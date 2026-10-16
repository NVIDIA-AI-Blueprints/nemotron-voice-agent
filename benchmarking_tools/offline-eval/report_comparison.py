# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report paired offline/live recordings without merging incompatible clocks.

This reads completed runs only and never makes provider requests. Missing runs,
unequal sources, and ambiguous server session matches fail explicitly. Trace
LLM requests and TTS contexts remain separate from wire response records unless
an explicit response-id correlation is recorded; matching ordinal counts alone
does not prove that relationship.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import wave
from pathlib import Path
from typing import Any

_SPEC = importlib.util.spec_from_file_location(
    "nva_comparison_offline_extractor", Path(__file__).with_name("extract_metrics.py")
)
assert _SPEC is not None and _SPEC.loader is not None
extractor = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = extractor
_SPEC.loader.exec_module(extractor)

DEFINITIONS = {
    "units": "Every duration/position reported in time uses milliseconds (_ms). Sample counts are integer PCM samples.",
    "valid_response": (
        "A response with at least one committed output PCM sample; silence-only file regions are not turns."
    ),
    "source_pairing": (
        "Actual input.wav PCM bytes and format are compared; matching filenames or manifest claims are insufficient."
    ),
    "configuration": (
        "Input equality does not establish equal provider settings. Offline configuration fingerprint and accepted "
        "session file hashes are retained for configuration review; no automatic prompt/model parity claim is made."
    ),
    "trace_pairing": "accepted_session.json id must match exactly one trace configuration.realtime_session_id.",
    "trace_response_association": (
        "Trace LLM request indices and TTS context ids are not wire response ordinals. "
        "A direct realtime_response_created trace event with response_id is required to associate them."
    ),
    "asr_callback": (
        "Live asr_provider_result is the existing event-loop result callback, not raw gRPC first byte. "
        "Worker-start and first-audio-request-yield baselines are reported separately."
    ),
    "asr_worker_lifetime": "Includes source pacing and teardown; not isolated inference computation.",
    "llm_trace": (
        "Service LLM start frame to first nonempty text/end frame using pipeline_timestamp_ns. "
        "Observer callback elapsed clocks can lag; they are not substituted. Not necessarily one provider RPC. "
        "Provider wrapper and pipeline clock origins are uncalibrated, so LLM-frame-to-TTS times remain null."
    ),
    "tts_trace": (
        "First filtered text submitted to run_tts to first generated PCM queued before paced playback. "
        "Worker return includes upstream text stalls. Provider errors may be caught internally, so normal return "
        "does not prove successful RPC EOF. Cancelled workers never populate returned-duration fields."
    ),
    "wire_server_metrics": (
        "Live response.done nvidia_metrics are original Pipecat accumulator windows between completions. "
        "ASR TTFB means inferred speech-end to final transcript; LLM TTFT can include a role-only first delta. "
        "They are distinct from offline whole-stream ASR and nonempty-text provider request timers."
    ),
    "turn_gap": (
        "Offline rendered_turn_gap_ms uses last detected VAD speech end to committed PCM. "
        "Live wire gateway_stop_to_playback_ms uses semantic-release cursor to client playback and is different. "
        "Live server_e2e_ms is the server's VAD-silence-to-BotStartedSpeaking observation."
    ),
    "completion_ratio": (
        "online/offline client completion wall ratio includes different setup/publication and the online idle drain. "
        "It is run throughput evidence, not a response-latency or inference-speed ratio."
    ),
    "missing": (
        "Null and blank CSV cells mean absent/unobserved, never zero. No acoustic scoring or task accuracy is inferred."
    ),
}


def read_json(path: Path) -> Any:
    """Read required evidence, failing with a useful file-level diagnostic."""
    if not path.is_file():
        raise ValueError(f"Required completed-run evidence is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        raise ValueError(f"Invalid JSON evidence: {path}") from error


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _ms(event: dict[str, Any] | None) -> float | None:
    if event is None:
        return None
    ns = _number(event.get("elapsed_ns"))
    value = None if ns is None else ns / 1_000_000
    stated = _number(event.get("elapsed_ms"))
    if value is not None and stated is not None and abs(value - stated) > 0.000001:
        raise ValueError("Server trace elapsed_ns/elapsed_ms disagree")
    return value if value is not None else stated


def _delta(start: dict[str, Any] | None, end: dict[str, Any] | None) -> float | None:
    first, last = _ms(start), _ms(end)
    if first is None or last is None:
        return None
    if last < first:
        raise ValueError("Trace duration has an end preceding its start")
    return last - first


def _pipeline_delta(start: dict[str, Any] | None, end: dict[str, Any] | None) -> float | None:
    first = _number((start or {}).get("pipeline_timestamp_ns"))
    last = _number((end or {}).get("pipeline_timestamp_ns"))
    if first is None or last is None:
        return None
    if last < first:
        raise ValueError("Pipeline frame clock has an end preceding its start")
    return (last - first) / 1_000_000


def _first(events: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    return next((event for event in events if event.get("event") == kind), None)


def _mean(values) -> float | None:
    numbers = [number for value in values if (number := _number(value)) is not None]
    return statistics.mean(numbers) if numbers else None


def _metric_average(value: Any) -> float | None:
    if isinstance(value, list):
        return _mean(value)
    if isinstance(value, dict):
        return _number(value.get("average_ms"))
    return None


def wav_info(path: Path) -> tuple[dict[str, Any], bytes]:
    """Hash actual WAV contents and retain format/duration without interpreting speech."""
    if not path.is_file():
        raise ValueError(f"Required recording is missing: {path}")
    with wave.open(str(path), "rb") as source:
        rate, channels, width, frames = (
            source.getframerate(),
            source.getnchannels(),
            source.getsampwidth(),
            source.getnframes(),
        )
        if source.getcomptype() != "NONE":
            raise ValueError(f"Comparison requires uncompressed PCM WAV: {path}")
        pcm = source.readframes(frames)
    if len(pcm) != frames * channels * width:
        raise ValueError(f"Recording is truncated: {path}")
    return {
        "path": str(path),
        "sample_rate": rate,
        "channels": channels,
        "sample_width_bytes": width,
        "frames": frames,
        "duration_ms": frames * 1000 / rate,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
    }, pcm


def completed_cases(root: Path, mode: str) -> dict[str, Path]:
    """Find actual successful case results; never synthesize pending rows."""
    if not root.is_dir():
        raise ValueError(f"{mode} results directory does not exist: {root}")
    cases = {}
    for path in sorted(root.rglob("result.json")):
        result = read_json(path)
        valid = (
            (result.get("server_completion", {}).get("status") == "ok")
            if mode == "offline"
            else (
                result.get("mode") == "online_1x_normal_realtime_pipeline"
                and result.get("completion_reason") == "post_input_idle"
            )
        )
        if not valid:
            raise ValueError(f"Incomplete or unexpected {mode} result: {path}")
        if path.parent.name in cases:
            raise ValueError(f"Duplicate {mode} case id: {path.parent.name}")
        cases[path.parent.name] = path.parent
    if not cases:
        raise ValueError(f"No completed {mode} recordings found: {root}")
    return cases


def trace_index(directory: Path) -> dict[str, list[tuple[Path, dict[str, Any]]]]:
    """Index passive traces by exact live session ids rather than file order."""
    if not directory.is_dir():
        raise ValueError(f"Server traces directory does not exist: {directory}")
    index: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path in sorted(directory.glob("*.json")):
        trace = read_json(path)
        if trace.get("schema") != "nva.evaluation.trace.v1":
            continue
        session = trace.get("configuration", {}).get("realtime_session_id")
        if isinstance(session, str) and session:
            index.setdefault(session, []).append((path, trace))
    return index


def trace_tables(trace: dict[str, Any], case_id: str) -> dict[str, list[dict[str, Any]]]:
    """Keep ASR worker, LLM frame, TTS context, and source metrics in their true scopes."""
    events = trace.get("events", [])
    if not isinstance(events, list):
        raise ValueError("Server trace events must be a list")
    times = [_ms(event) for event in events]
    if any(value is None for value in times) or any(
        left > right for left, right in zip(times, times[1:], strict=False)
    ):
        raise ValueError("Server trace events require nondecreasing observed clocks")
    common = {"case_id": case_id, "trace_session_id": trace.get("session_id")}
    asr_rows, windows, active = [], [], []
    for event in events:
        if event["event"] == "asr_worker_started":
            window = {"start": event, "end": None, "events": [], "ambiguous": bool(active)}
            for previous in active:
                previous["ambiguous"] = True
            windows.append(window)
            active.append(window)
        elif event["event"] == "asr_worker_finished":
            if len(active) == 1:
                active[0]["end"] = event
                active.clear()
            elif active:
                for window in active:
                    window["ambiguous"] = True
                # No worker id is present, so an overlapping finish cannot be assigned.
        elif len(active) == 1:
            active[0]["events"].append(event)
    for index, window in enumerate(windows, 1):
        inside = [] if window["ambiguous"] else window["events"]
        supply = next(
            (event for event in inside if event["event"] == "asr_request_supplied" and event.get("audio_bytes", 0) > 0),
            None,
        )
        callbacks = [event for event in inside if event["event"] == "asr_provider_result"]
        first = callbacks[0] if callbacks else None
        text = next(
            (
                event
                for event in callbacks
                if any(str(result.get("transcript", "")).strip() for result in event.get("results", []))
            ),
            None,
        )
        final = next(
            (
                event
                for event in callbacks
                if any(
                    result.get("is_final") and str(result.get("transcript", "")).strip()
                    for result in event.get("results", [])
                )
            ),
            None,
        )
        row = {
            **common,
            "asr_worker_index": index,
            "association_ambiguous": window["ambiguous"],
            "worker_started_elapsed_ms": _ms(window["start"]),
            "worker_finished_elapsed_ms": _ms(window["end"]),
            "observed_worker_lifetime_ms": _delta(window["start"], window["end"]),
            "first_audio_supplied_elapsed_ms": _ms(supply),
            "callback_count": len(callbacks),
        }
        for name, observation in (
            ("first_callback", first),
            ("first_nonempty_transcript", text),
            ("first_final_transcript", final),
        ):
            row[f"{name}_from_worker_start_ms"] = _delta(window["start"], observation)
            row[f"{name}_from_first_audio_supply_ms"] = _delta(supply, observation)
        asr_rows.append(row)

    explicit: dict[int, set[str]] = {}
    for event in events:
        if event.get("event") == "realtime_response_created" and isinstance(event.get("response_id"), str):
            index = event.get("llm_request_index")
            if isinstance(index, int) and not isinstance(index, bool):
                explicit.setdefault(index, set()).add(event["response_id"])
    reverse: dict[str, set[int]] = {}
    for index, ids in explicit.items():
        for response_id in ids:
            reverse.setdefault(response_id, set()).add(index)

    def correlation(index):
        ids = explicit.get(index, set())
        response_id = next(iter(ids)) if len(ids) == 1 else None
        proven = (
            response_id is not None
            and len(reverse[response_id]) == 1
            and not trace.get("dropped_events", 0)
            and not trace.get("observation_errors", 0)
        )
        return {"wire_response_id": response_id if proven else None, "wire_association_proven": proven}

    contexts: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        context_id = event.get("context_id")
        if isinstance(context_id, str) and context_id and event["event"].startswith("tts_"):
            contexts.setdefault(context_id, []).append(event)
    tts_rows = []
    for context_id, context in contexts.items():
        first_text = _first(context, "tts_text_submitted")
        first_pcm = next(
            (event for event in context if event["event"] == "tts_pcm_ready" and event.get("audio_bytes", 0) > 0), None
        )
        starts = [event for event in context if event["event"] == "tts_worker_started"]
        ends = [event for event in context if event["event"] == "tts_worker_finished"]
        start = starts[0] if len(starts) == 1 else None
        end = ends[0] if len(ends) == 1 else None
        outcome = end.get("outcome") if end else None
        index = (first_text or start or first_pcm or {}).get("llm_request_index")
        tts_rows.append(
            {
                **common,
                "context_id": context_id,
                "llm_request_index": index,
                **correlation(index),
                "synthesis_mode": (first_text or {}).get("synthesis_mode"),
                "first_text_submitted_elapsed_ms": _ms(first_text),
                "first_pcm_ready_elapsed_ms": _ms(first_pcm),
                "submission_to_first_pcm_ms": _delta(first_text, first_pcm),
                "worker_observed_lifetime_ms": _delta(start, end),
                "worker_outcome": outcome,
                "worker_stop_requested": end.get("stop_requested") if end else None,
                "submission_to_worker_return_ms": _delta(first_text, end) if outcome == "returned" else None,
                "worker_returned_lifetime_ms": _delta(start, end) if outcome == "returned" else None,
                "rpc_success_confirmed": False,
                "completion_scope": "provider_success_unknown" if outcome == "returned" else outcome or "unobserved",
                "generated_pcm_duration_ms": sum(
                    event["audio_bytes"] / (2 * event.get("num_channels", 1)) * 1000 / event["sample_rate"]
                    for event in context
                    if event["event"] == "tts_pcm_ready" and event.get("sample_rate", 0) > 0
                ),
            }
        )

    indices = sorted(
        {
            event["llm_request_index"]
            for event in events
            if isinstance(event.get("llm_request_index"), int) and not isinstance(event.get("llm_request_index"), bool)
        }
    )
    llm_rows = []
    for index in indices:
        request = [event for event in events if event.get("llm_request_index") == index]
        starts = [event for event in request if event["event"] == "LLMFullResponseStartFrame"]
        ends = [event for event in request if event["event"] == "LLMFullResponseEndFrame"]
        start, end = starts[0] if len(starts) == 1 else None, ends[0] if len(ends) == 1 else None
        first_text = next(
            (event for event in request if event["event"] == "llm_text" and str(event.get("text", "")).strip()), None
        )
        first_pcm = next(
            (event for event in request if event["event"] == "tts_pcm_ready" and event.get("audio_bytes", 0) > 0), None
        )
        llm_rows.append(
            {
                **common,
                "llm_request_index": index,
                **correlation(index),
                "llm_start_elapsed_ms": _ms(start),
                "first_text_elapsed_ms": _ms(first_text),
                "llm_end_elapsed_ms": _ms(end),
                "llm_frame_to_first_nonempty_text_ms": _pipeline_delta(start, first_text),
                "llm_frame_to_end_ms": _pipeline_delta(start, end),
                "llm_frame_to_first_generated_pcm_ms": None,
                "synthesis_context_count": sum(row["llm_request_index"] == index for row in tts_rows),
            }
        )
    server_messages = []
    for event in events:
        if event["event"] != "server_message":
            continue
        data = event.get("data", {})
        kind = data.get("type")
        value = data.get("latency") if kind == "user-bot-latency" else data.get("vad_smart_turn")
        number = _number(value)
        if kind in {"user-bot-latency", "latency-breakdown"}:
            server_messages.append(
                {
                    **common,
                    "event": kind,
                    "observed_elapsed_ms": _ms(event),
                    "llm_request_index_at_observation": event.get("llm_request_index"),
                    "first": data.get("first"),
                    "value_ms": None if number is None else number * 1000,
                }
            )
    return {
        "asr_workers": asr_rows,
        "llm_requests": llm_rows,
        "tts_contexts": tts_rows,
        "server_messages": server_messages,
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    names = list(dict.fromkeys(key for record in records for key in record))
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=names)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: json.dumps(value, allow_nan=False) if isinstance(value, (list, dict)) else value
                    for key, value in record.items()
                }
            )


def build_report(offline_root: Path, online_root: Path, traces_root: Path, output: Path) -> dict[str, Any]:
    """Validate evidence first, then produce portable paired CSV/JSON tables."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new or empty comparison output directory")
    offline, online, traces = (
        completed_cases(offline_root, "offline"),
        completed_cases(online_root, "online"),
        trace_index(traces_root),
    )
    if set(offline) != set(online):
        raise ValueError("Offline/online completed case ids differ; paired results are incomplete")
    report: dict[str, Any] = {
        "schema": "nva.offline_live.comparison.v1",
        "definitions": DEFINITIONS,
        "cases": [],
        "offline_valid_responses": [],
        "online_valid_responses": [],
        "asr_workers": [],
        "llm_requests": [],
        "tts_contexts": [],
        "server_messages": [],
        "wav_inventory": [],
        "transcripts": [],
        "warnings": [],
    }
    for case_id in sorted(offline):
        off_dir, on_dir = offline[case_id], online[case_id]
        off_input, off_pcm = wav_info(off_dir / "input.wav")
        on_input, on_pcm = wav_info(on_dir / "input.wav")
        if any(
            (item["sample_rate"], item["channels"], item["sample_width_bytes"]) != (16000, 1, 2)
            for item in (off_input, on_input)
        ):
            raise ValueError(f"Paired case {case_id} requires canonical 16 kHz mono PCM16")
        if off_pcm != on_pcm:
            raise ValueError(f"Offline/online source PCM differs for case {case_id}")
        accepted, on_result = read_json(on_dir / "accepted_session.json"), read_json(on_dir / "result.json")
        off_result = read_json(off_dir / "result.json")
        if (
            off_result.get("source_sha256") != off_input["pcm_sha256"]
            or on_result.get("input_pcm_sha256") != on_input["pcm_sha256"]
        ):
            raise ValueError(f"Recorded source hash does not match saved input PCM for case {case_id}")
        session_id = accepted.get("id")
        matches = traces.get(session_id, []) if isinstance(session_id, str) else []
        if not isinstance(session_id, str) or len(matches) != 1:
            raise ValueError(
                f"Expected exactly one server trace matching accepted session id for case {case_id}; "
                f"found {len(matches)}"
            )
        trace_path, trace = matches[0]
        off_report = extractor.extract_case(off_dir, "paired_offline")
        off_session = off_report["session"]
        off_rows = [row for row in off_report["responses"] if row["committed_audio"]]
        on_rows = [
            {"case_id": case_id, **row}
            for row in on_result.get("responses", [])
            if row.get("committed_audio_samples", 0) > 0
        ]
        tables = trace_tables(trace, case_id)
        for key, records in tables.items():
            report[key].extend(records)
        off_wall, on_wall = (
            _number(off_session.get("client_total_wall_ms")),
            _number(on_result.get("client_total_wall_ms")),
        )
        last_output = max((row.get("last_audio_receive_ms", 0) for row in on_rows), default=0)
        active_end = max(on_result.get("client_input_send_ms", 0), last_output)
        sends_path = on_dir / "send_trace.json"
        send_lateness = (
            [row["send_start_ms"] - row["start_sample"] * 1000 / 16000 for row in read_json(sends_path)]
            if sends_path.exists()
            else []
        )
        report["cases"].append(
            {
                "case_id": case_id,
                "paired_input_pcm_equal": True,
                "input_pcm_sha256": off_input["pcm_sha256"],
                "input_duration_ms": off_input["duration_ms"],
                "offline_valid_response_count": len(off_rows),
                "online_valid_response_count": len(on_rows),
                "offline_generated_response_count": len(off_report["responses"]),
                "offline_client_completion_wall_ms": off_wall,
                "online_client_completion_wall_ms": on_wall,
                "online_client_setup_ms": on_result.get("client_setup_ms"),
                "online_client_active_input_or_output_wall_ms": active_end,
                "online_client_idle_close_overhead_ms": None
                if on_wall is None
                else on_wall - on_result.get("client_setup_ms", 0) - active_end,
                "online_to_offline_client_completion_wall_ratio": on_wall / off_wall
                if on_wall is not None and off_wall and off_wall > 0
                else None,
                "online_target_pacing": "1x absolute source deadlines; late packets can catch up",
                "online_max_send_deadline_lateness_ms": max(send_lateness) if send_lateness else None,
                "offline_server_batch_e2e_ms": off_session.get("server_batch_e2e_ms"),
                "offline_server_runtime_wall_ms": off_session.get("server_runtime_wall_ms"),
                "offline_asr_stream_wall_ms": off_session.get("asr_total_wall_ms"),
                "offline_asr_stream_first_response_ms": off_session.get("asr_first_response_ms"),
                "offline_asr_stream_first_nonempty_text_ms": off_session.get("asr_first_transcript_ms"),
                "offline_mean_valid_llm_rpc_first_text_ms": _mean(row.get("llm_rpc_ttft_ms") for row in off_rows),
                "offline_mean_valid_tts_input_to_first_pcm_ms": _mean(
                    row.get("tts_first_audio_latency_ms") for row in off_rows
                ),
                "offline_mean_valid_rendered_turn_gap_ms": _mean(row.get("rendered_turn_gap_ms") for row in off_rows),
                "online_mean_valid_wire_server_e2e_ms": _mean(
                    _metric_average(row.get("server_e2e_ms")) for row in on_rows
                ),
                "online_mean_valid_wire_llm_first_choices_chunk_ms": _mean(
                    _metric_average(row.get("llm_ttft_ms")) for row in on_rows
                ),
                "online_mean_valid_wire_tts_ttfb_ms": _mean(_metric_average(row.get("tts_ttfb_ms")) for row in on_rows),
                "realtime_session_id": session_id,
                "offline_configuration_fingerprint": off_result["server_completion"].get("configuration_fingerprint"),
                "online_accepted_session_sha256": hashlib.sha256(
                    (on_dir / "accepted_session.json").read_bytes()
                ).hexdigest(),
                "server_trace": str(trace_path),
                "server_trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                "server_trace_dropped_events": trace.get("dropped_events", 0),
                "server_trace_observation_errors": trace.get("observation_errors", 0),
            }
        )
        report["offline_valid_responses"].extend(off_rows)
        report["online_valid_responses"].extend(on_rows)
        for mode, directory in (("offline", off_dir), ("online", on_dir)):
            wav_info(directory / "full_output.wav")  # Required real output, not an invented silent file.
            for path in sorted(directory.glob("*.wav")):
                info, _ = wav_info(path)
                report["wav_inventory"].append({"case_id": case_id, "mode": mode, **info})
        manifest_path = off_dir / "server_manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        report["transcripts"].append(
            {
                "case_id": case_id,
                "mode": "offline",
                "scope": "committed_conversation_messages",
                "messages": manifest.get("messages"),
            }
        )
        transcript_path = on_dir / "transcripts.json"
        report["transcripts"].append(
            {
                "case_id": case_id,
                "mode": "online",
                "scope": "wire_transcription_events",
                "events": read_json(transcript_path) if transcript_path.exists() else None,
            }
        )
        if trace.get("dropped_events", 0):
            report["warnings"].append(
                f"{case_id}: server trace dropped events; trace-to-wire associations are disabled"
            )
        if trace.get("observation_errors", 0):
            report["warnings"].append(
                f"{case_id}: server trace observation errors occurred; trace-to-wire associations are disabled "
                "and request/context tables may be incomplete"
            )
        if not any(row["wire_association_proven"] for row in tables["llm_requests"]):
            report["warnings"].append(
                f"{case_id}: no explicit one-to-one LLM request / wire response correlation; tables remain separate"
            )
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for key in (
        "cases",
        "offline_valid_responses",
        "online_valid_responses",
        "asr_workers",
        "llm_requests",
        "tts_contexts",
        "server_messages",
        "wav_inventory",
    ):
        _write_csv(output / f"{key}_ms.csv", report[key])
    (output / "transcripts.json").write_text(
        json.dumps(report["transcripts"], indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    """Extract paired evidence from successful offline/live runs and passive server traces."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-results-dir", type=Path, required=True)
    parser.add_argument("--online-results-dir", type=Path, required=True)
    parser.add_argument("--server-traces-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.offline_results_dir, args.online_results_dir, args.server_traces_dir, args.output_dir)
    print(f"Reported {len(report['cases'])} verified paired recordings in {args.output_dir}")


if __name__ == "__main__":
    main()
