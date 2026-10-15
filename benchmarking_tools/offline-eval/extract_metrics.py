# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract observed execution and source-timeline metrics from saved offline runs.

This reads recorded traces only. Missing timers stay null. Conversation gaps and
bulk request elapsed times are distinct quantities and are never added together.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

RATE = 16000

DEFINITIONS = {
    "client_total_wall_ms": ("Client connect/upload/wait/receive/render completion elapsed; not server-only latency."),
    "server_batch_e2e_ms": (
        "Source sealed to server pre-completion; excludes final metadata rewrite and completion-frame send."
    ),
    "server_runtime_wall_ms": (
        "Observed runtime.run duration, including parallel source analysis, response generation, and runtime cleanup."
    ),
    "server_first_wire_audio_ms": (
        "Source sealed to first committed PCM packet send completion. Offline publication "
        "occurs after whole-session generation/recording."
    ),
    "asr_total_wall_ms": (
        "One continuous ASR request over the complete finite source, including transport, queueing, and EOF draining."
    ),
    "asr_first_response_ms": ("ASR request start to first provider response, even if it contains no transcript."),
    "asr_first_transcript_ms": "ASR request start to first nonempty interim or final transcript.",
    "asr_first_final_transcript_ms": "ASR request start to first nonempty final transcript.",
    "asr_post_last_audio_supply_wall_ms": (
        "ASR stream EOF minus last source chunk supplied to its request iterator; includes bulk backlog/draining."
    ),
    "asr_final_observations": (
        "Observed arrival/supply/residual and source dependency for each final. These do not "
        "isolate backend GPU processing."
    ),
    "llm_first_text_ms": (
        "Response capture start to first nonempty LLM text, including history preparation and request setup."
    ),
    "llm_rpc_ttft_ms": "Actual LLM RPC dispatch to first nonempty text delta, for one model request round.",
    "llm_text_pipeline_complete_ms": (
        "Response capture start to LLM text stream completion and aggregation/filter flush."
    ),
    "tts_first_input_ms": "Response capture start to first text yielded into the synthesis input iterator.",
    "tts_first_audio_latency_ms": (
        "First synthesis text input to first PCM audio response; includes "
        "request/transport/queueing. Not first audio minus first LLM token."
    ),
    "tts_stream_elapsed_ms": (
        "First synthesis text input to synthesis response EOF; LLM may still be supplying text during this interval."
    ),
    "response_first_audio_ms": (
        "Response capture start to first generated PCM, including LLM, aggregation, and TTS; "
        "provisional audio may later be discarded."
    ),
    "response_generation_wall_ms": (
        "Whole overlapping LLM/TTS response capture, including generation later discarded by interruption."
    ),
    "endpoint_wait_ms": ("Source turn endpoint minus last detected VAD speech end for the last response-owned turn."),
    "post_endpoint_ready_wait_ms": (
        "Response request offset minus that endpoint. Includes gating/deferment, not pure ASR execution."
    ),
    "rendered_turn_gap_ms": (
        "First committed PCM sample minus last detected VAD speech end, on the rendered "
        "timeline. PCM may contain leading silence; this is not acoustic onset or live wall "
        "E2E."
    ),
    "projected_response_first_audio_ms": ("Response first-audio availability rounded forward to a sample boundary."),
    "mean_fields": (
        "Session means use response records with non-null values. Rendered-gap means exclude "
        "responses cancelled before any playback; response_count includes them."
    ),
    "units": (
        "Times ending _ms are milliseconds; source offsets are integer 16 kHz samples. "
        "Monotonic durations and source positions remain separate."
    ),
    "missing": ("Null/blank means the timer was not recorded or the event did not occur. It never means zero latency."),
}


def ms(value):
    """Convert integer nanoseconds while retaining absent observations."""
    return None if value is None else value / 1_000_000


def mean_present(values):
    """Exclude missing measurements, including unplayed cancelled responses."""
    present = [value for value in values if value is not None]
    return statistics.mean(present) if present else None


def read_json(path):
    """Load an already saved public run artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_events(directory):
    """Accept server event logs or the client's wire-wrapped equivalent."""
    path = directory / "server_events.jsonl"
    if not path.exists():
        path = directory / "events.jsonl"
    values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [value["event"] if value.get("type") == "offline.event" else value for value in values]


def extract_case(directory: Path, run_label: str):
    """Keep observed execution, generation availability, and playback distinct."""
    result = read_json(directory / "result.json")
    manifest_path = directory / "server_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    metrics = manifest.get("metrics", result["server_completion"]["metrics"])
    events = read_events(directory)
    turns = {}
    for event in events:
        key = event.get("turn_id")
        if key is None:
            continue
        turn = turns.setdefault(key, {})
        if event["type"] == "user_speech_end":
            turn["speech_end_sample"] = max(turn.get("speech_end_sample", 0), event["sample"])
        elif event["type"] == "user_turn_endpoint":
            turn["endpoint_sample"] = event["sample"]
        elif event["type"] == "turn_endpoint":
            turn.update(
                endpoint_sample=event["sample"],
                speech_end_sample=event["speech_end_sample"],
                endpoint_reason=event.get("reason"),
                endpoint_compute_ms=ms(event.get("compute_ns")),
            )
    responses = []
    llm_rounds = []
    for response in metrics["responses"]:
        provider = response["provider"]
        turn = turns[response["turn_ids"][-1]]
        spans = [
            event
            for event in events
            if event["type"] == "bot_audio" and event["response_id"] == response["response_id"]
        ]
        first = min((span["sample"] for span in spans), default=None)
        end = max((span["end_sample"] for span in spans), default=None)
        observed_audio = provider.get("first_audio_ready_ns")
        projected_audio_samples = (
            None if observed_audio is None else (observed_audio * RATE + 999_999_999) // 1_000_000_000
        )
        row = {
            "run_label": run_label,
            "case_id": directory.name,
            "response_id": response["response_id"],
            "turn_ids": ",".join(response["turn_ids"]),
            "coalesced_turn_count": len(response["turn_ids"]),
            "interrupted": response["interrupted"],
            "committed_audio": first is not None,
            "endpoint_reason": turn.get("endpoint_reason"),
            "last_vad_speech_end_sample": turn["speech_end_sample"],
            "endpoint_sample": turn["endpoint_sample"],
            "response_request_sample": response["request_sample"],
            "first_committed_audio_sample": first,
            "last_committed_audio_sample": end,
            "endpoint_wait_ms": (turn["endpoint_sample"] - turn["speech_end_sample"]) * 1000 / RATE,
            "endpoint_compute_ms": turn.get("endpoint_compute_ms"),
            "post_endpoint_ready_wait_ms": (response["request_sample"] - turn["endpoint_sample"]) * 1000 / RATE,
            "llm_first_text_ms": ms(provider.get("llm_first_text_ns")),
            "llm_text_pipeline_complete_ms": ms(provider.get("llm_complete_ns")),
            "response_first_audio_ms": ms(observed_audio),
            "projected_response_first_audio_ms": None
            if projected_audio_samples is None
            else projected_audio_samples * 1000 / RATE,
            "tts_first_text_queued_ms": ms(provider.get("tts_first_text_queued_ns")),
            "tts_first_input_ms": ms(provider.get("tts_first_input_ns")),
            "tts_first_audio_latency_ms": ms(provider.get("tts_first_audio_latency_ns")),
            "tts_stream_elapsed_ms": ms(provider.get("tts_stream_elapsed_ns")),
            "tts_last_audio_ready_ms": ms(provider.get("tts_last_audio_ready_ns")),
            "response_generation_wall_ms": ms(response["compute_ns"]),
            "rendered_turn_gap_ms": None if first is None else (first - turn["speech_end_sample"]) * 1000 / RATE,
            "generated_audio_ms": response["generated_samples"] * 1000 / RATE,
            "committed_audio_ms": response["committed_samples"] * 1000 / RATE,
        }
        if first is not None:
            expected = response["request_sample"] + projected_audio_samples
            if first != expected:
                raise ValueError("First committed audio does not match its recorded availability")
            parts = (
                row["endpoint_wait_ms"] + row["post_endpoint_ready_wait_ms"] + row["projected_response_first_audio_ms"]
            )
            if abs(parts - row["rendered_turn_gap_ms"]) > 1e-8:
                raise ValueError("Turn gap decomposition lost source sample alignment")
        rpc = provider.get("llm_provider_metrics", {})
        for round_record in rpc.get("rounds", ()):
            llm_rounds.append(
                {
                    "run_label": run_label,
                    "case_id": directory.name,
                    "response_id": response["response_id"],
                    **round_record,
                }
            )
        row["llm_rpc_ttft_ms"] = mean_present(ms(item.get("first_text_latency_ns")) for item in rpc.get("rounds", ()))
        row["llm_rpc_first_response_ms"] = mean_present(
            ms(item.get("first_response_latency_ns")) for item in rpc.get("rounds", ())
        )
        row["llm_rpc_stream_elapsed_ms"] = (
            sum(ms(item["stream_elapsed_ns"]) for item in rpc.get("rounds", ())) if rpc.get("rounds") else None
        )
        responses.append(row)
    recognition = metrics["recognition"]
    server = metrics.get("server", {})
    detector = next((event for event in events if event["type"] == "detector_complete"), {})
    session = {
        "run_label": run_label,
        "case_id": directory.name,
        "input_duration_s": result["source_duration_s"],
        "output_duration_s": result["output_duration_s"],
        "response_count": len(responses),
        "committed_response_count": sum(row["committed_audio"] for row in responses),
        "client_upload_wall_ms": result["upload_wall_s"] * 1000,
        "client_total_wall_ms": result["total_wall_s"] * 1000,
        "server_batch_e2e_ms": ms(server.get("source_sealed_to_precomplete_wall_ns")),
        "server_runtime_wall_ms": ms(server.get("runtime_run_wall_ns")),
        "server_build_wall_ms": ms(server.get("runtime_build_wall_ns")),
        "server_first_wire_audio_ms": ms(server.get("source_sealed_to_first_output_packet_sent_wall_ns")),
        "server_upload_wall_ms": ms(server.get("upload_accepted_to_source_sealed_wall_ns")),
        "server_generation_complete_ms": ms(server.get("source_sealed_to_generation_complete_wall_ns")),
        "server_output_recording_wall_ms": ms(server.get("output_recording_wall_ns")),
        "server_publication_wall_ms": ms(server.get("output_publication_wall_ns")),
        "asr_total_wall_ms": ms(recognition["asr_observed_wall_ns"]),
        "asr_first_response_ms": ms(recognition.get("asr_first_response_ns")),
        "asr_first_transcript_ms": ms(recognition.get("asr_first_transcript_ns")),
        "asr_first_final_transcript_ms": ms(recognition.get("asr_first_final_transcript_ns")),
        "asr_input_exhausted_ms": ms(recognition.get("asr_input_exhausted_ns")),
        "asr_first_audio_supplied_ms": ms(recognition.get("asr_first_audio_supplied_ns")),
        "asr_last_audio_supplied_ms": ms(recognition.get("asr_last_audio_supplied_ns")),
        "asr_post_last_audio_supply_wall_ms": (
            ms(recognition["asr_stream_end_ns"] - recognition["asr_last_audio_supplied_ns"])
            if recognition.get("asr_stream_end_ns") is not None
            and recognition.get("asr_last_audio_supplied_ns") is not None
            else None
        ),
        "vad_total_model_work_ms": ms(detector.get("vad_compute_ns")),
        "smart_turn_total_model_work_ms": ms(detector.get("turn_compute_ns")),
        "response_generation_wall_sum_ms": sum(row["response_generation_wall_ms"] for row in responses),
        "mean_llm_rpc_ttft_ms": mean_present(row["llm_rpc_ttft_ms"] for row in responses),
        "mean_llm_rpc_first_response_ms": mean_present(row["llm_rpc_first_response_ms"] for row in responses),
        "mean_llm_rpc_stream_elapsed_ms": mean_present(row["llm_rpc_stream_elapsed_ms"] for row in responses),
        "mean_llm_first_text_ms": mean_present(row["llm_first_text_ms"] for row in responses),
        "mean_tts_first_audio_latency_ms": mean_present(row["tts_first_audio_latency_ms"] for row in responses),
        "mean_tts_stream_elapsed_ms": mean_present(row["tts_stream_elapsed_ms"] for row in responses),
        "mean_response_first_audio_ms": mean_present(row["response_first_audio_ms"] for row in responses),
        "mean_rendered_turn_gap_ms": mean_present(row["rendered_turn_gap_ms"] for row in responses),
        "timing_instrumentation_present": bool(server) and "asr_first_response_ns" in recognition,
    }
    asr_finals = [
        {"run_label": run_label, "case_id": directory.name, **record} for record in recognition.get("asr_finals", ())
    ]
    return {"session": session, "responses": responses, "asr_finals": asr_finals, "llm_rounds": llm_rounds}


def write_csv(path, records):
    """Use blank CSV cells for null observations; preserve exact numeric values."""
    names = list(dict.fromkeys(name for record in records for name in record))
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=names)
        writer.writeheader()
        writer.writerows(records)


def extract(input_dir: Path, output_dir: Path, run_label: str):
    """Produce portable metrics from all successful saved cases under one root."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Choose an empty metrics output directory")
    directories = sorted(path.parent for path in input_dir.rglob("result.json") if "diagnostics" not in path.parts)
    if not directories:
        raise ValueError("No successful saved cases found")
    report = {
        "run_label": run_label,
        "definitions": DEFINITIONS,
        "sessions": [],
        "responses": [],
        "asr_finals": [],
        "llm_rounds": [],
        "source_files_sha256": {},
    }
    for directory in directories:
        case = extract_case(directory, run_label)
        report["sessions"].append(case["session"])
        for name in ("responses", "asr_finals", "llm_rounds"):
            report[name].extend(case[name])
        for name in ("result.json", "server_manifest.json", "server_events.jsonl", "events.jsonl"):
            path = directory / name
            if path.exists():
                report["source_files_sha256"][str(path.relative_to(input_dir))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("sessions", "responses", "asr_finals", "llm_rounds"):
        write_csv(output_dir / (name + ".csv"), report[name])
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    """Extract existing observations without making network or model requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    arguments = parser.parse_args()
    report = extract(arguments.input_dir, arguments.output_dir, arguments.run_label)
    print(
        json.dumps(
            {
                "sessions": len(report["sessions"]),
                "responses": len(report["responses"]),
                "output": str(arguments.output_dir),
            }
        )
    )


if __name__ == "__main__":
    main()
