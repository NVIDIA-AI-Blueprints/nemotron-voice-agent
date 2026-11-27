# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Timestamped session evidence and conservative post-test gap classification."""

from __future__ import annotations

import collections
import math
import time
from typing import Any

from pipecat.utils.security.redaction import redact_secrets

GAP_CLASSES = ("tool_wait", "unknown")
LEGACY_GAP_CLASSES = ("within_sentence", "between_sentences", "interruption")
NO_VALID_TURNS = "No valid benchmark turns completed"
MAX_ERROR_MESSAGE_CHARS = 2048

_NATIVE_MARKERS = {
    "bot-llm-started": "llm_start",
    "bot-llm-stopped": "llm_end",
    "bot-tts-started": "tts_start",
    "bot-tts-stopped": "tts_end",
    "bot-started-speaking": "speech_start",
    "bot-stopped-speaking": "speech_end",
    "bot-interrupted": "interruption",
    "user-started-speaking": "user_start",
    "user-stopped-speaking": "user_end",
    "llm-function-call-started": "tool_announced",
    "llm-function-call-in-progress": "tool_start",
    "llm-function-call-stopped": "tool_end",
    "bot-output": "text_segment",
    "nvidia.tool.started": "tool_start",
    "nvidia.tool.completed": "tool_end",
    "response.created": "response_start",
    "response.done": "response_end",
    "input_audio_buffer.speech_started": "user_start",
    "input_audio_buffer.speech_stopped": "user_end",
}


def bounded_error_message(message: str) -> str:
    """Redact credential values before limiting the report text."""
    return redact_secrets(str(message)).strip()[:MAX_ERROR_MESSAGE_CHARS]


def _component(name: str) -> str:
    name = name.lower()
    for category, parts in (("asr", ("asr", "stt", "transcri")), ("tts", ("tts", "texttospeech")), ("llm", ("llm",))):
        if any(part in name for part in parts):
            return category
    return "unknown"


def _ms(start_ns: int | None, end_ns: int | None) -> float | None:
    if start_ns is None or end_ns is None:
        return None
    return (end_ns - start_ns) / 1_000_000


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


class SessionDiagnostics:
    """Collect bounded playback and native protocol evidence on the client."""

    def __init__(self, *, max_events: int = 50000):
        """Bound per-session storage and report any evidence loss explicitly."""
        self.max_events = max_events
        self.events: list[dict] = []
        self.audio_packets: list[dict] = []
        self.gaps: list[dict] = []
        self.errors: list[dict] = []
        self.audio_sequence = 0
        self.audio_ms = 0.0
        self.start: float | None = None
        self.duration: float | None = None
        self.local_dropped = 0
        self.audio_discontinuity_after_sequence: int | None = None
        self._current_turn: dict | None = None
        self._turns: list[dict] = []
        self._turn_sequence = 0

    def mark_audio_discontinuity(self) -> None:
        """Mark a point where received Realtime PCM was discarded outside playback."""
        if self.audio_discontinuity_after_sequence is None:
            self.audio_discontinuity_after_sequence = self.audio_sequence

    def begin_turn(self, audio_file: str, *, monotonic_ns: int | None = None, unix_s: float | None = None) -> None:
        """Start one benchmark input/response observation window."""
        if self._current_turn is not None:
            self.end_turn(error="turn ended before diagnostic completion")
        self._turn_sequence += 1
        unix_s = time.time() if unix_s is None else unix_s
        self._current_turn = {
            "turn": self._turn_sequence,
            "input_file": audio_file,
            "started_unix_s": unix_s,
            "started_monotonic_ns": time.monotonic_ns() if monotonic_ns is None else monotonic_ns,
            "phase": self.phase(unix_s),
            "event_start": len(self.events),
            "audio_start": len(self.audio_packets),
        }

    def end_turn(
        self,
        *,
        input_end_monotonic_s: float | None = None,
        error: str | None = None,
        client_timing: dict | None = None,
    ) -> None:
        """Close a turn, including one with an error or missing audio."""
        if self._current_turn is None:
            return
        self._append(
            self._turns,
            {
                **self._current_turn,
                "input_end_monotonic_ns": round(input_end_monotonic_s * 1e9)
                if input_end_monotonic_s is not None
                else None,
                "ended_monotonic_ns": time.monotonic_ns(),
                "event_end": len(self.events),
                "audio_end": len(self.audio_packets),
                "error": bounded_error_message(error) if error else None,
                "client_timing": client_timing or {},
            },
        )
        self._current_turn = None

    def phase(self, unix_s: float) -> str:
        """Preserve warmup and reply-drain evidence outside the measurement."""
        if self.start is None or self.duration is None:
            return "unknown"
        if unix_s < self.start:
            return "warmup"
        return "measurement" if unix_s <= self.start + self.duration else "reply_drain"

    def _append(self, items: list, item: dict) -> None:
        if len(items) < self.max_events:
            items.append(item)
        else:
            self.local_dropped += 1

    def add_error(self, message: str, *, source: str, component: str = "unknown", **fields) -> None:
        """Keep reported errors even when later turns finish successfully."""
        now = fields.pop("received_unix_s", time.time())
        entry = {
            "message": bounded_error_message(message),
            "source": source,
            "component": component,
            "received_unix_s": now,
            "phase": self.phase(now) if now is not None else "unknown",
            **fields,
        }
        # An exception at teardown can repeat a protocol/handshake error just observed.
        if source == "client" and entry.get("code") == "session_error":
            for old in reversed(self.errors):
                old_time = old.get("received_unix_s")
                if now is None or old_time is None or abs(now - old_time) > 2:
                    break
                if (
                    len(entry["message"]) < MAX_ERROR_MESSAGE_CHARS
                    and old["message"] == entry["message"]
                    and old["source"] in {"handshake", "protocol", "pipecat_frame"}
                    and source not in old.get("also_reported_by", [])
                ):
                    old["also_reported_by"] = sorted(set(old.get("also_reported_by", []) + [source]))
                    return
        self._append(self.errors, entry)

    def ingest(
        self,
        message: dict[str, Any],
        *,
        received_monotonic_ns: int | None = None,
        received_unix_s: float | None = None,
    ) -> None:
        """Record existing RTVI and Realtime protocol events as metadata."""
        kind = str(message.get("type") or "")
        if kind == "server-message" and isinstance(message.get("data"), dict):
            self.ingest(
                message["data"],
                received_monotonic_ns=received_monotonic_ns,
                received_unix_s=received_unix_s,
            )
            return
        marker = _NATIVE_MARKERS.get(kind)
        if not marker and kind != "error" and not kind.endswith(".failed"):
            return
        unix_s = time.time() if received_unix_s is None else received_unix_s
        mono_ns = time.monotonic_ns() if received_monotonic_ns is None else received_monotonic_ns
        data = message.get("data")
        data = data if isinstance(data, dict) else {}
        response = message.get("response")
        response = response if isinstance(response, dict) else {}
        response_status = response.get("status")

        if kind == "error" or kind.endswith(".failed"):
            detail = message.get("error", data)
            if isinstance(detail, dict):
                text = detail.get("error") or detail.get("message") or str(detail)
                self.add_error(
                    str(text),
                    source="protocol",
                    component=_component(str(detail.get("processor", "")) + str(text)),
                    fatal=detail.get("fatal"),
                    code=detail.get("code"),
                    event_type=kind,
                    received_unix_s=unix_s,
                )
            else:
                self.add_error(
                    str(detail),
                    source="protocol",
                    component=_component(str(detail)),
                    event_type=kind,
                    received_unix_s=unix_s,
                )
        elif kind == "response.done" and response_status in ("failed", "incomplete"):
            self.add_error(
                str(response.get("status_details") or response_status),
                source="protocol",
                event_type=kind,
                received_unix_s=unix_s,
            )

        if marker is None:
            return
        event = {
            "kind": marker,
            "protocol_event": kind,
            "received_monotonic_ns": mono_ns,
            "received_unix_s": unix_s,
            "audio_sequence": self.audio_sequence,
        }
        if marker.startswith("tool_"):
            event.update(
                tool_call_id=_string(data.get("tool_call_id") or message.get("tool_call_id")),
                tool_name=_string(data.get("function_name") or message.get("name")),
                cancelled=bool(data.get("cancelled") or message.get("cancelled")),
            )
        elif marker.startswith("response_"):
            event.update(
                response_id=_string(response.get("id") or message.get("response_id")),
                status=_string(response_status),
            )
        elif marker == "text_segment":
            event.update(segment_id=_string(data.get("segment_id")), output_type=_string(data.get("output_type")))
        self._append(self.events, event)

    def audio(
        self,
        *,
        nbytes: int,
        sample_rate: int,
        channels: int,
        monotonic_ns: int,
        unix_s: float,
        deficit_ms: float | None,
        detect: bool,
        response_id: str | None = None,
    ) -> None:
        """Record PCM arrival and the detector's actual cumulative deficit."""
        if not nbytes:
            return
        if sample_rate <= 0 or channels <= 0:
            self.local_dropped += 1
            return
        duration = nbytes * 1000 / (2 * sample_rate * channels)
        if not math.isfinite(duration):
            self.local_dropped += 1
            return
        self.audio_sequence += 1
        packet = {
            "audio_sequence": self.audio_sequence,
            "received_monotonic_ns": monotonic_ns,
            "received_unix_s": unix_s,
            "duration_ms": duration,
            "audio_ms_before": self.audio_ms,
            "phase": self.phase(unix_s),
            "playback_deficit_ms": deficit_ms,
        }
        if response_id is not None:
            packet["response_id"] = response_id
        self._append(self.audio_packets, packet)
        if detect and deficit_ms is not None and deficit_ms > 20:
            self._append(self.gaps, {**packet, "deficit_ms": deficit_ms})
        self.audio_ms += duration

    def _turn_report(self, raw: dict, attributed: dict[int, dict]) -> dict:
        events = self.events[raw["event_start"] : raw["event_end"]]
        packets = self.audio_packets[raw["audio_start"] : raw["audio_end"]]
        packet_by_seq = {p["audio_sequence"]: p for p in packets}
        first_audio = packets[0]["received_monotonic_ns"] if packets else None
        rounds: list[dict] = []
        tools: dict[str, dict] = {}
        fences: list[int] = []
        resumed: list[tuple[int, int]] = []
        response_starts: list[tuple[int, str]] = []
        stopped_at = None

        for index, event in enumerate(events):
            kind = event["kind"]
            sequence = event["audio_sequence"]
            timestamp = event["received_monotonic_ns"]
            if kind == "llm_start":
                rounds.append(
                    {
                        "round_trip": len(rounds) + 1,
                        "event_index": index,
                        "started_monotonic_ns": timestamp,
                        "ended_monotonic_ns": None,
                        "first_pcm_after_start": packet_by_seq.get(sequence + 1),
                    }
                )
            elif kind == "llm_end" and rounds and rounds[-1]["ended_monotonic_ns"] is None:
                rounds[-1]["ended_monotonic_ns"] = timestamp
            elif kind in {"tool_start", "tool_end"} and isinstance(event.get("tool_call_id"), str):
                tool_id = event["tool_call_id"]
                tool = tools.setdefault(tool_id, {"tool_call_id": tool_id, "tool_name": None})
                tool["tool_name"] = tool["tool_name"] or event.get("tool_name")
                tool.setdefault(kind, (index, timestamp, bool(event.get("cancelled"))))
            elif kind == "speech_end":
                stopped_at = sequence
            elif kind == "speech_start":
                if stopped_at == sequence and sequence in packet_by_seq and sequence + 1 in packet_by_seq:
                    resumed.append((index, sequence + 1))
                stopped_at = None
            elif kind == "response_start" and isinstance(event.get("response_id"), str):
                response_starts.append((index, event["response_id"]))
            elif kind in {"interruption", "user_start"} or (
                kind == "response_end" and event.get("status") in {"cancelled", "failed", "incomplete"}
            ):
                fences.append(index)
                stopped_at = None

        for position, llm_round in enumerate(rounds):
            pcm = llm_round["first_pcm_after_start"]
            if pcm is None:
                continue
            next_round = rounds[position + 1]["event_index"] if position + 1 < len(rounds) else len(events)
            next_fence = next((fence for fence in fences if fence > llm_round["event_index"]), len(events))
            boundary = min(next_round, next_fence)
            if boundary < len(events) and pcm["received_monotonic_ns"] >= events[boundary]["received_monotonic_ns"]:
                llm_round["first_pcm_after_start"] = None

        followups = [
            self._tool_report(tool, packets, packet_by_seq, rounds, response_starts, fences, resumed, attributed)
            for tool in tools.values()
        ]
        return {
            **{k: v for k, v in raw.items() if k not in {"event_start", "event_end", "audio_start", "audio_end"}},
            "first_audio_received_monotonic_ns": first_audio,
            "first_audio_latency_ms": _ms(raw["input_end_monotonic_ns"], first_audio),
            "audio_packets_observed": len(packets),
            "llm_round_trips": len(rounds) if rounds else None,
            "llm_round_trips_completed": sum(r["ended_monotonic_ns"] is not None for r in rounds) if rounds else None,
            "tool_calls_observed": len(tools),
            "tool_announcements_observed": sum(e["kind"] == "tool_announced" for e in events),
            "round_trips": [
                {
                    "round_trip": r["round_trip"],
                    "started_monotonic_ns": r["started_monotonic_ns"],
                    "ended_monotonic_ns": r["ended_monotonic_ns"],
                    "first_pcm_after_start_monotonic_ns": (r["first_pcm_after_start"] or {}).get(
                        "received_monotonic_ns"
                    ),
                }
                for r in rounds
            ],
            "tool_followups": followups,
        }

    def _tool_report(
        self,
        tool: dict,
        packets: list[dict],
        packet_by_seq: dict[int, dict],
        rounds: list[dict],
        response_starts: list[tuple[int, str]],
        fences: list[int],
        resumed: list[tuple[int, int]],
        attributed: dict[int, dict],
    ) -> dict:
        started = tool.get("tool_start")
        completed = tool.get("tool_end")
        start_ns = started[1] if started else None
        end_ns = completed[1] if completed else None
        cancelled = bool(completed and completed[2])
        evidence = ["matched_tool_markers"] if started and completed else ["incomplete_tool_markers"]

        following_round = None
        candidate_audio = None
        boundary_audio = None
        response_id = None
        if started and completed and not cancelled and started[0] < completed[0]:
            fence = next((i for i in fences if i > started[0]), math.inf)
            following_round = next((r for r in rounds if completed[0] < r["event_index"] < fence), None)
            if following_round:
                evidence.append("later_native_llm_start")
                next_round = next(
                    (r["event_index"] for r in rounds if r["event_index"] > following_round["event_index"]),
                    math.inf,
                )
                candidate_audio = following_round["first_pcm_after_start"]
                resume_sequence = next(
                    (
                        sequence
                        for index, sequence in resumed
                        if following_round["event_index"] < index < min(next_round, fence)
                    ),
                    None,
                )
                # Require the first subsequent PCM at the bracketed boundary.
                # Another segment later in the response cannot establish it.
                if candidate_audio and candidate_audio["audio_sequence"] == resume_sequence:
                    boundary_audio = candidate_audio
            else:
                for index, candidate_id in response_starts:
                    if completed[0] < index < fence:
                        response_id = candidate_id
                        candidate_audio = next((p for p in packets if p.get("response_id") == response_id), None)
                        evidence.append("later_realtime_response")
                        break
        elif cancelled:
            evidence.append("tool_cancelled")

        audio = boundary_audio or candidate_audio
        audio_seq = audio["audio_sequence"] if audio else None
        prior = packet_by_seq.get(audio_seq - 1) if audio_seq is not None else None
        observed_audio_ns = audio["received_monotonic_ns"] if audio else None
        previous_audio_ns = prior["received_monotonic_ns"] if prior else None
        playback_gap_ms = None
        if boundary_audio:
            evidence.append("native_speech_stop_start_before_pcm")
            discontinuity = self.audio_discontinuity_after_sequence
            if self.local_dropped or (discontinuity is not None and audio_seq > discontinuity):
                evidence.append("incomplete_audio_evidence")
            elif boundary_audio["playback_deficit_ms"] is not None:
                playback_gap_ms = max(0.0, boundary_audio["playback_deficit_ms"])
                tagged = attributed.setdefault(audio_seq, {"tool_call_ids": []})
                tagged["tool_call_ids"].append(tool["tool_call_id"])
        elif candidate_audio:
            evidence.append("pcm_observed_after_marker_without_speech_alignment")
        else:
            evidence.append("no_followup_audio_observed")

        llm_start_ns = following_round["started_monotonic_ns"] if following_round else None
        return {
            "tool_call_id": tool["tool_call_id"],
            "tool_name": tool["tool_name"],
            "tool_started_monotonic_ns": start_ns,
            "tool_completed_monotonic_ns": end_ns,
            "tool_observed_ms": _ms(start_ns, end_ns) if started and completed and started[0] < completed[0] else None,
            "cancelled": cancelled,
            "next_llm_round_trip": following_round["round_trip"] if following_round else None,
            "next_llm_started_monotonic_ns": llm_start_ns,
            "tool_completion_to_llm_start_ms": _ms(end_ns, llm_start_ns),
            "realtime_response_id": response_id,
            "previous_audio_received_monotonic_ns": previous_audio_ns,
            "followup_audio_received_monotonic_ns": observed_audio_ns,
            "tool_completion_to_followup_audio_ms": _ms(end_ns, observed_audio_ns),
            "llm_start_to_followup_audio_ms": _ms(llm_start_ns, observed_audio_ns),
            "previous_to_followup_audio_received_ms": _ms(previous_audio_ns, observed_audio_ns),
            "playback_gap_ms": playback_gap_ms,
            "evidence": evidence,
        }

    def finish(self) -> dict:
        """Compute diagnostics from native event and audio receive order after the benchmark."""
        attributed: dict[int, dict] = {}
        turns = [self._turn_report(turn, attributed) for turn in self._turns]
        classified = []
        for gap in self.gaps:
            if tool := attributed.get(gap["audio_sequence"]):
                classified.append(
                    {
                        **gap,
                        **tool,
                        "classification": "tool_wait",
                        "confidence": "native_marker_context",
                        "evidence": ["later_llm_round_and_native_playback_boundary", "client_receive_order_only"],
                    }
                )
            else:
                classified.append(
                    {
                        **gap,
                        "classification": "unknown",
                        "confidence": "unavailable",
                        "evidence": ["native_markers_do_not_align_sentences"],
                    }
                )
        return {
            "schema_version": 2,
            "coverage": {
                "native_event_counts": dict(collections.Counter(e["protocol_event"] for e in self.events)),
                "local_dropped_events": self.local_dropped,
                "audio_discontinuity_after_sequence": self.audio_discontinuity_after_sequence,
            },
            "events": self.events,
            "audio_packets": self.audio_packets,
            "turns": turns,
            "glitch_events": classified,
            "glitch_summary": {
                label: {
                    "events": sum(g["classification"] == label for g in classified),
                    "deficit_ms": sum(g["deficit_ms"] for g in classified if g["classification"] == label),
                }
                for label in GAP_CLASSES
            },
        }


def session_errors(client: dict) -> list[dict]:
    """Extract structured errors, including legacy results without diagnostics."""
    collector = SessionDiagnostics()
    if isinstance(client.get("session_errors"), list):
        collector.errors = [
            {**error, "message": bounded_error_message(error.get("message", "Session error"))}
            for error in client["session_errors"]
            if isinstance(error, dict) and error.get("code") != "no_valid_turns"
        ]
    else:
        for message in client.get("rtvi_messages", []):
            payload = message.get("data")
            if isinstance(payload, dict):
                previous = len(collector.errors)
                collector.ingest(payload)
                for error in collector.errors[previous:]:
                    # Legacy recordings have no monotonic error timing. Preserve
                    # their recorded timestamp instead of inventing a new time.
                    error.update(received_unix_s=None, phase="unknown", recorded_timestamp=message.get("timestamp"))
        for turn in client.get("realtime_turn_metrics", []):
            if turn.get("error"):
                collector.add_error(str(turn["error"]), source="protocol", received_unix_s=None)
    if client.get("error") and client["error"] != NO_VALID_TURNS:
        message = bounded_error_message(client["error"])
        if not any(e["message"] == message for e in collector.errors):
            collector.add_error(message, source="client", component="client", received_unix_s=None)
    return collector.errors


def has_valid_response(client: dict) -> bool:
    """Use the benchmark's recorded valid-turn count and latency."""
    try:
        return client.get("average_latency") is not None and int(client.get("num_valid_turns") or 0) > 0
    except (TypeError, ValueError):
        return False


def is_hard_deadline(client: dict) -> bool:
    """Identify forced benchmark window expiry in new and legacy artifacts."""
    return str(client.get("error") or "").startswith("Hard deadline reached")


def quality_status(client: dict, reported_errors: list[dict] | None = None) -> str:
    """Separate usable completion from error-free completion and runtime failure."""
    error = str(client.get("error") or "")
    if error and error != NO_VALID_TURNS and not is_hard_deadline(client):
        return "failed"
    if not has_valid_response(client):
        return "no_response"
    if reported_errors is None:
        reported_errors = session_errors(client)
    if error or client.get("failed_turns", 0) or reported_errors:
        return "degraded"
    return "successful"


def write_diagnostic_reports(directory, clients: list[dict]) -> dict:
    """Write post-test event counts and session errors without dropping failed clients."""
    import csv
    import json

    gaps = []
    errors = []
    qualities = []
    turn_timing = []
    tool_followups = []
    for client in clients:
        identifier = client.get("stream_id", "unknown")
        diagnostics = client.get("diagnostics") or {}
        for gap in diagnostics.get("glitch_events", []):
            gaps.append({"client": identifier, **gap})
        for turn in diagnostics.get("turns", []):
            turn_timing.append(
                {"client": identifier, **{key: value for key, value in turn.items() if key != "tool_followups"}}
            )
            for followup in turn.get("tool_followups", []):
                tool_followups.append(
                    {"client": identifier, "turn": turn["turn"], "phase": turn.get("phase"), **followup}
                )
        reported_errors = session_errors(client)
        status = quality_status(client, reported_errors)
        for error in reported_errors:
            errors.append({"client": identifier, "session_status": status, **error})
        qualities.append(
            {
                "client": identifier,
                "status": status,
                "valid_turns": client.get("num_valid_turns", 0),
                "failed_turns": client.get("failed_turns", 0),
                "reported_errors": len(reported_errors),
                "raw_glitch_flag": bool(client.get("glitch_detected")),
                "captured_glitch_events": len(diagnostics.get("glitch_events", [])),
                "diagnostic_coverage": diagnostics.get("coverage", {}),
            }
        )
    reports = (
        ("glitch_events", gaps),
        ("session_errors", errors),
        ("session_quality", qualities),
        ("turn_timing", turn_timing),
        ("tool_followups", tool_followups),
    )
    for filename, items in reports:
        (directory / f"{filename}.json").write_text(json.dumps(items, indent=2), encoding="utf-8")
        columns = list(dict.fromkeys(key for item in items for key in item)) or ["client"]
        with (directory / f"{filename}.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(
                {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in item.items()} for item in items
            )
    measured = [gap for gap in gaps if gap.get("phase") == "measurement"]
    total_ms = sum(gap["deficit_ms"] for gap in measured)
    observed_labels = {gap["classification"] for gap in measured}
    labels = (*GAP_CLASSES, *(label for label in LEGACY_GAP_CLASSES if label in observed_labels))
    summary = {
        label: {
            "events": sum(g["classification"] == label for g in measured),
            "deficit_ms": sum(g["deficit_ms"] for g in measured if g["classification"] == label),
        }
        for label in labels
    }
    legacy_flagged = sum(q["raw_glitch_flag"] and not q["captured_glitch_events"] for q in qualities)
    lines = [
        f"Clients flagged without captured gap events: {legacy_flagged} (event count/duration unavailable).",
        "Measurement-window buffer deficits (>20ms), not listening-confirmed audio defects.",
        "Unknown indicates insufficient evidence to identify the context of a detected gap.",
        "Native boundary classifications describe the order observed by the client.",
        "Classification       Events  Event share  Deficit(ms)  Duration share",
    ]
    for label, values in summary.items():
        values["event_percent"] = 100 * values["events"] / len(measured) if measured else 0
        values["duration_percent"] = 100 * values["deficit_ms"] / total_ms if total_ms else 0
        lines.append(
            f"{label:20} {values['events']:6} {values['event_percent']:11.2f}% "
            f"{values['deficit_ms']:12.3f} {values['duration_percent']:13.2f}%"
        )
    (directory / "glitch_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "measurement": summary,
        "all_phase_events": len(gaps),
        "reported_errors": len(errors),
        "clients_with_errors": sum(q["reported_errors"] > 0 for q in qualities),
        "flagged_clients_without_event_evidence": legacy_flagged,
        "turn_timing_rows": len(turn_timing),
        "turns_with_llm_round_trip_evidence": sum(t.get("llm_round_trips") is not None for t in turn_timing),
        "tool_followup_rows": len(tool_followups),
    }
