# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Timestamped session evidence and conservative post-test gap classification."""

from __future__ import annotations

import collections
import math
import time
from typing import Any

from pipecat.utils.security.redaction import redact_secrets

DIAGNOSTICS_TYPE = "nva.perf.diagnostics"
GAP_CLASSES = ("within_sentence", "between_sentences", "tool_wait", "interruption", "unknown")
NO_VALID_TURNS = "No valid benchmark turns completed"
MAX_ERROR_MESSAGE_CHARS = 2048


def bounded_error_message(message: str) -> str:
    """Redact credential values before limiting the report text."""
    return redact_secrets(str(message)).strip()[:MAX_ERROR_MESSAGE_CHARS]


def _component(name: str) -> str:
    name = name.lower()
    for category, parts in (("asr", ("asr", "stt", "transcri")), ("tts", ("tts",)), ("llm", ("llm",))):
        if any(part in name for part in parts):
            return category
    return "unknown"


class SessionDiagnostics:
    """Collect metadata independently of the valid-turn latency population."""

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
        self.server_dropped = 0
        self.server_session: str | None = None
        self.last_sequence = 0
        self.sequence_gaps = 0
        self.audio_discontinuity_after_sequence: int | None = None

    def mark_audio_discontinuity(self) -> None:
        """Mark a point where received Realtime PCM was discarded outside playback."""
        if self.audio_discontinuity_after_sequence is None:
            self.audio_discontinuity_after_sequence = self.audio_sequence

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
        # RTVI emits the same ErrorFrame independently of the diagnostic observer.
        # A tool error remains separate from a provider error with identical text.
        for old in reversed(self.errors):
            if now is None or old.get("received_unix_s") is None or abs(now - old["received_unix_s"]) > 2:
                break
            duplicate_frame = {old["source"], source} == {"protocol", "pipecat_frame"} and (
                old["component"] == component or "unknown" in (old["component"], component)
            )
            duplicate_terminal = (
                source == "client"
                and entry.get("code") == "session_error"
                and old["source"] in {"handshake", "protocol", "pipecat_frame"}
            )
            if (
                len(entry["message"]) < MAX_ERROR_MESSAGE_CHARS
                and old["message"] == entry["message"]
                and (duplicate_frame or duplicate_terminal)
                and source not in old.get("also_reported_by", [])
            ):
                if source == "pipecat_frame":
                    previous_source = old["source"]
                    old.update({**entry, "received_unix_s": old["received_unix_s"], "phase": old["phase"]})
                else:
                    previous_source = source
                old["also_reported_by"] = sorted(set(old.get("also_reported_by", []) + [previous_source]))
                return
        self._append(self.errors, entry)

    def ingest(self, message: dict[str, Any]) -> None:
        """Consume RTVI envelopes, native Realtime errors, and diagnostic batches."""
        kind = str(message.get("type") or "")
        if kind == "server-message" and isinstance(message.get("data"), dict):
            self.ingest(message["data"])
            return
        if kind == DIAGNOSTICS_TYPE:
            if message.get("schema_version") != 1:
                self.local_dropped += 1
                return
            session = message.get("session_id")
            if self.server_session is not None and session != self.server_session:
                self.local_dropped += 1
                return
            self.server_session = session
            self.server_dropped = max(self.server_dropped, int(message.get("dropped_events") or 0))
            for event in message.get("events", []):
                if not isinstance(event, dict) or not isinstance(event.get("seq"), int):
                    self.local_dropped += 1
                    continue
                seq = event["seq"]
                if seq <= self.last_sequence:
                    continue
                self.sequence_gaps += max(0, seq - self.last_sequence - 1)
                self.last_sequence = seq
                event = dict(event)
                if event.get("kind") == "error":
                    event["message"] = bounded_error_message(event.get("message", "Pipecat error"))
                    self.add_error(
                        event["message"],
                        source="pipecat_frame",
                        component=event.get("component", "unknown"),
                        frame_id=event.get("frame_id"),
                        processor=event.get("processor"),
                        fatal=event.get("fatal"),
                        category=event.get("category"),
                        exception_type=event.get("exception_type"),
                        server_time_ns=event.get("server_time_ns"),
                    )
                elif event.get("kind") == "tool_end" and event.get("error"):
                    event["error"] = bounded_error_message(event["error"])
                    self.add_error(
                        event["error"], source="tool", component="tool", tool_call_id=event.get("tool_call_id")
                    )
                self._append(self.events, event)
            return
        if kind == "error" or kind.endswith(".failed"):
            detail = message.get("error", message.get("data", {}))
            if isinstance(detail, dict):
                text = detail.get("error") or detail.get("message") or str(detail)
                component = _component(str(detail.get("processor", "")) + str(text))
                self.add_error(
                    str(text),
                    source="protocol",
                    component=component,
                    fatal=detail.get("fatal"),
                    code=detail.get("code"),
                    event_type=kind,
                )
            else:
                self.add_error(str(detail), source="protocol", component=_component(str(detail)), event_type=kind)
        elif kind == "response.done":
            response = message.get("response", {})
            if response.get("status") in {"failed", "incomplete"}:
                self.add_error(
                    str(response.get("status_details") or response.get("status")), source="protocol", event_type=kind
                )

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
    ) -> None:
        """Record PCM arrival and the detector's actual cumulative deficit."""
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
        }
        self._append(self.audio_packets, packet)
        if detect and deficit_ms is not None and deficit_ms > 20:
            self._append(self.gaps, {**packet, "deficit_ms": deficit_ms})
        self.audio_ms += duration

    def _classify(self, gap: dict, starts: list[dict], ends: dict, boundaries: dict) -> dict:
        result = {**gap, "classification": "unknown", "confidence": "unavailable", "evidence": []}
        if self.local_dropped or self.server_dropped or self.sequence_gaps:
            result["evidence"].append("diagnostic_capture_incomplete")
            return result
        seq = gap["audio_sequence"]
        discontinuity = self.audio_discontinuity_after_sequence
        if discontinuity is not None and seq > discontinuity:
            result["evidence"].append("realtime_audio_discontinuity")
            return result
        preceding = [e for e in starts if e["audio_sequence"] < seq]
        if not preceding:
            result["evidence"].append("no_audio_epoch")
            return result
        epoch = preceding[-1]
        identifier = epoch.get("epoch")
        result["epoch"] = identifier
        baseline = next(
            (p["audio_ms_before"] for p in self.audio_packets if p["audio_sequence"] == epoch["audio_sequence"] + 1),
            None,
        )
        if baseline is None or identifier not in ends:
            result["evidence"].append("epoch_start_or_end_missing")
            return result
        end = ends[identifier]
        if discontinuity is not None and end["audio_sequence"] > discontinuity:
            result["evidence"].append("realtime_audio_discontinuity")
            return result
        if seq > end["audio_sequence"]:
            result["evidence"].append("outside_completed_audio_epoch")
            return result
        packets = [
            p for p in self.audio_packets if epoch["audio_sequence"] < p["audio_sequence"] <= end["audio_sequence"]
        ]
        client_ms = sum(p["duration_ms"] for p in packets)
        server_ms = end.get("output_audio_ms", -1) - epoch.get("output_audio_ms", 0)
        if abs(client_ms - server_ms) > 1 or len(packets) != end["audio_sequence"] - epoch["audio_sequence"]:
            result["evidence"].append("client_server_audio_alignment_mismatch")
            return result
        offset = gap["audio_ms_before"] - baseline
        result["epoch_audio_offset_ms"] = offset
        units = sorted(boundaries.get(identifier, []), key=lambda e: e["source_audio_ms"])
        # First packet after a tool-driven LLM followup belongs to a separate wait,
        # rather than a claim that audio stalled inside the following sentence.
        if seq == epoch["audio_sequence"] + 1 and epoch.get("tool_call_ids"):
            result.update(
                classification="tool_wait",
                confidence="frame_evidence",
                evidence=["first_audio_of_tool_followup", "not_proof_of_tool_execution_duration"],
            )
            return result
        previous_seq = seq - 1
        if any(
            e.get("kind") == "interruption" and previous_seq <= e.get("audio_sequence", -1) < seq for e in self.events
        ):
            result.update(
                classification="interruption",
                confidence="frame_evidence",
                evidence=["interruption_near_playback_boundary"],
            )
            return result
        if not units or not all(e.get("sentence_aligned") for e in units):
            result["evidence"].append("sentence_audio_alignment_unavailable")
            return result
        first_packet = seq == epoch["audio_sequence"] + 1
        if first_packet:
            previous = next(
                (
                    e
                    for e in reversed(preceding[:-1])
                    if ends.get(e.get("epoch"), {}).get("audio_sequence") == epoch["audio_sequence"]
                ),
                None,
            )
            previous_units = boundaries.get(previous.get("epoch"), []) if previous else []
            if not previous_units or not all(e.get("sentence_aligned") for e in previous_units):
                result["evidence"].append("preceding_sentence_alignment_unavailable")
                return result
        if any(abs(e["source_audio_ms"] - offset) <= 1 for e in units):
            result.update(
                classification="between_sentences",
                confidence="audio_offset",
                evidence=["sentence_boundary_at_playback_cursor"],
            )
        elif seq != epoch["audio_sequence"] + 1 and any(e["source_audio_ms"] < offset for e in units):
            result.update(
                classification="within_sentence",
                confidence="audio_offset",
                evidence=["playback_cursor_inside_aligned_sentence_unit"],
            )
        else:
            result["evidence"].append("no_bracketing_sentence_boundary")
        return result

    def finish(self) -> dict:
        """Classify after delayed semantic markers have been received."""
        starts = sorted(
            (e for e in self.events if e.get("kind") == "audio_epoch_start"), key=lambda e: e["audio_sequence"]
        )
        ends = {e.get("epoch"): e for e in self.events if e.get("kind") == "audio_epoch_end"}
        boundaries = collections.defaultdict(list)
        for event in self.events:
            if event.get("kind") == "sentence_boundary":
                boundaries[event.get("epoch")].append(event)
        classified = [self._classify(g, starts, ends, boundaries) for g in self.gaps]
        return {
            "schema_version": 1,
            "server_session_id": self.server_session,
            "coverage": {
                "server_events_available": self.server_session is not None,
                "server_dropped_events": self.server_dropped,
                "local_dropped_events": self.local_dropped,
                "sequence_gaps": self.sequence_gaps,
                "audio_discontinuity_after_sequence": self.audio_discontinuity_after_sequence,
            },
            "events": self.events,
            "audio_packets": self.audio_packets,
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
    for client in clients:
        identifier = client.get("stream_id", "unknown")
        diagnostics = client.get("diagnostics") or {}
        for gap in diagnostics.get("glitch_events", []):
            gaps.append({"client": identifier, **gap})
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
    for filename, items in (("glitch_events", gaps), ("session_errors", errors), ("session_quality", qualities)):
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
    summary = {
        label: {
            "events": sum(g["classification"] == label for g in measured),
            "deficit_ms": sum(g["deficit_ms"] for g in measured if g["classification"] == label),
        }
        for label in GAP_CLASSES
    }
    legacy_flagged = sum(q["raw_glitch_flag"] and not q["captured_glitch_events"] for q in qualities)
    lines = [
        f"Clients flagged without captured gap events: {legacy_flagged} (event count/duration unavailable).",
        "Measurement-window buffer deficits (>20ms), not listening-confirmed audio defects.",
        "Unknown means insufficient or unaligned sentence evidence, not a clean pass.",
        "Boundary classifications describe context, not proven root cause.",
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
    }
