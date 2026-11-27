"""Count detected playback-buffer underruns for every completed client."""

import argparse
import bisect
import csv
import datetime
import json
import re
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
rows = []
events = []
for p in sorted(args.root.rglob("result_client*.json")):
    d = json.loads(p.read_text())
    level = next(
        (
            int(m.group(1))
            for s in p.parts
            if (m := re.fullmatch(r"run_(\d+)_clients", s))
        ),
        None,
    )
    stage = ""
    duration = None
    for parent in p.parents:
        metadata = parent / "raw-summary.json"
        if metadata.exists():
            run = json.loads(metadata.read_text())
            level = run.get("concurrency", level)
            stage = parent.name
            duration = run.get("measurement_seconds")
            break
    provider = [
        m.get("data", {}).get("data", {}).get("error", "provider error")
        for m in d.get("rtvi_messages", [])
        if m.get("type") == "error"
    ]
    client_events = []
    logs = sorted(p.parent.glob("benchmark_client*.log"))
    markers = []
    for log in logs:
        for line in log.read_text(errors="replace").splitlines():
            marker = re.search(
                r'^\[([^]]+)\].*"type":\s*"(bot-started-speaking|bot-stopped-speaking|bot-llm-started|llm-function-call-started)"',
                line,
            )
            if marker:
                marker_time = (
                    datetime.datetime.strptime(marker.group(1), "%Y-%m-%d %H:%M:%S.%f")
                    .replace(tzinfo=datetime.timezone.utc)
                    .timestamp()
                )
                markers.append((marker_time, marker.group(2)))
    markers.sort()
    speaking = [
        (t, kind)
        for t, kind in markers
        if kind in ["bot-started-speaking", "bot-stopped-speaking"]
    ]
    speaking_times = [t for t, kind in speaking]
    for log in logs:
        for line in log.read_text(errors="replace").splitlines():
            match = re.search(r"^\[([^]]+)\].*buffer underrun ([0-9.]+)ms", line)
            if match:
                event_time = (
                    datetime.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f")
                    .replace(tzinfo=datetime.timezone.utc)
                    .timestamp()
                )
                start = d.get("metrics_start_time")
                seconds = d.get("test_duration")
                phase = "unknown"
                if start is not None and seconds is not None:
                    phase = (
                        "connection_warmup"
                        if event_time < start
                        else "measurement"
                        if event_time <= start + seconds
                        else "final_reply_drain"
                    )
                index = bisect.bisect_right(speaking_times, event_time) - 1
                lifecycle = "unknown"
                if index >= 0:
                    if speaking[index][1] == "bot-started-speaking":
                        lifecycle = "within_speaking_segment"
                    else:
                        lifecycle = "after_bot_stopped"
                        stop_time = speaking[index][0]
                        previous_start = next(
                            (
                                t
                                for t, kind in reversed(speaking[:index])
                                if kind == "bot-started-speaking"
                            ),
                            stop_time,
                        )
                        tool = any(
                            previous_start - 2 <= t <= stop_time
                            and kind == "llm-function-call-started"
                            for t, kind in markers
                        )
                        followup = any(
                            stop_time <= t <= event_time + 0.1
                            and kind == "bot-llm-started"
                            for t, kind in markers
                        )
                        if tool and followup:
                            lifecycle = "tool_followup_boundary"
                event = {
                    "stage": stage,
                    "concurrency": level,
                    "client": d["stream_id"],
                    "timestamp": match.group(1),
                    "time_epoch": event_time,
                    "phase": phase,
                    "lifecycle_class": lifecycle,
                    "underrun_ms": float(match.group(2)),
                }
                client_events.append(event)
                events.append(event)
    durations = [e["underrun_ms"] for e in client_events]
    rows.append(
        {
            "stage": stage,
            "concurrency": level,
            "measurement_seconds": duration,
            "client": d["stream_id"],
            "valid_turns": d.get("num_valid_turns", 0),
            "failed_turns": d.get("failed_turns", 0),
            "provider_error_count": len(provider),
            "client_error": d.get("error"),
            "glitch_flag": bool(d.get("glitch_detected")),
            "client_logs_available": bool(logs),
            "underrun_events": len(durations) if logs else None,
            "measurement_underrun_events": sum(
                e["phase"] == "measurement" for e in client_events
            )
            if logs
            else None,
            "warmup_underrun_events": sum(
                e["phase"] == "connection_warmup" for e in client_events
            )
            if logs
            else None,
            "drain_underrun_events": sum(
                e["phase"] == "final_reply_drain" for e in client_events
            )
            if logs
            else None,
            "unknown_phase_underrun_events": sum(
                e["phase"] == "unknown" for e in client_events
            )
            if logs
            else None,
            "within_speech_underrun_events": sum(
                e["lifecycle_class"] == "within_speaking_segment" for e in client_events
            )
            if logs
            else None,
            "tool_boundary_gap_events": sum(
                e["lifecycle_class"] == "tool_followup_boundary" for e in client_events
            )
            if logs
            else None,
            "other_boundary_gap_events": sum(
                e["lifecycle_class"] == "after_bot_stopped" for e in client_events
            )
            if logs
            else None,
            "unknown_lifecycle_events": sum(
                e["lifecycle_class"] == "unknown" for e in client_events
            )
            if logs
            else None,
            "measurement_within_speech_underrun_events": sum(
                e["lifecycle_class"] == "within_speaking_segment"
                and e["phase"] == "measurement"
                for e in client_events
            )
            if logs
            else None,
            "max_within_speech_underrun_ms": max(
                (
                    e["underrun_ms"]
                    for e in client_events
                    if e["lifecycle_class"] == "within_speaking_segment"
                ),
                default=0,
            )
            if logs
            else None,
            "sum_underrun_ms": sum(durations) if logs else None,
            "max_underrun_ms": max(durations, default=0) if logs else None,
            "mean_underrun_ms": (statistics.mean(durations) if durations else 0)
            if logs
            else None,
            "provider_errors": provider,
            "result_file": str(p),
        }
    )
args.output.mkdir(parents=True, exist_ok=True)
with (args.output / "per-client-glitches.csv").open("w") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0])
        if rows
        else ["concurrency", "client", "underrun_events"],
    )
    writer.writeheader()
    writer.writerows(rows)
levels = []
for stage, level in sorted(
    {(r["stage"], r["concurrency"]) for r in rows},
    key=lambda x: (x[0], -1 if x[1] is None else x[1]),
):
    cs = [r for r in rows if r["concurrency"] == level and r["stage"] == stage]
    ds = [
        e["underrun_ms"]
        for e in events
        if e["concurrency"] == level and e["stage"] == stage
    ]
    levels.append(
        {
            "stage": stage,
            "concurrency": level,
            "clients": len(cs),
            "valid_turns": sum(c["valid_turns"] for c in cs),
            "failed_turns": sum(c["failed_turns"] for c in cs),
            "clients_with_underruns": sum(bool(c["underrun_events"]) for c in cs),
            "observed_underrun_events": len(ds),
            "underrun_events": len(ds)
            if all(c["client_logs_available"] for c in cs)
            else None,
            "measurement_underrun_events": sum(
                c["measurement_underrun_events"] for c in cs
            )
            if all(c["client_logs_available"] for c in cs)
            else None,
            "warmup_underrun_events": sum(c["warmup_underrun_events"] for c in cs)
            if all(c["client_logs_available"] for c in cs)
            else None,
            "drain_underrun_events": sum(c["drain_underrun_events"] for c in cs)
            if all(c["client_logs_available"] for c in cs)
            else None,
            "within_speech_underrun_events": sum(
                c["within_speech_underrun_events"] for c in cs
            )
            if all(c["client_logs_available"] for c in cs)
            else None,
            "tool_boundary_gap_events": sum(c["tool_boundary_gap_events"] for c in cs)
            if all(c["client_logs_available"] for c in cs)
            else None,
            "other_boundary_gap_events": sum(c["other_boundary_gap_events"] for c in cs)
            if all(c["client_logs_available"] for c in cs)
            else None,
            "measurement_within_speech_underrun_events": sum(
                c["measurement_within_speech_underrun_events"] for c in cs
            )
            if all(c["client_logs_available"] for c in cs)
            else None,
            "event_counts_complete": all(c["client_logs_available"] for c in cs),
            "clients_with_event_logs": sum(c["client_logs_available"] for c in cs),
            "events_per_client_min": min(
                (c["underrun_events"] for c in cs if c["client_logs_available"]),
                default=None,
            ),
            "events_per_client_median": statistics.median(
                c["underrun_events"] for c in cs if c["client_logs_available"]
            )
            if any(c["client_logs_available"] for c in cs)
            else None,
            "events_per_client_max": max(
                (c["underrun_events"] for c in cs if c["client_logs_available"]),
                default=None,
            ),
            "within_speech_events_per_client_median": statistics.median(
                c["within_speech_underrun_events"]
                for c in cs
                if c["client_logs_available"]
            )
            if any(c["client_logs_available"] for c in cs)
            else None,
            "within_speech_events_per_client_max": max(
                (
                    c["within_speech_underrun_events"]
                    for c in cs
                    if c["client_logs_available"]
                ),
                default=None,
            ),
            "max_within_speech_underrun_ms": max(
                (
                    c["max_within_speech_underrun_ms"]
                    for c in cs
                    if c["client_logs_available"]
                ),
                default=None,
            ),
            "max_underrun_ms": max(ds, default=0),
            "sum_underrun_ms": sum(ds),
            "provider_error_count": sum(c["provider_error_count"] for c in cs),
        }
    )
report = {
    "definition": "Each logged playback-buffer deficit exceeding 20 ms is one detected underrun event. Total counts include connection warmup and final-reply drain; phase-specific counts are provided separately. Timestamps are interpreted in the host UTC timezone. Lifecycle classes distinguish deficits inside a bot-speaking segment from tool-followup boundaries, other after-stop gaps, and unknown cases. Boundary labels alone do not prove that every pause is intentional. These are not confirmed distinct perceptual audio defects.",
    "levels": levels,
    "clients": rows,
}
(args.output / "per-client-glitches.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
(args.output / "underrun-events.json").write_text(json.dumps(events, indent=2) + "\n")
lines = [
    "Stage  Concurrency  Clients  RawEvents  InSpeechEvents  ToolBoundaryGaps  OtherBoundaryGaps  MeasuredInSpeechEvents  InSpeechPerClientMedian  InSpeechPerClientMax  WorstInSpeechMs  RawPerClientMedian  RawPerClientMax  WorstRawMs  ValidTurns  FailedTurns  ProviderErrors"
]
for level in levels:
    lines.append(
        "  ".join(
            str(level.get(k, "N/A"))
            for k in [
                "stage",
                "concurrency",
                "clients",
                "underrun_events",
                "within_speech_underrun_events",
                "tool_boundary_gap_events",
                "other_boundary_gap_events",
                "measurement_within_speech_underrun_events",
                "within_speech_events_per_client_median",
                "within_speech_events_per_client_max",
                "max_within_speech_underrun_ms",
                "events_per_client_median",
                "events_per_client_max",
                "max_underrun_ms",
                "valid_turns",
                "failed_turns",
                "provider_error_count",
            ]
        )
    )
(args.output / "glitch-results.txt").write_text("\n".join(lines) + "\n")
print(json.dumps({"levels": levels}, indent=2))
