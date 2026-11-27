"""Correlate client underruns with metadata probes; report evidence, not inferred certainty."""

import argparse
import ast
import bisect
import collections
import datetime
import json
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--stage")
parser.add_argument("--within-speech-only", action="store_true")
args = parser.parse_args()
root = args.root
clients = []
windows = []
for p in sorted((root / "runs").rglob("result_client*.json")):
    stage = next(
        (parent.name for parent in p.parents if (parent / "raw-summary.json").exists()),
        "unknown",
    )
    if args.stage and stage != args.stage:
        continue
    d = json.loads(p.read_text())
    peer = None
    glitches = []
    allowed = None
    if args.within_speech_only:
        q = root / "runs" / stage / "glitch-counts/underrun-events.json"
        allowed = (
            {
                e["timestamp"]
                for e in json.loads(q.read_text())
                if e["client"] == d["stream_id"]
                and e.get("lifecycle_class") == "within_speaking_segment"
            }
            if q.exists()
            else set()
        )
    for log in p.parent.glob("benchmark_client*.log"):
        for line in log.read_text(errors="replace").splitlines():
            m = re.search(r"diagnostic_connection local=(\(.+\))", line)
            if m:
                address = ast.literal_eval(m.group(1))
                peer = f"{address[0]}:{address[1]}"
            m = re.search(r"^\[([^]]+)\].*buffer underrun ([0-9.]+)ms", line)
            if m and (allowed is None or m.group(1) in allowed):
                t = (
                    datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
                    .replace(tzinfo=datetime.timezone.utc)
                    .timestamp()
                )
                glitches.append(
                    {"time": t, "utc": m.group(1), "underrun_ms": float(m.group(2))}
                )
                windows.append((t - 3, t + 1))
    clients.append(
        {"stage": stage, "client": d["stream_id"], "peer": peer, "glitches": glitches}
    )
windows.sort()
starts = [a for a, b in windows]


def near(t):
    i = bisect.bisect_right(starts, t)
    return any(a <= t <= b for a, b in windows[max(0, i - 3) : i])


peer_strings = [
    json.dumps(key, separators=(",", ":"))
    for c in clients
    if c["glitches"]
    for key in [c["peer"], c["client"]]
    if key
]
server = collections.defaultdict(list)
server_lag = collections.defaultdict(list)
client_lag = collections.defaultdict(list)
for p in (root / "diagnostics/server").rglob("events-*.jsonl"):
    text = p.read_text()
    if not any(peer in text for peer in peer_strings):
        continue
    for line in text.splitlines():
        if (
            not any(peer in line for peer in peer_strings)
            and '"kind":"event_loop_lag"' not in line
        ):
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = e["time_ns"] / 1e9
        if not near(t):
            continue
        if e["kind"] == "event_loop_lag":
            server_lag[(e.get("worker_id"), e["pid"])].append(e)
        elif e["kind"] != "event_loop_summary":
            server[e["session"]].append(e)
for stage_dir in (root / "diagnostics/client").iterdir():
    if args.stage and stage_dir.name != args.stage:
        continue
    for p in stage_dir.glob("events-*.jsonl"):
        es = []
        names = set()
        for line in p.open():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e["kind"] == "client_audio":
                names.add(e["session"])
            if e["kind"] == "event_loop_lag" and near(e["time_ns"] / 1e9):
                es.append(e)
        for name in names:
            client_lag[(stage_dir.name, name)].extend(es)
rows = []
for client in clients:
    events = sorted(
        server.get(client["client"], server.get(client["peer"], [])),
        key=lambda e: e["time_ns"],
    )
    for glitch in client["glitches"]:
        t = glitch["time"]
        around = [e for e in events if t - 2 <= e["time_ns"] / 1e9 <= t + 0.5]
        pids = {(e.get("worker_id"), e["pid"]) for e in around}
        sleeps = [e for e in around if e["kind"] == "transport_sleep"]
        wire = [e for e in around if e["kind"] == "wire_send_start"]
        pcm = [e for e in around if e["kind"] == "tts_pcm"]
        enqueue = [e for e in around if e["kind"] == "output_enqueue"]
        sl = [
            e["lag_ms"]
            for pid in pids
            for e in server_lag[pid]
            if t - 2 <= e["time_ns"] / 1e9 <= t + 0.5
        ]
        cl = [
            e["lag_ms"]
            for e in client_lag[(client["stage"], client["client"])]
            if t - 2 <= e["time_ns"] / 1e9 <= t + 0.5
        ]
        max_sleep = max((e["overshoot_ms"] for e in sleeps), default=0)
        max_server = max(sl, default=0)
        max_client = max(cl, default=0)
        wire_gaps = [
            {
                "gap_ms": (b["time_ns"] - a["time_ns"]) / 1e6,
                "previous_audio_ms": a.get("audio_ms"),
            }
            for a, b in zip(wire, wire[1:])
        ]
        pcm_gaps = [(b["time_ns"] - a["time_ns"]) / 1e6 for a, b in zip(pcm, pcm[1:])]
        mismatches = [
            e["configured_interval_ms"] - e["previous_packet_audio_ms"]
            for e in sleeps
            if e.get("previous_packet_audio_ms") is not None
        ]
        flags = []
        if max_sleep > 20:
            flags.append("transport_sleep_woke_more_than_20ms_late")
        if max_server > 20:
            flags.append("server_event_loop_delayed_more_than_20ms")
        if max_client > 20:
            flags.append("client_event_loop_delayed_more_than_20ms")
        if max(mismatches, default=0) > 0.5:
            flags.append("fixed_sleep_interval_exceeds_packet_audio_duration")
        if max(pcm_gaps, default=0) > 100 and max_server < 10 and max_sleep < 10:
            flags.append("upstream_pcm_gap_without_observed_server_loop_or_sleep_delay")
        if any(e["kind"] == "tts_input_tokens" for e in around):
            flags.append("new_sentence_session_in_same_time_window")
        rows.append(
            {
                **{k: v for k, v in client.items() if k != "glitches"},
                **glitch,
                "max_transport_sleep_overshoot_ms": max_sleep,
                "max_server_loop_lag_ms": max_server,
                "max_client_loop_lag_ms": max_client,
                "max_fixed_interval_minus_packet_ms": max(mismatches, default=None),
                "wire_gaps": wire_gaps,
                "pcm_receive_gaps_ms": pcm_gaps,
                "output_enqueue_events_in_window": len(enqueue),
                "evidence_flags": flags or ["no_decisive_timing_evidence"],
                "nearby_server_events": around,
            }
        )
report = {
    "scope": "Timing evidence near each detected underrun. Evidence flags are not proof of root cause. PCM receipt includes Python scheduling; missing lag evidence does not rule out CPU stalls.",
    "events": rows,
    "flag_counts": dict(
        collections.Counter(flag for row in rows for flag in row["evidence_flags"])
    ),
}
out = root / "attribution"
out.mkdir(exist_ok=True)
path = out / (
    f"{args.stage}-timing-evidence.json" if args.stage else "timing-evidence.json"
)
path.write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {
            "events": len(rows),
            "flag_counts": report["flag_counts"],
            "output": str(path),
        },
        indent=2,
    )
)
