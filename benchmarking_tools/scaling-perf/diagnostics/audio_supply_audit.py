"""Offline server-side PCM supply, output readiness and synthesis-rate accounting.

Reads metadata probes only. No model inference or audio playback is performed.
PCM arrival is measured at the pipeline, not inside the model's GPU engine.
"""

import argparse
import ast
import bisect
import collections
import csv
import json
import statistics
from pathlib import Path
import re


def audio_ms(event):
    rate = event.get("sample_rate")
    channels = event.get("channels") or 1
    if not rate or not channels:
        return None
    return event["nbytes"] * 1000 / (2 * channels * rate)


def supply_stats(chunks, prebuffer_ms=0):
    """Counterfactual continuous playback from a chosen initial audio buffer."""
    if not chunks:
        return {"chunks": 0, "audio_ms": 0, "deficits": []}
    total = 0.0
    start = None
    for index, (timestamp, duration) in enumerate(chunks):
        total += duration
        if total >= prebuffer_ms:
            start = index
            break
    if start is None:
        start = len(chunks) - 1
    credit = sum(d for _, d in chunks[: start + 1])
    previous = chunks[start][0]
    deficits = []
    for timestamp, duration in chunks[start + 1 :]:
        credit -= (timestamp - previous) / 1e6
        if credit < -20:
            deficits.append({"time_ns": timestamp, "deficit_ms": -credit})
        credit = max(0, credit) + duration
        previous = timestamp
    return {
        "chunks": len(chunks),
        "audio_ms": sum(d for _, d in chunks),
        "playback_start_ns": chunks[start][0],
        "prebuffer_ms": prebuffer_ms,
        "deficits": deficits,
    }


def curve(events):
    times = []
    amounts = []
    total = 0
    for e in events:
        duration = audio_ms(e)
        if duration is not None:
            total += duration
            times.append(e["time_ns"])
            amounts.append(total)
    return times, amounts


def available_at(c, when):
    i = bisect.bisect_right(c[0], when) - 1
    return c[1][i] if i >= 0 else 0.0


def ready_at(c, required):
    i = bisect.bisect_left(c[1], required - 0.05)
    return c[0][i] if i < len(c[0]) else None


def distribution(values):
    xs = sorted(x for x in values if x is not None)
    if not xs:
        return None

    def percentile(p):
        return xs[min(len(xs) - 1, max(0, __import__("math").ceil(len(xs) * p) - 1))]

    return {
        "count": len(xs),
        "min": xs[0],
        "p05": percentile(0.05),
        "median": statistics.median(xs),
        "p95": percentile(0.95),
        "max": xs[-1],
        "mean": statistics.mean(xs),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--stage", required=True)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    root = a.root
    out = a.output or root / "audio-supply" / a.stage
    out.mkdir(parents=True, exist_ok=True)
    clients = {}
    by_session = collections.defaultdict(list)
    for path in (root / "runs" / a.stage).rglob("result_client*.json"):
        d = json.loads(path.read_text())
        peer = None
        for log in path.parent.glob("benchmark_client*.log"):
            for line in log.open():
                match = re.search(r"diagnostic_connection local=(\(.+\))", line)
                if match:
                    addr = ast.literal_eval(match.group(1))
                    peer = f"{addr[0]}:{addr[1]}"
                    break
        entry = {
            "client": d["stream_id"],
            "peer": peer,
            "start": d.get("metrics_start_time", 0),
            "duration": d.get("test_duration", 0),
            "events": [],
            "result": d,
        }
        clients[d["stream_id"]] = entry
        for key in set([d["stream_id"], peer]) - {None}:
            by_session[key].append(entry)
    desired = {
        "session_attach",
        "tts_pcm",
        "tts_input_tokens",
        "tts_text_ready",
        "tts_provider_input",
        "llm_output",
        "tts_control",
        "output_control",
        "output_enqueue",
        "wire_send_start",
        "transport_sleep",
        "wire_send_end",
        "audio_socket_send_start",
        "audio_socket_send_end",
    }
    lag_by_pid = collections.defaultdict(list)
    for path in (root / "diagnostics/server").rglob("events-*.jsonl"):
        for line in path.open():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e["kind"] == "event_loop_lag":
                lag_by_pid[e["pid"]].append((e["time_ns"], e["lag_ms"]))
                continue
            if e["kind"] not in desired or e.get("session") not in by_session:
                continue
            t = e["time_ns"] / 1e9
            for client in by_session[e["session"]]:
                if (
                    client["start"] - 40
                    <= t
                    <= client["start"] + client["duration"] + 400
                ):
                    client["events"].append(e)
    client_probes = collections.defaultdict(list)
    for path in (root / "diagnostics/client" / a.stage).rglob("events-*.jsonl"):
        for line in path.open():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event["kind"] == "client_audio":
                client_probes[event["session"]].append(event)
    for events in client_probes.values():
        events.sort(key=lambda event: event["time_ns"])
    requests = []
    chunks_rows = []
    gap_rows = []
    event_path = root / "runs" / a.stage / "glitch-counts/underrun-events.json"
    observed = json.loads(event_path.read_text()) if event_path.exists() else []
    for client in clients.values():
        es = sorted(client["events"], key=lambda e: e["time_ns"])
        active = None
        pending = None
        local_requests = []
        for e in es:
            if e["kind"] == "tts_input_tokens":
                pending = e
            if e["kind"] == "tts_control" and e["event"] == "audio.start":
                active = {
                    "client": client["client"],
                    "request_id": e.get("request_id"),
                    "context": e.get("context"),
                    "start_ns": e["time_ns"],
                    "input_ns": pending["time_ns"]
                    if pending
                    and pending.get("context") == e.get("context")
                    and 0 < e["time_ns"] - pending["time_ns"] < 30e9
                    else None,
                    "pcm": [],
                    "done_ns": None,
                    "error": False,
                }
                pending = None
                local_requests.append(active)
            elif e["kind"] == "tts_pcm" and active is not None:
                if e.get("context") == active["context"]:
                    active["pcm"].append(e)
            elif e["kind"] == "tts_control" and active is not None:
                if e["event"] == "error" or e.get("error"):
                    active["error"] = True
                if e["event"] == "audio.done":
                    active["done_ns"] = e["time_ns"]
                if e["event"] == "session.done":
                    active["done_ns"] = active["done_ns"] or e["time_ns"]
                    active = None
        for req in local_requests:
            pcs = req["pcm"]
            pairs = [
                (e["time_ns"], audio_ms(e)) for e in pcs if audio_ms(e) is not None
            ]
            first = pairs[0][0] if pairs else None
            last = pairs[-1][0] if pairs else None
            end = req["done_ns"]
            total = sum(d for _, d in pairs)
            elapsed = (end - req["start_ns"]) / 1e6 if end else None
            stats = supply_stats(pairs, 399.092970521542)
            row = {k: v for k, v in req.items() if k != "pcm"}
            row.update(
                {
                    "stage": a.stage,
                    "phase": "measurement"
                    if client["start"]
                    <= req["start_ns"] / 1e9
                    <= client["start"] + client["duration"]
                    else "outside_measurement",
                    "audio_ms": total,
                    "chunks": len(pairs),
                    "first_pcm_from_audio_start_ms": (first - req["start_ns"]) / 1e6
                    if first
                    else None,
                    "first_pcm_from_input_ms": (first - req["input_ns"]) / 1e6
                    if first and req["input_ns"]
                    else None,
                    "request_wall_ms": elapsed,
                    "audio_per_request_wall_rtf_x": total / elapsed
                    if elapsed and elapsed > 0
                    else None,
                    "after_first_chunk_rtf_x": sum(d for _, d in pairs[1:])
                    / ((last - first) / 1e6)
                    if first and last > first
                    else None,
                    "within_request_counterfactual_deficits_with_399ms_prebuffer": len(
                        stats["deficits"]
                    ),
                    "worst_counterfactual_deficit_ms": max(
                        (x["deficit_ms"] for x in stats["deficits"]), default=0
                    ),
                }
            )
            requests.append(row)
            for index, (timestamp, duration) in enumerate(pairs):
                prior = pairs[index - 1] if index else None
                chunks_rows.append(
                    {
                        "stage": a.stage,
                        "client": client["client"],
                        "request_id": req["request_id"],
                        "context": req["context"],
                        "chunk": index,
                        "time_ns": timestamp,
                        "audio_ms": duration,
                        "interarrival_ms": (timestamp - prior[0]) / 1e6
                        if prior
                        else None,
                        "previous_chunk_audio_ms": prior[1] if prior else None,
                    }
                )
        starts = [
            e
            for e in es
            if e["kind"] == "output_control"
            and e.get("frame_type") == "TTSStartedFrame"
            and e.get("context")
        ]
        for gap in observed:
            if (
                gap["client"] != client["client"]
                or gap.get("lifecycle_class") != "within_speaking_segment"
            ):
                continue
            t = int(gap["time_epoch"] * 1e9)
            eligible = [e for e in starts if e["time_ns"] <= t]
            row = {
                "stage": a.stage,
                "client": client["client"],
                "utc": gap["timestamp"],
                "client_deficit_ms": gap["underrun_ms"],
                "phase": gap.get("phase"),
                "coverage": "incomplete",
            }
            if not eligible:
                gap_rows.append(row)
                continue
            begin = eligible[-1]
            context = begin["context"]
            until = min(
                (e["time_ns"] for e in starts if e["time_ns"] > begin["time_ns"]),
                default=10**30,
            )
            wires = [
                e
                for e in es
                if e["kind"] == "wire_send_start"
                and begin["time_ns"] <= e["time_ns"] < until
            ]
            all_wires = [event for event in es if event["kind"] == "wire_send_start"]
            arrivals = client_probes.get(client["client"], [])
            candidates = [
                i
                for i, event in enumerate(arrivals)
                if abs(event["time_ns"] - t) < 5e6
                and event.get("last_update_wall_s") is not None
                and (
                    event["time_ns"] / 1e9
                    - event["last_update_wall_s"]
                    - event["buffer_before_s"]
                )
                * 1000
                > 19
            ]
            probe_index = (
                min(
                    candidates,
                    key=lambda i: abs(
                        (
                            arrivals[i]["time_ns"] / 1e9
                            - arrivals[i]["last_update_wall_s"]
                            - arrivals[i]["buffer_before_s"]
                        )
                        * 1000
                        - gap["underrun_ms"]
                    ),
                )
                if candidates
                else None
            )
            exact_match = (
                probe_index is not None
                and probe_index < len(all_wires)
                and all(
                    abs(audio_ms(arrivals[i]) - audio_ms(all_wires[i])) < 0.05
                    for i in range(probe_index + 1)
                )
            )
            matched = all_wires[probe_index] if exact_match else None
            index = wires.index(matched) if matched in wires else -1
            row["packet_match"] = (
                "validated_duration_sequence_and_client_buffer_probe"
                if exact_match
                else "unmatched"
            )
            if index < 0 and wires:
                index = min(
                    range(len(wires)), key=lambda i: abs(wires[i]["time_ns"] - t)
                )
                row["packet_match"] = "nearest_timestamp_only"
            if index < 1 or (
                row["packet_match"] == "nearest_timestamp_only"
                and abs(wires[index]["time_ns"] - t) > 50e6
            ):
                gap_rows.append(row)
                continue
            now, previous = wires[index], wires[index - 1]
            packet_id = now.get("queue_frame_id")
            socket_end = next(
                (e for e in es if e["kind"] == "audio_socket_send_end"
                 and e.get("queue_frame_id") == packet_id),
                None,
            ) if packet_id is not None else None
            deadline = (
                int(
                    (
                        arrivals[probe_index]["last_update_wall_s"]
                        + arrivals[probe_index]["buffer_before_s"]
                    )
                    * 1e9
                )
                if exact_match
                else t - int(gap["underrun_ms"] * 1e6)
            )
            sent = sum(audio_ms(e) for e in wires[:index])
            required = sent + audio_ms(now)
            pc = curve(
                [
                    e
                    for e in es
                    if e["kind"] == "tts_pcm" and e.get("context") == context
                ]
            )
            eq = curve(
                [
                    e
                    for e in es
                    if e["kind"] == "output_enqueue" and e.get("context") == context
                ]
            )
            pr = ready_at(pc, required)
            er = ready_at(eq, required)
            p_left = available_at(pc, deadline) - sent
            e_left = available_at(eq, deadline) - sent
            label = "insufficient_coverage"
            if pr is not None and er is not None and min(p_left, e_left) > -40:
                if pr > deadline + 20e6:
                    label = (
                        "not_enough_provider_pcm_for_next_packet_by_playback_deadline"
                    )
                elif er > deadline + 20e6:
                    label = "pcm_available_but_late_at_output_processor"
                elif now["time_ns"] > deadline + 20e6:
                    label = "audio_at_output_but_late_wire_send"
                elif (socket_end and socket_end["time_ns"] > deadline + 20e6
                      and socket_end["elapsed_ms"] > 20):
                    label = "websocket_send_crossed_playback_deadline"
                else:
                    label = "server_wire_gap_not_over_20ms_check_client_or_buffer_model"
            near = [
                e
                for e in es
                if e["kind"] == "transport_sleep"
                and deadline - 1e9 <= e["time_ns"] <= now["time_ns"]
            ]
            llm = [e for e in es if e["kind"] == "llm_output"]
            lifecycle = [e for e in llm if e.get("frame_type") in {
                "LLMFullResponseStartFrame", "LLMFullResponseEndFrame"} and e["time_ns"] <= deadline]
            llm_active = bool(lifecycle and lifecycle[-1]["frame_type"] == "LLMFullResponseStartFrame")
            texts = [e["time_ns"] for e in llm if e.get("frame_type") == "LLMTextFrame" and e.get("characters")]
            previous_text = max((ns for ns in texts if ns <= deadline), default=None)
            next_text = min((ns for ns in texts if ns > deadline), default=None)
            next_tts_text = min((e["time_ns"] for e in es if e["kind"] == "tts_text_ready"
                                 and e["time_ns"] > deadline), default=None)
            row.update({
                "llm_response_active_at_deadline": llm_active,
                "last_llm_text_before_deadline_ms": (deadline - previous_text) / 1e6 if previous_text else None,
                "next_llm_text_after_deadline_ms": (next_text - deadline) / 1e6 if next_text else None,
                "next_tts_text_ready_after_deadline_ms": (next_tts_text - deadline) / 1e6 if next_tts_text else None,
            })
            row.update(
                {
                    "context": context,
                    "worker_pid": now["pid"],
                    "wire_monotonic_ns": now["monotonic_ns"],
                    "wire_complete_monotonic_ns": socket_end["monotonic_ns"] if socket_end else None,
                    "deadline_monotonic_ns": deadline - (now["time_ns"] - now["monotonic_ns"]),
                    "queue_frame_id": now.get("queue_frame_id"),
                    "deadline_definition": "client prior update plus actual recorded buffer credit"
                    if exact_match
                    else "client logged deficit subtracted from detection timestamp",
                    "provider_request_active_at_deadline": any(
                        q["context"] == context
                        and q["start_ns"] <= deadline
                        and (q["done_ns"] is None or q["done_ns"] > deadline)
                        for q in local_requests
                    ),
                    "client_detection_minus_nearest_wire_ms": (t - now["time_ns"])
                    / 1e6,
                    "coverage": "accounted"
                    if label != "insufficient_coverage"
                    else "incomplete",
                    "wire_gap_ms": (now["time_ns"] - previous["time_ns"]) / 1e6,
                    "previous_packet_audio_ms": audio_ms(previous),
                    "next_packet_audio_ms": audio_ms(now),
                    "provider_pcm_remaining_at_deadline_ms": p_left,
                    "output_pcm_remaining_at_deadline_ms": e_left,
                    "provider_packet_ready_vs_deadline_ms": (pr - deadline) / 1e6
                    if pr
                    else None,
                    "output_packet_ready_vs_deadline_ms": (er - deadline) / 1e6
                    if er
                    else None,
                    "wire_after_output_packet_ready_ms": (now["time_ns"] - er) / 1e6
                    if er
                    else None,
                    "max_nearby_server_loop_lag_ms": max(
                        (ms for stamp, ms in lag_by_pid.get(now["pid"], [])
                         if deadline - 200e6 <= stamp <= now["time_ns"] + 50e6),
                        default=None,
                    ),
                    "max_nearby_sleep_overshoot_ms": max(
                        (e["overshoot_ms"] for e in near), default=None
                    ),
                    "evidence": label,
                }
            )
            gap_rows.append(row)
    measured = [
        r
        for r in requests
        if r["phase"] == "measurement"
        and r["done_ns"]
        and not r["error"]
        and r["audio_ms"] > 0
    ]
    fully_measured = [
        request
        for request in measured
        if request["done_ns"]
        <= int(
            (
                clients[request["client"]]["start"]
                + clients[request["client"]]["duration"]
            )
            * 1e9
        )
    ]
    strict_window = {
        "scope": "Only complete non-error requests starting and finishing within the client measurement window; excludes drain.",
        "requests": len(fully_measured),
        "request_rtf_x": distribution(
            request["audio_per_request_wall_rtf_x"] for request in fully_measured
        ),
        "below_one": sum(
            request["audio_per_request_wall_rtf_x"] < 1 for request in fully_measured
        ),
        "counterfactual_within_request_deficits": sum(
            request["within_request_counterfactual_deficits_with_399ms_prebuffer"] > 0
            for request in fully_measured
        ),
    }
    (out / "fully-measured-summary.json").write_text(
        json.dumps(strict_window, indent=2) + "\n"
    )
    summary = {
        "measurement_definition": "Main request statistics select complete non-error requests by start time; some end in reply drain. fully-measured-summary.json excludes those completions.",
        "stage": a.stage,
        "clients": len(clients),
        "clients_with_server_events": sum(bool(c["events"]) for c in clients.values()),
        "measured_complete_nonerror_requests": len(measured),
        "request_rtf_x": distribution(
            r["audio_per_request_wall_rtf_x"] for r in measured
        ),
        "first_pcm_from_input_ms": distribution(
            r["first_pcm_from_input_ms"] for r in measured
        ),
        "after_first_chunk_rtf_x": distribution(
            r["after_first_chunk_rtf_x"] for r in measured
        ),
        "requests_rtf_x_below_one": sum(
            r["audio_per_request_wall_rtf_x"] < 1 for r in measured
        ),
        "requests_with_counterfactual_within_request_deficit": sum(
            r["within_request_counterfactual_deficits_with_399ms_prebuffer"] > 0
            for r in measured
        ),
        "gap_evidence_counts": dict(
            collections.Counter(
                r.get("evidence", "insufficient_coverage") for r in gap_rows
            )
        ),
        "limitations": [
            "PCM timestamps are pipeline receipt times: provider generation plus network and receiver scheduling. They are not GPU kernel timestamps.",
            "Request RTFx includes frontend/stream delivery from audio.start through audio.done; it is not GPU compute RTFx.",
            "Counterfactual per-request playback uses399.093ms initial prebuffer and excludes pauses between requests.",
            "Cumulative PCM/output/wire accounting assumes audio conservation within context. Interruptions, dropped or transformed audio can invalidate it; negative balances and missing data are reported as incomplete.",
            "Playback deadlines are reconstructed from client buffer-deficit logs on the same host clock; they are not physical audio-device timestamps. Output-processor entry precedes internal queueing/resampling. A downstream delay does not by itself prove transport sleep was responsible.",
            "Zero-event sessions do not establish successful admission; client errors remain in original raw results.",
        ],
    }
    for name, rows in [
        ("requests", requests),
        ("chunks", chunks_rows),
        ("gap-attribution", gap_rows),
    ]:
        (out / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n")
        if rows:
            keys = list(dict.fromkeys(k for row in rows for k in row))
            with (out / f"{name}.csv").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
