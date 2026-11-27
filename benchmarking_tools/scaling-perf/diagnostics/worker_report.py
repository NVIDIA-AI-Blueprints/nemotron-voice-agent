"""Join measured audio paths, worker CPU, and scheduler intervals offline."""

import argparse
import ast
import bisect
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


NS = 1_000_000_000
MS = 1_000_000
SYSCALLS = {
    0: "read",
    1: "write",
    7: "poll",
    23: "select",
    35: "nanosleep",
    44: "sendto",
    45: "recvfrom",
    46: "sendmsg",
    47: "recvmsg",
    202: "futex",
    230: "clock_nanosleep",
    232: "epoll_wait",
    270: "pselect6",
    271: "ppoll",
    281: "epoll_pwait",
    441: "epoll_pwait2",
    449: "futex_waitv",
}


def read_json(path):
    return json.loads(path.read_text())


def json_lines(path):
    with path.open() as file:
        for line in file:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if line.endswith("\n"):
                    raise


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[math.ceil(len(ordered) * q) - 1], 3)


def fields(name, values):
    return {
        name + "_samples": len(values),
        name + "_p95": percentile(values, 0.95),
        name + "_p99": percentile(values, 0.99),
        name + "_max": round(max(values), 3) if values else None,
    }


def write_csv(path, rows, default_columns):
    names = list(dict.fromkeys(key for row in rows for key in row)) or default_columns
    with path.open("w") as file:
        writer = csv.DictWriter(file, names)
        writer.writeheader()
        writer.writerows(rows)


def windows_and_peers(root):
    windows = {}
    peers = defaultdict(list)
    for path in (root / "suite").glob("run_*_clients/benchmark_summary.json"):
        n = int(path.parent.name.split("_")[1])
        config = read_json(path)["config"]
        start = config["metrics_start_time"]
        end = start + config["test_duration"]
        windows[n] = (int(start * NS), int(end * NS))
        for result_path in (root / "runs" / f"scaling-{n:03}").rglob("result_client*.json"):
            stream = read_json(result_path)["stream_id"]
            log = result_path.with_name(f"benchmark_{stream}.log")
            with log.open() as lines:
                for line in lines:
                    if match := re.search(r"diagnostic_connection local=(\(.+\))", line):
                        ip, port = ast.literal_eval(match[1])[:2]
                        peers[f"{ip}:{port}"].append((n, int((start - 45) * NS), int(end * NS)))
                        break
    return windows, peers


def load_scheduler(root):
    scheduler = defaultdict(list)
    with (root / "scheduler.csv").open() as file:
        for row in csv.reader(file):
            if len(row) == 6 and row[0] in {"blocked", "runq"}:
                pid, start, end, syscall, state = map(int, row[1:])
                if end >= start:
                    scheduler[pid].append((start, end, row[0], syscall, state))
    return scheduler


def intersection(start, end, low, high):
    return max(0, min(end, high) - max(start, low))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    root, out = args.root, args.report
    windows, peers = windows_and_peers(root)
    starts = sorted((start, n) for n, (start, _) in windows.items())

    def level(timestamp):
        index = bisect.bisect_right(starts, (timestamp, math.inf)) - 1
        if index >= 0:
            _, n = starts[index]
            if timestamp <= windows[n][1]:
                return n
        return None

    topology = read_json(root / "worker-topology.json")
    host = {int(ns): pid for ns, pid in topology["workers"].items()}
    scheduler = load_scheduler(root)
    gaps = []
    for path in (out / "timing").glob("scaling-*/gap-attribution.json"):
        gaps.extend(gap for gap in read_json(path) if gap["phase"] == "measurement")
    gap_frames = {
        (gap["worker_pid"], gap["queue_frame_id"])
        for gap in gaps if gap.get("queue_frame_id") is not None
    }
    by_gap_frame = defaultdict(dict)
    loop = defaultdict(list)
    metrics = defaultdict(lambda: defaultdict(list))
    attachments = Counter()
    not_main_thread = set()
    monotonic_windows = {}
    cgroup_samples = defaultdict(list)
    untraced = set()
    previous = None
    for sample in json_lines(root / "worker-samples.jsonl"):
        n = level(sample["time_ns"])
        tick = sample["monotonic_ns"]
        offset = sample["time_ns"] - tick
        if n is not None:
            start, end = windows[n]
            monotonic_windows[n] = (start - offset, end - offset)
            cgroup_samples[n].append(sample["cgroup_cpu"])
            untraced.update(sample["untraced_active_workers"])
        if n is not None and previous and n == level(previous["time_ns"]):
            interval_s = (tick - previous["monotonic_ns"]) / NS
            for key, now in sample["workers"].items():
                earlier = previous["workers"].get(key)
                if now is None or earlier is None or interval_s <= 0:
                    continue
                ns = int(key)
                worker = metrics[n, ns]
                group_cpu = (now[0] - earlier[0]) / topology["clock_ticks_per_second"]
                main_cpu = (now[1] - earlier[1]) / NS
                if group_cpu >= 0 and main_cpu >= 0:
                    worker["cpu_interval_s"].append(interval_s)
                    worker["worker_cpu_s"].append(group_cpu)
                    worker["main_cpu_s"].append(main_cpu)
                    worker["worker_cpu_cores"].append(group_cpu / interval_s)
                    worker["main_cpu_cores"].append(main_cpu / interval_s)
                worker["logical_cpu"].append(now[3])
                worker["sampled_task_state"].append(now[2])
        previous = sample

    for path in (root / "diagnostics/server").glob("events-*.jsonl"):
        queued = {}
        dequeued = {}
        for event in json_lines(path):
            n = level(event["time_ns"])
            ns = event["pid"]
            kind = event["kind"]
            mono = event["monotonic_ns"]
            if kind == "session_attach":
                matches = [count for count, lo, hi in peers[event["session"]]
                           if lo <= event["time_ns"] <= hi]
                if len(matches) == 1:
                    attachments[matches[0], ns] += 1
                if event.get("native_tid") != ns:
                    not_main_thread.add(ns)
            elif kind == "event_loop_lag":
                loop[ns].append(event)
                if n is not None:
                    metrics[n, ns]["loop_lag_observed_ms"].append(event["lag_ms"])
            key = (ns, event.get("queue_frame_id", event.get("frame_id")))
            if key in gap_frames:
                by_gap_frame[key][kind] = event
            if n is None:
                continue
            metric = metrics[n, ns]
            if kind == "audio_queue_put":
                queued[event["frame_id"]] = mono
                metric["queue_depth_frames"].append(event["queue_frames"])
            elif kind == "audio_queue_get":
                if (start := queued.pop(event["frame_id"], None)) is not None:
                    metric["queue_dwell_ms"].append((mono - start) / MS)
                dequeued[event["frame_id"]] = mono
            elif kind == "audio_queue_reset_pop":
                queued.pop(event["frame_id"], None)
            elif kind == "wire_send_start":
                if (start := dequeued.pop(event.get("queue_frame_id"), None)) is not None:
                    metric["queue_get_to_write_ms"].append((mono - start) / MS)
            elif kind == "audio_socket_send_end":
                metric["socket_send_ms"].append(event["elapsed_ms"])
            elif kind == "transport_sleep":
                metric["sleep_overshoot_ms"].append(event["overshoot_ms"])
            elif kind == "event_loop_summary":
                metric["loop_probe_ticks"].append(event["samples"])

    for (n, ns), count in attachments.items():
        metrics[n, ns]["sessions"].append(count)

    worker_rows = []
    for (n, ns), values in sorted(metrics.items()):
        if not values["sessions"] and not values["queue_dwell_ms"]:
            continue
        interval = sum(values["cpu_interval_s"])
        begin, end = monotonic_windows.get(n, (0, 0))
        sched = scheduler[host.get(ns)]
        wait = [intersection(s, e, begin, end) / MS for s, e, kind, _, _ in sched
                if kind == "runq" and e > begin and s < end]
        blocked = [intersection(s, e, begin, end) / MS for s, e, kind, _, _ in sched
                   if kind == "blocked" and e > begin and s < end]
        worker_rows.append({
            "clients_requested": n,
            "namespace_pid": ns,
            "host_pid": host.get(ns),
            "sessions_attached": sum(values["sessions"]),
            "cpu_coverage_s": round(interval, 3),
            "worker_cpu_mean_cores": round(sum(values["worker_cpu_s"]) / interval, 3) if interval else None,
            "loop_thread_mean_cores": round(sum(values["main_cpu_s"]) / interval, 3) if interval else None,
            "loop_probe_ticks": sum(values["loop_probe_ticks"]),
            "runq_total_over_threshold_ms": round(sum(wait), 3),
            "observed_block_total_over_threshold_ms": round(sum(blocked), 3),
            "distinct_last_cpus": len(set(values["logical_cpu"])),
            **fields("worker_cpu_cores", values["worker_cpu_cores"]),
            **fields("loop_thread_cpu_cores", values["main_cpu_cores"]),
            **fields("runnable_wait_ms", wait),
            **fields("observed_block_ms", blocked),
            **{k: v for name in ("loop_lag_observed_ms", "queue_depth_frames",
                                "queue_dwell_ms", "queue_get_to_write_ms",
                                "sleep_overshoot_ms", "socket_send_ms")
               for k, v in fields(name, values[name]).items()},
        })

    for gap in gaps:
        ns = gap.get("worker_pid")
        pid = host.get(ns)
        gap["host_pid"] = pid
        low = gap.get("deadline_monotonic_ns")
        high = gap.get("wire_complete_monotonic_ns") or gap.get("wire_monotonic_ns")
        if pid is None or low is None or high is None or high <= low:
            continue
        relevant_loop = [
            event for event in loop[ns] if
            intersection(event["window_start_ns"] + 50 * MS, event["sample_end_ns"], low, high)
        ]
        loop_max = max(relevant_loop, key=lambda event: event["lag_ms"], default=None)
        if loop_max:
            gap["overlapping_loop_lag_ms"] = round(loop_max["lag_ms"], 3)
            gap["loop_cpu_during_max_probe_ms"] = round(loop_max["thread_cpu_ms"], 3)
            gap["max_loop_probe_window_ms"] = round(
                (loop_max["sample_end_ns"] - loop_max["window_start_ns"]) / MS, 3
            )
        relevant = [
            (kind, syscall, state, intersection(start, end, low, high) / MS)
            for start, end, kind, syscall, state in scheduler[pid]
            if start < high and end > low
        ]
        gap["runnable_wait_overlap_ms"] = round(sum(ms for kind, _, _, ms in relevant if kind == "runq"), 3)
        blocked = [(ms, syscall, state) for kind, syscall, state, ms in relevant if kind == "blocked"]
        gap["blocked_overlap_ms"] = round(sum(ms for ms, _, _ in blocked), 3)
        if blocked:
            duration, syscall, state = max(blocked)
            gap["longest_block_overlap_ms"] = round(duration, 3)
            gap["longest_block_syscall"] = SYSCALLS.get(syscall, str(syscall))
            gap["longest_block_state"] = state
        frame = by_gap_frame[ns, gap["queue_frame_id"]]
        for kind, label in (("audio_queue_put", "packet_queue_enter"),
                            ("audio_queue_get", "packet_queue_leave"),
                            ("audio_socket_send_start", "socket_send_start")):
            if event := frame.get(kind):
                gap[label + "_vs_deadline_ms"] = round((event["monotonic_ns"] - low) / MS, 3)
                if "queue_frames" in event:
                    gap[label + "_queue_frames"] = event["queue_frames"]
        if send := frame.get("audio_socket_send_end"):
            gap["socket_send_duration_ms"] = round(send["elapsed_ms"], 3)
        if frame.get("audio_queue_put") and frame.get("audio_queue_get"):
            gap["packet_queue_dwell_ms"] = round(
                (frame["audio_queue_get"]["monotonic_ns"] - frame["audio_queue_put"]["monotonic_ns"]) / MS, 3
            )

    level_rows = []
    for n in sorted(windows):
        workers = [row for row in worker_rows if row["clients_requested"] == n]
        cgroup = cgroup_samples[n]
        first, last = (cgroup[0], cgroup[-1]) if cgroup else ({}, {})
        shares = [row["worker_cpu_mean_cores"] for row in workers
                  if row["worker_cpu_mean_cores"] is not None]
        counts = sorted((row["sessions_attached"] for row in workers), reverse=True)
        valid = read_json(root / "suite" / f"run_{n}_clients/benchmark_summary.json")["results"]
        measured = [gap for gap in gaps if gap["stage"] == f"scaling-{n:03}"]
        level_rows.append({
            "clients_requested": n,
            "accepted_sessions_matched": sum(counts),
            "workers_used": sum(count > 0 for count in counts),
            "sessions_per_worker_descending": "/".join(str(count) for count in counts if count),
            "reported_successful_clients": valid["successful_clients"],
            "mean_worker_cpu_cores": round(mean(shares), 3) if shares else None,
            "hottest_worker_mean_cores": max(shares, default=None),
            "hottest_loop_p99_cpu_cores": max(
                (row["loop_thread_cpu_cores_p99"] for row in workers
                 if row["loop_thread_cpu_cores_p99"] is not None), default=None,
            ),
            "max_loop_lag_over_10ms": max(
                (row["loop_lag_observed_ms_max"] for row in workers
                 if row["loop_lag_observed_ms_max"] is not None), default=None,
            ),
            "max_runnable_wait_ms": max(
                (row["runnable_wait_ms_max"] for row in workers if row["runnable_wait_ms_max"] is not None),
                default=None,
            ),
            "max_packet_queue_dwell_ms": max(
                (row["queue_dwell_ms_max"] for row in workers if row["queue_dwell_ms_max"] is not None),
                default=None,
            ),
            "cgroup_cpu_s": round((last["usage_usec"] - first["usage_usec"]) / 1e6, 3)
            if "usage_usec" in first and "usage_usec" in last else None,
            "cgroup_throttled_count": last["nr_throttled"] - first["nr_throttled"]
            if "nr_throttled" in first and "nr_throttled" in last else None,
            "speech_gap_events": len(measured),
            "speech_gap_ms": round(sum(g["client_deficit_ms"] for g in measured), 3),
            "speech_gaps_with_worker_correlation": sum(g["host_pid"] is not None for g in measured),
            "speech_gaps_with_exact_queue_timing": sum(g.get("packet_queue_dwell_ms") is not None for g in measured),
        })

    write_csv(out / "worker-profile.csv", worker_rows, ["clients_requested"])
    write_csv(out / "worker-profile-by-concurrency.csv", level_rows, ["clients_requested"])
    write_csv(out / "speech-gap-worker-timing.csv", gaps, ["stage", "client"])
    (out / "worker-profile-coverage.json").write_text(json.dumps({
        "topology": topology,
        "completed_levels": sorted(windows),
        "untraced_active_workers": sorted(untraced),
        "non_main_loop_workers": sorted(not_main_thread),
        "rows": level_rows,
        "limitations": [
            "Per-process and main-loop CPU are based on 250 ms samples.",
            "Scheduler events contain runnable waits >=0.25 ms and selected blocked intervals >=5 ms.",
            "Normal interruptible epoll waits under 180 ms are excluded.",
            "Blocked syscall or lag proximity alone does not establish which callback caused a gap.",
            "Loop CPU is sampled over the whole 50 ms probe interval plus overshoot.",
        ],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
