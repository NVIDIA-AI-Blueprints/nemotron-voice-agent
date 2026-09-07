# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Run and suite aggregation for scaling-perf."""

# ruff: noqa: D103

from __future__ import annotations

import datetime as dt
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from benchmark_core import SERVER_METRIC_KEYS, average_or_none, round3

SUITE_HEADERS = (
    "Parallel Streams",
    "Successful",
    "Failures",
    "No Response",
    "Avg Latency",
    "P95 Latency",
    "Min Latency",
    "Max Latency",
    "LLM TTFT",
    "TTS TTFB",
    "ASR TTFB",
    "Server E2E",
    "VAD+Smart Turn",
    "LLM Proc Time",
    "LLM Tok/s",
    "Glitches",
)
assert len(SUITE_HEADERS) == 9 + len(SERVER_METRIC_KEYS), "SUITE_HEADERS must cover every server metric key"
CLIENT_HEADERS = ("Client", *SUITE_HEADERS[1:])


def calculate_p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil(0.95 * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def _client_int(client: dict, key: str, default: int = 0) -> int:
    try:
        return int(client.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _client_id(client: dict) -> str:
    return str(client.get("stream_id") or "(unknown)")


def _client_valid_turns(client: dict) -> int:
    return _client_int(client, "num_valid_turns")


def _has_valid_response(client: dict) -> bool:
    return client.get("average_latency") is not None and _client_valid_turns(client) > 0


def _is_hard_deadline(client: dict) -> bool:
    return str(client.get("error") or "").startswith("Hard deadline reached")


def _has_core_server_metric(client: dict) -> bool:
    counts = (client.get("server_metrics") or {}).get("sample_counts") or {}
    for key in ("asr_ttfb", "llm_ttft", "tts_ttfb"):
        try:
            if int(counts.get(key, 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _has_runtime_error(client: dict) -> bool:
    return bool(client.get("error")) and not _is_hard_deadline(client)


def _is_failed(client: dict) -> bool:
    return _has_runtime_error(client)


def _is_no_response(client: dict) -> bool:
    return not _is_failed(client) and not _has_valid_response(client)


def _is_successful(client: dict) -> bool:
    return not _is_failed(client) and _has_valid_response(client)


def _valid_latencies(client: dict) -> list[float]:
    values = client.get("valid_latencies")
    if not isinstance(values, list):
        return []
    return [float(value) for value in values if isinstance(value, (int, float))]


def _latency_summary(client: dict) -> dict[str, float | None]:
    values = _valid_latencies(client)
    return {
        "avg_latency": average_or_none(values),
        "p95_latency": calculate_p95(values),
        "min_latency": min(values) if values else None,
        "max_latency": max(values) if values else None,
    }


def _average_latency_columns(clients: Iterable[dict]) -> dict[str, float | None]:
    summaries = [_latency_summary(client) for client in clients]
    return {
        key: average_or_none([summary[key] for summary in summaries if summary[key] is not None])
        for key in ("avg_latency", "p95_latency", "min_latency", "max_latency")
    }


def _server_average(client: dict, key: str) -> float | None:
    value = ((client.get("server_metrics") or {}).get("average") or {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _client_sort_key(client: dict) -> tuple[int, str]:
    stream_id = str(client.get("stream_id") or "")
    parts = stream_id.split("_")
    if len(parts) > 1:
        try:
            return int(parts[1]), stream_id
        except ValueError:
            pass
    return sys.maxsize, stream_id


def _weighted_average(clients: list[dict], key: str) -> tuple[float | None, int]:
    weighted = 0.0
    total = 0
    for client in clients:
        metrics = client.get("server_metrics") or {}
        average = (metrics.get("average") or {}).get(key)
        count = (metrics.get("sample_counts") or {}).get(key, 0)
        if average is not None and count:
            weighted += float(average) * int(count)
            total += int(count)
    return ((weighted / total) if total else None), total


def aggregate_run_dir(run_dir: Path, num_clients: int | None) -> Path:
    """Collapse client result JSON files into ``benchmark_summary.json``."""
    clients = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(run_dir.glob("client_*/result_*.json"))]
    if num_clients is None:
        num_clients = len(clients)

    hard_deadline = [client for client in clients if _is_hard_deadline(client)]
    latency_clients = [client for client in clients if _has_valid_response(client)]
    latency_columns = _average_latency_columns(latency_clients)
    barge_only = [
        client for client in clients if _client_valid_turns(client) == 0 and _client_int(client, "num_turns") > 0
    ]
    hard_with_valid = [client for client in hard_deadline if _has_valid_response(client)]
    hard_without_valid = [client for client in hard_deadline if not _has_valid_response(client)]
    runtime_errors = [client for client in clients if _has_runtime_error(client)]
    no_response = [client for client in clients if _is_no_response(client)]
    metric_only_no_response = [
        client for client in no_response if not _has_valid_response(client) and _has_core_server_metric(client)
    ]
    failed_ids = {_client_id(client) for client in runtime_errors}
    missing = max(0, num_clients - len(clients))
    failed = len(runtime_errors) + missing
    successful = max(0, num_clients - failed - len(no_response))

    server_average: dict[str, float | None] = {}
    server_counts: dict[str, int] = {}
    for key in SERVER_METRIC_KEYS:
        average, count = _weighted_average(clients, key)
        server_average[key] = average
        server_counts[key] = count

    hard_ids = [_client_id(client) for client in hard_deadline]
    runtime_ids = [_client_id(client) for client in runtime_errors]
    hard_without_ids = [_client_id(client) for client in hard_without_valid]
    summary = {
        "timestamp": dt.datetime.now().isoformat(),
        "config": {
            "num_clients": num_clients,
            "test_duration": clients[0].get("test_duration") if clients else None,
            "metrics_start_time": clients[0].get("metrics_start_time") if clients else None,
            "reverse_barge_in_threshold": (clients[0].get("reverse_barge_in_threshold") if clients else None),
            "turn_response_timeout": clients[0].get("turn_response_timeout") if clients else None,
            "concurrency_mode": "process-per-client",
        },
        "results": {
            "configured_clients": num_clients,
            "successful_clients": successful,
            "failed_clients": failed,
            "failed_client_ids": sorted(failed_ids),
            "runtime_error_clients": len(runtime_errors),
            "runtime_error_client_ids": runtime_ids,
            "missing_clients": missing,
            "latency_sample_clients": len(latency_clients),
            "metric_only_success_clients": 0,
            "metric_only_no_response_clients": len(metric_only_no_response),
            "barge_in_only_clients": len(barge_only),
            "no_response_clients": len(no_response),
            "no_response_client_ids": [_client_id(client) for client in no_response],
            "hard_deadline_clients": len(hard_deadline),
            "hard_deadline_with_valid_response_clients": len(hard_with_valid),
            "hard_deadline_with_success_signal_clients": len(hard_with_valid),
            "hard_deadline_successful_clients": len(hard_with_valid),
            "hard_deadline_no_success_signal_clients": len(hard_without_valid),
            "hard_deadline_no_valid_response_clients": len(hard_without_valid),
            "hard_deadline_client_ids": hard_ids,
            "hard_deadline_no_valid_response_client_ids": hard_without_ids,
            "total_turns": sum(_client_int(client, "num_turns") for client in clients),
            "total_valid_turns": sum(_client_valid_turns(client) for client in latency_clients),
            "total_barge_ins": sum(_client_int(client, "reverse_barge_ins_count") for client in clients),
            "total_failed_turns": sum(_client_int(client, "failed_turns") for client in clients),
            "aggregate_average_latency": latency_columns["avg_latency"],
            "p95_client_latency": latency_columns["p95_latency"],
            "min_client_latency": latency_columns["min_latency"],
            "max_client_latency": latency_columns["max_latency"],
            "server_metrics": {"average": server_average, "sample_counts": server_counts},
            "glitch_detection": {
                "clients_with_glitches": sum(1 for client in latency_clients if client.get("glitch_detected")),
                "total_clients": len(latency_clients),
                "affected_client_ids": [
                    _client_id(client) for client in latency_clients if client.get("glitch_detected")
                ],
            },
            "error_detection": {
                "total_clients": len(clients),
                "clients_with_errors": len(runtime_errors),
                "runtime_error_clients": len(runtime_errors),
                "runtime_error_client_ids": runtime_ids,
                "hard_deadline_clients": len(hard_deadline),
                "hard_deadline_successful_clients": len(hard_with_valid),
                "hard_deadline_with_success_signal_clients": len(hard_with_valid),
                "hard_deadline_no_success_signal_clients": len(hard_without_valid),
                "hard_deadline_no_valid_response_clients": len(hard_without_valid),
                "hard_deadline_client_ids": hard_ids,
                "client_error_counts": {_client_id(client): 1 for client in clients if client.get("error")},
            },
        },
        "clients": clients,
    }
    output = run_dir / "benchmark_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output


def _summary_row(summary: dict[str, Any], num_clients: int) -> dict[str, Any]:
    results = summary["results"]
    server = results["server_metrics"]["average"]
    return {
        "parallel_streams": num_clients,
        "successful_streams": results["successful_clients"],
        "configured_streams": results["configured_clients"],
        "failed_streams": results.get("failed_clients", 0),
        "no_response_streams": results.get("no_response_clients", 0),
        "avg_latency": results["aggregate_average_latency"],
        "p95_latency": results["p95_client_latency"],
        "min_latency": results["min_client_latency"],
        "max_latency": results["max_client_latency"],
        **{key: server.get(key) for key in SERVER_METRIC_KEYS},
        "audio_glitches": results["glitch_detection"]["clients_with_glitches"],
    }


def _client_row(client: dict[str, Any]) -> dict[str, Any]:
    return {
        "client": _client_id(client),
        "successful_streams": 1 if _is_successful(client) else 0,
        "configured_streams": 1,
        "failed_streams": 1 if _is_failed(client) else 0,
        "no_response_streams": 1 if _is_no_response(client) else 0,
        **_latency_summary(client),
        **{key: _server_average(client, key) for key in SERVER_METRIC_KEYS},
        "audio_glitches": 1 if client.get("glitch_detected") else 0,
    }


def _client_rows(summary: dict[str, Any], num_clients: int) -> list[dict[str, Any]]:
    rows = [_client_row(client) for client in sorted(summary.get("clients", []), key=_client_sort_key)]
    average = _summary_row(summary, num_clients)
    average["client"] = "AVERAGE"
    average.update(_average_latency_columns(summary.get("clients", [])))
    rows.append(average)
    return rows


def _row_strings(row: dict[str, Any], label: str) -> list[str]:
    return [
        str(row[label]),
        f"{row['successful_streams']}/{row['configured_streams']}",
        str(row["failed_streams"]),
        str(row["no_response_streams"]),
        *(round3(row[key]) for key in ("avg_latency", "p95_latency", "min_latency", "max_latency")),
        *(round3(row[key]) for key in SERVER_METRIC_KEYS),
        str(row["audio_glitches"]),
    ]


def _format_table(headers: Iterable[str], rows: list[list[str]]) -> list[str]:
    headers = list(headers)
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def is_number(value: str) -> bool:
        try:
            float(value.strip())
        except ValueError:
            return False
        return bool(value.strip())

    right = [bool(rows) and index > 0 and all(is_number(row[index]) for row in rows) for index in range(len(headers))]

    def render(values: list[str]) -> str:
        return "  ".join(
            values[index].rjust(widths[index]) if right[index] else values[index].ljust(widths[index])
            for index in range(len(headers))
        )

    return [
        render(headers),
        "  ".join("-" * width for width in widths),
        *(render(row) for row in rows),
    ]


def aggregate_suite_dir(suite_dir: Path) -> tuple[Path, Path, Path]:
    """Emit unchanged ``results.tsv``, ``results.txt``, and ``results.json``."""
    run_dirs = sorted(
        (path for path in suite_dir.iterdir() if path.is_dir() and path.name.startswith("run_")),
        key=lambda path: int(path.name.split("_")[1]) if path.name.split("_")[1].isdigit() else 0,
    )
    rows: list[dict[str, Any]] = []
    headers = SUITE_HEADERS
    label = "parallel_streams"
    if run_dirs:
        for run_dir in run_dirs:
            try:
                count = int(run_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            summary_path = run_dir / "benchmark_summary.json"
            if summary_path.exists():
                rows.append(_summary_row(json.loads(summary_path.read_text(encoding="utf-8")), count))
    else:
        summary_path = suite_dir / "benchmark_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            count = int(summary.get("config", {}).get("num_clients") or summary["results"]["configured_clients"])
            rows = _client_rows(summary, count)
            headers = CLIENT_HEADERS
            label = "client"

    string_rows = [_row_strings(row, label) for row in rows]
    tsv = suite_dir / "results.tsv"
    txt = suite_dir / "results.txt"
    js = suite_dir / "results.json"
    with tsv.open("w", encoding="utf-8") as output:
        output.write("\t".join(headers) + "\n")
        for row in string_rows:
            output.write("\t".join(row) + "\n")
    txt.write_text("\n".join(_format_table(headers, string_rows)) + "\n", encoding="utf-8")
    js.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return tsv, txt, js


def run_aggregate_run(run_dir: Path, num_clients: int | None) -> int:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        print(f"--aggregate-run-dir not found: {run_dir}", file=sys.stderr)
        return 2
    summary_path = aggregate_run_dir(run_dir, num_clients)
    results = json.loads(summary_path.read_text(encoding="utf-8"))["results"]
    print(
        f"run aggregated: {summary_path}  "
        f"successful={results['successful_clients']}/{results['configured_clients']}  "
        f"failures={results.get('failed_clients', 0)}  "
        f"no_response={results.get('no_response_clients', 0)}  "
        f"avg_latency={round3(results['aggregate_average_latency'])}s  "
        f"p95={round3(results['p95_client_latency'])}s"
    )
    return 0


def run_aggregate_suite(suite_dir: Path) -> int:
    suite_dir = suite_dir.resolve()
    if not suite_dir.is_dir():
        print(f"--aggregate-suite-dir not found: {suite_dir}", file=sys.stderr)
        return 2
    tsv, txt, js = aggregate_suite_dir(suite_dir)
    print(f"suite aggregated: {suite_dir}")
    print(f"  TXT:  {txt}")
    print(f"  TSV:  {tsv}")
    print(f"  JSON: {js}")
    return 0
