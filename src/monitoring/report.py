# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare recorded sessions grouped by pipeline variant.

Groups sessions by fields of their config snapshot (default: LLM, ASR and TTS
models plus language) and prints, per group, latency percentiles (user→bot, also
split by turn kind, per-service TTFB, LLM TTFT split text-only vs. with images),
interruption rate and ASR WER from ``reasr`` annotations.

The turn-kind split needs the per-turn rows of ``monitoring.turn_metrics``
(written at the end of each recorded turn; run ``python -m monitoring.turn_metrics``
once for sessions recorded before they existed).

Usage::

    uv run python -m monitoring.report
    uv run python -m monitoring.report --since-days 7 --group-by llm.model,turn_detection.silero_vad_only
    uv run python -m monitoring.report --session 3f2a9c1d0b7e --csv report.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import inspect, select

from monitoring import schema
from monitoring.config import load_monitoring_config
from monitoring.store import SessionStore
from monitoring.turn_metrics import KINDS, STAGES, service_role

# A trend bucket is flagged when its median response latency is clearly worse than
# the previous bucket's: both ratio and absolute difference, on enough turns.
REGRESSION_RATIO = 1.2
REGRESSION_MIN_DELTA_SECS = 0.1
REGRESSION_MIN_TURNS = 5
_TURN_METRIC_COLUMNS = [c.name for c in schema.turn_metrics.columns if c.name != "segments"]

DEFAULT_GROUP_BY = "llm.model,asr.model,tts.model,language"


def _dig(config: dict[str, Any], dotted: str) -> str:
    value: Any = config
    for part in dotted.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return "" if value is None else str(value)


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(sorted(values), n=100, method="inclusive")[round(q) - 1]


def _fmt(value: float | None, unit: str = "s") -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%" if unit == "%" else f"{value:.3f}{unit}"


def _service(processor: str | None) -> str:
    return re.sub(r"#\d+$", "", processor or "?")


def collect(store: SessionStore, *, since: float | None, session_ids: list[str] | None, group_by: list[str]):
    """Return ``{group_key: stats}`` for the selected sessions."""
    return collect_with_sessions(store, since=since, session_ids=session_ids, group_by=group_by)[0]


def collect_with_sessions(
    store: SessionStore, *, since: float | None, session_ids: list[str] | None, group_by: list[str]
) -> tuple[dict[tuple, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return ``({group_key: stats}, {session_id: per-session stats})``."""
    with store.engine.connect() as conn:
        stmt = select(schema.sessions)
        if since:
            stmt = stmt.where(schema.sessions.c.started_at >= since)
        if session_ids:
            stmt = stmt.where(schema.sessions.c.id.in_(session_ids))
        sessions = [dict(r) for r in conn.execute(stmt).mappings()]
        ids = [s["id"] for s in sessions]

        def rows(table, *columns):
            if not ids:
                return []
            query = select(*(table.c[c] for c in columns)).where(table.c.session_id.in_(ids))
            return [dict(r) for r in conn.execute(query).mappings()]

        metrics = rows(schema.metrics, "session_id", "turn_idx", "processor", "name", "value")
        # A database not yet migrated to schema 4 simply has no per-turn rows.
        derived = inspect(conn).has_table(schema.turn_metrics.name)
        turn_metrics = rows(schema.turn_metrics, *_TURN_METRIC_COLUMNS) if derived else []
        tool_calls = rows(schema.tool_calls, *(c.name for c in schema.tool_calls.columns)) if derived else []
        turns = rows(schema.turns, "session_id", "idx", "user_text", "interrupted")
        llm_calls = rows(schema.llm_calls, "session_id", "ttfb", "n_images", "prompt_tokens", "completion_tokens")
        wers = [
            a
            for a in rows(schema.annotations, "session_id", "kind", "source", "value", "id")
            if a["kind"] == "wer_summary"
        ]

    group_of = {s["id"]: tuple(_dig(s["config"] or {}, g) for g in group_by) for s in sessions}
    groups: dict[tuple, dict[str, Any]] = defaultdict(
        lambda: {
            "sessions": 0,
            "turns": 0,
            "interrupted": 0,
            "latency": [],
            "latency_by_kind": defaultdict(list),
            "turn_metrics": [],
            "tool_calls": [],
            "first_latency": [],
            "ttfb": defaultdict(list),
            "ttft_text": [],
            "ttft_vision": [],
            "prompt_tokens": [],
            "wer": defaultdict(lambda: [0, 0]),
        }
    )
    per_session: dict[str, dict[str, Any]] = {
        s["id"]: {
            "started_at": s["started_at"],
            "group": group_of[s["id"]],
            "git_sha": (s["config"] or {}).get("git_sha"),
            "latency": [],
            "turn_metrics": [],
            "wer": {},
        }
        for s in sessions
    }
    kind_of = {(row["session_id"], row["idx"]): row["kind"] for row in turn_metrics}
    for row in turn_metrics:
        groups[group_of[row["session_id"]]]["turn_metrics"].append(row)
        per_session[row["session_id"]]["turn_metrics"].append(row)
    for call in tool_calls:
        groups[group_of[call["session_id"]]]["tool_calls"].append(call)
    for session in sessions:
        groups[group_of[session["id"]]]["sessions"] += 1
    for turn in turns:
        if turn["user_text"]:
            g = groups[group_of[turn["session_id"]]]
            g["turns"] += 1
            g["interrupted"] += bool(turn["interrupted"])
    for metric in metrics:
        g = groups[group_of[metric["session_id"]]]
        if metric["name"] == "user_bot_latency":
            g["latency"].append(metric["value"])
            per_session[metric["session_id"]]["latency"].append(metric["value"])
            kind = kind_of.get((metric["session_id"], metric["turn_idx"]))
            if kind:
                g["latency_by_kind"][kind].append(metric["value"])
        elif metric["name"] == "first_bot_speech_latency":
            g["first_latency"].append(metric["value"])
        elif metric["name"] == "ttfb" and (metric["value"] or 0) > 0:
            # Zero-valued samples are start-up artifacts of pipecat, not measurements.
            g["ttfb"][_service(metric["processor"])].append(metric["value"])
    for call in llm_calls:
        g = groups[group_of[call["session_id"]]]
        if call["ttfb"] is not None:
            (g["ttft_vision"] if call["n_images"] else g["ttft_text"]).append(call["ttfb"])
        if call["prompt_tokens"]:
            g["prompt_tokens"].append(call["prompt_tokens"])
    latest_wer: dict[tuple[str, str], dict] = {}
    for annotation in sorted(wers, key=lambda a: a["id"]):
        latest_wer[(annotation["session_id"], annotation["source"])] = annotation["value"] or {}
    for (session_id, source), value in latest_wer.items():
        acc = groups[group_of[session_id]]["wer"][source]
        acc[0] += value.get("word_errors") or 0
        acc[1] += value.get("ref_words") or 0
        per_session[session_id]["wer"][source] = value.get("wer")
    return groups, per_session


def _dist(values: list[float]) -> dict[str, float | int | None]:
    return {"n": len(values), "p10": _pct(values, 10), "p50": _pct(values, 50), "p90": _pct(values, 90)}


def _values(rows: list[dict[str, Any]], column: str) -> list[float]:
    return [row[column] for row in rows if row.get(column) is not None]


def _kind_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Latency and stage distributions of a set of ``turn_metrics`` rows (``turns`` is the sample count)."""
    flagged = [row["barge_in"] for row in rows if row.get("barge_in") is not None]
    loads, peaks = _values(rows, "gpu_load_mean"), _values(rows, "gpu_load_peak")
    return {
        "turns": len(rows),
        "response": _dist(_values(rows, "response_latency")),
        "voice": _dist(_values(rows, "voice_latency")),
        "stages": {stage: _dist(_values(rows, f"{stage}_secs")) for stage in (*STAGES, "unexplained")},
        "barge_in_rate": sum(map(bool, flagged)) / len(flagged) if flagged else None,
        "llm_calls_mean": statistics.fmean(_values(rows, "n_llm_calls")) if rows else None,
        "prompt_tokens_p50": _pct(_values(rows, "prompt_tokens"), 50),
        "completion_tokens_p50": _pct(_values(rows, "completion_tokens"), 50),
        "gpu_load_mean": statistics.fmean(loads) if loads else None,
        "gpu_load_peak": max(peaks) if peaks else None,
    }


def _by_kind(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """``{"all": stats, <kind>: stats}`` for the kinds present, so like is compared with like."""
    out = {"all": _kind_stats(rows)}
    for kind in KINDS:
        subset = [row for row in rows if row["kind"] == kind]
        if subset:
            out[kind] = _kind_stats(subset)
    return out


def _trend(buckets: list[tuple[str, str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    """Per-bucket stats in order, flagging a bucket whose p50 is clearly worse than the previous one."""
    out: list[dict[str, Any]] = []
    previous: dict[str, tuple[str, float]] = {}  # kind -> (bucket label, response p50)
    for key, label, sessions in buckets:
        rows = [row for session in sessions for row in session["turn_metrics"]]
        by_kind = _by_kind(rows)
        for kind, stats in by_kind.items():
            response = stats["response"]
            stats["regression"] = None
            if response["n"] < REGRESSION_MIN_TURNS:
                continue
            before = previous.get(kind)
            if (
                before
                and response["p50"] > before[1] * REGRESSION_RATIO
                and response["p50"] - before[1] > REGRESSION_MIN_DELTA_SECS
            ):
                stats["regression"] = {"previous": before[0], "previous_p50": before[1]}
            previous[kind] = (label, response["p50"])
        out.append(
            {
                "key": key,
                "label": label,
                "started_at": min(session["started_at"] for session in sessions),
                "sessions": len(sessions),
                "by_kind": by_kind,
            }
        )
    return out


def _trends(per_session: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(per_session.values(), key=lambda s: s["started_at"])
    days: dict[str, list] = defaultdict(list)
    revisions: dict[str, list] = defaultdict(list)  # insertion order = first time a revision was seen
    for session in ordered:
        days[datetime.fromtimestamp(session["started_at"]).strftime("%Y-%m-%d")].append(session)
        revisions[session["git_sha"] or ""].append(session)
    return {
        "by_day": _trend([(day, day, sessions) for day, sessions in days.items()]),
        "by_revision": _trend([(sha, sha[:8] or "unknown", sessions) for sha, sessions in revisions.items()]),
    }


def _tool_table(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (tool, trigger, target): calls, durations, failure and timeout rates."""
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        grouped[(call["name"], call["trigger"], call["target"])].append(call)
    table = []
    for (name, trigger, target), rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        outcomes = defaultdict(int)
        for row in rows:
            outcomes[row["outcome"]] += 1
        table.append(
            {
                "name": name,
                "trigger": trigger,
                "target": target,
                "perceivable": any(row["perceivable"] for row in rows),
                "calls": len(rows),
                "duration": _dist(_values(rows, "duration_secs")),
                "failure_rate": (len(rows) - outcomes["ok"]) / len(rows),
                "error_rate": outcomes["error"] / len(rows),
                "timeout_rate": outcomes["timeout"] / len(rows),
                "cancelled_rate": outcomes["cancelled"] / len(rows),
            }
        )
    return table


def metrics_json(
    store: SessionStore, *, since: float | None, group_by: list[str], session_ids: list[str] | None = None
) -> dict[str, Any]:
    """Numeric, JSON-ready variant comparison (used by the review UI)."""
    groups, per_session = collect_with_sessions(store, since=since, session_ids=session_ids, group_by=group_by)
    keys = sorted(groups, key=lambda k: (-groups[k]["sessions"], k))
    index = {key: i for i, key in enumerate(keys)}
    variants = []
    for key in keys:
        g = groups[key]
        ttfb: dict[str, list[float]] = defaultdict(list)
        for service, values in g["ttfb"].items():
            ttfb[service_role(service)].extend(values)
        variants.append(
            {
                "key": dict(zip(group_by, key, strict=True)),
                "label": " · ".join(v or "–" for v in key) or "all sessions",
                "sessions": g["sessions"],
                "turns": g["turns"],
                "interrupt_rate": g["interrupted"] / g["turns"] if g["turns"] else None,
                "latency": _dist(g["latency"]),
                "by_kind": _by_kind(g["turn_metrics"]),
                "first_speech": _dist(g["first_latency"]),
                "ttfb": {role: _dist(values) for role, values in sorted(ttfb.items())},
                "ttft": {"text": _dist(g["ttft_text"]), "vision": _dist(g["ttft_vision"])},
                "prompt_tokens_p50": _pct(g["prompt_tokens"], 50),
                "wer": {
                    source: {"wer": errors / words if words else None, "ref_words": words}
                    for source, (errors, words) in sorted(g["wer"].items())
                },
            }
        )
    all_latency = [v for g in groups.values() for v in g["latency"]]
    live_errors = sum(e for g in groups.values() for s, (e, _) in g["wer"].items() if s.startswith("live:"))
    live_words = sum(w for g in groups.values() for s, (_, w) in g["wer"].items() if s.startswith("live:"))
    all_turn_metrics = [row for g in groups.values() for row in g["turn_metrics"]]
    return {
        "group_by": group_by,
        "kinds": [kind for kind in KINDS if any(row["kind"] == kind for row in all_turn_metrics)],
        "stages": [*STAGES, "unexplained"],
        "totals": {
            "sessions": sum(g["sessions"] for g in groups.values()),
            "turns": sum(g["turns"] for g in groups.values()),
            "latency_p50": _pct(all_latency, 50),
            "by_kind": _by_kind(all_turn_metrics),
            "live_wer": live_errors / live_words if live_words else None,
        },
        "variants": variants,
        "trend": _trends(per_session),
        "tools": _tool_table([call for g in groups.values() for call in g["tool_calls"]]),
        "sessions": sorted(
            (
                {
                    "id": sid,
                    "started_at": s["started_at"],
                    "variant": index[s["group"]],
                    "latency_p50": _pct(s["latency"], 50),
                    "response_p50": _pct(_values(s["turn_metrics"], "response_latency"), 50),
                    "live_wer": next((w for src, w in s["wer"].items() if src.startswith("live:")), None),
                }
                for sid, s in per_session.items()
            ),
            key=lambda s: s["started_at"],
        ),
    }


def summarize(groups, group_by: list[str]) -> list[dict[str, str]]:
    """Flatten grouped stats into printable rows."""
    rows = []
    for key, g in sorted(groups.items()):
        row = {name: value or "-" for name, value in zip(group_by, key, strict=True)}
        row.update(
            {
                "sessions": str(g["sessions"]),
                "turns": str(g["turns"]),
                "interrupt_rate": _fmt(g["interrupted"] / g["turns"] if g["turns"] else None, "%"),
                "user_bot_p50": _fmt(_pct(g["latency"], 50)),
                "user_bot_p90": _fmt(_pct(g["latency"], 90)),
                "first_speech_p50": _fmt(_pct(g["first_latency"], 50)),
                "llm_ttft_text_p50": _fmt(_pct(g["ttft_text"], 50)),
                "llm_ttft_vision_p50": _fmt(_pct(g["ttft_vision"], 50)),
                "prompt_tokens_p50": str(int(_pct(g["prompt_tokens"], 50))) if g["prompt_tokens"] else "-",
            }
        )
        for kind in KINDS:
            values = g["latency_by_kind"].get(kind)
            if values:
                row[f"turns[{kind}]"] = str(len(values))
                row[f"user_bot_p50[{kind}]"] = _fmt(_pct(values, 50))
                row[f"user_bot_p90[{kind}]"] = _fmt(_pct(values, 90))
        for service, values in sorted(g["ttfb"].items()):
            row[f"ttfb_p50[{service}]"] = _fmt(_pct(values, 50))
            row[f"ttfb_p90[{service}]"] = _fmt(_pct(values, 90))
        for source, (errors, words) in sorted(g["wer"].items()):
            row[f"wer[{source}]"] = _fmt(errors / words if words else None, "%")
        rows.append(row)
    return rows


def main() -> int:
    """Entry point."""
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group-by", default=DEFAULT_GROUP_BY, help=f"config fields (default: {DEFAULT_GROUP_BY})")
    parser.add_argument("--since-days", type=float, help="only sessions started in the last N days")
    parser.add_argument("--session", action="append", help="restrict to session id(s)")
    parser.add_argument("--csv", help="also write the table to this CSV file")
    args = parser.parse_args()

    group_by = [g.strip() for g in args.group_by.split(",") if g.strip()]
    store = SessionStore(load_monitoring_config().db_url)
    since = time.time() - args.since_days * 86400 if args.since_days else None
    rows = summarize(collect(store, since=since, session_ids=args.session, group_by=group_by), group_by)
    if not rows:
        print("No recorded sessions match (is MONITORING_ENABLED=true?).")
        return 0
    for i, row in enumerate(rows, 1):
        print(f"── group {i} " + "─" * 60)
        width = max(len(k) for k in row)
        for key, value in row.items():
            print(f"  {key:<{width}}  {value}")
    if args.csv:
        columns = list(dict.fromkeys(k for row in rows for k in row))
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, restval="-")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
