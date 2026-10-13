# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Delegation event log -> the per-turn records ``fba_voice_metrics.py`` reads (plan section 14).

``fba_voice_metrics.py`` (tau2-bench-smasurekar) assumes one backend operation per
frontend turn. Here a turn may start a run, steer another turn's run, ask for status or
stay local, and a run may span several turns. This adapter attributes every backend
run (usage, latency, tool calls, answer) to the turn that **started** it and writes a
legacy-shaped JSONL:

* passed through: ``session_start``, ``session_end``, ``speech_stopped``, ``tool_output_in``,
  ``response_done``, ``turn_latency``;
* synthesized: ``agent_turn_start``, ``agent_turn_done`` (with ``frontend``/``backend`` role
  usage), ``filler_timing`` (delegated turns whose filler was spoken).

Self-checks (exit code 1): a client tool ``call_id`` with no run, or a run with no starting turn.

    PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.cli.report_adapter \
        logs/fdh_voice_events.jsonl --out logs/fdh_voice_events.legacy.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

#: The suffix of the relay's own format-retry call ids (``tools/relay.py`` ``RETRY_SUFFIX``).
RETRY_SUFFIX = ".fmt"
PASS_THROUGH = ("session_start", "session_end", "speech_stopped", "tool_output_in", "response_done", "turn_latency")
STARTING_ACTIONS = ("start", "continue", "pending_steer", "retry")


def _role(calls: int, usage: dict[str, Any], latency_ms: float) -> dict[str, Any]:
    prompt = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    completion = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    return {
        "calls": calls,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cached_tokens": int(usage.get("cached_tokens", 0) or 0),
        "total_tokens": prompt + completion,
        "latency_ms": round(latency_ms, 1),
    }


def adapt(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Adapt every session in ``records``; returns (legacy records, self-check problems)."""
    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        sessions[str(record.get("session_id"))].append(record)
    out: list[dict[str, Any]] = []
    problems: list[str] = []
    for session_id, rows in sessions.items():
        legacy, issues = _adapt_session(session_id, rows)
        out.extend(legacy)
        problems.extend(issues)
    out.sort(key=lambda row: float(row.get("timestamp") or 0))
    return out, problems


def _adapt_session(session_id: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    decisions = {r["turn_id"]: r for r in rows if r["kind"] == "delegation_decision"}
    run_start: dict[str, dict[str, Any]] = {}
    turn_links: dict[int, list[str]] = defaultdict(list)
    for r in rows:
        if r["kind"] != "backend_action" or not r.get("run_id"):
            continue
        action = r.get("action")
        if action in STARTING_ACTIONS and r["run_id"] not in run_start:
            run_start[r["run_id"]] = r
        turn_links[int(r["turn_id"])].append(str(action))
    runs = [r for r in rows if r["kind"] == "backend_run_done"]
    fillers = {r["turn_id"]: r for r in rows if r["kind"] == "filler_timing"}
    answers = defaultdict(list)
    for r in rows:
        if r["kind"] == "backend_answer" and r.get("run_id"):
            answers[r["run_id"]].append(r)

    problems: list[str] = []
    by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
    call_to_run: dict[str, str] = {}
    for run in runs:
        started = run_start.get(run["run_id"])
        owner = started.get("turn_id") if started else (run.get("turn_ids") or [None])[0]
        if owner is None:
            problems.append(f"{session_id}: run {run['run_id']} has no starting turn")
            continue
        by_turn[int(owner)].append(run)
        for call_id in run.get("tool_call_ids") or ():
            call_to_run[str(call_id)] = run["run_id"]
    for r in rows:
        # A format retry (``<id>.fmt``, A1) is the relay's own call for the backend call ``<id>``.
        call_id = str(r.get("call_id")).removesuffix(RETRY_SUFFIX)
        if r["kind"] == "tool_output_in" and not r.get("local_executor") and call_id not in call_to_run:
            problems.append(f"{session_id}: tool call {r.get('call_id')} has no backend run")

    legacy = [dict(r) for r in rows if r["kind"] in PASS_THROUGH]
    for turn_id, decision in decisions.items():
        ts = float(decision["timestamp"])
        frontend_ms = float(decision.get("latency_ms") or 0)
        legacy.append({"timestamp": ts - frontend_ms / 1000.0, "kind": "agent_turn_start", "session_id": session_id,
                       "turn_id": turn_id, "text": decision.get("text", "")})  # fmt: skip
        owned = by_turn.get(int(turn_id), [])
        backend_usage: dict[str, Any] = defaultdict(int)
        backend_ms = 0.0
        backend_calls = 0
        for run in owned:
            for key, value in (run.get("usage") or {}).items():
                if isinstance(value, (int, float)):
                    backend_usage[key] += value
            started = run_start.get(run["run_id"])
            if started is not None:
                backend_ms += max(0.0, (float(run["timestamp"]) - float(started["timestamp"])) * 1000.0)
            backend_calls += int((run.get("usage") or {}).get("api_calls", 1) or 1)
        links = turn_links.get(int(turn_id), [])
        if owned:
            outcome = "text" if any(answers.get(run["run_id"]) for run in owned) else str(owned[-1].get("status"))
        elif "status" in links:
            outcome = "status"
        elif any(link in ("steer", "redirect", "queued_in_delay") for link in links):
            outcome = "steer"
        else:
            outcome = "text" if decision.get("filler") else "no_reply"
        frontend = _role(1, decision.get("usage") or {}, frontend_ms)
        backend = _role(backend_calls, backend_usage, backend_ms)
        legacy.append({
            "timestamp": ts + backend_ms / 1000.0, "kind": "agent_turn_done", "session_id": session_id,
            "turn_id": turn_id, "outcome": outcome, "latency_ms": round(frontend_ms + backend_ms, 1),
            "input_tokens": frontend["prompt_tokens"] + backend["prompt_tokens"],
            "output_tokens": frontend["completion_tokens"] + backend["completion_tokens"],
            "frontend": frontend, "backend": backend, "backend_link": links[0] if links else "",
            "delegate": bool(decision.get("delegate")), "request": decision.get("request"),
        })  # fmt: skip
        timing = fillers.get(turn_id)
        if decision.get("delegate") and timing is not None:
            legacy.append({
                "timestamp": float(timing["timestamp"]), "kind": "filler_timing", "session_id": session_id,
                "turn_id": turn_id, "mode": "speak", "text": decision.get("filler", ""), "would_have_spoken": True,
                "outcome": "answer" if outcome == "text" else outcome,
                "filler_ready": {
                    "frontend_latency_ms": frontend_ms,
                    "since_turn_end_ms": decision.get("user_stop_to_decision_ms"),
                },
                "user_stop_to_first_audio_ms": timing.get("user_stop_to_first_audio_ms"),
            })  # fmt: skip
    return legacy, problems


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("event_log", help="the voice server's JSONL event log (FDH_EVENT_LOG)")
    parser.add_argument("--out", default="", help="legacy JSONL (default: <event_log>.legacy.jsonl)")
    args = parser.parse_args(argv)
    source = Path(args.event_log)
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    legacy, problems = adapt(records)
    out = Path(args.out) if args.out else source.with_suffix(".legacy.jsonl")
    out.write_text("".join(json.dumps(row, default=str) + "\n" for row in legacy), encoding="utf-8")
    turns = sum(1 for row in legacy if row["kind"] == "agent_turn_done")
    print(f"wrote {len(legacy)} records ({turns} turns) to {out}")
    for problem in problems:
        print(f"CHECK FAILED: {problem}", file=sys.stderr)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
