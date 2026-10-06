# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Replay barge-in verdicts against the live frontend model (barge-in plan, section 11.7).

It measures a change to the frontend prompt templates or the same-query guard
before anyone talks to the agent. For each ``barge_in_verdict`` in a recorded
event log, and for each case of an optional cases file, it asks the frontend
the probe question again, through the real runner with the current templates,
and writes one JSONL record per probe:

* ``source``: ``log`` or ``case``; ``session_id`` / ``turn_id`` for log items;
* ``new_words``, ``running_query``, ``old_verdict`` / ``old_reason`` (log items),
  ``expected`` (cases);
* ``verdict``, ``reason``, ``model_task``, ``probe_query``, ``latency_ms``.

Each probe starts from an empty conversation (the log does not hold the earlier
turns), and the frontend's capability list comes from the profile's config
tools only, so a profile with client tools (tau3) replays without them. Run
from the repository root, with the frontend model's credentials in the
environment::

    PYTHONPATH=src uv run python -m examples.frontend_backend_verdict.voice.cli.verdict_replay \
        --events logs/fba_voice_web_events.jsonl \
        --config src/prototypes/voice_frontend_backend_agent/config/profiles/browser_demo_frontend_verdict.yaml \
        --cases misc/prototypes/voice/verdict_cases.jsonl --guard off --out /tmp/verdict_replay.jsonl

A cases file holds one JSON object per line: ``request`` (the running turn's
words followed by the new words), ``new_words``, ``running_query``, optional
``filler`` and ``filler_heard``, and ``expected`` (``continue`` or ``new``).
The summary goes to stderr. The log must not have been written with
``logging.redact_content: true``.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TextIO

from examples.frontend_backend_verdict.voice.agent.port import InFlight
from examples.frontend_backend_verdict.voice.agent.runner import (
    AgentClients,
    TextAgentRunner,
    frontend_verdict_settings,
)
from examples.frontend_backend_verdict.voice.agent.sinks import EventLog, SessionRoutingSink
from examples.frontend_backend_verdict.voice.config import VoiceConfig, load_voice_config, prompt_context


@dataclass(frozen=True, slots=True)
class Item:
    """One probe question: a recorded verdict or a fixed case."""

    source: str
    request: str
    new_words: str
    running_query: str
    filler: str = ""
    filler_heard: bool = False
    expected: str = ""
    old_verdict: str = ""
    old_reason: str = ""
    session_id: str = ""
    turn_id: int | None = None


def _records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def log_items(records: Iterable[dict[str, Any]], model: str = "") -> list[Item]:
    """The recorded ``barge_in_verdict`` records that carried text (not ``no_transcript``), in log order."""
    models: dict[str, str] = {}
    filler: dict[str, str] = {}
    heard: dict[tuple[str, Any], bool] = {}
    items: list[Item] = []
    for record in records:
        session_id, kind = str(record.get("session_id") or ""), record.get("kind")
        if kind == "session_start":
            models[session_id] = str(record.get("model") or "")
        if model and models.get(session_id) != model:
            continue
        if kind == "filler":
            filler[session_id] = str(record.get("text") or "")
        elif kind == "barge_in_review":
            heard[(session_id, record.get("turn_id"))] = bool(record.get("filler_spoken"))
        elif kind == "barge_in_verdict" and record.get("merged_text") and record.get("utterance"):
            items.append(
                Item(
                    source="log",
                    request=str(record["merged_text"]),
                    new_words=str(record["utterance"]),
                    running_query=str(record.get("running_query") or ""),
                    filler=filler.get(session_id, ""),
                    filler_heard=heard.get((session_id, record.get("turn_id")), False),
                    old_verdict=str(record.get("verdict") or ""),
                    old_reason=str(record.get("reason") or ""),
                    session_id=session_id,
                    turn_id=record.get("turn_id"),
                )
            )
    return items


def case_items(path: Path) -> list[Item]:
    """The fixed cases of a cases file."""
    return [
        Item(
            source="case",
            request=str(case["request"]),
            new_words=str(case["new_words"]),
            running_query=str(case["running_query"]),
            filler=str(case.get("filler") or ""),
            filler_heard=bool(case.get("filler_heard", False)),
            expected=str(case.get("expected") or ""),
        )
        for case in _records(path)
    ]


def _with_guard(config: VoiceConfig, guard: str) -> VoiceConfig:
    if guard == "config":
        return config
    verdict = replace(config.barge_in.frontend_verdict, same_query_guard=guard == "on")
    return replace(config, barge_in=replace(config.barge_in, frontend_verdict=verdict))


def build_runner(config: VoiceConfig, clients: AgentClients) -> TextAgentRunner:
    """A runner as the server builds it, with the profile's config tools and no session instructions."""
    runner = TextAgentRunner(
        base_config=config.agent,
        tools_config=config.tools,
        instructions_config=config.instructions,
        clients=clients,
        sink=SessionRoutingSink(EventLog()),
        session_id="verdict-replay",
        normalization=config.normalization,
        barge_in=frontend_verdict_settings(config),
        prompt_context=prompt_context(config),
    )
    runner.configure(tools=(), instructions="")
    return runner


async def replay(config: VoiceConfig, clients: AgentClients, items: list[Item]) -> list[dict[str, Any]]:
    """Probe every item (sequentially: latencies stay comparable) and return the output records."""
    results: list[dict[str, Any]] = []
    for item in items:
        runner = build_runner(config, clients)  # a fresh, empty conversation per probe
        in_flight = InFlight(query=item.running_query, filler_text=item.filler)
        record: dict[str, Any] = {
            "source": item.source,
            "session_id": item.session_id or None,
            "turn_id": item.turn_id,
            "new_words": item.new_words,
            "running_query": item.running_query,
            "old_verdict": item.old_verdict or None,
            "old_reason": item.old_reason or None,
            "expected": item.expected or None,
        }
        try:
            probe = await runner.probe_request(
                item.request, in_flight=in_flight, new_words=item.new_words, filler_spoken=item.filler_heard
            )
        except Exception as exc:  # noqa: BLE001 - one failed probe must not end the replay
            record.update(verdict="error", reason=f"{type(exc).__name__}: {exc}")
        else:
            record.update(
                verdict=probe.verdict,
                reason=probe.reason,
                model_task=probe.model_task,
                probe_query=probe.probe_query,
                latency_ms=round(probe.latency_ms, 1),
            )
        results.append(record)
    return results


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * len(ordered) + 0.5) - 1))]


def summary(results: list[dict[str, Any]], guard: bool) -> dict[str, Any]:
    """Counts for stderr: accuracy on cases, changes against the log, guard overrides, latency."""
    counts: collections.Counter[str] = collections.Counter()
    for record in results:
        counts[f"{record['source']}:{record['verdict']}"] += 1
        if record.get("expected"):
            counts["cases_expected"] += 1
            counts["cases_correct"] += record["verdict"] == record["expected"]
        if record.get("old_verdict"):
            counts["log_changed"] += record["verdict"] != record["old_verdict"]
        counts["same_query_overrides"] += record.get("reason") == "same_query"
    latencies = [float(r["latency_ms"]) for r in results if r.get("latency_ms") is not None]
    return {
        "guard": guard,
        **dict(counts),
        "latency_ms_p50": _percentile(latencies, 0.5),
        "latency_ms_p95": _percentile(latencies, 0.95),
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", type=Path, default=None, help="event log JSONL with barge_in_verdict records")
    parser.add_argument("--config", required=True, type=Path, help="voice profile with the frontend verdict")
    parser.add_argument("--model", default="", help="only sessions whose session_start model equals this")
    parser.add_argument("--cases", type=Path, default=None, help="fixed cases JSONL (see above)")
    parser.add_argument(
        "--guard", choices=("config", "on", "off"), default="config", help="same_query_guard (default: the profile)"
    )
    parser.add_argument("--out", type=Path, default=None, help="output JSONL (default: stdout)")
    args = parser.parse_args(argv)
    if args.events is None and args.cases is None:
        parser.error("give --events, --cases or both")
    return args


def main(argv: list[str] | None = None, *, clients: AgentClients | None = None) -> dict[str, Any]:
    """Run the replay; ``clients`` replaces the profile's live LLM clients (tests)."""
    args = _parse_args(argv)
    config = load_voice_config(args.config)
    if config.barge_in.while_thinking != "frontend_verdict":
        raise SystemExit(f"{args.config}: barge_in.while_thinking must be frontend_verdict to replay verdicts")
    config = _with_guard(config, args.guard)
    if clients is None:
        from examples.frontend_backend_verdict.voice.server import build_clients  # noqa: PLC0415 - live runs only

        clients = build_clients(config)
    items = (log_items(_records(args.events), args.model) if args.events else []) + (
        case_items(args.cases) if args.cases else []
    )
    results = asyncio.run(replay(config, clients, items))
    out: TextIO = args.out.open("w", encoding="utf-8") if args.out else sys.stdout
    try:
        for record in results:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if args.out:
            out.close()
    report = summary(results, config.barge_in.frontend_verdict.same_query_guard)
    print(json.dumps(report, sort_keys=True), file=sys.stderr)
    return report


if __name__ == "__main__":
    main()
