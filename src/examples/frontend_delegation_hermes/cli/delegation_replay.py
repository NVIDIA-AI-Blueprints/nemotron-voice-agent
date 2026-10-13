# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Replay frontend decisions offline against the live frontend model (plan 6.4, gate G3).

Each case of ``delegation_cases.jsonl`` gives a backend state, a conversation and the
latest user turn, and the expected ``delegate`` / ``request`` / empty-filler outcome.
The replay builds exactly the request the voice server would send and reports
accuracy per workflow cell and per class (CSV cells, answers to backend questions,
regular speech: backchannels, vocal tics, non-directed speech).

A case with ``requires_feature`` (a frontend prompt variant, e.g. ``replay_intent``) only
runs when the profile turns that variant on (``--config .../tau3_arm_m2_replay.yaml``).

    PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.cli.delegation_replay \
        --config src/examples/frontend_delegation_hermes/config/profiles/tau3_eval.yaml --domain mock
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from examples.frontend_delegation_hermes.core.agent_tools import capability_lines, realtime_tools_to_specs
from examples.frontend_delegation_hermes.core.config import load_delegation_config
from examples.frontend_delegation_hermes.core.frontend.decider import DecisionContext, LLMDecider, apply_guards
from examples.frontend_delegation_hermes.core.frontend.llm import OpenAIChatModel
from examples.frontend_delegation_hermes.core.frontend.prompts import load_prompts

REPO = Path(__file__).resolve().parents[4]
DEFAULT_CASES = REPO / "src" / "examples" / "frontend_delegation_hermes" / "fixtures" / "delegation_cases.jsonl"
TAU2_FIXTURES = REPO / "tests" / "unit" / "frontend_delegation_hermes" / "tau" / "fixtures" / "tau2_session_update"

#: Gate G3 thresholds.
GATES = {"delegate": 0.95, "request_working": 0.95, "backend_answer": 1.0, "regular_speech_class": 0.95}


def _capabilities(domain: str) -> list[str]:
    if not domain:
        return []
    update = json.loads((TAU2_FIXTURES / f"{domain}.json").read_text())
    return list(capability_lines(realtime_tools_to_specs(update["session"]["tools"])))


def _check(case: dict[str, Any], decision: Any) -> dict[str, bool]:
    expect = case["expect"]
    checks = {"delegate": decision.delegate == expect["delegate"]}
    if "request" in expect:
        checks["request"] = decision.request == expect["request"]
    if expect.get("filler_empty"):
        checks["filler_empty"] = decision.filler_text == ""
    return checks


async def replay(args: argparse.Namespace) -> int:
    """Run every case and print (and optionally write) the report; exit 1 when a gate fails."""
    config = load_delegation_config(args.config)
    prompts = load_prompts(config.delegation.prompts_path, prompt_features=config.prompt_features)
    model = OpenAIChatModel(config.frontend.llm)
    decider = LLMDecider(
        model,
        prompts,
        timeout_ms=config.frontend.timeout_ms,
        max_repair_attempts=config.delegation.max_repair_attempts,
        on_contract_error=config.delegation.on_contract_error,
        tool_choice=config.frontend.tool_choice,
        parallel_tool_calls=config.frontend.parallel_tool_calls,
        hedge_after_ms=config.frontend.hedge_after_ms,
    )
    note = prompts.render("normalization_note") if config.voice.normalization.transcript.enabled else ""
    system = prompts.render("frontend_system", capabilities=_capabilities(args.domain), normalization_note=note)
    cases = [json.loads(line) for line in Path(args.cases).read_text().splitlines() if line.strip()]
    if args.only:
        cases = [case for case in cases if args.only in case["id"] or args.only in case["cell"]]
    skipped = [case["id"] for case in cases if not config.prompt_features.get(case.get("requires_feature", ""), True)]
    cases = [case for case in cases if case["id"] not in skipped]
    if skipped:
        print(f"skipped {len(skipped)} case(s) whose prompt variant is off: {', '.join(skipped)}")
    semaphore = asyncio.Semaphore(args.concurrency)

    async def run(case: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            messages = [*case["history"], {"role": "user", "content": case["user"]}]
            context = DecisionContext(
                system_prompt=system,
                messages=[m for m in messages if m["role"] in ("user", "assistant")],
                user_text=case["user"],
                state=case["state"],
                current_request=case.get("current_request", ""),
                elapsed_s=4 if case["state"] == "WORKING" else None,
                recent_activity=case.get("recent_activity", []),
                backend_asked_question=bool(case.get("backend_asked_question")),
                last_answer_unheard=bool(case.get("last_answer_unheard")),
            )
            decision = await decider.decide(context)
            if not args.no_guards:
                decision = apply_guards(
                    decision,
                    context,
                    backend_question_needs_delegate=config.delegation.backend_question_needs_delegate,
                    backchannel_words=config.delegation.backchannel_words,
                )
            checks = _check(case, decision)
            return {
                "id": case["id"],
                "cell": case["cell"],
                "class": case["class"],
                "state": case["state"],
                "user": case["user"],
                "expect": case["expect"],
                "got": {
                    "delegate": decision.delegate,
                    "request": decision.request,
                    "filler_text": decision.filler_text,
                },
                "repair": decision.repair,
                "latency_ms": int(decision.latency_ms),
                "ok": all(checks.values()),
                "checks": checks,
            }

    results = await asyncio.gather(*(run(case) for case in cases))
    await model.aclose()
    report = _report(results)
    for row in results:
        mark = "ok  " if row["ok"] else "FAIL"
        print(f"{mark} {row['id']:<24} {row['state']:<10} {row['user'][:48]!r:<52} -> {row['got']}")
    print(json.dumps(report["summary"], indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": report["summary"], "results": results}, indent=2))
    return 0 if report["summary"]["gates_passed"] else 1


def _report(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_cell: dict[str, list[bool]] = defaultdict(list)
    by_class: dict[str, list[bool]] = defaultdict(list)
    delegate_ok, request_working = [], []
    for row in results:
        by_cell[row["cell"]].append(row["ok"])
        cls = row["class"] if row["class"] != "regular_speech" else f"regular_speech:{row['cell'].split('|')[0]}"
        by_class[cls].append(row["ok"])
        delegate_ok.append(row["checks"]["delegate"])
        if row["state"] == "WORKING" and "request" in row["checks"]:
            request_working.append(row["checks"]["request"])

    def rate(values: list[bool]) -> float:
        return round(sum(values) / len(values), 3) if values else 1.0

    classes = {name: rate(values) for name, values in sorted(by_class.items())}
    gates = {
        "delegate": rate(delegate_ok) >= GATES["delegate"],
        "request_working": rate(request_working) >= GATES["request_working"],
        "backend_answer": classes.get("backend_answer", 1.0) >= GATES["backend_answer"],
        "regular_speech": all(
            v >= GATES["regular_speech_class"] for k, v in classes.items() if k.startswith("regular")
        ),
    }
    latencies = sorted(row["latency_ms"] for row in results) or [0]
    summary = {
        "cases": len(results),
        "accuracy": rate([row["ok"] for row in results]),
        "delegate_accuracy": rate(delegate_ok),
        "request_accuracy_working": rate(request_working),
        "by_class": classes,
        "by_cell": {name: rate(values) for name, values in sorted(by_cell.items())},
        "latency_ms_p50": latencies[len(latencies) // 2],
        "latency_ms_p90": latencies[int(len(latencies) * 0.9) - 1 if len(latencies) > 1 else 0],
        "repairs": sum(1 for row in results if row["repair"]),
        "gates": gates,
        "gates_passed": all(gates.values()),
    }
    return {"summary": summary}


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config", default=str(REPO / "src/examples/frontend_delegation_hermes/config/profiles/tau3_eval.yaml")
    )
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument(
        "--domain", default="mock", help="tau2 fixture whose tools give the capability lines ('' = none)"
    )
    parser.add_argument("--only", default="", help="substring filter on case id or cell")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--no-guards", action="store_true", help="score the raw model decision (no code guards)")
    parser.add_argument("--out", default="", help="write the JSON report here")
    sys.exit(asyncio.run(replay(parser.parse_args(argv))))


if __name__ == "__main__":
    main()
