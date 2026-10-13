# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Gate G2 of tau3-geval-failure-fixes-plan.md: replay the backend's write and transfer decisions with an arm's prompt.

Input: the decision traces of a clean control run (``hermes.backend_trace``, one
``<session>.trace.jsonl`` per worker in ``workers.log_dir``). Two steps:

``extract`` lists every decision point: a model response that calls a tool that is not a read
tool (by the session's schemas, ``schema_rules.is_read_tool``; transfers are write tools too).
Each gets the label ``U`` (unchanged). Before any arm is replayed, label the decisions of the
passing tasks (plan section 5): ``P`` for an expected policy correction (a write made without the
confirmation the policy requires, or a reduced change the user never agreed to), ``T`` for a
targeted decision of a failing task (reported, never gated), ``skip`` for a failing task's
decision that is not targeted. Optionally set ``next_user`` on a ``P`` decision (the user's own
later yes from the control; default ``"yes"``).

``replay`` re-sends each labelled decision's request with the arm's backend prompt (the control's
rendered ``backend_system`` inside the system message is replaced by the arm's render of the same
policy), ``--samples`` times at temperature 0:

* ``U``: every sample makes the control response's complete, ordered list of tool calls (names and
  canonical JSON arguments; fix-plan-f1-f2-f4-f8.md section 7). Decisions extracted before ``calls``
  existed fall back to "the same tool with the same arguments is among the sample's calls";
* ``P``: every sample first answers without a tool call (the confirmation); then, with the
  confirmation and ``next_user`` appended, every sample makes the same call;
* ``T``: how many samples changed the decision (information only).

Exit 1 when a ``U`` or ``P`` decision fails. ``--dry-run`` checks the prompt substitution only.

    PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.cli.backend_decision_replay \
        extract logs/fdh_workers --out logs/g2_decisions.jsonl
    PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.cli.backend_decision_replay \
        replay logs/g2_decisions.jsonl --traces logs/fdh_workers \
        --gateway-config src/examples/frontend_delegation_hermes/config/gateway.eval_b1_write_consent_v2.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from examples.frontend_delegation_hermes.core.normalization.schema_rules import is_read_tool, tool_function

LABELS = ("U", "P", "T", "skip")
POLICY_OPEN, POLICY_CLOSE = "<policy>\n", "\n</policy>"


@dataclass(slots=True)
class Decision:
    """One write or transfer decision of the control."""

    decision_id: str
    session_id: str
    api_request_id: str
    tool: str
    arguments: str
    label: str = "U"
    next_user: str = "yes"
    note: str = ""
    #: The control response's complete ordered tool calls ``[[name, canonical arguments], ...]``.
    calls: list[list[str]] = field(default_factory=list)


@dataclass(slots=True)
class Trace:
    """One session's trace records."""

    session: dict[str, Any] = field(default_factory=dict)
    requests: dict[str, dict[str, Any]] = field(default_factory=dict)
    responses: list[dict[str, Any]] = field(default_factory=list)


# -- traces and decisions ---------------------------------------------------------------------------


def load_traces(paths: Iterable[Path]) -> dict[str, Trace]:
    """``session_id -> Trace`` from trace files or directories of them."""
    traces: dict[str, Trace] = {}
    for path in paths:
        files = sorted(path.glob("*.trace.jsonl")) if path.is_dir() else [path]
        for file in files:
            for line in file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                trace = traces.setdefault(str(record.get("session_id") or file.stem), Trace())
                kind = record.get("kind")
                if kind == "session":
                    trace.session = record  # a respawned worker appends a new one; the tools are the same
                elif kind == "request":
                    trace.requests[str(record.get("api_request_id"))] = record
                elif kind == "response":
                    trace.responses.append(record)
    return traces


def write_tools(session: Mapping[str, Any]) -> set[str]:
    """The session's tools that are not read tools (writes, transfers, anything unknown)."""
    out = set()
    for tool in session.get("tools") or []:
        function = tool_function(tool)
        name = str(function.get("name") or "")
        if name and not is_read_tool(name, str(function.get("description") or "")):
            out.add(name)
    return out


def tool_calls(message: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """``(name, canonical arguments)`` of an assistant message's tool calls."""
    out = []
    for call in (message or {}).get("tool_calls") or []:
        function = call.get("function") if isinstance(call, Mapping) else None
        if not isinstance(function, Mapping):
            continue
        out.append((str(function.get("name") or ""), canonical_arguments(function.get("arguments"))))
    return out


def canonical_arguments(raw: Any) -> str:
    """Arguments as sorted-key JSON (a string that is not JSON is kept as is)."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return raw
    return json.dumps(raw, sort_keys=True, ensure_ascii=False)


def extract(traces: Mapping[str, Trace]) -> list[Decision]:
    """Every response that calls a write tool, one decision per such call."""
    decisions: list[Decision] = []
    for session_id, trace in sorted(traces.items()):
        writes = write_tools(trace.session)
        for response in trace.responses:
            request_id = str(response.get("api_request_id"))
            if request_id not in trace.requests:
                continue
            calls = tool_calls(response.get("assistant_message"))
            for index, (name, arguments) in enumerate(calls):
                if name in writes:
                    decisions.append(
                        Decision(
                            decision_id=f"{session_id}:{request_id}:{index}",
                            session_id=session_id,
                            api_request_id=request_id,
                            tool=name,
                            arguments=arguments,
                            calls=[list(call) for call in calls],
                        )
                    )
    return decisions


# -- the arm's prompt -------------------------------------------------------------------------------


def policy_of(rendered: str) -> str:
    """The ``instructions`` inside a rendered ``backend_system`` ("" when it has no policy block)."""
    start = rendered.find(POLICY_OPEN)
    end = rendered.rfind(POLICY_CLOSE)
    return rendered[start + len(POLICY_OPEN) : end] if start >= 0 and end > start else ""


def arm_messages(messages: list[dict[str, Any]], control_prompt: str, arm_prompt: str) -> list[dict[str, Any]]:
    """The request's messages with the control's ``backend_system`` replaced by the arm's, exactly once."""
    out = [dict(message) for message in messages]
    hits = 0
    for message in out:
        if message.get("role") != "system":
            continue
        content = message.get("content")
        if isinstance(content, str):
            hits += content.count(control_prompt)
            message["content"] = content.replace(control_prompt, arm_prompt)
        elif isinstance(content, list):  # content parts (prompt-cache decoration)
            parts = [dict(part) if isinstance(part, Mapping) else part for part in content]
            for part in parts:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    hits += part["text"].count(control_prompt)
                    part["text"] = part["text"].replace(control_prompt, arm_prompt)
            message["content"] = parts
    if hits != 1:
        raise ValueError(f"the control's backend_system occurs {hits} times in the system messages (expected 1)")
    return out


def render_arm_prompt(gateway_config: str, control_prompt: str) -> str:
    """The arm's ``backend_system`` for the same policy."""
    from examples.frontend_delegation_hermes.core.prompt_features import render_backend_system  # noqa: PLC0415
    from examples.frontend_delegation_hermes.core.sidecar.gateway_config import load_gateway_config  # noqa: PLC0415
    from examples.frontend_delegation_hermes.core.sidecar.templates import BackendTemplates  # noqa: PLC0415

    config = load_gateway_config(gateway_config)
    templates = BackendTemplates(config.hermes.prompts_path, prompt_features=config.prompt_features)
    return render_backend_system(templates, policy_of(control_prompt))


# -- verdicts ---------------------------------------------------------------------------------------


def verdict(decision: Decision, first: list[list[tuple[str, str]]], second: list[list[tuple[str, str]]]) -> str:
    """``pass`` / ``fail`` (U, P) or ``changed`` / ``unchanged`` (T) from the sampled tool calls."""
    wanted = (decision.tool, decision.arguments)
    if decision.label == "U":
        if decision.calls:
            sequence = [tuple(call) for call in decision.calls]
            return "pass" if all(calls == sequence for calls in first) else "fail"
        return "pass" if all(wanted in calls for calls in first) else "fail"
    if decision.label == "P":
        asked = all(not calls for calls in first)
        made = bool(second) and all(wanted in calls for calls in second)
        return "pass" if asked and made else "fail"
    return "unchanged" if all(wanted in calls for calls in first) else "changed"


async def _sample(client: Any, model: str, messages: list, tools: list, overrides: Mapping[str, Any]) -> dict:
    response = await client.chat.completions.create(
        model=model, messages=messages, tools=tools or None, temperature=0, **dict(overrides)
    )
    message = response.choices[0].message
    calls = [
        {"function": {"name": call.function.name, "arguments": call.function.arguments}}
        for call in (message.tool_calls or [])
    ]
    return {"role": "assistant", "content": message.content or "", "tool_calls": calls}


async def replay_decision(
    client: Any, decision: Decision, trace: Trace, arm_prompt: str, *, model: str, overrides: Mapping, samples: int
) -> dict[str, Any]:
    """Sample the arm at one decision point (and, for ``P``, at the turn after the confirmation)."""
    control_prompt = str(trace.session.get("ephemeral_system_prompt") or "")
    request = trace.requests[decision.api_request_id]
    messages = arm_messages(list(request.get("messages") or []), control_prompt, arm_prompt)
    tools = list(trace.session.get("tools") or [])
    first_messages = [await _sample(client, model, messages, tools, overrides) for _ in range(samples)]
    first = [tool_calls(message) for message in first_messages]
    second: list[list[tuple[str, str]]] = []
    if decision.label == "P" and all(not calls for calls in first):
        for message in first_messages:
            follow = [*messages, {"role": "assistant", "content": message["content"]}]
            follow.append({"role": "user", "content": decision.next_user})
            second.append(tool_calls(await _sample(client, model, follow, tools, overrides)))
    return {
        **asdict(decision),
        "verdict": verdict(decision, first, second),
        "first": [[list(c) for c in calls] for calls in first],
        "second": [[list(c) for c in calls] for calls in second],
        "confirmations": [m["content"] for m in first_messages] if decision.label == "P" else [],
    }


# -- CLI ------------------------------------------------------------------------------------------


def _read_decisions(path: Path) -> list[Decision]:
    decisions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            data = json.loads(line)
            if data.get("label") not in LABELS:
                raise SystemExit(f"{data.get('decision_id')}: label must be one of {LABELS}")
            decisions.append(Decision(**{k: v for k, v in data.items() if k in Decision.__dataclass_fields__}))
    return decisions


def _cmd_extract(args: argparse.Namespace) -> int:
    decisions = extract(load_traces(Path(p) for p in args.traces))
    lines = [json.dumps(asdict(d), ensure_ascii=False) for d in decisions]
    Path(args.out).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"{len(decisions)} decisions in {len({d.session_id for d in decisions})} sessions -> {args.out}")
    print(f"tools: {dict(Counter(d.tool for d in decisions))}")
    print("label them (U / P / T / skip) before any arm is replayed (plan section 5)")
    return 0


async def _cmd_replay(args: argparse.Namespace) -> int:
    from examples.frontend_delegation_hermes.core.sidecar.gateway_config import load_gateway_config  # noqa: PLC0415

    decisions = [d for d in _read_decisions(Path(args.decisions)) if d.label != "skip"]
    traces = load_traces(Path(p) for p in args.traces)
    config = load_gateway_config(args.gateway_config)
    arm_prompts: dict[str, str] = {}
    for decision in decisions:
        trace = traces.get(decision.session_id)
        if trace is None or decision.api_request_id not in trace.requests:
            raise SystemExit(f"{decision.decision_id}: no trace request (wrong --traces?)")
        control = str(trace.session.get("ephemeral_system_prompt") or "")
        arm = arm_prompts.setdefault(decision.session_id, render_arm_prompt(args.gateway_config, control))
        arm_messages(list(trace.requests[decision.api_request_id].get("messages") or []), control, arm)
    print(f"{len(decisions)} decisions; prompt substitution OK in every request")
    if args.dry_run:
        return 0
    from openai import AsyncOpenAI  # noqa: PLC0415

    hermes = config.hermes
    client = AsyncOpenAI(base_url=hermes.base_url, api_key=os.environ.get(hermes.api_key_env, ""))
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(decision: Decision) -> dict[str, Any]:
        async with semaphore:
            return await replay_decision(
                client,
                decision,
                traces[decision.session_id],
                arm_prompts[decision.session_id],
                model=hermes.model,
                overrides=hermes.request_overrides,
                samples=args.samples,
            )

    results = await asyncio.gather(*(one(d) for d in decisions))
    counts = Counter((r["label"], r["verdict"]) for r in results)
    report = {
        "gateway_config": args.gateway_config,
        "samples": args.samples,
        "counts": {f"{label}:{outcome}": n for (label, outcome), n in sorted(counts.items())},
        "decisions": results,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for result in results:
        if result["verdict"] in ("fail", "changed"):
            print(f"{result['label']} {result['verdict']}: {result['decision_id']} {result['tool']}")
    print(json.dumps(report["counts"], sort_keys=True))
    failed = sum(n for (label, outcome), n in counts.items() if outcome == "fail" and label in ("U", "P"))
    print("G2: OK" if not failed else f"G2: FAILED ({failed} decision(s))")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("extract", help="list the write and transfer decisions of control traces")
    p.add_argument("traces", nargs="+", help="trace files or directories (workers.log_dir)")
    p.add_argument("--out", required=True, help="decisions JSONL to label")
    p = sub.add_parser("replay", help="replay labelled decisions with an arm's backend prompt")
    p.add_argument("decisions", help="labelled decisions JSONL")
    p.add_argument("--traces", nargs="+", required=True, help="the control's trace files or directories")
    p.add_argument("--gateway-config", required=True, help="the arm's gateway config")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--out", default="", help="JSON report")
    p.add_argument("--dry-run", action="store_true", help="check the prompt substitution only (no model calls)")
    args = parser.parse_args(argv)
    if args.command == "extract":
        sys.exit(_cmd_extract(args))
    sys.exit(asyncio.run(_cmd_replay(args)))


if __name__ == "__main__":
    main()
