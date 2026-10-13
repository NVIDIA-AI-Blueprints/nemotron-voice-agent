# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Gate G1 of tau3-geval-failure-fixes-plan.md: replay recorded tool calls through an arm's relay.

Every backend tool call of a voice event log (``tool_calls_out`` with ``call_ids`` and
``arguments``, written since 2026-10-07) and its client output (``tool_output_in``) is fed
through two relays built from the session's tool schemas: the control profile's and the arm's.
The result each relay returns to the backend is compared per call:

* outside a format-retry trigger the two must be byte-identical (``unexpected`` otherwise);
* every trigger must be on a genuine phone field of a read tool (``format_retry`` events);
* a retry the arm's relay issues itself (A1 on a declared read-only tool) is answered with a
  recorded output of the same tool and arguments from anywhere in the log, else as a transient
  error; each found-after-retry result is listed for the identity check.

The session's tool schemas come from ``--session-update`` files (a Realtime ``session.update``,
such as the tau2 fixtures), matched to each session by ``session_tools_sha256``. A3 is not
replayed here (it needs the user turns): review the ``field_provenance`` events of an A3 run.
Exit 1 on any unexpected difference.

    PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.cli.relay_replay \
        logs/fdh_voice_events.realtime_eval.jsonl \
        --profile src/examples/frontend_delegation_hermes/config/profiles/realtime_eval_arm_a1_format_retry.yaml \
        --session-update tests/unit/frontend_delegation_hermes/tau/fixtures/tau2_session_update/telecom.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from examples.frontend_delegation_hermes.core.normalization.schema_rules import tools_sha256
from examples.frontend_delegation_hermes.core.text.messages import canonical_json

DEFAULT_CONTROL = Path(__file__).resolve().parents[1] / "config" / "profiles" / "realtime_eval_control.yaml"
NO_RECORD = "Error: no recorded output for this call (relay_replay)"


def sessions_of(events: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """``session_id -> events`` that matter here, in log order."""
    kinds = ("session_rules", "tool_calls_out", "tool_output_in", "tool_cancelled")
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("kind") in kinds:
            out[str(event.get("session_id"))].append(dict(event))
    return out


def recorded_outputs(sessions: Mapping[str, list[dict[str, Any]]]) -> dict[tuple[str, str], str]:
    """``(tool, canonical arguments) -> client output`` over the whole log (the first one wins)."""
    table: dict[tuple[str, str], str] = {}
    for events in sessions.values():
        calls: dict[str, tuple[str, str]] = {}
        for event in events:
            if event["kind"] == "tool_calls_out":
                for call_id, name, arguments in zip(
                    event.get("call_ids") or (), event.get("calls") or (), event.get("arguments") or (), strict=False
                ):
                    calls[str(call_id)] = (str(name), _canonical(arguments))
            elif event["kind"] == "tool_output_in" and str(event.get("call_id")) in calls:
                table.setdefault(calls[str(event["call_id"])], str(event.get("output") or ""))
    return table


def _canonical(arguments: Any) -> str:
    try:
        return canonical_json(json.loads(arguments or "{}"))
    except (TypeError, json.JSONDecodeError):
        return str(arguments)


class _Wire:
    """One relay fed with a session's recorded traffic; synthetic retries answered from the log."""

    def __init__(self, runtime: Any, tools: Sequence[Mapping[str, Any]], table: Mapping[tuple[str, str], str]):
        from examples.frontend_delegation_hermes.core.tools.relay import ToolRelay  # noqa: PLC0415

        rules = runtime.session_rules(tools)
        self.results: dict[str, str] = {}
        self.logs: list[tuple[str, dict[str, Any]]] = []
        self._table = table
        self._emitted: list[Any] = []
        self.relay = ToolRelay(
            executor="wire",
            send_result=lambda call_id, _epoch, output, _local: self.results.__setitem__(call_id, output),
            emit_calls=self._emitted.append,
            log=lambda kind, **data: self.logs.append((kind, data)),
            normalizer=rules.argument_normalizer,
            result_hints=rules.result_hints,
            batch_window_ms=1,
            format_retry=rules.format_retry,
            format_messages=runtime.format_messages,
            result_timeout_s=5.0,
        )

    async def batch(self, call_ids: Sequence[str], names: Sequence[str], arguments: Sequence[str]) -> None:
        for call_id, name, args in zip(call_ids, names, arguments, strict=False):
            self.relay.on_call({"call_id": call_id, "epoch": 1, "name": name, "arguments": args})
        await asyncio.sleep(0.005)
        self._emitted.clear()  # the recorded batch: its outputs come from the log

    async def output(self, call_id: str, output: str) -> None:
        try:
            self.relay.on_function_output(call_id, output)
        except Exception:  # noqa: BLE001 - an id this relay answered locally and never sent
            return
        while self.relay.take_ack():
            pass
        while self._emitted:  # the relay's own format retries: answered from the log
            for retry_id, name, args in self._emitted.pop(0).calls:
                self.relay.on_function_output(retry_id, self._table.get((name, _canonical(args)), NO_RECORD))
                while self.relay.take_ack():
                    pass

    def cancel(self, call_id: str) -> None:
        self.relay.on_cancel(call_id, "replay")

    async def close(self) -> None:
        await self.relay.close()


async def replay_session(
    events: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    control: Any,
    arm: Any,
    table: Mapping[tuple[str, str], str],
) -> dict[str, Any]:
    """Feed one session to both relays; the per-call differences and the arm's format-retry events."""
    wires = (_Wire(control, tools, table), _Wire(arm, tools, table))
    calls = 0
    for event in events:
        if event["kind"] == "tool_calls_out" and not event.get("format_retry"):
            ids, names, arguments = event.get("call_ids"), event.get("calls"), event.get("arguments")
            if ids is None or arguments is None:
                return {"skipped": "tool_calls_out without call_ids/arguments (log older than 2026-10-07)"}
            calls += len(ids)
            for wire in wires:
                await wire.batch([str(i) for i in ids], [str(n) for n in names], [str(a) for a in arguments])
        elif event["kind"] == "tool_output_in" and not str(event.get("call_id")).endswith(".fmt"):
            for wire in wires:
                await wire.output(str(event["call_id"]), str(event.get("output") or ""))
        elif event["kind"] == "tool_cancelled":
            for wire in wires:
                wire.cancel(str(event["call_id"]))
    for wire in wires:
        await wire.close()
    base, new = wires
    retries = [data for kind, data in new.logs if kind == "format_retry"]
    triggered = {data["call_id"] for data in retries}
    changed = sorted(call_id for call_id in new.results if new.results[call_id] != base.results.get(call_id))
    unexpected = [call_id for call_id in changed if call_id not in triggered]
    found = [
        {"call_id": d["call_id"], "tool": d["tool"], "after": d["after"], "output": new.results.get(d["call_id"], "")}
        for d in retries
        if d["outcome"] == "found"
    ]
    return {"calls": calls, "changed": changed, "unexpected": unexpected, "format_retry": retries, "found": found}


def _runtime(profile: str) -> Any:
    from examples.frontend_delegation_hermes.core.config import load_delegation_config  # noqa: PLC0415
    from examples.frontend_delegation_hermes.core.runtime import DelegationRuntime  # noqa: PLC0415

    return DelegationRuntime(load_delegation_config(profile), chat_model=object())  # type: ignore[arg-type]


async def run(args: argparse.Namespace) -> int:
    """Replay every session whose tools match a ``--session-update``."""
    events = [json.loads(line) for line in Path(args.events).read_text(encoding="utf-8").splitlines() if line.strip()]
    sessions = sessions_of(events)
    tools_by_sha: dict[str, list[dict[str, Any]]] = {}
    for path in args.session_update:
        tools = list(json.loads(Path(path).read_text(encoding="utf-8"))["session"]["tools"])
        tools_by_sha[tools_sha256(tools)] = tools
    control, arm = _runtime(args.control_profile), _runtime(args.profile)
    table = recorded_outputs(sessions)
    totals: Counter[str] = Counter()
    triggers: Counter[str] = Counter()
    report: dict[str, Any] = {"profile": args.profile, "control_profile": args.control_profile, "sessions": {}}
    for session_id, session_events in sessions.items():
        sha = next((e.get("session_tools_sha256") for e in session_events if e["kind"] == "session_rules"), "")
        tools = tools_by_sha.get(str(sha))
        if tools is None:
            totals["sessions_without_tools"] += 1
            continue
        result = await replay_session(session_events, tools, control, arm, table)
        report["sessions"][session_id] = result
        if "skipped" in result:
            totals["sessions_skipped"] += 1
            continue
        totals["sessions"] += 1
        totals["calls"] += result["calls"]
        totals["changed"] += len(result["changed"])
        totals["unexpected"] += len(result["unexpected"])
        totals["found_after_retry"] += len(result["found"])
        for retry in result["format_retry"]:
            triggers[f"{retry['tool']}.{retry['argument']} {retry['mode']}:{retry['outcome']}"] += 1
        for call_id in result["unexpected"]:
            print(f"UNEXPECTED {session_id} {call_id}: the arm changed a result without a format-retry trigger")
        for found in result["found"]:
            print(f"FOUND-AFTER-RETRY {session_id} {found['call_id']} as {found['after']}: check the identity")
    report["totals"], report["triggers"] = dict(totals), dict(triggers)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"totals": dict(totals), "triggers": dict(triggers)}, indent=2, sort_keys=True))
    ok = totals["unexpected"] == 0 and totals["sessions"] > 0
    print("G1: OK" if ok else "G1: FAILED")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("events", help="voice event log (tool_calls_out with call_ids and arguments)")
    parser.add_argument("--profile", required=True, help="the arm's profile")
    parser.add_argument("--control-profile", default=str(DEFAULT_CONTROL), help="default: realtime_eval_control.yaml")
    parser.add_argument("--session-update", nargs="+", required=True, help="session.update JSON files (tool schemas)")
    parser.add_argument("--out", default="", help="JSON report")
    args = parser.parse_args(argv)
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
