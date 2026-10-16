# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Drive the backend gateway directly, without audio (P0/P2 probe; plan section 16).

Plays the voice server's role over ``WS /v1/backend``: opens a session with a demo
``get_order`` tool, delegates a task, executes the tool calls itself (after
``--tool-delay`` seconds), then exercises a steer during a slow tool call and a status
request while the backend is WORKING, and closes.

    PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.cli.backend_probe \\
        --url ws://127.0.0.1:8790/v1/backend
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from typing import Any

import websockets

from examples.frontend_delegation_hermes.core.backend import protocol as proto

GET_ORDER = {
    "type": "function",
    "name": "get_order",
    "description": "Look up an order by its numeric id and return its status.",
    "parameters": {
        "type": "object",
        "properties": {"order_id": {"type": "string", "description": "The order id, digits only."}},
        "required": ["order_id"],
    },
}
ORDERS = {"1234": "cancelled", "5678": "shipped", "9012": "processing"}
INSTRUCTIONS = "You are a retail order-status agent. Look orders up with get_order; never guess a status."


class Probe:
    """A minimal voice-side stand-in."""

    def __init__(self, ws: Any, *, tool_delay: float, verbose: bool) -> None:
        """Bind the socket."""
        self.ws = ws
        self.tool_delay = tool_delay
        self.verbose = verbose
        self.t0 = time.monotonic()
        self.seq = 0
        self.answers: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.first_tool_call = asyncio.Event()
        self.answer_event = asyncio.Event()
        self.configured = asyncio.Event()
        self._tasks: set[asyncio.Task[None]] = set()

    def stamp(self) -> str:
        """Seconds since start."""
        return f"{time.monotonic() - self.t0:6.2f}s"

    async def send(self, type_: str, **fields: Any) -> None:
        """Send one voice→gateway message."""
        await self.ws.send(proto.encode(proto.VOICE_TO_GATEWAY, type_, **fields))
        if self.verbose:
            print(f"{self.stamp()} -> {type_} {json.dumps(fields)[:160]}")

    async def delegate(self, turn_id: int, text: str, request: str = "task") -> None:
        """Delegate the user's words."""
        self.seq += 1
        print(f"{self.stamp()} USER(turn {turn_id}, {request}): {text}")
        await self.send("delegate", turn_id=turn_id, request=request, run_input=[{"seq": self.seq, "text": text}])

    async def reader(self) -> None:
        """Consume gateway messages; run tools; collect answers."""
        async for raw in self.ws:
            msg = proto.decode(proto.GATEWAY_TO_VOICE, raw)
            self.events.append(msg)
            kind = msg["type"]
            detail = {k: v for k, v in msg.items() if k not in ("v", "type")}
            if kind == "tool.call":
                self.tool_calls.append(msg)
                self.first_tool_call.set()
                print(f"{self.stamp()} TOOL CALL {msg['name']}({msg['arguments']}) epoch={msg['epoch']}")
                task = asyncio.create_task(self._run_tool(msg))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
            elif kind == "answer":
                self.answers.append(msg)
                print(f"{self.stamp()} ANSWER[{msg['kind']}] turns={msg.get('turn_ids')}: {msg['text']}")
                self.answer_event.set()
            elif kind == "status":
                print(f"{self.stamp()} STATUS turn={msg['turn_id']}: {json.dumps(msg['summary'])}")
            elif kind == "action":
                print(f"{self.stamp()} ACTION turn={msg['turn_id']} kind={msg['kind']} run={msg.get('run_id')}")
            elif kind == "session.configured":
                print(f"{self.stamp()} CONFIGURED {detail}")
                self.configured.set()
            elif kind == "error":
                print(f"{self.stamp()} ERROR {detail}")
                if kind == "error" and not self.configured.is_set():
                    self.configured.set()
            elif self.verbose or kind in ("worker", "backend_run_done", "session.ready"):
                print(f"{self.stamp()} {kind} {json.dumps(detail)[:240]}")

    async def _run_tool(self, msg: dict[str, Any]) -> None:
        await asyncio.sleep(self.tool_delay)
        try:
            order_id = str(json.loads(msg["arguments"]).get("order_id", "")).strip()
        except ValueError:
            order_id = ""
        status = ORDERS.get(order_id)
        output = json.dumps(
            {"order_id": order_id, "status": status} if status else {"error": f"order {order_id} not found"}
        )
        await self.send("tool.result", call_id=msg["call_id"], epoch=msg["epoch"], output=output, local=False)

    async def wait_answer(self, timeout: float) -> None:
        """Wait for the next answer."""
        self.answer_event.clear()
        await asyncio.wait_for(self.answer_event.wait(), timeout=timeout)


async def run(url: str, tool_delay: float, slow_tool_delay: float, timeout: float, verbose: bool) -> int:
    """The probe scenario; returns a process exit code."""
    async with websockets.connect(url, max_size=64 * 1024 * 1024, ping_interval=None) as ws:
        probe = Probe(ws, tool_delay=tool_delay, verbose=verbose)
        reader = asyncio.create_task(probe.reader())
        session_id = f"probe{uuid.uuid4().hex[:6]}"
        await probe.send(
            "session.open",
            session_id=session_id,
            settings={"simulated_delay": {"seconds": 0, "where": "per_delegation"}, "steer_mode": "auto"},
        )
        await probe.send("session.configure", tools=[GET_ORDER], instructions=INSTRUCTIONS)
        await asyncio.wait_for(probe.configured.wait(), timeout=timeout)
        probe.seq += 1
        await probe.send(
            "history.append",
            entries=[
                {
                    "seq": probe.seq,
                    "origin": "frontend",
                    "kind": "frontend_speech",
                    "text": "Hi! How can I help you today?",
                    "outcome": "heard",
                }
            ],
        )

        print("\n== 1. new task (NO_SESSION -> start) ==")
        await probe.delegate(1, "What's the status of order 1234?")
        await probe.wait_answer(timeout)

        print("\n== 2. follow-up while IDLE (continue), steer during a slow tool, status while WORKING ==")
        probe.tool_delay = slow_tool_delay
        probe.first_tool_call.clear()
        answers_before = len(probe.answers)
        await probe.delegate(2, "Can you also check order 5678?")
        await asyncio.wait_for(probe.first_tool_call.wait(), timeout=timeout)
        await asyncio.sleep(0.5)
        await probe.delegate(3, "What's the update so far?", request="status")
        await asyncio.sleep(0.5)
        await probe.delegate(4, "And please check order 9012 as well.")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(0.25)
            last = probe.events[-1] if probe.events else {}
            done = [e for e in probe.events if e["type"] == "state" and e.get("state") == "IDLE"]
            if len(probe.answers) > answers_before and done and last.get("type") == "state":
                break
        print("\n== 3. close ==")
        await probe.send("session.close")
        await asyncio.sleep(0.5)
        reader.cancel()
        steer_actions = [e for e in probe.events if e["type"] == "action" and e["kind"] in ("steer", "redirect")]
        statuses = [e for e in probe.events if e["type"] == "status"]
        ok = bool(probe.answers) and bool(statuses) and bool(steer_actions)
        calls = [json.loads(c["arguments"]) for c in probe.tool_calls]
        print(
            f"\nSUMMARY answers={len(probe.answers)} tool_calls={calls} "
            f"steers={[e['kind'] for e in steer_actions]} statuses={len(statuses)} -> {'OK' if ok else 'INCOMPLETE'}"
        )
        return 0 if ok else 1


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Probe the fdh backend gateway without audio")
    parser.add_argument("--url", default="ws://127.0.0.1:8790/v1/backend")
    parser.add_argument("--tool-delay", type=float, default=0.3, help="seconds before answering a tool call")
    parser.add_argument("--slow-tool-delay", type=float, default=6.0, help="tool delay in the steer/status step")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    raise SystemExit(asyncio.run(run(args.url, args.tool_delay, args.slow_tool_delay, args.timeout, args.verbose)))


if __name__ == "__main__":
    main()
