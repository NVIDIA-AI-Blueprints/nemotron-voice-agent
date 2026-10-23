# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D100, D101, D102, D103, D107

"""Process-per-session isolation with REAL worker subprocesses (``agent_kind: fake``, this interpreter).

Real Unix sockets, real process groups, real signals — no Hermes needed. Covers the
re-review items: schema isolation for overlapping tool names, results never crossing
sessions, a hung or crashed worker not affecting another session, respawn from the
settled history, capacity only after ``max_sessions`` live sessions, disconnect cleanup
with SIGTERM → SIGKILL escalation, and the start timeout.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

from _fdh_backend_fakes import TOOL
from fastapi.testclient import TestClient

from examples.frontend_delegation_hermes.core.backend import protocol as proto
from examples.frontend_delegation_hermes.core.sidecar.events import GatewayEventLog
from examples.frontend_delegation_hermes.core.sidecar.gateway_config import (
    FAKE_GATEWAY_CONFIG,
    GatewayConfig,
    load_gateway_config,
)
from examples.frontend_delegation_hermes.core.sidecar.gateway_server import build_gateway_app
from examples.frontend_delegation_hermes.core.sidecar.inprocess_link import InProcessBackendLink
from examples.frontend_delegation_hermes.core.sidecar.session_runtime import SessionRuntime


def process_config(root: Path, **overrides: Any) -> GatewayConfig:
    workers = {
        "mode": "process_per_session",
        "agent_kind": "fake",
        "python": "self",
        "start_timeout_s": 5.0,
        "configure_timeout_s": 5.0,
        "stop_timeout_s": 2.0,
        "kill_grace_s": 0.5,
        "home_root": str(root / "homes"),
        "socket_dir": str(root / "sock"),
        "log_dir": str(root / "logs"),
    }
    workers.update(overrides.pop("workers", {}))
    hermes = {"unwind_timeout_s": 0.5, "run_budget_seconds": 0.5, "run_hard_deadline_s": 1.5, "steer_timeout_s": 1.0}
    hermes.update(overrides.pop("hermes", {}))
    return load_gateway_config(
        None, overrides={"workers": workers, "hermes": hermes, "gateway": {"log": ""}, **overrides}
    )


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0); check its state.
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


class Session:
    """One gateway session driven like the voice server would, answering tool calls itself."""

    def __init__(self, config: GatewayConfig, name: str, *, tool_output: str | None = None) -> None:
        self.name = name
        self.messages: list[dict[str, Any]] = []
        self.log = GatewayEventLog(None)
        self.tool_output = tool_output or f"{name}-result"
        self.runtime = SessionRuntime(config, self._on, event_log=self.log)
        self.seq = 0

    def _on(self, msg: dict[str, Any]) -> None:
        self.messages.append(msg)
        if msg["type"] == "tool.call":
            asyncio.get_running_loop().create_task(
                self.runtime.handle(
                    proto.message(
                        proto.VOICE_TO_GATEWAY,
                        "tool.result",
                        call_id=msg["call_id"],
                        epoch=msg["epoch"],
                        output=self.tool_output,
                        local=False,
                    )
                )
            )

    async def open(self, tools: list[dict[str, Any]] | None = None) -> None:
        await self.runtime.handle(
            proto.message(proto.VOICE_TO_GATEWAY, "session.open", session_id=self.name, settings={})
        )
        await self.runtime.handle(
            proto.message(proto.VOICE_TO_GATEWAY, "session.configure", tools=tools or [TOOL], instructions="POLICY")
        )

    async def delegate(self, turn_id: int, text: str) -> None:
        self.seq += 1
        await self.runtime.handle(
            proto.message(
                proto.VOICE_TO_GATEWAY,
                "delegate",
                turn_id=turn_id,
                request="task",
                run_input=[{"seq": self.seq, "text": text}],
            )
        )

    async def wait(self, kind: str, count: int = 1, timeout: float = 8.0, **match: Any) -> list[dict[str, Any]]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            found = [m for m in self.messages if m["type"] == kind and all(m.get(k) == v for k, v in match.items())]
            if len(found) >= count:
                return found
            await asyncio.sleep(0.02)
        raise AssertionError(f"{self.name}: no {count}x {kind} {match}; got {[m['type'] for m in self.messages]}")

    def pid(self) -> int:
        worker = self.runtime.controller._worker  # noqa: SLF001 - test introspection
        assert worker is not None and worker.pid is not None
        return worker.pid


class GatewayProcessTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="fdh-gw-"))
        self.sessions: list[Session] = []

    async def asyncTearDown(self) -> None:
        for session in self.sessions:
            await session.runtime.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def session(self, name: str, config: GatewayConfig | None = None, **kwargs: Any) -> Session:
        session = Session(config or process_config(self.root), name, **kwargs)
        self.sessions.append(session)
        return session

    async def test_overlapping_tool_names_keep_their_own_schemas_and_results(self) -> None:
        a, b = self.session("sa"), self.session("sb")
        tool_b = {**TOOL, "parameters": {"type": "object", "properties": {"sku": {"type": "string"}}}}
        await asyncio.gather(a.open([TOOL]), b.open([tool_b]))
        self.assertNotEqual(a.pid(), b.pid())
        await asyncio.gather(
            a.delegate(1, '[tool:get_order {"order_id": "1"}] [schemas]'),
            b.delegate(1, '[tool:get_order {"sku": "X"}] [schemas]'),
        )
        (answer_a,), (answer_b,) = await asyncio.gather(a.wait("answer"), b.wait("answer"))
        self.assertIn("sa-result", answer_a["text"])
        self.assertIn("get_order=['order_id']", answer_a["text"])
        self.assertNotIn("sb-result", answer_a["text"])
        self.assertNotIn("sku", answer_a["text"])
        self.assertIn("sb-result", answer_b["text"])
        self.assertIn("get_order=['sku']", answer_b["text"])
        self.assertNotIn("sa-result", answer_b["text"])

    async def test_hung_worker_is_killed_without_affecting_another_session(self) -> None:
        a, b = self.session("hung"), self.session("fine")
        await asyncio.gather(a.open(), b.open())
        hung_pid = a.pid()
        await a.delegate(1, "[hang]")
        await b.delegate(1, "[slow:0.3] hello")
        (answer_b,) = await b.wait("answer")
        self.assertEqual(answer_b["kind"], "hermes")
        (done,) = await a.wait("backend_run_done", timeout=8)
        self.assertEqual((done["status"], done["history"]), ("deadline", "rolled_back"))
        self.assertEqual((await a.wait("answer"))[0]["kind"], "apology")
        await a.wait("worker", event="respawned")
        self.assertFalse(alive(hung_pid))
        self.assertNotEqual(a.pid(), hung_pid)
        self.assertTrue(alive(b.pid()))

    async def test_crash_rolls_back_and_the_respawn_continues_from_settled_history(self) -> None:
        a = self.session("crash")
        await a.open()
        await a.delegate(1, "hello")
        await a.wait("answer")
        crashed = a.pid()
        await a.delegate(2, "[crash]")
        done = await a.wait("backend_run_done", count=2)
        self.assertEqual((done[1]["status"], done[1]["history"]), ("worker_died", "rolled_back"))
        await a.wait("worker", event="respawned")
        self.assertNotEqual(a.pid(), crashed)
        await a.delegate(3, "[histlen]")
        answers = await a.wait("answer", count=3)
        self.assertIn("history=2", answers[2]["text"])  # the first turn's committed history, not the crashed one

    async def test_disconnect_escalates_to_sigkill_and_cleans_up(self) -> None:
        a = self.session("stuck")
        await a.open()
        pid = a.pid()
        worker = a.runtime.controller._worker  # noqa: SLF001
        home, sock = worker.home, worker._socket  # noqa: SLF001
        self.assertTrue(home.is_dir())
        os.kill(pid, signal.SIGSTOP)  # cannot process `close` or SIGTERM
        started = time.monotonic()
        await a.runtime.close()
        stopped = [r for r in a.log.records if r["event"] == "worker_stopped"]
        self.assertEqual(stopped[-1]["how"], "sigkill")
        self.assertGreaterEqual(time.monotonic() - started, 2.0)
        self.assertFalse(alive(pid))
        self.assertFalse(home.exists())
        self.assertFalse(sock.exists())

    async def test_graceful_close_and_home_removed(self) -> None:
        a = self.session("graceful")
        await a.open()
        pid, home = a.pid(), a.runtime.controller._worker.home  # noqa: SLF001
        await a.runtime.close()
        stopped = [r for r in a.log.records if r["event"] == "worker_stopped"]
        self.assertEqual(stopped[-1]["how"], "closed")
        self.assertFalse(alive(pid))
        self.assertFalse(home.exists())

    async def test_start_timeout_is_reported_and_the_process_killed(self) -> None:
        script = self.root / "never_ready.sh"
        script.write_text("#!/bin/sh\nexec sleep 30\n")
        script.chmod(0o755)
        config = process_config(self.root, workers={"python": str(script), "start_timeout_s": 0.7})
        a = self.session("slow", config)
        await a.runtime.handle(proto.message(proto.VOICE_TO_GATEWAY, "session.open", session_id="slow", settings={}))
        (error,) = await a.wait("error")
        self.assertEqual((error["code"], error["fatal"]), ("worker_start_timeout", True))
        self.assertEqual(list((self.root / "homes").glob("slow-*")), [])

    async def test_in_process_link_with_the_fake_gateway_file(self) -> None:
        link = InProcessBackendLink(load_gateway_config(FAKE_GATEWAY_CONFIG))
        self.assertEqual(link._config.workers.mode, "in_process_fake")  # noqa: SLF001
        got: list[dict[str, Any]] = []

        def on(msg: dict[str, Any]) -> None:
            got.append(msg)
            if msg["type"] == "tool.call":
                link.send(
                    {
                        "type": "tool.result",
                        "call_id": msg["call_id"],
                        "epoch": msg["epoch"],
                        "output": '{"status": "shipped"}',
                        "local": False,
                    }
                )

        await link.open(on)
        link.send({"type": "session.open", "session_id": "stub", "settings": {}})
        link.send({"type": "session.configure", "tools": [TOOL], "instructions": ""})
        link.send(
            {
                "type": "delegate",
                "turn_id": 1,
                "request": "task",
                "run_input": [{"seq": 1, "text": "status of order 1234"}],
            }
        )
        deadline = time.monotonic() + 5
        while not any(m["type"] == "answer" for m in got) and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        await link.close()
        answer = next(m for m in got if m["type"] == "answer")
        self.assertIn("shipped", answer["text"])
        self.assertEqual(
            json.loads(next(m for m in got if m["type"] == "tool.call")["arguments"]), {"order_id": "1234"}
        )


class GatewayCapacityTest(unittest.TestCase):
    def test_capacity_refusal_only_after_max_live_sessions(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="fdh-cap-"))
        try:
            config = process_config(root, gateway={"max_sessions": 1, "log": ""})
            with TestClient(build_gateway_app(config)) as client:
                first = client.websocket_connect("/v1/backend")
                first.__enter__()
                first.send_text(proto.encode(proto.VOICE_TO_GATEWAY, "session.open", session_id="c1", settings={}))
                kinds = []
                while "session.ready" not in kinds:
                    kinds.append(json.loads(first.receive_text())["type"])
                self.assertEqual(client.get("/health").json()["sessions"], 1)
                with client.websocket_connect("/v1/backend") as second:
                    refused = json.loads(second.receive_text())
                self.assertEqual((refused["code"], refused["fatal"]), ("capacity", True))
                first.send_text(proto.encode(proto.VOICE_TO_GATEWAY, "session.close"))
                first.__exit__(None, None, None)
                deadline = time.monotonic() + 5
                while client.get("/health").json()["sessions"] and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertEqual(client.get("/health").json()["sessions"], 0)
                with client.websocket_connect("/v1/backend") as third:
                    third.send_text(proto.encode(proto.VOICE_TO_GATEWAY, "session.open", session_id="c3", settings={}))
                    kinds = []
                    while "session.ready" not in kinds and "error" not in kinds:
                        kinds.append(json.loads(third.receive_text())["type"])
                    self.assertIn("session.ready", kinds)
                    third.send_text(proto.encode(proto.VOICE_TO_GATEWAY, "session.close"))
                health = client.get("/health").json()
                self.assertEqual(health["refused"], 1)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
