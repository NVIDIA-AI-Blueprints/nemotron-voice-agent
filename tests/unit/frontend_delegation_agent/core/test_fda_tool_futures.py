# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D100, D101, D102, D103, D107

"""Keyed tool futures and the worker core (plan sections 8.1 and 9.2)."""

from __future__ import annotations

import asyncio
import threading
import time
import unittest
from typing import Any

from examples.frontend_delegation_agent.core.backends.emulator.host import EmulatorAgentHost
from examples.frontend_delegation_agent.core.worker import tool_futures as tf
from examples.frontend_delegation_agent.core.worker.worker_core import WorkerCore


class Collector:
    """Thread-safe emit target."""

    def __init__(self, *, closed: bool = False) -> None:
        self.messages: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.closed = closed
        self.event = threading.Event()

    def __call__(self, message: dict[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("link closed")
        with self.lock:
            self.messages.append(message)
        self.event.set()

    def wait_call(self, timeout: float = 2.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                calls = [m for m in self.messages if m["type"] == "tool.call"]
            if calls:
                return calls[-1]
            time.sleep(0.005)
        raise AssertionError("no tool.call")


def call_in_thread(futures: tf.ToolFutures, task_id: str = "s:1") -> tuple[threading.Thread, dict[str, str]]:
    box: dict[str, str] = {}
    thread = threading.Thread(target=lambda: box.update(out=futures.call("get_order", {"order_id": "1"}, task_id)))
    thread.start()
    return thread, box


class ToolFuturesTest(unittest.TestCase):
    def make(self, **kwargs: Any) -> tuple[tf.ToolFutures, Collector]:
        emit = kwargs.pop("emit", None) or Collector()
        futures = tf.ToolFutures(session_id="s", emit=emit, poll_s=0.01, **kwargs)
        futures.current_epoch = 1
        return futures, emit

    def test_waits_for_the_result_not_the_send(self) -> None:
        futures, emit = self.make()
        thread, box = call_in_thread(futures)
        call = emit.wait_call()
        self.assertEqual((call["epoch"], call["name"]), (1, "get_order"))
        time.sleep(0.05)
        self.assertTrue(thread.is_alive())  # still waiting after the send
        self.assertEqual(futures.resolve(call["call_id"], "RESULT"), "ok")
        thread.join(1)
        self.assertEqual(box["out"], "RESULT")

    def test_stale_epoch_is_not_executed(self) -> None:
        futures, emit = self.make()
        self.assertEqual(futures.call("get_order", {}, "s:0"), tf.STALE_ERROR)
        self.assertEqual(futures.call("get_order", {}, None), tf.STALE_ERROR)
        self.assertEqual(emit.messages, [])

    def test_timeout_resolves_and_cancels(self) -> None:
        futures, emit = self.make(timeout_s=0.05)
        thread, box = call_in_thread(futures)
        thread.join(1)
        self.assertEqual(box["out"], tf.TIMEOUT_ERROR)
        self.assertEqual([m["type"] for m in emit.messages], ["tool.call", "tool.cancel"])
        call_id = emit.messages[0]["call_id"]
        self.assertEqual(futures.resolve(call_id, "late"), "late")

    def test_interrupt_and_close_resolve_everything(self) -> None:
        futures, emit = self.make()
        threads = [call_in_thread(futures) for _ in range(3)]
        deadline = time.monotonic() + 2
        while len(futures.outstanding()) < 3 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(futures.interrupt_all(), 3)
        for thread, box in threads:
            thread.join(1)
            self.assertEqual(box["out"], tf.INTERRUPTED_ERROR)
        futures.clear_interrupt()
        thread, box = call_in_thread(futures)
        emit.wait_call()
        futures.close_all()
        thread.join(1)
        self.assertEqual(box["out"], tf.CLOSED_ERROR)

    def test_duplicate_unknown_and_stale_results(self) -> None:
        futures, emit = self.make()
        thread, _box = call_in_thread(futures)
        call = emit.wait_call()
        self.assertEqual(futures.resolve(call["call_id"], "x", epoch=7), "stale_epoch")
        self.assertEqual(futures.resolve(call["call_id"], "x", epoch=1), "ok")
        thread.join(1)
        self.assertEqual(futures.resolve(call["call_id"], "again"), "duplicate")
        self.assertEqual(futures.resolve("nope", "x"), "unknown")

    def test_closed_link_resolves_at_once(self) -> None:
        futures, _emit = self.make(emit=Collector(closed=True))
        started = time.monotonic()
        self.assertEqual(futures.call("get_order", {}, "s:1"), tf.CLOSED_ERROR)
        self.assertLess(time.monotonic() - started, 0.5)

    def test_epoch_of(self) -> None:
        self.assertEqual(tf.epoch_of("sess:with:colons:12"), 12)
        self.assertIsNone(tf.epoch_of("nocolon"))
        self.assertIsNone(tf.epoch_of("s:x"))


class WorkerCoreTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.messages: list[dict[str, Any]] = []
        self.core = WorkerCore(
            session_id="s",
            host_factory=lambda futures, on_activity: EmulatorAgentHost(futures, on_activity, allow_crash=False),
            emit=lambda m: self.loop.call_soon_threadsafe(self.messages.append, m),
        )
        await self.core.handle(
            {
                "type": "configure",
                "tools": [{"name": "get_order", "parameters": {"properties": {"order_id": {}}}}],
                "instructions": "x",
                "backend": {"agent_construct": "eager", "unwind_timeout_s": 1.0},
            }
        )

    async def asyncTearDown(self) -> None:
        await self.core.close()

    async def wait_for(self, kind: str, timeout: float = 3.0) -> dict[str, Any]:
        deadline = self.loop.time() + timeout
        while self.loop.time() < deadline:
            found = [m for m in self.messages if m["type"] == kind]
            if found:
                return found[-1]
            await asyncio.sleep(0.01)
        raise AssertionError(f"no {kind}: {[m['type'] for m in self.messages]}")

    def run_msg(self, epoch: int, text: str) -> dict[str, Any]:
        return {
            "type": "run",
            "epoch": epoch,
            "task_id": f"s:{epoch}",
            "user_message": text,
            "conversation_history": [],
        }

    async def test_steer_during_tool_and_redirect_during_model_request(self) -> None:
        await self.wait_for("configured")
        await self.core.handle(self.run_msg(1, '[tool:get_order {"order_id": "1"}]'))
        call = await self.wait_for("tool.call")
        await self.core.handle({"type": "steer", "epoch": 1, "text": "also order 2", "mode": "auto"})
        result = await self.wait_for("steer_result")
        self.assertEqual((result["accepted"], result["via"]), (True, "steer"))
        await self.core.handle({"type": "tool.result", "call_id": call["call_id"], "output": "ok"})
        outcome = await self.wait_for("run_outcome")
        self.assertEqual(outcome["status"], "ok")
        self.assertIn("steered: also order 2", outcome["final_response"])
        self.messages.clear()
        await self.core.handle(self.run_msg(2, "[slow:2]"))
        await asyncio.sleep(0.1)
        await self.core.handle({"type": "steer", "epoch": 2, "text": "do this instead", "mode": "auto"})
        result = await self.wait_for("steer_result")
        self.assertEqual(result["via"], "redirect")
        outcome = await self.wait_for("run_outcome")
        self.assertIn("redirected: do this instead", outcome["final_response"])

    async def test_steer_for_a_finished_or_other_epoch_is_refused(self) -> None:
        await self.core.handle({"type": "steer", "epoch": 5, "text": "x"})
        result = await self.wait_for("steer_result")
        self.assertFalse(result["accepted"])

    async def test_usage_is_a_per_run_delta_and_errors_are_outcomes(self) -> None:
        await self.core.handle(self.run_msg(1, "hello"))
        first = await self.wait_for("run_outcome")
        self.messages.clear()
        await self.core.handle(self.run_msg(2, "hello again"))
        second = await self.wait_for("run_outcome")
        self.assertGreater(first["usage_delta"]["input_tokens"], 0)
        self.assertLess(second["usage_delta"]["input_tokens"], 100)  # not cumulative
        self.messages.clear()
        await self.core.handle(self.run_msg(3, "[raise]"))
        outcome = await self.wait_for("run_outcome")
        self.assertEqual(outcome["status"], "error")
        self.assertIn("RuntimeError", outcome["error"])

    async def test_interrupt_ends_a_run_and_close_reports_unwound(self) -> None:
        await self.core.handle(self.run_msg(1, "[slow:5]"))
        await asyncio.sleep(0.05)
        await self.core.handle({"type": "interrupt", "hard": True})
        outcome = await self.wait_for("run_outcome")
        self.assertEqual(outcome["status"], "interrupted")
        await self.core.close()
        closed = await self.wait_for("closed")
        self.assertTrue(closed["unwound"])


if __name__ == "__main__":
    unittest.main()
