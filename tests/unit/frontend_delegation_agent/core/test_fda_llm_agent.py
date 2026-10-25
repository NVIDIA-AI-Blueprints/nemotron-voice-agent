# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D100, D101, D102, D103, D107

"""``LLMAgent`` against a scripted OpenAI-compatible server (llm-backend-plan.md section 6).

The loop (tools, unknown names, iteration limit, reasoning-only replies, retries) and the controls
the controller uses mid-run: steer, redirect (prefill, mid-stream, crossing, during a tool, idle)
and hard interrupt.
"""

from __future__ import annotations

import os
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from _fda_llm_server import Reply, ScriptedOpenAI, tool

from examples.frontend_delegation_agent.core.backends.llm import HARNESS_KEYS
from examples.frontend_delegation_agent.core.backends.llm.host import LLMAgentHost
from examples.frontend_delegation_agent.core.backends.llm.settings import worker_prompts
from examples.frontend_delegation_agent.core.sidecar.templates import BackendTemplates

PROMPTS = Path(__file__).resolve().parents[4] / "src" / "examples" / "frontend_delegation_agent" / "config" / "prompts"
TOOLS = [
    {
        "name": "get_order",
        "description": "Look up an order.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    },
    {"name": "cancel_order", "description": "Cancel an order.", "parameters": {"type": "object", "properties": {}}},
]
TEMPLATES = BackendTemplates(PROMPTS / "customer_service" / "backend" / "llm.yaml", required_keys=HARNESS_KEYS)


class Bridge:
    """The worker's ``ToolFutures.call``: returns scripted results, optionally slowly."""

    def __init__(self, results: dict[str, str] | None = None, delay_s: float = 0.0) -> None:
        self.results = results or {}
        self.delay_s = delay_s
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.entered = threading.Event()
        self.host: LLMAgentHost | None = None

    def call(self, name: str, args: dict[str, Any], task_id: str) -> str:
        self.calls.append((name, args))
        self.entered.set()
        deadline = time.monotonic() + self.delay_s
        while time.monotonic() < deadline:
            if self.host is not None and self.host.is_interrupted():
                return "Error: interrupted"
            time.sleep(0.01)
        return self.results.get(name, '{"status": "shipped"}')


class AgentTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ScriptedOpenAI().start()
        self.addCleanup(self.server.stop)
        patcher = mock.patch.dict(os.environ, {"FDA_FAKE_KEY": "sk-fake"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def agent(self, bridge: Bridge | None = None, **settings: Any) -> Any:
        bridge = bridge or Bridge()
        host = LLMAgentHost(bridge, lambda *_: None, session_id="s1")
        bridge.host = host
        host.configure(
            tools=TOOLS,
            instructions="POLICY",
            settings={
                "model": "fake-model",
                "base_url": self.server.base_url,
                "api_key_env": "FDA_FAKE_KEY",
                "max_iterations": settings.pop("max_iterations", 30),
                "request_overrides": {},
                "reasoning_effort": "medium",
                "request_timeout_s": 5.0,
                "max_retries": 3,
                **worker_prompts(TEMPLATES),
                **settings,
            },
        )
        self.bridge = bridge
        return host.agent()

    def run_async(self, agent: Any, text: str, history: list | None = None) -> tuple[threading.Thread, dict]:
        out: dict[str, Any] = {}
        thread = threading.Thread(target=lambda: out.update(agent.run_conversation(text, history or [], "s1:1")))
        thread.start()
        return thread, out

    def roles(self, messages: list[dict]) -> list[str]:
        return [m["role"] for m in messages]


class LoopTest(AgentTestCase):
    def test_tool_round_then_answer(self) -> None:
        self.server.queue(
            Reply(content="Checking.", tool_calls=[tool("get_order", order_id="1")]), Reply(content="It shipped.")
        )
        result = self.agent().run_conversation("where is order 1", [], "s1:1")
        self.assertEqual(result["final_response"], "It shipped.")
        self.assertEqual(self.bridge.calls, [("get_order", {"order_id": "1"})])
        messages = result["messages"]
        self.assertEqual(self.roles(messages), ["user", "assistant", "tool", "assistant"])
        self.assertEqual(messages[1]["content"], "Checking.")  # text next to a tool call is kept
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["arguments"], '{"order_id":"1"}')
        first = self.server.requests[0]
        self.assertEqual(first["messages"][0]["role"], "system")
        self.assertTrue(first["messages"][0]["content"].endswith("POLICY"))
        self.assertEqual(first["reasoning_effort"], "medium")
        self.assertNotIn("system", self.roles(messages))
        self.assertEqual(self.server.requests[1]["messages"][0], first["messages"][0])  # the same system bytes

    def test_unknown_tool_and_repair(self) -> None:
        self.server.queue(
            Reply(tool_calls=[tool("lookup_everything")]),
            Reply(tool_calls=[tool("GetOrder", order_id="7")]),
            Reply(content="Done."),
        )
        result = self.agent().run_conversation("order 7", [], "s1:1")
        tool_rows = [m for m in result["messages"] if m["role"] == "tool"]
        self.assertEqual(
            tool_rows[0]["content"], "Tool 'lookup_everything' does not exist. Available tools: cancel_order, get_order"
        )
        self.assertEqual(self.bridge.calls, [("get_order", {"order_id": "7"})])  # repaired, and replayed repaired
        self.assertEqual(result["messages"][3]["tool_calls"][0]["function"]["name"], "get_order")

    def test_three_rounds_of_unknown_names_end_the_run(self) -> None:
        self.server.queue(*(Reply(tool_calls=[tool("check_wifi_status")]) for _ in range(3)))
        result = self.agent().run_conversation("wifi", [], "s1:1")
        self.assertEqual(result["final_response"], "Model generated invalid tool call: check_wifi_status")
        self.assertTrue(result["partial"])
        self.assertEqual(result["messages"][-1], {"role": "assistant", "content": result["final_response"]})

    def test_bad_arguments_get_an_error_result(self) -> None:
        self.server.queue(Reply(tool_calls=[("get_order", "{oops")]), Reply(content="Sorry."))
        result = self.agent().run_conversation("x", [], "s1:1")
        row = next(m for m in result["messages"] if m["role"] == "tool")
        self.assertTrue(row["content"].startswith("Error: Invalid JSON arguments. "))
        self.assertEqual(self.bridge.calls, [])

    def test_iteration_limit_asks_for_a_summary_without_tools(self) -> None:
        self.server.queue(
            *(Reply(tool_calls=[tool("get_order", order_id=str(i))]) for i in range(2)), Reply(content="Summary.")
        )
        result = self.agent(max_iterations=2).run_conversation("x", [], "s1:1")
        self.assertEqual(result["final_response"], "Summary.")
        last = self.server.requests[-1]
        self.assertNotIn("tools", last)
        self.assertTrue(last["messages"][-1]["content"].startswith("You've reached the maximum number"))

    def test_reasoning_only_reply_is_the_answer_or_a_stall(self) -> None:
        self.server.queue(Reply(reasoning="The order shipped on Monday."))
        self.assertEqual(
            self.agent().run_conversation("x", [], "s1:1")["final_response"], "The order shipped on Monday."
        )
        self.server.queue(
            Reply(reasoning="The user wants the status. Let me look up the order."),
            Reply(tool_calls=[tool("get_order", order_id="2")]),
            Reply(content="It shipped."),
        )
        result = self.agent().run_conversation("x", [], "s1:1")
        self.assertEqual(result["final_response"], "It shipped.")
        nudge = result["messages"][2]
        self.assertEqual(nudge["content"], TEMPLATES.render("harness_continue_now"))

    def test_continue_nudge_is_capped(self) -> None:
        self.server.queue(*(Reply(content="I'll now check the order.") for _ in range(3)))
        result = self.agent().run_conversation("x", [], "s1:1")
        self.assertEqual(result["final_response"], "I'll now check the order.")
        self.assertEqual(len(self.server.requests), 3)

    def test_empty_reply_is_retried_once(self) -> None:
        self.server.queue(Reply(content=""), Reply(content="Here."))
        self.assertEqual(self.agent().run_conversation("x", [], "s1:1")["final_response"], "Here.")

    def test_retries(self) -> None:
        self.server.queue(Reply(status=500), Reply(status=429, retry_after="0"), Reply(content="ok"))
        with mock.patch("examples.frontend_delegation_agent.core.backends.llm.client.BACKOFF_BASE_S", 0.01):
            self.assertEqual(self.agent().run_conversation("x", [], "s1:1")["final_response"], "ok")
        self.server.queue(Reply(content="partial text", cut=True), Reply(content="whole"))
        with mock.patch("examples.frontend_delegation_agent.core.backends.llm.client.BACKOFF_BASE_S", 0.01):
            self.assertEqual(self.agent().run_conversation("x", [], "s1:1")["final_response"], "whole")
        self.server.queue(Reply(status=400))
        result = self.agent().run_conversation("x", [], "s1:1")
        self.assertTrue(result["failed"])
        self.assertIn("BadRequestError", result["error"])

    def test_usage_is_cumulative(self) -> None:
        agent = self.agent()
        agent.run_conversation("x", [], "s1:1")
        agent.run_conversation("y", [], "s1:2")
        self.assertEqual(agent.usage_snapshot(), {"input_tokens": 22, "output_tokens": 14, "reasoning_tokens": 6})


class ControlTest(AgentTestCase):
    def wait(self, event: threading.Event, what: str) -> None:
        self.assertTrue(event.wait(5), f"timed out waiting for {what}")

    def test_steer_during_a_tool_is_delivered_after_its_result(self) -> None:
        self.server.queue(Reply(tool_calls=[tool("get_order", order_id="1")]), Reply(content="Cancelled instead."))
        agent = self.agent(Bridge(delay_s=0.3))
        thread, out = self.run_async(agent, "order 1")
        self.wait(self.bridge.entered, "the tool")
        self.assertTrue(agent.steer('The user just added: "cancel it"'))
        thread.join(5)
        self.assertEqual(self.roles(out["messages"]), ["user", "assistant", "tool", "user", "assistant"])
        steer = out["messages"][3]["content"]
        self.assertTrue(steer.startswith("[OUT-OF-BAND USER MESSAGE — a direct message from the user"))
        self.assertIn('The user just added: "cancel it"', steer)
        self.assertNotIn("pending_steer", out)

    def test_steer_goes_after_an_older_turns_tool_row(self) -> None:
        history = [
            {"role": "user", "content": "order 1"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "a", "type": "function", "function": {"name": "get_order", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "a", "content": "shipped"},
            {"role": "assistant", "content": "It shipped."},
        ]
        agent = self.agent()
        agent.steer("also the other one")
        result = agent.run_conversation("and order 2?", history, "s1:2")
        self.assertEqual(result["messages"][3]["role"], "user")
        self.assertIn("also the other one", result["messages"][3]["content"])

    def test_undelivered_steer_is_returned(self) -> None:
        agent = self.agent()
        agent.steer("one more thing")
        self.assertEqual(agent.run_conversation("hi", [], "s1:1")["pending_steer"], "one more thing")

    def test_redirect_mid_stream_retries_with_the_correction(self) -> None:
        self.server.queue(Reply(content="x" * 400, chunk_s=0.05), Reply(content="Order 2 shipped."))
        agent = self.agent()
        thread, out = self.run_async(agent, "order 1")
        self.wait(self.server.started, "the request")
        time.sleep(0.2)
        self.assertTrue(agent.model_request_active())
        self.assertTrue(agent.redirect("I meant order 2"))
        thread.join(5)
        self.assertEqual(out["final_response"], "Order 2 shipped.")
        self.assertEqual(self.roles(out["messages"]), ["user", "assistant", "user", "assistant"])
        self.assertEqual(out["messages"][1]["content"], "")  # no partial text
        self.assertEqual(
            out["messages"][2]["content"],
            "[Context from the interrupted assistant response]\n"
            "[This response was interrupted by a user correction.]\n\nI meant order 2",
        )
        sent = self.server.requests[1]["messages"]
        self.assertEqual(sent[2], {"role": "assistant", "content": "[response interrupted]"})
        self.assertEqual(agent.get_activity_summary()["api_call_count"], 1)  # the aborted call does not count

    def test_redirect_during_prefill_aborts_before_the_first_token(self) -> None:
        self.server.queue(Reply(content="slow", hold_s=3.0), Reply(content="fast"))
        agent = self.agent()
        thread, out = self.run_async(agent, "q")
        self.wait(self.server.started, "the request")
        time.sleep(0.1)
        started = time.monotonic()
        self.assertTrue(agent.redirect("never mind"))
        thread.join(5)
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertEqual(out["final_response"], "fast")

    def test_a_reply_that_crosses_a_redirect_is_discarded(self) -> None:
        self.server.queue(Reply(tool_calls=[tool("cancel_order")]), Reply(content="Kept it."))
        agent = self.agent()
        original = agent._client.complete  # noqa: SLF001

        def complete_then_redirect(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            if len(self.server.requests) == 1:
                with agent._lock:  # noqa: SLF001 - the redirect lands after the stream ended
                    agent._redirects.append("do not cancel")  # noqa: SLF001
            return result

        agent._client.complete = complete_then_redirect  # noqa: SLF001
        result = agent.run_conversation("cancel order", [], "s1:1")
        self.assertEqual(self.bridge.calls, [])  # the crossed tool call never ran
        self.assertEqual(result["final_response"], "Kept it.")
        self.assertIn("do not cancel", result["messages"][2]["content"])

    def test_redirect_during_a_tool_is_a_steer_and_idle_is_refused(self) -> None:
        agent = self.agent(Bridge(delay_s=0.3))
        self.assertFalse(agent.redirect("idle"))
        self.server.queue(Reply(tool_calls=[tool("get_order", order_id="1")]), Reply(content="ok"))
        thread, out = self.run_async(agent, "q")
        self.wait(self.bridge.entered, "the tool")
        self.assertTrue(agent.redirect("actually order 3"))
        thread.join(5)
        self.assertIn("actually order 3", out["messages"][3]["content"])
        self.assertTrue(out["messages"][3]["content"].startswith("[OUT-OF-BAND USER MESSAGE"))

    def test_hard_interrupt_during_a_stream_and_during_a_tool(self) -> None:
        self.server.queue(Reply(content="y" * 400, chunk_s=0.05))
        agent = self.agent()
        thread, out = self.run_async(agent, "q")
        self.wait(self.server.started, "the request")
        time.sleep(0.2)
        agent.hard_interrupt("fda: interrupted")
        thread.join(5)
        self.assertTrue(out["interrupted"])
        self.assertEqual((out["final_response"], out["turn_exit_reason"]), (None, "interrupted_by_user"))
        self.server.queue(Reply(tool_calls=[tool("get_order", order_id="1")]))
        agent = self.agent(Bridge(delay_s=5.0))
        agent.clear_interrupt()
        thread, out = self.run_async(agent, "q")
        self.wait(self.bridge.entered, "the tool")
        agent.hard_interrupt("stop")
        thread.join(5)
        self.assertTrue(out["interrupted"])
        self.assertEqual(self.roles(out["messages"])[-2:], ["tool", "assistant"])
        self.assertEqual(out["messages"][-1]["content"], "Operation interrupted.")


if __name__ == "__main__":
    unittest.main()
