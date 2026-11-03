# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Tests for the round cap and the request log: the cap ends with an answer, not a failure."""

import unittest
from unittest.mock import patch

from delegation_core.backend.llm import LlmSettings
from delegation_core.backend.providers import ChatCompletionsBackend
from tests.unit.contracts.backend import call_round, text_round
from tests.unit.delegation_core.backend.llm.doubles import (
    ENDPOINT,
    LIMITS,
    FakeChat,
    IMTask,
    RecordingTools,
    ScriptedProvider,
    adapter,
    chunk,
    drive,
    items_of,
    tool_fragment,
)

CAPPED = LlmSettings(max_tool_rounds=2)


class RoundCapTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_capped_round_answers_with_no_tool_offered_and_the_configured_note(self):
        provider = ScriptedProvider(
            call_round("lookup", {}, "call_1"), call_round("lookup", {}, "call_2"), text_round("Here is what I have.")
        )
        task = IMTask()
        events = await drive(adapter(provider, tools=[RecordingTools()], settings=CAPPED).start(task), task)
        self.assertEqual(items_of(events, "message")[-1]["phase"], "final_answer")
        self.assertEqual([payload["tool_choice"] for payload in provider.payloads], ["auto", "auto", "none"])
        offered = [[tool["name"] for tool in payload["tools"]] for payload in provider.payloads]
        self.assertEqual(offered, [["get_menu", "lookup"], ["get_menu", "lookup"], []])  # no tool to call or write out
        note = task.texts.round_cap_note
        self.assertEqual(provider.payloads[2]["instructions"], f"COMPOSED-INSTRUCTIONS\n\n{note}")
        outputs = [
            item["call_id"] for item in provider.payloads[2]["input"] if item.get("type") == "function_call_output"
        ]
        self.assertEqual(outputs, ["call_1", "call_2"])  # the last tool output is kept, not thrown away

    async def test_a_chat_endpoint_gets_the_capped_round_with_no_tools_and_the_earlier_calls_in_its_history(self):
        chat = FakeChat(
            [chunk(tool_calls=[tool_fragment(0, "p1", "lookup", "{}")]), chunk(finish="tool_calls")],
            [chunk(tool_calls=[tool_fragment(0, "p2", "lookup", "{}")]), chunk(finish="tool_calls")],
            [chunk("Here is what I have."), chunk(finish="stop")],
        )
        task = IMTask()
        backend = adapter(ChatCompletionsBackend(ENDPOINT, LIMITS, chat), tools=[RecordingTools()], settings=CAPPED)
        events = await drive(backend.start(task), task)
        self.assertEqual(items_of(events, "message")[-1]["phase"], "final_answer")
        capped = chat.requests[2]
        self.assertEqual(("tools" in capped, "tool_choice" in capped), (False, False))
        self.assertTrue(capped["messages"][0]["content"].endswith(task.texts.round_cap_note))
        calls = [call["function"]["name"] for message in capped["messages"] for call in message.get("tool_calls", [])]
        self.assertEqual(calls, ["lookup", "lookup"])
        self.assertEqual([message["role"] for message in capped["messages"]].count("tool"), 2)

    async def test_calls_past_the_cap_fail_the_task(self):
        provider = ScriptedProvider(
            call_round("lookup", {}, "call_1"), call_round("lookup", {}, "call_2"), call_round("lookup", {}, "call_3")
        )
        shop, task, events = RecordingTools(), IMTask(), []
        with self.assertRaisesRegex(RuntimeError, "past the round limit"):
            async for event in adapter(provider, tools=[shop], settings=CAPPED).start(task).events():
                task.see(event)
                events.append(event)
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertEqual(len(shop.calls), 2)

    async def test_what_each_round_was_sent_is_logged_at_debug(self):
        task = IMTask()
        with patch("delegation_core.backend.llm.run.logger") as logger:
            await drive(adapter(ScriptedProvider(text_round("Done."))).start(task), task)
        [line] = [
            call.args[0].format(*call.args[1:])
            for call in logger.debug.call_args_list
            if "backend.request" in call.args[0]
        ]
        for part in ("work=W1", "task=W1.1", "round=1", "tools=['get_menu']", "COMPOSED-INSTRUCTIONS", "Do the task."):
            self.assertIn(part, line)


if __name__ == "__main__":
    unittest.main()
