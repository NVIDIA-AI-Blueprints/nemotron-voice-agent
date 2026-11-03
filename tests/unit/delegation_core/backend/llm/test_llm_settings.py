# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Tests for per-round settings and the settings the backend LLM cannot honor."""

import unittest

from delegation_core.backend.llm import LlmSettings
from delegation_core.backend.providers import OpenAIResponsesBackend
from tests.unit.contracts.backend import call_round, text_round
from tests.unit.delegation_core.backend.llm.doubles import IMTask, RecordingTools, ScriptedProvider, adapter, source

CART = {"type": "function", "name": "manage_cart", "parameters": {"type": "object", "properties": {}}}


class SettingsTests(unittest.IsolatedAsyncioTestCase):
    def test_web_search_is_listed_with_the_chat_provider_and_passed_on_by_the_responses_one(self):
        settings = {"tools": [{"type": "web_search"}], "tool_choice": "auto", "service_tier": "priority"}
        self.assertEqual([param for param, _ in source().unsupported(settings)], ["tools", "service_tier"])
        self.assertEqual(source(settings=LlmSettings(provider=OpenAIResponsesBackend)).unsupported(settings), [])

    async def test_a_changed_function_list_reaches_the_next_round(self):
        provider = ScriptedProvider(call_round("lookup", {}), text_round("Done.", "resp_2"))
        task = IMTask()
        run = adapter(provider, tools=[RecordingTools()]).start(task)
        async for event in run.events():
            task.see(event)
            if event["type"] == "response.completed" and len(task.requests) == 1:
                task.declared.append(CART)  # e.g. a session.update the IM took before the next round
        self.assertEqual([tool["name"] for tool in provider.payloads[0]["tools"]], ["get_menu", "lookup"])
        self.assertEqual(
            [tool["name"] for tool in provider.payloads[1]["tools"]], ["get_menu", "manage_cart", "lookup"]
        )

    async def test_a_continuations_settings_reach_its_round(self):
        provider = ScriptedProvider(call_round("get_menu", {}), text_round("Done.", "resp_2"))
        task = IMTask()
        run = adapter(provider).start(task)
        async for event in run.events():
            task.see(event)
        task.instructions = "The continuation's instructions."
        task.give_outputs([{"type": "function_call_output", "call_id": "call_1", "output": "{}"}])
        await run.resume()
        async for event in run.events():
            task.see(event)
        self.assertEqual(
            [payload["instructions"] for payload in provider.payloads],
            ["COMPOSED-INSTRUCTIONS", "The continuation's instructions."],
        )


if __name__ == "__main__":
    unittest.main()
