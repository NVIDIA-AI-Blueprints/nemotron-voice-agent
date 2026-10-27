# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Tests that every round sends the whole history and that nothing is stored."""

import json
import unittest

from delegation_core.backend.providers.chat_completions import ChatCompletionsBackend
from delegation_core.backend.providers.openai_responses import OpenAIResponsesBackend
from delegation_core.config import ModelEndpoint
from tests.unit.contracts.backend import call_round, text_round
from tests.unit.delegation_core.backend.llm.doubles import (
    LIMITS,
    IMTask,
    RecordingTools,
    ScriptedProvider,
    adapter,
    drive,
)

HISTORY = [
    {"type": "message", "role": "user", "content": "Find teas."},
    {"type": "function_call", "call_id": "call_1", "name": "get_menu", "arguments": "{}"},
    {"type": "function_call_output", "call_id": "call_1", "output": "{}"},
]
MENU = {"type": "function", "name": "get_menu", "parameters": {"type": "object", "properties": {}}}


def endpoint(**extra):
    return ModelEndpoint(model="test/model", base_url="http://localhost:1/v1", extra_params=extra)


def payload(**changes):
    return {"model": "test/model", "instructions": "Be brief.", "input": HISTORY, "tools": [MENU], **changes}


class ResponsesRequestTests(unittest.TestCase):
    def test_each_round_sends_the_whole_history_unchained_and_unstored(self):
        provider = OpenAIResponsesBackend(endpoint(), LIMITS, client=object())
        first, second = provider.request(payload()), provider.request(payload())
        for request in (first, second):
            self.assertEqual(request["input"], HISTORY)
            self.assertIs(request["store"], False)
            self.assertNotIn("previous_response_id", request)
        self.assertEqual(first["prompt_cache_key"], second["prompt_cache_key"])
        self.assertTrue(first["prompt_cache_key"].startswith("cache_"))
        other = OpenAIResponsesBackend(endpoint(), LIMITS, client=object()).request(payload())
        self.assertNotEqual(other["prompt_cache_key"], first["prompt_cache_key"])

    def test_the_round_settings_and_the_endpoint_defaults_are_mapped(self):
        provider = OpenAIResponsesBackend(
            endpoint(temperature=0.2, max_tokens=300, extra_body={"x": 1}, stream_options={"include_usage": True}),
            LIMITS,
            client=object(),
        )
        request = provider.request(
            payload(reasoning={"effort": "low"}, text={"format": {"type": "text"}}, parallel_tool_calls=False)
        )
        self.assertEqual(
            {key: request[key] for key in ("temperature", "max_output_tokens", "extra_body", "reasoning", "text")},
            {
                "temperature": 0.2,
                "max_output_tokens": 300,
                "extra_body": {"x": 1},
                "reasoning": {"effort": "low"},
                "text": {"format": {"type": "text"}},
            },
        )
        self.assertEqual((request["parallel_tool_calls"], request["tool_choice"]), (False, "auto"))
        self.assertNotIn("stream_options", request)  # a chat-only setting
        self.assertEqual(provider.request(payload(max_output_tokens=50))["max_output_tokens"], 50)  # the round wins

    def test_no_tool_choice_without_tools(self):
        request = OpenAIResponsesBackend(endpoint(), LIMITS, client=object()).request(payload(tools=[]))
        self.assertNotIn("tools", request)
        self.assertNotIn("tool_choice", request)


class RunPayloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_round_sends_the_whole_history_and_no_chain_field(self):
        provider = ScriptedProvider(call_round("lookup", {}), text_round("Done.", "resp_2"))
        task = IMTask(history=[{"type": "message", "role": "user", "content": "Find teas."}])
        await drive(adapter(provider, tools=[RecordingTools()]).start(task), task)
        first, second = provider.payloads
        self.assertEqual(first["input"], [{"type": "message", "role": "user", "content": "Find teas."}])
        self.assertEqual(
            [item.get("type", "message") for item in second["input"]],
            ["message", "function_call", "function_call_output"],
        )
        for sent in provider.payloads:
            for field in ("store", "previous_response_id", "prompt_cache_key"):
                self.assertNotIn(field, sent)


class ChatRequestTests(unittest.TestCase):
    def test_a_chat_round_sends_the_whole_history_and_no_chain_field(self):
        request = ChatCompletionsBackend(endpoint(), LIMITS, client=object()).request(payload())
        self.assertEqual([message["role"] for message in request["messages"]], ["system", "user", "assistant", "tool"])
        for field in ("store", "previous_response_id", "prompt_cache_key"):
            self.assertNotIn(field, request)
        json.dumps(request)  # plain JSON


if __name__ == "__main__":
    unittest.main()
