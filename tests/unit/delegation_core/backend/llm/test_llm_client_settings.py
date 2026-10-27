# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Tests that the chat provider honors the client's settings and reports what it cannot."""

import unittest

from delegation_core.backend.providers.chat_completions import ChatCompletionsBackend
from delegation_core.backend.providers.openai_responses import OpenAIResponsesBackend
from delegation_core.config import ModelEndpoint
from tests.unit.contracts.base import collect
from tests.unit.delegation_core.backend.llm.doubles import LIMITS, FakeChat, chunk, tool_fragment

MENU = {"type": "function", "name": "get_menu", "parameters": {"type": "object", "properties": {}}}
CART = {"type": "function", "name": "manage_cart", "parameters": {"type": "object", "properties": {}}}
THINKING = {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
# What tau2 sends: functions, tool_choice auto, one call per round, maybe an effort.
TAU2 = {
    "model": "test/model",
    "instructions": "Policy.",
    "tools": [MENU],
    "tool_choice": "auto",
    "parallel_tool_calls": False,
    "reasoning": {"effort": "low"},
}


def call(index, name):
    return tool_fragment(index, f"p{index}", name, "{}")


def provider(*script, **extra):
    endpoint = ModelEndpoint(model="test/model", base_url="http://localhost:1/v1", extra_params=extra)
    return ChatCompletionsBackend(endpoint, LIMITS, FakeChat(*script))


def payload(**settings):
    return {"model": "test/model", "instructions": "Be brief.", "input": [], "tools": [MENU, CART], **settings}


class ClientSettingsTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_call_per_round_keeps_only_the_first_call(self):
        both = [chunk(tool_calls=[call(0, "get_menu"), call(1, "manage_cart")]), chunk(finish="tool_calls")]
        for parallel, expected in ((False, ["get_menu"]), (True, ["get_menu", "manage_cart"])):
            with self.subTest(parallel=parallel):
                events = await collect(provider(list(both)).stream(payload(parallel_tool_calls=parallel)))
                done = [e["item"] for e in events if e["type"] == "response.output_item.done"]
                self.assertEqual([item["name"] for item in done], expected)
        chat = provider([chunk(finish="stop")])
        await collect(chat.stream(payload(parallel_tool_calls=False)))
        self.assertNotIn("parallel_tool_calls", chat.client.requests[0])  # never sent: an endpoint may refuse it

    def test_effort_none_turns_thinking_off_where_the_endpoint_has_the_switch(self):
        chat = provider(**THINKING)
        off = chat.request(payload(reasoning={"effort": "none"}))
        self.assertIs(off["extra_body"]["chat_template_kwargs"]["enable_thinking"], False)
        self.assertIs(chat.endpoint.extra_params["extra_body"]["chat_template_kwargs"]["enable_thinking"], True)
        self.assertIs(
            chat.request(payload(reasoning={"effort": "low"}))["extra_body"]["chat_template_kwargs"]["enable_thinking"],
            True,
        )
        self.assertNotIn("extra_body", provider().request(payload(reasoning={"effort": "none"})))

    def test_text_format_becomes_the_response_format(self):
        schema = {"type": "object", "properties": {"speak": {"type": "string"}}}
        cases = [
            (
                {"type": "json_schema", "name": "reply", "schema": schema, "strict": True},
                {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema, "strict": True}},
            ),
            ({"type": "json_object"}, {"type": "json_object"}),
        ]
        for text_format, expected in cases:
            with self.subTest(text_format=text_format["type"]):
                self.assertEqual(provider().request(payload(text={"format": text_format}))["response_format"], expected)
        self.assertNotIn("response_format", provider().request(payload(text={"format": {"type": "text"}})))

    def test_what_it_cannot_honor_is_named(self):
        settings = {
            "tools": [MENU, {"type": "web_search"}, {"type": "web_search"}, {"type": "file_search"}],
            "service_tier": "flex",
            "text": {"verbosity": "low"},
        }
        problems = ChatCompletionsBackend.unsupported(settings)
        self.assertEqual([param for param, _ in problems], ["tools", "tools", "service_tier", "text.verbosity"])
        self.assertIn("web_search", problems[0][1])
        self.assertEqual(ChatCompletionsBackend.unsupported({"service_tier": "auto"}), [])

    def test_tau2s_session_body_reports_nothing(self):
        self.assertEqual(ChatCompletionsBackend.unsupported(TAU2), [])
        self.assertEqual(OpenAIResponsesBackend.unsupported(TAU2), [])
        self.assertEqual(OpenAIResponsesBackend.unsupported({"tools": [{"type": "web_search"}]}), [])


if __name__ == "__main__":
    unittest.main()
