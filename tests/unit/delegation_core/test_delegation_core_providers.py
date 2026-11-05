# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Chat-completions to Responses translation and the chat-completions backend provider."""

import json
import unittest
from dataclasses import replace

import httpx
from openai import APIError, APITimeoutError, BadRequestError

from examples.frontend_backend_live.delegation_core.backend.providers.chat_completions import ChatCompletionsBackend
from examples.frontend_backend_live.delegation_core.backend.providers.translation import (
    ChatResponseTranslator,
    responses_items_to_messages,
    to_chat_tools,
)
from examples.frontend_backend_live.delegation_core.backend.tools.cafe.schemas import CAFE_TOOLS
from tests.unit.core_helpers import ENDPOINT, LIMITS, FakeChat, _timeout, chunk, tool_fragment


def translated(*chunks, response_id: str | None = None) -> list[dict]:
    """The Responses events one streamed reply of ``chunks`` becomes."""
    translator = ChatResponseTranslator("nvidia/test", **({"response_id": response_id} if response_id else {}))
    events = translator.start()
    for item in chunks:
        events += translator.feed(item)
    return events + translator.finish()


def done_items(events: list[dict]) -> list[dict]:
    return [event["item"] for event in events if event["type"] == "response.output_item.done"]


LENGTH_CUT = {"reason": "max_output_tokens"}  # a reply cut by the token limit


def text_of(content: str) -> dict:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": content}]}


class RequestTests(unittest.TestCase):
    def test_items_become_chat_messages_with_their_tool_calls_and_results(self):
        said = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Checking."}]}
        items = [
            text_of("a"),
            text_of("b"),
            {"type": "function_call", "call_id": "call_1", "name": "get_menu", "arguments": '{"category":"pastry"}'},
            {"type": "function_call", "call_id": "call_2", "name": "manage_cart", "arguments": ""},
            {"type": "function_call_output", "call_id": "call_1", "output": '{"items":[]}'},
            {"type": "function_call_output", "call_id": "call_2", "output": "{}"},
            {"type": "reasoning", "summary": []},
            said,
            {"type": "function_call", "call_id": "c", "name": "get_menu", "arguments": "{}"},
        ]
        system, merged, calls, first, second, last = responses_items_to_messages("Be brief.", items)
        self.assertEqual((system["content"], merged), ("Be brief.", {"role": "user", "content": "a\nb"}))
        self.assertEqual([call["id"] for call in calls["tool_calls"]], ["call_1", "call_2"])
        self.assertEqual(calls["tool_calls"][1]["function"]["arguments"], "{}")  # empty arguments stay valid JSON
        self.assertEqual(first, {"role": "tool", "tool_call_id": "call_1", "content": '{"items":[]}'})
        self.assertEqual(second["tool_call_id"], "call_2")  # the reasoning item is dropped
        self.assertEqual((last["content"], last["tool_calls"][0]["id"]), ("Checking.", "c"))  # text, then its call

    def test_hosted_tools_are_dropped_and_function_tools_converted(self):
        function = {"name": "get_menu", "description": "The menu", "parameters": {"type": "object", "properties": {}}}
        tools, dropped = to_chat_tools([{"type": "web_search"}, {"type": "function", **function, "strict": False}])
        self.assertEqual((tools, dropped), ([{"type": "function", "function": function}], ["web_search"]))


class TranslationTests(unittest.TestCase):
    def test_a_text_reply_streams_the_responses_lifecycle_in_order_with_its_usage(self):
        usage = {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}
        events = translated(chunk("The pastries"), chunk(" are croissants."), chunk(finish="stop"), chunk(usage=usage))
        kinds = ["created", "in_progress", "output_item.added", "content_part.added", "output_text.delta"]
        kinds += ["output_text.delta", "output_text.done", "content_part.done", "output_item.done", "completed"]
        self.assertEqual([event["type"] for event in events], [f"response.{kind}" for kind in kinds])
        self.assertEqual([event["sequence_number"] for event in events], list(range(1, len(events) + 1)))
        (item,) = done_items(events)
        self.assertEqual(item["content"][0]["text"], "The pastries are croissants.")
        completed = translated(chunk("Hi."), chunk(finish="stop"), chunk(usage=usage), response_id="resp_x")[-1]
        self.assertEqual((completed["response"]["id"], completed["response"]["status"]), ("resp_x", "completed"))
        self.assertEqual(completed["response"]["usage"], {"input_tokens": 7, "output_tokens": 5, "total_tokens": 12})

    def test_call_fragments_become_function_calls_with_ids_of_our_own_after_any_text(self):
        menu = [tool_fragment(0, "call_abc", "get_menu", ""), tool_fragment(0, arguments='{"cate')]
        rest = [tool_fragment(0, arguments='gory":"pastry"}'), tool_fragment(1, None, "manage_cart", "{}")]
        fragments = [chunk(tool_calls=[menu[0]]), chunk(tool_calls=[menu[1]]), chunk(tool_calls=rest)]
        events = translated(chunk("One moment."), *fragments, chunk(finish="tool_calls"))
        message, menu, cart = done_items(events)
        self.assertEqual((menu["name"], json.loads(menu["arguments"])), ("get_menu", {"category": "pastry"}))
        self.assertEqual(cart["name"], "manage_cart")
        self.assertTrue(all(call["call_id"].startswith("call_") for call in (menu, cart)))
        self.assertNotIn(menu["call_id"], ("call_abc", cart["call_id"]))  # ours, and distinct
        deltas = "".join(e["delta"] for e in events if e["type"] == "response.function_call_arguments.delta")
        self.assertEqual(deltas, menu["arguments"] + cart["arguments"])
        added = [e["output_index"] for e in events if e["type"] == "response.output_item.added"]
        self.assertEqual((added, message["type"], events[-1]["type"]), ([0, 1, 2], "message", "response.completed"))

    def test_reasoning_never_reaches_the_wire_and_a_cut_reply_is_incomplete(self):
        thoughts = [chunk(None, reasoning_content="private plan"), chunk(None, reasoning=" more")]  # either name
        reasoned = translated(*thoughts, chunk("$4.07."), chunk(finish="stop"))
        self.assertNotIn("private plan", json.dumps(reasoned))
        self.assertNotIn(" more", json.dumps(reasoned))
        self.assertEqual([e["delta"] for e in reasoned if e["type"] == "response.output_text.delta"], ["$4.07."])
        only_thinking = translated(chunk(None, reasoning_content="still thinking"), chunk(finish="length"))
        self.assertEqual((done_items(only_thinking), only_thinking[-1]["type"]), ([], "response.incomplete"))
        cut = translated(chunk("Half an ans"), chunk(finish="length"))[-1]
        self.assertEqual((cut["type"], cut["response"]["incomplete_details"]), ("response.incomplete", LENGTH_CUT))
        nameless = translated(chunk(tool_calls=[tool_fragment(0, "call_1", None, "{}")]), chunk(finish="stop"))
        self.assertEqual(done_items(nameless), [])  # never surfaced as a call to run


class ChatBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_round_sends_the_tools_and_history_and_translates_a_function_call(self):
        fragments = [tool_fragment(0, "call_1", "get_menu", '{"category":'), tool_fragment(0, arguments='"tea"}')]
        client = FakeChat([chunk(tool_calls=fragments[:1]), chunk(tool_calls=fragments[1:], finish="tool_calls")])
        tools = CAFE_TOOLS + [{"type": "web_search"}]
        payload = {"model": "test/model", "instructions": "Use tools.", "tools": tools, "tool_choice": "auto"}
        payload["input"] = [text_of("teas?")]
        events = [event async for event in ChatCompletionsBackend(ENDPOINT, LIMITS, client).stream(payload)]
        request = client.requests[0]
        self.assertEqual((request["temperature"], request["max_tokens"], request["tool_choice"]), (0.7, 1024, "auto"))
        self.assertEqual([t["function"]["name"] for t in request["tools"]], ["get_menu", "manage_cart", "place_order"])
        self.assertEqual([m["role"] for m in request["messages"]], ["system", "user"])
        (call,) = [item for item in done_items(events) if item["type"] == "function_call"]
        self.assertEqual((call["name"], json.loads(call["arguments"])), ("get_menu", {"category": "tea"}))
        self.assertEqual(events[-1]["type"], "response.completed")

    async def test_a_failure_that_opens_the_stream_is_retried_but_a_rejected_request_or_a_spent_retry_is_not(self):
        request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
        overloaded = APIError("Service temporarily overloaded", request, body=None)
        for opening in (_timeout(), overloaded):
            backend = ChatCompletionsBackend(ENDPOINT, LIMITS, FakeChat(opening, [chunk("Hello", finish="stop")]))
            self.assertIn("response.output_text.delta", [e["type"] async for e in backend.stream({"input": []})])
        rejected = BadRequestError("bad request", response=httpx.Response(400, request=request), body=None)
        for error, limits in ((rejected, LIMITS), (_timeout(), replace(LIMITS, provider_retries=0))):
            backend = ChatCompletionsBackend(ENDPOINT, limits, FakeChat(error, [chunk("Hello", finish="stop")]))
            with self.subTest(type(error).__name__), self.assertRaises((BadRequestError, APITimeoutError)):
                [e async for e in backend.stream({"input": []})]
