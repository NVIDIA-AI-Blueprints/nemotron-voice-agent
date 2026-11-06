# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Tests for the model providers: the provider contract, the round's limits, the requests sent and the chat reply."""

import unittest

import httpx
from openai import BadRequestError

from examples.frontend_backend_live.delegation_core.backend.providers import (
    BrokenStreamError,
    ChatCompletionsBackend,
    OpenAIResponsesBackend,
    ProviderLimits,
)
from examples.shared.delegation_contract import ModelEndpoint
from tests.unit.contracts.backend import ProviderContract, a_round, call_events, message_events, text_round
from tests.unit.contracts.base import collect
from tests.unit.delegation_core.backend.llm.doubles import (
    ENDPOINT,
    LIMITS,
    FakeChat,
    FakeResponses,
    ScriptedProvider,
    Sleep,
    chunk,
    tool_fragment,
)

QUICK = ProviderLimits(provider_retries=1, first_event_timeout_seconds=0.2, idle_timeout_seconds=0.2)
THINKING = {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
MENU = {"type": "function", "name": "get_menu", "parameters": {"type": "object", "properties": {}}}
CART = {"type": "function", "name": "manage_cart", "parameters": {"type": "object", "properties": {}}}
HISTORY = [
    {"type": "message", "role": "user", "content": "Find teas."},
    {"type": "function_call", "call_id": "call_1", "name": "get_menu", "arguments": "{}"},
    {"type": "function_call_output", "call_id": "call_1", "output": "{}"},
]


def endpoint(**extra):
    return ModelEndpoint(model="test/model", base_url="http://localhost:1/v1", extra_params=extra)


def payload(**settings):
    return {"model": "test/model", "instructions": "Be brief.", "input": HISTORY, "tools": [MENU, CART], **settings}


def done(events):
    return [event["item"] for event in events if event["type"] == "response.output_item.done"]


def chat(*script, limits=LIMITS, **extra):
    """The chat provider on an endpoint with ``extra``, its client streaming ``script`` (or no client)."""
    return ChatCompletionsBackend(endpoint(**extra), limits, FakeChat(*script) if script else object())


class ChatCompletionsProviderTests(ProviderContract, unittest.IsolatedAsyncioTestCase):
    def make_provider(self, script):
        rounds = {
            "text": [chunk("Hel"), chunk("lo."), chunk(finish="stop")],
            "call": [chunk(tool_calls=[tool_fragment(0, "c1", "get_menu", "{}")]), chunk(finish="tool_calls")],
        }
        return ChatCompletionsBackend(ENDPOINT, LIMITS, FakeChat(rounds[script]))


class OpenAIResponsesProviderTests(ProviderContract, unittest.IsolatedAsyncioTestCase):
    def make_provider(self, script):
        rounds = {"text": a_round(message_events("Hello.")), "call": a_round(call_events("get_menu", {}))}
        return OpenAIResponsesBackend(ENDPOINT, LIMITS, FakeResponses(rounds[script]))


class RoundLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_silence_past_its_limit_ends_the_round_and_only_opening_is_retried(self):
        round_ = text_round("Done.")
        slow_first = ScriptedProvider([Sleep(1.0), *round_], round_, limits=QUICK)
        silent = ScriptedProvider([round_[0], Sleep(1.0), *round_[1:]], limits=QUICK)
        thinking = [chunk(None, reasoning_content="Let me"), Sleep(1.0), chunk("Done."), chunk(finish="stop")]
        stalled_chat = ChatCompletionsBackend(endpoint(**THINKING), QUICK, FakeChat(thinking))
        for provider in (slow_first, silent, stalled_chat):
            with self.subTest(provider=provider), self.assertRaises(TimeoutError):
                await collect(provider.stream(payload()))
        self.assertEqual((slow_first.opened, silent.opened), (1, 1))

    async def test_a_round_that_streams_steadily_is_never_cut_however_long_it_takes(self):
        # 0.4 s in all, but never 0.2 s without an event, or a reasoning chunk of a thinking model
        steady = ScriptedProvider([step for event in text_round("Done.") for step in (Sleep(0.1), event)], limits=QUICK)
        pieces = ("Let me", " think", " it", " over.")
        thinking = [step for piece in pieces for step in (Sleep(0.1), chunk(None, reasoning_content=piece))]
        chat = FakeChat([*thinking, chunk("A latte is $4.50."), chunk(finish="stop")])
        for provider in (steady, ChatCompletionsBackend(endpoint(**THINKING), QUICK, chat)):
            events = await collect(provider.stream(payload()))
            self.assertEqual(events[-1]["type"], "response.completed")
        [message] = done(events)
        self.assertEqual(message["content"][0]["text"], "A latte is $4.50.")  # the reasoning is no message

    async def test_a_refused_request_is_never_retried_and_reading_stops_at_the_first_terminal(self):
        request = httpx.Request("POST", "http://localhost:1/v1/chat/completions")
        refused = BadRequestError("bad request", response=httpx.Response(400, request=request), body=None)
        provider = ScriptedProvider([refused], text_round("Done."), limits=QUICK)
        with self.assertRaises(BadRequestError):
            await collect(provider.stream({}))
        self.assertEqual(provider.opened, 1)
        with self.assertRaises(BrokenStreamError):
            await collect(ScriptedProvider([], limits=ProviderLimits(provider_retries=0)).stream({}))  # no event
        provider = ScriptedProvider([*text_round("Done."), RuntimeError("after the terminal")], limits=QUICK)
        events = await collect(provider.stream({}))
        self.assertEqual((events[-1]["type"], provider.read_after_terminal), ("response.completed", False))


class RequestTests(unittest.TestCase):
    def test_every_round_sends_the_whole_history_with_nothing_stored_or_chained(self):
        provider = OpenAIResponsesBackend(endpoint(), LIMITS, client=object())
        first, second = provider.request(payload()), provider.request(payload())
        chat = ChatCompletionsBackend(endpoint(), LIMITS, client=object()).request(payload())
        self.assertEqual((first["input"], first["store"], second["input"]), (HISTORY, False, HISTORY))
        self.assertEqual([message["role"] for message in chat["messages"]], ["system", "user", "assistant", "tool"])
        for request, field in ((first, "previous_response_id"), (chat, "previous_response_id"), (chat, "store")):
            self.assertNotIn(field, request)
        self.assertEqual(first["prompt_cache_key"], second["prompt_cache_key"])  # one key per provider
        other = OpenAIResponsesBackend(endpoint(), LIMITS, client=object()).request(payload())
        self.assertNotEqual(other["prompt_cache_key"], first["prompt_cache_key"])

    def test_the_responses_provider_maps_the_round_settings_over_the_endpoint_defaults(self):
        defaults = endpoint(
            temperature=0.2, max_tokens=300, extra_body={"x": 1}, stream_options={"include_usage": True}
        )
        provider = OpenAIResponsesBackend(defaults, LIMITS, client=object())
        settings = {"reasoning": {"effort": "low"}, "text": {"format": {"type": "text"}}, "parallel_tool_calls": False}
        request = provider.request(payload(**settings))
        expected = {**settings, "temperature": 0.2, "max_output_tokens": 300, "extra_body": {"x": 1}}
        self.assertEqual({key: request[key] for key in expected}, expected)
        self.assertEqual(request["tool_choice"], "auto")
        self.assertNotIn("stream_options", request)  # a chat-only setting
        self.assertEqual(provider.request(payload(max_output_tokens=50))["max_output_tokens"], 50)  # the round wins
        bare = provider.request(payload(tools=[]))
        self.assertEqual(("tools" in bare, "tool_choice" in bare), (False, False))

    def test_the_chat_provider_maps_the_effort_the_format_the_tool_choice_and_the_input_to_its_own_settings(self):
        thinking, plain = chat(**THINKING), chat()
        efforts = [thinking.request(payload(reasoning={"effort": effort}))["extra_body"] for effort in ("none", "low")]
        self.assertEqual([body["chat_template_kwargs"]["enable_thinking"] for body in efforts], [False, True])
        self.assertIs(thinking.endpoint.extra_params["extra_body"]["chat_template_kwargs"]["enable_thinking"], True)
        self.assertNotIn("extra_body", plain.request(payload(reasoning={"effort": "none"})))
        strict = {"name": "reply", "schema": {"type": "object"}, "strict": True}
        for setting, key, expected in (
            (
                {"text": {"format": {"type": "json_schema", **strict}}},
                "response_format",
                {"type": "json_schema", "json_schema": strict},
            ),
            ({"text": {"format": {"type": "json_object"}}}, "response_format", {"type": "json_object"}),
            (
                {"tool_choice": {"type": "function", "name": "get_menu"}},
                "tool_choice",
                {"type": "function", "function": {"name": "get_menu"}},
            ),
            ({"tool_choice": {"type": "web_search"}}, "tool_choice", "auto"),  # a hosted tool's choice: any of ours
            ({"tool_choice": "sometimes"}, "tool_choice", "auto"),
            ({"tool_choice": "required"}, "tool_choice", "required"),
        ):
            self.assertEqual(plain.request(payload(**setting))[key], expected)
        bare = plain.request(payload(text={"format": {"type": "text"}}, tools=[], tool_choice="none"))
        self.assertEqual([key in bare for key in ("response_format", "tools", "tool_choice")], [False, False, False])
        image = {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "low"}
        by_file = {"type": "input_image", "file_id": "file_1"}  # left out: a chat endpoint cannot read it
        typed = [{"type": "input_text", "text": "What is this?"}, image, by_file]
        sent = [{"type": "message", "role": "user", "content": content} for content in (typed, "And the price?")]
        self.assertEqual(
            plain.request(payload(input=sent))["messages"][1]["content"],
            [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "low"}},
                {"type": "text", "text": "And the price?"},
            ],
        )
        text_only = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "a"}, by_file]}]
        self.assertEqual(plain.request(payload(input=text_only))["messages"][1], {"role": "user", "content": "a"})


class ChatReplyTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_call_per_round_keeps_the_first_call_and_ends_the_round_when_a_second_starts(self):
        def call(index, name, arguments="{}"):
            return tool_fragment(index, f"p{index}", name, arguments)

        both = [chunk(tool_calls=[call(0, "get_menu"), call(1, "manage_cart")]), chunk(finish="tool_calls")]
        # The output cap cuts the second call, after the round ended with the first: a cut never costs the first.
        cut = [chunk(tool_calls=[call(0, "get_menu")]), chunk(tool_calls=[call(1, "manage_cart", '{"item": "te')])]
        cut += [Sleep(5), chunk(finish="length")]  # never read: a provider that read on would hit its silence limit
        for script, parallel, names in (
            (both, True, ["get_menu", "manage_cart"]),
            (both, False, ["get_menu"]),
            (cut, False, ["get_menu"]),
        ):
            with self.subTest(parallel=parallel, steps=len(script)):
                client = FakeChat(list(script))
                events = await collect(
                    ChatCompletionsBackend(endpoint(), LIMITS, client).stream(payload(parallel_tool_calls=parallel))
                )
                self.assertEqual(
                    ([item["name"] for item in done(events)], events[-1]["type"]), (names, "response.completed")
                )
                self.assertNotIn("parallel_tool_calls", client.requests[0])  # never sent: an endpoint may refuse it
                self.assertNotIn("p0", [item["call_id"] for item in done(events)])  # its call ids are our own

    async def test_a_reply_cut_short_never_gives_its_call_and_a_stream_that_ends_inside_a_call_is_broken(self):
        for finish, arguments, reason in (
            ("length", '{"seat": "1', "max_output_tokens"),  # the call itself was cut
            ("content_filter", '{"seat": "1A"}', "content_filter"),  # a whole call, never given either
        ):
            with self.subTest(finish):
                cut = [
                    chunk("Booking"),
                    chunk(tool_calls=[tool_fragment(0, "c1", "book", arguments)]),
                    chunk(finish=finish),
                ]
                events = await collect(chat(cut).stream(payload()))
                self.assertEqual([item["type"] for item in done(events)], ["message"])
                self.assertEqual(events[-1]["response"]["incomplete_details"], {"reason": reason})
        with self.assertRaises(BrokenStreamError):
            await collect(chat([chunk(tool_calls=[tool_fragment(0, "c1", "book", '{"seat": ')])]).stream(payload()))


if __name__ == "__main__":
    unittest.main()
