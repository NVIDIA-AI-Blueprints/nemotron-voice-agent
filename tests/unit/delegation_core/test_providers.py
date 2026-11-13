# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The model providers: chat-completions to Responses translation, the requests they send and the rounds they read."""

import json
import unittest
from types import SimpleNamespace

import httpx
from openai import APIError, APITimeoutError, BadRequestError

from examples.frontend_backend_live.delegation_core.backend.providers import (
    BrokenStreamError,
    ChatCompletionsBackend,
    OpenAIResponsesBackend,
    ProviderLimits,
)
from examples.frontend_backend_live.delegation_core.backend.providers.retry import StreamErrorEvent
from examples.frontend_backend_live.delegation_core.backend.providers.translation import (
    ChatResponseTranslator,
    responses_items_to_messages,
    to_chat_tools,
)
from examples.frontend_backend_live.delegation_core.backend.tools.cafe.schemas import CAFE_TOOLS
from examples.shared.delegation_contract import ModelEndpoint
from tests.unit.core_helpers import FakeChat, FakeStream, Sleep, _timeout, chunk, tool_fragment
from tests.unit.delegation_core.helpers import (
    ENDPOINT,
    LIMITS,
    MENU,
    NO_RETRY,
    ScriptedProvider,
    call_round,
    finished_items,
    kinds,
    read_all,
    text_round,
)

QUICK = ProviderLimits(provider_retries=1, first_event_timeout_seconds=0.2, idle_timeout_seconds=0.2)
THINKING = {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
CART = {"type": "function", "name": "manage_cart", "parameters": {"type": "object", "properties": {}}}
HISTORY = [
    {"type": "message", "role": "user", "content": "Find teas."},
    {"type": "function_call", "call_id": "call_1", "name": "get_menu", "arguments": "{}"},
    {"type": "function_call_output", "call_id": "call_1", "output": "{}"},
]
LENGTH_CUT = {"reason": "max_output_tokens"}


def text_of(content: str) -> dict:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": content}]}


def translated(*chunks) -> list[dict]:
    translator = ChatResponseTranslator("nvidia/test", single_call=False)
    events = translator.start()
    for item in chunks:
        events += translator.feed(item)
    return events + translator.finish()


def endpoint(**extra):
    return ModelEndpoint(model="test/model", base_url="http://localhost:1/v1", extra_params=extra)


def payload(**settings):
    return {"model": "test/model", "instructions": "Be brief.", "input": HISTORY, "tools": [MENU, CART], **settings}


def chat(*script, limits=NO_RETRY, **extra):
    """Build the chat provider on an endpoint with ``extra``, its client streaming ``script`` (or no client)."""
    return ChatCompletionsBackend(endpoint(**extra), limits, FakeChat(*script) if script else object())


class FakeResponses:
    """Scripted ``responses.create``: each call streams the next round's events, as the client's models."""

    def __init__(self, *rounds):
        self.rounds, self.responses = list(rounds), self

    async def create(self, **request):
        dumps = [lambda event=event, **_: json.loads(json.dumps(event)) for event in self.rounds.pop(0)]
        return FakeStream([SimpleNamespace(model_dump=dump) for dump in dumps])


class TranslationTests(unittest.TestCase):
    def test_items_become_chat_messages_with_calls_and_results(self):
        said = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Checking."}]}
        parts = [{"type": "input_text", "text": "Teas."}, {"type": "input_image", "image_url": "https://x/t.png"}]
        array = [*parts, {"type": "input_text", "text": "Hot."}]
        items = [
            text_of("a"),
            {"role": "user", "content": "b"},
            {"type": "function_call", "call_id": "call_1", "name": "get_menu", "arguments": '{"category":"pastry"}'},
            {"type": "function_call", "call_id": "call_2", "name": "manage_cart", "arguments": ""},
            {"type": "function_call_output", "call_id": "call_1", "output": '{"items":[]}'},
            {"type": "function_call_output", "call_id": "call_2", "output": array},
            {"type": "reasoning", "summary": []},
            said,
            {"type": "function_call", "call_id": "c", "name": "get_menu", "arguments": "{}"},
        ]
        system, merged, calls, first, second, last = responses_items_to_messages("Be brief.", items)
        self.assertEqual((system["content"], merged), ("Be brief.", {"role": "user", "content": "a\nb"}))
        self.assertEqual([call["id"] for call in calls["tool_calls"]], ["call_1", "call_2"])
        self.assertEqual(calls["tool_calls"][1]["function"]["arguments"], "{}")
        self.assertEqual(first, {"role": "tool", "tool_call_id": "call_1", "content": '{"items":[]}'})
        self.assertEqual(second, {"role": "tool", "tool_call_id": "call_2", "content": "Teas.\nHot."})
        self.assertEqual((last["content"], last["tool_calls"][0]["id"]), ("Checking.", "c"))
        for output in ({"stock": 3}, None, 42, True, [{"id": "o1"}], []):
            with self.subTest(output=output):
                (tool,) = responses_items_to_messages(None, [{**items[4], "output": output}])
                self.assertEqual(tool, {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(output)})
        with self.assertRaisesRegex(ValueError, "^The result for call call_1 has no output$"):
            responses_items_to_messages(None, [{"type": "function_call_output", "call_id": "call_1"}])

    def test_hosted_tools_are_dropped_and_functions_converted(self):
        function = {"name": "get_menu", "description": "The menu", "parameters": {"type": "object", "properties": {}}}
        tools, dropped = to_chat_tools([{"type": "web_search"}, {"type": "function", **function, "strict": False}])
        self.assertEqual((tools, dropped), ([{"type": "function", "function": function}], ["web_search"]))

    def test_a_text_reply_streams_the_lifecycle_with_its_usage(self):
        usage = {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}
        events = translated(chunk("The pastries"), chunk(" are croissants."), chunk(finish="stop"), chunk(usage=usage))
        steps = ["created", "in_progress", "output_item.added", "content_part.added", "output_text.delta"]
        steps += ["output_text.delta", "output_text.done", "content_part.done", "output_item.done", "completed"]
        self.assertEqual(kinds(events), [f"response.{step}" for step in steps])
        self.assertEqual([event["sequence_number"] for event in events], list(range(1, len(events) + 1)))
        (item,) = finished_items(events)
        self.assertEqual(item["content"][0]["text"], "The pastries are croissants.")
        created, completed = events[0]["response"], events[-1]["response"]
        self.assertEqual((completed["id"], completed["status"]), (created["id"], "completed"))
        self.assertEqual(completed["usage"], {"input_tokens": 7, "output_tokens": 5, "total_tokens": 12})

    def test_call_fragments_become_calls_with_our_own_ids(self):
        menu = [tool_fragment(0, "call_abc", "get_menu", ""), tool_fragment(0, arguments='{"cate')]
        rest = [tool_fragment(0, arguments='gory":"pastry"}'), tool_fragment(1, None, "manage_cart", "{}")]
        fragments = [chunk(tool_calls=[menu[0]]), chunk(tool_calls=[menu[1]]), chunk(tool_calls=rest)]
        events = translated(chunk("One moment."), *fragments, chunk(finish="tool_calls"))
        message, menu, cart = finished_items(events)
        self.assertEqual((menu["name"], json.loads(menu["arguments"])), ("get_menu", {"category": "pastry"}))
        self.assertEqual(cart["name"], "manage_cart")
        self.assertTrue(all(call["call_id"].startswith("call_") for call in (menu, cart)))
        self.assertNotIn(menu["call_id"], ("call_abc", cart["call_id"]))
        deltas = "".join(e["delta"] for e in events if e["type"] == "response.function_call_arguments.delta")
        self.assertEqual(deltas, menu["arguments"] + cart["arguments"])
        added = [e["output_index"] for e in events if e["type"] == "response.output_item.added"]
        self.assertEqual((added, message["type"], events[-1]["type"]), ([0, 1, 2], "message", "response.completed"))

    def test_reasoning_never_reaches_the_wire(self):
        # A provider names its reasoning either way.
        thoughts = [chunk(None, reasoning_content="private plan"), chunk(None, reasoning=" more")]
        reasoned = translated(*thoughts, chunk("$4.07."), chunk(finish="stop"))
        self.assertNotIn("private plan", json.dumps(reasoned))
        self.assertNotIn(" more", json.dumps(reasoned))
        self.assertEqual([e["delta"] for e in reasoned if e["type"] == "response.output_text.delta"], ["$4.07."])
        only_thinking = translated(chunk(None, reasoning_content="still thinking"), chunk(finish="length"))
        self.assertEqual((finished_items(only_thinking), only_thinking[-1]["type"]), ([], "response.incomplete"))
        cut = translated(chunk("Half an ans"), chunk(finish="length"))[-1]
        self.assertEqual((cut["type"], cut["response"]["incomplete_details"]), ("response.incomplete", LENGTH_CUT))
        nameless = translated(chunk(tool_calls=[tool_fragment(0, "call_1", None, "{}")]), chunk(finish="stop"))
        self.assertEqual(finished_items(nameless), [])


class RequestTests(unittest.TestCase):
    def test_every_round_sends_the_whole_history_unstored(self):
        provider = OpenAIResponsesBackend(endpoint(), NO_RETRY, client=object())
        first, second = provider.request(payload()), provider.request(payload())
        sent = ChatCompletionsBackend(endpoint(), NO_RETRY, client=object()).request(payload())
        self.assertEqual((first["input"], first["store"], second["input"]), (HISTORY, False, HISTORY))
        self.assertEqual([message["role"] for message in sent["messages"]], ["system", "user", "assistant", "tool"])
        for request, field in ((first, "previous_response_id"), (sent, "previous_response_id"), (sent, "store")):
            self.assertNotIn(field, request)
        self.assertEqual(first["prompt_cache_key"], second["prompt_cache_key"])
        other = OpenAIResponsesBackend(endpoint(), NO_RETRY, client=object()).request(payload())
        self.assertNotEqual(other["prompt_cache_key"], first["prompt_cache_key"])

    def test_the_responses_provider_sends_input_messages_without_their_ids(self):
        provider = OpenAIResponsesBackend(endpoint(), NO_RETRY, client=object())
        said = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Checking."}]}
        notes = [{"type": "message", "role": role, "content": "Be brief."} for role in ("system", "developer")]
        for item in (text_of("Teas?"), *notes, said):
            with self.subTest(role=item["role"]):
                self.assertEqual(provider.request(payload(input=[{**item, "id": "D1.2"}]))["input"], [item])
        # An output message, a call and reasoning keep their ids.
        kept = [{**said, "id": "msg_1", "status": "completed"}, {**HISTORY[1], "id": "fc_call_1"}]
        kept.append({"type": "reasoning", "id": "rs_1", "summary": []})
        self.assertEqual(provider.request(payload(input=kept))["input"], kept)

    def test_the_responses_provider_maps_the_round_settings(self):
        defaults = endpoint(
            temperature=0.2, max_tokens=300, extra_body={"x": 1}, stream_options={"include_usage": True}
        )
        provider = OpenAIResponsesBackend(defaults, NO_RETRY, client=object())
        settings = {"reasoning": {"effort": "low"}, "text": {"format": {"type": "text"}}, "parallel_tool_calls": False}
        request = provider.request(payload(**settings))
        expected = {**settings, "temperature": 0.2, "max_output_tokens": 300, "extra_body": {"x": 1}}
        self.assertEqual({key: request[key] for key in expected}, expected)
        self.assertEqual(request["tool_choice"], "auto")
        self.assertNotIn("stream_options", request)
        self.assertEqual(provider.request(payload(max_output_tokens=50))["max_output_tokens"], 50)
        bare = provider.request(payload(tools=[]))
        self.assertEqual(("tools" in bare, "tool_choice" in bare), (False, False))
        # A tool the client runs is left out, and a choice that only a tool left out matches becomes auto.
        custom, local = {"type": "custom", "name": "run_sql"}, {"type": "shell", "environment": {"type": "local"}}
        searched, quiet = {"type": "web_search"}, {"type": "mcp", "server_label": "A", "require_approval": "never"}
        asking = [
            {"type": "mcp", "server_label": "B"},
            {"type": "mcp", "server_label": "C", "require_approval": {"never": {"read_only": True}}},
        ]
        for choice, sent in (
            (custom, "auto"),
            (searched, searched),
            ({"type": "shell"}, "auto"),
            ({"type": "mcp", "server_label": "A"}, {"type": "mcp", "server_label": "A"}),
            ({"type": "mcp", "server_label": "B"}, "auto"),
            ("required", "required"),
        ):
            mixed = provider.request(payload(tools=[MENU, custom, searched, local, quiet, *asking], tool_choice=choice))
            self.assertEqual((mixed["tools"], mixed["tool_choice"]), ([MENU, searched, quiet], sent))
        only = provider.request(payload(tools=[custom], tool_choice=custom))
        self.assertEqual(("tools" in only, "tool_choice" in only), (False, False))
        off = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        on, quiet, plain = (
            OpenAIResponsesBackend(endpoint(**extra), NO_RETRY, object()) for extra in (THINKING, off, {})
        )
        cases = {
            "thinking on": (on, None, True, None),
            "another effort": (on, {"effort": "low"}, True, {"effort": "low"}),
            "the round's effort none": (on, {"effort": "none"}, False, {"effort": "none"}),
            "thinking off": (quiet, None, False, {"effort": "none"}),
            "thinking off over an effort": (
                quiet,
                {"effort": "high", "summary": "auto"},
                False,
                {"effort": "none", "summary": "auto"},
            ),
            "no switch": (plain, {"effort": "none"}, None, {"effort": "none"}),
        }
        for name, (provider, reasoning, switch, sent) in cases.items():
            with self.subTest(name):
                request = provider.request(payload(reasoning=reasoning))
                kwargs = (request.get("extra_body") or {}).get("chat_template_kwargs") or {}
                self.assertEqual((kwargs.get("enable_thinking"), request.get("reasoning")), (switch, sent))
        self.assertIs(on.endpoint.extra_params["extra_body"]["chat_template_kwargs"]["enable_thinking"], True)

    def test_the_chat_provider_maps_the_round_settings(self):
        thinking, plain = chat(**THINKING), chat()
        efforts = [thinking.request(payload(reasoning={"effort": effort}))["extra_body"] for effort in ("none", "low")]
        self.assertEqual([body["chat_template_kwargs"]["enable_thinking"] for body in efforts], [False, True])
        self.assertIs(thinking.endpoint.extra_params["extra_body"]["chat_template_kwargs"]["enable_thinking"], True)
        self.assertNotIn("extra_body", plain.request(payload(reasoning={"effort": "none"})))
        strict = {"name": "reply", "schema": {"type": "object"}, "strict": True}
        schema = {"type": "json_schema", "json_schema": strict}
        named = {"type": "function", "function": {"name": "get_menu"}}
        for setting, key, expected in (
            ({"text": {"format": {"type": "json_schema", **strict}}}, "response_format", schema),
            ({"text": {"format": {"type": "json_object"}}}, "response_format", {"type": "json_object"}),
            ({"tool_choice": {"type": "function", "name": "get_menu"}}, "tool_choice", named),
            # A hosted tool's choice becomes any of ours.
            ({"tool_choice": {"type": "web_search"}}, "tool_choice", "auto"),
            ({"tool_choice": "sometimes"}, "tool_choice", "auto"),
            ({"tool_choice": "required"}, "tool_choice", "required"),
        ):
            self.assertEqual(plain.request(payload(**setting))[key], expected)
        bare = plain.request(payload(text={"format": {"type": "text"}}, tools=[], tool_choice="none"))
        self.assertEqual([key in bare for key in ("response_format", "tools", "tool_choice")], [False, False, False])
        image = {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "low"}
        # Left out, as a chat endpoint cannot read it.
        by_file = {"type": "input_image", "file_id": "file_1"}
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


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_chat_round_sends_tools_and_history_and_gets_a_call(self):
        fragments = [tool_fragment(0, "call_1", "get_menu", '{"category":'), tool_fragment(0, arguments='"tea"}')]
        client = FakeChat([chunk(tool_calls=fragments[:1]), chunk(tool_calls=fragments[1:], finish="tool_calls")])
        tools = CAFE_TOOLS + [{"type": "web_search"}]
        request = {"model": "test/model", "instructions": "Use tools.", "tools": tools, "tool_choice": "auto"}
        events = await read_all(
            ChatCompletionsBackend(ENDPOINT, LIMITS, client).stream({**request, "input": [text_of("teas?")]})
        )
        request = client.requests[0]
        self.assertEqual((request["temperature"], request["max_tokens"], request["tool_choice"]), (0.7, 1024, "auto"))
        self.assertEqual([t["function"]["name"] for t in request["tools"]], ["get_menu", "manage_cart", "place_order"])
        self.assertEqual([m["role"] for m in request["messages"]], ["system", "user"])
        (call,) = [item for item in finished_items(events) if item["type"] == "function_call"]
        self.assertEqual((call["name"], json.loads(call["arguments"])), ("get_menu", {"category": "tea"}))
        self.assertEqual(events[-1]["type"], "response.completed")

    async def test_the_responses_provider_reads_text_and_call_rounds(self):
        for round_, item_type in ((text_round("Hello."), "message"), (call_round("get_menu", {}), "function_call")):
            with self.subTest(item_type):
                provider = OpenAIResponsesBackend(ENDPOINT, LIMITS, FakeResponses(round_))
                events = await read_all(provider.stream(payload()))
                self.assertEqual((kinds(events)[0], kinds(events)[-1]), ("response.created", "response.completed"))
                self.assertEqual([item["type"] for item in finished_items(events)], [item_type])

    async def test_only_a_failure_opening_the_stream_is_retried(self):
        request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
        overloaded = APIError("Service temporarily overloaded", request, body=None)
        for opening in (_timeout(), overloaded):
            backend = ChatCompletionsBackend(ENDPOINT, LIMITS, FakeChat(opening, [chunk("Hello", finish="stop")]))
            events = await read_all(backend.stream({"input": []}, retries=LIMITS.provider_retries))
            self.assertIn("response.output_text.delta", kinds(events))
        midway = ChatCompletionsBackend(ENDPOINT, LIMITS, FakeChat([chunk("Hel"), overloaded]))
        # Transient after the first chunk too: the run may try again.
        with self.assertRaises(StreamErrorEvent):
            await read_all(midway.stream({"input": []}, retries=LIMITS.provider_retries))
        rejected = BadRequestError("bad request", response=httpx.Response(400, request=request), body=None)
        backend = ChatCompletionsBackend(ENDPOINT, LIMITS, FakeChat(rejected, [chunk("Hello", finish="stop")]))
        with self.assertRaises(BadRequestError):
            await read_all(backend.stream({"input": []}, retries=LIMITS.provider_retries))
        # By default a round opens once: its caller decides whether to try it again.
        client = FakeChat(_timeout(), [chunk("Hello", finish="stop")])
        with self.assertRaises(APITimeoutError):
            await read_all(ChatCompletionsBackend(ENDPOINT, LIMITS, client).stream({"input": []}))
        self.assertEqual(len(client.requests), 1)

    async def test_a_silence_past_its_limit_ends_the_round(self):
        round_ = text_round("Done.")
        slow_first = ScriptedProvider([Sleep(1.0), *round_], round_, limits=QUICK)
        silent = ScriptedProvider([round_[0], Sleep(1.0), *round_[1:]], limits=QUICK)
        thinking = [chunk(None, reasoning_content="Let me"), Sleep(1.0), chunk("Done."), chunk(finish="stop")]
        stalled_chat = ChatCompletionsBackend(endpoint(**THINKING), QUICK, FakeChat(thinking))
        for provider in (slow_first, silent, stalled_chat):
            with self.subTest(provider=provider), self.assertRaises(TimeoutError):
                await read_all(provider.stream(payload(), retries=QUICK.provider_retries))
        self.assertEqual((slow_first.opened, silent.opened), (1, 1))

    async def test_a_steady_round_is_never_cut(self):
        # 0.4 s in all, but never 0.2 s without an event or a thinking model's reasoning chunk.
        steady = ScriptedProvider([step for event in text_round("Done.") for step in (Sleep(0.1), event)], limits=QUICK)
        pieces = ("Let me", " think", " it", " over.")
        thinking = [step for piece in pieces for step in (Sleep(0.1), chunk(None, reasoning_content=piece))]
        client = FakeChat([*thinking, chunk("A latte is $4.50."), chunk(finish="stop")])
        for provider in (steady, ChatCompletionsBackend(endpoint(**THINKING), QUICK, client)):
            events = await read_all(provider.stream(payload()))
            self.assertEqual(events[-1]["type"], "response.completed")
        [message] = finished_items(events)
        self.assertEqual(message["content"][0]["text"], "A latte is $4.50.")

    async def test_a_refused_request_is_never_retried(self):
        request = httpx.Request("POST", "http://localhost:1/v1/chat/completions")
        refused = BadRequestError("bad request", response=httpx.Response(400, request=request), body=None)
        provider = ScriptedProvider([refused], text_round("Done."), limits=QUICK)
        with self.assertRaises(BadRequestError):
            await read_all(provider.stream({}, retries=QUICK.provider_retries))
        self.assertEqual(provider.opened, 1)
        with self.assertRaises(BrokenStreamError):
            await read_all(ScriptedProvider([], text_round("Done."), limits=QUICK).stream({}, retries=1))
        provider = ScriptedProvider([*text_round("Done."), RuntimeError("after the terminal")], limits=QUICK)
        events = await read_all(provider.stream({}))
        self.assertEqual((events[-1]["type"], provider.read_after_terminal), ("response.completed", False))

    async def test_one_call_per_round_keeps_the_first_call(self):
        def call(index, name, arguments="{}"):
            return tool_fragment(index, f"p{index}", name, arguments)

        both = [chunk(tool_calls=[call(0, "get_menu"), call(1, "manage_cart")]), chunk(finish="tool_calls")]
        cut = [chunk(tool_calls=[call(0, "get_menu")]), chunk(tool_calls=[call(1, "manage_cart", '{"item": "te')])]
        # Never read: a provider that read on would hit its silence limit.
        cut += [Sleep(5), chunk(finish="length")]
        for script, parallel, names in (
            (both, True, ["get_menu", "manage_cart"]),
            (both, False, ["get_menu"]),
            (cut, False, ["get_menu"]),
        ):
            with self.subTest(parallel=parallel, steps=len(script)):
                client = FakeChat(list(script))
                provider = ChatCompletionsBackend(endpoint(), NO_RETRY, client)
                events = await read_all(provider.stream(payload(parallel_tool_calls=parallel)))
                names_given = [item["name"] for item in finished_items(events)]
                self.assertEqual((names_given, events[-1]["type"]), (names, "response.completed"))
                # Never sent, as an endpoint may refuse it.
                self.assertNotIn("parallel_tool_calls", client.requests[0])
                self.assertNotIn("p0", [item["call_id"] for item in finished_items(events)])

    async def test_a_cut_reply_never_gives_its_call(self):
        for finish, arguments, reason in (
            ("length", '{"item_id": "la', "max_output_tokens"),
            ("content_filter", '{"item_id": "lat"}', "content_filter"),
        ):
            with self.subTest(finish):
                cut = [
                    chunk("Adding"),
                    chunk(tool_calls=[tool_fragment(0, "c1", "manage_cart", arguments)]),
                    chunk(finish=finish),
                ]
                events = await read_all(chat(cut).stream(payload()))
                self.assertEqual([item["type"] for item in finished_items(events)], ["message"])
                self.assertEqual(events[-1]["response"]["incomplete_details"], {"reason": reason})
        with self.assertRaises(BrokenStreamError):
            await read_all(
                chat([chunk(tool_calls=[tool_fragment(0, "c1", "manage_cart", '{"item_id": ')])]).stream(payload())
            )
