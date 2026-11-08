# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The delegation core's Responses API end to end, through its in-process client, on a core of the shipped settings."""

import asyncio
import contextlib
import unittest
from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from unittest import mock

from examples.frontend_backend_live.delegation_core.api.local import LocalResponsesClient
from examples.frontend_backend_live.delegation_core.api.service import (
    MAX_REQUEST_DEPTH,
    InvalidRequest,
    NotFound,
    ResponsesService,
)
from examples.frontend_backend_live.delegation_core.app import build_core
from examples.frontend_backend_live.delegation_core.backend.adapters import ADAPTERS
from examples.frontend_backend_live.delegation_core.backend.llm import LlmSettings, LlmSource
from examples.frontend_backend_live.delegation_core.backend.providers.chat_completions import ChatCompletionsBackend
from examples.frontend_backend_live.delegation_core.im.input import BackendTask
from examples.shared.delegation_contract import TERMINALS, ModelEndpoint
from tests.unit.core_helpers import FakeChat, Sleep, chunk
from tests.unit.delegation_core.helpers import (
    CORE_SETTINGS,
    LIMITS,
    MENU,
    TEXTS,
    ScriptedProvider,
    answer_text,
    call_round,
    text_round,
)

ENDPOINT = ModelEndpoint(model="org/test-model", base_url="http://localhost:1/v1")
DEFAULTS = BackendTask("org/default-model", "Default.", [])
CALL = {"type": "function_call", "call_id": "call_1", "name": "get_menu", "arguments": "{}"}
OUTPUT = {"type": "function_call_output", "call_id": "call_1", "output": '{"items": ["chai"]}'}
CREATED = {"type": "response.created", "response": {"id": "r", "status": "in_progress"}}
# The core stores no responses.
NOT_STORED = "previous_response_not_found"
NOTES = (TEXTS.caller_note.strip(), TEXTS.voice_note.strip())


def user(text: str) -> dict:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


USER = user("What teas do you have?")


def nested(levels: int):
    value = "x"
    for _ in range(levels):
        value = [value]
    return value


Model = partial(ScriptedProvider, endpoint=ENDPOINT, limits=LIMITS)


class Rules:
    """A backend's rules: it serves org/a only, and cannot honor a service tier or a verbosity."""

    UNSUPPORTED = {"service_tier": "Only the default tier.", "text": "No verbosity."}

    async def check_model(self, model):
        if model != "org/a":
            raise ValueError(f"The backend serves org/a; not {model!r}")

    def unsupported(self, settings):
        return [(param, self.UNSUPPORTED[param]) for param in settings if param in self.UNSUPPORTED]


class CoreCase(unittest.IsolatedAsyncioTestCase):
    def core(self, *models, rules=None, **delegation):
        """Return the in-process client of a core of the shipped settings whose sessions run on ``models`` in turn."""
        settings, providers = CORE_SETTINGS, list(models)
        delegation = settings.delegation.model_copy(update={"close_grace_seconds": 0.01, **delegation})

        class ScriptedLlm(LlmSource):
            @classmethod
            def from_settings(cls, block, *, endpoints, tools, vocabulary):
                make = lambda endpoint: providers.pop(0)  # noqa: E731
                return cls(LlmSettings.model_validate(block), endpoints=endpoints, tools=tools, make_provider=make)

        endpoints = lambda model: replace(ENDPOINT, model=model or ENDPOINT.model)  # noqa: E731
        settings = settings.model_copy(update={"delegation": delegation})
        adapters = {**ADAPTERS, "llm": lambda: ScriptedLlm}
        core = build_core(settings, endpoints=endpoints, defaults=DEFAULTS, adapters=adapters)
        self.addAsyncCleanup(core.aclose)
        self.service = ResponsesService(core.session, stateless_session=core.one_shot_session, backend=rules or core)
        self.client = LocalResponsesClient(self.service, close_seconds=core.close_seconds, close_core=core.aclose)
        return self.client

    async def open(self, metadata=None) -> str:
        conversation = (await self.client.create_conversation(metadata or {"live_session_id": "live_1"}))["id"]

        async def delete():
            with contextlib.suppress(NotFound):
                await self.client.delete_conversation(conversation)

        self.addAsyncCleanup(delete)
        return conversation

    async def ask(self, request: dict) -> list[dict]:
        stream = await self.client.create_response(request)
        async with asyncio.timeout(2):
            return [event async for event in stream]

    @staticmethod
    def request(conversation, text="Hi.", **fields) -> dict:
        metadata = fields.pop("metadata", {"live_delegation_id": "item_1"})
        return {"input": [user(text)], "conversation": conversation, "metadata": metadata, "stream": True, **fields}


class ConversationTests(CoreCase):
    async def test_a_response_opens_queued_and_ends_once_in_order(self):
        gate = asyncio.Event()
        model = Model([gate, *text_round("We open at nine.")])
        self.core(model)
        conversation = await self.open()
        with self.assertRaises(NotFound):
            await self.client.create_response(self.request("conv_unknown"))
        request = self.request({"id": conversation}, "When do you open?")
        stream = await self.client.create_response(request)
        # What crosses the client is a copy.
        request["input"][0]["content"][0]["text"] = "changed after it was sent"
        gate.set()
        async with asyncio.timeout(2):
            events = [event async for event in stream]
        kinds = [event["type"] for event in events]
        self.assertEqual(kinds[:3], ["response.created", "response.queued", "response.in_progress"])
        self.assertEqual([kind for kind in kinds if kind in TERMINALS], ["response.completed"])
        self.assertEqual([event["sequence_number"] for event in events], list(range(1, len(events) + 1)))
        created, completed = events[0]["response"], events[-1]["response"]
        self.assertEqual((created["status"], created["id"]), ("queued", completed["id"]))
        self.assertEqual((kinds[-1], answer_text(events[-1])), ("response.completed", "We open at nine."))
        self.assertEqual(model.payloads[0]["model"], DEFAULTS.model)
        page = await self.client.list_conversation_items(conversation)
        self.assertEqual(sorted(page), ["data", "first_id", "has_more", "last_id", "object"])
        self.assertEqual((page["object"], page["has_more"]), ("list", False))
        self.assertEqual([item.get("role") for item in page["data"]], ["user", "assistant"])
        self.assertEqual(page["data"][0]["content"][0]["text"], "When do you open?")
        self.assertEqual(model.payloads[0]["input"][0]["content"][0]["text"], "When do you open?")

    async def test_a_delegation_states_what_was_heard_in_metadata_and_a_bad_value_states_nothing(self):
        self.core(Model(text_round("Done.")), Model(text_round("Done.")), Model(text_round("Done.")))
        conversation = await self.open()
        im = self.service._session(conversation)
        cases = (
            ("item_1", {"live_reply_unheard": "question"}, "question"),
            ("item_2", {"live_reply_unheard": "maybe"}, None),
        )
        for delegation, stated, level in cases:
            with self.subTest(stated=stated):
                await self.ask(self.request(conversation, metadata={"live_delegation_id": delegation, **stated}))
                heard = im.heard(delegation)
                self.assertEqual(heard.level.value if heard is not None else None, level)
        # A request of the client's own names no delegation, so it states no fact.
        await self.ask(self.request(conversation, metadata={"live_reply_unheard": "question"}))
        self.assertTrue(all(incoming.brings is None or incoming.heard is None for incoming in im.incoming.values()))

    async def test_a_deleted_conversation_stops_and_is_gone(self):
        # The round still runs when the conversation is deleted.
        model = Model([asyncio.Event(), *text_round("Done.")])
        self.core(model, close_grace_seconds=0.2)
        conversation = await self.open()
        stream = await self.client.create_response(self.request(conversation))
        async with asyncio.timeout(2):
            while not model.payloads:
                await asyncio.sleep(0.005)
        deleting = asyncio.create_task(self.client.delete_conversation(conversation))
        await asyncio.sleep(0.05)
        self.assertEqual((await self.client.list_conversation_items(conversation))["object"], "list")
        self.assertEqual(await deleting, {"id": conversation, "object": "conversation.deleted", "deleted": True})
        ended = [event async for event in stream][-1]
        self.assertEqual((ended["type"], ended["response"]["status"]), ("response.incomplete", "cancelled"))
        self.assertEqual(model.closed, 1)
        for call in (
            self.client.delete_conversation,
            self.client.list_conversation_items,
            lambda conversation: self.client.create_response(self.request(conversation)),
        ):
            with self.subTest(call), self.assertRaises(NotFound):
                await call(conversation)
        ledger = SimpleNamespace(session=SimpleNamespace(id="conv_1"))
        im = mock.Mock(ledger=ledger, history=[], close=mock.AsyncMock(side_effect=RuntimeError("cut short")))
        cut_short = ResponsesService(lambda: im, stateless_session=lambda: im, backend=Rules())
        conversation = (await cut_short.create_conversation())["id"]
        with self.assertRaisesRegex(RuntimeError, "cut short"):
            await cut_short.delete_conversation(conversation)
        with self.assertRaises(NotFound):
            await cut_short.list_conversation_items(conversation)

    async def test_two_conversations_run_apart(self):
        gate = asyncio.Event()
        first_model, second_model = Model([gate, *text_round("First.")]), Model(text_round("Second."))
        self.core(first_model, second_model)
        first, second = await self.open(), await self.open({"live_session_id": "live_2"})
        held = await self.client.create_response(self.request(first, "One."))
        self.assertEqual(answer_text((await self.ask(self.request(second, "Two.")))[-1]), "Second.")
        self.assertEqual((await self.client.list_conversation_items(second))["data"][0], user("Two."))
        await self.client.delete_conversation(second)
        self.assertEqual((first_model.closed, second_model.closed), (0, 1))
        gate.set()
        self.assertEqual(answer_text([event async for event in held][-1]), "First.")
        self.assertEqual(len(first_model.payloads), 1)

    async def test_each_request_kind_takes_its_admission(self):
        im = mock.Mock(ledger=SimpleNamespace(session=SimpleNamespace(id="conv_1")), history=[])
        service = ResponsesService(lambda: im, stateless_session=lambda: im, backend=Rules())
        conversation = (await service.create_conversation())["id"]
        typed = {"live_delegation_id": "item_3", "live_typed_run": "true"}
        for request in (
            self.request(conversation),
            {**self.request(conversation), "input": [OUTPUT]},
            self.request(conversation, "When do you open?", metadata=typed),
            self.request(conversation, "Untagged.", metadata={}),
        ):
            await service.create_response(request)
        admitted = [(name, kwargs["delegation_id"]) for name, _, kwargs in im.mock_calls if name.startswith("admit")]
        client = "admit_client_request"
        self.assertEqual(admitted, [("admit", "item_1"), (client, "item_1"), (client, "item_3"), (client, None)])

    async def test_a_malformed_request_is_refused_before_any_event(self):
        model = Model(text_round("Ok."))
        self.core(model)
        conversation = await self.open()
        base, deep = self.request(conversation), nested(MAX_REQUEST_DEPTH)
        cases = (
            ({**base, "input": "not a list"}, "input", None),
            ({**base, "input": [USER, "What teas?"]}, "input", None),
            ({**base, "metadata": ["item_1"]}, "metadata", None),
            ({**base, "previous_response_id": "resp_1"}, "previous_response_id", None),
            ({"input": [USER], "previous_response_id": "resp_1"}, "previous_response_id", NOT_STORED),
            ({**base, "meta": deep}, None, None),
            ({"input": [{**USER, "meta": nested(MAX_REQUEST_DEPTH - 2)}]}, None, None),
            ({"input": [USER], "tools": [{**MENU, "parameters": {"enum": deep}}]}, None, None),
            ({"input": [{**USER, "meta": nested(5000)}]}, None, None),
        )
        for request, param, code in cases:
            with self.subTest(param=param, code=code), self.assertRaises(InvalidRequest) as refused:
                await self.client.create_response(request)
            self.assertEqual((refused.exception.param, refused.exception.code), (param, code))
        self.assertIsInstance(refused.exception, ValueError)
        self.assertIn("at most 64 levels deep", str(refused.exception))
        # The request, its input and the item are 3 levels: 61 lists more make 64, the most a request may nest.
        deepest = await self.ask({"input": [{**USER, "meta": nested(MAX_REQUEST_DEPTH - 3)}]})
        self.assertEqual((deepest[-1]["type"], model.opened), ("response.completed", 1))
        self.assertEqual((await self.client.list_conversation_items(conversation))["data"], [])

    async def test_input_tokens_and_the_model_check_admit_nothing(self):
        self.core(Model(text_round("Done.")), rules=Rules())
        served = await self.client.retrieve_model("org/a")
        self.assertEqual((served["id"], served["object"], served["owned_by"]), ("org/a", "model", "delegation_core"))
        conversation = await self.open()
        with self.assertRaisesRegex(ValueError, "not 'org/b'"):
            await self.client.retrieve_model("org/b")
        await self.ask(self.request(conversation, "Find teas."))
        counted = await self.client.input_tokens(self.request(conversation, instructions="Be brief."))
        self.assertEqual((sorted(counted), counted["object"]), (["input_tokens", "object"], "response.input_tokens"))
        alone = await self.client.input_tokens({"input": [user("Hi.")], "instructions": "Be brief."})
        self.assertGreater(counted["input_tokens"], alone["input_tokens"])
        self.assertGreater((await self.client.input_tokens({"model": "org/a"}))["input_tokens"], 0)
        self.assertEqual(len((await self.client.list_conversation_items(conversation))["data"]), 2)
        settings = {"model": "org/a", "service_tier": "priority", "text": {"verbosity": "low"}}
        with self.assertRaises(InvalidRequest) as refused:
            await self.client.input_tokens({"input": [], **settings})
        self.assertEqual(refused.exception.param, "text")
        self.assertIn("text: No verbosity.", str(refused.exception))
        self.assertIn("service_tier: Only the default tier.", str(refused.exception))
        for request, refusal, param in (
            ({"input": [], "model": "org/b"}, InvalidRequest, "model"),
            ({"input": "Hi."}, InvalidRequest, "input"),
            ({"input": [user("Hi.")], "meta": nested(MAX_REQUEST_DEPTH)}, InvalidRequest, None),
            (self.request("conv_unknown"), NotFound, None),
        ):
            with self.subTest(param=param, refusal=refusal.__name__), self.assertRaises(refusal) as refused:
                await self.client.input_tokens(request)
            self.assertEqual(getattr(refused.exception, "param", None), param)


class StatelessTests(CoreCase):
    def assertEnded(self, events: list[dict], kind: str) -> None:
        kinds = [event["type"] for event in events]
        self.assertEqual((kinds[0], events[0]["response"]["status"]), ("response.created", "in_progress"))
        self.assertEqual(([k for k in kinds if k in TERMINALS], kinds[-1]), ([kind], kind))
        self.assertEqual([event["sequence_number"] for event in events], list(range(1, len(events) + 1)))

    async def test_the_input_and_settings_reach_the_model_as_sent(self):
        first = Model(text_round("We have chai."))
        second = Model(call_round("get_menu", {}, call_id="call_9", said="Checking the menu."))
        self.core(first, second)
        # Far over the core's context budget, and never trimmed.
        sent = [user("x" * 120_000), USER, CALL, OUTPUT]
        events = await self.ask({"input": sent, "stream": True, "store": False})
        self.assertEnded(events, "response.completed")
        created, completed = events[0]["response"], events[-1]["response"]
        self.assertEqual((created["id"], answer_text(events[-1])), (completed["id"], "We have chai."))
        self.assertNotEqual(completed["id"], "resp_1")
        payload, composed = first.payloads[0], "\n".join(["Default.", *NOTES])
        self.assertEqual(payload["input"], sent)
        self.assertEqual((payload["model"], payload["instructions"]), ("org/default-model", composed))
        self.assertEqual(first.closed, 1)
        settings = {"model": "org/other", "instructions": "Be terse.", "tools": [MENU], "tool_choice": "auto"}
        # Within 2 s, where the client-tool wait is 60 s.
        events = await self.ask({"input": [USER], **settings})
        payload = second.payloads[0]
        self.assertEqual((payload["model"], payload["tools"], payload["tool_choice"]), ("org/other", [MENU], "auto"))
        self.assertEqual(payload["instructions"], "\n".join(["Be terse.", *NOTES]))
        self.assertEqual((events[-1]["response"]["model"], "previous_response_id" in payload), ("org/other", False))
        self.assertEnded(events, "response.completed")
        said, call = events[-1]["response"]["output"]
        self.assertEqual((said["phase"], said["content"][0]["text"]), ("commentary", "Checking the menu."))
        self.assertEqual((call["type"], second.closed), ("function_call", 1))

    async def test_closing_the_stream_early_releases_the_model(self):
        model = Model([CREATED, Sleep(30.0)])
        self.core(model)
        stream = await self.client.create_response({"input": [USER]})
        self.assertEqual((await anext(stream))["type"], "response.created")
        # The reader stops, as a client's disconnect does over HTTP.
        await stream.aclose()
        self.assertEqual(model.closed, 1)
        ended = [event async for event in stream][-1]
        self.assertEqual((ended["type"], ended["response"]["status"]), ("response.incomplete", "cancelled"))
        await stream.aclose()
        self.assertEqual(model.closed, 1)

    async def test_a_malformed_item_fails_its_own_response(self):
        for item in (
            {"type": "function_call", "name": "get_menu", "arguments": "{}"},
            {"type": "function_call", "call_id": "call_9", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1"},
            {"type": "function_call_output", "output": "{}"},
            {"type": "message", "role": "user", "content": 5},
        ):
            with self.subTest(item):
                chat = FakeChat([chunk("never"), chunk(finish="stop")])
                self.core(ChatCompletionsBackend(ENDPOINT, LIMITS, chat))
                self.assertEnded(await self.ask({"input": [USER, item]}), "response.failed")
                self.assertEqual(chat.requests, [])
        chat = FakeChat([chunk("We have chai."), chunk(finish="stop")])
        self.core(ChatCompletionsBackend(ENDPOINT, LIMITS, chat))
        events = await self.ask({"input": [USER, {"type": "reasoning", "summary": []}, {"odd": True}]})
        self.assertEqual(answer_text(events[-1]), "We have chai.")
        self.assertEqual([message["role"] for message in chat.requests[0]["messages"]], ["system", "user"])

    async def test_stateless_requests_run_at_once_in_own_sessions(self):
        gate = asyncio.Event()
        models = [Model([gate, *text_round(f"Answer {n}.")]) for n in range(10)]
        self.core(*models)
        streams = [await self.client.create_response({"input": [user(f"Q{n}")]}) for n in range(10)]
        async with asyncio.timeout(2):
            while not all(model.opened for model in models):
                await asyncio.sleep(0.005)
        gate.set()
        answers = [answer_text([event async for event in stream][-1]) for stream in streams]
        self.assertEqual(answers, [f"Answer {n}." for n in range(10)])
        self.assertEqual([model.payloads[0]["input"] for model in models], [[user(f"Q{n}")] for n in range(10)])
        self.assertEqual([model.closed for model in models], [1] * 10)

    async def test_a_stateless_request_leaves_conversations_alone(self):
        conversation_model, stateless_model = Model(text_round("Hello.")), Model(text_round("Ok."))
        # In the order the sessions' first rounds start.
        self.core(stateless_model, conversation_model)
        conversation = await self.open()
        await self.ask({"input": [USER, CALL, OUTPUT], "metadata": {"live_delegation_id": "item_1"}})
        self.assertEqual((await self.client.list_conversation_items(conversation))["data"], [])
        answered = await self.ask(self.request(conversation))
        self.assertEqual(answer_text(answered[-1]), "Hello.")
        self.assertEqual(stateless_model.payloads[0]["input"], [USER, CALL, OUTPUT])
