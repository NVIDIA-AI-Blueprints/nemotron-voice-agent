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

from loguru import logger

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
from examples.frontend_backend_live.delegation_core.im.session import DelegationError, IMSession
from examples.shared.delegation_contract import TERMINALS, ModelEndpoint
from tests.unit.core_helpers import FakeChat, Sleep, chunk
from tests.unit.delegation_core.helpers import (
    CORE_SETTINGS,
    LIMITS,
    MARK,
    MENU,
    TEXTS,
    ClosesWhenLetGo,
    ScriptedAdapter,
    ScriptedProvider,
    StartEach,
    answer,
    answer_text,
    call_round,
    marks,
    text_round,
    until,
    wait,
)

ENDPOINT = ModelEndpoint(model="org/test-model", base_url="http://localhost:1/v1")
DEFAULTS = BackendTask("org/default-model", "Default.", [])
CALL = {"type": "function_call", "call_id": "call_1", "name": "get_menu", "arguments": "{}"}
OUTPUT = {"type": "function_call_output", "call_id": "call_1", "output": '{"items": ["chai"]}'}
CREATED = {"type": "response.created", "response": {"id": "r", "status": "in_progress"}}
# The core stores no responses.
NOT_STORED = "previous_response_not_found"
NOTES = (TEXTS.caller_note.strip(), TEXTS.voice_note.strip())
STORED = {"live_session_id": "live_1", "live_store": "true"}


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
        cut_short = ResponsesService(lambda **hooks: im, stateless_session=lambda: im, backend=Rules())
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
        page = await self.client.list_conversation_items(second)
        self.assertEqual(page["data"][0]["content"], user("Two.")["content"])
        await self.client.delete_conversation(second)
        self.assertEqual((first_model.closed, second_model.closed), (0, 1))
        gate.set()
        self.assertEqual(answer_text([event async for event in held][-1]), "First.")
        self.assertEqual(len(first_model.payloads), 1)

    async def test_a_stored_conversation_with_no_work_after_the_call_is_forgotten_at_its_hang_up(self):
        self.core(Model(text_round("Hi.")))
        conversation = await self.open(STORED)
        self.assertTrue((await self.client.delete_conversation(conversation))["deleted"])
        with self.assertRaises(NotFound):
            await self.client.list_conversation_items(conversation)

    async def test_a_requests_messages_take_their_ids_from_its_delegation(self):
        self.core(Model(text_round("Green and black.")))
        conversation = await self.open()
        mine = {**user("Which teas?"), "id": "msg_mine"}
        await self.ask({**self.request(conversation), "input": [mine, user("Hot ones.")]})
        page = await self.client.list_conversation_items(conversation)
        self.assertEqual([item.get("id") for item in page["data"][:2]], ["item_1", "item_1.2"])

    async def test_each_request_kind_takes_its_admission(self):
        im = mock.Mock(ledger=SimpleNamespace(session=SimpleNamespace(id="conv_1")), history=[])
        service = ResponsesService(lambda **hooks: im, stateless_session=lambda: im, backend=Rules())
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


class KeptConversationTests(CoreCase):
    """Conversations whose work goes on after their call: the hang-up that keeps one, and a later call's attach."""

    def kept_core(self, *scripts, backend=ScriptedAdapter, **delegation):
        """Return the in-process client of a core whose conversations play ``scripts`` in turn, three tasks at once.

        Each request starts a Work of its own; ``backend`` is the scripted adapter's class.
        """
        settings = CORE_SETTINGS.delegation.model_copy(update={"close_grace_seconds": 0.01, **delegation})
        waiting = list(scripts)

        def new_backend():
            made = backend(conductor=StartEach(), max_running=3)
            made.scripts = waiting
            return made

        def session(**hooks):
            return IMSession(new_backend(), DEFAULTS, settings, TEXTS, renew=new_backend, **hooks)

        self.service = ResponsesService(session, stateless_session=session, backend=Rules())
        self.client = LocalResponsesClient(self.service, close_seconds=None, close_core=lambda: asyncio.sleep(0))
        self.addAsyncCleanup(self.client.aclose)
        return self.client

    async def marked(self, metadata=STORED) -> tuple[str, IMSession]:
        """Open a conversation whose first request's task marks its Work; return it once marked, with its session."""
        conversation = await self.open(metadata)
        await self.client.create_response(self.request(conversation, "Compare every supplier."))
        session = self.service.sessions[conversation]
        await until(lambda: marks(session), "the mark")
        return conversation, session

    async def test_a_kept_conversation_is_left_and_attached_by_a_later_call(self):
        gate = asyncio.Event()
        self.kept_core([MARK, wait(gate), answer("The comparison is ready.")])
        conversation, session = await self.marked()
        deleted = await self.client.delete_conversation(conversation)
        self.assertEqual(deleted, {"id": conversation, "object": "conversation.deleted", "deleted": False})
        self.assertEqual((await self.client.list_conversation_items(conversation))["object"], "list")
        attached = await self.client.update_conversation(conversation, {**STORED, "live_session_id": "live_2"})
        self.assertEqual((attached["id"], attached["metadata"]["live_session_id"]), (conversation, "live_2"))

        async def refusal(target: str, kind: type[Exception]) -> str:
            with self.assertRaises(kind) as refused:
                await self.client.update_conversation(target, STORED)
            return str(refused.exception)

        # One call holds it, an id names none, and one has ended: no refusal names the conversation.
        refusals = [await refusal(conversation, DelegationError), await refusal("conv_unknown", NotFound)]
        gate.set()
        await session.close()
        refusals.append(await refusal(conversation, NotFound))
        self.assertEqual([text for text in refusals if "conv_" in text], [])

    async def test_a_delete_cancelled_while_it_leaves_keeps_the_conversation(self):
        gate, other = asyncio.Event(), asyncio.Event()
        scripts = ([MARK, wait(gate), answer("The comparison is ready.")], [wait(other), answer("Free all day.")])
        self.kept_core(*scripts, close_grace_seconds=5)
        conversation, session = await self.marked()
        calendar = {"live_delegation_id": "item_2"}
        await self.client.create_response(self.request(conversation, "What is on my calendar?", metadata=calendar))
        await until(lambda: "W2" in session.works, "the second Work")
        deleting = asyncio.create_task(self.client.delete_conversation(conversation))
        await until(lambda: session.detached, "the hang-up")
        deleting.cancel()
        await asyncio.gather(deleting, return_exceptions=True)
        # The second Work ends in the hang-up's close grace; the hang-up goes on to its end.
        other.set()
        self.assertEqual((deleting.cancelled(), await session.leave()), (True, True))
        self.assertEqual((await self.client.list_conversation_items(conversation))["object"], "list")
        await self.client.update_conversation(conversation, STORED)
        self.assertEqual(session.ledger.session.state, "active")
        gate.set()

    async def test_work_after_the_call_is_refused_beyond_the_conversations_allowed(self):
        gate = asyncio.Event()
        self.kept_core(*([MARK, wait(gate), answer("Done.")] for _ in range(2)), max_after_call_conversations=1)
        sessions = [(await self.marked({**STORED, "live_session_id": f"live_{n}"}))[1] for n in (1, 2)]
        self.assertEqual(
            [marks(session) for session in sessions], [[TEXTS.continued_output], [TEXTS.not_continued_output]]
        )
        gate.set()

    async def test_a_failed_close_at_the_keep_times_end_is_logged(self):
        class FailsToClose(ScriptedAdapter):
            async def close(self):
                raise RuntimeError("the backend did not close")

        logged = []
        self.addCleanup(logger.remove, logger.add(logged.append, level="WARNING", format="{message}"))
        self.kept_core([MARK, wait(asyncio.Event()), answer("Never.")], backend=FailsToClose, after_call_seconds=0.05)
        conversation, _ = await self.marked()
        self.assertFalse((await self.client.delete_conversation(conversation))["deleted"])
        await until(lambda: [line for line in logged if "in the background" in line], "the keep time's close")
        failed = [line for line in logged if "in the background" in line]
        self.assertEqual(len(failed), 1)
        with self.assertRaises(NotFound):
            await self.client.list_conversation_items(conversation)

    async def test_an_attach_while_the_keep_times_close_runs_is_not_found(self):
        job = [MARK, wait(asyncio.Event()), answer("Never.")]
        self.kept_core(job, backend=ClosesWhenLetGo, after_call_seconds=0.05)
        # Before the client's own cleanup, which waits for the close.
        self.addCleanup(lambda: [each.backend.may_close.set() for each in self.service.sessions.values()])
        conversation, session = await self.marked()
        self.assertFalse((await self.client.delete_conversation(conversation))["deleted"])
        await until(lambda: session.closed, "the keep time's end")
        # The close waits for the backend: the conversation is still listed, and has ended.
        self.assertEqual((await self.client.list_conversation_items(conversation))["object"], "list")
        with self.assertRaises(NotFound):
            await self.client.update_conversation(conversation, STORED)
