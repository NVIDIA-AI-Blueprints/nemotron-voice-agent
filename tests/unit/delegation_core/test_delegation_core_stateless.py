# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D107

"""Stateless Responses requests end to end, each served by a one-shot IM session on the backend LLM."""

import asyncio
import gc
import unittest
import weakref
from dataclasses import replace

from delegation_core.api.local import LocalResponsesClient
from delegation_core.api.service import MAX_REQUEST_DEPTH, InvalidRequest, ResponsesService
from delegation_core.app import build_core
from delegation_core.backend.adapters import ADAPTERS
from delegation_core.backend.llm import LlmSettings, LlmSource
from delegation_core.backend.providers import ProviderLimits
from delegation_core.backend.providers.chat_completions import ChatCompletionsBackend
from delegation_core.config import CoreSettings, ModelEndpoint
from delegation_core.im.input import BackendTask, compose
from delegation_core.im.oneshot import OneShotSession
from tests.unit.contracts.backend import a_round, call_events, message_events
from tests.unit.core_helpers import LIMITS, FakeChat, answer_text, chunk
from tests.unit.delegation_core.backend.llm.doubles import ScriptedProvider, Sleep
from tests.unit.delegation_core.test_delegation_core_api import CORE_YAML

ENDPOINT = ModelEndpoint(model="org/test-model", base_url="http://localhost:1/v1")
DEFAULTS = BackendTask("org/default-model", "Default.", [])
USER = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What teas do you have?"}]}
CALL = {"type": "function_call", "call_id": "call_1", "name": "get_menu", "arguments": "{}"}
OUTPUT = {"type": "function_call_output", "call_id": "call_1", "output": '{"items": ["chai"]}'}
MENU = {"type": "function", "name": "get_menu", "parameters": {"type": "object", "properties": {}}}
TERMINALS = {"response.completed", "response.failed", "response.incomplete"}


def answer_round(text):
    """A backend round that answers ``text``."""
    return a_round(message_events(text))


def call_round():
    """A backend round that ends with one function call for the client."""
    return a_round(call_events("get_menu", {}, call_id="call_9"))


def empty_round():
    """A backend round that completes with no items."""
    return a_round()


class Model(ScriptedProvider):
    """A session's scripted provider (through the provider template, so its silence limits apply); it counts closes."""

    def __init__(self, *rounds, limits=LIMITS):
        super().__init__(*rounds, endpoint=ENDPOINT, limits=limits)
        self.closes = 0

    async def close(self):
        self.closes += 1


def scripted_llm(providers: list):
    """The adapter table, its backend LLM giving each session the next of ``providers`` when its first round starts."""

    class ScriptedLlm(LlmSource):
        @classmethod
        def from_settings(cls, block, *, endpoints, tools, vocabulary):
            settings = LlmSettings.parse(block)
            return cls(settings, endpoints=endpoints, tools=tools, make_provider=lambda _: providers.pop(0))

    return {**ADAPTERS, "llm": lambda: ScriptedLlm}


class StatelessCase(unittest.IsolatedAsyncioTestCase):
    def core(self, *models, one_shot=True):
        """A core from the shipped file that serves stateless requests, each on the next of ``models``."""
        settings = CoreSettings.load(CORE_YAML)
        # A reader that stops closes its session: running work is stopped after the close's grace, kept short here.
        settings = replace(settings, delegation=replace(settings.delegation, close_grace_seconds=0.01))
        core = build_core(
            settings,
            endpoints=lambda model: ModelEndpoint(model=model or ENDPOINT.model, base_url=ENDPOINT.base_url),
            defaults=DEFAULTS,
            adapters=scripted_llm(list(models)),
        )
        self.addAsyncCleanup(core.aclose)
        self.texts = core.settings.texts
        stateless = {"stateless_session": core.one_shot_session} if one_shot else {}
        self.service = ResponsesService(core.session, backend=core, **stateless)
        self.client = LocalResponsesClient(self.service)
        return self.client

    @staticmethod
    async def read(stream, seconds=2):
        async with asyncio.timeout(seconds):
            return [event async for event in stream]

    async def ask(self, request, seconds=2):
        """Send ``request`` (stateless unless it says otherwise) and read its whole response."""
        return await self.read(await self.client.create_response(request), seconds)

    def assert_ended(self, events, kind):
        kinds = [event["type"] for event in events]
        self.assertEqual(kinds[0], "response.created")
        self.assertEqual(events[0]["response"]["status"], "in_progress")  # a one-shot response opens in progress
        terminals = [k for k in kinds if k in TERMINALS]
        self.assertEqual((terminals, kinds[-1]), ([kind], kind))
        self.assertEqual([event["sequence_number"] for event in events], list(range(1, len(events) + 1)))


class StatelessRequestTests(StatelessCase):
    async def test_the_whole_input_reaches_the_model_as_sent_and_the_session_ends_with_the_response(self):
        model = Model(answer_round("We have chai."))
        self.core(model)
        stream = await self.client.create_response({"input": [USER, CALL, OUTPUT], "stream": True, "store": False})
        session = stream._stream.session  # the request's one-shot IM session
        self.assertIsInstance(session, OneShotSession)
        events = await self.read(stream)
        self.assert_ended(events, "response.completed")
        created, completed = events[0]["response"], events[-1]["response"]
        self.assertEqual(created["id"], completed["id"])
        self.assertTrue(completed["id"].startswith("resp_"))  # the core's response, not the round's
        self.assertEqual(answer_text(events[-1]), "We have chai.")
        # Never read as a continuation: the function output is part of the context, and the model is asked.
        self.assertEqual(model.payloads[0]["input"], [USER, CALL, OUTPUT])
        settings = (model.payloads[0]["model"], model.payloads[0]["instructions"])
        expected = ("org/default-model", compose("Default.", self.texts.voice_notes["v2"]))
        self.assertEqual(settings, expected)  # the IM's defaults fill what it left out; the voice note joins them
        # The response's end ended the session: the model is released, and no conversation was opened.
        self.assertEqual(model.closes, 1)
        self.assertEqual(session.ledger.session.state, "ended")
        self.assertEqual(self.service.sessions, {})

    async def test_what_the_client_set_and_the_voice_note_reach_the_model(self):
        model = Model(answer_round("Hello."))
        self.core(model)
        request = {
            "input": [USER],
            "model": "org/other",
            "instructions": "Be terse.",
            "tools": [MENU],
            "tool_choice": "auto",
        }
        events = await self.ask(request)
        payload = model.payloads[0]
        self.assertEqual((payload["model"], payload["tools"], payload["tool_choice"]), ("org/other", [MENU], "auto"))
        self.assertEqual(payload["instructions"], compose("Be terse.", self.texts.voice_notes["v2"]))
        self.assertEqual(events[-1]["response"]["model"], "org/other")

    async def test_function_calls_end_the_response_and_nothing_waits_for_their_outputs(self):
        model = Model(call_round())
        self.core(model)
        stream = await self.client.create_response({"input": [USER], "tools": [MENU]})
        session = stream._stream.session
        events = await self.read(stream)  # within 2 s, where the client-tool wait would take 60 s
        self.assert_ended(events, "response.completed")
        output = events[-1]["response"]["output"]
        self.assertEqual([item["type"] for item in output], ["function_call"])
        self.assertEqual([work.state for work in session.ledger.works.values()], ["answered"])  # never waiting_client
        self.assertEqual(model.closes, 1)

    async def test_a_failed_round_is_tried_once_and_ends_failed(self):
        model = Model([RuntimeError("model down")], answer_round("never"))
        self.core(model)
        events = await self.ask({"input": [USER]})
        self.assert_ended(events, "response.failed")
        self.assertEqual(model.opened, 1)  # one attempt
        self.assertEqual(model.closes, 1)
        self.assertNotIn("output_item", str([event["type"] for event in events]))  # no apology, no item at all

    async def test_a_round_that_ends_unfinished_or_without_a_terminal_ends_failed(self):
        unfinished = [
            {"type": "response.created", "response": {"id": "r", "status": "in_progress"}},
            {"type": "response.incomplete", "response": {"id": "r", "status": "incomplete"}},
        ]
        cut_short = [{"type": "response.created", "response": {"id": "r", "status": "in_progress"}}]
        for name, step in (("incomplete", unfinished), ("no terminal", cut_short)):
            with self.subTest(name):
                model = Model(step, answer_round("never"))
                self.core(model)
                self.assert_ended(await self.ask({"input": [USER]}), "response.failed")
                self.assertEqual(model.opened, 1)

    async def test_a_round_with_no_items_is_no_answer_and_ends_failed_without_a_retry(self):
        model = Model(empty_round(), answer_round("never"))
        self.core(model)
        events = await self.ask({"input": [USER]})
        self.assert_ended(events, "response.failed")  # an empty round is retried once, never in one-shot
        self.assertEqual(events[-1]["response"]["error"]["code"], "server_error")
        self.assertNotIn("response.output_item.done", [event["type"] for event in events])
        self.assertEqual(model.opened, 1)

    async def test_a_round_past_its_silence_limit_ends_failed_and_releases_the_model(self):
        quick = ProviderLimits(provider_retries=0, first_event_timeout_seconds=2.0, idle_timeout_seconds=0.2)
        model = Model(
            [{"type": "response.created", "response": {"id": "r", "status": "in_progress"}}, Sleep(30)], limits=quick
        )
        self.core(model)
        events = await self.ask({"input": [USER]})
        self.assert_ended(events, "response.failed")
        self.assertEqual((model.opened, model.closes), (1, 1))

    async def test_the_input_is_never_trimmed_or_chained(self):
        long = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x" * 120_000}]}
        first, second = Model(answer_round("Ok.")), Model(answer_round("Ok."))
        self.core(first, second)
        for model in (first, second):  # a second request is not chained to the first one's response either
            await self.ask({"input": [long, USER]})  # far over the core's context budget
            payload = model.payloads[0]
            self.assertEqual(payload["input"], [long, USER])
            self.assertNotIn("previous_response_id", payload)

    async def test_closing_the_stream_early_stops_the_work_and_releases_the_model(self):
        model = Model([{"type": "response.created", "response": {"id": "r", "status": "in_progress"}}, Sleep(30)])
        self.core(model)
        stream = await self.client.create_response({"input": [USER]})
        session = stream._stream.session
        self.assertEqual((await stream.__anext__())["type"], "response.created")
        await stream.aclose()  # the reader stops: over HTTP, the client's disconnect
        self.assertEqual(model.closes, 1)
        self.assertEqual([work.state for work in session.ledger.works.values()], ["cancelled"])
        rest = await self.read(stream)
        self.assertEqual((rest[-1]["type"], rest[-1]["response"]["status"]), ("response.incomplete", "cancelled"))
        await stream.aclose()  # safe twice
        self.assertEqual(model.closes, 1)

    async def test_a_previous_response_id_without_a_conversation_is_not_found(self):
        model = Model()
        self.core(model)
        with self.assertRaisesRegex(InvalidRequest, "the core stores no responses") as refused:
            await self.client.create_response({"input": [USER], "previous_response_id": "resp_1"})
        # The official 400: an invalid request whose code is previous_response_not_found.
        self.assertEqual(
            (refused.exception.code, refused.exception.param), ("previous_response_not_found", "previous_response_id")
        )
        self.assertEqual((model.payloads, model.closes), ([], 0))  # refused before any session was made


class MalformedItemTests(StatelessCase):
    """The client's items reach the provider as sent; one the provider cannot read fails its own round only."""

    def chat_core(self, chat):
        provider = ChatCompletionsBackend(ENDPOINT, LIMITS, chat)
        self.core(provider)

    async def test_a_malformed_item_ends_its_response_failed_and_never_hangs_or_stops_the_core(self):
        malformed = {
            "a function call without call_id": {"type": "function_call", "name": "get_menu", "arguments": "{}"},
            "a function call without name": {"type": "function_call", "call_id": "call_9", "arguments": "{}"},
            "an output without output": {"type": "function_call_output", "call_id": "call_1"},
            "an output without call_id": {"type": "function_call_output", "output": "{}"},
            "message content that is a number": {"type": "message", "role": "user", "content": 5},
        }
        for name, item in malformed.items():
            with self.subTest(name):
                chat = FakeChat([chunk("never"), chunk(finish="stop")])
                self.chat_core(chat)
                stream = await self.client.create_response({"input": [USER, item]})
                self.assert_ended(await self.read(stream), "response.failed")
                self.assertEqual(chat.requests, [])  # the provider could not build the model request
        # Items the provider does not know are left out of the model request.
        chat = FakeChat([chunk("We have chai."), chunk(finish="stop")])
        self.chat_core(chat)
        events = await self.ask({"input": [USER, {"type": "reasoning", "summary": []}, {"odd": True}]})
        self.assertEqual(answer_text(events[-1]), "We have chai.")
        self.assertEqual([message["role"] for message in chat.requests[0]["messages"]], ["system", "user"])

    async def test_an_item_that_is_not_an_object_is_refused_before_any_event(self):
        model = Model()
        self.core(model)
        with self.assertRaisesRegex(ValueError, "must be a list of input items"):
            await self.client.create_response({"input": [USER, "What teas?"]})
        self.assertEqual(model.closes, 0)

    async def test_a_request_nested_too_deeply_is_refused_before_any_session(self):
        # The IM copies each request, and Python's copy recurses: a deep one would fail inside the core instead.
        def nested(levels):
            value = "x"
            for _ in range(levels):
                value = [value]
            return value

        model = Model(answer_round("Ok."))
        self.core(model)
        # The request, its input and the item are 3 levels: 61 lists more make 64, the most a request may nest.
        events = await self.ask({"input": [{**USER, "meta": nested(MAX_REQUEST_DEPTH - 3)}]})
        self.assert_ended(events, "response.completed")
        too_deep = {
            "in its input": {"input": [{**USER, "meta": nested(MAX_REQUEST_DEPTH - 2)}]},
            "in its tools": {"input": [USER], "tools": [{**MENU, "parameters": {"enum": nested(MAX_REQUEST_DEPTH)}}]},
            "far deeper": {"input": [{**USER, "meta": nested(5000)}]},
        }
        for name, request in too_deep.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, "at most 64 levels deep"):
                await self.client.create_response(request)
        self.assertEqual((model.opened, model.closes), (1, 1))  # no session was made for the others


class Gated(Model):
    """A model whose round waits at ``barrier``: it finishes only once every round of the barrier has started."""

    def __init__(self, index, barrier):
        super().__init__(answer_round(f"Answer {index}."))
        self.barrier = barrier

    async def _events(self, payload):
        await self.barrier.wait()
        async for event in super()._events(payload):
            yield event


class LifetimeTests(StatelessCase):
    async def test_many_stateless_requests_run_at_once_each_in_a_session_of_its_own(self):
        count = 20
        barrier = asyncio.Barrier(count)
        models = [Gated(index, barrier) for index in range(count)]
        self.core(*models)
        streams = []
        for index in range(count):
            question = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": f"Q{index}"}]}
            streams.append(await self.client.create_response({"input": [question]}))
        answers = await asyncio.gather(*(self.read(stream) for stream in streams))  # one session would never pass
        self.assertEqual([answer_text(events[-1]) for events in answers], [f"Answer {i}." for i in range(count)])
        questions = [model.payloads[0]["input"][0]["content"][0]["text"] for model in models]
        self.assertEqual(questions, [f"Q{i}" for i in range(count)])
        self.assertEqual([model.closes for model in models], [1] * count)
        self.assertEqual(len({stream._stream.session.ledger.session.id for stream in streams}), count)
        self.assertEqual(self.service.sessions, {})

    async def test_no_record_of_a_request_outlives_its_response(self):
        # An answer, client calls, no items, a failure.
        self.core(
            Model(answer_round("Ok.")), Model(call_round()), Model(empty_round()), Model([RuntimeError("model down")])
        )
        sessions = []
        for _ in range(4):
            stream = await self.client.create_response({"input": [USER, CALL, OUTPUT]})
            sessions.append(weakref.ref(stream._stream.session))
            await self.read(stream)
            del stream
        await asyncio.sleep(0)  # asyncio's own callbacks for the last session's ended tasks run first
        gc.collect()
        self.assertEqual([session() for session in sessions], [None] * 4)  # every record went with its session
        self.assertEqual(self.service.sessions, {})

    async def test_a_stateless_request_leaves_the_cores_conversations_alone(self):
        conversation_model, stateless_model = Model(answer_round("Hello.")), Model(answer_round("Ok."))
        self.core(stateless_model, conversation_model)  # in the order the sessions' first rounds start
        conversation = (await self.client.create_conversation({"live_session_id": "live_1"}))["id"]
        self.addAsyncCleanup(self.client.delete_conversation, conversation)
        im = self.service.sessions[conversation]
        await self.ask({"input": [USER, CALL, OUTPUT], "metadata": {"live_delegation_id": "item_1"}})
        self.assertEqual((im.ledger.delegations, im.history), ({}, []))  # it saw nothing of the stateless request
        request = {"input": [USER], "conversation": conversation, "metadata": {"live_delegation_id": "item_1"}}
        self.assertEqual(answer_text((await self.ask(request))[-1]), "Hello.")
        self.assertEqual(stateless_model.payloads[0]["input"], [USER, CALL, OUTPUT])


if __name__ == "__main__":
    unittest.main()
