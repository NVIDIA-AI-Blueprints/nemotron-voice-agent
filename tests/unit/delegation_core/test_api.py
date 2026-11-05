# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The delegation core's Responses API end to end, through its in-process client, on a core built from core.yaml."""

import asyncio
import contextlib
import json
import unittest
from dataclasses import replace
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
from examples.frontend_backend_live.delegation_core.backend.providers import Backend, ProviderLimits
from examples.frontend_backend_live.delegation_core.backend.providers.chat_completions import ChatCompletionsBackend
from examples.frontend_backend_live.delegation_core.config import CoreSettings
from examples.frontend_backend_live.delegation_core.im.input import BackendTask
from examples.shared.delegation_contract import ModelEndpoint
from tests.unit.contracts.backend import a_round, call_round, text_round
from tests.unit.contracts.responses import ResponsesClientContract
from tests.unit.core_helpers import CORE_TEXTS, LIMITS, FakeChat, answer_text, chunk
from tests.unit.delegation_core.helpers import CORE_YAML, MENU, TEXTS

ENDPOINT = ModelEndpoint(model="org/test-model", base_url="http://localhost:1/v1")
DEFAULTS = BackendTask("org/default-model", "Default.", [])
CALL = {"type": "function_call", "call_id": "call_1", "name": "get_menu", "arguments": "{}"}
OUTPUT = {"type": "function_call_output", "call_id": "call_1", "output": '{"items": ["chai"]}'}
CREATED = {"type": "response.created", "response": {"id": "r", "status": "in_progress"}}
TERMINALS = {"response.completed", "response.failed", "response.incomplete"}
NOT_STORED = "previous_response_not_found"  # the core stores no responses
NOTES = (TEXTS.caller_notes["default"].strip(), TEXTS.voice_notes["default"].strip())


def user(text: str) -> dict:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


USER = user("What teas do you have?")


def nested(levels: int):
    value = "x"
    for _ in range(levels):
        value = [value]
    return value


def session(**members) -> mock.Mock:
    """An IM session as the API calls it; it records every call made to it."""
    return mock.Mock(ledger=SimpleNamespace(session=SimpleNamespace(id="conv_1")), history=[], **members)


class Model(Backend):
    """A session's model provider: one script per round of events, exceptions, gates and seconds of silence."""

    def __init__(self, *rounds, limits: ProviderLimits = LIMITS):
        super().__init__(ENDPOINT, limits)
        self.rounds, self.payloads, self.opened, self.closes = list(rounds), [], 0, 0

    async def _events(self, payload):
        self.payloads.append(json.loads(json.dumps(payload)))
        self.opened += 1
        for step in self.rounds.pop(0):
            if isinstance(step, Exception):
                raise step
            if isinstance(step, asyncio.Event):
                await step.wait()
            elif isinstance(step, float):
                await asyncio.sleep(step)
            else:
                yield step

    async def close(self):
        self.closes += 1


class Rules:
    """A backend's rules: it serves org/a only, and cannot honor a service tier or a verbosity."""

    UNSUPPORTED = {"service_tier": "Only the default tier.", "text": "No verbosity."}

    async def check_model(self, model):
        if model != "org/a":
            raise ValueError(f"The backend serves org/a; not {model!r}")

    def unsupported(self, settings):
        return [(param, self.UNSUPPORTED[param]) for param in settings if param in self.UNSUPPORTED]


class CoreCase(unittest.IsolatedAsyncioTestCase):
    def core(self, *models, rules=None, stateless=True, **delegation):
        """A core from core.yaml whose IM sessions run on ``models`` in turn, reached through its in-process client."""
        settings, providers = CoreSettings.load(CORE_YAML, CORE_TEXTS), list(models)
        delegation = replace(settings.delegation, **{"close_grace_seconds": 0.01, **delegation})

        class ScriptedLlm(LlmSource):
            @classmethod
            def from_settings(cls, block, *, endpoints, tools, vocabulary):
                make = lambda endpoint: providers.pop(0)  # noqa: E731
                return cls(LlmSettings.parse(block), endpoints=endpoints, tools=tools, make_provider=make)

        endpoints = lambda model: replace(ENDPOINT, model=model or ENDPOINT.model)  # noqa: E731
        settings, adapters = replace(settings, delegation=delegation), {**ADAPTERS, "llm": lambda: ScriptedLlm}
        core = build_core(settings, endpoints=endpoints, defaults=DEFAULTS, adapters=adapters)
        self.addAsyncCleanup(core.aclose)
        one_shot = core.one_shot_session if stateless else None
        self.service = ResponsesService(core.session, stateless_session=one_shot, backend=rules or core)
        self.client = LocalResponsesClient(self.service)
        return self.client

    async def open(self, metadata=None) -> str:
        """A new conversation's id; it is deleted when the test ends."""
        conversation = (await self.client.create_conversation(metadata or {"live_session_id": "live_1"}))["id"]

        async def delete():
            with contextlib.suppress(NotFound):
                await self.client.delete_conversation(conversation)

        self.addAsyncCleanup(delete)
        return conversation

    async def ask(self, request: dict) -> list[dict]:
        """Send ``request`` and read its whole response."""
        stream = await self.client.create_response(request)
        async with asyncio.timeout(2):
            return [event async for event in stream]

    @staticmethod
    def request(conversation, text="Hi.", **fields) -> dict:
        metadata = fields.pop("metadata", {"live_delegation_id": "item_1"})
        return {"input": [user(text)], "conversation": conversation, "metadata": metadata, "stream": True, **fields}


class LocalResponsesClientContractTests(ResponsesClientContract, CoreCase):
    async def make_client(self):
        return self.core(Model(text_round("Done.")))


class ConversationTests(CoreCase):
    async def test_closing_the_client_runs_the_close_its_root_handed_it(self):
        close_core = mock.AsyncMock()
        self.core()
        await LocalResponsesClient(self.service, close_core=close_core).aclose()
        await self.client.aclose()  # handed none: nothing to release
        close_core.assert_awaited_once()

    async def test_a_response_opens_queued_runs_and_ends_once_numbered_from_one(self):
        model = Model(text_round("Done."))
        self.core(model)
        conversation = await self.open()
        with self.assertRaises(NotFound):
            await self.client.create_response(self.request("conv_unknown"))
        events = await self.ask(self.request({"id": conversation}))  # named by its id, or as an object
        kinds = [event["type"] for event in events]
        self.assertEqual(kinds[:3], ["response.created", "response.queued", "response.in_progress"])
        self.assertEqual([kind for kind in kinds if kind in TERMINALS], ["response.completed"])
        self.assertEqual([event["sequence_number"] for event in events], list(range(1, len(events) + 1)))
        created, completed = events[0]["response"], events[-1]["response"]
        self.assertEqual((created["status"], created["id"]), ("queued", completed["id"]))
        self.assertEqual((kinds[-1], answer_text(events[-1])), ("response.completed", "Done."))
        self.assertEqual(model.payloads[0]["model"], DEFAULTS.model)  # a request that sets nothing: the defaults

    async def test_the_items_are_the_conversations_history_and_what_crosses_the_client_is_a_copy(self):
        gate = asyncio.Event()
        model = Model([gate, *text_round("We open at nine.")])
        self.core(model)
        conversation = await self.open()
        request = self.request(conversation, "When do you open?")
        stream = await self.client.create_response(request)
        request["input"][0]["content"][0]["text"] = "changed after it was sent"
        gate.set()
        [event async for event in stream]
        page = await self.client.list_conversation_items(conversation)
        self.assertEqual(sorted(page), ["data", "first_id", "has_more", "last_id", "object"])
        self.assertEqual((page["object"], page["has_more"]), ("list", False))
        self.assertEqual([item.get("role") for item in page["data"]], ["user", "assistant"])
        self.assertEqual(page["data"][0]["content"][0]["text"], "When do you open?")
        self.assertEqual(model.payloads[0]["input"][0]["content"][0]["text"], "When do you open?")

    async def test_a_deleted_conversation_stops_its_work_releases_its_model_and_is_gone_from_every_endpoint(self):
        model = Model([asyncio.Event(), *text_round("Done.")])  # the round still runs when the conversation is deleted
        self.core(model, close_grace_seconds=0.2)
        conversation = await self.open()
        stream = await self.client.create_response(self.request(conversation))
        async with asyncio.timeout(2):
            while not model.payloads:
                await asyncio.sleep(0.005)
        deleting = asyncio.create_task(self.client.delete_conversation(conversation))
        await asyncio.sleep(0.05)
        self.assertEqual((await self.client.list_conversation_items(conversation))["object"], "list")  # still closing
        self.assertEqual(await deleting, {"id": conversation, "object": "conversation.deleted", "deleted": True})
        ended = [event async for event in stream][-1]
        self.assertEqual((ended["type"], ended["response"]["status"]), ("response.incomplete", "cancelled"))
        self.assertEqual(model.closes, 1)
        for call in (
            self.client.delete_conversation,
            self.client.list_conversation_items,
            lambda conversation: self.client.create_response(self.request(conversation)),
        ):
            with self.subTest(call), self.assertRaises(NotFound):
                await call(conversation)
        breaks = mock.AsyncMock(side_effect=RuntimeError("cut short"))
        cut_short = ResponsesService(lambda: session(close=breaks), backend=Rules())
        conversation = (await cut_short.create_conversation())["id"]
        with self.assertRaisesRegex(RuntimeError, "cut short"):
            await cut_short.delete_conversation(conversation)
        with self.assertRaises(NotFound):  # a close cut short still forgets the conversation
            await cut_short.list_conversation_items(conversation)

    async def test_two_conversations_run_apart_and_one_ends_while_the_others_work_goes_on(self):
        gate = asyncio.Event()
        first_model, second_model = Model([gate, *text_round("First.")]), Model(text_round("Second."))
        self.core(first_model, second_model)
        first, second = await self.open(), await self.open({"live_session_id": "live_2"})
        held = await self.client.create_response(self.request(first, "One."))
        self.assertEqual(answer_text((await self.ask(self.request(second, "Two.")))[-1]), "Second.")  # not queued
        self.assertEqual((await self.client.list_conversation_items(second))["data"][0], user("Two."))
        await self.client.delete_conversation(second)
        self.assertEqual((first_model.closes, second_model.closes), (0, 1))
        gate.set()
        self.assertEqual(answer_text([event async for event in held][-1]), "First.")
        self.assertEqual(len(first_model.payloads), 1)

    async def test_each_request_goes_to_the_admission_its_kind_takes(self):
        im = session()
        service = ResponsesService(lambda: im, backend=Rules())
        conversation = (await service.create_conversation())["id"]
        output = {"type": "function_call_output", "call_id": "call_1", "output": "{}"}
        typed = {"live_delegation_id": "item_3", "live_typed_run": "true"}
        for request in (
            self.request(conversation),  # a delegation: it names itself and brings no outputs
            {**self.request(conversation), "input": [output]},
            self.request(conversation, "When do you open?", metadata=typed),
            self.request(conversation, "Untagged.", metadata={}),  # a plain Responses client's request
            {**self.request(conversation, model="org/a", instructions="Be brief."), "input": []},
        ):
            await service.create_response(request)
        admitted = [(name, kwargs["delegation_id"]) for name, _, kwargs in im.mock_calls if name.startswith("admit")]
        client, delegation = "admit_client_request", "admit"
        expected = [(delegation, "item_1"), (client, "item_1"), (client, "item_3"), (client, None)]
        self.assertEqual(admitted, [*expected, (delegation, "item_1")])
        self.assertEqual(im.admit.call_args.kwargs["settings"], {"model": "org/a", "instructions": "Be brief."})


class CheckTests(CoreCase):
    async def test_a_malformed_request_is_refused_before_any_event_with_its_param_and_code(self):
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
        self.assertIsInstance(refused.exception, ValueError)  # a 400
        self.assertIn("at most 64 levels deep", str(refused.exception))
        # The request, its input and the item are 3 levels: 61 lists more make 64, the most a request may nest.
        deepest = await self.ask({"input": [{**USER, "meta": nested(MAX_REQUEST_DEPTH - 3)}]})
        self.assertEqual((deepest[-1]["type"], model.opened), ("response.completed", 1))
        self.assertEqual((await self.client.list_conversation_items(conversation))["data"], [])  # nothing was admitted
        self.core(stateless=False)
        with self.assertRaises(InvalidRequest) as refused:
            await self.client.create_response({"input": [USER]})
        self.assertEqual(refused.exception.param, "conversation")  # a core with no one-shot sessions

    async def test_the_backends_rules_check_a_model_and_input_tokens_counts_a_request_and_admits_nothing(self):
        self.core(Model(text_round("Done.")), rules=Rules())
        served = await self.client.retrieve_model("org/a")  # no conversation was ever opened
        self.assertEqual((served["id"], served["object"], served["owned_by"]), ("org/a", "model", "delegation_core"))
        conversation = await self.open()
        with self.assertRaisesRegex(ValueError, "not 'org/b'"):
            await self.client.retrieve_model("org/b")
        await self.ask(self.request(conversation, "Find teas."))
        counted = await self.client.input_tokens(self.request(conversation, instructions="Be brief."))
        self.assertEqual((sorted(counted), counted["object"]), (["input_tokens", "object"], "response.input_tokens"))
        alone = await self.client.input_tokens({"input": [user("Hi.")], "instructions": "Be brief."})
        self.assertGreater(counted["input_tokens"], alone["input_tokens"])  # the conversation's items count too
        self.assertGreater((await self.client.input_tokens({"model": "org/a"}))["input_tokens"], 0)  # no input given
        self.assertEqual(len((await self.client.list_conversation_items(conversation))["data"]), 2)
        settings = {"model": "org/a", "service_tier": "priority", "text": {"verbosity": "low"}}
        with self.assertRaises(InvalidRequest) as refused:
            await self.client.input_tokens({"input": [], **settings})
        self.assertEqual(refused.exception.param, "text")  # the first in the request's settings
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

    async def test_the_whole_input_and_the_clients_settings_reach_the_model_as_sent_with_the_notes(self):
        first, second = Model(text_round("We have chai.")), Model(text_round("Hello."))
        self.core(first, second)
        sent = [user("x" * 120_000), USER, CALL, OUTPUT]  # far over the core's context budget: never trimmed
        events = await self.ask({"input": sent, "stream": True, "store": False})
        self.assertEnded(events, "response.completed")
        created, completed = events[0]["response"], events[-1]["response"]
        self.assertEqual((created["id"], answer_text(events[-1])), (completed["id"], "We have chai."))
        self.assertNotEqual(completed["id"], "resp_1")  # the core's response, not the round's
        # Never read as a continuation: the function output is part of the context, and the model is asked.
        payload, composed = first.payloads[0], "\n".join(["Default.", *NOTES])
        self.assertEqual(payload["input"], sent)
        self.assertEqual((payload["model"], payload["instructions"]), ("org/default-model", composed))
        self.assertEqual(first.closes, 1)  # the response's end ended its session, which released the model
        settings = {"model": "org/other", "instructions": "Be terse.", "tools": [MENU], "tool_choice": "auto"}
        events = await self.ask({"input": [USER], **settings})
        payload = second.payloads[0]
        self.assertEqual((payload["model"], payload["tools"], payload["tool_choice"]), ("org/other", [MENU], "auto"))
        self.assertEqual(payload["instructions"], "\n".join(["Be terse.", *NOTES]))
        self.assertEqual((events[-1]["response"]["model"], "previous_response_id" in payload), ("org/other", False))

    async def test_function_calls_end_the_response_and_nothing_waits_for_their_outputs(self):
        model = Model(call_round("get_menu", {}, call_id="call_9", said="Checking the menu."))
        self.core(model)
        events = await self.ask({"input": [USER], "tools": [MENU]})  # within 2 s, where the client-tool wait is 60 s
        self.assertEnded(events, "response.completed")
        said, call = events[-1]["response"]["output"]
        self.assertEqual((said["phase"], said["content"][0]["text"]), ("commentary", "Checking the menu."))
        self.assertEqual((call["type"], model.closes), ("function_call", 1))

    async def test_a_round_that_fails_is_tried_once_and_ends_the_response_failed(self):
        quick = ProviderLimits(provider_retries=0, first_event_timeout_seconds=2.0, idle_timeout_seconds=0.2)
        incomplete = {"type": "response.incomplete", "response": {"id": "r", "status": "incomplete"}}
        cases = {
            "an error": ([RuntimeError("model down")], LIMITS),
            "an incomplete round": ([CREATED, incomplete], LIMITS),
            "no terminal": ([CREATED], LIMITS),
            "no items": (a_round(), LIMITS),  # an empty round is retried once, never in one-shot
            "silent past its limit": ([CREATED, 30.0], quick),
        }
        for name, (step, limits) in cases.items():
            with self.subTest(name):
                model = Model(step, text_round("never"), limits=limits)
                self.core(model)
                events = await self.ask({"input": [USER]})
                self.assertEnded(events, "response.failed")
                self.assertNotIn("response.output_item.done", [event["type"] for event in events])  # no apology
                self.assertEqual((model.opened, model.closes), (1, 1))

    async def test_closing_the_stream_early_stops_the_work_and_releases_the_model(self):
        model = Model([CREATED, 30.0])
        self.core(model)
        stream = await self.client.create_response({"input": [USER]})
        self.assertEqual((await anext(stream))["type"], "response.created")
        await stream.aclose()  # the reader stops: over HTTP, the client's disconnect
        self.assertEqual(model.closes, 1)
        ended = [event async for event in stream][-1]
        self.assertEqual((ended["type"], ended["response"]["status"]), ("response.incomplete", "cancelled"))
        await stream.aclose()  # safe twice
        self.assertEqual(model.closes, 1)

    async def test_a_malformed_item_fails_its_own_response_and_an_unknown_one_is_left_out(self):
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
                self.assertEqual(chat.requests, [])  # the provider could not build the model request
        chat = FakeChat([chunk("We have chai."), chunk(finish="stop")])
        self.core(ChatCompletionsBackend(ENDPOINT, LIMITS, chat))
        events = await self.ask({"input": [USER, {"type": "reasoning", "summary": []}, {"odd": True}]})
        self.assertEqual(answer_text(events[-1]), "We have chai.")
        self.assertEqual([message["role"] for message in chat.requests[0]["messages"]], ["system", "user"])

    async def test_many_stateless_requests_run_at_once_each_in_a_session_of_its_own(self):
        gate = asyncio.Event()
        models = [Model([gate, *text_round(f"Answer {n}.")]) for n in range(10)]
        self.core(*models)
        streams = [await self.client.create_response({"input": [user(f"Q{n}")]}) for n in range(10)]
        async with asyncio.timeout(2):
            while not all(model.opened for model in models):  # every round runs before any of them ends
                await asyncio.sleep(0.005)
        gate.set()
        answers = [answer_text([event async for event in stream][-1]) for stream in streams]
        self.assertEqual(answers, [f"Answer {n}." for n in range(10)])
        self.assertEqual([model.payloads[0]["input"] for model in models], [[user(f"Q{n}")] for n in range(10)])
        self.assertEqual([model.closes for model in models], [1] * 10)

    async def test_a_stateless_request_leaves_the_cores_conversations_alone(self):
        conversation_model, stateless_model = Model(text_round("Hello.")), Model(text_round("Ok."))
        self.core(stateless_model, conversation_model)  # in the order the sessions' first rounds start
        conversation = await self.open()
        await self.ask({"input": [USER, CALL, OUTPUT], "metadata": {"live_delegation_id": "item_1"}})
        self.assertEqual((await self.client.list_conversation_items(conversation))["data"], [])  # it saw nothing
        answered = await self.ask(self.request(conversation))  # item_1 is free there
        self.assertEqual(answer_text(answered[-1]), "Hello.")
        self.assertEqual(stateless_model.payloads[0]["input"], [USER, CALL, OUTPUT])
