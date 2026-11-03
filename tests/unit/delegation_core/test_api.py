# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""The Responses API surface's own checks, on a fake IM session and the emulator's rules."""

import asyncio
import unittest
from itertools import count
from types import SimpleNamespace

from delegation_core.api.local import LocalResponsesClient
from delegation_core.api.service import MAX_REQUEST_DEPTH, InvalidRequest, NotFound, ResponsesService
from tests.unit.delegation_core.emulator import emulator_source

CONVERSATIONS = count(1)


def user(text):
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


class Stream:
    """A response stream as the IM gives it: no events here."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def aclose(self):
        pass


class FakeSession:
    """An IM session as the API surface calls it: it records each admission and its close."""

    def __init__(self, close_gate=None, fails_to_close=False):
        self.ledger = SimpleNamespace(session=SimpleNamespace(id=f"conv_{next(CONVERSATIONS)}"))
        self.history = [user("An earlier request.")]
        self.admitted = []  # (how, input items, delegation id, settings), in order
        self.started = self.closed = False
        self.close_gate, self.fails_to_close = close_gate, fails_to_close
        self.seen_during_close = None

    def start(self):
        self.started = True

    async def close(self):
        if self.close_gate is not None:
            await self.close_gate.wait()
        self.closed = True
        if self.fails_to_close:
            raise RuntimeError("the close was cut short")

    def admit(self, input_items, *, delegation_id=None, settings=None):
        self.admitted.append(("admit", input_items, delegation_id, settings))
        return SimpleNamespace(stream=Stream())

    def admit_client_request(self, input_items, *, delegation_id=None, settings=None):
        self.admitted.append(("client", input_items, delegation_id, settings))
        return SimpleNamespace(stream=Stream())


class SurfaceCase(unittest.IsolatedAsyncioTestCase):
    def surface(self, backend=None, **session):
        """A service on fake IM sessions, with ``backend`` as its rules (by default an emulator serving any model)."""
        self.sessions = []

        def new_session():
            self.sessions.append(FakeSession(**session))
            return self.sessions[-1]

        self.service = ResponsesService(new_session, backend=backend or emulator_source())
        self.client = LocalResponsesClient(self.service)
        return self.client

    async def conversation(self):
        return (await self.client.create_conversation({"live_session_id": "live_1"}))["id"]

    @staticmethod
    def request(conversation, text="Find teas.", **fields):
        metadata = fields.pop("metadata", {"live_delegation_id": "item_1"})
        return {"input": [user(text)], "conversation": conversation, "metadata": metadata, **fields}


# ---------------------------------------------------------------------------------------- GET /v1/models/{model}
class ModelCheckTests(SurfaceCase):
    async def test_the_backends_rules_answer_with_or_without_a_conversation_open(self):
        self.surface(backend=emulator_source({"models": ["org/served"]}))
        served = await self.client.retrieve_model("org/served")  # no conversation was ever opened
        self.assertEqual(
            (served["id"], served["object"], served["owned_by"]), ("org/served", "model", "delegation_core")
        )
        with self.assertRaisesRegex(ValueError, "not 'org/other'"):
            await self.client.retrieve_model("org/other")
        self.assertEqual(self.sessions, [])
        await self.conversation()
        with self.assertRaisesRegex(ValueError, "not 'org/other'"):
            await self.client.retrieve_model("org/other")  # the backend's rule, not a conversation's IM session


# -------------------------------------------------------------------------------------------- the request checks
def nested(levels):
    value = "x"
    for _ in range(levels):
        value = [value]
    return value


class RequestCheckTests(SurfaceCase):
    async def test_a_conversations_request_nested_too_deeply_is_refused_before_the_im_sees_it(self):
        self.surface()
        conversation = await self.conversation()
        deep = self.request(conversation, meta=nested(MAX_REQUEST_DEPTH))
        with self.assertRaisesRegex(InvalidRequest, "at most 64 levels deep"):  # every request, not stateless only
            await self.client.create_response(deep)
        await self.client.create_response(self.request(conversation, meta=nested(MAX_REQUEST_DEPTH - 2)))
        self.assertEqual(len(self.sessions[0].admitted), 1)

    async def test_each_malformed_field_is_named_and_crosses_the_client_with_its_code(self):
        self.surface()
        conversation = await self.conversation()
        cases = {
            "input": (self.request(conversation) | {"input": "not a list"}, None),
            "metadata": (self.request(conversation, metadata=["item_1"]), None),
            "previous_response_id": (self.request(conversation, previous_response_id="resp_1"), None),
        }
        for param, (request, code) in cases.items():
            with self.subTest(param), self.assertRaises(InvalidRequest) as refused:
                await self.client.create_response(request)
            self.assertEqual((refused.exception.param, refused.exception.code), (param, code))
        self.assertEqual(self.sessions[0].admitted, [])
        with self.assertRaises(InvalidRequest) as refused:  # no conversation: a response the core never stored
            await self.client.create_response({"input": [user("Hi.")], "previous_response_id": "resp_1"})
        self.assertEqual(refused.exception.code, "previous_response_not_found")
        self.assertIsInstance(refused.exception, ValueError)  # a 400 wherever a ValueError is refused

    async def test_a_core_without_one_shot_sessions_refuses_a_request_with_no_conversation(self):
        self.surface()
        with self.assertRaises(InvalidRequest) as refused:
            await self.client.create_response({"input": [user("Hi.")]})
        self.assertEqual(refused.exception.param, "conversation")


# ------------------------------------------------------------------------------------- which admission
class PassThroughTests(SurfaceCase):
    async def test_each_request_goes_to_the_admission_its_kind_takes(self):
        self.surface()
        conversation = await self.conversation()
        output = {"type": "function_call_output", "call_id": "call_1", "output": "{}"}
        typed = {"live_delegation_id": "item_3", "live_typed_run": "true"}
        requests = [
            self.request(conversation),  # a delegation: it names itself and brings no outputs
            self.request(conversation, metadata={"live_delegation_id": "item_1"}) | {"input": [output]},
            self.request(conversation, "When do you open?", metadata=typed),
            self.request(conversation, "Untagged.", metadata={}),  # a plain Responses client's request
            self.request(conversation, model="org/a", instructions="Be brief.") | {"input": []},
        ]
        for request in requests:
            await self.client.create_response(request)
        admitted = [(how, delegation_id) for how, _, delegation_id, _ in self.sessions[0].admitted]
        self.assertEqual(
            admitted,
            [("admit", "item_1"), ("client", "item_1"), ("client", "item_3"), ("client", None), ("admit", "item_1")],
        )
        self.assertEqual(self.sessions[0].admitted[-1][3], {"model": "org/a", "instructions": "Be brief."})


# ------------------------------------------------------------------------------- POST /v1/responses/input_tokens
class InputTokensTests(SurfaceCase):
    async def test_a_request_is_counted_with_its_conversations_items_and_nothing_is_admitted(self):
        self.surface(backend=emulator_source())
        conversation = await self.conversation()
        counted = await self.client.input_tokens(self.request(conversation, instructions="Be brief."))
        self.assertEqual(sorted(counted), ["input_tokens", "object"])
        self.assertEqual(counted["object"], "response.input_tokens")
        alone = await self.client.input_tokens({"input": [user("Find teas.")], "instructions": "Be brief."})
        self.assertGreater(counted["input_tokens"], alone["input_tokens"])  # the conversation's items count too
        self.assertGreater((await self.client.input_tokens({"model": "org/a"}))["input_tokens"], 0)  # no input given
        self.assertEqual(self.sessions[0].admitted, [])

    async def test_the_settings_the_backend_cannot_honor_are_named(self):
        unsupported = {"service_tier": "Only the default tier.", "text": "No verbosity."}
        self.surface(backend=emulator_source({"models": ["org/a"], "unsupported": unsupported}))
        settings = {"model": "org/a", "service_tier": "priority", "text": {"verbosity": "low"}}
        with self.assertRaises(InvalidRequest) as refused:
            await self.client.input_tokens({"input": [], **settings})
        self.assertEqual(refused.exception.param, "text")  # the first in the request's settings
        self.assertIn("text: No verbosity.", str(refused.exception))
        self.assertIn("service_tier: Only the default tier.", str(refused.exception))
        with self.assertRaises(InvalidRequest) as refused:
            await self.client.input_tokens({"input": [], "model": "org/b"})
        self.assertEqual(refused.exception.param, "model")
        self.assertEqual(
            (await self.client.input_tokens({"input": [], "model": "org/a"}))["object"], "response.input_tokens"
        )

    async def test_it_makes_the_same_checks_as_a_response(self):
        self.surface(backend=emulator_source())
        with self.assertRaises(NotFound):
            await self.client.input_tokens(self.request("conv_unknown"))
        with self.assertRaisesRegex(InvalidRequest, "levels deep"):
            await self.client.input_tokens({"input": [user("Hi.")], "meta": nested(MAX_REQUEST_DEPTH)})
        with self.assertRaises(InvalidRequest) as refused:
            await self.client.input_tokens({"input": "Hi."})
        self.assertEqual(refused.exception.param, "input")


# ---------------------------------------------------------------------- DELETE /v1/conversations/{id}
class CloseTests(SurfaceCase):
    async def test_the_conversation_is_forgotten_only_once_its_im_session_has_closed(self):
        gate = asyncio.Event()
        self.surface(close_gate=gate)
        conversation = await self.conversation()
        delete = asyncio.create_task(self.client.delete_conversation(conversation))
        await asyncio.sleep(0)
        self.assertIn(conversation, self.service.sessions)  # its responses are still ending
        self.assertEqual((await self.client.list_conversation_items(conversation))["object"], "list")
        gate.set()
        self.assertEqual(await delete, {"id": conversation, "object": "conversation.deleted", "deleted": True})
        self.assertTrue(self.sessions[0].closed)
        self.assertEqual((self.service.sessions, self.service.conversations), ({}, {}))
        with self.assertRaises(NotFound):
            await self.client.delete_conversation(conversation)

    async def test_a_close_cut_short_still_forgets_the_conversation(self):
        self.surface(fails_to_close=True)
        conversation = await self.conversation()
        with self.assertRaisesRegex(RuntimeError, "cut short"):
            await self.client.delete_conversation(conversation)
        self.assertEqual(self.service.sessions, {})


if __name__ == "__main__":
    unittest.main()
