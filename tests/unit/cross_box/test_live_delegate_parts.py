# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""``POST /v1/live/delegate`` through the server's route, the root's one delegation core and the backend LLM.

Each round is one stateless request that a one-shot session serves; only the model client is scripted.
"""

import asyncio
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import httpx
import openai
import yaml
from fastapi.testclient import TestClient

import server
from delegation_core.backend.providers import base as providers
from delegation_core.config import CoreSettings
from delegation_core.im.input import compose
from examples.frontend_backend_live import factory
from examples.frontend_backend_live.prompts.catalog import catalog_prompt
from tests.unit.cascade_helpers import MENU_TOOL, tool_call_round
from tests.unit.core_helpers import FakeChat, chunk
from tests.unit.frontend_backend_live.root.root_helpers import clear_root_caches
from utils import clear_service_context, set_active_services

CORE = Path(factory.__file__).with_name("core.yaml")
USER = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What teas do you have?"}]}
LATER = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "And coffee?"}]}
FAILED = (502, "The backend could not complete the round")


def answer(text):
    return [chunk(text), chunk(finish="stop")]


def voice_note(name="default"):
    return CoreSettings.load(CORE).texts.voice_notes[name]


class Hanging:
    """A model stream that never sends its first chunk."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(30)
        raise StopAsyncIteration

    async def close(self):
        pass


class HangingChat(FakeChat):
    """A model client whose rounds never start answering."""

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return Hanging()


class ServedRoute(unittest.TestCase):
    """The route as the server serves it, on a core the root builds fresh for each ``served`` block."""

    def setUp(self):
        # The bundled files and the registry's hooks; the catalog's self-hosted section, named outright, and no NVIDIA
        # Cloud key, so no reachability probe picks another. The test client sends Host: testserver, which the console
        # route answers with no token only when it is allowed.
        live = {"LIVE_ENABLED": "true", "LIVE_API_TOKEN": "", "LIVE_ALLOWED_HOSTS": "testserver"}
        env = patch.dict(os.environ, {**live, "SERVICE_RECIPE": "server", "NVIDIA_API_KEY": ""})
        env.start()
        self.addCleanup(env.stop)
        for name in ("SERVICES_PATH", "LIVE_CONFIG_PATH", "CORE_CONFIG_PATH", "LIVE_ADAPTER_FACTORY"):
            os.environ.pop(name, None)
        # create_app binds the selected example's services process-wide; leave no binding behind.
        self.addCleanup(clear_service_context)
        self.addCleanup(set_active_services, None)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)

    def profile(self, **blocks) -> Path:
        """A core profile over the shipped ``core.yaml`` that sets ``blocks`` (merged key by key)."""
        path = self.folder / f"core.{len(list(self.folder.iterdir()))}.yaml"
        path.write_text(yaml.safe_dump({"extends": str(CORE), **blocks}), encoding="utf-8")
        return path

    @contextmanager
    def served(self, client, *, core=None):
        """The server's app on a new core (``core``: a profile), its providers' clients made as ``client``.

        Yields the test client and the requests the route sent to the core; ``self.service`` is the core's service.
        """
        clear_root_caches(self)
        sent: list[dict] = []
        make = factory.core_client

        def spied(built):
            core_client = make(built)
            create = core_client.create_response

            async def create_response(params):
                sent.append(json.loads(json.dumps(params)))
                return await create(params)

            core_client.create_response = create_response
            self.service = core_client.service
            return core_client

        with (
            patch.dict(os.environ, {"CORE_CONFIG_PATH": str(core or CORE)}),
            patch.object(providers, "AsyncOpenAI", return_value=client),
            patch.object(factory, "core_client", spied),
            TestClient(server.create_app()) as http,
        ):
            yield http, sent

    def round(self, body, *script, core=None, chat=None):
        """One console round on a fresh core: the reply, the model client and the requests sent to the core."""
        chat = chat or FakeChat(*script)
        with self.served(chat, core=core) as (http, sent):
            response = http.post("/v1/live/delegate", json=body)
        return response, chat, sent


# --------------------------------------------------------------------------------------------- the rounds
class RoundTests(ServedRoute):
    """Each round reaches the root's core as one stateless request, and a one-shot session's backend LLM runs it."""

    def assert_stateless(self, sent, items):
        # One stateless request: no conversation and no previous response, the client's whole history as its input.
        # A one-shot session served it, so the core opened no conversation.
        (request,) = sent
        self.assertNotIn("conversation", request)
        self.assertNotIn("previous_response_id", request)
        self.assertEqual(request["input"], items)
        self.assertEqual(self.service.sessions, {})

    def test_a_resent_history_that_holds_function_outputs_reaches_the_backend_model(self):
        # Round 1: the backend model asks for a tool; the round returns the call for the client to run.
        response, chat, sent = self.round({"input": [USER], "tools": [MENU_TOOL]}, tool_call_round("call_1"))
        self.assertEqual(response.status_code, 200, response.text)
        self.assert_stateless(sent, [USER])
        reply = response.json()
        call = reply["response"]["output"][0]
        self.assertEqual((call["type"], call["name"]), ("function_call", "get_menu"))
        self.assertTrue(reply["response"]["id"])
        self.assertEqual(reply["sent_request"]["tools"], ["get_menu"])
        self.assertEqual([m["role"] for m in chat.requests[0]["messages"]], ["system", "user"])
        self.assertTrue(chat.closed)  # the one-shot session's model client, closed after its round
        # Round 2: the client resends its whole history with the function's output, under the call id the core gave.
        # A request in a conversation would continue open work; a stateless one reaches the model.
        output = {"type": "function_call_output", "call_id": call["call_id"], "output": '{"items": ["green tea"]}'}
        response, chat, sent = self.round(
            {"input": [USER, call, output], "tools": [MENU_TOOL]}, answer("We have green tea and chai.")
        )
        self.assert_stateless(sent, [USER, call, output])
        message = response.json()["response"]["output"][0]
        self.assertEqual(message["type"], "message")
        messages = chat.requests[0]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "tool"])
        self.assertEqual(messages[2]["tool_calls"][0]["id"], call["call_id"])
        self.assertEqual((messages[3]["tool_call_id"], messages[3]["content"]), (call["call_id"], output["output"]))
        # The first round of a later delegation carries the earlier delegation's items too, and still reaches the model.
        _, chat, _ = self.round({"input": [USER, call, output, message, LATER]}, answer("We have espresso."))
        self.assertEqual(len(chat.requests), 1)
        self.assertEqual(
            [m["role"] for m in chat.requests[0]["messages"]],
            ["system", "user", "assistant", "tool", "assistant", "user"],
        )

    def test_the_core_fills_what_the_client_left_out_and_sends_what_it_chose(self):
        chat = FakeChat(answer("Hello."), answer("Hello."))
        with self.served(chat) as (http, _):
            info = http.get("/v1/live/info").json()  # what the console reads (app.js)
            default = http.post("/v1/live/delegate", json={"input": [USER]}).json()
            model = next(name for name in info["backend_models"] if name != info["backend"]["model"])
            tools = [MENU_TOOL, {"type": "web_search"}]
            body = {"input": [USER], "model": model, "instructions": "Be terse.", "tools": tools}
            chosen = http.post("/v1/live/delegate", json=body).json()
        first, second = chat.requests
        # The core's defaults: the backend slot's default model and its prompt as written, then the voice note. The
        # console's prefill is the cafe demo's prompt instead, which it sends with the cafe's sample tools.
        prompt = catalog_prompt(info["backend"]["prompt"])
        self.assertEqual((first["model"], default["sent_request"]["model"]), (info["backend"]["model"],) * 2)
        self.assertEqual(first["messages"][0]["content"], compose(prompt, voice_note()))
        self.assertNotEqual(info["instructions"]["backend"], prompt)
        self.assertNotIn("tools", first)
        # The client's own: a model the catalog lists for the slot, its instructions and its tools. The reply names a
        # hosted tool by its type; the chat-completions backend gets the function tools only.
        self.assertEqual((second["model"], chosen["sent_request"]["model"]), (model, model))
        self.assertEqual(second["messages"][0]["content"], compose("Be terse.", voice_note()))
        self.assertEqual(chosen["sent_request"]["tools"], ["get_menu", "web_search"])
        self.assertEqual([tool["function"]["name"] for tool in second["tools"]], ["get_menu"])

    def test_a_round_with_no_items_fails(self):
        # An empty round is no answer, so the round ends failed.
        response, chat, _ = self.round({"input": [USER]}, [chunk(finish="stop")], answer("never"))
        self.assertEqual((response.status_code, response.json()["detail"]), FAILED)
        self.assertEqual(len(chat.requests), 1)  # one attempt: a one-shot session's adapter tries a round once

    def test_a_failed_round_is_tried_once_and_ends_failed_in_the_core(self):
        response, chat, _ = self.round({"input": [USER]}, RuntimeError("model down"), answer("never"))
        self.assertEqual((response.status_code, response.json()["detail"]), FAILED)
        self.assertEqual(len(chat.requests), 1)  # the round is not tried again
        self.assertTrue(chat.closed)

    def test_a_transient_failure_gets_the_providers_own_retries_and_no_more(self):
        # Opening the round is retried on a transient error, provider_retries times, and nothing on top.
        error = openai.APIConnectionError(request=httpx.Request("POST", "http://backend.test/v1/chat/completions"))
        retries = CoreSettings.load(CORE).backend.blocks["llm"]["provider_retries"]
        with patch("delegation_core.backend.providers.retry.retry_delay", return_value=0):
            response, chat, _ = self.round({"input": [USER]}, *[error] * (2 * retries + 2), answer("never"))
        self.assertEqual((response.status_code, response.json()["detail"]), FAILED)
        self.assertEqual(len(chat.requests), retries + 1)
        self.assertTrue(chat.closed)

    def test_a_round_past_the_backends_time_limit_is_a_502_and_its_model_client_is_closed(self):
        # The backend LLM times each round (backend.llm.first_event_timeout_seconds), well before the route's
        # backstop, and the one-shot session releases its model client when its response ends.
        fast = self.profile(backend={"llm": {"first_event_timeout_seconds": 0.2}})
        response, chat, _ = self.round({"input": [USER]}, chat=HangingChat(), core=fast)
        self.assertEqual((response.status_code, response.json()["detail"]), FAILED)
        self.assertTrue(chat.closed)


# --------------------------------------------------------------------------------------- the server's shutdown
class ShutdownTests(ServedRoute):
    def test_the_servers_shutdown_closes_the_roots_core_that_served_its_rounds(self):
        # The gateway closes the core it holds through its client: the root handed the client the core's close, which
        # releases each backend source (the shipped core.yaml has one, llm, for both kinds of session).
        with self.served(FakeChat(answer("Hello."))) as (http, _):
            self.assertEqual(http.post("/v1/live/delegate", json={"input": [USER]}).status_code, 200)
            released = self.enterContext(patch.object(factory.delegation_core().source, "aclose"))
            released.assert_not_awaited()  # a round leaves the core open
        released.assert_awaited_once()


# ------------------------------------------------------------------------------- a broken Responses backend
class _Event(dict):
    """A Responses streaming event as the OpenAI client yields it."""

    def model_dump(self, **kwargs):
        return dict(self)


class _Events:
    def __init__(self, events):
        self.events = events

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for event in self.events:
            yield _Event(event)

    async def close(self):
        pass


class FakeResponses:
    """Scripted ``responses.create`` of an OpenAI client: it notes the request and streams ``events``."""

    def __init__(self, events):
        self.events, self.requests, self.closed = events, [], False
        self.responses = self

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return _Events(self.events)

    async def close(self):
        self.closed = True


def _response(status, *output):
    return {"id": "resp_up", "object": "response", "created_at": 1, "model": "m", "status": status, "output": [*output]}


def _message(item_id, text):
    content = [{"type": "output_text", "text": text, "annotations": []}]
    return {"type": "message", "id": item_id, "role": "assistant", "status": "completed", "content": content}


def _added(item_id):
    item = {"type": "message", "id": item_id, "role": "assistant", "status": "in_progress", "content": []}
    return {"type": "response.output_item.added", "output_index": 0, "item": item}


def _part(item_id):
    part = {"type": "output_text", "text": "", "annotations": []}
    where = {"item_id": item_id, "output_index": 0, "content_index": 0}
    return {"type": "response.content_part.added", **where, "part": part}


def _delta(item_id, text):
    where = {"item_id": item_id, "output_index": 0, "content_index": 0}
    return {"type": "response.output_text.delta", **where, "delta": text}


def _done(item):
    return {"type": "response.output_item.done", "output_index": 0, "item": item}


def _completed(*output):
    return {"type": "response.completed", "response": _response("completed", *output)}


CREATED = {"type": "response.created", "response": _response("in_progress")}
TEA, MORE = _message("msg_1", "We have chai."), _message("msg_2", "And green tea.")
CALL_WITHOUT_ID = {"type": "function_call", "id": "fc_1", "name": "get_menu", "arguments": "{}", "status": "completed"}


class MalformedBackendStreamTests(ServedRoute):
    """The route on the real ``openai-responses`` provider, whose stream breaks the Responses event rules.

    The backend LLM reads a round by structure: its items come from ``output_item.done``, reading stops at the first
    terminal, a second ``response.created`` breaks the round, and a round with no answer fails.
    """

    def play(self, events, body=None):
        client = FakeResponses(events)
        core = self.profile(backend={"llm": {"provider": "openai-responses"}})
        response, _, _ = self.round(body or {"input": [USER]}, core=core, chat=client)
        self.assertEqual(len(client.requests), 1)
        self.assertTrue(client.closed)
        return response

    def test_the_reply_is_the_items_the_round_finished(self):
        cases = {
            # The item is the one output_item.done gives, whatever id it was added with.
            "an added id that differs from the done one": [
                CREATED,
                _added("msg_a"),
                _delta("msg_a", "We have chai."),
                _done(TEA),
                _completed(TEA),
            ],
            # Reading stops at the first terminal: an item after it is not the round's.
            "items after the terminal": [CREATED, _done(TEA), _completed(TEA), _done(MORE)],
        }
        for name, events in cases.items():
            with self.subTest(name):
                response = self.play(events)
                self.assertEqual(response.status_code, 200, response.text)
                [item] = response.json()["response"]["output"]
                self.assertEqual((item["type"], item["content"][0]["text"]), ("message", "We have chai."))

    def test_a_round_the_backend_llm_cannot_read_fails(self):
        failed = {"type": "response.failed", "response": _response("failed")}
        error = {"type": "error", "code": "server_error", "message": "boom", "param": None}
        cases = {
            # The open message is no item, so the round has no answer.
            "a message left open": [CREATED, _added("msg_1"), _part("msg_1"), _delta("msg_1", "We have"), _completed()],
            # A second response.created is out of order: the round is broken.
            "a second response.created over an open message": [
                CREATED,
                _added("msg_0"),
                _delta("msg_0", "We"),
                CREATED,
                _done(TEA),
                _completed(TEA),
            ],
            "an item, then a second response.created": [CREATED, _done(MORE), CREATED, _done(TEA), _completed(TEA)],
            # No terminal, only an error.
            "a stream that ends after an error event": [CREATED, _done(TEA), error],
            # The first terminal decides.
            "a failed terminal, then a completed one": [CREATED, _done(TEA), failed, _completed(TEA)],
        }
        for name, events in cases.items():
            with self.subTest(name):
                response = self.play(events)
                self.assertEqual((response.status_code, response.json()["detail"]), FAILED)

    def test_a_function_call_without_a_call_id_fails_the_round(self):
        # The backend LLM keys each call by its call_id. Only the chat-completions translation gives calls the core's
        # own ids; a Responses item keeps the provider's.
        response = self.play([CREATED, _done(CALL_WITHOUT_ID), _completed()], {"input": [USER], "tools": [MENU_TOOL]})
        self.assertEqual((response.status_code, response.json()["detail"]), FAILED)


# ------------------------------------------------------------------------------------------- the route's seam
class RouteSeamTests(ServedRoute):
    """The body the route reads, through the gateway's guard, before any round."""

    def post_text(self, text, chat, **headers):
        """POST ``text`` as the body, as written: Python's JSON reader also takes NaN, Infinity and lone surrogates."""
        with self.served(chat) as (http, _):
            return http.post("/v1/live/delegate", content=text, headers={"content-type": "application/json", **headers})

    def test_a_value_that_plain_json_cannot_carry_is_a_400_and_reaches_no_model(self):
        call = '{"type": "function_call", "call_id": "c", "name": "get_menu", "arguments": Infinity}'
        cases = {
            "NaN": '{"input": [{"type": "function_call_output", "call_id": "c", "output": NaN}]}',
            "Infinity": f'{{"input": [{json.dumps(USER)}, {call}]}}',
            "a lone surrogate": '{"input": [{"role": "user", "content": "\\ud800"}]}',
        }
        for name, text in cases.items():
            with self.subTest(name):
                chat = FakeChat(answer("never"))
                response = self.post_text(text, chat)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(chat.requests, [])

    def test_a_request_nested_too_deeply_is_a_400_and_reaches_no_model(self):
        # The core refuses it before any session: the IM copies each request, and Python's copy recurses.
        chat = FakeChat(answer("never"))
        nested = "[" * 600 + "1" + "]" * 600
        response = self.post_text(f'{{"input": [{{"role": "user", "content": "hi", "meta": {nested}}}]}}', chat)
        detail = "A request may nest objects and lists at most 64 levels deep"
        self.assertEqual((response.status_code, response.json()["detail"]), (400, detail))
        self.assertEqual(chat.requests, [])

    def test_the_route_reads_its_body_through_the_gateways_guard(self):
        # The gateway's guard: a JSON object, within the console's size limit, from the same origin.
        big = json.dumps({"input": [USER] * 12000})
        cases = {
            "over the console's limit": ({"content": big, "headers": {"content-type": "application/json"}}, 413),
            "a JSON array": ({"json": [USER]}, 400),
            "not JSON": ({"content": b"hello", "headers": {"content-type": "application/json"}}, 400),
            "a cross-site origin": ({"json": {"input": [USER]}, "headers": {"Origin": "http://evil.example"}}, 403),
        }
        for name, (kwargs, status) in cases.items():
            with self.subTest(name):
                chat = FakeChat(answer("never"))
                with self.served(chat) as (http, _):
                    self.assertEqual(http.post("/v1/live/delegate", **kwargs).status_code, status)
                self.assertEqual(chat.requests, [])


if __name__ == "__main__":
    unittest.main()
