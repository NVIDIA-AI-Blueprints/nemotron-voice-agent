# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""``POST /v1/live/delegate`` on the server's app, through the root's delegation core to its backend LLM."""

import json
import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from fastapi.testclient import TestClient

import examples_registry
import server
from examples.frontend_backend_live import factory
from examples.frontend_backend_live.delegation_core.backend.providers import base as providers
from examples.frontend_backend_live.delegation_core.config import CoreSettings
from examples.frontend_backend_live.delegation_core.im.input import compose
from examples.frontend_backend_live.prompts import load_prompts
from tests.unit.cascade_helpers import MENU_TOOL
from tests.unit.core_helpers import CORE_BLOCK, CORE_TEXTS, EXAMPLE_SETTINGS, FakeChat, chunk
from tests.unit.frontend_helpers import clear_root_caches, tool_call
from utils import clear_service_context, set_active_services

EXAMPLE = "frontend-backend-live"
USER = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What teas do you have?"}]}
LATER = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "And coffee?"}]}
FAILED = (502, "The backend could not complete the round")


def answer(text):
    return [chunk(text), chunk(finish="stop")]


class RouteCase(unittest.TestCase):
    """The route as the server serves it, on a core the root builds fresh for each ``served`` block."""

    def setUp(self):
        # The catalog's self-hosted section, every service in it up and no NVIDIA Cloud key, so no probe picks another.
        env = {"LIVE_ENABLED": "true", "LIVE_API_TOKEN": "", "SERVICE_RECIPE": "server", "NVIDIA_API_KEY": ""}
        self.enterContext(patch.dict(os.environ, env))
        os.environ.pop("SERVICES_PATH", None)
        self.enterContext(patch.object(server, "_probe_status", lambda address, expects_ready_json: "ready"))
        selection = examples_registry.Selection(EXAMPLE, True, (EXAMPLE,), EXAMPLE)
        self.enterContext(patch.object(examples_registry, "_SELECTION", selection))
        hosts = {"allowed_hosts": ["testserver"]}
        self.enterContext(patch.dict(examples_registry.EXAMPLES[EXAMPLE]["live"]["settings"], hosts))
        # create_app binds the selected example's services process-wide; leave no binding behind.
        self.addCleanup(clear_service_context)
        self.addCleanup(set_active_services, None)

    @contextmanager
    def served(self, client, *, settings=None):
        """Yield the server's test client on a new core, with its model clients as ``client``, and what it was sent."""
        clear_root_caches(self)
        sent, make = [], factory.core_client

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
            patch.dict(examples_registry.EXAMPLES[EXAMPLE], {"config": settings or EXAMPLE_SETTINGS}),
            patch.object(providers, "AsyncOpenAI", return_value=client),
            patch.object(factory, "core_client", spied),
            TestClient(server.create_app()) as http,
        ):
            yield http, sent

    def round(self, body, *script, settings=None, chat=None):
        """Run one delegate round on a fresh core and return the reply, the model client and what the core got."""
        chat = chat or FakeChat(*script)
        with self.served(chat, settings=settings) as (http, sent):
            response = http.post("/v1/live/delegate", json=body)
        return response, chat, sent


class RouteTests(RouteCase):
    def assert_stateless(self, sent, items):
        (request,) = sent
        self.assertNotIn("conversation", request)
        self.assertNotIn("previous_response_id", request)
        self.assertEqual(request["input"], items)
        self.assertEqual(self.service.sessions, {})

    def test_a_resent_history_with_outputs_reaches_the_model(self):
        body = {"input": [USER], "tools": [MENU_TOOL]}
        response, chat, sent = self.round(body, tool_call("get_menu", '{"category":"tea"}'))
        self.assertEqual(response.status_code, 200, response.text)
        self.assert_stateless(sent, [USER])
        reply = response.json()
        call = reply["response"]["output"][0]
        self.assertEqual((call["type"], call["name"]), ("function_call", "get_menu"))
        self.assertTrue(reply["response"]["id"])
        self.assertEqual(reply["sent_request"]["tools"], ["get_menu"])
        self.assertEqual([m["role"] for m in chat.requests[0]["messages"]], ["system", "user"])
        self.assertTrue(chat.closed)
        output = {"type": "function_call_output", "call_id": call["call_id"], "output": '{"items": ["green tea"]}'}
        body = {"input": [USER, call, output], "tools": [MENU_TOOL]}
        response, chat, sent = self.round(body, answer("We have green tea and chai."))
        self.assert_stateless(sent, [USER, call, output])
        message = response.json()["response"]["output"][0]
        self.assertEqual(message["type"], "message")
        messages = chat.requests[0]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "tool"])
        self.assertEqual(messages[2]["tool_calls"][0]["id"], call["call_id"])
        self.assertEqual((messages[3]["tool_call_id"], messages[3]["content"]), (call["call_id"], output["output"]))
        _, chat, _ = self.round({"input": [USER, call, output, message, LATER]}, answer("We have espresso."))
        self.assertEqual(len(chat.requests), 1)
        roles = ["system", "user", "assistant", "tool", "assistant", "user"]
        self.assertEqual([m["role"] for m in chat.requests[0]["messages"]], roles)

    def test_the_core_sends_the_clients_choices_and_fills_the_rest(self):
        chat = FakeChat(answer("Hello."), answer("Hello."))
        with self.served(chat) as (http, _):
            info = http.get("/v1/live/info").json()
            default = http.post("/v1/live/delegate", json={"input": [USER]}).json()
            model = next(name for name in info["backend_models"] if name != info["backend"]["model"])
            tools = [MENU_TOOL, {"type": "web_search"}]
            body = {"input": [USER], "model": model, "instructions": "Be terse.", "tools": tools}
            chosen = http.post("/v1/live/delegate", json=body).json()
        first, second = chat.requests
        texts = CoreSettings.parse(CORE_BLOCK, CORE_TEXTS).texts
        notes = texts.caller_note, texts.voice_note
        prompt = load_prompts().text("Backend", info["backend"]["prompt"])
        self.assertEqual((first["model"], default["sent_request"]["model"]), (info["backend"]["model"],) * 2)
        self.assertEqual(first["messages"][0]["content"], compose(prompt, *notes))
        self.assertNotIn("tools", first)
        # The reply names a hosted tool by its type; the chat-completions backend gets only the function tools.
        self.assertEqual((second["model"], chosen["sent_request"]["model"]), (model, model))
        self.assertEqual(second["messages"][0]["content"], compose("Be terse.", *notes))
        self.assertEqual(chosen["sent_request"]["tools"], ["get_menu", "web_search"])
        self.assertEqual([tool["function"]["name"] for tool in second["tools"]], ["get_menu"])

    def test_the_servers_shutdown_closes_the_roots_core(self):
        with self.served(FakeChat(answer("Hello."))) as (http, _):
            self.assertEqual(http.post("/v1/live/delegate", json={"input": [USER]}).status_code, 200)
            released = self.enterContext(patch.object(factory.delegation_core().source, "aclose"))
            released.assert_not_awaited()
        released.assert_awaited_once()

    def test_a_body_the_route_refuses_reaches_no_model(self):
        # Each body is POSTed as written: Python's JSON reader also takes NaN and lone surrogates.
        nested = "[" * 600 + "1" + "]" * 600
        cases = {
            "NaN": ('{"input": [{"type": "function_call_output", "call_id": "c", "output": NaN}]}', None),
            "a lone surrogate": ('{"input": [{"role": "user", "content": "\\ud800"}]}', None),
            "nested too deeply": (
                f'{{"input": [{{"role": "user", "content": "hi", "meta": {nested}}}]}}',
                "A request may nest objects and lists at most 64 levels deep",
            ),
            "not JSON": (b"hello", None),
        }
        for name, (body, detail) in cases.items():
            with self.subTest(name):
                chat = FakeChat(answer("never"))
                with self.served(chat) as (http, _):
                    headers = {"content-type": "application/json"}
                    response = http.post("/v1/live/delegate", content=body, headers=headers)
                self.assertEqual(response.status_code, 400, response.text)
                if detail:
                    self.assertEqual(response.json()["detail"], detail)
                self.assertEqual(chat.requests, [])


class Event(dict):
    """A Responses streaming event as the OpenAI client yields it."""

    def model_dump(self, **kwargs):
        return dict(self)


def response(status, *output):
    return {"id": "resp_up", "object": "response", "created_at": 1, "model": "m", "status": status, "output": [*output]}


def message(item_id, text):
    content = [{"type": "output_text", "text": text, "annotations": []}]
    return {"type": "message", "id": item_id, "role": "assistant", "status": "completed", "content": content}


def added(item_id):
    item = {"type": "message", "id": item_id, "role": "assistant", "status": "in_progress", "content": []}
    return {"type": "response.output_item.added", "output_index": 0, "item": item}


def text_event(kind, item_id, **fields):
    return {"type": f"response.{kind}", "item_id": item_id, "output_index": 0, "content_index": 0, **fields}


def done(item):
    return {"type": "response.output_item.done", "output_index": 0, "item": item}


def completed(*output):
    return {"type": "response.completed", "response": response("completed", *output)}


CREATED = {"type": "response.created", "response": response("in_progress")}
TEA, MORE = message("msg_1", "We have chai."), message("msg_2", "And green tea.")


class ResponsesStreamTests(RouteCase):
    def setUp(self):
        super().setUp()
        read = factory.ExampleCatalog.read

        def speaking_responses():
            """Return the catalog with every backend entry naming the Responses API."""
            catalog = read()
            for entry in catalog.offered.get("thinker-llm", []):
                entry["api"] = "openai-responses"
            return catalog

        self.enterContext(patch.object(factory.ExampleCatalog, "read", speaking_responses))

    def play(self, events, body=None):
        client = FakeChat([Event(event) for event in events])
        client.responses = client  # its one ``responses.create`` streams ``events``
        reply, _, _ = self.round(body or {"input": [USER]}, chat=client)
        self.assertEqual(len(client.requests), 1)
        self.assertTrue(client.closed)
        return reply

    def test_the_reply_is_the_items_the_round_finished(self):
        cases = {
            "an added id that differs from the done one": [
                CREATED,
                added("msg_a"),
                text_event("output_text.delta", "msg_a", delta="We have chai."),
                done(TEA),
                completed(TEA),
            ],
            "items after the terminal": [CREATED, done(TEA), completed(TEA), done(MORE)],
        }
        for name, events in cases.items():
            with self.subTest(name):
                reply = self.play(events)
                self.assertEqual(reply.status_code, 200, reply.text)
                [item] = reply.json()["response"]["output"]
                self.assertEqual((item["type"], item["content"][0]["text"]), ("message", "We have chai."))

    def test_a_round_the_backend_llm_cannot_read_fails(self):
        part = {"type": "output_text", "text": "", "annotations": []}
        cases = {
            "a message left open": [
                CREATED,
                added("msg_1"),
                text_event("content_part.added", "msg_1", part=part),
                text_event("output_text.delta", "msg_1", delta="We have"),
                completed(),
            ],
            "a second response.created over an open message": [
                CREATED,
                added("msg_0"),
                text_event("output_text.delta", "msg_0", delta="We"),
                CREATED,
                done(TEA),
                completed(TEA),
            ],
            "an item, then a second response.created": [CREATED, done(MORE), CREATED, done(TEA), completed(TEA)],
            "a stream that ends after an error event": [
                CREATED,
                done(TEA),
                {"type": "error", "code": "server_error", "message": "boom", "param": None},
            ],
            "a failed terminal, then a completed one": [
                CREATED,
                done(TEA),
                {"type": "response.failed", "response": response("failed")},
                completed(TEA),
            ],
        }
        for name, events in cases.items():
            with self.subTest(name):
                reply = self.play(events)
                self.assertEqual((reply.status_code, reply.json()["detail"]), FAILED)

    def test_a_function_call_without_a_call_id_fails_the_round(self):
        # The backend LLM keys each call by its call_id, which a Responses item keeps from its provider.
        call = {"type": "function_call", "id": "fc_1", "name": "get_menu", "arguments": "{}", "status": "completed"}
        reply = self.play([CREATED, done(call), completed()], {"input": [USER], "tools": [MENU_TOOL]})
        self.assertEqual((reply.status_code, reply.json()["detail"]), FAILED)
