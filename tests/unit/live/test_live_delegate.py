# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""``POST /v1/live/delegate`` on a fake core: one stateless round for a client that owns its delegation."""

import asyncio
import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from live.gateway import LiveSettings
from live.mount import mount_live
from tests.unit.live_helpers import FakeAdapter

FUNCTION_CALL = {"type": "function_call", "call_id": "call_1", "name": "get_menu", "arguments": '{"category":"tea"}'}
TOOLS = [{"type": "function", "name": "get_menu", "parameters": {"type": "object", "properties": {}}}]
INPUT = [{"role": "user", "content": [{"type": "input_text", "text": "What teas do you have?"}]}]


def round_with(*items, finish="response.completed"):
    response = {"id": "resp_1", "model": "org/backend", "status": "completed", "output": list(items)}
    return [{"type": "response.output_item.done", "item": item} for item in items] + [
        {"type": finish, "response": response}
    ]


class RoundStream:
    """A console round's stream: its events, an optional wait, and whether the route closed it."""

    def __init__(self, events, delay=0.0):
        self.events, self.delay, self.closed = list(events), delay, False

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(self.delay)
        if not self.events:
            raise StopAsyncIteration
        return self.events.pop(0)

    async def aclose(self):
        self.closed = True


class RoundCore(FakeAdapter):
    """The gateway's core: it answers each console round with ``events``, or refuses it with ``refusal``."""

    def __init__(self, events=None, refusal=None, delay=0.0):
        super().__init__()
        self.events = round_with(FUNCTION_CALL) if events is None else events
        self.refusal, self.delay = refusal, delay
        self.requests: list[dict] = []
        self.streams: list[RoundStream] = []

    async def create_response(self, params):
        self.requests.append(params)
        if self.refusal is not None:
            raise self.refusal
        self.streams.append(RoundStream(self.events, self.delay))
        return self.streams[-1]


async def no_session(config, transport, connection):
    raise AssertionError("no session is started here")


class DelegateRouteTests(unittest.TestCase):
    def setUp(self):
        # The test client sends Host: testserver, a name the console route answers to only when it is allowed.
        env = patch.dict(os.environ, {"LIVE_ENABLED": "true", "LIVE_API_TOKEN": "", "LIVE_ALLOWED_HOSTS": "testserver"})
        env.start()
        self.addCleanup(env.stop)

    def post(self, body, core=None, headers=None, settings=None):
        self.core = core if core is not None else RoundCore()
        app = FastAPI()
        mount_live(app, factory=no_session, core=self.core.core, settings=settings)
        with TestClient(app) as client:
            return client.post("/v1/live/delegate", json=body, headers=headers or {})

    def test_a_round_returns_its_output_items_so_the_client_can_run_the_function_calls(self):
        response = self.post({"input": INPUT, "tools": TOOLS})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "response": {"id": "resp_1", "output": [FUNCTION_CALL]},
                "sent_request": {"model": "org/backend", "tools": ["get_menu"]},
            },
        )
        [request] = self.core.requests
        self.assertEqual(
            request,
            {"input": INPUT, "tools": TOOLS, "tool_choice": "auto", "stream": True, "store": False},
        )  # stateless: no conversation; the core fills the model and the instructions
        self.assertTrue(self.core.streams[0].closed)

    def test_the_client_can_choose_the_model_and_the_instructions(self):
        self.post({"input": INPUT, "model": "org/other", "instructions": "Be terse."})
        request = self.core.requests[0]
        self.assertEqual((request["model"], request["instructions"]), ("org/other", "Be terse."))

    def test_a_bad_request_is_a_400_and_runs_no_round(self):
        for body, detail in (
            ({}, "input must be a non-empty list of items"),
            ({"input": []}, "input must be a non-empty list of items"),
            ({"input": ["text"]}, "input must be a non-empty list of items"),
            ({"input": INPUT, "tools": [{"type": "function"}]}, "tools must be function or web_search tools"),
            ({"input": INPUT, "model": 7}, "model must be a string"),
        ):
            with self.subTest(body=body):
                response = self.post(body)
                self.assertEqual((response.status_code, response.json()["detail"]), (400, detail))
                self.assertEqual(self.core.requests, [])

    def test_a_request_the_core_refuses_is_a_400_with_its_reason(self):
        response = self.post({"input": INPUT}, RoundCore(refusal=ValueError("A request may nest at most 64 levels")))
        self.assertEqual(
            (response.status_code, response.json()["detail"]), (400, "A request may nest at most 64 levels")
        )

    def test_a_round_the_backend_could_not_finish_is_a_502_without_its_detail(self):
        for finish in ("response.failed", "response.incomplete"):
            with self.subTest(finish=finish):
                response = self.post({"input": INPUT}, RoundCore(events=round_with(finish=finish)))
                self.assertEqual(response.status_code, 502)
                self.assertNotIn(finish, response.text)
                self.assertTrue(self.core.streams[0].closed)
        failing = self.post({"input": INPUT}, RoundCore(refusal=RuntimeError("/srv/config.yaml is unreadable")))
        self.assertEqual(
            (failing.status_code, failing.json()["detail"]), (502, "The backend could not complete the round")
        )

    def test_a_round_past_the_backstop_is_a_502_and_its_work_is_stopped(self):
        settings = LiveSettings(console_round_seconds=0.05, allowed_hosts=("testserver",))
        response = self.post({"input": INPUT}, RoundCore(delay=1.0), settings=settings)
        self.assertEqual(response.status_code, 502)
        self.assertTrue(self.core.streams[0].closed)

    def test_a_server_with_no_core_answers_503_without_detail(self):
        app = FastAPI()
        mount_live(app, factory=no_session)
        with TestClient(app) as client:
            response = client.post("/v1/live/delegate", json={"input": INPUT})
        self.assertEqual(
            (response.status_code, response.json()["detail"]), (503, "The delegation core is not available")
        )

    def test_it_needs_the_bearer_token_when_one_is_configured(self):
        with patch.dict(os.environ, {"LIVE_API_TOKEN": "secret"}):
            self.assertEqual(self.post({"input": INPUT}).status_code, 401)
            ok = self.post({"input": INPUT}, headers={"Authorization": "Bearer secret"})
        self.assertEqual(ok.status_code, 200)

    def test_it_is_not_served_when_the_live_routes_are_not(self):
        app = FastAPI()
        with patch.dict(os.environ, {"LIVE_ENABLED": "false"}):
            mount_live(app)
        with TestClient(app) as client:
            self.assertEqual(client.post("/v1/live/delegate", json={"input": INPUT}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
