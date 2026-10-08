# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""``POST /v1/live/delegate``: one backend round for a client that owns its delegation."""

import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from live.mount import mount_live

FUNCTION_CALL = {"type": "function_call", "call_id": "call_1", "name": "get_menu", "arguments": '{"category":"tea"}'}
TOOLS = [{"type": "function", "name": "get_menu", "parameters": {"type": "object", "properties": {}}}]
INPUT = [{"role": "user", "content": [{"type": "input_text", "text": "What teas do you have?"}]}]


class FakeThinker:
    """Stands in for the backend: plays the events of one round, and records what it was asked."""

    def __init__(self, events):
        self.events, self.requests, self.closed = events, [], False
        self.backend = self

    async def stream(self, request):
        self.requests.append(request)
        for event in self.events:
            yield event

    async def close(self):
        self.closed = True


def round_with(*items, finish="response.completed"):
    return [
        *({"type": "response.output_item.done", "item": item} for item in items),
        {"type": finish, "response": {"id": "r1"}},
    ]


class DelegateRouteTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"LIVE_ENABLED": "true", "LIVE_API_TOKEN": ""})
        env.start()
        self.addCleanup(env.stop)

    def post(self, body, events=None, headers=None):
        thinker = FakeThinker(events if events is not None else round_with(FUNCTION_CALL))
        app = FastAPI()
        with patch("examples.frontend_backend_live.engine.factory.build_thinker", return_value=thinker):
            mount_live(app)
            with TestClient(app) as client:
                response = client.post("/v1/live/delegate", json=body, headers=headers or {})
        return response, thinker

    def test_a_round_returns_its_output_items_so_the_client_can_run_the_function_calls(self):
        response, thinker = self.post({"input": INPUT, "tools": TOOLS})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["response"], {"id": "r1", "output": [FUNCTION_CALL]})
        self.assertEqual(body["sent_request"]["tools"], ["get_menu"])
        request = thinker.requests[0]
        self.assertEqual(request["input"], INPUT)
        self.assertIn("Bluebird Cafe", request["instructions"])  # the example's backend prompt when none is sent
        self.assertTrue(thinker.closed)

    def test_the_client_can_choose_the_model_and_the_instructions(self):
        _, thinker = self.post({"input": INPUT, "model": "org/other", "instructions": "Be terse."})
        self.assertEqual(
            (thinker.requests[0]["model"], thinker.requests[0]["instructions"]), ("org/other", "Be terse.")
        )

    def test_a_bad_request_is_a_400(self):
        for body in ({}, {"input": []}, {"input": ["text"]}, {"input": INPUT, "tools": [{"type": "function"}]}):
            with self.subTest(body=body):
                self.assertEqual(self.post(body)[0].status_code, 400)

    def test_a_round_the_backend_could_not_finish_is_a_502_without_its_detail(self):
        response, _ = self.post({"input": INPUT}, events=round_with(finish="response.failed"))
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("response.failed", response.text)

    def test_it_needs_the_bearer_token_when_one_is_configured(self):
        with patch.dict(os.environ, {"LIVE_API_TOKEN": "secret"}):
            self.assertEqual(self.post({"input": INPUT})[0].status_code, 401)
            ok = self.post({"input": INPUT}, headers={"Authorization": "Bearer secret"})[0]
        self.assertEqual(ok.status_code, 200)

    def test_it_is_not_served_when_the_live_routes_are_not(self):
        app = FastAPI()
        with patch.dict(os.environ, {"LIVE_ENABLED": "false"}):
            mount_live(app)
        with TestClient(app) as client:
            self.assertEqual(client.post("/v1/live/delegate", json={"input": INPUT}).status_code, 404)

    def test_an_engine_without_the_hook_does_not_serve_it(self):
        async def factory(config, transport, connection):
            raise AssertionError("no session is started here")

        app = FastAPI()
        mount_live(app, factory)
        with TestClient(app) as client:
            self.assertEqual(client.post("/v1/live/delegate", json={"input": INPUT}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
