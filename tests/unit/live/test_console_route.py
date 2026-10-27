# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""The console route's limits: rounds at once, served models, body size and the ``Host`` header.

With no token, the route answers only an IP literal, ``localhost`` or a name in ``LIVE_ALLOWED_HOSTS``, which refuses
a page on another site whose name resolves to this server (DNS rebinding).
"""

import asyncio
import os
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from live.gateway import LiveSettings
from live.gateway.security import host_name
from live.mount import mount_live
from tests.unit.live_helpers import FakeAdapter

INPUT = [{"role": "user", "content": [{"type": "input_text", "text": "What teas do you have?"}]}]
DONE = {"type": "response.completed", "response": {"id": "resp_1", "model": "m", "output": []}}


class Round:
    """A round's stream: it waits for ``go`` (when given), then completes."""

    def __init__(self, go=None):
        self.go, self.done = go, False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.done:
            raise StopAsyncIteration
        if self.go is not None:
            await self.go.wait()
        self.done = True
        return DONE

    async def aclose(self):
        pass


class ConsoleCore(FakeAdapter):
    """The gateway's core for console rounds; ``go`` holds every round until it is set."""

    def __init__(self, go=None):
        super().__init__()
        self.go = go
        self.rounds: list[dict] = []
        self.checked_models: list[str] = []

    async def retrieve_model(self, model):
        self.checked_models.append(model)
        return await super().retrieve_model(model)

    async def create_response(self, params):
        self.rounds.append(params)
        return Round(self.go)


def app_with(core, settings=None):
    async def no_session(config, transport, connection):
        raise AssertionError("no session is started here")

    app = FastAPI()
    mount_live(app, factory=no_session, core=core.core, settings=settings or LiveSettings())
    return app


class LimitCase(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"LIVE_ENABLED": "true", "LIVE_API_TOKEN": ""})
        env.start()
        self.addCleanup(env.stop)

    def post(self, body, host="127.0.0.1", settings=None, core=None, headers=None):
        self.core = core or ConsoleCore()
        with TestClient(app_with(self.core, settings)) as client:
            return client.post("/v1/live/delegate", json=body, headers={"host": host, **(headers or {})})


class CapTests(unittest.IsolatedAsyncioTestCase):
    async def test_over_the_cap_gives_429_and_a_finished_round_frees_its_place(self):
        go = asyncio.Event()
        core = ConsoleCore(go)
        app = app_with(core, LiveSettings(max_console_rounds=1))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as http:
            first = asyncio.create_task(http.post("/v1/live/delegate", json={"input": INPUT}))
            async with asyncio.timeout(2):
                while not core.rounds:
                    await asyncio.sleep(0.005)
            over = await http.post("/v1/live/delegate", json={"input": INPUT})
            go.set()
            done = await first
            again = await http.post("/v1/live/delegate", json={"input": INPUT})
        self.assertEqual((over.status_code, over.json()["detail"]), (429, "Too many console rounds: at most 1 at once"))
        self.assertEqual((done.status_code, again.status_code), (200, 200))
        self.assertEqual(len(core.rounds), 2)  # the refused one ran nothing

    def test_the_cap_is_an_operator_setting(self):
        with patch.dict(os.environ, {"LIVE_MAX_CONSOLE_ROUNDS": "4"}):
            self.assertEqual(LiveSettings.from_env().max_console_rounds, 4)
        with patch.dict(os.environ, {"LIVE_MAX_CONSOLE_ROUNDS": "0"}), self.assertRaises(ValueError):
            LiveSettings.from_env()


class ModelTests(LimitCase):
    def test_an_unknown_model_gives_400_and_runs_no_round(self):
        response = self.post({"input": INPUT, "model": "bad-model"})
        self.assertEqual((response.status_code, response.json()["detail"]), (400, "Unknown model 'bad-model'"))
        self.assertEqual((self.core.checked_models, self.core.rounds), (["bad-model"], []))

    def test_a_served_model_is_checked_then_used(self):
        response = self.post({"input": INPUT, "model": "org/other"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual((self.core.checked_models, self.core.rounds[0]["model"]), (["org/other"], "org/other"))

    def test_with_no_model_the_core_uses_its_own_and_nothing_is_checked(self):
        self.assertEqual(self.post({"input": INPUT}).status_code, 200)
        self.assertEqual(self.core.checked_models, [])


class HostTests(LimitCase):
    def test_a_foreign_name_is_refused_and_runs_no_round(self):
        response = self.post({"input": INPUT}, host="evil.example")
        self.assertEqual(response.status_code, 403)
        self.assertIn("LIVE_ALLOWED_HOSTS", response.json()["detail"])
        self.assertEqual(self.core.rounds, [])

    def test_an_ip_literal_or_localhost_is_accepted(self):
        for host in ("127.0.0.1", "10.0.0.5:8000", "[::1]:8000", "localhost:7860", "LOCALHOST"):
            with self.subTest(host=host):
                self.assertEqual(self.post({"input": INPUT}, host=host).status_code, 200)

    def test_an_allowed_name_is_accepted(self):
        settings = LiveSettings(allowed_hosts=("console.lab",))
        self.assertEqual(self.post({"input": INPUT}, host="Console.Lab:8000", settings=settings).status_code, 200)
        with patch.dict(os.environ, {"LIVE_ALLOWED_HOSTS": " console.lab , Other.Lab "}):
            self.assertEqual(LiveSettings.from_env().allowed_hosts, ("console.lab", "other.lab"))

    def test_with_a_token_the_token_decides_not_the_name(self):
        settings = LiveSettings(api_token="secret")
        authorized = {"Authorization": "Bearer secret"}
        self.assertEqual(self.post({"input": INPUT}, "evil.example", settings, headers=authorized).status_code, 200)
        self.assertEqual(self.post({"input": INPUT}, "evil.example", settings).status_code, 401)

    def test_the_host_name_drops_its_port(self):
        self.assertEqual(
            [host_name(h) for h in ("app:8000", "[::1]:8000", "::1", "10.0.0.5", "")],
            ["app", "::1", "::1", "10.0.0.5", ""],
        )


class BodyTests(LimitCase):
    def test_a_body_over_the_console_limit_is_a_413_and_runs_no_round(self):
        settings = LiveSettings(console_body_bytes=1024)
        big = [{"role": "user", "content": [{"type": "input_text", "text": "x" * 2048}]}]
        response = self.post({"input": big}, settings=settings)
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.core.rounds, [])

    def test_the_console_limit_is_smaller_than_the_session_limit(self):
        from live.gateway.security import MAX_BODY_BYTES

        self.assertLess(LiveSettings().console_body_bytes, MAX_BODY_BYTES)


if __name__ == "__main__":
    unittest.main()
