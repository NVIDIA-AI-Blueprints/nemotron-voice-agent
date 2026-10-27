# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""What ``GET /v1/live/info`` reports: the protocol's voices, the session limit and the root's ``describe()``."""

import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from live.gateway import LiveSettings
from live.mount import mount_live
from tests.unit.live_helpers import FakeAdapter


async def factory(config, transport, connection):
    return FakeAdapter().parts()


class InfoTests(unittest.TestCase):
    def info(self, describe=None):
        app = FastAPI()
        mount_live(app, factory=factory, describe=describe, settings=LiveSettings(max_sessions=3))
        with TestClient(app) as client:
            return client.get("/v1/live/info").json()

    def test_info_adds_what_the_roots_describe_hook_says_to_the_voices_and_the_session_limit(self):
        async def describe():
            return {"frontend": {"model": "talker"}, "warnings": []}

        info = self.info(describe)
        self.assertIn("marin", info["voices"])
        self.assertEqual(
            {key: info[key] for key in ("max_sessions", "frontend", "warnings")},
            {"max_sessions": 3, "frontend": {"model": "talker"}, "warnings": []},
        )

    def test_without_a_describe_hook_info_has_the_protocols_part_only(self):
        self.assertEqual(set(self.info()), {"voices", "max_sessions"})

    def test_info_needs_the_bearer_token_when_one_is_configured(self):
        app = FastAPI()
        with patch.dict(os.environ, {"LIVE_API_TOKEN": "secret"}):
            mount_live(app, factory=factory)
        with TestClient(app) as client:
            self.assertEqual(client.get("/v1/live/info").status_code, 401)
            self.assertEqual(client.get("/v1/live/info", headers={"Authorization": "Bearer secret"}).status_code, 200)


if __name__ == "__main__":
    unittest.main()
