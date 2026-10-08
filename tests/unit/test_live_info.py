# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

"""What `GET /v1/live/info` reports for the registered example."""

import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from live.mount import mount_live


class DefaultEngineInfoTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"LIVE_ENABLED": "true"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_valid_configuration_reports_no_warnings(self):
        app = FastAPI()
        mount_live(app)
        with TestClient(app) as client:
            self.assertEqual(client.get("/v1/live/info").json()["warnings"], [])

    def test_info_includes_what_the_default_engine_describes(self):
        app = FastAPI()
        mount_live(app)
        with TestClient(app) as client:
            info = client.get("/v1/live/info").json()
        self.assertEqual({"provider", "slot", "model", "base_url", "prompt"}, set(info["frontend"]))
        self.assertEqual(info["backend"]["provider"], "chat-completions")
        self.assertEqual([t["name"] for t in info["sample_tools"]], ["get_menu", "manage_cart", "place_order"])
        self.assertNotIn("api_key", str(info).lower())


if __name__ == "__main__":
    unittest.main()
