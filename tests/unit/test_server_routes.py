# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103

"""HTTP route behavior for the application server."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import server
from utils import clear_service_context, set_active_services


class ServerRouteTests(unittest.TestCase):
    """Routes that must remain reachable when the client is built."""

    def tearDown(self) -> None:
        clear_service_context()
        set_active_services(None)

    def test_health_returns_json_when_client_dist_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client_dist = Path(directory)
            (client_dist / "index.html").write_text("<html>client</html>")
            with patch.object(server, "CLIENT_DIST", client_dist):
                response = TestClient(server.create_app()).get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/json")
        self.assertEqual(response.json(), {"status": "ok"})

    def test_session_config_rejects_non_string_service_ids(self) -> None:
        with TestClient(server.create_app()) as client:
            for field, value in (("llm_id", {"bad": "value"}), ("asr_id", 123), ("tts_id", True)):
                with self.subTest(field=field):
                    response = client.post("/api/session-config", json={field: value})

                    self.assertEqual(response.status_code, 400)
                    self.assertEqual(response.json(), {"detail": f"{field} must be a string"})
