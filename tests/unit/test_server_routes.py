# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103

"""HTTP route behavior for the application server."""

from __future__ import annotations

import os
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

    def test_transcribe_language_endpoint_uses_static_catalog_without_grpc(self) -> None:
        transcribe = {"asr_type": "transcribe", "supported_languages": ["de", "hi", "es"]}
        with (
            patch.object(server, "load_service_entry_by_id", return_value=transcribe),
            patch.object(server, "load_service_entry", return_value={"supported_languages": ["de", "es"]}),
            patch("examples.shared.prewarm.prewarm_tts", return_value={"languages": ["de-DE", "hi-IN", "fr-FR"]}),
            patch("examples.shared.prewarm.prewarm_asr") as probe,
            TestClient(server.create_app()) as client,
        ):
            response = client.get(
                "/api/tts-config",
                params={
                    "pipeline_mode": "multilingual-assistant",
                    "asr_id": "self-hosted:nemotron-transcribe",
                    "asr_server": "http://nim/v1",
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["languages"], ["de-DE"])
        probe.assert_not_called()

    def test_transcribe_api_defaults_to_session_locale_and_preserves_explicit_locale(self) -> None:
        with patch.dict(os.environ, {"SERVICE_RECIPE": "server"}):
            for mode, requested, expected in (
                ("multilingual-assistant", None, "de-DE"),
                ("multilingual-assistant", "", "de-DE"),
                ("multilingual-assistant", "es-ES", "es-ES"),
                ("generic-assistant", None, "en-US"),
            ):
                with self.subTest(mode=mode, language=requested):
                    body = {"pipeline_mode": mode, "asr_id": "self-hosted:nemotron-transcribe"}
                    if requested is not None:
                        body["asr_language_code"] = requested
                    config = server._sanitize_session_config(body)
                    self.assertEqual(config["asr_language_code"], expected)

    def test_language_endpoint_probes_selected_asr_when_only_id_is_supplied(self) -> None:
        selected = {"server": "selected:50052", "model": "selected-model", "function_id": "selected-function"}
        with (
            patch.object(server, "load_service_entry_by_id", return_value=selected),
            patch("examples.shared.prewarm.prewarm_tts", return_value={"languages": ["de-DE"]}),
            patch("examples.shared.prewarm.prewarm_asr", return_value={"languages": ["de-DE"]}) as probe,
            TestClient(server.create_app()) as client,
        ):
            response = client.get(
                "/api/tts-config",
                params={"pipeline_mode": "multilingual-assistant", "asr_id": "self-hosted:parakeet-rnnt"},
            )
        self.assertEqual(response.status_code, 200)
        probe.assert_called_once_with("selected:50052", "selected-model", "selected-function")
