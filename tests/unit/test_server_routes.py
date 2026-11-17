# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103

"""HTTP route behavior for the application server."""

from __future__ import annotations

import os
import socket
import tempfile
import unittest
import urllib.error
from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import examples_registry
import server
from utils import clear_service_context, set_active_services

# A single-gpu catalog whose first backend entry for Frontend/Backend Live (thinker-llm: super, then lightning) is down.
REALTIME_CATALOG = dedent(
    """\
    single-gpu:
      llm:
        nemotron-super:
          model_id: org/super
          base_url: http://super.test:8000/v1
          health_path: /health
        nemotron-lightning:
          model_id: org/lightning
          base_url: http://lightning.test:8000/v1
          health_path: /health
      asr:
        nemotron-asr-streaming-english:
          server: asr.test:50051
          model: asr-model
        parakeet-ctc:
          server: ctc.test:50051
          model: ctc-model
      tts:
        magpie-multilingual-tts:
          server: tts.test:50051
          model: magpie-tts-multilingual
          voice_id: Magpie.EN-US.Aria
    """
)
DOWN = frozenset({"super.test"})


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


class RealtimeRouteTests(unittest.TestCase):
    """WS ``/v1/realtime`` on its catalog's services, reached through the network calls of their probes."""

    def setUp(self) -> None:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (directory / "services.yaml").write_text(REALTIME_CATALOG, encoding="utf-8")
        (directory / "settings.yaml").write_text("{}\n", encoding="utf-8")
        env = {"SERVICES_PATH": str(directory / "services.yaml"), "SERVICE_RECIPE": "single-gpu", "NVIDIA_API_KEY": ""}
        self.enterContext(patch.dict(os.environ, {**env, "APP_RUNTIME": "container", "LIVE_ENABLED": "false"}))
        self.probed: list[str] = []
        self.bodies: list[dict] = []
        self.down: frozenset[str] = DOWN
        self.enterContext(patch("urllib.request.urlopen", self.urlopen))
        self.enterContext(patch("socket.create_connection", self.connect))
        self.enterContext(patch("examples.shared.prewarm.prewarm_tts", return_value={}))
        self.enterContext(patch.object(examples_registry, "resolve_bot", lambda example: self.bot))

    def tearDown(self) -> None:
        clear_service_context()
        set_active_services(None)

    def urlopen(self, url: str, timeout: float | None = None) -> MagicMock:
        host = urlparse(url).hostname
        self.probed.append(host)
        if host in self.down:
            raise urllib.error.URLError(socket.gaierror())  # an unresolvable host: the service is not running
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"status":"ready"}'
        return response

    def connect(self, address: tuple[str, int], timeout: float | None = None) -> MagicMock:
        self.probed.append(address[0])
        return MagicMock()

    async def bot(self, runner_args) -> None:
        self.bodies.append(runner_args.body)

    def start_bot(self, nvidia: dict) -> dict:
        """Start a Realtime session whose ``session.nvidia`` is ``nvidia``; return the body its bot starts with."""
        with TestClient(server.create_app()) as client, client.websocket_connect("/v1/realtime") as ws:
            self.assertEqual(ws.receive_json()["type"], "session.created")
            ws.send_json({"type": "session.update", "session": {"type": "realtime", "nvidia": nvidia}})
            self.assertEqual(ws.receive_json()["type"], "session.updated")
            with self.assertRaises(WebSocketDisconnect):
                ws.receive_json()  # the route closes the socket once its bot returns
        [body] = self.bodies
        return body

    def test_a_slot_the_session_cannot_name_starts_on_the_entry_the_ui_selects(self) -> None:
        # Its readiness check passes on that entry; an entry the session names itself is kept.
        for down, selected in ((DOWN, "lightning"), (frozenset(), "super")):
            with self.subTest(down=sorted(down)):
                server._seen_service_targets.clear()
                server._failed_service_targets.clear()
                self.down, self.bodies = down, []
                body = self.start_bot({"pipeline_mode": "frontend-backend-live", "asr_id": "self-hosted:parakeet-ctc"})
                backend = (body["thinker_llm_id"], body["thinker_base_url"])
                self.assertEqual(backend, (f"self-hosted:nemotron-{selected}", f"http://{selected}.test:8000/v1"))
                self.assertEqual((body["asr_id"], body["asr_server"]), ("self-hosted:parakeet-ctc", "ctc.test:50051"))

    def test_an_example_whose_slots_a_session_can_name_lists_no_services(self) -> None:
        # Only the entries it starts on are probed; a listing would probe each entry of its slots.
        body = self.start_bot({"pipeline_mode": "generic-assistant"})
        self.assertEqual(sorted(set(self.probed)), ["asr.test", "lightning.test", "tts.test"])
        self.assertNotIn("thinker_llm_id", body)
