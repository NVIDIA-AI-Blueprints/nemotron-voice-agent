# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""A tau2-shaped client over the τ³ head's real FastAPI WebSocket (stub speech, stub-gate runtime)."""

from __future__ import annotations

import base64
import json
import unittest
from typing import Any

from _voice_fakes import pcmu_silence, pcmu_speech, stub_gate_config, stub_services, tau2_session_update
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from examples.frontend_delegation_hermes.adapters.tau.app import build_delegation_app
from examples.frontend_delegation_hermes.adapters.tau.server import ServerOptions
from examples.frontend_delegation_hermes.core.server_common import SHOW_FILLER_PARAM


def _send_audio(ws: Any, audio: bytes) -> None:
    for offset in range(0, len(audio), 160):
        chunk = audio[offset : offset + 160]
        ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode()}))


def _until(ws: Any, kind: str, seen: list[dict]) -> dict:
    while True:
        event = json.loads(ws.receive_text())
        seen.append(event)
        if event["type"] == kind:
            return event


def _app(**server: Any) -> Any:
    return build_delegation_app(
        stub_gate_config(**server), options=ServerOptions(stub_speech=True), services=stub_services()
    )


class WebSocketEndToEndTests(unittest.TestCase):
    def test_turn_with_a_tool_round_trip(self) -> None:
        seen: list[dict] = []
        with TestClient(_app()) as client:
            self.assertEqual(client.get("/health").json()["status"], "ok")
            with client.websocket_connect("/v1/realtime?model=pine-e2e", headers={"Authorization": "Bearer x"}) as ws:
                first = json.loads(ws.receive_text())
                self.assertEqual(first["type"], "session.created")
                ws.send_text(json.dumps(tau2_session_update("mock")))
                _until(ws, "session.updated", seen)
                _send_audio(ws, pcmu_speech(600) + pcmu_silence(1200))
                call = _until(ws, "response.function_call_arguments.done", seen)
                _until(ws, "response.done", seen)
                ws.send_text(
                    json.dumps(
                        {
                            "type": "conversation.item.create",
                            "item": {"type": "function_call_output", "call_id": call["call_id"], "output": "{}"},
                        }
                    )
                )
                ws.send_text(json.dumps({"type": "response.create"}))
                _send_audio(ws, pcmu_silence(400))
                _until(ws, "conversation.item.added", seen)
            health = client.get("/health").json()
        self.assertEqual(health["total_sessions"], 1)
        types = [event["type"] for event in seen]
        self.assertEqual(types.count("input_audio_buffer.speech_started"), 1)
        self.assertNotIn("error", types)
        transcripts = [
            e["transcript"] for e in seen if e["type"] == "conversation.item.input_audio_transcription.completed"
        ]
        self.assertEqual(transcripts, ["utterance 1"])

    def test_health_reports_the_delegation_runtime(self) -> None:
        with TestClient(_app()) as client:
            health = client.get("/health").json()
        self.assertEqual(health["prototype"], "frontend-delegation-hermes")
        self.assertEqual(health["backend"]["link"], "in_process_fake")

    def test_browser_page_is_served_and_asks_for_silent_filler(self) -> None:
        config = stub_gate_config()
        with TestClient(_app()) as client:
            page = client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("text/html", page.headers["content-type"])
        self.assertIn(f"{config.voice.server.path}?model=pine-browser&{SHOW_FILLER_PARAM}=1", page.text)

    def test_bearer_token_is_enforced_when_configured(self) -> None:
        with TestClient(_app(require_bearer=True, bearer_token="s3cret")) as client:
            with (
                self.assertRaises(WebSocketDisconnect),
                client.websocket_connect("/v1/realtime", headers={"Authorization": "Bearer wrong"}) as ws,
            ):
                ws.receive_text()
            with client.websocket_connect("/v1/realtime", headers={"Authorization": "Bearer s3cret"}) as ws:
                self.assertEqual(json.loads(ws.receive_text())["type"], "session.created")

    def test_capacity_refusal_is_server_busy(self) -> None:
        with TestClient(_app(max_sessions=1)) as client, client.websocket_connect("/v1/realtime") as first:
            self.assertEqual(json.loads(first.receive_text())["type"], "session.created")
            with client.websocket_connect("/v1/realtime") as second:
                refused = json.loads(second.receive_text())
                self.assertEqual(refused["error"]["code"], "server_busy")


if __name__ == "__main__":
    unittest.main()
