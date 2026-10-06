# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Rule 19: a tau2-shaped client over a real FastAPI WebSocket (stub speech, scripted agent)."""

from __future__ import annotations

import base64
import json
import unittest
from dataclasses import replace
from typing import Any

from _fbv_voice_fakes import base_config, pcmu_silence, pcmu_speech, tau2_session_update
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from examples.frontend_backend_verdict.voice.server import SHOW_FILLER_PARAM, ServerOptions, build_app
from examples.frontend_backend_verdict.voice.speech.ports import IdentityNormalizer, SpeechServices
from examples.frontend_backend_verdict.voice.speech.stubs import StubRecognizer, ToneSynthesizer
from examples.frontend_backend_verdict.voice.speech.vad_energy import EnergyVad


def _services() -> SpeechServices:
    return SpeechServices(
        recognizer=StubRecognizer(),
        synthesizer=ToneSynthesizer(sample_rate=16000, ms_per_char=5.0),
        vad_factory=EnergyVad,
        normalizer=IdentityNormalizer(),
    )


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


class WebSocketEndToEndTests(unittest.TestCase):
    def test_three_turns_with_one_tool_round_trip(self) -> None:
        config = base_config()
        config = replace(config, server=replace(config.server, warmup=False))
        app = build_app(config, options=ServerOptions(stub_speech=True, stub_agent="scripted"), services=_services())
        seen: list[dict] = []
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").json()["status"], "ok")
            with client.websocket_connect("/v1/realtime?model=pine-e2e", headers={"Authorization": "Bearer x"}) as ws:
                first = json.loads(ws.receive_text())
                self.assertEqual(first["type"], "session.created")
                ws.send_text(json.dumps(tau2_session_update("mock")))
                _until(ws, "session.updated", seen)
                turn = pcmu_speech(600) + pcmu_silence(800)

                _send_audio(ws, turn)  # turn 1: the scripted agent calls a tool
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
                _until(ws, "response.done", seen)

                _send_audio(ws, pcmu_silence(3000) + turn)  # turn 2: an answer
                _until(ws, "response.done", seen)
                _send_audio(ws, pcmu_silence(3000) + turn)  # turn 3: transfer_to_human_agents
                transfer = _until(ws, "response.function_call_arguments.done", seen)
                _until(ws, "response.done", seen)
            health = client.get("/health").json()
        self.assertEqual(transfer["name"], "transfer_to_human_agents")
        self.assertEqual(health["total_sessions"], 1)
        types = [event["type"] for event in seen]
        self.assertEqual(types.count("response.created"), types.count("response.done"))
        self.assertEqual(types.count("input_audio_buffer.speech_started"), 3)
        self.assertNotIn("error", types)
        transcripts = [
            e["transcript"] for e in seen if e["type"] == "conversation.item.input_audio_transcription.completed"
        ]
        self.assertEqual(transcripts, ["utterance 1", "utterance 2", "utterance 3"])

    def test_browser_page_is_served_and_asks_for_silent_filler(self) -> None:
        config = replace(base_config(), server=replace(base_config().server, warmup=False))
        app = build_app(config, options=ServerOptions(stub_speech=True, stub_agent="scripted"), services=_services())
        with TestClient(app) as client:
            page = client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("text/html", page.headers["content-type"])
        self.assertIn(f"{config.server.path}?model=pine-browser&{SHOW_FILLER_PARAM}=1", page.text)

    def test_bearer_token_is_enforced_when_configured(self) -> None:
        config = base_config()
        config = replace(
            config, server=replace(config.server, warmup=False, require_bearer=True, bearer_token="s3cret")
        )
        app = build_app(config, options=ServerOptions(stub_speech=True, stub_agent="scripted"), services=_services())
        with TestClient(app) as client:
            with (
                self.assertRaises(WebSocketDisconnect),
                client.websocket_connect("/v1/realtime", headers={"Authorization": "Bearer wrong"}) as ws,
            ):
                ws.receive_text()
            with client.websocket_connect("/v1/realtime", headers={"Authorization": "Bearer s3cret"}) as ws:
                self.assertEqual(json.loads(ws.receive_text())["type"], "session.created")
