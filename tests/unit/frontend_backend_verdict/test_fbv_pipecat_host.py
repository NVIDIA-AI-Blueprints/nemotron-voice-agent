# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The Pipecat host serves the prototype session with the prototype's exact event stream.

The same tau2-shaped client script runs against two hosts of the same copied session:
the prototype's own FastAPI app (``voice/server.py`` ``build_app``) and this example's
Pipecat pipeline (``pipeline.bot``). Both use the prototype's stub speech (energy VAD,
"utterance N" recognizer, tone synthesizer) and its scripted agent. The event streams
must be identical after normalizing only generated ids, timestamps, and audio bytes.
"""

from __future__ import annotations

import base64
import json
import re
import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest import mock

from _fbv_voice_fakes import base_config, pcmu_silence, pcmu_speech, tau2_session_update
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from examples.frontend_backend_verdict import pipeline
from examples.frontend_backend_verdict.bridge.runtime import VerdictRuntime
from examples.frontend_backend_verdict.voice.server import ServerOptions, build_app
from examples.frontend_backend_verdict.voice.speech.ports import IdentityNormalizer, SpeechServices
from examples.frontend_backend_verdict.voice.speech.stubs import StubRecognizer, ToneSynthesizer
from examples.frontend_backend_verdict.voice.speech.vad_energy import EnergyVad

_OPTIONS = ServerOptions(stub_speech=True, stub_agent="scripted")
_ID_KEYS = {"event_id", "item_id", "response_id", "previous_item_id", "id", "call_id"}
_ID_VALUE = re.compile(r"^(event|item|resp|sess|call|conv)_[A-Za-z0-9]+$")


def _services() -> SpeechServices:
    return SpeechServices(
        recognizer=StubRecognizer(),
        synthesizer=ToneSynthesizer(sample_rate=16000, ms_per_char=5.0),
        vad_factory=EnergyVad,
        normalizer=IdentityNormalizer(),
    )


def _config() -> Any:
    config = base_config()
    return replace(config, server=replace(config.server, warmup=False))


def _pipecat_app(config: Any) -> FastAPI:
    runtime = VerdictRuntime(config, profile="test", options=_OPTIONS, services=_services())
    app = FastAPI()

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket) -> None:
        # What the Realtime gateway does for a session-owned pipeline: accept, then hand over.
        offered = [value.strip() for value in websocket.headers.get("sec-websocket-protocol", "").split(",")]
        await websocket.accept(subprotocol="realtime" if "realtime" in offered else None)
        with mock.patch.object(pipeline, "runtime_for", return_value=runtime):
            await pipeline.bot(SimpleNamespace(websocket=websocket, body={"protocol": "realtime"}))
        await websocket.close()

    return app


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


def _conversation(app: FastAPI) -> list[dict]:
    """Three turns with one tool round trip, then a tool call (the prototype's own scenario)."""
    seen: list[dict] = []
    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/realtime?model=pine-e2e", headers={"Authorization": "Bearer x"}) as ws,
    ):
        seen.append(json.loads(ws.receive_text()))
        ws.send_text(json.dumps(tau2_session_update("mock")))
        _until(ws, "session.updated", seen)
        turn = pcmu_speech(600) + pcmu_silence(800)
        _send_audio(ws, turn)
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
        _send_audio(ws, pcmu_silence(3000) + turn)
        _until(ws, "response.done", seen)
        _send_audio(ws, pcmu_silence(3000) + turn)
        _until(ws, "response.function_call_arguments.done", seen)
        _until(ws, "response.done", seen)
    return seen


def _barge_in_conversation(app: FastAPI) -> list[dict]:
    """Speech over the agent's answer, the client's truncate of the heard audio, then the next turn."""
    seen: list[dict] = []
    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/realtime?model=pine-e2e", headers={"Authorization": "Bearer x"}) as ws,
    ):
        seen.append(json.loads(ws.receive_text()))
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
        answer = _until(ws, "response.output_item.added", seen)
        _until(ws, "response.done", seen)
        # The caller talks over the answer before it has finished playing (input-audio clock).
        _send_audio(ws, pcmu_silence(200) + turn)
        _until(ws, "input_audio_buffer.speech_started", seen)
        ws.send_text(
            json.dumps(
                {
                    "type": "conversation.item.truncate",
                    "item_id": answer["item"]["id"],
                    "content_index": 0,
                    "audio_end_ms": 150,
                }
            )
        )
        # The session handles messages in arrival order: the new turn's response is done
        # before the truncate, sent after all of the audio, is acknowledged.
        _until(ws, "conversation.item.truncated", seen)
    return seen


def _normalize(events: list[dict]) -> list[Any]:
    """Replace generated ids by first-seen order, and drop audio bytes and wall-clock values."""
    names: dict[str, str] = {}

    def walk(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {k: walk(v, k) for k, v in sorted(value.items())}
        if isinstance(value, list):
            return [walk(v, key) for v in value]
        if isinstance(value, str) and (key in _ID_KEYS or _ID_VALUE.match(value)):
            return names.setdefault(value, f"<id{len(names)}>")
        if key in {"expires_at", "created_at"}:
            return "<time>"
        return value

    normalized = []
    for event in events:
        event = dict(event)
        if event.get("type") == "response.output_audio.delta":
            event["delta"] = "<audio>"
        normalized.append(walk(event))
    return normalized


class PipecatHostParityTests(unittest.TestCase):
    def test_pipecat_host_matches_the_prototype_server_event_for_event(self) -> None:
        config = _config()
        prototype = _conversation(build_app(config, options=_OPTIONS, services=_services()))
        hosted = _conversation(_pipecat_app(config))
        self.assertEqual([e["type"] for e in hosted], [e["type"] for e in prototype])
        self.assertEqual(_normalize(hosted), _normalize(prototype))
        types = [e["type"] for e in hosted]
        self.assertEqual(types[0], "session.created")
        self.assertEqual(types.count("response.created"), types.count("response.done"))
        self.assertNotIn("error", types)

    def test_barge_in_and_truncate_match_the_prototype_server(self) -> None:
        config = _config()
        prototype = _barge_in_conversation(build_app(config, options=_OPTIONS, services=_services()))
        hosted = _barge_in_conversation(_pipecat_app(config))
        self.assertEqual(_normalize(hosted), _normalize(prototype))
        types = [e["type"] for e in hosted]
        self.assertIn("conversation.item.truncated", types)
        self.assertEqual(types.count("response.created"), types.count("response.done"))
        self.assertNotIn("error", types)
