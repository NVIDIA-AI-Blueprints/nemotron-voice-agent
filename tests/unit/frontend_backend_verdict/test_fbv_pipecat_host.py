# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The registry's ``bot`` serves a real WebSocket with v1's decisions, and admits sessions as v1 did.

A tau2-shaped client script runs through FastAPI's test client against ``pipeline.bot`` (the
Pipecat adapter, with Pipecat stub speech and the scripted agent). The same script ran
against v1's hosts at ``fbv-v1-baseline``; that stream is ``fixtures/v1_golden/pipecat_host.json``.
The streams are compared as G2 compares them (plan §10.1): events per delivery class in order,
with ids, timestamps, audio and audio-clock fields set aside. ``bot`` admission follows v1
(S12): the Realtime protocol only, a profile error closes 1011, a full server closes 1013.
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

from _fbv_core_fakes import FIXTURES, base_config, pcmu_silence, pcmu_speech, tau2_session_update
from _fbv_parity import INTERIM_DELTA, TIMING_KEYS, event_class
from _fbv_pipecat_fakes import EnergyVADAnalyzer, StubSTT, ToneSynthesizer, ToneTTS
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from examples.frontend_backend_verdict import pipeline
from examples.frontend_backend_verdict.adapters.pipecat import runtime as pipecat_runtime
from examples.frontend_backend_verdict.adapters.pipecat.pipeline import SpeechStack
from examples.frontend_backend_verdict.adapters.pipecat.runtime import PipecatRuntime
from examples.frontend_backend_verdict.adapters.pipecat.settings import load_pipecat_settings
from examples.frontend_backend_verdict.core.agent.factory import ServerOptions
from examples.frontend_backend_verdict.core.config.profiles import ProfileError

_OPTIONS = ServerOptions(stub_speech=True, stub_agent="scripted")
#: The baseline streams (both v1 hosts were identical at fbv-v1-baseline), recorded before R0.
_GOLDEN = json.loads((FIXTURES / "v1_golden" / "pipecat_host.json").read_text(encoding="utf-8"))
_ID_KEYS = {"event_id", "item_id", "response_id", "previous_item_id", "id", "call_id"}
_ID_VALUE = re.compile(r"^(event|item|resp|sess|call|conv)_[A-Za-z0-9]+$")
_PLACEHOLDER = re.compile(r"^<id\d+>$")


def _config(**server: Any) -> Any:
    config = base_config()
    return replace(config, server=replace(config.server, warmup=False, **server))


def _runtime(config: Any) -> PipecatRuntime:
    settings = load_pipecat_settings()
    settings = replace(settings, vad=replace(settings.vad, min_volume=0.0))  # see PipecatHarness

    def speech(_session: Any) -> SpeechStack:
        return SpeechStack(
            vad=EnergyVADAnalyzer(),
            stt=StubSTT(ttfs_p99=settings.turn.stt_ttfs_p99 or 1.0),
            tts=ToneTTS(ToneSynthesizer(sample_rate=16000, ms_per_char=5.0)),
        )

    return PipecatRuntime(config, profile="test", settings=settings, options=_OPTIONS, speech_factory=speech)


def _app(runtime: PipecatRuntime) -> FastAPI:
    app = FastAPI()

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket) -> None:
        # What the Realtime gateway does for a session-owned pipeline: accept, then hand over.
        offered = [value.strip() for value in websocket.headers.get("sec-websocket-protocol", "").split(",")]
        await websocket.accept(subprotocol="realtime" if "realtime" in offered else None)
        with mock.patch.object(pipecat_runtime, "runtime_for", return_value=runtime):
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
        # v1 handled messages in arrival order, so the new turn's response was done before the
        # truncate (sent after all of the audio) was acknowledged. The adapter acknowledges it
        # as an urgent event, ahead of queued response events (B10): read on until every
        # response is done.
        _until(ws, "conversation.item.truncated", seen)
        while sum(e["type"] == "response.created" for e in seen) > sum(e["type"] == "response.done" for e in seen):
            _until(ws, "response.done", seen)
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


def _decisions(events: list[dict]) -> dict[str, list[Any]]:
    """G2's view: per delivery class, in order, without ids, audio and audio-clock fields."""

    def walk(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {k: walk(v, k) for k, v in sorted(value.items()) if k not in TIMING_KEYS}
        if isinstance(value, list):
            return [walk(v, key) for v in value]
        if isinstance(value, str) and _PLACEHOLDER.match(value):
            return "<id>"
        return value

    out: dict[str, list[Any]] = {"urgent": [], "stream": [], "response": []}
    for event in events:
        cls = event_class(event)
        if cls == "audio" or event.get("type") == INTERIM_DELTA:
            continue
        out[cls].append(walk(event))
    return out


class PipecatHostParityTests(unittest.TestCase):
    def test_conversation_makes_v1s_decisions(self) -> None:
        hosted = _normalize(_conversation(_app(_runtime(_config()))))
        self.assertEqual(_decisions(hosted), _decisions(_GOLDEN["conversation"]))
        types = [e["type"] for e in hosted]
        self.assertEqual(types[0], "session.created")
        self.assertEqual(types.count("response.created"), types.count("response.done"))
        self.assertNotIn("error", types)

    def test_barge_in_and_truncate_make_v1s_decisions(self) -> None:
        hosted = _normalize(_barge_in_conversation(_app(_runtime(_config()))))
        self.assertEqual(_decisions(hosted), _decisions(_GOLDEN["barge_in_conversation"]))
        types = [e["type"] for e in hosted]
        self.assertIn("conversation.item.truncated", types)
        self.assertEqual(types.count("response.created"), types.count("response.done"))
        self.assertNotIn("error", types)


class _Socket:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed: tuple[int, str] | None = None
        self.query_params: dict[str, str] = {}

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = (code, reason)


class BotAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_the_realtime_protocol_is_served(self) -> None:
        with self.assertRaises(RuntimeError):
            await pipeline.bot(SimpleNamespace(websocket=_Socket(), body={"protocol": "webrtc"}))
        with self.assertRaises(RuntimeError):
            await pipeline.bot(SimpleNamespace(websocket=None, body={"protocol": "realtime"}))

    async def test_profile_error_is_reported_and_closes_1011(self) -> None:
        socket = _Socket()
        with mock.patch.object(pipecat_runtime, "runtime_for", side_effect=ProfileError("bad profile")):
            await pipeline.bot(SimpleNamespace(websocket=socket, body={"protocol": "realtime"}))
        self.assertEqual(socket.sent[0]["type"], "error")
        self.assertEqual(socket.sent[0]["error"]["code"], "server_error")
        self.assertIn("bad profile", socket.sent[0]["error"]["message"])
        self.assertEqual(socket.closed, (1011, "profile error"))

    async def test_full_server_is_busy_and_closes_1013(self) -> None:
        runtime = _runtime(_config(max_sessions=0))
        socket = _Socket()
        with mock.patch.object(pipecat_runtime, "runtime_for", return_value=runtime):
            await pipeline.bot(SimpleNamespace(websocket=socket, body={"protocol": "realtime"}))
        self.assertEqual(socket.sent[0]["error"]["code"], "server_busy")
        self.assertEqual(socket.closed[0], 1013)
        self.assertEqual(runtime.state.sessions, 0)


if __name__ == "__main__":
    unittest.main()
