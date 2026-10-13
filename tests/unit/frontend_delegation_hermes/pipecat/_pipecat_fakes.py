# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""A τ²-shaped client for the Pipecat adapter: real FastAPI WebSocket, stub speech, fake backend."""

from __future__ import annotations

import base64
import json
import queue
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from examples.frontend_delegation_hermes.adapters.pipecat.app import build_pipecat_app
from examples.frontend_delegation_hermes.adapters.pipecat.pipeline import SpeechStack
from examples.frontend_delegation_hermes.adapters.pipecat.services import EnergyVADAnalyzer, StubSTT, ToneTTS
from examples.frontend_delegation_hermes.adapters.pipecat.settings import load_pipecat_settings
from examples.frontend_delegation_hermes.core.audio.formats import AudioFormat
from examples.frontend_delegation_hermes.core.audio.pcm import silence, tone
from examples.frontend_delegation_hermes.core.config import load_delegation_config

REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE = REPO_ROOT / "src" / "examples" / "frontend_delegation_hermes"
FIXTURES = REPO_ROOT / "tests" / "unit" / "frontend_delegation_hermes" / "tau" / "fixtures" / "tau2_session_update"
PCMU = AudioFormat("audio/pcmu", 8000)


def tau2_session_update(domain: str = "mock") -> dict[str, Any]:
    return json.loads((FIXTURES / f"{domain}.json").read_text())


def stub_config(profile: str = "pipecat_realtime_eval.yaml", **server: Any) -> Any:
    """A Pipecat profile with the in-process fake gateway (calls the first tool) and the rule-based frontend."""
    config = load_delegation_config(PACKAGE / "config" / "profiles" / profile)
    voice = replace(
        config.voice,
        server=replace(config.voice.server, warmup=False, **server),
        logging=replace(config.voice.logging, event_log=""),
    )
    backend = replace(
        config.backend,
        link="in_process_fake",
        verbalizer_mode="template",
        gateway_config=PACKAGE / "config" / "gateway.stub_gate.yaml",
    )
    return replace(config, voice=voice, backend=backend, frontend=replace(config.frontend, decider="scripted"))


def stub_speech(_: Any = None) -> SpeechStack:
    return SpeechStack(vad=EnergyVADAnalyzer(), stt=StubSTT(), tts=ToneTTS(sample_rate=16000, ms_per_char=3.0))


def stub_app(profile: str = "pipecat_realtime_eval.yaml", settings: str = "adapters/pipecat.yaml", **server: Any):
    return build_pipecat_app(
        stub_config(profile, **server), load_pipecat_settings(settings), speech_factory=stub_speech
    )


class FailingTTS(ToneTTS):
    """Every synthesis fails (the service turns the exception into an ErrorFrame, no audio)."""

    async def run_tts(self, text: str, context_id: str):
        raise RuntimeError("synthesis failed")
        yield  # pragma: no cover - makes this an async generator


def failing_tts_app() -> Any:
    def speech(_: Any = None) -> SpeechStack:
        return SpeechStack(vad=EnergyVADAnalyzer(), stt=StubSTT(), tts=FailingTTS(sample_rate=16000))

    return build_pipecat_app(stub_config(), load_pipecat_settings("adapters/pipecat.yaml"), speech_factory=speech)


def pcmu_speech(ms: int) -> bytes:
    return PCMU.encode(tone(ms, 8000, frequency=300.0))


def pcmu_silence(ms: int) -> bytes:
    return PCMU.encode(silence(ms, 8000))


class Client:
    """Sends τ²-paced audio and collects every server event on a reader thread."""

    def __init__(self, ws: Any) -> None:
        self.ws = ws
        self.events: list[dict[str, Any]] = []
        self._inbox: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        try:
            while True:
                self._inbox.put(json.loads(self.ws.receive_text()))
        except Exception:  # noqa: BLE001 - the socket closed
            self._inbox.put(None)

    def send(self, event: dict[str, Any]) -> None:
        self.ws.send_text(json.dumps(event))

    def audio(self, payload: bytes, *, frame_ms: int = 20, realtime: float = 0.5) -> None:
        """160-byte μ-law frames; ``realtime`` scales the wall-clock pacing (Pipecat's timers are wall-clock)."""
        step = 8 * frame_ms
        for offset in range(0, len(payload), step):
            chunk = payload[offset : offset + step]
            self.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode()})
            time.sleep(frame_ms / 1000.0 * realtime)

    def _drain(self, timeout: float) -> None:
        try:
            event = self._inbox.get(timeout=timeout)
        except queue.Empty:
            return
        if event is not None:
            self.events.append(event)

    def wait_for(self, kind: str, count: int = 1, timeout: float = 15.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            matches = [e for e in self.events if e["type"] == kind]
            if len(matches) >= count:
                return matches[count - 1]
            self._drain(0.05)
        raise AssertionError(f"timed out waiting for {count} x {kind}; saw {[e['type'] for e in self.events]}")

    def settle(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._drain(0.05)

    def of_type(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == kind]

    def types(self) -> list[str]:
        return [e["type"] for e in self.events]
