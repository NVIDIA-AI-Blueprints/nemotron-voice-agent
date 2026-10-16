# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Offline doubles and a session harness for the τ³-head tests.

Named ``_voice_fakes`` because pytest imports test helpers by basename.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
from collections.abc import Sequence
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from examples.frontend_delegation_hermes.adapters.tau.session import RealtimeSession
from examples.frontend_delegation_hermes.adapters.tau.speech.ports import IdentityNormalizer, SpeechServices
from examples.frontend_delegation_hermes.adapters.tau.speech.stubs import StubRecognizer, ToneSynthesizer
from examples.frontend_delegation_hermes.adapters.tau.speech.vad_energy import EnergyVad
from examples.frontend_delegation_hermes.core.audio.formats import AudioFormat
from examples.frontend_delegation_hermes.core.audio.pcm import silence, tone
from examples.frontend_delegation_hermes.core.engine.filler import FillerLog
from examples.frontend_delegation_hermes.core.engine.sinks import EventLog, SessionRoutingSink
from examples.frontend_delegation_hermes.core.head_api import TurnManagerFactory
from examples.frontend_delegation_hermes.core.voice_config import VoiceConfig, load_voice_config

REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_DIR = REPO_ROOT / "src" / "examples" / "frontend_delegation_hermes"
BASE_CONFIG = PACKAGE_DIR / "config" / "voice_agent.yaml"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
PCMU = AudioFormat("audio/pcmu", 8000)
PCM16K = AudioFormat("audio/pcm", 16000)

_CONFIG_CACHE: dict[str, VoiceConfig] = {}


def base_config() -> VoiceConfig:
    if "base" not in _CONFIG_CACHE:
        _CONFIG_CACHE["base"] = load_voice_config(BASE_CONFIG)
    return _CONFIG_CACHE["base"]


def voice_config(**sections: Any) -> VoiceConfig:
    """The shipped base config with whole sections or fields replaced.

    ``filler_mode="speak"`` style keys are expanded as ``<section>_<field>``.
    """
    config = base_config()
    names = sorted((f.name for f in fields(config)), key=len, reverse=True)
    for key, value in sections.items():
        if key == "agent":
            config = replace(config, agent=value)
            continue
        section = next(name for name in names if key.startswith(f"{name}_"))
        name = key[len(section) + 1 :]
        current = getattr(config, section)
        config = replace(config, **{section: replace(current, **{name: value})})
    return config


# -- clock / transport ----------------------------------------------------------------


class FakeClock:
    def __init__(self, start: float = 1_790_000_000.0) -> None:
        self.now = start
        self.mono = 1000.0

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.mono

    def advance(self, seconds: float) -> None:
        self.now += seconds
        self.mono += seconds


class MemoryTransport:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.changed = asyncio.Event()

    async def send_text(self, text: str) -> None:
        self.events.append(json.loads(text))
        self.changed.set()


# -- tau2-shaped payloads ---------------------------------------------------------------


def tau2_session_update(domain: str = "mock") -> dict[str, Any]:
    return json.loads((FIXTURES / "tau2_session_update" / f"{domain}.json").read_text(encoding="utf-8"))


def pcmu_speech(ms: int, frequency: float = 300.0) -> bytes:
    return PCMU.encode(tone(ms, 8000, frequency=frequency))


def pcmu_silence(ms: int) -> bytes:
    return PCMU.encode(silence(ms, 8000))


def append_event(chunk: bytes) -> str:
    return json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode("ascii")})


# -- session harness ---------------------------------------------------------------------


class SessionHarness:
    """A ``RealtimeSession`` over an in-memory transport with stub speech."""

    def __init__(
        self,
        *,
        turn_manager_factory: TurnManagerFactory,
        config: VoiceConfig | None = None,
        synthesizer: ToneSynthesizer | None = None,
        recognizer: StubRecognizer | None = None,
        clock: FakeClock | None = None,
        filler_log: FillerLog | None = None,
        show_silent_filler: bool = False,
        event_log: EventLog | None = None,
    ) -> None:
        self.config = config or base_config()
        self.transport = MemoryTransport()
        self.recognizer = recognizer or StubRecognizer()
        self.synthesizer = synthesizer or ToneSynthesizer(sample_rate=16000, ms_per_char=10.0)
        services = SpeechServices(
            recognizer=self.recognizer,
            synthesizer=self.synthesizer,
            vad_factory=EnergyVad,
            normalizer=IdentityNormalizer(),
        )
        self.routing_sink = SessionRoutingSink(event_log or EventLog())
        self.filler_log = filler_log or FillerLog()
        self.clock = clock
        self.session = RealtimeSession(
            config=self.config,
            transport=self.transport,
            services=services,
            turn_manager_factory=turn_manager_factory,
            routing_sink=self.routing_sink,
            filler_log=self.filler_log,
            clock=clock,
            model="pine-test",
            show_silent_filler=show_silent_filler,
        )
        self._inbox: asyncio.Queue[str | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.transport.events

    def types(self) -> list[str]:
        return [event["type"] for event in self.events]

    def of_type(self, kind: str) -> list[dict[str, Any]]:
        return [event for event in self.events if event["type"] == kind]

    async def start(self, session_update: dict[str, Any] | None = None) -> None:
        async def receive() -> str | None:
            return await self._inbox.get()

        self._task = asyncio.create_task(self.session.run(receive))
        await self.wait_for("session.created")
        if session_update is not None:
            await self.send(session_update)
            await self.wait_for("session.updated")

    async def send(self, event: dict[str, Any] | str) -> None:
        text = event if isinstance(event, str) else json.dumps(event)
        await self.session.handle_message(text)
        await self.settle()

    async def feed(self, audio: bytes, *, chunk: int = 160) -> None:
        """Append ``audio`` as tau2 does: 160-byte μ-law frames (no wall-clock pacing)."""
        for offset in range(0, len(audio), chunk):
            piece = audio[offset : offset + chunk]
            await self.session.handle_message(append_event(piece + pcmu_silence(20)[len(piece) :]))
            await asyncio.sleep(0)
        await self.settle()

    async def speak(self, ms: int = 600, *, then_silence_ms: int = 700) -> None:
        await self.feed(pcmu_speech(ms) + pcmu_silence(then_silence_ms))

    async def settle(self, rounds: int = 20) -> None:
        for _ in range(rounds):
            await asyncio.sleep(0)
        await self.session.writer.drain()

    async def wait_for(self, kind: str, count: int = 1, timeout: float = 5.0) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            await self.session.writer.drain()
            matches = self.of_type(kind)
            if len(matches) >= count:
                return matches[count - 1]
            if loop.time() > deadline:
                raise AssertionError(f"timed out waiting for {count} x {kind}; saw {self.types()}")
            await asyncio.sleep(0.005)

    async def idle(self, ms: int = 2000) -> None:
        """Stream silence (keeps the audio clock and playout cursor moving)."""
        await self.feed(pcmu_silence(ms))

    async def close(self) -> None:
        await self._inbox.put(None)
        if self._task is not None:
            await asyncio.wait_for(self._task, timeout=5.0)


def tau2_update_with(tools: list[dict[str, Any]] | None = None, instructions: str = "policy") -> dict[str, Any]:
    update = copy.deepcopy(tau2_session_update("mock"))
    if tools is not None:
        update["session"]["tools"] = tools
    update["session"]["instructions"] = instructions
    return update


ID_KINDS = (("item_", "ITEM"), ("resp_", "RESP"), ("sess_", "SESS"), ("call_", "CALL"), ("fcall_", "FCALL"))
WALL_KEYS = frozenset({"event_id", "expires_at"})


def normalize_trace(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rule 9 normalization: drop ``event_id``, ordinal placeholders for generated ids, keep audio-clock fields."""
    mapping: dict[str, str] = {}
    counters: dict[str, int] = {}

    def norm(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: norm(item) for key, item in value.items() if key not in WALL_KEYS}
        if isinstance(value, list):
            return [norm(item) for item in value]
        if isinstance(value, str):
            for prefix, label in ID_KINDS:
                if value.startswith(prefix):
                    if value not in mapping:
                        counters[label] = counters.get(label, 0) + 1
                        mapping[value] = f"<{label}{counters[label]}>"
                    return mapping[value]
        return value

    return [norm(event) for event in events]


# -- the delegation runtime with the no-GPU stub gateway ---------------------------------


def stub_gate_config(**server: Any) -> Any:
    """``profiles/stub_gate.yaml``: rule-based frontend, in-process fake gateway that calls the first tool."""
    from examples.frontend_delegation_hermes.core.config import load_delegation_config  # noqa: PLC0415

    config = load_delegation_config(PACKAGE_DIR / "config" / "profiles" / "stub_gate.yaml")
    voice = replace(
        config.voice,
        server=replace(config.voice.server, warmup=False, **server),
        logging=replace(config.voice.logging, event_log=""),  # tests never write the repository's logs/
    )
    return replace(config, voice=voice)


def stub_runtime(**server: Any) -> Any:
    """A :class:`DelegationRuntime` over :func:`stub_gate_config` (no LLM, no Hermes, no network)."""
    from examples.frontend_delegation_hermes.core.runtime import DelegationRuntime  # noqa: PLC0415

    return DelegationRuntime(stub_gate_config(**server))


def stub_services(ms_per_char: float = 5.0) -> SpeechServices:
    return SpeechServices(
        recognizer=StubRecognizer(),
        synthesizer=ToneSynthesizer(sample_rate=16000, ms_per_char=ms_per_char),
        vad_factory=EnergyVad,
        normalizer=IdentityNormalizer(),
    )


_ASYNC_HOOKS = (
    "start",
    "close",
    "on_speech_started",
    "on_response_cancel",
    "on_output_audio_clear",
    "on_session_update",
)


class NullTurns:
    """A ``TurnManagerLike`` that answers nothing (tests of the τ³ head's input side)."""

    state = "IDLE"

    def __init__(self, context: Any) -> None:
        self.context = context
        self.advanced_ms = 0.0

    def on_audio_advance(self, step_ms: float) -> None:
        self.advanced_ms += step_ms

    def __getattr__(self, name: str) -> Any:
        async def noop_async(*args: Any, **kwargs: Any) -> None:
            return None

        def noop(*args: Any, **kwargs: Any) -> Any:
            return False

        if name in _ASYNC_HOOKS:
            return noop_async
        return noop
