# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Voice-side fakes for the delegation prototype: a scriptable backend link and a session harness."""

from __future__ import annotations

import asyncio
import base64
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from examples.frontend_delegation_hermes.core.backend.protocol import (
    GATEWAY_TO_VOICE,
    VOICE_TO_GATEWAY,
    message,
    validate,
)
from examples.frontend_delegation_hermes.core.config import DelegationConfig, load_delegation_config
from examples.frontend_delegation_hermes.core.engine.filler import FillerLog
from examples.frontend_delegation_hermes.core.engine.sinks import EventLog, SessionRoutingSink
from examples.frontend_delegation_hermes.core.engine.turn_manager import DelegationTurnManager
from examples.frontend_delegation_hermes.core.frontend.decider import Decider, RuleDecider
from examples.frontend_delegation_hermes.core.frontend.llm import ChatReply
from examples.frontend_delegation_hermes.core.head_api import TurnContext
from examples.frontend_delegation_hermes.core.runtime import DelegationRuntime

ROOT = Path(__file__).resolve().parents[4]
_VOICE_TESTS = ROOT / "tests" / "unit" / "frontend_delegation_hermes" / "voice"
if str(_VOICE_TESTS) not in sys.path:
    sys.path.insert(0, str(_VOICE_TESTS))

from _voice_fakes import MemoryTransport, pcmu_silence, pcmu_speech  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fdh_test_head.ports import IdentityNormalizer, SpeechServices  # noqa: E402
from _fdh_test_head.session import RealtimeSession  # noqa: E402
from _fdh_test_head.stubs import StubRecognizer, ToneSynthesizer  # noqa: E402
from _fdh_test_head.vad_energy import EnergyVad  # noqa: E402
from _fdh_variants import variant  # noqa: E402

PACKAGE = ROOT / "src" / "examples" / "frontend_delegation_hermes"
PROFILES = PACKAGE / "config" / "profiles"
FIXTURES = ROOT / "tests" / "unit" / "frontend_delegation_hermes" / "voice" / "fixtures" / "tau2_session_update"


def tau3_config(**backend: Any) -> DelegationConfig:
    """The tau3 control profile (every tau3-failure-fixes-plan.md switch off), scripted, no warm-up.

    Tests of one fix load its arm with :func:`profile_config`.
    """
    config = load_delegation_config(variant("profiles/tau3_eval_baseline.yaml"))
    voice = replace(config.voice, server=replace(config.voice.server, warmup=False))
    config = replace(config, voice=voice, frontend=replace(config.frontend, decider="scripted"))
    if backend:
        config = replace(config, backend=replace(config.backend, **backend))
    return config


def profile_config(name: str | Path) -> DelegationConfig:
    """A shipped profile (or a variant path) with the scripted frontend and no warm-up (like :func:`tau3_config`)."""
    config = load_delegation_config(PROFILES / name)
    voice = replace(config.voice, server=replace(config.voice.server, warmup=False))
    return replace(config, voice=voice, frontend=replace(config.frontend, decider="scripted"))


def browser_config() -> DelegationConfig:
    config = load_delegation_config(PROFILES / "browser_demo.yaml")
    voice = replace(config.voice, server=replace(config.voice.server, warmup=False))
    return replace(config, voice=voice, frontend=replace(config.frontend, decider="scripted"))


def tau2_session_update(domain: str = "mock") -> dict[str, Any]:
    return json.loads((FIXTURES / f"{domain}.json").read_text())


class FakeLink:
    """Records voice->gateway messages; answers open/configure; tests inject gateway messages."""

    def __init__(self, *, configure_error: str = "", fail_open: bool = False) -> None:
        self.sent: list[dict[str, Any]] = []
        self.on_message: Callable[[dict[str, Any]], None] | None = None
        self.configure_error = configure_error
        self.fail_open = fail_open
        self.closed = False
        self.configures = 0

    async def open(self, on_message: Callable[[dict[str, Any]], None]) -> None:
        if self.fail_open:
            raise ConnectionRefusedError("gateway down")
        self.on_message = on_message

    def send(self, data: dict[str, Any]) -> None:
        validate(VOICE_TO_GATEWAY, data)
        self.sent.append(data)
        kind = data["type"]
        loop = asyncio.get_running_loop()
        if kind == "session.open":
            loop.call_soon(self.deliver, "session.ready", {"worker_pid": 1234})
        elif kind == "session.configure":
            self.configures += 1
            if self.configure_error and self.configures > 1:
                loop.call_soon(
                    self.deliver, "error", {"code": self.configure_error, "message": "tools locked", "fatal": False}
                )
            else:
                names = [tool.get("name") for tool in data["tools"]]
                loop.call_soon(self.deliver, "session.configured", {"applied": True, "tools": names})

    def deliver(self, type_: str, fields: dict[str, Any] | None = None, **more: Any) -> None:
        assert self.on_message is not None
        self.on_message(message(GATEWAY_TO_VOICE, type_, **(fields or {}), **more))

    async def close(self) -> None:
        self.closed = True

    def of_type(self, kind: str) -> list[dict[str, Any]]:
        return [item for item in self.sent if item["type"] == kind]


class FakeChatModel:
    """Scripted ``ChatModel``: replies in order; ``delay`` seconds before each."""

    def __init__(self, replies: Sequence[ChatReply] = (), *, delay: float = 0.0) -> None:
        self.replies = list(replies)
        self.delay = delay
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self, messages: list[dict[str, Any]], *, tools=None, tool_choice: Any = "auto", parallel_tool_calls=None
    ) -> ChatReply:
        self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        if self.delay:
            await asyncio.sleep(self.delay)
        if not self.replies:
            raise RuntimeError("no scripted reply left")
        return self.replies.pop(0)


def delegate_reply(delegate: bool, filler: str = "", request: str | None = None) -> ChatReply:
    args: dict[str, Any] = {"delegate": delegate, "filler_text": filler}
    if request is not None:
        args["request"] = request
    return ChatReply(
        content=None, tool_calls=[("delegate", json.dumps(args))], usage={"input_tokens": 50, "output_tokens": 9}
    )


class DelegationHarness:
    """The test head's ``RealtimeSession`` with the delegation turn manager, stub speech and a :class:`FakeLink`."""

    def __init__(
        self,
        config: DelegationConfig | None = None,
        *,
        link: FakeLink | None = None,
        decider: Decider | None = None,
        transcripts: Sequence[str] = (),
        ms_per_char: float = 10.0,
    ) -> None:
        self.config = config or tau3_config()
        self.link = link or FakeLink()
        self.decider = decider or RuleDecider()
        self.recognizer = StubRecognizer(list(transcripts))
        self.events_log: list[tuple[str, dict[str, Any]]] = []
        runtime = DelegationRuntime(self.config, chat_model=FakeChatModel())
        self.runtime = runtime
        self.manager: DelegationTurnManager | None = None

        def factory(context: TurnContext) -> DelegationTurnManager:
            parts = runtime.parts(context)
            parts.link = self.link
            parts.decider = self.decider
            self.manager = DelegationTurnManager(context, self.config, parts)
            return self.manager

        services = SpeechServices(
            recognizer=self.recognizer,
            synthesizer=ToneSynthesizer(sample_rate=16000, ms_per_char=ms_per_char),
            vad_factory=EnergyVad,
            normalizer=IdentityNormalizer(),
        )
        self.transport = MemoryTransport()
        event_log = _RecordingLog(self.events_log)
        self.session = RealtimeSession(
            config=self.config.voice,
            transport=self.transport,
            services=services,
            routing_sink=SessionRoutingSink(event_log),
            filler_log=FillerLog(),
            model="pine-test",
            turn_manager_factory=factory,
        )
        self._inbox: asyncio.Queue[str | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.transport.events

    def of_type(self, kind: str) -> list[dict[str, Any]]:
        return [event for event in self.events if event["type"] == kind]

    def logged(self, kind: str) -> list[dict[str, Any]]:
        return [data for name, data in self.events_log if name == kind]

    async def start(self, session_update: dict[str, Any] | None = None) -> None:
        async def receive() -> str | None:
            return await self._inbox.get()

        self._task = asyncio.create_task(self.session.run(receive))
        await self.wait_for("session.created")
        if session_update is not None:
            await self.send(session_update)
            await self.wait_for("session.updated")

    async def send(self, event: dict[str, Any]) -> None:
        await self.session.handle_message(json.dumps(event))
        await self.settle()

    async def feed(self, audio: bytes, *, chunk: int = 160) -> None:
        for offset in range(0, len(audio), chunk):
            piece = audio[offset : offset + chunk]
            payload = base64.b64encode(piece + pcmu_silence(20)[len(piece) :]).decode()
            await self.session.handle_message(json.dumps({"type": "input_audio_buffer.append", "audio": payload}))
            await asyncio.sleep(0)
        await self.settle()

    async def speak(self, ms: int = 600, *, then_silence_ms: int = 900) -> None:
        await self.feed(pcmu_speech(ms) + pcmu_silence(then_silence_ms))

    async def idle(self, ms: int = 2000) -> None:
        await self.feed(pcmu_silence(ms))

    async def settle(self, rounds: int = 30) -> None:
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
                raise AssertionError(f"timed out waiting for {count} x {kind}; saw {[e['type'] for e in self.events]}")
            await asyncio.sleep(0.005)

    async def wait_sent(self, kind: str, count: int = 1, timeout: float = 5.0) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            matches = self.link.of_type(kind)
            if len(matches) >= count:
                return matches[count - 1]
            if loop.time() > deadline:
                raise AssertionError(f"timed out waiting for sent {kind}; sent {[m['type'] for m in self.link.sent]}")
            await asyncio.sleep(0.005)

    def transcripts(self) -> list[str]:
        """Assistant transcripts in order (from response.output_audio_transcript.done)."""
        return [event["transcript"] for event in self.of_type("response.output_audio_transcript.done")]

    async def close(self) -> None:
        await self._inbox.put(None)
        if self._task is not None:
            await asyncio.wait_for(self._task, timeout=5.0)


class _RecordingLog(EventLog):
    def __init__(self, sink: list[tuple[str, dict[str, Any]]]) -> None:
        super().__init__()
        self._sink = sink

    def write(self, kind: str, session_id: str, data: dict[str, Any]) -> None:
        self._sink.append((kind, dict(data)))
