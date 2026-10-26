# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""An in-process harness for the Pipecat adapter, with v1's ``SessionHarness`` interface.

Plan ``misc/frontend-backend-verdict/shared-stack-plan.md`` §8.4, §10.1 (G2). The session runs on
a fake WebSocket (no network) with Pipecat stub speech ported from FDH's ``services.py``
(``a0ef50c``): an energy VAD analyzer, an STT that finalizes "utterance N" at each VAD stop, and a
tone TTS through Pipecat's own TTS machinery at v1's 10 ms per character. The same golden scenarios
(``_fba_golden.py``) run on this harness and on v1's.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import struct
import unittest
from collections.abc import AsyncGenerator, Callable
from dataclasses import replace
from typing import Any

from _fba_core_fakes import append_event, base_config, pcmu_silence, pcmu_speech
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams, VADState
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.tts_service import TTSService
from pipecat.utils.time import time_now_iso8601
from starlette.websockets import WebSocketState

from examples.frontend_backend_agent.adapters.pipecat.pipeline import SpeechStack
from examples.frontend_backend_agent.adapters.pipecat.runtime import PipecatRuntime
from examples.frontend_backend_agent.adapters.pipecat.session import PipecatSession
from examples.frontend_backend_agent.adapters.pipecat.settings import load_pipecat_settings
from examples.frontend_backend_agent.core.agent.factory import ServerOptions
from examples.frontend_backend_agent.core.config.voice import VoiceConfig
from examples.frontend_backend_agent.core.engine.event_log import EventLog, SessionRoutingSink
from examples.frontend_backend_agent.core.engine.filler import FillerLog
from examples.frontend_backend_agent.core.engine.port import AgentPort
from examples.frontend_backend_agent.core.engine.runner import AgentClients, TextAgentRunner

# -- stub speech (FDH services.py stand-ins) -------------------------------------------------


class EnergyVADAnalyzer(VADAnalyzer):
    """Speech confidence from frame energy (a tone is speech, silence is not)."""

    def __init__(self, *, sample_rate: int | None = None, params: VADParams | None = None) -> None:
        super().__init__(sample_rate=sample_rate, params=params or VADParams())

    def num_frames_required(self) -> int:
        return int(self.sample_rate * 0.02) if self.sample_rate else 320

    async def analyze_audio(self, buffer: bytes) -> VADState:
        # Inline, not on Pipecat's thread executor: the harness advances by event-loop turns.
        return self._run_analyzer(buffer)

    def voice_confidence(self, buffer: bytes) -> float:
        count = len(buffer) // 2
        if not count:
            return 0.0
        samples = struct.unpack(f"<{count}h", buffer[: count * 2])
        rms = math.sqrt(sum(sample * sample for sample in samples) / count)
        return 1.0 if rms > 500 else 0.0


class StubRecognizer:
    """What the stub STT says, with v1's ``StubRecognizer`` interface (plus scripted failures).

    The N-th utterance (1-based) is ``transcripts[N-1]`` while scripted transcripts last, then
    ``"<prefix> N"``. An utterance whose number is in ``fail`` raises an ASR error instead.
    """

    def __init__(
        self, transcripts: list[str] | None = None, *, prefix: str = "utterance", fail: set[int] | None = None
    ) -> None:
        self._scripted = list(transcripts or [])
        self._prefix = prefix
        self.fail = set(fail or ())
        self.count = 0

    def next(self) -> str | None:
        """The next utterance's transcript, or ``None`` when it fails."""
        self.count += 1
        text = self._scripted.pop(0) if self._scripted else f"{self._prefix} {self.count}"
        return None if self.count in self.fail else text


class StubSTT(FrameProcessor):
    """Emits one finalized transcript per VAD stop from a :class:`StubRecognizer` (v1's stub recognizer).

    Like Pipecat's STT services it announces its ``ttfs_p99_latency`` after the ``StartFrame``,
    so the user-turn stop strategy waits for the final transcript as it does in production.
    Like v1's stub stream it reports the whole transcript as one interim just before the final.
    """

    def __init__(self, recognizer: StubRecognizer | None = None, *, ttfs_p99: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.recognizer = recognizer or StubRecognizer()
        self._ttfs_p99 = ttfs_p99

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(STTMetadataFrame(service_name=self.name, ttfs_p99_latency=self._ttfs_p99))
        if isinstance(frame, VADUserStoppedSpeakingFrame) and direction == FrameDirection.DOWNSTREAM:
            text = self.recognizer.next()
            if text is None:
                await self.push_error("ASR endpoint unavailable")
                return
            now = time_now_iso8601()
            if text:
                await self.push_frame(InterimTranscriptionFrame(text=text, user_id="", timestamp=now))
            await self.push_frame(TranscriptionFrame(text=text, user_id="", timestamp=now, finalized=True))


class ToneSynthesizer:
    """What the tone TTS does, with v1's ``ToneSynthesizer`` options plus the test variants.

    * ``gate``: every sentence waits until it is set (v1's ``gate``);
    * ``block_on``: a sentence containing it blocks until cancelled (v1 tests' ``BlockingSynthesizer``);
    * ``hold_at``: the N-th sentence (1-based) waits for :attr:`release` (the golden ``HoldingSynthesizer``);
    * ``fail``: every sentence fails with a TTS error (the golden ``FailingSynthesizer``).

    ``requests`` (also ``started``) and ``cancelled`` record sentences, as v1's synthesizers did.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        ms_per_char: float = 10.0,
        gate: asyncio.Event | None = None,
        block_on: str | None = None,
        hold_at: int | None = None,
        fail: str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.ms_per_char = ms_per_char
        self.gate = gate
        self.block_on = block_on
        self.hold_at = hold_at
        self.fail = fail
        self.release = asyncio.Event()
        self.requests: list[str] = []
        self.started: list[str] = []
        self.cancelled: list[str] = []


class ToneTTS(TTSService):
    """A 440 Hz tone, ``ms_per_char`` long per character, through Pipecat's own TTS machinery."""

    def __init__(self, synthesizer: ToneSynthesizer | None = None, **kwargs: Any) -> None:
        self.synthesizer = synthesizer or ToneSynthesizer()
        super().__init__(sample_rate=self.synthesizer.sample_rate, push_stop_frames=True, **kwargs)

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        spec = self.synthesizer
        spec.requests.append(text)
        spec.started.append(text)
        if spec.fail:
            await self.push_error(spec.fail)
            return
        try:
            if spec.gate is not None:
                await spec.gate.wait()
            if spec.block_on and spec.block_on in text:
                await asyncio.Event().wait()
            if spec.hold_at is not None and len(spec.requests) == spec.hold_at:
                await spec.release.wait()
        except asyncio.CancelledError:
            spec.cancelled.append(text)
            raise
        rate = self.sample_rate
        total = int(rate * spec.ms_per_char * max(1, len(text)) / 1000)
        chunk = int(rate * 0.02)
        for start in range(0, total, chunk):
            count = min(chunk, total - start)
            pcm = struct.pack(
                f"<{count}h",
                *(int(8000 * math.sin(2 * math.pi * 440 * (start + i) / rate)) for i in range(count)),
            )
            yield TTSAudioRawFrame(audio=pcm, sample_rate=rate, num_channels=1, context_id=context_id)


# -- fake WebSocket ----------------------------------------------------------------------------


class FakeWebSocket:
    """Enough of Starlette's ``WebSocket`` for Pipecat's FastAPI transport."""

    def __init__(self) -> None:
        self.inbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.events: list[dict[str, Any]] = []
        self.client_state = WebSocketState.CONNECTED
        self.application_state = WebSocketState.CONNECTED
        self.headers: dict[str, str] = {}
        self.query_params: dict[str, str] = {}
        self.close_code: int | None = None

    async def receive(self) -> dict[str, Any]:
        return await self.inbox.get()

    async def send_text(self, text: str) -> None:
        self.events.append(json.loads(text))

    async def send_bytes(self, data: bytes) -> None:  # pragma: no cover - Realtime is text only
        raise AssertionError("unexpected binary frame")

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.close_code = code
        self.client_state = WebSocketState.DISCONNECTED
        self.application_state = WebSocketState.DISCONNECTED

    def put_text(self, text: str) -> None:
        self.inbox.put_nowait({"type": "websocket.receive", "text": text})

    def disconnect(self) -> None:
        self.inbox.put_nowait({"type": "websocket.disconnect", "code": 1000})


# -- harness -----------------------------------------------------------------------------------


#: Event-loop turns per input frame: enough for a frame to cross the pipeline's processors.
FRAME_ROUNDS = 30


#: Harnesses not closed yet (``PipecatTestCase`` closes them after each test).
_OPEN: list[PipecatHarness] = []


async def _no_sleep() -> None:
    return None


class PipecatHarness:
    """A Pipecat-adapter session over a fake WebSocket with stub speech (v1 ``SessionHarness`` interface)."""

    def __init__(
        self,
        *,
        config: VoiceConfig | None = None,
        agent_factory: Callable[[str], AgentPort] | None = None,
        clients: AgentClients | None = None,
        synthesizer: ToneSynthesizer | None = None,
        recognizer: StubRecognizer | None = None,
        clock: Any = None,
        filler_log: FillerLog | None = None,
        show_silent_filler: bool = False,
        event_log: EventLog | None = None,
        pace_output: bool = False,
    ) -> None:
        self.config = config or base_config()
        self.runners: list[TextAgentRunner] = []
        options = ServerOptions(stub_speech=True, stub_agent="" if clients is not None else "scripted")
        settings = load_pipecat_settings()
        # The energy VAD is already a stand-in: Pipecat's smoothed-volume gate (min_volume) needs
        # about 400 ms of audio to ramp up, longer than part of a 600 ms stub utterance.
        settings = replace(settings, vad=replace(settings.vad, min_volume=0.0))
        # The output transport runs at the tone TTS's rate (production: the profile's TTS rate),
        # so Pipecat never resamples: its libsoxr int16 output dithers differently run to run.
        self.synthesizer = synthesizer or ToneSynthesizer()
        settings = replace(settings, audio=replace(settings.audio, out_sample_rate=self.synthesizer.sample_rate))
        self.runtime = PipecatRuntime(
            self.config,
            profile="test",
            settings=settings,
            options=options,
            clients=clients,
            speech_factory=self._speech,
        )
        self.runtime.state.routing_sink = SessionRoutingSink(event_log or EventLog())
        if filler_log is not None:
            self.runtime.state.filler_log = filler_log
        self.filler_log = self.runtime.state.filler_log
        if agent_factory is not None:
            self.runtime.agent_factory = agent_factory
        elif clients is not None:
            base = self.runtime.agent_factory

            def recording(session_id: str) -> AgentPort:
                runner = base(session_id)
                self.runners.append(runner)
                return runner

            self.runtime.agent_factory = recording
        self.recognizer = recognizer or StubRecognizer()
        self.tts = ToneTTS(self.synthesizer)
        self.stt = StubSTT(self.recognizer, ttfs_p99=settings.turn.stt_ttfs_p99 or 1.0)
        self.websocket = FakeWebSocket()
        self.session = PipecatSession(
            runtime=self.runtime,
            websocket=self.websocket,  # type: ignore[arg-type]
            model="pine-test",
            show_silent_filler=show_silent_filler,
            speech=self._speech,
            clock=clock,
        )
        if not pace_output:
            # G2 runs without wall-clock sleeps: audio is written as soon as it is synthesized, as
            # v1's golden capture did (``pace_output: false``). Pipecat's transport otherwise
            # paces its writes at twice real time.
            self.session.pipeline.transport.output()._write_audio_sleep = _no_sleep  # type: ignore[method-assign]
        self._task: asyncio.Task[None] | None = None
        _OPEN.append(self)

    def _speech(self, _session: Any) -> SpeechStack:
        return SpeechStack(vad=EnergyVADAnalyzer(), stt=self.stt, tts=self.tts)

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.websocket.events

    def types(self) -> list[str]:
        return [event["type"] for event in self.events]

    def of_type(self, kind: str) -> list[dict[str, Any]]:
        return [event for event in self.events if event["type"] == kind]

    async def start(self, session_update: dict[str, Any] | None = None) -> None:
        self._task = asyncio.create_task(self.session.run())
        await self.wait_for("session.created")
        if session_update is not None:
            await self.send(session_update)
            await self.wait_for("session.updated")

    async def send(self, event: dict[str, Any] | str) -> None:
        self.websocket.put_text(event if isinstance(event, str) else json.dumps(event))
        await self.settle()

    async def feed(self, audio: bytes, *, chunk: int = 160) -> None:
        """Append ``audio`` as the benchmark does: 160-byte mu-law frames, each run through before the next (G2)."""
        for offset in range(0, len(audio), chunk):
            piece = audio[offset : offset + chunk]
            self.websocket.put_text(append_event(piece + pcmu_silence(20)[len(piece) :]))
            await self._drain_frame()
        await self.settle()

    async def _drain_frame(self, rounds: int = FRAME_ROUNDS) -> None:
        """Let the transport take the frame, then give the pipeline's tasks ``rounds`` loop turns."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 2.0
        while not self.websocket.inbox.empty():
            if loop.time() > deadline:
                raise AssertionError("the transport stopped reading input (session ended?)")
            await asyncio.sleep(0.001)
        for _ in range(rounds):
            await asyncio.sleep(0)

    async def speak(self, ms: int = 600, *, then_silence_ms: int = 700) -> None:
        await self.feed(pcmu_speech(ms) + pcmu_silence(then_silence_ms))

    async def settle(self, seconds: float = 0.05) -> None:
        """Let the pipeline drain what was queued (the transport reads the inbox on its own task)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 2.0
        while not self.websocket.inbox.empty() and loop.time() < deadline:
            await asyncio.sleep(0.005)
        await asyncio.sleep(seconds)

    async def wait_for(self, kind: str, count: int = 1, timeout: float = 10.0) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            matches = self.of_type(kind)
            if len(matches) >= count:
                return matches[count - 1]
            if loop.time() > deadline:
                raise AssertionError(f"timed out waiting for {count} x {kind}; saw {self.types()}")
            await asyncio.sleep(0.005)

    async def idle(self, ms: int = 2000) -> None:
        await self.feed(pcmu_silence(ms))

    async def close(self) -> None:
        if self in _OPEN:
            _OPEN.remove(self)
        self.synthesizer.release.set()
        if self.synthesizer.gate is not None:
            self.synthesizer.gate.set()
        self.websocket.disconnect()
        if self._task is not None:
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(self._task, timeout=10.0)


class PipecatTestCase(unittest.IsolatedAsyncioTestCase):
    """Closes every harness a test left open (a failed assertion skips ``close``).

    Without it, the event loop's teardown cancels all tasks at once, and Pipecat's pipeline
    worker then waits forever for processors whose tasks were cancelled in the same sweep.
    """

    async def asyncTearDown(self) -> None:
        for harness in list(_OPEN):
            with contextlib.suppress(Exception):
                await harness.close()
        await super().asyncTearDown()
