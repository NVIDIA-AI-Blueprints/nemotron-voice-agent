# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The adapter's Pipecat pipeline, and ``bot()`` for the session-owned Realtime route.

Ported from FDH ``adapters/pipecat/pipeline.py`` (``a0ef50c``) plus v1's ``pipeline.py``
admission (plan §2, §3.1)::

    FastAPIWebsocketTransport.input()    client messages (router), input audio decoded as v1 decodes it
      -> VADProcessor(Silero)            shared Pipecat VAD, Tau settings (§7.2)
      -> InputCursor                     input-audio clock; onset and offset; playout cursor (S1)
      -> AsrErrorTap                     the STT's upstream errors -> the bridge (S5)
      -> NvidiaForceEouSTTService        shared STT
      -> UserTurnProcessor               Pipecat user turns (speech-timeout stop, no Smart Turn)
      -> FbvTurnBridge                   drives the turn manager
      -> NvidiaTTSService                shared Pipecat TTS, one request per sentence
      -> OutputTagger                    transcript/audio order, playout progress
      -> FastAPIWebsocketTransport.output()

Barge-in (B9): Pipecat's user-turn start strategies broadcast the interruption; whether a turn
interrupts is decided per turn by the same rule the turn manager applies
(``barge_in.enabled`` and the client's ``interrupt_response``).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from loguru import logger
from pipecat.frames.frames import Frame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy, VADUserTurnStartStrategy
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from examples.frontend_backend_verdict.adapters.pipecat.serializer import RealtimeSerializer
from examples.frontend_backend_verdict.adapters.pipecat.services import build_stt, build_tts, build_vad
from examples.frontend_backend_verdict.adapters.pipecat.speech_output import OutputTagger
from examples.frontend_backend_verdict.adapters.pipecat.timing import InputCursor
from examples.frontend_backend_verdict.adapters.pipecat.turn_bridge import AsrErrorTap, FbvTurnBridge
from examples.frontend_backend_verdict.core.config.profiles import ProfileError
from examples.frontend_backend_verdict.core.engine import server_events as ev
from examples.frontend_backend_verdict.core.engine.filler import SHOW_FILLER_PARAM

if TYPE_CHECKING:
    from fastapi import WebSocket
    from pipecat.audio.vad.vad_analyzer import VADAnalyzer

    from examples.frontend_backend_verdict.adapters.pipecat.session import PipecatSession
    from examples.frontend_backend_verdict.core.engine.session_settings import TurnDetectionSettings


class SpeechStack:
    """The three speech pieces a pipeline needs (shared Pipecat services by default; tests pass stubs)."""

    def __init__(self, *, vad: VADAnalyzer, stt: FrameProcessor, tts: FrameProcessor) -> None:
        """Bind the VAD analyzer and the STT/TTS processors."""
        self.vad = vad
        self.stt = stt
        self.tts = tts


def real_speech(session: PipecatSession) -> SpeechStack:
    """Silero + the shared NVIDIA STT/TTS services on the profile's endpoints."""
    return SpeechStack(
        vad=build_vad(session.settings),
        stt=build_stt(session.config, session.settings),
        tts=build_tts(session.config, session.settings),
    )


class _GatedStart:
    """Mixin: decide per turn whether Pipecat broadcasts the interruption (B9)."""

    _allowed: Callable[[], bool]
    _enable_interruptions: bool

    async def trigger_user_turn_started(
        self,
        *,
        enable_interruptions: bool | None = None,
        enable_user_speaking_frames: bool | None = None,
    ) -> None:
        if enable_interruptions is None:
            enable_interruptions = self._enable_interruptions and self._allowed()
        await super().trigger_user_turn_started(  # type: ignore[misc]
            enable_interruptions=enable_interruptions,
            enable_user_speaking_frames=enable_user_speaking_frames,
        )


class GatedVADStart(_GatedStart, VADUserTurnStartStrategy):
    """Pipecat's VAD start strategy with a per-turn interruption gate."""

    def __init__(self, allowed: Callable[[], bool], **kwargs: Any) -> None:
        """``allowed`` is read when a turn starts."""
        super().__init__(**kwargs)
        self._allowed = allowed


class GatedTranscriptionStart(_GatedStart, TranscriptionUserTurnStartStrategy):
    """Pipecat's transcription start strategy with a per-turn interruption gate."""

    def __init__(self, allowed: Callable[[], bool], **kwargs: Any) -> None:
        """``allowed`` is read when a turn starts."""
        super().__init__(**kwargs)
        self._allowed = allowed


class SessionPipeline:
    """Builds and owns one session's processors, transport and worker."""

    def __init__(self, session: PipecatSession, websocket: WebSocket, speech: SpeechStack) -> None:
        """Assemble the pipeline (it starts in :meth:`PipecatSession.run`)."""
        settings = session.settings
        self.cursor = InputCursor()
        self.tagger = OutputTagger()
        turn = settings.turn
        stop_kwargs: dict[str, Any] = {}
        if turn.user_speech_timeout is not None:
            stop_kwargs["user_speech_timeout"] = turn.user_speech_timeout
        if turn.wait_for_transcript is not None:
            stop_kwargs["wait_for_transcript"] = turn.wait_for_transcript
        stop = SpeechTimeoutUserTurnStopStrategy(**stop_kwargs)
        start = [GatedVADStart(session.interruptions_allowed), GatedTranscriptionStart(session.interruptions_allowed)]
        turn_kwargs: dict[str, Any] = {"user_turn_strategies": UserTurnStrategies(start=start, stop=[stop])}
        if turn.user_turn_stop_timeout is not None:
            turn_kwargs["user_turn_stop_timeout"] = turn.user_turn_stop_timeout
        self.user_turns = UserTurnProcessor(**turn_kwargs)
        self.bridge = FbvTurnBridge(
            config=session.config,
            emit_urgent=session.emit_urgent,
            conversation=session.conversation,
            cursor=self.cursor,
            tagger=self.tagger,
            log=session.log,
            clock=session.clock,
            session_id=session.session_id,
        )
        self.asr_errors = AsrErrorTap(self.bridge.on_asr_error)
        self.asr_errors.stt = speech.stt
        audio = settings.audio
        out_rate = audio.out_sample_rate or session.config.tts.sample_rate
        self.serializer = RealtimeSerializer(session)
        self.transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_in_sample_rate=audio.in_sample_rate,
                audio_out_enabled=True,
                audio_out_sample_rate=out_rate,
                audio_out_10ms_chunks=audio.out_10ms_chunks,
                audio_out_auto_silence=False,
                audio_out_end_silence_secs=0,
                audio_out_write_timeout_secs=audio.write_timeout_s,
                serializer=self.serializer,
                session_timeout=None,
            ),
        )
        self.tts = speech.tts
        self.vad = speech.vad
        self._vad_overrides = settings.vad.overrides()
        self._vad_params: Any = None
        self._vad_pending = False
        self.pipeline = Pipeline(
            [
                self.transport.input(),
                VADProcessor(vad_analyzer=speech.vad, **settings.vad.processor_kwargs()),
                self.cursor,
                self.asr_errors,
                speech.stt,
                self.user_turns,
                self.bridge,
                speech.tts,
                self.tagger,
                self.transport.output(),
            ]
        )
        self.worker = PipelineWorker(
            self.pipeline,
            params=PipelineParams(
                audio_in_sample_rate=audio.in_sample_rate,
                audio_out_sample_rate=out_rate,
            ),
            enable_rtvi=False,
            idle_timeout_secs=None,
            cancel_on_idle_timeout=False,
        )

        @self.transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(_transport: Any, _websocket: Any) -> None:
            await self.worker.cancel()

    def apply_turn_detection(self, turn: TurnDetectionSettings) -> None:
        """End turns on the session's effective silence, as v1's segmenter does (B1).

        ``stop_secs`` follows ``silence_duration_ms`` of the session's turn detection (the
        profile's value when ``honor_client_values`` is off, 800 ms for every Tau profile); the
        other ``VADParams`` come from the settings file. Pipecat's ``set_params`` resets the VAD
        state and needs the sample rate (known once the pipeline started), so params are applied
        only when they change, and params set before the start wait for :meth:`apply_pending_vad`.
        """
        if not turn.vad_enabled:
            return
        from pipecat.audio.vad.vad_analyzer import VADParams  # noqa: PLC0415

        params = VADParams(**{**self._vad_overrides, "stop_secs": turn.silence_duration_ms / 1000.0})
        if params == self._vad_params:
            return
        self._vad_params = params
        self._vad_pending = True
        self.apply_pending_vad()

    def apply_pending_vad(self) -> None:
        """Apply turn-detection params waiting for the VAD's sample rate (called before each audio frame)."""
        if self._vad_pending and self._vad_params is not None and self.vad.sample_rate:
            self.vad.set_params(self._vad_params)
            self._vad_pending = False

    async def push_to_tts(self, frame: Frame) -> None:
        """Push a frame into the TTS service's input (from the bridge, upstream of it)."""
        await self.bridge.push_frame(frame)

    async def interrupt_output(self) -> None:
        """Flush queued agent audio (the turn manager stopped its own output)."""
        await self.bridge.broadcast_interruption()


async def bot(runner_args: RunnerArguments) -> None:
    """Serve one OpenAI Realtime connection with the Pipecat adapter (v1's admission rules, S12)."""
    from examples.frontend_backend_verdict.adapters.pipecat.runtime import runtime_for  # noqa: PLC0415
    from examples.frontend_backend_verdict.adapters.pipecat.session import PipecatSession  # noqa: PLC0415

    body = getattr(runner_args, "body", None) or {}
    websocket = getattr(runner_args, "websocket", None)
    if body.get("protocol") != "realtime" or websocket is None:
        raise RuntimeError(
            "frontend-backend-verdict-agent serves the OpenAI Realtime WebSocket only "
            "(model nvidia/nemotron-realtime-frontend-backend-verdict on /v1/realtime)"
        )
    query = websocket.query_params
    model = query.get("model", "")
    show_silent_filler = query.get(SHOW_FILLER_PARAM, "").lower() in ("1", "true")
    try:
        runtime = runtime_for(body)
    except ProfileError as exc:
        logger.error(f"frontend-backend-verdict-agent: {exc}")
        await websocket.send_text(json.dumps(ev.error(str(exc), code="server_error", error_type="server_error")))
        await websocket.close(code=1011, reason="profile error")
        return
    if not await runtime.admit(websocket):
        return
    try:
        session = PipecatSession(
            runtime=runtime,
            websocket=websocket,
            model=model,
            show_silent_filler=show_silent_filler,
            speech=runtime.speech_factory,
        )
        await session.run()
    finally:
        runtime.release()
        logger.info(f"frontend/backend verdict session finished (model={model or '-'})")
