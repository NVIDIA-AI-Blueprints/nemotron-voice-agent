# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The adapter's Pipecat pipeline, and ``bot()``: the session-owned Realtime route and the browser UI.

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

That is the OpenAI Realtime pipeline (always the ``eval`` profile). Browser UI sessions run the
same processors from VAD to the output tagger on the repository's UI transport, with RTVI
(``ui_pipeline.py``, ``ui_session.py``); with ``smart_turn`` set in the settings file the user
turn ends on the shared Smart Turn stop strategy instead of the speech-timeout stop.
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

from examples.frontend_backend_verdict.adapters.pipecat import switches, turn_modes
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
    from examples.frontend_backend_verdict.adapters.pipecat.settings import PipecatSettings
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
    """Builds and owns one OpenAI Realtime session's processors, transport and worker.

    The processors between the transport's input and output are shared with the browser UI
    pipeline (:class:`UiSessionPipeline`): :meth:`_build_turn_processors` and
    :meth:`_middle_processors`. The Realtime transport, serializer and worker are built here
    exactly as before the browser UI was added (``test_fbv_realtime_baseline``).
    """

    def __init__(self, session: PipecatSession, websocket: WebSocket, speech: SpeechStack) -> None:
        """Assemble the pipeline (it starts in :meth:`PipecatSession.run`)."""
        settings = session.settings
        self._build_turn_processors(session, speech)
        audio = settings.audio
        out_rate = audio.out_sample_rate or session.config.tts.sample_rate
        self.serializer: RealtimeSerializer | None = RealtimeSerializer(session)
        self.transport: Any = FastAPIWebsocketTransport(
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
        self.pipeline = Pipeline(
            [self.transport.input(), *self._middle_processors(speech, settings), self.transport.output()]
        )
        # Tracing is opt-in for Realtime sessions (switches.py); off, the worker is built as before.
        tracing: dict[str, Any] = {"enable_tracing": True} if switches.realtime_tracing() else {}
        self.worker = PipelineWorker(
            self.pipeline,
            params=PipelineParams(
                audio_in_sample_rate=audio.in_sample_rate,
                audio_out_sample_rate=out_rate,
            ),
            enable_rtvi=False,
            idle_timeout_secs=None,
            cancel_on_idle_timeout=False,
            **tracing,
        )
        self._cancel_on_disconnect()

    def _build_turn_processors(self, session: PipecatSession, speech: SpeechStack) -> None:
        """The input cursor, user-turn processor, turn bridge, ASR error tap and output tagger."""
        settings = session.settings
        self.turn_modes: turn_modes.ModeSwitchStopStrategy | None = None
        self.cursor = InputCursor()
        self.tagger = OutputTagger()
        turn = settings.turn
        if turn.smart_turn:
            from examples.shared.pipeline_utils import build_smart_turn_stop_strategies  # noqa: PLC0415

            wait = True if turn.wait_for_transcript is None else turn.wait_for_transcript
            stops: list[Any] = list(build_smart_turn_stop_strategies(wait_for_transcript=wait))
        else:
            stop_kwargs: dict[str, Any] = {}
            if turn.user_speech_timeout is not None:
                stop_kwargs["user_speech_timeout"] = turn.user_speech_timeout
            if turn.wait_for_transcript is not None:
                stop_kwargs["wait_for_transcript"] = turn.wait_for_transcript
            stops = [SpeechTimeoutUserTurnStopStrategy(**stop_kwargs)]
            modes = session.allowed_turn_modes()
            if modes != {turn_modes.SERVER_VAD}:
                # Opt-in Realtime turn modes (turn_modes.py); off, the strategies are built as before.
                by_mode: dict[str, list[Any]] = {turn_modes.SERVER_VAD: stops}
                if turn_modes.MANUAL in modes:
                    by_mode[turn_modes.MANUAL] = []
                if turn_modes.SEMANTIC_VAD in modes:
                    from examples.shared.pipeline_utils import build_smart_turn_stop_strategies  # noqa: PLC0415

                    wait = bool(turn.wait_for_transcript)
                    by_mode[turn_modes.SEMANTIC_VAD] = list(build_smart_turn_stop_strategies(wait_for_transcript=wait))
                self.turn_modes = turn_modes.ModeSwitchStopStrategy(by_mode)
                stops = [self.turn_modes]
        start = [GatedVADStart(session.interruptions_allowed), GatedTranscriptionStart(session.interruptions_allowed)]
        turn_kwargs: dict[str, Any] = {"user_turn_strategies": UserTurnStrategies(start=start, stop=stops)}
        if turn.user_turn_stop_timeout is not None:
            turn_kwargs["user_turn_stop_timeout"] = turn.user_turn_stop_timeout
        self.user_turns = UserTurnProcessor(**turn_kwargs)
        if self.turn_modes is not None:
            self.turn_modes.bind_controller(self.user_turns._user_turn_controller)
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
        self.tts = speech.tts
        self.vad = speech.vad
        self._vad_overrides = settings.vad.overrides()
        self._vad_params: Any = None
        self._vad_pending = False

    def _middle_processors(self, speech: SpeechStack, settings: PipecatSettings) -> list[FrameProcessor]:
        """VAD, input cursor, ASR, user turns, the turn bridge, TTS and the output tagger, in order."""
        return [
            VADProcessor(vad_analyzer=speech.vad, **settings.vad.processor_kwargs()),
            self.cursor,
            self.asr_errors,
            speech.stt,
            self.user_turns,
            self.bridge,
            speech.tts,
            self.tagger,
        ]

    def _cancel_on_disconnect(self) -> None:
        """End the session when the client goes away."""

        @self.transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(_transport: Any, _client: Any) -> None:
            await self.worker.cancel()

    def apply_turn_detection(self, turn: TurnDetectionSettings, *, stop_secs: float | None = None) -> None:
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

        stop = turn.silence_duration_ms / 1000.0 if stop_secs is None else stop_secs
        params = VADParams(**{**self._vad_overrides, "stop_secs": stop})
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
    """Serve one session: OpenAI Realtime (always the ``eval`` profile) or the browser UI (``default``).

    The standard example entry point (``examples_registry.yaml``): ``src/server.py`` marks the
    Realtime route's sessions with ``protocol: "realtime"`` and browser sessions (WebRTC, the UI
    WebSocket) with ``protocol: "rtvi"``. With ``FBV_PROFILE=eval`` browser sessions are refused.
    """
    body = getattr(runner_args, "body", None) or {}
    if body.get("protocol") == "realtime":
        await serve_realtime(runner_args)
        return
    from examples.frontend_backend_verdict.core.config.profiles import REALTIME_PROFILE, profile_name  # noqa: PLC0415

    if profile_name() == REALTIME_PROFILE:
        raise RuntimeError(
            "frontend-backend-verdict-agent runs with FBV_PROFILE=eval, which serves the OpenAI Realtime "
            "WebSocket only (model nvidia/nemotron-realtime-frontend-backend-verdict on /v1/realtime); "
            "deploy with FBV_PROFILE=default (or unset) for the browser UI"
        )
    from examples.frontend_backend_verdict.adapters.pipecat.ui_session import serve_ui  # noqa: PLC0415

    await serve_ui(runner_args)


async def serve_realtime(runner_args: RunnerArguments) -> None:
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
