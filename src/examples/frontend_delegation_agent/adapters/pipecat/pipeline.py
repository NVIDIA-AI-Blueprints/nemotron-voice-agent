# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Pipecat adapter's per-connection pipeline.

::

    transport.input()            Realtime events in (RealtimeSerializer)
      -> VADProcessor            Silero, VADParams from the settings file (Pipecat defaults)
      -> InputCursor             audio now, VAD onset/offset on the input-audio clock
      -> STT                     NvidiaForceEouSTTService (catalog ASR)
      -> UserTurnProcessor       default start strategies; stop = speech timeout (no Smart Turn)
      -> FdaTurnBridge           user turns -> Realtime input events and the turn manager
      -> TTS                     NvidiaTTSService, per_sentence (catalog TTS)
      -> OutputTagger            transcript before audio, ResponseProgress, item markers
      -> transport.output()      Realtime events out, paced by Pipecat

RTVI is off, there is no idle timeout and no automatic output silence: τ² sends no pings,
pauses for seconds, and counts every agent audio byte as speech.

That is the OpenAI Realtime pipeline (always the ``eval`` profile). Browser UI sessions run the
same processors from the VAD to the output tagger on the repository's UI transport, with RTVI
(``ui_pipeline.py``, ``ui_session.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pipecat.frames.frames import Frame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from examples.frontend_delegation_agent.adapters.pipecat.serializer import RealtimeSerializer
from examples.frontend_delegation_agent.adapters.pipecat.services import build_stt, build_tts, build_vad
from examples.frontend_delegation_agent.adapters.pipecat.speech_output import OutputTagger
from examples.frontend_delegation_agent.adapters.pipecat.timing import InputCursor
from examples.frontend_delegation_agent.adapters.pipecat.turn_bridge import FdaTurnBridge

if TYPE_CHECKING:
    from fastapi import WebSocket
    from pipecat.audio.vad.vad_analyzer import VADAnalyzer

    from examples.frontend_delegation_agent.adapters.pipecat.session import PipecatSession
    from examples.frontend_delegation_agent.adapters.pipecat.settings import PipecatHeadSettings


class SpeechStack:
    """The three speech pieces a pipeline needs (real by default; tests pass stubs)."""

    def __init__(self, *, vad: VADAnalyzer, stt: FrameProcessor, tts: FrameProcessor) -> None:
        """Bind the VAD analyzer and the STT/TTS processors."""
        self.vad = vad
        self.stt = stt
        self.tts = tts


def real_speech(session: PipecatSession) -> SpeechStack:
    """Silero + NVIDIA ASR/TTS from the catalog endpoints."""
    return SpeechStack(
        vad=build_vad(session.settings),
        stt=build_stt(session.config),
        tts=build_tts(session.config, session.settings),
    )


class SessionPipeline:
    """Builds and owns one session's processors, transport and worker."""

    def __init__(self, session: PipecatSession, websocket: WebSocket, speech: SpeechStack) -> None:
        """Assemble the pipeline (it starts in :meth:`PipecatSession.run`)."""
        settings = session.settings
        self._build_turn_processors(session)
        audio = settings.audio
        self.transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_in_sample_rate=audio.in_sample_rate,
                audio_out_enabled=True,
                audio_out_sample_rate=audio.out_sample_rate,
                audio_out_10ms_chunks=audio.out_10ms_chunks,
                audio_out_auto_silence=False,
                audio_out_end_silence_secs=0,
                audio_out_write_timeout_secs=audio.write_timeout_s,
                serializer=RealtimeSerializer(session),
                session_timeout=None,
            ),
        )
        self.tts = speech.tts
        self.pipeline = Pipeline(
            [self.transport.input(), *self._middle_processors(speech, settings), self.transport.output()]
        )
        self.worker = PipelineWorker(
            self.pipeline,
            params=PipelineParams(
                audio_in_sample_rate=audio.in_sample_rate,
                audio_out_sample_rate=audio.out_sample_rate,
            ),
            enable_rtvi=False,
            idle_timeout_secs=None,
            cancel_on_idle_timeout=False,
        )

        self._cancel_on_disconnect()

    def _build_turn_processors(self, session: PipecatSession) -> None:
        """The input cursor, output tagger, user-turn processor and turn bridge (shared with the browser UI)."""
        settings = session.settings
        self.cursor = InputCursor()
        self.tagger = OutputTagger()
        turn = settings.turn
        stop = (
            SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=turn.user_speech_timeout)
            if turn.user_speech_timeout is not None
            else SpeechTimeoutUserTurnStopStrategy()
        )
        turn_kwargs: dict[str, Any] = {"user_turn_strategies": UserTurnStrategies(stop=[stop])}
        if turn.user_turn_stop_timeout is not None:
            turn_kwargs["user_turn_stop_timeout"] = turn.user_turn_stop_timeout
        self.user_turns = UserTurnProcessor(**turn_kwargs)
        self.bridge = FdaTurnBridge(
            config=session.config,
            emit_urgent=session.emit_urgent,
            conversation=session.conversation,
            cursor=self.cursor,
            tagger=self.tagger,
            log=session.log,
            clock=session.clock,
            session_id=session.session_id,
        )

    def _middle_processors(self, speech: SpeechStack, settings: PipecatHeadSettings) -> list[FrameProcessor]:
        """VAD, input cursor, STT, user turns, the turn bridge, TTS and the output tagger, in order."""
        return [
            VADProcessor(vad_analyzer=speech.vad, **settings.vad.processor_kwargs()),
            self.cursor,
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

    async def push_to_tts(self, frame: Frame) -> None:
        """Push a frame into the TTS service's input (from the bridge, upstream of it)."""
        await self.bridge.push_frame(frame)

    async def interrupt_output(self) -> None:
        """Flush queued agent audio (the turn manager stopped its own output)."""
        await self.bridge.broadcast_interruption()
