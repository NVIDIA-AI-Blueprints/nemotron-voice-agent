# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The browser UI pipeline: the Realtime pipeline's processors on the repository's UI transport.

Plan: ``misc/frontend-backend-verdict/ui-transport-plan.md``, section 4.3::

    transport.input()                  WebRTC or the UI WebSocket (examples/shared create_transport)
      -> VadParamsGate                 applies the session's turn-end silence once the VAD knows its rate
      -> VADProcessor -> InputCursor -> AsrErrorTap -> STT -> UserTurnProcessor
      -> FbaTurnBridge -> TTS -> OutputTagger          (shared with SessionPipeline)
      -> RealtimeFrameFilter           drops the Realtime event frames and item markers
    transport.output()

The turn manager still emits its Realtime events, and the output tagger still orders them with
the audio, because both rely on them; the browser receives RTVI messages instead (the worker's
RTVI observer: transcripts, bot speaking; ``user-bot-latency`` from :mod:`ui_session`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pipecat.frames.frames import Frame, InputAudioRawFrame
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from examples.frontend_backend_agent.adapters.pipecat import switches
from examples.frontend_backend_agent.adapters.pipecat.frames import (
    ItemAudioFrame,
    ResponseEventFrame,
    StreamEventFrame,
    UrgentEventFrame,
)
from examples.frontend_backend_agent.adapters.pipecat.pipeline import SessionPipeline

if TYPE_CHECKING:
    from examples.frontend_backend_agent.adapters.pipecat.pipeline import SpeechStack
    from examples.frontend_backend_agent.adapters.pipecat.session import PipecatSession

#: Frames that only an OpenAI Realtime client understands; the browser never receives them.
REALTIME_ONLY_FRAMES = (ItemAudioFrame, StreamEventFrame, ResponseEventFrame, UrgentEventFrame)


class RealtimeFrameFilter(FrameProcessor):
    """Drops the Realtime event frames and item markers after the output tagger has used them."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Pass every frame except :data:`REALTIME_ONLY_FRAMES`."""
        await super().process_frame(frame, direction)
        if isinstance(frame, REALTIME_ONLY_FRAMES):
            return
        await self.push_frame(frame, direction)


class VadParamsGate(FrameProcessor):
    """Applies pending VAD params before each input audio frame reaches the VAD.

    On the Realtime path the session does this when it decodes each appended chunk; a browser
    transport delivers audio frames directly, so this processor takes that role.
    """

    def __init__(self, pipeline: SessionPipeline) -> None:
        """Bind the pipeline whose pending VAD params to apply."""
        super().__init__()
        self._owner = pipeline

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Apply pending params on input audio, then pass the frame on."""
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            self._owner.apply_pending_vad()
        await self.push_frame(frame, direction)


class UiSessionPipeline(SessionPipeline):
    """The shared processors on a browser transport, with RTVI and the user-to-bot latency observer."""

    def __init__(
        self,
        session: PipecatSession,
        transport: Any,
        speech: SpeechStack,
        *,
        idle_timeout_secs: float | None = None,
    ) -> None:
        """Assemble the pipeline on ``transport`` (from ``create_transport``); it starts with the session."""
        settings = session.settings
        self._build_turn_processors(session, speech)
        self.serializer = None
        self.transport = transport
        self.frame_filter = RealtimeFrameFilter()
        self.pipeline = Pipeline(
            [
                transport.input(),
                VadParamsGate(self),
                *self._middle_processors(speech, settings),
                self.frame_filter,
                transport.output(),
            ]
        )
        self.latency_observer = UserBotLatencyObserver()
        self.worker = PipelineWorker(
            self.pipeline,
            params=PipelineParams(
                audio_in_sample_rate=settings.audio.in_sample_rate,
                audio_out_sample_rate=settings.audio.out_sample_rate or session.config.tts.sample_rate,
                enable_metrics=True,
            ),
            observers=[self.latency_observer],
            enable_rtvi=True,
            idle_timeout_secs=idle_timeout_secs,
            **({"enable_tracing": True} if switches.ui_tracing() else {}),
        )
        self._cancel_on_disconnect()
