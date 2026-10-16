# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The browser UI pipeline: the Realtime pipeline's processors on the repository's UI transport.

::

    transport.input()                  WebRTC or the UI WebSocket (examples/shared create_transport)
      -> VADProcessor -> InputCursor -> STT -> UserTurnProcessor
      -> FdhTurnBridge -> TTS -> OutputTagger          (shared with SessionPipeline)
      -> UiOutputGate                  drops the Realtime-only frames; feeds the playout estimate
    transport.output()

The turn manager still emits its Realtime events, and the output tagger still orders them with
the audio, because both rely on them; the browser receives RTVI messages instead (the worker's
RTVI observer: transcripts, bot speaking; ``user-bot-latency`` from :mod:`ui_session`).

On the Realtime path the serializer feeds the playout estimate (what the client heard, for
barge-in truncation and unheard answers) from each audio chunk it writes; here
:class:`UiOutputGate` does the same from the audio it passes to the browser transport.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pipecat.frames.frames import Frame, InterruptionFrame, OutputAudioRawFrame
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from examples.frontend_delegation_hermes.adapters.pipecat.frames import (
    ItemAudioFrame,
    ResponseEventFrame,
    StreamEventFrame,
    UrgentEventFrame,
)
from examples.frontend_delegation_hermes.adapters.pipecat.pipeline import SessionPipeline

if TYPE_CHECKING:
    from examples.frontend_delegation_hermes.adapters.pipecat.pipeline import SpeechStack
    from examples.frontend_delegation_hermes.adapters.pipecat.session import PipecatSession
    from examples.frontend_delegation_hermes.adapters.pipecat.timing import PlayoutEstimator

#: Frames that only an OpenAI Realtime client understands; the browser never receives them.
REALTIME_ONLY_FRAMES = (StreamEventFrame, ResponseEventFrame, UrgentEventFrame)


class UiOutputGate(FrameProcessor):
    """Drops the Realtime event frames and item markers; counts the agent audio sent to the browser.

    Agent audio belongs to an item between its ``begin`` and end :class:`ItemAudioFrame`, as on the
    Realtime path, where only that audio becomes ``response.output_audio.delta`` and feeds the
    playout estimate. An interruption drops what the browser has not played.
    """

    def __init__(self, playout: PlayoutEstimator | None = None) -> None:
        """``playout`` is bound by the session once it exists (:meth:`bind`)."""
        super().__init__()
        self._playout = playout
        self._in_item = False

    def bind(self, playout: PlayoutEstimator) -> None:
        """The session's playout estimate."""
        self._playout = playout

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Pass every frame except the Realtime-only ones; note item audio and interruptions."""
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, ItemAudioFrame):
                self._in_item = frame.begin
                return
            if isinstance(frame, REALTIME_ONLY_FRAMES):
                return
            if isinstance(frame, InterruptionFrame):
                self._in_item = False
                if self._playout is not None:
                    self._playout.on_interruption()
            elif isinstance(frame, OutputAudioRawFrame) and self._in_item and frame.audio and frame.sample_rate:
                if self._playout is not None:
                    channels = max(1, frame.num_channels)
                    self._playout.on_written(len(frame.audio) * 1000.0 / (frame.sample_rate * 2 * channels))
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
        self._build_turn_processors(session)
        self.transport = transport
        self.tts = speech.tts
        self.gate = UiOutputGate()
        self.pipeline = Pipeline(
            [transport.input(), *self._middle_processors(speech, settings), self.gate, transport.output()]
        )
        self.latency_observer = UserBotLatencyObserver()
        self.worker = PipelineWorker(
            self.pipeline,
            params=PipelineParams(
                audio_in_sample_rate=settings.audio.in_sample_rate,
                audio_out_sample_rate=settings.audio.out_sample_rate,
                enable_metrics=True,
            ),
            observers=[self.latency_observer],
            enable_rtvi=True,
            idle_timeout_secs=idle_timeout_secs,
        )
        self._cancel_on_disconnect()
