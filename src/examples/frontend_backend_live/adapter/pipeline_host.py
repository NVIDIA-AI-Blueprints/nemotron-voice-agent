# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hosting a Pipecat pipeline for a live session: the caller's transport around the frontend's processors.

The pipeline runs at the shared rates (``examples.shared.pipeline_utils``): caller audio at 16 kHz, reply audio at
``PIPELINE_AUDIO_OUT_SAMPLE_RATE``, which every Magpie deployment accepts. The edges convert: the input gate hands the
frontend 16 kHz audio, and both outputs send the caller reply audio at the exchange rate (``live.ports.EXCHANGE_RATE``).
Each transport is a class of :data:`MEDIA_EDGES`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol

from loguru import logger
from pipecat.frames.frames import InputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, ProcessorUnusablePolicy
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner

from examples.frontend_backend_live.adapter.bridge import (
    LiveEventObserver,
    LiveInputGate,
    LiveOutputTranscript,
    LiveWebSocketAudioOutput,
)
from examples.frontend_backend_live.media.playout import PlayoutTracker
from examples.shared.pipeline_utils import build_pipeline_params
from live.ports import EXCHANGE_RATE
from utils import parse_env_bool, parse_env_int

if TYPE_CHECKING:
    from live.ports import MediaConnection, SessionPort

START_TIMEOUT_SECONDS = 30
# The switch the repository's tracing setup reads (``src/tracing.py``); the roots import that module, which starts
# OpenTelemetry, and the adapter only turns pipeline tracing on.
TRACING_SWITCH = "ENABLE_TRACING"
# How many 10 ms chunks WebRTC reply audio is written in, as the shared WebRTC transport reads it.
AUDIO_OUT_CHUNKS_SETTING = "AUDIO_OUT_10MS_CHUNKS"
WEBRTC_AUDIO_OUT_CHUNKS = 5


class MediaEdges(Protocol):
    """Where a session's caller audio enters its pipeline and where the reply audio leaves it, for one transport."""

    def wrap(self, inner: list, session: SessionPort, connection: MediaConnection | None, audio_in_rate: int) -> list:
        """``inner`` (the input gate, then the frontend's processors) between this transport's input and output."""


class WebRtcEdges:
    """A WebRTC peer: Pipecat's transport takes the caller's audio from the peer's track and plays the reply on it."""

    def wrap(self, inner: list, session: SessionPort, connection: MediaConnection | None, audio_in_rate: int) -> list:
        """Pipecat's WebRTC input, ``inner``, then its output, which writes reply audio at the exchange rate."""
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

        webrtc = SmallWebRTCTransport(
            webrtc_connection=connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_in_sample_rate=audio_in_rate,
                audio_out_enabled=True,
                audio_out_sample_rate=EXCHANGE_RATE,
                audio_out_10ms_chunks=parse_env_int(AUDIO_OUT_CHUNKS_SETTING, WEBRTC_AUDIO_OUT_CHUNKS),
            ),
        )
        return [webrtc.input(), *inner, webrtc.output()]


class WebSocketEdges:
    """A primary WebSocket: the gateway feeds the caller's audio in (``feed_audio``), a clocked output sends replies."""

    def wrap(self, inner: list, session: SessionPort, connection: MediaConnection | None, audio_in_rate: int) -> list:
        """``inner``, then the WebSocket's clocked reply-audio output."""
        return [*inner, LiveWebSocketAudioOutput(session)]


# Read per session, like the gateway's delegation parts: the transport is the client's choice of connection.
MEDIA_EDGES: dict[str, type[MediaEdges]] = {"webrtc": WebRtcEdges, "websocket": WebSocketEdges}


class PipelineHost:
    """A frontend adapter whose frontend is a Pipecat pipeline.

    The host wraps the frontend's processors with the caller's transport, the input gate (the session's input clock and
    mute) and the playout tracker (what the caller heard), so every adapter it hosts has all three.
    """

    def __init__(self):
        """Start empty: the session's view comes with the start, and the pipeline once :meth:`run_pipeline` runs."""
        self.session: SessionPort | None = None  # the frontend adapter's view of the session, from its start
        self.gate: LiveInputGate | None = None  # ``input_ms`` is the session's input media clock
        self.playout: PlayoutTracker | None = None  # what each line played; its facts go to ``playout.listen()``
        self.task: PipelineWorker | None = None
        self.runner_task: asyncio.Task | None = None
        self.background: set[asyncio.Task] = set()  # background commands; stop_pipeline cancels them

    def _spawn(self, coro) -> None:
        """Run a command in the background; a failure is reported to the client without ending the session."""
        task = asyncio.create_task(coro)
        self.background.add(task)

        def done(finished: asyncio.Task) -> None:
            self.background.discard(finished)
            if not finished.cancelled() and finished.exception():
                logger.error(f"Live command failed: {finished.exception()!r}")
                report = asyncio.create_task(self.session.report_error("A command could not be processed."))
                self.background.add(report)
                report.add_done_callback(self.background.discard)

        task.add_done_callback(done)

    def audio_pipeline(self, middle: list, *, transport: str, connection, audio_in_rate: int) -> list:
        """Wrap the frontend's processors with the input gate, the caller's transport and the playout tracker.

        Args:
            middle: The frontend's processors, from the speech-to-text to the speech output.
            transport: The session's transport, a key of :data:`MEDIA_EDGES` (``webrtc`` or ``websocket``).
            connection: The WebRTC connection (``webrtc`` only).
            audio_in_rate: The sample rate the frontend wants caller audio at.
        """
        self.gate = LiveInputGate(self.session, audio_in_rate)
        self.playout = PlayoutTracker(LiveOutputTranscript(self.session))
        edges = MEDIA_EDGES[transport]()
        return [*edges.wrap([self.gate, *middle], self.session, connection, audio_in_rate), self.playout]

    async def run_pipeline(self, processors: list, *, stt, tts) -> None:
        """Start the pipeline and return once it is running; closing the pipeline closes the session.

        The worker runs at the shared rates and is traced, under the session's id, when tracing is on.

        Args:
            processors: The complete pipeline, from :meth:`audio_pipeline`.
            stt: The processor whose frames carry the caller's transcripts.
            tts: The processor whose frames carry the spoken reply.
        """
        session = self.session
        self.task = PipelineWorker(
            Pipeline(processors),
            params=build_pipeline_params(),
            idle_timeout_secs=None,
            enable_rtvi=False,
            enable_tracing=parse_env_bool(TRACING_SWITCH),
            conversation_id=session.id,
            observers=[LiveEventObserver(session, self.gate, stt, tts)],
            processor_unusable_policy=ProcessorUnusablePolicy.END,
            setup_timeout_secs=120.0,
        )
        started = asyncio.Event()

        @self.task.event_handler("on_pipeline_started")
        async def on_started(worker, frame):
            started.set()

        @self.task.event_handler("on_pipeline_finished")
        async def on_finished(worker, frame):
            if session.is_active():
                session.request_close("connection_lost")

        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(self.task)
        self.runner_task = asyncio.create_task(runner.run(), name=f"{session.id}:pipeline")
        async with asyncio.timeout(START_TIMEOUT_SECONDS):
            await started.wait()

    async def stop_pipeline(self) -> None:
        """Cancel the background commands and the pipeline."""
        for background in tuple(self.background):
            background.cancel()
        if self.task is not None:
            await self.task.cancel()
        if self.runner_task is not None:
            await asyncio.gather(self.runner_task, return_exceptions=True)

    async def set_muted(self, muted: bool) -> None:
        """Stop or resume consuming caller audio."""
        self.gate.muted = muted

    async def feed_audio(self, pcm: bytes, sample_rate: int) -> None:
        """Deliver caller audio from a primary WebSocket into the pipeline; the gate converts it to the input rate."""
        await self.task.queue_frame(InputAudioRawFrame(audio=pcm, sample_rate=sample_rate, num_channels=1))
