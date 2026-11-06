# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hosting a Pipecat pipeline for a live session: the caller's transport around the frontend's processors.

The pipeline runs at the shared rates (``examples.shared.pipeline_utils``): caller audio at 16 kHz, reply audio at
``PIPELINE_AUDIO_OUT_SAMPLE_RATE``, which every Magpie deployment accepts. The edges convert: each transport's input
hands the frontend 16 kHz audio, and its output sends the caller reply audio at the transport's own rate (WebRTC's, or
a primary WebSocket's wire rate). Each transport is a class of :data:`MEDIA_EDGES`, and the host gives the frontend's
build (``examples.frontend_backend_live.frontend``) the session's edges.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol

from loguru import logger
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.worker import PipelineWorker
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner

from examples.frontend_backend_live.adapter.bridge import (
    LiveEventObserver,
    LiveInputGate,
    LiveOutputTranscript,
    LiveWebSocketSilence,
)
from examples.frontend_backend_live.adapter.serializer import LiveFrameSerializer
from examples.frontend_backend_live.frontend import FrontendEdges
from examples.shared.audio_recorder import create_audio_recorder
from examples.shared.pipeline_utils import build_pipeline_params
from utils import parse_env_bool, parse_env_int

if TYPE_CHECKING:
    from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
    from pipecat.processors.frame_processor import FrameProcessor

    from gateways.live.ports import MediaConnection, SessionPort

START_TIMEOUT_SECONDS = 30
# The switch the repository's tracing setup reads (``src/tracing.py``); the roots import that module, which starts
# OpenTelemetry, and the adapter only turns pipeline tracing on.
TRACING_SWITCH = "ENABLE_TRACING"
# How many 10 ms chunks reply audio is written in, as the shared transports read it: WebRTC's, and a WebSocket's.
AUDIO_OUT_CHUNKS_SETTING = "AUDIO_OUT_10MS_CHUNKS"
WEBRTC_AUDIO_OUT_CHUNKS = 5
WEBSOCKET_AUDIO_OUT_CHUNKS = 10
# The rate WebRTC reply audio is written at, which the peer's audio track takes.
WEBRTC_AUDIO_OUT_RATE = 24_000


class MediaEdges(Protocol):
    """Where a session's caller audio enters its pipeline and where the reply audio leaves it, for one transport."""

    def ends(
        self, session: SessionPort, connection: MediaConnection | None, audio_in_rate: int
    ) -> tuple[list[FrameProcessor], list[FrameProcessor]]:
        """This transport's processors before the input gate, and after the frontend's speech output."""


class WebRtcEdges:
    """A WebRTC peer: Pipecat's transport takes the caller's audio from the peer's track and plays the reply on it."""

    def ends(
        self, session: SessionPort, connection: MediaConnection | None, audio_in_rate: int
    ) -> tuple[list[FrameProcessor], list[FrameProcessor]]:
        """Pipecat's WebRTC input, and its output, which writes reply audio at :data:`WEBRTC_AUDIO_OUT_RATE`."""
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

        webrtc = SmallWebRTCTransport(
            webrtc_connection=connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_in_sample_rate=audio_in_rate,
                audio_out_enabled=True,
                audio_out_sample_rate=WEBRTC_AUDIO_OUT_RATE,
                audio_out_10ms_chunks=parse_env_int(AUDIO_OUT_CHUNKS_SETTING, WEBRTC_AUDIO_OUT_CHUNKS),
            ),
        )
        return [webrtc.input()], [webrtc.output()]


class WebSocketEdges:
    """A primary WebSocket: Pipecat's WebSocket transport reads the gateway's socket, with the Live serializer.

    The gateway's socket (the session view's ``socket``) runs every client event but caller audio, and the serializer
    turns caller audio into frames and reply audio into ``session.output_audio.delta`` events at the wire rate, which
    the session sends. The silence between lines follows the output.
    """

    def ends(
        self, session: SessionPort, connection: MediaConnection | None, audio_in_rate: int
    ) -> tuple[list[FrameProcessor], list[FrameProcessor]]:
        """Pipecat's WebSocket input, and its output with the silence between lines after it."""
        from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

        serializer = LiveFrameSerializer(session.socket, session, audio_in_rate)
        websocket = FastAPIWebsocketTransport(
            websocket=session.socket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_in_sample_rate=audio_in_rate,
                audio_out_enabled=True,
                audio_out_sample_rate=serializer.rate,
                audio_out_10ms_chunks=parse_env_int(AUDIO_OUT_CHUNKS_SETTING, WEBSOCKET_AUDIO_OUT_CHUNKS),
                # No trailing silence of Pipecat's own: LiveWebSocketSilence sends the silence between lines.
                audio_out_end_silence_secs=0,
                # The gateway checked the client's origin before it accepted the socket.
                allowed_origins=[],
                serializer=serializer,
            ),
        )
        return [websocket.input()], [websocket.output(), LiveWebSocketSilence(session, serializer)]


# Read per session, like the gateway's delegation parts: the transport is the client's choice of connection.
MEDIA_EDGES: dict[str, type[MediaEdges]] = {"webrtc": WebRtcEdges, "websocket": WebSocketEdges}


def latency_observer() -> UserBotLatencyObserver:
    """Pipecat's user-to-bot latency observer, which logs each turn's latency and its breakdown as the UI bot does.

    The breakdown names each part of the turn, from the caller's silence to the bot's first audio, with the metrics the
    session's worker turns on. A live session has no RTVI client to send them to, so the handlers only log.
    """
    observer = UserBotLatencyObserver()

    @observer.event_handler("on_latency_measured")
    async def on_latency(observer, latency):
        logger.info(f"User-to-bot latency: {latency:.3f}s")

    @observer.event_handler("on_latency_breakdown")
    async def on_breakdown(observer, breakdown):
        if events := breakdown.turn_contribution_lines():
            logger.info(f"Latency breakdown: {' | '.join(events)}")

    return observer


class PipelineHost:
    """A frontend adapter whose frontend is a Pipecat pipeline.

    The host wraps the frontend's processors with the caller's transport, the input gate (the session's input clock and
    mute) and the output transcript (the sentences the caller heard), so every adapter it hosts has all three.
    """

    def __init__(self):
        """Start empty: the session's view comes with the start, and the pipeline once :meth:`run_pipeline` runs."""
        self.session: SessionPort | None = None  # the frontend adapter's view of the session, from its start
        self.gate: LiveInputGate | None = None  # ``input_ms`` is the session's input media clock
        self.recorder: AudioBufferProcessor | None = None  # the shared audio recorder; None while no audio dump is on
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

    def edges(self, *, transport: str, connection, audio_in_rate: int, stt) -> FrontendEdges:
        """The session's edges around the frontend: the caller's transport, the input gate and the output transcript.

        The transcript, after the output, sends the sentences that played. The worker runs at the shared rates with no
        RTVI and no idle close, is traced, under the session's id, when tracing is on, and reports the caller's
        transcripts and the pipeline's errors as live events. It runs with Pipecat's metrics on, as the UI bot's does,
        and logs each turn's latency (:func:`latency_observer`).

        When ``ENABLE_ASR_AUDIO_DUMP`` or ``ENABLE_TTS_AUDIO_DUMP`` is on, the shared audio recorder
        (``examples.shared.audio_recorder``) follows the transcript, as it follows the UI bot's output, and records from
        the pipeline's start.

        Args:
            transport: The session's transport, a key of :data:`MEDIA_EDGES` (``webrtc`` or ``websocket``).
            connection: The WebRTC connection (``webrtc`` only).
            audio_in_rate: The sample rate the frontend wants caller audio at.
            stt: The processor whose frames carry the caller's transcripts.
        """
        self.gate = LiveInputGate(self.session)
        self.recorder = create_audio_recorder()
        inputs, outputs = MEDIA_EDGES[transport]().ends(self.session, connection, audio_in_rate)
        worker = {
            "params": build_pipeline_params(enable_metrics=True, enable_usage_metrics=True),
            "idle_timeout_secs": None,
            "enable_rtvi": False,
            "enable_tracing": parse_env_bool(TRACING_SWITCH),
            "conversation_id": self.session.id,
            "observers": [LiveEventObserver(self.session, self.gate, stt), latency_observer()],
        }
        transcript = LiveOutputTranscript(self.session)
        after = [transcript, *([self.recorder] if self.recorder else [])]
        return FrontendEdges(inputs=[*inputs, self.gate], outputs=outputs, worker=worker, after=after)

    async def run_pipeline(self, worker: PipelineWorker, processors: list) -> None:
        """Run ``worker``'s pipeline and return once it is running; closing the pipeline closes the session.

        Args:
            worker: The pipeline's worker, on the session's :meth:`edges`.
            processors: The pipeline's processors, each of which must have set up.
        """
        session = self.session
        self.task = worker
        started = asyncio.Event()
        # A transport's input reads the caller while later processors set up: the gate drops that audio until the start.
        self.gate.muted = True

        @self.task.event_handler("on_pipeline_started")
        async def on_started(worker, frame):
            self.gate.muted = False
            if self.recorder:
                await self.recorder.start_recording()
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
        # Pipecat starts the pipeline once each processor's setup has ended, a failed one included: it fails the start.
        if failed := [str(processor) for processor in processors if not processor.is_usable]:
            raise RuntimeError(f"Processors that could not set up: {', '.join(failed)}")

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
