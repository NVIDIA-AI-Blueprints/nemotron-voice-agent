# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hosting a live session's Pipecat pipeline: the caller's transport and the live edges around the frontend."""

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
# The roots import ``tracing``, which starts OpenTelemetry; the adapter only turns pipeline tracing on.
TRACING_SWITCH = "ENABLE_TRACING"
AUDIO_OUT_CHUNKS_SETTING = "AUDIO_OUT_10MS_CHUNKS"
WEBRTC_AUDIO_OUT_CHUNKS = 5
WEBSOCKET_AUDIO_OUT_CHUNKS = 10
WEBRTC_AUDIO_OUT_RATE = 24_000


class MediaEdges(Protocol):
    """Where a session's caller audio enters its pipeline and its reply audio leaves, for one transport."""

    def ends(
        self, session: SessionPort, connection: MediaConnection | None, audio_in_rate: int
    ) -> tuple[list[FrameProcessor], list[FrameProcessor]]:
        """Return this transport's processors before the input gate, and after the frontend's speech output."""


class WebRtcEdges:
    """A WebRTC peer: Pipecat's transport takes the caller's audio from the peer's track and plays the reply on it."""

    def ends(
        self, session: SessionPort, connection: MediaConnection | None, audio_in_rate: int
    ) -> tuple[list[FrameProcessor], list[FrameProcessor]]:
        """Return Pipecat's WebRTC input, and its output at :data:`WEBRTC_AUDIO_OUT_RATE`."""
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
    """A primary WebSocket: Pipecat's WebSocket transport on the gateway's socket, with the Live serializer."""

    def ends(
        self, session: SessionPort, connection: MediaConnection | None, audio_in_rate: int
    ) -> tuple[list[FrameProcessor], list[FrameProcessor]]:
        """Return Pipecat's WebSocket input, and its output followed by the silence between lines."""
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


MEDIA_EDGES: dict[str, type[MediaEdges]] = {"webrtc": WebRtcEdges, "websocket": WebSocketEdges}


def latency_observer() -> UserBotLatencyObserver:
    """Return Pipecat's user-to-bot latency observer, which logs each turn's latency and its breakdown."""
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
    """A frontend adapter whose frontend is a Pipecat pipeline, wrapped in the caller's transport and the live edges."""

    def __init__(self):
        """Start empty: the session's view comes with the start, and the pipeline once :meth:`run_pipeline` runs."""
        self.session: SessionPort | None = None
        self.gate: LiveInputGate | None = None
        self.recorder: AudioBufferProcessor | None = None
        self.task: PipelineWorker | None = None
        self.runner_task: asyncio.Task | None = None
        self.background: set[asyncio.Task] = set()

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
                report.add_done_callback(lambda done: done.cancelled() or done.exception())

        task.add_done_callback(done)

    def edges(self, *, transport: str, connection, audio_in_rate: int, stt) -> FrontendEdges:
        """Return the session's edges around the frontend: its transport, the input gate and the output transcript."""
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
        """Run ``worker``'s pipeline and return once it runs; a processor that could not set up raises."""
        session = self.session
        self.task = worker
        started = asyncio.Event()
        # The input reads the caller while later processors set up: the gate drops that audio until the start.
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
        # Pipecat starts the pipeline even when a processor's setup failed: such a processor fails the start.
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
