# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hosting a Pipecat pipeline for a live session: the caller's transport around the frontend's processors."""

from __future__ import annotations

import asyncio

from pipecat.frames.frames import InputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, ProcessorUnusablePolicy
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner

from examples.frontend_backend_live.engine.base import BaseLiveEngine
from examples.shared.pipeline_utils import build_pipeline_params
from live.media.processors import LiveEventObserver, LiveInputGate, LiveWebSocketAudioOutput

LIVE_AUDIO_OUT_RATE = 24000
START_TIMEOUT_SECONDS = 30


class PipelineLiveEngine(BaseLiveEngine):
    """A live engine whose frontend is a Pipecat pipeline (the cascade, or a remote realtime model's processor)."""

    def __init__(self, *args, **kwargs):
        """Pass everything to :class:`BaseLiveEngine`; the pipeline exists once :meth:`run_pipeline` runs."""
        super().__init__(*args, **kwargs)
        self.gate: LiveInputGate | None = None  # ``input_ms`` is the session's input media clock
        self.task: PipelineWorker | None = None
        self.runner_task: asyncio.Task | None = None

    def audio_pipeline(
        self, middle: list, *, transport: str, connection, audio_in_rate: int, tail: list | None = None
    ) -> list:
        """Wrap the frontend's processors with the caller's transport (WebRTC or the primary WebSocket).

        Args:
            middle: The frontend's processors, from the input gate to the speech output.
            transport: ``webrtc`` or ``websocket``.
            connection: The WebRTC connection (``webrtc`` only).
            audio_in_rate: The sample rate the frontend wants caller audio at.
            tail: Processors that follow the audio output (for example an assistant aggregator).
        """
        tail = tail or []
        if transport == "webrtc":
            from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

            webrtc = SmallWebRTCTransport(
                webrtc_connection=connection,
                params=TransportParams(
                    audio_in_enabled=True,
                    audio_in_sample_rate=audio_in_rate,
                    audio_out_enabled=True,
                    audio_out_sample_rate=LIVE_AUDIO_OUT_RATE,
                    audio_out_10ms_chunks=5,
                ),
            )
            return [webrtc.input(), *middle, webrtc.output(), *tail]
        return [*middle, LiveWebSocketAudioOutput(self.session), *tail]

    async def run_pipeline(self, processors: list, *, stt, tts) -> None:
        """Start the pipeline and return once it is running; closing the pipeline closes the session.

        Args:
            processors: The complete pipeline, from :meth:`audio_pipeline`.
            stt: The processor whose frames carry the caller's transcripts.
            tts: The processor whose frames carry the spoken reply.
        """
        session = self.session
        self.task = PipelineWorker(
            Pipeline(processors),
            params=build_pipeline_params(audio_out_sample_rate=LIVE_AUDIO_OUT_RATE),
            idle_timeout_secs=None,
            enable_rtvi=False,
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
            if session.status == "active":
                session.spawn(session.close("connection_lost"), "pipeline-finished")

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

    async def feed_audio(self, pcm24k: bytes) -> None:
        """Deliver caller audio from a primary WebSocket into the pipeline."""
        await self.task.queue_frame(InputAudioRawFrame(audio=pcm24k, sample_rate=LIVE_AUDIO_OUT_RATE, num_channels=1))
