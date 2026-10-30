# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Shared doubles of the frontend adapter's Pipecat-side tests."""

import asyncio

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

from live.ports import EXCHANGE_RATE

SILENCE = 0
PACKET_BYTES = EXCHANGE_RATE // 10 * 2  # 100 ms of PCM16 at the exchange rate


def tone(ms: int, rate: int = EXCHANGE_RATE, level: int = 0x1010) -> bytes:
    """``ms`` of a constant PCM16 level at ``rate``."""
    return level.to_bytes(2, "little", signed=True) * (rate * ms // 1000)


class RecordingPort:
    """A frontend adapter's view of its session that notes what the adapter sends, in order."""

    def __init__(self, active=True, session_id="live_test"):
        self.id, self.input_ms, self.active = session_id, 0, active
        # ("emit", kind, fields), ("audio", pcm, rate), ("error", message) or ("close", reason), in order
        self.log: list[tuple] = []

    def is_active(self):
        return self.active

    async def emit(self, kind, **fields):
        self.log.append(("emit", kind, fields))

    async def advance_input_clock(self, ms):
        self.input_ms = max(self.input_ms, ms)

    async def send_output_audio(self, pcm, sample_rate):
        self.log.append(("audio", pcm, sample_rate))

    async def report_error(self, message, client_event_id=None):
        self.log.append(("error", message))

    def request_close(self, reason):
        self.log.append(("close", reason))

    def packets(self):
        return [entry for entry in self.log if entry[0] == "audio"]

    def voiced(self):
        """The packets that carry sound."""
        return [entry for entry in self.packets() if any(entry[1])]

    def transcript(self):
        return [entry[2] for entry in self.log if entry[:2] == ("emit", "session.output_transcript.delta")]


class Probe(FrameProcessor):
    """Records every frame that reaches it, with its direction, and passes it on."""

    def __init__(self):
        super().__init__()
        self.frames = []
        self.directions = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        self.frames.append(frame)
        self.directions.append(direction)
        await self.push_frame(frame, direction)

    def of(self, *kinds, direction=FrameDirection.DOWNSTREAM):
        return [f for f, d in zip(self.frames, self.directions, strict=True) if isinstance(f, kinds) and d is direction]


FLUSH = object()  # a step of run_frames: wait until every frame queued before it has passed the whole pipeline


async def run_frames(processors, frames, *, settle=0.15, gap=0.02, observers=None, params=None):
    """Run ``processors`` in a pipeline: once started, queue ``frames`` ``gap`` apart, wait ``settle``, then cancel it.

    A processor takes the frames waiting for it system frames first, so a cut queued right behind a line can overtake
    the line. A :data:`FLUSH` step before the cut keeps it behind the line, as a text-to-speech passes a cut on only
    after the frames it pushed before it.
    """
    worker = PipelineWorker(Pipeline(processors), observers=observers or [], idle_timeout_secs=None, **(params or {}))
    started = asyncio.Event()
    worker.add_event_handler("on_pipeline_started", lambda worker, frame: started.set())
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    await asyncio.wait_for(started.wait(), timeout=5)
    for frame in frames:
        if frame is FLUSH:
            await worker.flush_pipeline()
        else:
            await worker.queue_frame(frame)
        await asyncio.sleep(gap)
    await asyncio.sleep(settle)
    await worker.cancel()
    await asyncio.gather(running, return_exceptions=True)
