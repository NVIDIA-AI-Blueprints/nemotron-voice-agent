# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Incremental native output buffering with sample-clock WebSocket writes."""

import asyncio
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from math import ceil

from pipecat.frames.frames import BotSpeakingFrame, MetricsFrame, OutputAudioRawFrame, TTSStoppedFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_output import BOT_VAD_STOP_FALLBACK_SECS, BaseOutputTransport
from pipecat.transports.base_transport import TransportParams

from .models import RATE, AudioSpan, ns_to_samples
from .playback import WebsocketSinkClock


@dataclass(frozen=True)
class OutputDelivery:
    """A native FIFO dequeue, distinct from completion and client playout."""

    generation: str | None
    frame: object | None
    complete_ns: int
    notifications: tuple


class _Sink:
    def __init__(self):
        self.notifications = []

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.notifications.append((frame, direction))

    async def broadcast_frame(self, cls):
        await self.push_frame(cls())
        await self.push_frame(cls(), FrameDirection.UPSTREAM)

    async def write_transport_frame(self, frame):
        pass


class _SampleSender(BaseOutputTransport.MediaSender):
    """Change only the native speaking heartbeat's clock source."""

    logical_ns = 0

    async def _bot_currently_speaking(self):
        await self._bot_started_speaking()
        now = self.logical_ns / 1_000_000_000
        if now - self._bot_speaking_frame_time >= self._bot_speaking_frame_period:
            await self._transport.broadcast_frame(BotSpeakingFrame)
            self._bot_speaking_frame_time = now
        self._bot_speech_last_time = now


class IncrementalOutput:
    """Retain MediaSender's buffers, FIFO, stop flush and bot notifications.

    The scheduler supplies provider arrivals and calls pop at the next write or
    starvation deadline. No socket sleep is executed. Native-rate emitted PCM
    is resampled once at recording finalization, with cumulative sample cuts.
    """

    def __init__(self, *, sample_rate, chunks_10ms):
        """Create a persistent native buffer without wall-clock playback tasks."""
        if sample_rate <= 0 or chunks_10ms <= 0:
            raise ValueError("Output sample rate and chunk count must be positive")
        self.sink = _Sink()
        self.sender = _SampleSender(
            self.sink,
            destination=None,
            sample_rate=sample_rate,
            audio_chunk_size=(sample_rate // 100) * 2 * chunks_10ms,
            params=TransportParams(audio_out_enabled=True),
        )
        self.sender._audio_queue = asyncio.Queue()
        self.rate = sample_rate
        self.interval = Fraction(self.sender._audio_chunk_size * 1_000_000_000, sample_rate * 2)
        self.pacing = WebsocketSinkClock()
        self.pending = deque()
        self.consumer_ns = Fraction(0)
        self.idle_ns = None
        self.generation = None
        self.recordings = {}
        self.cutoffs = {}
        self.presentation_sample = 0

    @property
    def next_ns(self):
        """Earliest FIFO dequeue or native empty-queue stop, if any."""
        if self.pending:
            return ceil(max(self.consumer_ns, self.pending[0][0]))
        return ceil(self.idle_ns) if self.idle_ns is not None else None

    async def receive(self, generation, ready_ns, frame):
        """Pass one arrived provider frame through the native packing methods."""
        if isinstance(frame, MetricsFrame):
            return
        if isinstance(frame, OutputAudioRawFrame):
            if frame.num_channels != 1 or len(frame.audio) % 2 or not frame.interruptible:
                raise ValueError("Offline output supports interruptible mono PCM16")
            await self.sender.handle_audio_frame(frame)
        elif isinstance(frame, TTSStoppedFrame):
            await self.sender.handle_tts_stopped(frame)
        else:
            await self.sender.handle_sync_frame(frame)
        while not self.sender._audio_queue.empty():
            self.pending.append((ready_ns, generation, self.sender._audio_queue.get_nowait()))

    async def pop(self, at_ns):
        """Run the next native consumer operation at its scheduled offset."""
        expected = self.next_ns
        if expected is None or at_ns < expected:
            raise ValueError("Output dequeue precedes its availability")
        self.sender.logical_ns = at_ns
        self.sink.notifications.clear()
        if not self.pending:
            await self.sender._bot_stopped_speaking()
            self.idle_ns = None
            return OutputDelivery(self.generation, None, at_ns, tuple(self.sink.notifications))
        _, generation, frame = self.pending.popleft()
        self.generation = generation
        await self.sender._handle_frame(frame)
        complete = Fraction(at_ns)
        if isinstance(frame, OutputAudioRawFrame):
            complete = self.pacing.write_complete(complete, self.interval)
            recording = self.recordings.setdefault(generation, {"pcm": [], "segments": [], "native_samples": 0})
            before = round(recording["native_samples"] * RATE / self.rate)
            recording["native_samples"] += len(frame.audio) // 2
            end = round(recording["native_samples"] * RATE / self.rate)
            start = max(ns_to_samples(at_ns), self.presentation_sample)
            recording["pcm"].append(frame.audio)
            recording["segments"].append((start, before, end))
            self.presentation_sample = start + end - before
        self.consumer_ns = complete
        self.idle_ns = (
            complete + round(BOT_VAD_STOP_FALLBACK_SECS * 1_000_000_000) if self.sender._bot_speaking else None
        )
        return OutputDelivery(generation, frame, ceil(complete), tuple(self.sink.notifications))

    async def interrupt(self, generation, at_ns):
        """Apply native queue cancellation and clip client-buffered recording PCM."""
        self.cutoffs[generation] = ns_to_samples(at_ns)
        self.pending = deque(item for item in self.pending if item[1] != generation)
        self.sender.logical_ns = at_ns
        self.sink.notifications.clear()
        await self.sender._bot_stopped_speaking()
        self.sender._clear_audio_buffer()
        await self.sender._resampler.reset()
        self.pacing.interrupt()
        self.consumer_ns = Fraction(at_ns)
        self.presentation_sample = ns_to_samples(at_ns)
        self.idle_ns = None
        return tuple(self.sink.notifications)

    def spans(self):
        """Render emitted native PCM; never infer text timing from the waveform."""
        import numpy as np
        import soxr

        spans = []
        for generation, record in self.recordings.items():
            pcm = b"".join(record["pcm"])
            if self.rate != RATE:
                pcm = soxr.resample(np.frombuffer(pcm, dtype=np.int16), self.rate, RATE, quality="VHQ").tobytes()
            cutoff = self.cutoffs.get(generation, float("inf"))
            for start, before, end in record["segments"]:
                count = min(end - before, max(0, cutoff - start))
                if count:
                    spans.append(AudioSpan(start, pcm[before * 2 : (before + count) * 2], generation))
        return tuple(sorted(spans, key=lambda s: s.start_sample))

    def close(self):
        """Release the native sender's executor without starting playback workers."""
        self.sender._executor.shutdown(wait=False)
