# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reuse the live output buffer and keep its audio/text queue order."""

import asyncio
import copy
from dataclasses import dataclass
from fractions import Fraction
from math import ceil

from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    MetricsFrame,
    OutputAudioRawFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_output import BOT_VAD_STOP_FALLBACK_SECS, BaseOutputTransport
from pipecat.transports.base_transport import TransportParams

from .models import RATE, GeneratedAudio, ResponseTrace


@dataclass(frozen=True)
class PackedFrame:
    """One dequeued output item, retaining provider availability and FIFO order."""

    frame: object
    ready_offset_ns: int
    delivery_offset_ns: int | None = None
    write_complete_offset_ns: int | None = None


@dataclass
class WebsocketSinkClock:
    """Native WS write pacing state, separate from client PCM presentation.

    Pipecat sends a packet before sleeping. The first write sleeps zero, so a
    second queued packet can also be sent immediately. Text and stop frames
    follow write completion, which can precede buffered client audio's end.
    """

    next_send_ns: Fraction = Fraction(0)

    def write_complete(self, at, interval):
        """Apply FastAPIWebsocketOutputTransport._write_audio_sleep exactly."""
        if self.next_send_ns <= at:
            self.next_send_ns = at + interval
            return at
        completed = self.next_send_ns
        self.next_send_ns += interval
        return completed

    def interrupt(self):
        """The native WS transport resets pacing only on interruption."""
        self.next_send_ns = Fraction(0)


async def pack_frames(
    trace: ResponseTrace, chunks_10ms: int, *, sample_rate=RATE, sink_clock=None, start_ns=0
) -> tuple[PackedFrame, ...]:
    """Run actual Pipecat buffering without starting socket or pacing workers.

    Native-rate chunks use the online transport's integer 10 ms byte calculation.
    Provider arrivals and virtual FIFO playback are interleaved so a starvation
    stop resets partial audio before later chunks arrive. TTS stop flushes the
    partial audio before queued text and control frames as the live sender does.
    """
    if chunks_10ms <= 0 or sample_rate <= 0:
        raise ValueError("Output chunk count and sample rate must be positive")

    class _Sink:
        async def push_frame(self, *args, **kwargs):
            # Speaking notifications are reproduced by the session scheduler.
            pass

    sender = BaseOutputTransport.MediaSender(
        _Sink(),
        destination=None,
        sample_rate=sample_rate,
        audio_chunk_size=(sample_rate // 100) * 2 * chunks_10ms,
        params=TransportParams(audio_out_enabled=True),
    )
    sink_clock = sink_clock if sink_clock is not None else WebsocketSinkClock()
    interval = Fraction(sender._audio_chunk_size * 1_000_000_000, sample_rate * 2)
    producer_at = 0
    queued_at = {}

    class _Queue(asyncio.Queue):
        async def put(self, frame):
            queued_at[id(frame)] = producer_at
            await super().put(frame)

    sender._audio_queue = _Queue()
    result = []
    consumer_at = Fraction(start_ns)
    idle_at = None

    async def drain(limit, *, inclusive):
        """Run the FIFO consumer up to the next provider arrival.

        The live sender can reset its input buffer while waiting for another
        output item. Consequently future provider arrivals must not be packed
        before that timeout has run. Only the wait is replaced with an explicit
        deadline: buffer clearing and resampler reset remain MediaSender's.
        """
        nonlocal consumer_at, idle_at
        while True:
            if not sender._audio_queue.empty():
                if consumer_at > limit or (consumer_at == limit and not inclusive):
                    return
                frame = sender._audio_queue.get_nowait()
                # Keep the enqueue offset separate from virtual playback.
                # The session uses cumulative canonical sample pointers to
                # serialize audio without native-to-canonical rounding gaps.
                delivered = consumer_at
                if isinstance(frame, OutputAudioRawFrame):
                    if isinstance(frame, TTSAudioRawFrame) or legacy:
                        sender._tts_audio_received = True
                        sender._bot_speaking = True
                    consumer_at = sink_clock.write_complete(consumer_at, interval)
                elif isinstance(frame, TTSStoppedFrame) and sender._tts_audio_received:
                    await sender._bot_stopped_speaking()
                result.append(
                    PackedFrame(
                        frame,
                        queued_at.pop(id(frame)),
                        ceil(delivered - start_ns),
                        ceil(consumer_at - start_ns),
                    )
                )
                idle_at = (
                    consumer_at + round(BOT_VAD_STOP_FALLBACK_SECS * 1_000_000_000) if sender._bot_speaking else None
                )
            elif idle_at is not None and (idle_at < limit or (inclusive and idle_at == limit)):
                await sender._bot_stopped_speaking()
                idle_at = None
            else:
                return

    frames = getattr(trace, "frames", None)
    legacy = frames is None
    if frames is None:
        # Scripted/legacy providers have no live frame stream. Production uses
        # TTSResponseTrace; this compatibility branch never infers word timing.
        frames = [PackedFrame(OutputAudioRawFrame(chunk.pcm, RATE, 1), chunk.ready_offset_ns) for chunk in trace.audio]
        frames.append(PackedFrame(TTSStoppedFrame(), trace.compute_ns))
        if trace.text:
            frames.append(PackedFrame(TTSTextFrame(trace.text, "sentence"), trace.compute_ns))
        frames.append(PackedFrame(LLMFullResponseEndFrame(), trace.compute_ns))
    previous = 0
    try:
        for item in frames:
            if getattr(item, "direction", FrameDirection.DOWNSTREAM) != FrameDirection.DOWNSTREAM:
                continue
            if not previous <= item.ready_offset_ns <= trace.compute_ns:
                raise ValueError("Provider frames must have monotonic causal availability")
            previous = item.ready_offset_ns
            producer_at = item.ready_offset_ns
            # An arrival tied to a queue timeout supplies the next item first,
            # matching the session's source-events-before-timers tie policy.
            arrived = start_ns + item.ready_offset_ns
            await drain(arrived, inclusive=False)
            if isinstance(item.frame, MetricsFrame):
                # BaseOutputTransport forwards these SystemFrames directly;
                # they cannot renew MediaSender's queue wait. The provider
                # trace has already retained their native timing measurements.
                continue
            if sender._audio_queue.empty():
                consumer_at = max(consumer_at, arrived)
            frame = item.frame
            if isinstance(frame, OutputAudioRawFrame):
                if frame.num_channels != 1 or len(frame.audio) % 2:
                    raise ValueError("Provider audio must be mono PCM16")
                await sender.handle_audio_frame(frame)
            elif isinstance(frame, TTSStoppedFrame):
                await sender.handle_tts_stopped(frame)
            else:
                await sender.handle_sync_frame(frame)
            await drain(arrived, inclusive=True)
        await drain(start_ns + trace.compute_ns, inclusive=False)
        producer_at = trace.compute_ns
        if sender._audio_queue.empty():
            consumer_at = max(consumer_at, start_ns + trace.compute_ns)
        await sender._enqueue_flushed_audio_buffer()
        await drain(float("inf"), inclusive=True)
    finally:
        sender._executor.shutdown(wait=False)
    return tuple(result)


def canonicalize(frames: tuple[PackedFrame, ...]) -> tuple[PackedFrame, ...]:
    """Convert packed native PCM once, splitting at cumulative sample pointers.

    Resampling the complete finite audio avoids adding a resampler's internal
    buffering latency to provider readiness or rounding each chunk independently.
    Control frames retain their exact position relative to preceding PCM bytes.
    """
    import numpy as np
    import soxr

    audio = [item.frame for item in frames if isinstance(item.frame, OutputAudioRawFrame)]
    if not audio:
        return frames
    rate = audio[0].sample_rate
    if any(frame.sample_rate != rate for frame in audio):
        raise ValueError("Packed output sample rate changed")
    if rate == RATE:
        return frames
    pcm = np.frombuffer(b"".join(frame.audio for frame in audio), dtype=np.int16)
    converted = soxr.resample(pcm, rate, RATE, quality="VHQ").tobytes()
    result = []
    native_end = canonical_end = 0
    for item in frames:
        if isinstance(item.frame, OutputAudioRawFrame):
            native_end += len(item.frame.audio) // 2
            end = round(native_end * RATE / rate)
            frame = copy.copy(item.frame)
            frame.audio = converted[canonical_end * 2 : end * 2]
            frame.sample_rate = RATE
            result.append(
                PackedFrame(frame, item.ready_offset_ns, item.delivery_offset_ns, item.write_complete_offset_ns)
            )
            canonical_end = end
        else:
            result.append(item)
    if canonical_end * 2 != len(converted):
        raise ValueError("Resampled output does not cover packed sample pointers")
    return tuple(result)


async def pack_output(trace: ResponseTrace, chunks_10ms: int) -> tuple[GeneratedAudio, ...]:
    """Compatibility view of the same live chunker's canonical PCM output."""
    return tuple(
        GeneratedAudio(item.frame.audio, item.ready_offset_ns)
        for item in await pack_frames(trace, chunks_10ms)
        if isinstance(item.frame, OutputAudioRawFrame)
    )
