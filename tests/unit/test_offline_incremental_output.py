# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Incremental output must preserve the native finite-trace reference contract."""

import asyncio
from functools import wraps
from types import SimpleNamespace

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    LLMFullResponseEndFrame,
    OutputAudioRawFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from examples.generic.offline.output_stream import IncrementalOutput
from examples.generic.offline.playback import canonicalize, pack_frames


def _async_case(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return run


@_async_case
@pytest.mark.parametrize("rate", [16000, 22050])
async def test_incremental_dequeues_match_native_reference(rate):
    """Chunk bytes, FIFO and virtual write completions match native packing."""
    audio = b"\x10\x00" * round(rate * 1.1)
    arrivals = [
        (0, TTSAudioRawFrame(audio[: len(audio) // 2 // 2 * 2], rate, 1)),
        (100_000_000, TTSAudioRawFrame(audio[len(audio) // 2 // 2 * 2 :], rate, 1)),
        (150_000_000, TTSStoppedFrame()),
        (150_000_000, TTSTextFrame("Retain FIFO text.", "sentence")),
        (150_000_000, LLMFullResponseEndFrame()),
    ]
    trace = SimpleNamespace(
        frames=[SimpleNamespace(frame=f, ready_offset_ns=t) for t, f in arrivals], compute_ns=150_000_000
    )
    reference = canonicalize(await pack_frames(trace, 40, sample_rate=rate))
    output = IncrementalOutput(sample_rate=rate, chunks_10ms=40)
    delivered = []
    notifications = []

    async def drain(until, inclusive):
        while output.next_ns is not None and (output.next_ns <= until if inclusive else output.next_ns < until):
            at = output.next_ns
            item = await output.pop(at)
            notifications.extend(item.notifications)
            if item.frame is not None:
                delivered.append((at, item))

    try:
        for at, frame in arrivals:
            await drain(at, False)
            await output.receive("r1", at, frame)
            await drain(at, True)
        await drain(float("inf"), True)
        assert [type(d.frame) for _, d in delivered] == [type(item.frame) for item in reference]
        assert [at for at, _ in delivered] == pytest.approx([f.delivery_offset_ns for f in reference], abs=2)
        assert [d.complete_ns for _, d in delivered] == pytest.approx(
            [f.write_complete_offset_ns for f in reference], abs=2
        )
        assert b"".join(span.pcm for span in output.spans()) == b"".join(
            item.frame.audio for item in reference if isinstance(item.frame, OutputAudioRawFrame)
        )
        downstream = [f for f, direction in notifications if direction == FrameDirection.DOWNSTREAM]
        assert sum(isinstance(f, BotStartedSpeakingFrame) for f in downstream) == 1
        assert sum(isinstance(f, BotStoppedSpeakingFrame) for f in downstream) == 1
    finally:
        output.close()


@_async_case
async def test_interrupt_discards_partial_and_queued_audio_without_new_generation_leak():
    """Cancel sent and unsent audio without contaminating the next generation."""
    output = IncrementalOutput(sample_rate=16000, chunks_10ms=40)
    try:
        await output.receive("old", 0, TTSAudioRawFrame(b"\x01\x00" * 20000, 16000, 1))
        await output.pop(0)
        await output.pop(0)
        await output.interrupt("old", 100_000_000)
        assert not output.pending
        assert output.sender._buffered_audio_bytes == 0
        await output.receive("new", 200_000_000, TTSAudioRawFrame(b"\x02\x00" * 6400, 16000, 1))
        await output.pop(200_000_000)
        spans = output.spans()
        old = [s for s in spans if s.response_id == "old"]
        new = [s for s in spans if s.response_id == "new"]
        assert all(s.end_sample <= 1600 for s in old)
        assert new[0].start_sample == 3200
        assert new[0].pcm == b"\x02\x00" * 6400
    finally:
        output.close()


@_async_case
async def test_starvation_resets_native_partial_buffer_before_later_audio():
    """A native empty-queue stop discards the pending partial chunk."""
    output = IncrementalOutput(sample_rate=16000, chunks_10ms=40)
    try:
        await output.receive("r", 0, TTSAudioRawFrame(b"\x01\x00" * 7000, 16000, 1))
        await output.pop(0)
        assert output.sender._buffered_audio_bytes == 1200
        stopped = await output.pop(3_000_000_000)
        assert stopped.frame is None
        assert output.sender._buffered_audio_bytes == 0
        await output.receive("r", 4_000_000_000, TTSAudioRawFrame(b"\x02\x00" * 6400, 16000, 1))
        await output.pop(4_000_000_000)
        assert output.spans()[-1].pcm == b"\x02\x00" * 6400
    finally:
        output.close()
