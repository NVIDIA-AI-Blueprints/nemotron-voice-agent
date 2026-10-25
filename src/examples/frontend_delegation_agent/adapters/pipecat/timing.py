# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Pipecat head's simple timing estimate: audio now, speech onset, and heard audio.

* **audio now**: input audio received, counted in pipeline order by :class:`InputCursor`
  (right after the VAD, so a VAD event arrives after exactly the audio it analyzed).
* **onset / offset**: the cursor when Pipecat's VAD confirms a start or stop, minus the
  VAD's ``start_secs`` / ``stop_secs`` (the confirmation delay), so ``audio_start_ms`` points
  at the speech itself, as the τ³ harness expects. A turn Pipecat starts from a transcript
  (no VAD onset) uses the cursor at turn start.
* **heard**: Pipecat's WebSocket output writes audio at twice real time, so the written
  amount is not what the client played. :class:`PlayoutEstimator` advances the playout
  cursor at real-time speed from the first written chunk, never past what was written,
  and stops at the written amount on an interruption.

This is an estimate on wall-clock time. A client that pauses its playback (τ²'s tick
loop) hears less than estimated; the README documents the limitation.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class InputCursor(FrameProcessor):
    """Counts input audio in pipeline order and records VAD onsets and offsets."""

    def __init__(self, **kwargs: object) -> None:
        """Start at audio time zero."""
        super().__init__(**kwargs)
        self.now_ms = 0.0
        self.onset_ms: float | None = None
        self.offset_ms: float | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Advance on input audio; note the VAD events that follow it downstream."""
        await super().process_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, InputAudioRawFrame) and frame.sample_rate > 0:
            samples = len(frame.audio) // (2 * max(1, frame.num_channels))
            self.now_ms += samples * 1000.0 / frame.sample_rate
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self.onset_ms = max(0.0, self.now_ms - frame.start_secs * 1000.0)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self.offset_ms = max(0.0, self.now_ms - frame.stop_secs * 1000.0)
        await self.push_frame(frame, direction)

    def take_onset(self) -> tuple[float, str]:
        """The onset of the turn that just started, and where it came from (``vad`` or ``transcription``)."""
        onset, self.onset_ms = self.onset_ms, None
        if onset is None:
            return self.now_ms, "transcription"
        return onset, "vad"

    def take_offset(self) -> float:
        """The offset of the turn that just stopped (the cursor when there was no VAD stop)."""
        offset, self.offset_ms = self.offset_ms, None
        return self.now_ms if offset is None else offset


class PlayoutEstimator:
    """Feeds ``on_audio_advance`` with a real-time estimate of the audio the client has played."""

    def __init__(
        self,
        advance: Callable[[float], None],
        *,
        tick_s: float = 0.02,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bind the turn manager's ``on_audio_advance``."""
        self._advance = advance
        self._tick_s = tick_s
        self._monotonic = monotonic
        self.written_ms = 0.0
        self.heard_ms = 0.0
        self._last: float | None = None
        self._task: asyncio.Task[None] | None = None

    def on_written(self, chunk_ms: float) -> None:
        """The transport wrote ``chunk_ms`` of audio to the client."""
        self.written_ms += chunk_ms
        if self._task is None or self._task.done():
            self._last = self._monotonic()
            self._task = asyncio.get_running_loop().create_task(self._run(), name="fda-playout")

    def on_interruption(self) -> None:
        """The client drops what it has not played: nothing more will be heard."""
        self.written_ms = self.heard_ms

    def step(self) -> float:
        """Advance by the wall time since the last step, capped at what was written; the step in ms."""
        now = self._monotonic()
        last = self._last if self._last is not None else now
        self._last = now
        step = min(max(0.0, (now - last) * 1000.0), max(0.0, self.written_ms - self.heard_ms))
        if step > 0:
            self.heard_ms += step
            self._advance(step)
        return step

    async def _run(self) -> None:
        while self.heard_ms < self.written_ms:
            await asyncio.sleep(self._tick_s)
            self.step()

    async def aclose(self) -> None:
        """Stop the ticker."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
