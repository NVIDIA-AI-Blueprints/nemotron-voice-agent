# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The reply-audio pre-roll in front of a live session's audio output."""

from __future__ import annotations

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    SystemFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class SpeechPreroll(FrameProcessor):
    """Hold the start of each reply until ``preroll_ms`` of audio is ready, then play it on.

    Synthesis runs several times faster than playback but delivers audio in uneven chunks; without a head start,
    playback runs dry between chunks and the caller hears a stutter. Every downstream frame of the reply is held in
    order (so word timings stay with their audio) until enough audio is buffered or the reply ends. An interruption
    drops what is held. System frames pass at once.
    """

    def __init__(self, preroll_ms: int, **kwargs):
        """Hold ``preroll_ms`` of audio at the start of each reply (0 passes everything through)."""
        super().__init__(**kwargs)
        self.preroll_ms = preroll_ms
        self._held: list = []
        self._held_ms = 0.0
        self._holding = False

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        """Hold a reply's first frames, release them once enough audio is ready, and pass everything else."""
        await super().process_frame(frame, direction)
        if direction is not FrameDirection.DOWNSTREAM or not self.preroll_ms:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, (InterruptionFrame, CancelFrame)):
            self._drop()
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, TTSStartedFrame) and not self._holding:
            self._holding = True
        if not self._holding:
            await self.push_frame(frame, direction)
            return
        self._held.append(frame)
        if isinstance(frame, OutputAudioRawFrame):
            self._held_ms += len(frame.audio) / 2 / frame.num_channels / frame.sample_rate * 1000
        if self._held_ms >= self.preroll_ms or isinstance(frame, (TTSStoppedFrame, EndFrame)):
            await self._release()

    async def _release(self) -> None:
        held, self._held, self._held_ms, self._holding = self._held, [], 0.0, False
        for frame in held:
            await self.push_frame(frame, FrameDirection.DOWNSTREAM)

    def _drop(self) -> None:
        self._held, self._held_ms, self._holding = [], 0.0, False
