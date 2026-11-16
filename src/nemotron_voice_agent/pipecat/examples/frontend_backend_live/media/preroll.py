# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The reply-audio pre-roll in front of the audio output."""

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
    """Hold each reply's start until ``preroll_ms`` of audio is ready, so playback never runs dry between chunks."""

    def __init__(self, preroll_ms: int, **kwargs):
        """Hold ``preroll_ms`` of audio at the start of each reply (0 passes everything through)."""
        super().__init__(**kwargs)
        self.preroll_ms = preroll_ms
        self._held: list = []
        self._held_ms = 0.0
        self._holding = False

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        """Hold a reply's frames in order until enough audio is ready; a cut keeps only the uninterruptible ones."""
        await super().process_frame(frame, direction)
        if direction is not FrameDirection.DOWNSTREAM or not self.preroll_ms:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, (InterruptionFrame, CancelFrame)):
            kept = [held for held in self._held if not held.interruptible]
            self._drop()
            await self.push_frame(frame, direction)
            for held in kept:
                await self.push_frame(held, direction)
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
            self._held_ms += frame.num_frames / frame.sample_rate * 1000
        if self._held_ms >= self.preroll_ms or isinstance(frame, (TTSStoppedFrame, EndFrame)):
            await self._release()

    async def _release(self) -> None:
        held, self._held, self._held_ms, self._holding = self._held, [], 0.0, False
        for frame in held:
            await self.push_frame(frame, FrameDirection.DOWNSTREAM)

    def _drop(self) -> None:
        self._held, self._held_ms, self._holding = [], 0.0, False
