# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Frame processor adapter for Pipecat user mute strategies."""

from collections.abc import Iterable

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserMuteStartedFrame,
    UserMuteStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.turns.user_mute.base_user_mute_strategy import BaseUserMuteStrategy

_MUTABLE_USER_FRAMES = (
    InterruptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)


class UserMuteProcessor(FrameProcessor):
    """Suppress user input frames according to Pipecat user mute strategies."""

    def __init__(self, *, strategies: Iterable[BaseUserMuteStrategy], **kwargs):
        """Initialize the processor with one or more mute strategies."""
        super().__init__(**kwargs)
        self._strategies = list(strategies)
        self._user_is_muted = False

    async def setup(self, setup: FrameProcessorSetup) -> None:
        """Set up mute strategies with the pipeline configuration."""
        await super().setup(setup)
        for strategy in self._strategies:
            await strategy.setup(setup)

    async def cleanup(self) -> None:
        """Clean up all configured mute strategies."""
        for strategy in self._strategies:
            await strategy.cleanup()
        await super().cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Suppress mutable user frames while the active strategy says muted."""
        await super().process_frame(frame, direction)

        if isinstance(frame, (StartFrame, EndFrame, CancelFrame)):
            await self.push_frame(frame, direction)
            return

        should_mute_frame = self._user_is_muted and isinstance(frame, _MUTABLE_USER_FRAMES)
        should_mute_next = False
        for strategy in self._strategies:
            should_mute_next |= await strategy.process_frame(frame)

        if should_mute_next != self._user_is_muted:
            self._user_is_muted = should_mute_next
            logger.debug(f"{self}: user is now {'muted' if self._user_is_muted else 'unmuted'}")
            await self.broadcast_frame(UserMuteStartedFrame if self._user_is_muted else UserMuteStoppedFrame)

        if should_mute_frame:
            logger.trace(f"{frame.name} suppressed - user currently muted")
            return

        await self.push_frame(frame, direction)
