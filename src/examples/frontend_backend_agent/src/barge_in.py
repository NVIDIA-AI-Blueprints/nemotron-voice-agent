# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Session-local barge-in state for the Frontend/Backend Agent."""

from __future__ import annotations

from loguru import logger
from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame, UserStartedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class BargeInState:
    """Remember whether the current user turn interrupted active bot speech."""

    def __init__(self) -> None:
        """Initialize idle speaking and interruption state."""
        self._bot_speaking = False
        self._interrupted_speech = False

    def observe(self, frame: Frame) -> None:
        """Update speaking state from pipeline lifecycle frames."""
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._interrupted_speech = self._bot_speaking
            if self._interrupted_speech:
                logger.info("Barge-in detected while bot audio was active")

    def consume_interrupted_speech(self) -> bool:
        """Return and clear the interruption marker for the current user turn."""
        interrupted = self._interrupted_speech
        self._interrupted_speech = False
        return interrupted


class BargeInTracker(FrameProcessor):
    """Observe speaking lifecycle frames without changing pipeline traffic."""

    def __init__(self, state: BargeInState) -> None:
        """Bind the transparent processor to session-local barge-in state."""
        super().__init__()
        self._state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Track barge-in state and forward every frame unchanged."""
        await super().process_frame(frame, direction)
        self._state.observe(frame)
        await self.push_frame(frame, direction)
