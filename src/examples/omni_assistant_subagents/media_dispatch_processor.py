# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Dispatch LLM-selected media analysis after the acknowledgement turn."""

from __future__ import annotations

from typing import Protocol

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class MediaDispatchHandler(Protocol):
    """Dispatch hook implemented by the transport agent."""

    async def start_pending_media_analysis(self) -> None:
        """Start any queued media analysis after the assistant ack turn completes."""

    async def start_pending_thinking(self) -> None:
        """Start any queued deliberate-reasoning pass after the stall turn completes."""

    async def on_user_voice_turn_started(self) -> None:
        """Notify that the user resumed control through speech."""

    async def on_user_voice_turn_stopped(self) -> None:
        """Notify that the user turn reached end-of-utterance."""

    async def on_user_interrupted_assistant(self) -> None:
        """Notify that user speech interrupted assistant audio."""

    async def on_assistant_speaking_started(self) -> None:
        """Notify that assistant audio started."""

    async def on_assistant_speaking_stopped(self) -> None:
        """Notify that assistant audio stopped."""


class PostAckMediaDispatchProcessor(FrameProcessor):
    """Starts LLM-queued media work after the spoken bot acknowledgement finishes."""

    def __init__(self, *, handler: MediaDispatchHandler) -> None:
        """Initialize the post-ack dispatcher."""
        super().__init__()
        self._handler = handler
        self._assistant_speaking = False
        self._assistant_interrupted = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Pass frames through, then dispatch pending analysis after bot speech stops."""
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if isinstance(frame, (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)):
            if self._assistant_speaking:
                self._assistant_interrupted = True
                await self._handler.on_user_interrupted_assistant()
            await self._handler.on_user_voice_turn_started()
            return

        if isinstance(frame, UserStoppedSpeakingFrame):
            await self._handler.on_user_voice_turn_stopped()
            return

        if direction == FrameDirection.UPSTREAM and isinstance(frame, BotStartedSpeakingFrame):
            self._assistant_speaking = True
            self._assistant_interrupted = False
            await self._handler.on_assistant_speaking_started()
            return

        if direction == FrameDirection.UPSTREAM and isinstance(frame, BotStoppedSpeakingFrame):
            was_interrupted = self._assistant_interrupted
            self._assistant_speaking = False
            self._assistant_interrupted = False
            await self._handler.on_assistant_speaking_stopped()
            if was_interrupted:
                return
            try:
                await self._handler.start_pending_media_analysis()
            except Exception as exc:
                logger.warning(f"Failed to start pending media analysis: {exc}")
            try:
                await self._handler.start_pending_thinking()
            except Exception as exc:
                logger.warning(f"Failed to start pending thinking: {exc}")
