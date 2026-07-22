# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-2-Clause

"""Proactive inactivity checks for cascaded voice pipelines."""

import asyncio
from collections.abc import Awaitable, Callable

from loguru import logger
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

ActivityCallback = Callable[[int], Awaitable[None]]
DisconnectCallback = Callable[[], Awaitable[None]]


class ActivityCheckProcessor(FrameProcessor):
    """Replace a silent hard timeout with two LLM/TTS activity checks.

    Timers begin when the bot finishes speaking. A VAD ``UserStartedSpeakingFrame``
    cancels the current countdown and returns the processor to normal operation.
    The callback is responsible for queueing an LLM run; the next timer is armed
    only after the resulting TTS audio has finished. The second activity check
    is the closing statement; the session disconnects when it finishes playing.
    """

    def __init__(
        self,
        *,
        activity_check_interval_s: float = 600.0,
        first_warning_s: float | None = None,
        second_warning_s: float = 30.0,
        on_warning: ActivityCallback,
        on_disconnect: DisconnectCallback,
    ) -> None:
        """Initialize two inactivity thresholds and their callbacks."""
        super().__init__(name="activity-check")
        self._intervals = (
            first_warning_s if first_warning_s is not None else activity_check_interval_s,
            second_warning_s,
        )
        if any(interval <= 0 for interval in self._intervals):
            raise ValueError("activity-check intervals must be greater than zero")
        self._on_warning = on_warning
        self._on_disconnect = on_disconnect
        self._stage = 0
        self._disconnect_after_speech = False
        self._timer: asyncio.Task[None] | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Track speech-boundary frames and forward every pipeline frame."""
        # Let Pipecat initialize this processor on StartFrame (and clean it up
        # on EndFrame/CancelFrame) before forwarding the frame ourselves.
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            if self._timer is not None or self._stage:
                logger.info(
                    "Activity check reset by user speech (stage={}, timer_pending={})",
                    self._stage,
                    self._timer is not None,
                )
            self.reset()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self._disconnect_after_speech:
                self._disconnect_after_speech = False
                logger.info("Final activity check finished; disconnecting session")
                self.create_task(self._on_disconnect(), "disconnect")
            else:
                self._arm_timer()
        elif isinstance(frame, (CancelFrame, EndFrame)):
            self.reset()

        await self.push_frame(frame, direction)

    def reset(self) -> None:
        """Cancel a pending check and return to ordinary conversation."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._stage = 0
        self._disconnect_after_speech = False

    def _arm_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        if self._stage >= len(self._intervals) or self._disconnect_after_speech:
            return
        delay = self._intervals[self._stage]
        logger.info("Activity check armed: next_stage={}, delay_s={}", self._stage + 1, delay)
        self._timer = self.create_task(self._wait_for_check(delay), "wait-for-check")

    async def _wait_for_check(self, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            self._timer = None
            self._stage += 1
            if self._stage == len(self._intervals):
                self._disconnect_after_speech = True
            logger.info("Activity check fired: stage={}", self._stage)
            await self._on_warning(self._stage)
        except asyncio.CancelledError:
            logger.debug("Activity check countdown cancelled")
            return

    async def cleanup(self) -> None:
        """Cancel any pending inactivity timer during pipeline teardown."""
        self.reset()
        await super().cleanup()
