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
        warning_completion_timeout_s: float = 45.0,
        on_warning: ActivityCallback,
        on_disconnect: DisconnectCallback,
    ) -> None:
        """Initialize inactivity thresholds, warning watchdogs, and callbacks."""
        super().__init__(name="activity-check")
        self._intervals = (
            first_warning_s if first_warning_s is not None else activity_check_interval_s,
            second_warning_s,
        )
        if any(interval <= 0 for interval in (*self._intervals, warning_completion_timeout_s)):
            raise ValueError("activity-check intervals must be greater than zero")
        self._on_warning = on_warning
        self._on_disconnect = on_disconnect
        self._warning_completion_timeout_s = warning_completion_timeout_s
        self._stage = 0
        self._disconnect_after_speech = False
        self._retired_warning_completions = 0
        self._timer: asyncio.Task[None] | None = None
        self._warning_completion_timer: asyncio.Task[None] | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Track speech-boundary frames and forward every pipeline frame."""
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
            self._handle_bot_stopped_speaking()
        elif isinstance(frame, (CancelFrame, EndFrame)):
            self.reset()

        await self.push_frame(frame, direction)

    def reset(self) -> None:
        """Cancel a pending check and return to ordinary conversation."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._cancel_warning_completion_timer()
        self._stage = 0
        self._disconnect_after_speech = False
        self._retired_warning_completions = 0

    def _handle_bot_stopped_speaking(self) -> None:
        """Handle a bot-speech completion for the currently active warning."""
        if self._retired_warning_completions:
            self._retired_warning_completions -= 1
            logger.info(
                "Ignoring delayed completion from retired activity warning (remaining={})",
                self._retired_warning_completions,
            )
            return

        self._cancel_warning_completion_timer()
        if self._disconnect_after_speech:
            self._disconnect_after_speech = False
            logger.info("Final activity check finished; disconnecting session")
            self.create_task(self._on_disconnect(), "disconnect")
        else:
            self._arm_timer()

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
            await self._emit_warning(self._stage + 1)
        except asyncio.CancelledError:
            logger.debug("Activity check countdown cancelled")
            return

    async def _emit_warning(self, stage: int) -> None:
        self._stage = stage
        self._disconnect_after_speech = stage == len(self._intervals)
        logger.info("Activity check fired: stage={}", stage)
        self._warning_completion_timer = self.create_task(
            self._wait_for_warning_completion(stage), "wait-for-warning-completion"
        )
        try:
            await self._on_warning(stage)
        except Exception:
            logger.exception("Activity check warning generation failed: stage={}", stage)

    async def _wait_for_warning_completion(self, stage: int) -> None:
        try:
            await asyncio.sleep(self._warning_completion_timeout_s)
            if self._stage != stage:
                return

            self._warning_completion_timer = None
            logger.warning(
                "Activity check warning did not finish before timeout: stage={}, timeout_s={}",
                stage,
                self._warning_completion_timeout_s,
            )
            if stage == len(self._intervals):
                self._disconnect_after_speech = False
                await self._on_disconnect()
            else:
                self._retired_warning_completions += 1
                await self._emit_warning(stage + 1)
        except asyncio.CancelledError:
            logger.debug("Activity check warning-completion watchdog cancelled")

    def _cancel_warning_completion_timer(self) -> None:
        if self._warning_completion_timer is not None:
            self._warning_completion_timer.cancel()
            self._warning_completion_timer = None

    async def cleanup(self) -> None:
        """Cancel any pending inactivity timer during pipeline teardown."""
        self.reset()
        await super().cleanup()
