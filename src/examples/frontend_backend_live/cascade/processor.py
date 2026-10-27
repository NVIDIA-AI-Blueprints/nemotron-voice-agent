# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Pipecat processor that takes the LLM service's place and binds the cascade to the pipeline.

Pipecat's user aggregator writes the turn into its context and queues the context frame before it broadcasts the stop,
so the turn is heard whichever of the two this processor handles first. Lines go out as the frames an LLM service
pushes; no assistant aggregator follows, because the cascade records each line in the talker's memory itself.
"""

from __future__ import annotations

import asyncio

from loguru import logger
from pipecat.frames.frames import (
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from examples.frontend_backend_live.cascade.session import CascadeSession


class LiveTalkerProcessor(FrameProcessor):
    """Runs one call's :class:`CascadeSession` inside a Pipecat pipeline."""

    def __init__(self, **kwargs):
        """Create the processor; call :meth:`attach` with the session before the pipeline starts."""
        super().__init__(**kwargs)
        self.session: CascadeSession | None = None
        # A lead-in, an answer and a commentary can each speak at once; their frames must not interleave.
        self._speak_lock = asyncio.Lock()

    def attach(self, session: CascadeSession) -> None:
        """Bind the session (its ``say`` callback is :meth:`speak`); the host that built it starts and closes it."""
        self.session = session

    async def interrupt(self) -> None:
        """Cut off the line in progress (an instruction): an interruption pushed downstream, not broadcast.

        A broadcast would also empty this processor's own queue, dropping a caller turn waiting in it.
        """
        if self.session is not None:
            self.session.interrupted()
        await self.push_frame(InterruptionFrame())

    async def speak(self, text: str) -> None:
        """Push ``text`` downstream as one LLM response, exactly as an LLM service would."""
        async with self._speak_lock:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(LLMTextFrame(text=text))
            await self.push_frame(LLMFullResponseEndFrame())

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        """Hand each caller turn's start, stop and decision to the session; pass everything else through."""
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            if getattr(frame, "speculation", False) or self.session is None:
                # A provisional context from early turn detection: the final frame will follow.
                return
            try:
                await self.session.on_turn()
            except Exception as exc:
                logger.exception(f"Live turn failed: {exc}")
                await self.push_error(error_msg=f"Live turn failed: {exc}", exception=exc)
            return
        if self.session is not None:
            if isinstance(frame, InterruptionFrame) and direction == FrameDirection.DOWNSTREAM:
                self.session.interrupted()
            elif isinstance(frame, UserStartedSpeakingFrame):
                self.session.caller_started()
            elif isinstance(frame, UserStoppedSpeakingFrame):
                self.session.caller_stopped()
        await self.push_frame(frame, direction)
