# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Pipecat processor that stands where the LLM service normally sits.

It receives the user aggregator's ``LLMContextFrame`` for each caller turn, runs the :class:`CascadeSession`
(talker decision, delegation, commentary) and speaks by pushing the same frames an LLM service does
(``LLMFullResponseStartFrame``, ``LLMTextFrame``, ``LLMFullResponseEndFrame``), so TTS and the assistant
aggregator need no changes.
"""

from __future__ import annotations

import asyncio

from loguru import logger
from pipecat.frames.frames import (
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
        """Bind the session (its ``say`` callback is :meth:`speak`) and start its delegation worker."""
        self.session = session
        session.start()

    async def interrupt(self) -> None:
        """Cut off whatever is being spoken (an instruction append changes direction mid-reply)."""
        await self.broadcast_interruption()

    async def speak(self, text: str) -> None:
        """Push ``text`` downstream as one LLM response, exactly as an LLM service would."""
        async with self._speak_lock:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(LLMTextFrame(text=text))
            await self.push_frame(LLMFullResponseEndFrame())

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        """Decide each caller turn; track whether the caller is speaking; pass everything else through."""
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
            if isinstance(frame, UserStartedSpeakingFrame):
                self.session.set_user_speaking(True)
            elif isinstance(frame, UserStoppedSpeakingFrame):
                self.session.set_user_speaking(False)
        await self.push_frame(frame, direction)

    async def cleanup(self) -> None:
        """Stop the delegation worker and release the models when the pipeline ends."""
        await super().cleanup()
        if self.session is not None:
            await self.session.close()
            self.session = None
