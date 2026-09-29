# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-2-Clause

"""Gate that stops pre-tool-call text from being spoken.

When the LLM decides to call a tool, the *content* it emits in that same
completion is never the real answer — the answer comes from the follow-up
completion after the tool result. With reasoning enabled, Nemotron occasionally
emits its chain-of-thought (or a "let me check…" stall) as that content, which
would otherwise be sent to TTS and spoken aloud — a prompt/CoT leak. The system
prompt already forbids announcing or stalling, so dropping a tool-call
completion's text is always correct.

Placed between the LLM and TTS. It buffers each LLM response's text and:
  * drops it entirely if that response also produces a tool call
    (``FunctionCallsStartedFrame`` / ``FunctionCallInProgressFrame`` arrive
    inside the response, before ``LLMFullResponseEndFrame`` — verified against
    pipecat's OpenAI service, which calls ``run_function_calls`` before pushing
    the end frame);
  * flushes it to TTS at ``LLMFullResponseEndFrame`` for a normal (no-tool)
    response — the post-tool-result answer completion streams through here.

Answers are short (one sentence), so buffering-until-end adds only the tiny
completion-generation latency while guaranteeing no leaked reasoning is spoken.
"""

from __future__ import annotations

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallsStartedFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class ToolCallSpeechGate(FrameProcessor):
    """Suppress text emitted in the same LLM response as a tool call."""

    def __init__(self) -> None:
        """Initialize per-response buffering and tool-call state."""
        super().__init__()
        self._buffer: list[LLMTextFrame] = []
        self._in_response = False
        self._tool_call = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Buffer downstream LLM text until the response is known to be tool-free."""
        await super().process_frame(frame, direction)

        # Only gate LLM->TTS (downstream) text; pass everything else straight through.
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            self._in_response = True
            self._tool_call = False
            self._buffer = []
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (FunctionCallsStartedFrame, FunctionCallInProgressFrame)):
            if self._buffer:
                dropped = sum(len(f.text) for f in self._buffer)
                logger.debug(f"ToolCallSpeechGate: dropped {dropped} chars of pre-tool-call text (not spoken)")
            self._tool_call = True
            self._buffer = []
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMTextFrame) and self._in_response:
            if self._tool_call:
                return  # text from a tool-call response is never spoken
            self._buffer.append(frame)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            if not self._tool_call:
                for buffered in self._buffer:
                    await self.push_frame(buffered, direction)
            self._buffer = []
            self._in_response = False
            self._tool_call = False
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)
