# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-turn tool-call reminder for the generic cascaded pipeline.

With reasoning off, Nemotron 3.5 Lightning often answers from memory instead of
calling a matching tool. The reminder points at the "# Tools" section the chat
template renders, so it stays valid for any ``tools_available`` set.
"""

import dataclasses

from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

TOOL_CALL_REMINDER = (
    "Reminder: check the functions listed in the # Tools section. If any function's description matches "
    "this request and its result is not in the conversation yet, call that function; do not answer from "
    "memory or guess values a function returns. If no function matches, answer directly without tools."
)


class ToolCallReminderProcessor(FrameProcessor):
    """Append ``TOOL_CALL_REMINDER`` to the user turn of each LLM request.

    Sits between the user aggregator and the LLM and forwards a copy of the
    context, so the stored chat history never contains the reminder.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Forward frames, adding the reminder to outbound context frames."""
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            frame = _with_reminder(frame)
        await self.push_frame(frame, direction)


def _with_reminder(frame: LLMContextFrame) -> LLMContextFrame:
    messages = list(frame.context.get_messages())
    last = messages[-1] if messages else None
    if not (isinstance(last, dict) and last.get("role") == "user" and isinstance(last.get("content"), str)):
        return frame
    messages[-1] = {**last, "content": f"{last['content']}\n\n{TOOL_CALL_REMINDER}"}
    context = LLMContext(messages, tools=frame.context.tools, tool_choice=frame.context.tool_choice)
    return dataclasses.replace(frame, context=context)
