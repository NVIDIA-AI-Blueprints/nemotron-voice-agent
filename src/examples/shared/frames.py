# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Ordered internal frames shared by voice-agent pipeline implementations."""

from dataclasses import dataclass
from typing import Literal, cast

from pipecat.frames.frames import ControlFrame, UninterruptibleFrame

USER_TRANSCRIPT_TURN_FRAME_ID_METADATA = "nvidia.user_transcript_turn_frame_id"

LLMProviderFinishReason = Literal[
    "stop",
    "length",
    "tool_calls",
    "content_filter",
    "function_call",
]

LLM_PROVIDER_FINISH_REASONS = frozenset(
    {
        "stop",
        "length",
        "tool_calls",
        "content_filter",
        "function_call",
    }
)


def require_llm_provider_finish_reason(value: object) -> LLMProviderFinishReason:
    """Return one supported provider terminal or reject the boundary value."""
    if not isinstance(value, str) or value not in LLM_PROVIDER_FINISH_REASONS:
        raise ValueError(f"Unsupported LLM provider finish reason: {value!r}")
    return cast(LLMProviderFinishReason, value)


@dataclass
class LLMProviderCompletionReasonFrame(ControlFrame):
    """Carry the provider's typed terminal reason through the pipeline.

    Pipecat 1.7 consumes ``ChatCompletionChunk.finish_reason`` internally but
    does not expose it on ``LLMFullResponseEndFrame``. This ordered frame is
    emitted after the provider's terminal chunk has been consumed and before
    Pipecat emits ``LLMFullResponseEndFrame``. Downstream protocol adapters can
    therefore distinguish a complete response from a token-limit terminal
    without inspecting generated text.
    """

    finish_reason: LLMProviderFinishReason


@dataclass
class UserTranscriptProducerEndedFrame(ControlFrame, UninterruptibleFrame):
    """Mark that a fused audio turn cannot emit another user transcript."""

    status: Literal["completed", "failed", "cancelled", "skipped", "overflowed"]
    turn_frame_id: int | None = None
