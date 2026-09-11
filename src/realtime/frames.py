# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Internal Pipecat frames used by the Realtime transport boundary."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from openai import NOT_GIVEN
from pipecat.frames.frames import (
    ControlFrame,
    DataFrame,
    InputAudioRawFrame,
    LLMContextFrame,
    LLMFullResponseStartFrame,
    UninterruptibleFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext

if TYPE_CHECKING:
    from realtime.response_config import RealtimeResponseInput


class RealtimeResponseOrigin(Enum):
    """Identify why a Realtime-owned provider response is starting."""

    AUTOMATIC_USER_TURN = auto()
    INTERNAL_TOOL_CONTINUATION = auto()
    SERVICE_INITIATED = auto()


class RealtimeResponseLLMContext(LLMContext):
    """An isolated response context carrying provider request overrides."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Any = NOT_GIVEN,
        tool_choice: Any = NOT_GIVEN,
        max_output_tokens: int | str | None = None,
        parallel_tool_calls: bool | None = None,
        truncation: str | dict[str, Any] | None = None,
        preserve_prompt_messages: int = 0,
        response_id: str | None = None,
        run_owner_id: str | None = None,
        activation_generation: int | None = None,
    ) -> None:
        """Build a normal Pipecat context with Realtime-only request metadata."""
        if isinstance(preserve_prompt_messages, bool) or not isinstance(preserve_prompt_messages, int):
            raise TypeError("preserve_prompt_messages must be an integer")
        if preserve_prompt_messages < 0 or preserve_prompt_messages > len(messages):
            raise ValueError("preserve_prompt_messages must identify a valid message prefix")
        super().__init__(messages, tools=tools, tool_choice=tool_choice)
        self.max_output_tokens = max_output_tokens
        self.parallel_tool_calls = parallel_tool_calls
        self.truncation = copy.deepcopy(truncation)
        self.preserve_prompt_messages = preserve_prompt_messages
        self.response_id = response_id
        self.run_owner_id = run_owner_id
        self.activation_generation = activation_generation


@dataclass
class RealtimeOwnedLLMFullResponseStartFrame(LLMFullResponseStartFrame):
    """Start a provider run bound to an exact Realtime response or reservation."""

    response_id: str = ""
    run_owner_id: str = ""
    activation_generation: int | None = None


@dataclass
class RealtimeConversationAppendFrame(ControlFrame, UninterruptibleFrame):
    """Apply and acknowledge a client conversation item at its context owner."""

    item_id: str
    context_message: dict[str, Any]
    events: tuple[dict[str, Any], ...]


@dataclass
class RealtimeIdleTimeoutFrame(ControlFrame, UninterruptibleFrame):
    """Commit a server-VAD idle turn at the canonical context owner."""

    audio_start_ms: int
    audio_end_ms: int
    generation: int
    timeout_ms: int
    preceding_assistant_item_id: str

    def __post_init__(self) -> None:
        """Initialize the Pipecat frame and validate its captured timer state."""
        super().__post_init__()
        for name, value in (
            ("audio_start_ms", self.audio_start_ms),
            ("audio_end_ms", self.audio_end_ms),
            ("generation", self.generation),
            ("timeout_ms", self.timeout_ms),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.audio_end_ms < self.audio_start_ms:
            raise ValueError("audio_end_ms must be greater than or equal to audio_start_ms")
        if self.timeout_ms == 0:
            raise ValueError("timeout_ms must be positive")
        if not isinstance(self.preceding_assistant_item_id, str) or not self.preceding_assistant_item_id:
            raise ValueError("preceding_assistant_item_id must be a non-empty string")


class RealtimeASREndpointSilenceFrame(InputAudioRawFrame):
    """Internal PCM silence that translates manual commit into an ASR endpoint.

    A Realtime manual commit is a semantic end-of-utterance boundary. NVIDIA's
    continuous streaming ASR consumes audio only, so this private frame gives
    its endpoint detector that boundary without billing the synthetic samples
    as client-provided input audio.
    """


@dataclass
class RealtimeInputTranscriptionErrorFrame(ControlFrame, UninterruptibleFrame):
    """Fail a Realtime input transcript whose ASR owner is not trustworthy."""

    code: str
    message: str

    def __post_init__(self) -> None:
        """Initialize the Pipecat frame and validate its public error detail."""
        super().__post_init__()
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("code must be a non-empty string")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("message must be a non-empty string")


def _validate_manual_audio_boundary(audio_sample_cursor: int, sample_rate: int) -> None:
    if isinstance(audio_sample_cursor, bool) or not isinstance(audio_sample_cursor, int) or audio_sample_cursor < 0:
        raise ValueError("audio_sample_cursor must be a non-negative integer")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")


@dataclass
class RealtimeManualUserStartedSpeakingFrame(UserStartedSpeakingFrame):
    """Start a manual turn at its captured committed-audio cursor."""

    audio_sample_cursor: int
    sample_rate: int

    def __post_init__(self) -> None:
        """Initialize the Pipecat frame and validate its audio clock."""
        super().__post_init__()
        _validate_manual_audio_boundary(self.audio_sample_cursor, self.sample_rate)


@dataclass
class RealtimeManualUserStoppedSpeakingFrame(UserStoppedSpeakingFrame):
    """Stop a manual turn at its captured committed-audio cursor."""

    audio_sample_cursor: int
    sample_rate: int

    def __post_init__(self) -> None:
        """Initialize the Pipecat frame and validate its audio clock."""
        super().__post_init__()
        _validate_manual_audio_boundary(self.audio_sample_cursor, self.sample_rate)


@dataclass
class RealtimeResponseCreateFrame(DataFrame):
    """Carry one validated response request to its response gate.

    A plain ``LLMRunFrame`` cannot carry a response-local inference override and
    can race ahead of the final transcript produced by a manually committed
    input buffer. The gate turns this marker into exactly one context run after
    every preceding committed turn has reached the user context.
    """

    response_id: str
    tool_choice: str | dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    mcp_pipeline_names: frozenset[str] | None = None
    client_tool_bindings: dict[str, str] | None = None
    instructions: str | None = None
    max_output_tokens: int | str | None = None
    parallel_tool_calls: bool | None = None
    truncation: str | dict[str, Any] | None = None
    input_messages: list[dict[str, Any]] | None = None
    output_modalities: list[str] | None = None
    audio_output: dict[str, Any] | None = None


@dataclass
class RealtimeDeferredResponseCreateFrame(ControlFrame, UninterruptibleFrame):
    """Queue a response behind an earlier response or input boundary.

    OpenAI clients can submit a function output as soon as the function item is
    complete, before the originating response's later ``response.done`` event.
    A manual audio turn can likewise be committed while Response A is active.
    Clients also send ``response.create`` immediately after a text
    ``conversation.item.create``. This frame preserves those client-event
    orders in Pipecat without blocking the WebSocket receive loop while the
    earlier lifecycle is still entering canonical context or publishing.
    """

    request_id: str
    event_id: str | None
    response_metadata: dict[str, Any] | None
    tool_call_ids: tuple[str, ...]
    activation_generation: int
    conversation_item_ids: tuple[str, ...] = ()
    tool_choice: str | dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    public_tools: list[dict[str, Any]] | None = None
    mcp_pipeline_names: frozenset[str] | None = None
    client_tool_bindings: dict[str, str] | None = None
    instructions: str | None = None
    max_output_tokens: int | str | None = None
    parallel_tool_calls: bool | None = None
    truncation: str | dict[str, Any] | None = None
    response_input: RealtimeResponseInput | None = None
    input_messages: list[dict[str, Any]] | None = None
    output_modalities: list[str] | None = None
    audio_output: dict[str, Any] | None = None


@dataclass
class RealtimeClientToolOutputFrame(ControlFrame, UninterruptibleFrame):
    """Apply one already-validated client output at Response A's boundary.

    The serializer stages the output before returning this frame, so parallel
    outputs all meet their client deadline and the socket remains readable.
    The frame is uninterruptible because a later cancellation must still retire
    that staged output deterministically instead of dropping its state update.
    """

    event_id: str | None
    call_id: str
    tool_name: str
    response_id: str
    output: str
    item_id: str | None = None
    previous_item_id: str | None = None
    previous_item_id_supplied: bool = False


@dataclass
class RealtimeResponseContextFrame(LLMContextFrame):
    """Run one response with an isolated context while retaining its owner.

    ``context`` is an independent snapshot whose request controls apply only to
    this model invocation. ``canonical_context`` remains the connection-owned
    context so stateful services do not accidentally retain the response
    override.
    """

    response_id: str
    canonical_context: Any
