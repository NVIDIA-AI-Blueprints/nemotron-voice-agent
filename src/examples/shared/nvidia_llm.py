# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""NVIDIA LLM service with typed forced-tool and structured completions.

Pipecat consumes OpenAI Chat Completions streams, but it does not dispatch a
function until the complete stream has been coalesced. For a tool choice that
requires a call, request the provider's complete Chat Completion and adapt its
typed ``message.tool_calls`` into the same chunks Pipecat already consumes.
This leaves parsing at the provider boundary without matching or rewriting
model text, and it preserves normal streaming for ``auto``, ``none``, and
text-only responses.

One-shot structured inference uses the provider's OpenAI-compatible JSON Schema
response format and validates the decoded response before returning it.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import aclosing
from contextvars import ContextVar
from enum import Enum, auto
from typing import Any

import httpx
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from loguru import logger
from openai import NOT_GIVEN as OPENAI_NOT_GIVEN
from openai import APITimeoutError, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from pipecat.frames.frames import (
    Frame,
    LLMConfigureOutputFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.nvidia.llm import (
    NvidiaLLMService as PipecatNvidiaLLMService,
)
from pipecat.services.nvidia.llm import NvidiaLLMSettings
from pipecat.services.settings import NOT_GIVEN, assert_given, is_given

from examples.shared.frames import (
    LLMProviderCompletionReasonFrame,
    LLMProviderFinishReason,
    require_llm_provider_finish_reason,
)
from realtime.frames import RealtimeOwnedLLMFullResponseStartFrame, RealtimeResponseLLMContext
from realtime.protocol import RealtimeProtocolError

__all__ = ["NvidiaLLMService", "NvidiaLLMSettings"]

RealtimeTokenCounter = Callable[[dict[str, Any]], Awaitable[tuple[int, int]]]
RealtimeResponseStartedHandler = Callable[[str], Awaitable[None]]
RealtimeDeferredResponseActivator = Callable[
    [RealtimeResponseStartedHandler | None],
    Awaitable[str | None],
]
RealtimeDeferredResponseSnapshotHook = Callable[
    [LLMContext],
    Awaitable[
        tuple[
            RealtimeResponseLLMContext,
            tuple[Frame, ...],
            RealtimeDeferredResponseActivator,
            Callable[[], None],
        ]
        | None
    ],
]


def _effective_realtime_output_token_limit(
    context: LLMContext,
    *,
    model_max_output_tokens: int | None,
) -> int | None:
    """Resolve the provider ceiling for one immutable Realtime response.

    OpenAI Realtime's ``"inf"`` value means the maximum available for the
    selected model, not whatever happens to remain after filling the context
    window. Local profiles therefore supply a trusted model ceiling that is
    also used as the truncation reserve. A client integer remains an upper
    bound and cannot raise that model ceiling.
    """
    if not isinstance(context, RealtimeResponseLLMContext):
        return None
    requested = context.max_output_tokens
    if requested is None or requested == "inf":
        return model_max_output_tokens
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise ValueError("Realtime max_output_tokens must be a positive integer or 'inf'")
    return min(requested, model_max_output_tokens) if model_max_output_tokens is not None else requested


def _realtime_tokenize_url(base_url: object) -> str:
    """Resolve NVIDIA/vLLM's documented tokenizer endpoint from an API root."""
    value = str(base_url or "").rstrip("/")
    if value.endswith("/v1"):
        value = value[:-3]
    if not value:
        raise ValueError("Realtime context truncation requires an LLM base URL")
    return f"{value}/tokenize"


def _conversation_units(messages: list[dict[str, Any]], preserve_count: int) -> list[list[dict[str, Any]]]:
    """Group removable history into complete user/system-led conversation units."""
    tail = messages[preserve_count:]
    if not tail:
        return []
    boundaries = [
        index
        for index, message in enumerate(tail)
        if index > 0 and isinstance(message, Mapping) and message.get("role") in {"system", "user"}
    ]
    starts = [0, *boundaries]
    return [tail[start:end] for start, end in zip(starts, [*boundaries, len(tail)], strict=True)]


async def _truncate_realtime_context(
    params: dict[str, Any],
    context: LLMContext,
    *,
    count_tokens: RealtimeTokenCounter,
    model_max_output_tokens: int | None = None,
) -> dict[str, Any]:
    """Apply OpenAI Realtime auto/retention truncation to one request snapshot.

    The canonical Pipecat conversation remains intact. Only the immutable
    response snapshot sent to the provider drops oldest complete turns, which
    preserves public item identity and exact delete/truncate ownership.
    """
    if not isinstance(context, RealtimeResponseLLMContext):
        return params
    # A route without an exact provider tokenizer does not advertise native
    # truncation. Keep its existing provider behavior instead of introducing
    # an approximate counter or probing an unsupported endpoint.
    if context.truncation is None:
        return params
    messages = params.get("messages")
    if not isinstance(messages, list) or any(not isinstance(message, dict) for message in messages):
        raise TypeError("Realtime provider messages must be an array of objects")

    preserve_count = context.preserve_prompt_messages
    if preserve_count > len(messages):
        raise RuntimeError("Realtime prompt ownership exceeds the provider message snapshot")
    full_count, max_model_len = await count_tokens(params)
    if full_count < 0 or max_model_len < 1:
        raise RuntimeError("The LLM tokenizer returned invalid context metadata")

    output_reserve = _effective_realtime_output_token_limit(
        context,
        model_max_output_tokens=model_max_output_tokens,
    )
    if output_reserve is None:
        raise RuntimeError('Realtime truncation with max_output_tokens="inf" requires a trusted model output limit')
    model_input_limit = max_model_len - output_reserve
    if model_input_limit < 1:
        raise RealtimeProtocolError(
            message="The configured output-token limit leaves no room for Realtime input context",
            code="context_length_exceeded",
            param="session.max_output_tokens",
        )
    truncation = context.truncation
    prompt_count: int | None = None

    async def count_prompt_tokens() -> int:
        nonlocal prompt_count
        if prompt_count is None:
            prompt_params = {**params, "messages": messages[:preserve_count]}
            prompt_count, prompt_max_model_len = await count_tokens(prompt_params)
            if prompt_count < 0 or prompt_max_model_len != max_model_len:
                raise RuntimeError("The LLM tokenizer returned inconsistent prompt metadata")
        return prompt_count

    input_limit = model_input_limit
    if isinstance(truncation, Mapping):
        limits = truncation.get("token_limits")
        if isinstance(limits, Mapping) and isinstance(limits.get("post_instructions"), int):
            post_instructions = limits["post_instructions"]
            if post_instructions > model_input_limit:
                raise RealtimeProtocolError(
                    message=(
                        "session.truncation.token_limits.post_instructions cannot exceed "
                        "the model context window minus max_output_tokens"
                    ),
                    code="invalid_value",
                    param="session.truncation.token_limits.post_instructions",
                )
            prompt_tokens = await count_prompt_tokens()
            input_limit = min(model_input_limit, prompt_tokens + post_instructions)
    if full_count <= input_limit:
        return params
    if truncation == "disabled":
        raise RealtimeProtocolError(
            message="The conversation exceeds the model input-token limit and truncation is disabled",
            code="context_length_exceeded",
            param="session.truncation",
        )

    units = _conversation_units(messages, preserve_count)
    if len(units) < 2:
        raise RealtimeProtocolError(
            message="The latest Realtime turn exceeds the model input-token limit",
            code="context_length_exceeded",
            param="conversation",
        )

    target = input_limit
    if isinstance(truncation, Mapping) and truncation.get("type") == "retention_ratio":
        ratio = truncation.get("retention_ratio")
        if isinstance(ratio, int | float) and not isinstance(ratio, bool) and math.isfinite(float(ratio)):
            prompt_tokens = await count_prompt_tokens()
            post_instruction_budget = max(0, input_limit - prompt_tokens)
            target = min(input_limit, prompt_tokens + math.floor(post_instruction_budget * float(ratio)))

    async def candidate(drop_units: int) -> tuple[dict[str, Any], int]:
        retained = [message for unit in units[drop_units:] for message in unit]
        candidate_params = {**params, "messages": [*messages[:preserve_count], *retained]}
        count, candidate_max_model_len = await count_tokens(candidate_params)
        if count < 0 or candidate_max_model_len != max_model_len:
            raise RuntimeError("The LLM tokenizer returned inconsistent candidate metadata")
        return candidate_params, count

    low = 1
    high = len(units) - 1
    selected: tuple[dict[str, Any], int, int] | None = None
    while low <= high:
        midpoint = (low + high) // 2
        candidate_params, candidate_count = await candidate(midpoint)
        if candidate_count <= target:
            selected = candidate_params, candidate_count, midpoint
            high = midpoint - 1
        else:
            low = midpoint + 1

    if selected is None:
        candidate_params, candidate_count = await candidate(len(units) - 1)
        if candidate_count <= input_limit:
            selected = candidate_params, candidate_count, len(units) - 1
    if selected is None:
        raise RealtimeProtocolError(
            message="The latest Realtime turn exceeds the model input-token limit",
            code="context_length_exceeded",
            param="conversation",
        )

    truncated, retained_count, dropped_units = selected
    logger.info(
        "Applied Realtime context truncation "
        f"input_tokens={full_count}->{retained_count} limit={input_limit} dropped_turns={dropped_units}"
    )
    return truncated


def _named_function(tool_choice: object) -> str | None:
    """Return the function name from an OpenAI named-function choice."""
    if not isinstance(tool_choice, Mapping):
        return None
    if tool_choice.get("type") != "function":
        return None
    function = tool_choice.get("function")
    if not isinstance(function, Mapping):
        raise ValueError("Named function tool_choice must contain a function object")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Named function tool_choice must contain a non-empty function name")
    return name


def _forced_tool_choice(tool_choice: object) -> tuple[str, str | None] | None:
    """Return the supported forced-tool mode and optional function name."""
    if tool_choice == "required":
        return "required", None
    name = _named_function(tool_choice)
    if name is not None:
        return "named", name
    return None


def _stop_sequences(value: object, *, label: str) -> list[str]:
    """Normalize an OpenAI ``stop`` value into validated sequences."""
    if value is None or value is NOT_GIVEN or value is OPENAI_NOT_GIVEN:
        return []
    if isinstance(value, str):
        sequences = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        sequences = list(value)
    else:
        raise ValueError(f"{label} must be a string or a list of strings")
    if not all(isinstance(sequence, str) and sequence for sequence in sequences):
        raise ValueError(f"{label} must contain only non-empty strings")
    if len(sequences) > 4:
        raise ValueError(f"{label} may contain at most four sequences")
    return sequences


def _apply_forced_tool_call_stops(params: dict[str, Any], configured: tuple[str, ...]) -> None:
    """Merge model-profile tool terminals into the standard OpenAI stop field."""
    if not configured:
        return
    requested = _stop_sequences(params.get("stop"), label="stop")
    merged = list(dict.fromkeys([*requested, *configured]))
    if len(merged) > 4:
        raise ValueError("stop and forced_tool_call_stops may contain at most four unique sequences")
    params["stop"] = merged[0] if len(merged) == 1 else merged


def _function_names(tools: object) -> set[str]:
    """Return the function names exposed by an OpenAI tools request."""
    if not isinstance(tools, list):
        raise ValueError("A forced tool choice requires a tools array")

    names: set[str] = set()
    for index, tool in enumerate(tools):
        if not isinstance(tool, Mapping):
            raise ValueError(f"Tool at index {index} must be an object")
        if tool.get("type") != "function":
            continue
        function = tool.get("function")
        if not isinstance(function, Mapping):
            raise ValueError(f"Function tool at index {index} must contain a function object")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Function tool at index {index} must contain a non-empty name")
        if name in names:
            raise ValueError(f"Function tool request contains duplicate name {name!r}")
        names.add(name)

    if not names:
        raise ValueError("A forced tool choice requires at least one function tool")
    return names


def _validate_forced_tool_choice(
    completion: ChatCompletion,
    *,
    mode: str,
    expected_name: str | None,
    available_names: set[str],
    parallel_tool_calls: bool,
) -> Choice:
    """Validate a provider's complete, typed forced-tool response."""
    if len(completion.choices) != 1:
        raise ValueError(f"Forced tool completion must contain exactly one choice; received {len(completion.choices)}")

    choice = completion.choices[0]
    if choice.finish_reason not in {"stop", "tool_calls"}:
        raise ValueError(
            "Forced tool completion did not finish with a complete tool call; "
            f"received finish_reason={choice.finish_reason!r}"
        )

    calls = choice.message.tool_calls
    if not calls:
        target = expected_name if expected_name is not None else "an available function"
        raise ValueError(f"Forced tool completion did not return a structured call for {target!r}")
    if not parallel_tool_calls and len(calls) != 1:
        raise ValueError("Forced tool completion returned parallel calls while parallel_tool_calls is false")

    call_ids: set[str] = set()
    for call in calls:
        if call.type != "function":
            raise ValueError(f"Forced tool completion returned unsupported tool type {call.type!r}")
        if not call.id:
            raise ValueError("Forced tool completion returned a call without an id")
        if call.id in call_ids:
            raise ValueError(f"Forced tool completion returned duplicate call id {call.id!r}")
        call_ids.add(call.id)

        function = call.function
        if function.name not in available_names:
            raise ValueError(f"Forced tool completion returned unavailable function {function.name!r}")
        if mode == "named" and function.name != expected_name:
            raise ValueError(f"Forced tool completion returned {function.name!r}; expected {expected_name!r}")
        try:
            arguments = json.loads(function.arguments)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Forced tool completion returned invalid JSON arguments for {function.name!r}") from exc
        if not isinstance(arguments, dict):
            raise ValueError(f"Forced tool completion arguments for {function.name!r} must be a JSON object")

    return choice


def _chunk(
    completion: ChatCompletion,
    *,
    delta: dict[str, Any] | None = None,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> ChatCompletionChunk:
    """Build one OpenAI Chat Completion chunk from a complete response."""
    payload: dict[str, Any] = {
        "id": completion.id,
        "choices": [],
        "created": completion.created,
        "model": completion.model,
        "object": "chat.completion.chunk",
    }
    if completion.system_fingerprint is not None:
        payload["system_fingerprint"] = completion.system_fingerprint
    if completion.service_tier is not None:
        payload["service_tier"] = completion.service_tier
    if delta is not None:
        payload["choices"] = [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ]
    if usage is not None:
        payload["usage"] = usage
    return ChatCompletionChunk.model_validate(payload)


async def _forced_tool_choice_chunks(
    completion: ChatCompletion,
    choice: Choice,
) -> AsyncIterator[ChatCompletionChunk]:
    """Project a typed completion into Pipecat's existing chunk contract."""
    message = choice.message
    role_pending = True

    reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
    if reasoning:
        delta: dict[str, Any] = {"role": "assistant", "reasoning": reasoning}
        role_pending = False
        yield _chunk(completion, delta=delta)

    if message.content:
        delta = {"content": message.content}
        if role_pending:
            delta["role"] = "assistant"
            role_pending = False
        yield _chunk(completion, delta=delta)

    for index, call in enumerate(message.tool_calls or []):
        delta = {
            "tool_calls": [
                {
                    "index": index,
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
            ]
        }
        if role_pending:
            delta["role"] = "assistant"
            role_pending = False
        yield _chunk(completion, delta=delta)

    # A complete typed call is authoritative. Some OpenAI-compatible providers
    # label that terminal as ``stop``; expose Pipecat's canonical tool terminal
    # without mutating the provider response.
    yield _chunk(completion, delta={}, finish_reason="tool_calls")
    if completion.usage is not None:
        yield _chunk(
            completion,
            usage=completion.usage.model_dump(exclude_none=True),
        )


async def _incomplete_completion_chunks(
    completion: ChatCompletion,
    choice: Choice,
) -> AsyncIterator[ChatCompletionChunk]:
    """Publish only an incomplete provider terminal, discarding partial output."""
    yield _chunk(completion, delta={}, finish_reason=choice.finish_reason)
    if completion.usage is not None:
        yield _chunk(
            completion,
            usage=completion.usage.model_dump(exclude_none=True),
        )


def _project_stream_chunk(
    chunk: ChatCompletionChunk,
    *,
    delta: Mapping[str, Any],
    finish_reason: str | None,
    preserve_usage: bool,
) -> ChatCompletionChunk:
    """Copy one stream chunk with an explicit single-choice projection."""
    if len(chunk.choices) != 1:
        raise ValueError(f"LLM stream chunk must contain exactly one choice; received {len(chunk.choices)}")
    payload = chunk.model_dump()
    choice = payload["choices"][0]
    choice["delta"] = dict(delta)
    choice["finish_reason"] = finish_reason
    payload["choices"] = [choice]
    if not preserve_usage:
        payload["usage"] = None
    return ChatCompletionChunk.model_validate(payload)


class _RealtimeAssistantOutput(Enum):
    """Track the first structured output kind in one Realtime response."""

    UNRESOLVED = auto()
    TEXT = auto()
    TOOL_CALL = auto()


def _is_formatting_only_content(content: object) -> bool:
    """Return whether a provider content delta has no semantic text."""
    return isinstance(content, str) and (not content or content.isspace())


def _without_formatting_only_content(chunk: ChatCompletionChunk) -> ChatCompletionChunk | None:
    """Remove only a formatting-only content field, retaining typed metadata."""
    if not chunk.choices:
        return chunk
    choice = chunk.choices[0]
    delta = choice.delta.model_dump(exclude_none=True) if choice.delta else {}
    if not _is_formatting_only_content(delta.get("content")):
        return chunk
    delta.pop("content")
    if not delta and choice.finish_reason is None and chunk.usage is None:
        return None
    return _project_stream_chunk(
        chunk,
        delta=delta,
        finish_reason=choice.finish_reason,
        preserve_usage=True,
    )


async def _normalize_realtime_assistant_output(
    stream: AsyncIterator[ChatCompletionChunk],
) -> AsyncIterator[ChatCompletionChunk]:
    """Drop formatting-only text only when the response is structurally tool-only.

    Providers can emit a role and line breaks before their first typed tool
    fragment. Hold that ambiguous prefix until a semantic content delta, a
    function call, or a terminal proves the response kind. Mixed text and tool
    responses retain their original content exactly.
    """
    output = _RealtimeAssistantOutput.UNRESOLVED
    pending: list[ChatCompletionChunk] = []
    try:
        async for chunk in stream:
            if output is not _RealtimeAssistantOutput.UNRESOLVED:
                yield chunk
                continue

            choice = chunk.choices[0] if chunk.choices else None
            if choice is None:
                pending.append(chunk)
                continue
            delta = choice.delta.model_dump(exclude_none=True) if choice.delta else {}
            content = delta.get("content")
            semantic_content = isinstance(content, str) and bool(content) and not content.isspace()
            tool_call = bool(choice.delta and choice.delta.tool_calls)

            if semantic_content:
                output = _RealtimeAssistantOutput.TEXT
                for staged in pending:
                    yield staged
                pending.clear()
                yield chunk
                continue

            if tool_call:
                output = _RealtimeAssistantOutput.TOOL_CALL
                for staged in pending:
                    normalized = _without_formatting_only_content(staged)
                    if normalized is not None:
                        yield normalized
                pending.clear()
                normalized = _without_formatting_only_content(chunk)
                if normalized is not None:
                    yield normalized
                continue

            if choice.finish_reason is not None:
                output = _RealtimeAssistantOutput.TEXT
                for staged in pending:
                    yield staged
                pending.clear()
                yield chunk
                continue

            pending.append(chunk)

        for staged in pending:
            yield staged
    finally:
        close = getattr(stream, "aclose", None) or getattr(stream, "close", None)
        if close is not None:
            await close()


def _merge_streamed_tool_call_fragments(
    calls: dict[int, dict[str, Any]],
    fragments: Sequence[Any],
) -> None:
    """Accumulate every provider-typed tool fragment by its declared index."""
    for fragment in fragments:
        index = fragment.index
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f"Streamed tool call has invalid index {index!r}")
        call = calls.setdefault(
            index,
            {"id": None, "type": None, "name": "", "arguments": ""},
        )
        if fragment.id:
            if call["id"] is not None and call["id"] != fragment.id:
                raise ValueError(f"Streamed tool call {index} changed its call id")
            call["id"] = fragment.id
        if fragment.type:
            if call["type"] is not None and call["type"] != fragment.type:
                raise ValueError(f"Streamed tool call {index} changed its type")
            call["type"] = fragment.type
        if fragment.function is not None:
            if fragment.function.name:
                call["name"] += fragment.function.name
            if fragment.function.arguments:
                call["arguments"] += fragment.function.arguments


def _validate_streamed_tool_calls(
    calls: Mapping[int, Mapping[str, Any]],
    *,
    available_names: set[str],
    parallel_tool_calls: bool,
) -> list[Mapping[str, Any]]:
    """Freeze complete streamed calls before any delta reaches Pipecat."""
    indices = sorted(calls)
    if indices != list(range(len(indices))):
        raise ValueError(f"Streamed tool-call indices must be contiguous from zero; received {indices}")
    if not indices:
        raise ValueError("Tool-call terminal did not contain a structured function call")
    if not parallel_tool_calls and len(indices) != 1:
        raise ValueError("Provider returned parallel calls while parallel_tool_calls is false")

    call_ids: set[str] = set()
    completed: list[Mapping[str, Any]] = []
    for index in indices:
        call = calls[index]
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError(f"Streamed tool call {index} is missing its call id")
        if call_id in call_ids:
            raise ValueError(f"Streamed tool call {index} reused call id {call_id!r}")
        call_ids.add(call_id)
        if call.get("type") != "function":
            raise ValueError(f"Streamed tool call {index} has unsupported type {call.get('type')!r}")
        name = call.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Streamed tool call {index} is missing its function name")
        if name not in available_names:
            raise ValueError(f"Streamed tool call {index} selected unavailable function {name!r}")
        arguments = call.get("arguments")
        if not isinstance(arguments, str) or not arguments:
            raise ValueError(f"Streamed tool call {index} is missing its function arguments")
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Streamed tool call {index} returned invalid JSON arguments for {name!r}") from exc
        if not isinstance(decoded, dict):
            raise ValueError(f"Streamed tool call arguments for {name!r} must be a JSON object")
        completed.append(call)
    return completed


def _canonical_streamed_tool_call_chunk(
    template: ChatCompletionChunk,
    *,
    index: int,
    call: Mapping[str, Any],
) -> ChatCompletionChunk:
    """Emit one complete call per chunk for Pipecat's sequential accumulator."""
    return _project_stream_chunk(
        template,
        delta={
            **({"role": "assistant"} if index == 0 else {}),
            "tool_calls": [
                {
                    "index": index,
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"],
                    },
                }
            ],
        },
        finish_reason=None,
        preserve_usage=False,
    )


async def _hold_streamed_tool_calls_until_terminal(
    stream: AsyncIterator[ChatCompletionChunk],
    *,
    available_names: set[str],
    parallel_tool_calls: bool,
) -> AsyncIterator[ChatCompletionChunk]:
    """Validate and canonicalize typed stream calls at the provider terminal.

    Pipecat executes any parseable accumulated tool arguments when a stream
    closes, irrespective of its finish reason, and reads only the first tool
    fragment in each chunk. Hold every typed fragment outside Pipecat until the
    provider reports a complete terminal, then replay one validated call per
    chunk. ``length`` and ``content_filter`` expose only a sanitized terminal.
    """
    calls: dict[int, dict[str, Any]] = {}
    saw_calls = False
    try:
        async for chunk in stream:
            if len(chunk.choices) > 1:
                raise ValueError(f"LLM stream chunk must contain at most one choice; received {len(chunk.choices)}")
            choice = chunk.choices[0] if chunk.choices else None
            finish_reason = choice.finish_reason if choice is not None else None
            fragments = list(choice.delta.tool_calls or []) if choice is not None and choice.delta else []
            if fragments:
                saw_calls = True
                _merge_streamed_tool_call_fragments(calls, fragments)

            if not saw_calls:
                if finish_reason in {"tool_calls", "function_call"}:
                    _validate_streamed_tool_calls(
                        calls,
                        available_names=available_names,
                        parallel_tool_calls=parallel_tool_calls,
                    )
                yield chunk
                continue

            if choice is None:
                yield chunk
                continue

            visible_delta = choice.delta.model_dump(exclude_none=True) if choice.delta else {}
            visible_delta.pop("tool_calls", None)
            if visible_delta or (finish_reason is None and chunk.usage is not None):
                yield _project_stream_chunk(
                    chunk,
                    delta=visible_delta,
                    finish_reason=None,
                    preserve_usage=finish_reason is None,
                )

            if finish_reason is None:
                continue

            if finish_reason in {"length", "content_filter"}:
                logger.warning(f"Discarding incomplete streamed tool-call output finish_reason={finish_reason!r}")
                calls.clear()
                yield _project_stream_chunk(
                    chunk,
                    delta={},
                    finish_reason=finish_reason,
                    preserve_usage=True,
                )
                continue

            if finish_reason not in {"stop", "tool_calls", "function_call"}:
                raise ValueError(f"Unsupported streamed tool-call finish reason {finish_reason!r}")
            completed = _validate_streamed_tool_calls(
                calls,
                available_names=available_names,
                parallel_tool_calls=parallel_tool_calls,
            )
            for index, call in enumerate(completed):
                yield _canonical_streamed_tool_call_chunk(chunk, index=index, call=call)
            calls.clear()
            yield _project_stream_chunk(
                chunk,
                delta={},
                finish_reason="tool_calls",
                preserve_usage=True,
            )
    finally:
        close = getattr(stream, "aclose", None) or getattr(stream, "close", None)
        if close is not None:
            await close()


def _set_completion_token_limit(params: dict[str, Any], max_tokens: int | None) -> None:
    """Select exactly one OpenAI completion-token field for a one-shot request."""
    if max_tokens is not None and (isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1):
        raise ValueError("max_tokens must be a positive integer")

    configured_max_completion_tokens = params.get("max_completion_tokens", NOT_GIVEN)
    use_max_completion_tokens = (
        is_given(configured_max_completion_tokens)
        and configured_max_completion_tokens is not OPENAI_NOT_GIVEN
        and configured_max_completion_tokens is not None
    )
    if use_max_completion_tokens:
        if max_tokens is not None:
            params["max_completion_tokens"] = max_tokens
        params.pop("max_tokens", None)
        return

    configured_max_tokens = params.get("max_tokens", NOT_GIVEN)
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    elif (
        not is_given(configured_max_tokens)
        or configured_max_tokens is OPENAI_NOT_GIVEN
        or configured_max_tokens is None
    ):
        params.pop("max_tokens", None)
    params.pop("max_completion_tokens", None)


def _apply_realtime_completion_token_limit(
    params: dict[str, Any],
    context: LLMContext,
    *,
    model_max_output_tokens: int | None = None,
) -> None:
    """Project one response-local Realtime limit onto the provider request only."""
    if not isinstance(context, RealtimeResponseLLMContext):
        return
    max_output_tokens = context.max_output_tokens
    effective_limit = _effective_realtime_output_token_limit(
        context,
        model_max_output_tokens=model_max_output_tokens,
    )
    if max_output_tokens is None and effective_limit is None:
        return
    if effective_limit is None:
        params.pop("max_tokens", None)
        params.pop("max_completion_tokens", None)
        return
    _set_completion_token_limit(params, effective_limit)


def _apply_realtime_parallel_tool_calls(
    params: dict[str, Any],
    context: LLMContext,
    *,
    session_parallel_tool_calls: bool | None = None,
) -> bool:
    """Project the effective Realtime tool-call policy onto one provider request."""
    configured = params.get("parallel_tool_calls", NOT_GIVEN)
    if isinstance(context, RealtimeResponseLLMContext) and context.parallel_tool_calls is not None:
        configured = context.parallel_tool_calls
    elif session_parallel_tool_calls is not None:
        configured = session_parallel_tool_calls

    if configured in (NOT_GIVEN, OPENAI_NOT_GIVEN, None):
        effective = True
    elif isinstance(configured, bool):
        effective = configured
    else:
        raise ValueError("parallel_tool_calls must be a boolean")

    tools = params.get("tools")
    tools_active = isinstance(tools, list) and any(
        isinstance(tool, Mapping) and tool.get("type") == "function" for tool in tools
    )
    if tools_active:
        params["parallel_tool_calls"] = effective
    else:
        # NVIDIA endpoints need not accept a tool policy on a request that has
        # no callable functions. The value remains effective vacuously and is
        # applied as soon as a response exposes tools.
        params.pop("parallel_tool_calls", None)
    return effective


def _normalize_realtime_empty_tool_request(params: dict[str, Any], context: LLMContext) -> None:
    """Omit vacuous Realtime tool controls from the provider request.

    A Realtime session may keep its standard ``tool_choice`` default of
    ``"auto"`` while exposing no function tools. OpenAI-compatible Chat
    Completions providers differ here: some reject ``tool_choice`` or an empty
    ``tools`` array even though there is no possible call. Preserve the public
    Realtime configuration and remove only the no-op provider fields. Forced
    choices remain untouched so the normal strict validation rejects them.
    """
    if not isinstance(context, RealtimeResponseLLMContext):
        return
    tools = params.get("tools")
    tools_active = isinstance(tools, list) and any(
        isinstance(tool, Mapping) and tool.get("type") == "function" for tool in tools
    )
    if tools_active:
        return
    tool_choice = params.get("tool_choice", NOT_GIVEN)
    if tool_choice not in (NOT_GIVEN, OPENAI_NOT_GIVEN, None, "auto", "none"):
        return
    params.pop("tools", None)
    params.pop("tool_choice", None)
    params.pop("parallel_tool_calls", None)


def _structured_extra_body(params: Mapping[str, Any]) -> dict[str, Any]:
    """Copy provider parameters and select concise structured-generation mode."""
    configured = params.get("extra_body")
    if configured is None or not is_given(configured):
        extra_body: dict[str, Any] = {}
    elif isinstance(configured, Mapping):
        extra_body = copy.deepcopy(dict(configured))
    else:
        raise ValueError("Structured inference extra_body must be an object")

    configured_template = extra_body.get("chat_template_kwargs")
    if configured_template is None:
        chat_template_kwargs: dict[str, Any] = {}
    elif isinstance(configured_template, Mapping):
        chat_template_kwargs = copy.deepcopy(dict(configured_template))
    else:
        raise ValueError("Structured inference chat_template_kwargs must be an object")
    chat_template_kwargs["enable_thinking"] = False
    extra_body["chat_template_kwargs"] = chat_template_kwargs

    # These controls are irrelevant when thinking is disabled. Removing them
    # also prevents unsupported provider-specific budget fields from changing
    # the deterministic structured-output contract.
    extra_body.pop("reasoning_budget", None)
    extra_body.pop("thinking_token_budget", None)
    return extra_body


class NvidiaLLMService(PipecatNvidiaLLMService):
    """Use complete typed responses when the request requires a tool call.

    ``forced_tool_call_stops`` describes model-native tool-call terminators for
    a serving profile. They are sent through the standard OpenAI ``stop``
    parameter only for ``required`` and named tool choices. The provider's
    configured tool parser remains the sole authority for structured calls.
    """

    def __init__(
        self,
        *,
        forced_tool_call_stops: str | Sequence[str] | None = None,
        realtime_parallel_tool_calls: bool | None = None,
        realtime_model_max_output_tokens: int | None = None,
        **kwargs,
    ) -> None:
        """Initialize with optional model-profile forced-tool terminators."""
        if realtime_parallel_tool_calls is not None and not isinstance(realtime_parallel_tool_calls, bool):
            raise ValueError("realtime_parallel_tool_calls must be a boolean")
        if realtime_model_max_output_tokens is not None and (
            isinstance(realtime_model_max_output_tokens, bool)
            or not isinstance(realtime_model_max_output_tokens, int)
            or not 1 <= realtime_model_max_output_tokens <= 4096
        ):
            raise ValueError("realtime_model_max_output_tokens must be an integer between 1 and 4096")
        base_url = kwargs.get("base_url")
        self._realtime_tokenize_url = _realtime_tokenize_url(base_url) if base_url else None
        api_key = kwargs.get("api_key")
        self._realtime_tokenize_headers = (
            {"Authorization": f"Bearer {api_key}"} if isinstance(api_key, str) and api_key else {}
        )
        self._forced_tool_call_stops = tuple(_stop_sequences(forced_tool_call_stops, label="forced_tool_call_stops"))
        self._realtime_parallel_tool_calls = realtime_parallel_tool_calls
        self._realtime_model_max_output_tokens = realtime_model_max_output_tokens
        self._realtime_run_owner: ContextVar[tuple[str, str, int | None] | None] = ContextVar(
            f"nvidia_realtime_run_owner_{id(self)}",
            default=None,
        )
        self._realtime_deferred_response_snapshot_hook: RealtimeDeferredResponseSnapshotHook | None = None
        super().__init__(**kwargs)

    def bind_realtime_deferred_response_snapshot(self, hook: RealtimeDeferredResponseSnapshotHook) -> None:
        """Bind ownership for assistant-aggregator tool follow-up runs."""
        if not callable(hook):
            raise TypeError("Realtime deferred response snapshot hook must be callable")
        if self._realtime_deferred_response_snapshot_hook is not None:
            raise RuntimeError("Realtime deferred response snapshot hook is already bound")
        self._realtime_deferred_response_snapshot_hook = hook

    async def _count_realtime_chat_tokens(self, params: dict[str, Any]) -> tuple[int, int]:
        """Count the exact rendered prompt with NVIDIA/vLLM's tokenizer API."""
        if self._realtime_tokenize_url is None:
            raise RuntimeError("Realtime context truncation requires the NVIDIA LLM base URL")
        payload: dict[str, Any] = {
            "model": params.get("model"),
            "messages": copy.deepcopy(params.get("messages")),
            "add_generation_prompt": True,
        }
        tools = params.get("tools")
        if isinstance(tools, list):
            payload["tools"] = copy.deepcopy(tools)
        extra_body = params.get("extra_body")
        if isinstance(extra_body, Mapping):
            chat_template_kwargs = extra_body.get("chat_template_kwargs")
            if isinstance(chat_template_kwargs, Mapping):
                payload["chat_template_kwargs"] = copy.deepcopy(dict(chat_template_kwargs))

        timeout = httpx.Timeout(10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                self._realtime_tokenize_url,
                json=payload,
                headers=self._realtime_tokenize_headers,
            )
            response.raise_for_status()
        decoded = response.json()
        if not isinstance(decoded, Mapping):
            raise RuntimeError("The LLM tokenizer returned a non-object response")
        count = decoded.get("count")
        max_model_len = decoded.get("max_model_len")
        if isinstance(count, bool) or not isinstance(count, int):
            raise RuntimeError("The LLM tokenizer response omitted its token count")
        if isinstance(max_model_len, bool) or not isinstance(max_model_len, int):
            raise RuntimeError("The LLM tokenizer response omitted max_model_len")
        return count, max_model_len

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Bind Pipecat's provider start frame to its immutable Realtime context."""
        if (
            direction == FrameDirection.UPSTREAM
            and isinstance(frame, LLMContextFrame)
            and not isinstance(frame.context, RealtimeResponseLLMContext)
            and self._realtime_deferred_response_snapshot_hook is not None
        ):
            await LLMService.process_frame(self, frame, direction)
            await self._process_deferred_realtime_context(frame.context)
            return

        run_owner: tuple[str, str, int | None] | None = None
        if isinstance(frame, LLMContextFrame) and isinstance(frame.context, RealtimeResponseLLMContext):
            run_owner = (
                frame.context.response_id or "",
                frame.context.run_owner_id or "",
                frame.context.activation_generation,
            )
        token = self._realtime_run_owner.set(run_owner)
        try:
            await super().process_frame(frame, direction)
        finally:
            self._realtime_run_owner.reset(token)

    async def _process_deferred_realtime_context(self, canonical_context: LLMContext) -> None:
        """Run a server-tool follow-up as a separately owned Realtime response."""
        hook = self._realtime_deferred_response_snapshot_hook
        if hook is None:
            raise RuntimeError("Realtime deferred response snapshot hook is not bound")
        prepared = await hook(canonical_context)
        if prepared is None:
            return
        run_context, setup_frames, activate, abort = prepared
        if not isinstance(run_context, RealtimeResponseLLMContext):
            abort()
            raise TypeError("Realtime deferred response hook did not return a Realtime response context")
        self._sync_registered_tool_handlers(run_context.tools)

        response_started = False
        metrics_started = False
        restore_skip_tts = False
        previous_skip_tts = self._skip_tts
        try:
            for setup_frame in setup_frames:
                if isinstance(setup_frame, LLMConfigureOutputFrame):
                    self._skip_tts = setup_frame.skip_tts
                    restore_skip_tts = True
                await self.push_frame(setup_frame, FrameDirection.DOWNSTREAM)

            async def _publish_owned_start(response_id: str) -> None:
                await self.push_frame(
                    RealtimeOwnedLLMFullResponseStartFrame(response_id=response_id),
                    FrameDirection.DOWNSTREAM,
                )

            response_id = await activate(_publish_owned_start)
            if response_id is None:
                return
            response_started = True
            run_context.response_id = response_id
            await self.start_processing_metrics()
            metrics_started = True
            await self._process_context(run_context)
        except asyncio.CancelledError:
            raise
        except httpx.TimeoutException as exc:
            await self._call_event_handler("on_completion_timeout")
            await self.push_error(error_msg="LLM completion timeout", exception=exc)
        except Exception as exc:
            await self.push_error(error_msg=f"Error during completion: {exc}", exception=exc)
        finally:
            if not response_started:
                abort()
            if metrics_started:
                await self.stop_processing_metrics()
            try:
                if response_started:
                    await self.push_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
            finally:
                if restore_skip_tts:
                    self._skip_tts = previous_skip_tts

    async def push_frame(
        self,
        frame: Frame,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ) -> None:
        """Replace only Pipecat's provider-start frame with its typed owner."""
        run_owner = self._realtime_run_owner.get()
        if type(frame) is LLMFullResponseStartFrame and run_owner is not None:
            response_id, owner_id, activation_generation = run_owner
            frame = RealtimeOwnedLLMFullResponseStartFrame(
                response_id=response_id,
                run_owner_id=owner_id,
                activation_generation=activation_generation,
            )
        await super().push_frame(frame, direction)

    def set_realtime_parallel_tool_calls(self, enabled: bool) -> None:
        """Apply the native Realtime session policy to subsequent requests.

        Pipecat service settings intentionally contain provider request fields
        only; ``parallel_tool_calls`` is a Realtime orchestration policy that
        this adapter projects after it has inspected the effective tool list.
        Keep the update explicit instead of mutating the private attribute from
        the transport binding.
        """
        if not isinstance(enabled, bool):
            raise TypeError("Realtime parallel_tool_calls must be a boolean")
        self._realtime_parallel_tool_calls = enabled

    async def _with_provider_completion_reason(
        self,
        stream: AsyncIterator[ChatCompletionChunk],
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Forward a normal stream and publish each typed terminal reason.

        Pipecat's OpenAI streaming loop consumes the terminal chunk without
        forwarding its ``finish_reason``. Every chunk is yielded unchanged so
        content and tool-call handling stay intact. The ordered metadata frame
        is published after the wrapped stream has finalized all buffered model
        output and before Pipecat emits ``LLMFullResponseEndFrame``.
        """
        terminal_reason: LLMProviderFinishReason | None = None
        async with aclosing(stream):
            async for chunk in stream:
                finish_reason = chunk.choices[0].finish_reason if chunk.choices else None
                if terminal_reason is not None and chunk.choices:
                    raise ValueError("Provider stream emitted choices after its terminal chunk")
                yield chunk
                if finish_reason is not None:
                    terminal_reason = require_llm_provider_finish_reason(finish_reason)
        if terminal_reason is None:
            raise ValueError("Provider stream ended without a finish_reason")
        await self.push_frame(LLMProviderCompletionReasonFrame(finish_reason=terminal_reason))

    async def _request_chat_completion(self, params: dict[str, Any]):
        """Send one Chat Completions request with Pipecat's retry policy."""
        if self._retry_on_timeout:
            try:
                return await asyncio.wait_for(
                    self._client.chat.completions.create(**params),
                    timeout=self._retry_timeout_secs,
                )
            except (TimeoutError, APITimeoutError):
                logger.debug(f"{self}: Retrying chat completion due to timeout")
        return await self._client.chat.completions.create(**params)

    async def run_structured_inference(
        self,
        context: LLMContext,
        *,
        schema: Mapping[str, Any],
        schema_name: str,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
    ) -> Any:
        """Run one deterministic JSON-Schema-constrained completion.

        Structured control-plane calls do not need a reasoning transcript. The
        request therefore disables thinking and uses temperature zero so the
        configured token budget is reserved for the schema-constrained result.
        All overrides are applied to a fresh request dictionary; the service's
        shared settings remain unchanged for normal pipeline inference.

        Args:
            context: Messages for the one-shot completion.
            schema: Draft 2020-12 JSON Schema for the decoded response.
            schema_name: Provider-visible name for the response schema.
            max_tokens: Optional completion-token limit for this request.
            system_instruction: Optional system-instruction override.

        Returns:
            The decoded response after Draft 2020-12 validation.

        Raises:
            TypeError: If the provider does not return a Chat Completion.
            ValueError: If the schema, completion terminal, JSON, or decoded
                response is invalid.
        """
        if not isinstance(schema, Mapping):
            raise ValueError("Structured inference schema must be an object")
        if not isinstance(schema_name, str) or not schema_name.strip():
            raise ValueError("Structured inference schema_name must be a non-empty string")
        schema_name = schema_name.strip()

        schema_copy = copy.deepcopy(dict(schema))
        Draft202012Validator.check_schema(schema_copy)
        validator = Draft202012Validator(schema_copy)

        effective_instruction = system_instruction or self._settings.system_instruction
        adapter = self.get_llm_adapter()
        invocation_params = adapter.get_llm_invocation_params(
            context,
            system_instruction=effective_instruction,
            convert_developer_to_user=not self.supports_developer_role,
        )
        params = self.build_chat_completion_params(invocation_params)
        params["stream"] = False
        params.pop("stream_options", None)
        _set_completion_token_limit(params, max_tokens)
        params["temperature"] = 0.0
        params["extra_body"] = _structured_extra_body(params)
        params["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema_copy,
            },
        }

        completion = await self._request_chat_completion(params)
        if not isinstance(completion, ChatCompletion):
            raise TypeError("Structured inference did not return a ChatCompletion")
        if len(completion.choices) != 1:
            raise ValueError(
                f"Structured inference completion must contain exactly one choice; received {len(completion.choices)}"
            )

        choice = completion.choices[0]
        if choice.finish_reason != "stop":
            raise ValueError(
                "Structured inference did not finish with a complete response; "
                f"received finish_reason={choice.finish_reason!r}"
            )
        if choice.message.refusal:
            raise ValueError("Structured inference was refused by the provider")
        content = choice.message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Structured inference returned empty content")
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Structured inference returned invalid JSON") from exc
        try:
            validator.validate(decoded)
        except JSONSchemaValidationError as exc:
            path = ".".join(str(part) for part in exc.absolute_path)
            location = f" at {path}" if path else ""
            raise ValueError(f"Structured inference response failed schema validation{location}") from exc
        return decoded

    async def get_chat_completions(
        self,
        context: LLMContext,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Create a normal stream or a typed forced-tool completion."""
        if not isinstance(context, RealtimeResponseLLMContext):
            return await super().get_chat_completions(context)

        adapter = self.get_llm_adapter()
        logger.debug(f"{self}: Generating chat from context {adapter.get_messages_for_logging(context)}")
        params_from_context = adapter.get_llm_invocation_params(
            context,
            system_instruction=assert_given(self._settings.system_instruction),
            convert_developer_to_user=not self.supports_developer_role,
        )
        params = self.build_chat_completion_params(params_from_context)
        _apply_realtime_completion_token_limit(
            params,
            context,
            model_max_output_tokens=self._realtime_model_max_output_tokens,
        )
        parallel_tool_calls = _apply_realtime_parallel_tool_calls(
            params,
            context,
            session_parallel_tool_calls=self._realtime_parallel_tool_calls,
        )
        _normalize_realtime_empty_tool_request(params, context)
        params = await _truncate_realtime_context(
            params,
            context,
            count_tokens=self._count_realtime_chat_tokens,
            model_max_output_tokens=self._realtime_model_max_output_tokens,
        )
        forced_choice = _forced_tool_choice(params.get("tool_choice"))
        if forced_choice is None:
            stream_tool_choice = params.get("tool_choice")
            available_names = (
                _function_names(params.get("tools"))
                if stream_tool_choice != "none" and isinstance(params.get("tools"), list) and params.get("tools")
                else set()
            )
            stream = await self._request_chat_completion(params)
            if not isinstance(stream, AsyncStream):
                raise TypeError("Streaming request did not return an AsyncStream")
            safe_stream = _hold_streamed_tool_calls_until_terminal(
                stream,
                available_names=available_names,
                parallel_tool_calls=parallel_tool_calls,
            )
            # Keep normalization downstream: validated calls are replayed only
            # at their terminal, after any later semantic content in the stream.
            normalized_stream = _normalize_realtime_assistant_output(safe_stream)
            return self._with_provider_completion_reason(self._handle_reasoning_content(normalized_stream))

        mode, expected_name = forced_choice
        available_names = _function_names(params.get("tools"))
        _apply_forced_tool_call_stops(params, self._forced_tool_call_stops)
        params["stream"] = False
        params.pop("stream_options", None)
        logger.debug(
            f"{self}: Requesting typed completion for forced tool choice mode={mode!r}, function={expected_name!r}"
        )

        completion = await self._request_chat_completion(params)
        if not isinstance(completion, ChatCompletion):
            raise TypeError("Forced tool request did not return a ChatCompletion")
        if len(completion.choices) != 1:
            raise ValueError(
                f"Forced tool completion must contain exactly one choice; received {len(completion.choices)}"
            )
        terminal_choice = completion.choices[0]
        if terminal_choice.finish_reason in {"length", "content_filter"}:
            logger.warning(
                f"{self}: Discarding incomplete forced-tool output finish_reason={terminal_choice.finish_reason!r}"
            )
            return self._with_provider_completion_reason(_incomplete_completion_chunks(completion, terminal_choice))
        choice = _validate_forced_tool_choice(
            completion,
            mode=mode,
            expected_name=expected_name,
            available_names=available_names,
            parallel_tool_calls=parallel_tool_calls,
        )
        if choice.finish_reason != "tool_calls":
            logger.debug(
                f"{self}: Normalizing forced tool completion finish_reason={choice.finish_reason!r} to 'tool_calls'"
            )
        forced_stream = _forced_tool_choice_chunks(completion, choice)
        normalized_stream = _normalize_realtime_assistant_output(forced_stream)
        return self._with_provider_completion_reason(self._handle_reasoning_content(normalized_stream))
