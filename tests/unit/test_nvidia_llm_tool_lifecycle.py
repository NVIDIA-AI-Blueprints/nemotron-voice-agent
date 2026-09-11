# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

from __future__ import annotations

import unittest
from collections.abc import AsyncIterator
from typing import Any

from openai.types.chat import ChatCompletion, ChatCompletionChunk

from examples.shared.nvidia_llm import (
    NvidiaLLMService,
    _forced_tool_choice_chunks,
    _hold_streamed_tool_calls_until_terminal,
    _normalize_realtime_assistant_output,
)


def _stream_chunk(
    delta: dict[str, Any],
    *,
    finish_reason: str | None = None,
) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl_stream",
            "created": 1,
            "model": "test-model",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
    )


def _tool_call(
    *,
    index: int = 0,
    call_id: str = "call_lookup",
    name: str = "lookup",
    arguments: str = "{}",
) -> dict[str, Any]:
    return {
        "index": index,
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _tool_delta(**overrides: Any) -> dict[str, Any]:
    return {"tool_calls": [_tool_call(**overrides)]}


def _tool_completion(content: str) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl_forced",
            "created": 1,
            "model": "test-model",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": _tool_delta()["tool_calls"],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
    )


def _usage_chunk() -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl_stream",
            "created": 1,
            "model": "test-model",
            "object": "chat.completion.chunk",
            "choices": [],
            "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
        }
    )


async def _stream(*chunks: ChatCompletionChunk) -> AsyncIterator[ChatCompletionChunk]:
    for chunk in chunks:
        yield chunk


async def _collect_stream(*chunks: ChatCompletionChunk) -> list[ChatCompletionChunk]:
    validated = _hold_streamed_tool_calls_until_terminal(
        _stream(*chunks),
        available_names={"lookup"},
        parallel_tool_calls=False,
    )
    return [chunk async for chunk in _normalize_realtime_assistant_output(validated)]


class _ProviderReasonSink:
    async def push_frame(self, frame: Any) -> None:
        pass


async def _collect_provider_stream(
    *chunks: ChatCompletionChunk,
    parallel_tool_calls: bool = False,
) -> list[ChatCompletionChunk]:
    validated = _hold_streamed_tool_calls_until_terminal(
        _stream(*chunks),
        available_names={"lookup"},
        parallel_tool_calls=parallel_tool_calls,
    )
    with_terminal = NvidiaLLMService._with_provider_completion_reason(_ProviderReasonSink(), validated)
    return [chunk async for chunk in with_terminal]


def _contents(chunks: list[ChatCompletionChunk]) -> list[str]:
    return [
        choice.delta.content
        for chunk in chunks
        for choice in chunk.choices
        if choice.delta and choice.delta.content is not None
    ]


def _tool_names(chunks: list[ChatCompletionChunk]) -> list[str]:
    return [
        call.function.name
        for chunk in chunks
        for choice in chunk.choices
        if choice.delta
        for call in choice.delta.tool_calls or []
    ]


class StreamedToolLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_provider_tool_calls_are_rejected_before_dispatch(self) -> None:
        duplicate_calls = {
            "tool_calls": [
                _tool_call(),
                _tool_call(index=1),
            ]
        }
        parallel_calls = {
            "tool_calls": [
                _tool_call(),
                _tool_call(index=1, call_id="call_other"),
            ]
        }
        terminal = _stream_chunk({}, finish_reason="tool_calls")
        cases = (
            (
                "unknown function",
                (_stream_chunk(_tool_delta(name="missing")), terminal),
                False,
                "Streamed tool call 0 selected unavailable function 'missing'",
            ),
            (
                "invalid arguments JSON",
                (_stream_chunk(_tool_delta(arguments="{")), terminal),
                False,
                "Streamed tool call 0 returned invalid JSON arguments for 'lookup'",
            ),
            (
                "duplicate call ID",
                (_stream_chunk(duplicate_calls), terminal),
                True,
                "Streamed tool call 1 reused call id 'call_lookup'",
            ),
            (
                "parallel calls prohibited",
                (_stream_chunk(parallel_calls), terminal),
                False,
                "Provider returned parallel calls while parallel_tool_calls is false",
            ),
            (
                "empty tool terminal",
                (_stream_chunk({}, finish_reason="tool_calls"),),
                False,
                "Tool-call terminal did not contain a structured function call",
            ),
            (
                "missing terminal",
                (_stream_chunk(_tool_delta()),),
                False,
                "Provider stream ended without a finish_reason",
            ),
        )

        for label, chunks, parallel_tool_calls, expected_error in cases:
            with self.subTest(label=label), self.assertRaises(ValueError) as raised:
                await _collect_provider_stream(*chunks, parallel_tool_calls=parallel_tool_calls)
            self.assertEqual(str(raised.exception), expected_error)

    async def test_formatting_only_prefix_is_not_published_for_a_tool_only_response(self) -> None:
        chunks = await _collect_stream(
            _stream_chunk({"role": "assistant", "content": "\n\u2003"}),
            _stream_chunk(_tool_delta()),
            _usage_chunk(),
            _stream_chunk({}, finish_reason="tool_calls"),
        )

        self.assertEqual(_contents(chunks), [])
        self.assertEqual(_tool_names(chunks), ["lookup"])
        self.assertEqual(chunks[0].choices[0].delta.role, "assistant")
        self.assertEqual([chunk.usage.total_tokens for chunk in chunks if chunk.usage], [11])
        self.assertEqual(chunks[-1].choices[0].finish_reason, "tool_calls")

    async def test_leading_formatting_is_preserved_when_semantic_text_follows(self) -> None:
        source = [
            _stream_chunk({"role": "assistant", "content": "\n"}),
            _stream_chunk({"content": "Hello"}),
            _stream_chunk({}, finish_reason="stop"),
        ]

        chunks = await _collect_stream(*source)

        self.assertEqual([chunk.model_dump() for chunk in chunks], [chunk.model_dump() for chunk in source])

    async def test_semantic_preamble_and_leading_formatting_are_preserved_before_a_tool(self) -> None:
        chunks = await _collect_stream(
            _stream_chunk({"role": "assistant", "content": "\n"}),
            _stream_chunk({"content": "I will check that."}),
            _stream_chunk(_tool_delta()),
            _stream_chunk({}, finish_reason="tool_calls"),
        )

        self.assertEqual(_contents(chunks), ["\n", "I will check that."])
        self.assertEqual(_tool_names(chunks), ["lookup"])

    async def test_formatting_before_late_semantic_tool_content_is_preserved(self) -> None:
        chunks = await _collect_stream(
            _stream_chunk({"role": "assistant", "content": "\n"}),
            _stream_chunk(_tool_delta()),
            _stream_chunk({"content": "I found it."}),
            _stream_chunk({}, finish_reason="tool_calls"),
        )

        self.assertEqual(_contents(chunks), ["\n", "I found it."])
        self.assertEqual(_tool_names(chunks), ["lookup"])


class ForcedToolLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_tool_content_is_normalized_without_losing_semantic_text(self) -> None:
        for content, expected in (("\n\n", []), ("\nI will check that.", ["\nI will check that."])):
            with self.subTest(content=content):
                completion = _tool_completion(content)
                raw_chunks = _forced_tool_choice_chunks(completion, completion.choices[0])
                chunks = [chunk async for chunk in _normalize_realtime_assistant_output(raw_chunks)]

                self.assertEqual(_contents(chunks), expected)
                self.assertEqual(_tool_names(chunks), ["lookup"])
