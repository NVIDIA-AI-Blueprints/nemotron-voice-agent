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
    _apply_forced_tool_call_stops,
    _apply_realtime_tool_loop_guard,
    _forced_tool_choice_chunks,
    _hold_streamed_tool_calls_until_terminal,
    _normalize_realtime_assistant_output,
    _strip_raw_tool_call_markup,
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

    async def test_empty_delta_usage_chunk_after_terminal_is_forwarded(self) -> None:
        usage_tail = ChatCompletionChunk.model_validate(
            {
                "id": "chatcmpl_stream",
                "created": 1,
                "model": "test-model",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
            }
        )

        chunks = await _collect_provider_stream(
            _stream_chunk({"role": "assistant", "content": "Hello"}),
            _stream_chunk({}, finish_reason="stop"),
            usage_tail,
        )

        self.assertEqual(_contents(chunks), ["Hello"])
        self.assertEqual([chunk.usage.total_tokens for chunk in chunks if chunk.usage], [11])

    async def test_output_after_terminal_is_rejected(self) -> None:
        cases = (
            ("content", _stream_chunk({"content": "late"})),
            ("second terminal", _stream_chunk({}, finish_reason="stop")),
        )

        for label, late_chunk in cases:
            with self.subTest(label=label), self.assertRaises(ValueError) as raised:
                await _collect_provider_stream(
                    _stream_chunk({"content": "Hello"}),
                    _stream_chunk({}, finish_reason="stop"),
                    late_chunk,
                )
            self.assertEqual(str(raised.exception), "Provider stream emitted choices after its terminal chunk")

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
    def test_profile_tool_terminal_uses_the_standard_stop_field(self) -> None:
        cases = (
            ({}, ("</tool_call>",), "</tool_call>"),
            ({"stop": "END"}, ("</tool_call>",), ["END", "</tool_call>"]),
            ({"stop": ["END", "</tool_call>"]}, ("</tool_call>",), ["END", "</tool_call>"]),
        )
        for params, configured, expected in cases:
            with self.subTest(params=params):
                _apply_forced_tool_call_stops(params, configured)
                self.assertEqual(params["stop"], expected)

    async def test_complete_tool_content_is_normalized_without_losing_semantic_text(self) -> None:
        for content, expected in (("\n\n", []), ("\nI will check that.", ["\nI will check that."])):
            with self.subTest(content=content):
                completion = _tool_completion(content)
                raw_chunks = _forced_tool_choice_chunks(completion, completion.choices[0])
                chunks = [chunk async for chunk in _normalize_realtime_assistant_output(raw_chunks)]

                self.assertEqual(_contents(chunks), expected)
                self.assertEqual(_tool_names(chunks), ["lookup"])


def _assistant_call(name: str, arguments: str, call_id: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}],
    }


def _loop_params(*turns: dict[str, Any], tool_choice: Any = "auto") -> dict[str, Any]:
    params: dict[str, Any] = {
        "messages": [{"role": "system", "content": "You are an agent."}, *turns],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
    }
    if tool_choice is not None:
        params["tool_choice"] = tool_choice
    return params


def _repeated_lookups(count: int, *, arguments: str = '{"id": "A1"}') -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for index in range(count):
        turns.append(_assistant_call("lookup", arguments, f"call_{index}"))
        turns.append({"role": "tool", "tool_call_id": f"call_{index}", "content": '{"status": "not_found"}'})
    return turns


class RealtimeToolLoopGuardTests(unittest.TestCase):
    def test_guard_is_disabled_without_configured_limits(self) -> None:
        params = _loop_params({"role": "user", "content": "Find A1"}, *_repeated_lookups(50))

        reason = _apply_realtime_tool_loop_guard(params, max_identical_tool_calls=None, max_tool_calls_per_turn=None)

        self.assertIsNone(reason)
        self.assertEqual(params["tool_choice"], "auto")

    def test_identical_calls_withhold_tools_at_the_limit(self) -> None:
        for count, expected in ((2, "auto"), (3, "none")):
            with self.subTest(count=count):
                params = _loop_params({"role": "user", "content": "Find A1"}, *_repeated_lookups(count))

                reason = _apply_realtime_tool_loop_guard(
                    params, max_identical_tool_calls=3, max_tool_calls_per_turn=None
                )

                self.assertEqual(params["tool_choice"], expected)
                self.assertEqual(reason is not None, expected == "none")
                self.assertEqual(len(params["tools"]), 1)
                last = params["messages"][-1]
                self.assertEqual(
                    last["role"] == "system" and "Tool calls are paused" in last["content"], expected == "none"
                )

    def test_guard_instruction_does_not_mutate_the_canonical_messages(self) -> None:
        messages = [{"role": "user", "content": "Find A1"}, *_repeated_lookups(3)]
        params = _loop_params(*messages)
        canonical = params["messages"]
        params["parallel_tool_calls"] = True

        _apply_realtime_tool_loop_guard(params, max_identical_tool_calls=3, max_tool_calls_per_turn=None)

        self.assertEqual(len(canonical), len(messages) + 1)
        self.assertEqual(len(params["messages"]), len(messages) + 2)
        self.assertNotIn("parallel_tool_calls", params)

    def test_raw_tool_call_markup_is_removed_from_speakable_text(self) -> None:
        cases = (
            ("<tool_call> <function=lookup> <parameter=id> A1 </parameter> </function> </tool_call>", ""),
            ("I could not find A1. <tool_call> <function=lookup> </function> </tool_call> ", "I could not find A1."),
            ("Please repeat your ID. <tool_call> <function=lookup> <parameter=id> A", "Please repeat your ID."),
            ("Nothing to strip here.", "Nothing to strip here."),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(_strip_raw_tool_call_markup(text), expected)

    def test_argument_key_order_does_not_hide_repetition(self) -> None:
        turns = [
            _assistant_call("lookup", '{"id": "A1", "date": "2024-05-15"}', "call_0"),
            {"role": "tool", "tool_call_id": "call_0", "content": "{}"},
            _assistant_call("lookup", '{"date":"2024-05-15","id":"A1"}', "call_1"),
            {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
        ]
        params = _loop_params({"role": "user", "content": "Find A1"}, *turns)

        reason = _apply_realtime_tool_loop_guard(params, max_identical_tool_calls=2, max_tool_calls_per_turn=None)

        self.assertIsNotNone(reason)
        self.assertEqual(params["tool_choice"], "none")

    def test_varied_calls_are_bounded_only_by_the_per_turn_limit(self) -> None:
        turns: list[dict[str, Any]] = []
        for index in range(6):
            turns.append(_assistant_call("lookup", f'{{"id": "A{index}"}}', f"call_{index}"))
            turns.append({"role": "tool", "tool_call_id": f"call_{index}", "content": "{}"})
        for per_turn, expected in ((None, "auto"), (7, "auto"), (6, "none")):
            with self.subTest(per_turn=per_turn):
                params = _loop_params({"role": "user", "content": "Find it"}, *turns)

                _apply_realtime_tool_loop_guard(params, max_identical_tool_calls=3, max_tool_calls_per_turn=per_turn)

                self.assertEqual(params["tool_choice"], expected)

    def test_new_user_message_resets_the_counters(self) -> None:
        params = _loop_params(
            {"role": "user", "content": "Find A1"},
            *_repeated_lookups(5),
            {"role": "assistant", "content": "I could not find A1."},
            {"role": "user", "content": "Try A1 again"},
            *_repeated_lookups(1),
        )

        reason = _apply_realtime_tool_loop_guard(params, max_identical_tool_calls=3, max_tool_calls_per_turn=4)

        self.assertIsNone(reason)
        self.assertEqual(params["tool_choice"], "auto")

    def test_forced_and_disabled_tool_choices_are_never_overridden(self) -> None:
        named = {"type": "function", "function": {"name": "lookup"}}
        for tool_choice in ("required", named, "none"):
            with self.subTest(tool_choice=tool_choice):
                params = _loop_params(
                    {"role": "user", "content": "Find A1"}, *_repeated_lookups(5), tool_choice=tool_choice
                )

                reason = _apply_realtime_tool_loop_guard(params, max_identical_tool_calls=2, max_tool_calls_per_turn=2)

                self.assertIsNone(reason)
                self.assertEqual(params["tool_choice"], tool_choice)

    def test_unset_tool_choice_is_treated_as_auto(self) -> None:
        params = _loop_params({"role": "user", "content": "Find A1"}, *_repeated_lookups(3), tool_choice=None)

        reason = _apply_realtime_tool_loop_guard(params, max_identical_tool_calls=3, max_tool_calls_per_turn=None)

        self.assertIsNotNone(reason)
        self.assertEqual(params["tool_choice"], "none")

    def test_service_rejects_invalid_limits(self) -> None:
        for field in ("realtime_max_identical_tool_calls", "realtime_max_tool_calls_per_turn"):
            for value in (0, -1, True, 2.5, "3"):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                    NvidiaLLMService(api_key="not-needed", base_url="http://localhost:8000/v1", **{field: value})


def _text_completion(content: str | None, *, finish_reason: str = "stop") -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl_guard",
            "created": 1,
            "model": "test-model",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
        }
    )


class RealtimeToolLoopGuardCompletionTests(unittest.IsolatedAsyncioTestCase):
    async def _guarded_chunks(self, *completions: ChatCompletion) -> tuple[list[ChatCompletionChunk], list[dict]]:
        service = NvidiaLLMService(api_key="not-needed", base_url="http://localhost:8000/v1")
        requests: list[dict] = []
        replies = iter(completions)

        async def fake_request(params: dict) -> ChatCompletion:
            requests.append(dict(params))
            return next(replies)

        service._request_chat_completion = fake_request  # type: ignore[method-assign]
        params = {"stream": True, "stream_options": {"include_usage": True}, "tool_choice": "none"}
        stream = await service._tool_loop_guard_text_chunks(params)
        chunks = [
            chunk async for chunk in NvidiaLLMService._with_provider_completion_reason(_ProviderReasonSink(), stream)
        ]
        return chunks, requests

    async def test_guarded_reply_is_requested_without_streaming_and_published_as_text(self) -> None:
        chunks, requests = await self._guarded_chunks(
            _text_completion("I could not find that ID. Could you repeat it?")
        )

        self.assertEqual(len(requests), 1)
        self.assertIs(requests[0]["stream"], False)
        self.assertNotIn("stream_options", requests[0])
        self.assertEqual(_contents(chunks), ["I could not find that ID. Could you repeat it?"])
        self.assertEqual(chunks[-2].choices[0].finish_reason, "stop")
        self.assertEqual(chunks[-1].usage.total_tokens, 25)

    async def test_markup_only_reply_is_retried_once(self) -> None:
        markup = "<tool_call> <function=lookup> <parameter=id> A1 </parameter> </function> </tool_call>"
        for second, expected in (
            (_text_completion("Could you repeat your ID?"), ["Could you repeat your ID?"]),
            (_text_completion(markup), []),
        ):
            with self.subTest(expected=expected):
                chunks, requests = await self._guarded_chunks(_text_completion(markup), second)

                self.assertEqual(len(requests), 2)
                self.assertEqual(_contents(chunks), expected)
                self.assertEqual(chunks[-2].choices[0].finish_reason, "stop")

    async def test_incomplete_guarded_reply_publishes_only_its_terminal(self) -> None:
        chunks, _ = await self._guarded_chunks(_text_completion("partial", finish_reason="length"))

        self.assertEqual(_contents(chunks), [])
        self.assertEqual(chunks[0].choices[0].finish_reason, "length")
