# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Contracts for recovering tool calls a model emitted as literal text."""

# Test names describe the contract; separate public API docstrings add no value here.
# ruff: noqa: D101, D102

from __future__ import annotations

import json
import unittest
from collections.abc import AsyncIterator

from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.completion_usage import CompletionUsage

from examples.frontend_backend_agent.src.reliable_talker import _collect_stream
from examples.shared.text_tool_calls import harvest_text_tool_calls, parse_text_tool_calls

XML_CALL = (
    "<tool_call>\n"
    "<function=call_backend>\n"
    "<parameter=query>\nGet the details of reservation ABC123.\n</parameter>\n"
    "<parameter=filler_text>\nLet me look up reservation ABC123.\n</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


def _chunk(content=None, tool_calls=None, usage=None):
    return ChatCompletionChunk(
        id="chunk",
        object="chat.completion.chunk",
        created=0,
        model="test",
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(role="assistant", content=content, tool_calls=tool_calls),
                finish_reason=None,
            )
        ],
        usage=usage,
    )


def _streamed(text: str) -> list[ChatCompletionChunk]:
    """Split ``text`` across chunks the way a streaming endpoint would."""
    return [_chunk(content=piece) for piece in (text[i : i + 7] for i in range(0, len(text), 7))]


async def _async_stream(chunks: list[ChatCompletionChunk]) -> AsyncIterator[ChatCompletionChunk]:
    for chunk in chunks:
        yield chunk


class ParseTextToolCallsTests(unittest.TestCase):
    def test_parses_the_native_xml_form(self):
        self.assertEqual(
            parse_text_tool_calls(XML_CALL),
            [
                (
                    "call_backend",
                    {
                        "query": "Get the details of reservation ABC123.",
                        "filler_text": "Let me look up reservation ABC123.",
                    },
                )
            ],
        )

    def test_parses_the_json_form(self):
        text = '<tool_call>{"name": "get_reservation_details", "arguments": {"reservation_id": "ABC123"}}</tool_call>'
        self.assertEqual(parse_text_tool_calls(text), [("get_reservation_details", {"reservation_id": "ABC123"})])

    def test_keeps_ambiguous_values_as_text_and_decodes_typed_ones(self):
        text = (
            "<tool_call><function=book>"
            "<parameter=code>ABC123</parameter>"
            "<parameter=seats>3</parameter>"
            "<parameter=insured>true</parameter>"
            "<parameter=legs>[1, 2]</parameter>"
            "</function></tool_call>"
        )
        self.assertEqual(
            parse_text_tool_calls(text),
            [("book", {"code": "ABC123", "seats": 3, "insured": True, "legs": [1, 2]})],
        )

    def test_ignores_prose_without_a_tool_call(self):
        self.assertEqual(parse_text_tool_calls("Let me look that up for you."), [])


class HarvestTextToolCallsTests(unittest.TestCase):
    def test_rewrites_a_streamed_text_call_into_structured_deltas(self):
        harvested = harvest_text_tool_calls(_streamed(XML_CALL))
        tool_calls = harvested[0].choices[0].delta.tool_calls
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].function.name, "call_backend")
        self.assertEqual(
            json.loads(tool_calls[0].function.arguments),
            {
                "query": "Get the details of reservation ABC123.",
                "filler_text": "Let me look up reservation ABC123.",
            },
        )
        self.assertIsNone(harvested[0].choices[0].delta.content)
        self.assertIsNone(harvested[0].choices[0].finish_reason)
        self.assertEqual(harvested[1].choices[0].finish_reason, "tool_calls")

    def test_emits_one_delta_per_parsed_call_before_the_terminal(self):
        second_call = XML_CALL.replace("call_backend", "cancel_backend").replace(
            "Get the details of reservation ABC123.",
            "Cancel reservation ABC123.",
        )

        harvested = harvest_text_tool_calls(_streamed(XML_CALL + second_call))
        call_chunks = harvested[:-1]

        self.assertEqual(len(call_chunks), 2)
        self.assertTrue(all(len(chunk.choices[0].delta.tool_calls) == 1 for chunk in call_chunks))
        self.assertEqual(
            [chunk.choices[0].delta.tool_calls[0].function.name for chunk in call_chunks],
            ["call_backend", "cancel_backend"],
        )
        self.assertEqual(harvested[-1].choices[0].finish_reason, "tool_calls")

    def test_leaves_structured_tool_calls_untouched(self):
        native = [
            _chunk(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        id="call-1",
                        type="function",
                        function=ChoiceDeltaToolCallFunction(name="call_backend", arguments="{}"),
                    )
                ]
            )
        ]
        self.assertIs(harvest_text_tool_calls(native), native)

    def test_leaves_plain_speech_untouched(self):
        speech = _streamed("Reservation ABC123 is confirmed.")
        self.assertIs(harvest_text_tool_calls(speech), speech)

    def test_preserves_usage_chunks_so_token_metrics_survive(self):
        usage = CompletionUsage(prompt_tokens=11, completion_tokens=3, total_tokens=14)
        harvested = harvest_text_tool_calls([*_streamed(XML_CALL), _chunk(usage=usage)])
        self.assertEqual([chunk.usage for chunk in harvested if chunk.usage is not None], [usage])

    def test_handles_an_empty_stream(self):
        self.assertEqual(harvest_text_tool_calls([]), [])


class CollectTextToolCallsTests(unittest.IsolatedAsyncioTestCase):
    async def test_synthesized_call_is_observed_once_after_harvesting(self):
        observed: list[ChatCompletionChunk] = []

        async def observe(chunk: ChatCompletionChunk) -> None:
            observed.append(chunk)

        harvested = await _collect_stream(_async_stream(_streamed(XML_CALL)), observe)
        synthesized = [chunk for chunk in observed if chunk.choices[0].delta.tool_calls]

        self.assertEqual(len(synthesized), 1)
        self.assertEqual(
            synthesized[0].choices[0].delta.tool_calls[0].id,
            harvested[0].choices[0].delta.tool_calls[0].id,
        )


if __name__ == "__main__":
    unittest.main()
