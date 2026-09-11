# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import unittest

from openai import NOT_GIVEN
from pipecat.frames.frames import LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext

from examples.multilingual.multilingual_processor import (
    PerTurnReminderProcessor,
    build_reminder,
    describe_language,
    with_reasoning,
)
from realtime.frames import RealtimeResponseContextFrame, RealtimeResponseLLMContext


class DescribeLanguageTests(unittest.TestCase):
    def test_maps_known_subtag_to_name(self) -> None:
        self.assertEqual(describe_language("de-DE"), "German (Deutsch)")

    def test_falls_back_to_raw_code_for_unknown(self) -> None:
        self.assertEqual(describe_language("xx-YY"), "xx-YY")

    def test_empty_code_returns_empty(self) -> None:
        self.assertEqual(describe_language(""), "")


class BuildReminderTests(unittest.TestCase):
    def test_names_language_and_forbids_mixing(self) -> None:
        reminder = build_reminder("hi-IN")
        self.assertIn("Hindi", reminder)
        self.assertIn("ONE single language", reminder)
        self.assertNotIn("JSON", reminder)


class WithReasoningTests(unittest.TestCase):
    def test_enables_thinking_without_mutating_input(self) -> None:
        base = {"extra_body": {"repetition_penalty": 1.05}}
        merged = with_reasoning(base, True)
        self.assertTrue(merged["extra_body"]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(merged["extra_body"]["repetition_penalty"], 1.05)
        self.assertNotIn("chat_template_kwargs", base["extra_body"])

    def test_disables_thinking_on_empty_input(self) -> None:
        merged = with_reasoning({}, False)
        self.assertFalse(merged["extra_body"]["chat_template_kwargs"]["enable_thinking"])


class PerTurnReminderProcessorTests(unittest.TestCase):
    def test_appends_reminder_to_last_user_message(self) -> None:
        processor = PerTurnReminderProcessor("REMINDER")
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        result = processor._append_reminder([dict(msg) for msg in messages])
        self.assertEqual(result[-1]["content"], "hello\n\nREMINDER")
        self.assertEqual(result[0]["content"], "sys")

    def test_appends_user_message_when_none_present(self) -> None:
        processor = PerTurnReminderProcessor("REMINDER")
        result = processor._append_reminder([{"role": "system", "content": "sys"}])
        self.assertEqual(result[-1], {"role": "user", "content": "REMINDER"})

    def test_preserves_realtime_response_context_and_frame_ownership(self) -> None:
        processor = PerTurnReminderProcessor("REMINDER", realtime=True)
        canonical = LLMContext([{"role": "user", "content": "canonical"}])
        source = RealtimeResponseLLMContext(
            [{"role": "user", "content": "response input"}],
            max_output_tokens=64,
            parallel_tool_calls=False,
            response_id="resp_123",
            run_owner_id="run_123",
            activation_generation=4,
        )
        frame = RealtimeResponseContextFrame(
            context=source,
            response_id="resp_123",
            canonical_context=canonical,
        )
        frame.pts = 123
        frame.metadata = {"owner": "realtime"}

        reminded = processor._reminded_frame(frame)

        self.assertIsInstance(reminded, RealtimeResponseContextFrame)
        self.assertEqual(reminded.response_id, "resp_123")
        self.assertIs(reminded.canonical_context, canonical)
        self.assertIsInstance(reminded.context, RealtimeResponseLLMContext)
        self.assertEqual(reminded.context.max_output_tokens, 64)
        self.assertIs(reminded.context.parallel_tool_calls, False)
        self.assertEqual(reminded.context.response_id, "resp_123")
        self.assertEqual(reminded.context.run_owner_id, "run_123")
        self.assertEqual(reminded.context.activation_generation, 4)
        self.assertEqual(reminded.context.get_messages()[-1]["content"], "response input\n\nREMINDER")
        self.assertEqual(source.get_messages()[-1]["content"], "response input")
        self.assertEqual(reminded.pts, 123)
        self.assertEqual(reminded.metadata, {"owner": "realtime"})
        self.assertIsNot(reminded.metadata, frame.metadata)

    def test_preserves_unowned_realtime_context_controls(self) -> None:
        processor = PerTurnReminderProcessor("REMINDER", realtime=True)
        source = RealtimeResponseLLMContext(
            [{"role": "user", "content": "tool result"}],
            max_output_tokens="inf",
            parallel_tool_calls=True,
            run_owner_id="run_tool_result",
            activation_generation=7,
        )

        reminded = processor._reminded_frame(LLMContextFrame(context=source))

        self.assertIs(type(reminded), LLMContextFrame)
        self.assertIsInstance(reminded.context, RealtimeResponseLLMContext)
        self.assertEqual(reminded.context.max_output_tokens, "inf")
        self.assertIs(reminded.context.parallel_tool_calls, True)
        self.assertEqual(reminded.context.run_owner_id, "run_tool_result")
        self.assertEqual(reminded.context.activation_generation, 7)

    def test_preserves_arbitrary_context_subtype_without_mutating_source(self) -> None:
        class ExtendedContext(LLMContext):
            def __init__(self, messages: list[dict], *, trace_id: str) -> None:
                super().__init__(messages)
                self.trace_id = trace_id

        processor = PerTurnReminderProcessor("REMINDER")
        source = ExtendedContext(
            [{"role": "user", "content": "extended"}],
            trace_id="trace-123",
        )

        reminded = processor._reminded_frame(LLMContextFrame(context=source))

        self.assertIsInstance(reminded.context, ExtendedContext)
        self.assertEqual(reminded.context.trace_id, "trace-123")
        self.assertIs(reminded.context.tools, NOT_GIVEN)
        self.assertIs(reminded.context.tool_choice, NOT_GIVEN)
        self.assertEqual(reminded.context.get_messages()[-1]["content"], "extended\n\nREMINDER")
        self.assertEqual(source.get_messages(), [{"role": "user", "content": "extended"}])


if __name__ == "__main__":
    unittest.main()
