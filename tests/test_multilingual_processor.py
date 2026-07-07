# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import unittest

from examples.multilingual.multilingual_processor import (
    PerTurnReminderProcessor,
    build_reminder,
    describe_language,
    with_reasoning,
)


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


class PerTurnReminderProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def test_appends_reminder_to_last_user_message(self) -> None:
        processor = PerTurnReminderProcessor("REMINDER")
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        result = processor._append_reminder([dict(msg) for msg in messages])
        self.assertEqual(result[-1]["content"], "hello\n\nREMINDER")
        self.assertEqual(result[0]["content"], "sys")

    async def test_appends_user_message_when_none_present(self) -> None:
        processor = PerTurnReminderProcessor("REMINDER")
        result = processor._append_reminder([{"role": "system", "content": "sys"}])
        self.assertEqual(result[-1], {"role": "user", "content": "REMINDER"})


if __name__ == "__main__":
    unittest.main()
