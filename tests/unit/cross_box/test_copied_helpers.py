# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The helpers each box keeps its own copy of, since the boxes share no module, behave alike in every copy."""

import asyncio
import json
import unittest
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from unittest.mock import patch

import httpx
from openai import APIConnectionError, RateLimitError

from delegation_core import ids as core_ids
from delegation_core import tables as core_tables
from delegation_core.backend.providers import retry as core_retry
from delegation_core.backend.providers import translation as core_translation
from examples.frontend_backend_live import tables as frontend_tables
from examples.frontend_backend_live.cascade import memory as frontend_memory
from examples.frontend_backend_live.models import chat_completions as frontend_chat
from examples.frontend_backend_live.models import retry as frontend_retry
from live.protocol import errors as gateway_errors

PICKS = {"core": core_tables.pick, "frontend": frontend_tables.pick}
STRIPS = {"core": core_translation.strip_reasoning, "frontend": frontend_chat.strip_reasoning}
ESTIMATES = {"core": core_ids.approx_tokens, "gateway": gateway_errors.approx_tokens}
RETRIES = {"core": core_retry, "frontend": frontend_retry}
REQUEST = httpx.Request("POST", "http://localhost:1/v1/chat/completions")


def too_many_requests(retry_after: str | None = None) -> RateLimitError:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    return RateLimitError("slow down", response=httpx.Response(429, headers=headers, request=REQUEST), body=None)


class PickTests(unittest.TestCase):
    def test_every_copy_picks_by_name_and_names_what_is_known(self):
        table = {"llm": 1, "harness": 2}
        for box, pick in PICKS.items():
            with self.subTest(box):
                self.assertEqual(pick(table, "harness", "backend adapter"), 2)
                with self.assertRaisesRegex(ValueError, r"^Unknown backend adapter 'hermes'; known: harness, llm$"):
                    pick(table, "hermes", "backend adapter")
                with self.assertRaisesRegex(ValueError, r"known: none$"):
                    pick({}, "llm", "backend adapter")
                with self.assertRaisesRegex(ValueError, r"^Unknown backend adapter \['llm'\]"):
                    pick(table, ["llm"], "backend adapter")  # an unhashable name is refused the same way


class ReasoningTests(unittest.TestCase):
    CASES = {
        "<think>check the menu</think>We have tea.": "We have tea.",
        "Plain answer.": "Plain answer.",
        "We have <think>a thought</think>tea.": "We have tea.",
        "Partial <think>cut off before it closed": "Partial",
        "<think>first</think><think>second</think>Answer.": "Answer.",
    }

    def test_every_copy_drops_hidden_reasoning_alike(self):
        for box, strip in STRIPS.items():
            for text, answer in self.CASES.items():
                with self.subTest(box=box, text=text):
                    self.assertEqual(strip(text), answer)


class TokenEstimateTests(unittest.TestCase):
    def test_every_copy_counts_about_four_characters_a_token(self):
        for box, estimate in ESTIMATES.items():
            with self.subTest(box):
                self.assertEqual([estimate(text) for text in ("", "a", "abcd", "abcde")], [0, 1, 1, 2])

    def test_the_talker_memorys_estimate_of_a_message_is_the_same_rule_on_its_json(self):
        message = {"role": "user", "content": "What teas do you have?"}
        text = json.dumps(message, ensure_ascii=False)
        self.assertEqual(frontend_memory.approx_tokens(message), core_ids.approx_tokens(text))


class RetryTests(unittest.IsolatedAsyncioTestCase):
    def test_every_copy_honors_a_short_retry_after_and_caps_a_long_one(self):
        later = format_datetime(datetime.now(UTC) - timedelta(seconds=30), usegmt=True)
        for box, module in RETRIES.items():
            with self.subTest(box):
                self.assertEqual(module.retry_delay(too_many_requests("2"), 1), 2.0)
                self.assertEqual(module.retry_delay(too_many_requests("120"), 1), module.MAX_RETRY_AFTER_SECONDS)
                self.assertEqual(module.retry_delay(too_many_requests(later), 1), 0.0)  # a date already past
                self.assertEqual(module.MAX_RETRY_AFTER_SECONDS, 3.0)

    def test_every_copy_backs_off_with_jitter_without_a_retry_after(self):
        for box, module in RETRIES.items():
            with self.subTest(box):
                first = [module.retry_delay(too_many_requests(), 1) for _ in range(50)]
                later = [module.retry_delay(too_many_requests(), 2) for _ in range(50)]
                self.assertTrue(all(0.3 <= delay <= 0.5 for delay in first), first)
                self.assertTrue(all(0.9 <= delay <= 1.5 for delay in later), later)

    async def test_every_copy_tries_a_transient_failure_again_and_raises_anything_else_at_once(self):
        for box, module in RETRIES.items():
            with self.subTest(box), patch.object(module, "retry_delay", lambda exc, attempt: 0.0):
                await self.check_retries(module)

    async def check_retries(self, module):
        calls = []

        async def flaky(errors):
            calls.append(1)
            if errors:
                raise errors.pop(0)
            return "answer"

        connection = [APIConnectionError(request=REQUEST)]
        self.assertEqual(await module.with_retries(lambda: flaky(connection), attempts=2, role="test"), "answer")
        self.assertEqual(len(calls), 2)
        calls.clear()
        with self.assertRaises(ValueError):  # not transient: no second attempt
            await module.with_retries(lambda: flaky([ValueError("bad request")]), attempts=3, role="test")
        self.assertEqual(len(calls), 1)
        calls.clear()
        limited = [too_many_requests(), too_many_requests()]
        with self.assertRaises(RateLimitError):  # transient every time: the last error, after every attempt
            await module.with_retries(lambda: flaky(limited), attempts=2, role="test")
        self.assertEqual(len(calls), 2)
        with self.assertRaises(ValueError):
            await module.with_retries(lambda: asyncio.sleep(0), attempts=0, role="test")


if __name__ == "__main__":
    unittest.main()
