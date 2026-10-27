# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The heard log carries the backend's input: one user message, a labeled transcript."""

import asyncio
import unittest

from examples.frontend_backend_live.cascade.heard_log import HEADER, MAX_TRANSCRIPT_CHARS, OMITTED, HeardLog


class HeldDelegation:
    """A delegation side whose ``delegate`` waits until the test lets it accept, or refuse, the request."""

    def __init__(self):
        self.requests: list[list[dict]] = []
        self.arrived = asyncio.Event()
        self.decided: asyncio.Future = asyncio.get_running_loop().create_future()

    async def delegate(self, input: list[dict]) -> str:
        self.requests.append(input)
        self.arrived.set()
        return await self.decided


def text_of(items):
    return items[0]["content"][0]["text"]


class TranscriptTests(unittest.TestCase):
    def test_one_user_message_with_labeled_lines_in_time_order_and_no_current_request(self):
        log = HeardLog()
        log.agent(1, "Welcome to Bluebird.")
        log.caller("What teas do you have?")
        log.context("The caller is a regular.")
        log.instruction("Answer in Spanish.")
        items, _ = log.delegation_input()
        self.assertEqual([item["role"] for item in items], ["user"])
        self.assertEqual(
            text_of(items),
            "\n".join(
                [
                    HEADER,
                    "Agent: Welcome to Bluebird.",
                    "Caller: What teas do you have?",
                    "App context: The caller is a regular.",
                    "App instruction: Answer in Spanish.",
                ]
            ),
        )
        self.assertNotIn("Current request", text_of(items))

    def test_a_cut_line_is_marked_and_an_unheard_line_is_gone(self):
        log = HeardLog()
        log.agent(1, "Your flight leaves at nine from gate twelve.")
        log.agent(2, "Anything else?")
        log.cut(1, "Your flight leaves at nine")
        log.unsay(2)
        self.assertEqual(log.transcript(), ["Agent: Your flight leaves at nine [cut off]"])

    def test_a_result_heard_in_full_is_left_to_the_backend_and_a_cut_one_is_a_delivery_note(self):
        log = HeardLog()
        log.agent(1, "Your total is five dollars.", result=True)
        log.agent(2, "Your order number is BB-42 and it will be ready soon.", result=True)
        log.cut(2, "Your order number is")
        self.assertEqual(log.transcript(), ["Agent: Your order number is [cut off]"])

    def test_the_oldest_lines_go_past_the_cap(self):
        log = HeardLog()
        for n in range(200):
            log.caller(f"Turn {n}: " + "words " * 10)
        text = text_of(log.delegation_input()[0])
        self.assertLessEqual(len(text), MAX_TRANSCRIPT_CHARS + len(HEADER) + len(OMITTED) + 100)
        self.assertTrue(text.startswith(f"{HEADER}\n{OMITTED}\n"))
        self.assertTrue(text.endswith("Caller: Turn 199: " + "words " * 10))

    def test_the_normalizers_rewrite_replaces_the_newest_caller_turn_noted_with_its_words(self):
        log = HeardLog()
        log.caller("two bags")
        log.agent(1, "Okay.")
        log.caller("two bags")
        log.replace_caller("two bags", "2 bags")
        log.replace_caller("missing", "x")
        self.assertEqual(log.transcript(), ["Caller: two bags", "Agent: Okay.", "Caller: 2 bags"])


class DelegationTests(unittest.IsolatedAsyncioTestCase):
    async def test_words_noted_while_the_request_waits_stay_for_the_next_delegation(self):
        log, side = HeardLog(), HeldDelegation()
        log.agent(1, "Welcome.")
        log.caller("What teas do you have?")
        sending = asyncio.create_task(log.delegate(side))
        await side.arrived.wait()
        log.agent(2, "Let me check.")
        log.context("The cart is empty.")
        side.decided.set_result("item_1")
        self.assertEqual(await sending, "item_1")
        self.assertIn("Caller: What teas do you have?", text_of(side.requests[0]))
        self.assertEqual(log.transcript(), ["Agent: Let me check.", "App context: The cart is empty."])

    async def test_a_refused_request_keeps_everything_noted(self):
        log, side = HeardLog(), HeldDelegation()
        log.caller("What teas do you have?")
        sending = asyncio.create_task(log.delegate(side))
        await side.arrived.wait()
        side.decided.set_exception(RuntimeError("Delegation queue is full"))
        with self.assertRaises(RuntimeError):
            await sending
        self.assertEqual(log.transcript(), ["Caller: What teas do you have?"])

    async def test_two_requests_in_flight_each_remove_only_what_they_carried(self):
        log, first, second = HeardLog(), HeldDelegation(), HeldDelegation()
        log.caller("Add a latte.")
        sending_first = asyncio.create_task(log.delegate(first))
        await first.arrived.wait()
        log.agent(1, "One moment.")
        log.caller("And a muffin.")
        sending_second = asyncio.create_task(log.delegate(second))
        await second.arrived.wait()
        log.agent(2, "Still working.")
        second.decided.set_result("item_2")  # the newer one is accepted first: it carried everything before it
        await sending_second
        log.unsay(2)  # a line nobody heard leaves, and the marks stay exact
        log.agent(3, "Done.")
        first.decided.set_result("item_1")
        await sending_first
        self.assertEqual(log.transcript(), ["Agent: Done."])
