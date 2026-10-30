# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The heard log carries the backend's input: one user message, a labeled transcript."""

import asyncio
import unittest
from dataclasses import replace

from examples.frontend_backend_live.cascade.heard_log import MAX_TRANSCRIPT_CHARS, Heard, HeardLog
from examples.frontend_backend_live.prompts import TranscriptText, get_prompt_set

PROMPTS = get_prompt_set("default")
WORDS = PROMPTS.transcript
CUT = PROMPTS.cut_mark


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


class AcceptingDelegation:
    """A delegation side that accepts every request at once."""

    def __init__(self):
        self.requests: list[list[dict]] = []

    async def delegate(self, input: list[dict]) -> str:
        self.requests.append(input)
        return f"item_{len(self.requests)}"


def text_of(items):
    return items[0]["content"][0]["text"]


def line(label, text):
    """A transcript line as the log writes it: its label, then its text."""
    return f"{label}: {text}"


class TranscriptTests(unittest.TestCase):
    def test_one_user_message_with_labeled_lines_in_time_order(self):
        log = HeardLog(PROMPTS)
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
                    WORDS.header,
                    line(WORDS.agent, "Welcome to Bluebird."),
                    line(WORDS.caller, "What teas do you have?"),
                    line(WORDS.context, "The caller is a regular."),
                    line(WORDS.instruction, "Answer in Spanish."),
                ]
            ),
        )

    def test_a_cut_line_is_marked_and_an_unheard_line_is_gone(self):
        log = HeardLog(PROMPTS)
        log.agent(1, "Your flight leaves at nine from gate twelve.")
        log.agent(2, "Anything else?")
        log.cut(1, "Your flight leaves at nine")
        log.unsay(2)
        self.assertEqual(log.transcript(), [line(WORDS.agent, "Your flight leaves at nine" + CUT)])

    def test_a_result_heard_in_full_is_left_to_the_backend_and_a_cut_one_is_a_delivery_note(self):
        log = HeardLog(PROMPTS)
        log.agent(1, "Your total is five dollars.", result=True)
        log.agent(2, "Your order number is BB-42 and it will be ready soon.", result=True)
        log.cut(2, "Your order number is")
        self.assertEqual(log.transcript(), [line(WORDS.agent, "Your order number is" + CUT)])

    def test_the_oldest_lines_go_past_the_cap(self):
        log = HeardLog(PROMPTS)
        for n in range(200):
            log.caller(f"Turn {n}: " + "words " * 10)
        text = text_of(log.delegation_input()[0])
        self.assertLessEqual(len(text), MAX_TRANSCRIPT_CHARS + len(WORDS.header) + len(WORDS.omitted) + 100)
        self.assertTrue(text.startswith(f"{WORDS.header}\n{WORDS.omitted}\n"))
        self.assertTrue(text.endswith(line(WORDS.caller, "Turn 199: " + "words " * 10)))

    def test_every_word_is_the_prompt_sets_and_each_line_keeps_its_kind(self):
        words = TranscriptText(
            header="HEADER",
            omitted="OMITTED",
            caller="CALLER",
            agent="AGENT",
            context="CONTEXT",
            instruction="NOTE",
            earlier="EARLIER",
            unheard="UNHEARD",
        )
        log = HeardLog(replace(PROMPTS, transcript=words, cut_mark=" <CUT>"))
        log.agent(1, "Welcome.")
        log.caller("Teas?")
        log.context("Regular.")
        log.instruction("Be brief.")
        log.cut(1, "Wel")
        self.assertEqual(
            log.lines(),
            [
                (Heard.AGENT, "AGENT: Wel <CUT>"),
                (Heard.CALLER, "CALLER: Teas?"),
                (Heard.CONTEXT, "CONTEXT: Regular."),
                (Heard.INSTRUCTION, "NOTE: Be brief."),
            ],
        )
        self.assertEqual(text_of(log.delegation_input()[0]).splitlines()[0], "HEADER")
        self.assertEqual(text_of(log.delegation_input(unheard=True)[0]).splitlines()[-1], "UNHEARD")
        for _ in range(200):
            log.caller("words " * 10)
        self.assertEqual(text_of(log.delegation_input()[0]).splitlines()[:2], ["HEADER", "OMITTED"])
        log.sent(log.delegation_input()[1])  # accepted: the instruction is an earlier one from now on
        log.caller("Hours?")
        self.assertEqual(
            text_of(log.delegation_input()[0]).splitlines(), ["EARLIER", "NOTE: Be brief.", "HEADER", "CALLER: Hours?"]
        )

    def test_the_unheard_line_ends_the_input_only_while_an_answer_waits_to_be_heard(self):
        log = HeardLog(PROMPTS)
        log.agent(1, "Let me check.")
        log.caller("Okay.")
        heard = [WORDS.header, line(WORDS.agent, "Let me check."), line(WORDS.caller, "Okay.")]
        self.assertEqual(text_of(log.delegation_input(unheard=True)[0]), "\n".join([*heard, WORDS.unheard]))
        self.assertEqual(text_of(log.delegation_input()[0]), "\n".join(heard))
        for n in range(200):  # past the cap the line still ends the input
            log.caller(f"Turn {n}: " + "words " * 10)
        self.assertEqual(text_of(log.delegation_input(unheard=True)[0]).splitlines()[-1], WORDS.unheard)

    def test_the_normalizers_rewrite_replaces_the_newest_caller_turn_noted_with_its_words(self):
        log = HeardLog(PROMPTS)
        log.caller("two bags")
        log.agent(1, "Okay.")
        log.caller("two bags")
        log.replace_caller("two bags", "2 bags")
        log.replace_caller("missing", "x")
        self.assertEqual(
            log.transcript(), [line(WORDS.caller, "two bags"), line(WORDS.agent, "Okay."), line(WORDS.caller, "2 bags")]
        )


class DelegationTests(unittest.IsolatedAsyncioTestCase):
    async def test_words_noted_while_the_request_waits_stay_for_the_next_delegation(self):
        log, side = HeardLog(PROMPTS), HeldDelegation()
        log.agent(1, "Welcome.")
        log.caller("What teas do you have?")
        sending = asyncio.create_task(log.delegate(side))
        await side.arrived.wait()
        log.agent(2, "Let me check.")
        log.context("The cart is empty.")
        side.decided.set_result("item_1")
        self.assertEqual(await sending, "item_1")
        self.assertIn(line(WORDS.caller, "What teas do you have?"), text_of(side.requests[0]))
        self.assertEqual(
            log.transcript(), [line(WORDS.agent, "Let me check."), line(WORDS.context, "The cart is empty.")]
        )

    async def test_a_refused_request_keeps_everything_noted(self):
        log, side = HeardLog(PROMPTS), HeldDelegation()
        log.caller("What teas do you have?")
        sending = asyncio.create_task(log.delegate(side))
        await side.arrived.wait()
        side.decided.set_exception(RuntimeError("Delegation queue is full"))
        with self.assertRaises(RuntimeError):
            await sending
        self.assertEqual(log.transcript(), [line(WORDS.caller, "What teas do you have?")])

    async def test_two_requests_in_flight_each_remove_only_what_they_carried(self):
        log, first, second = HeardLog(PROMPTS), HeldDelegation(), HeldDelegation()
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
        self.assertEqual(log.transcript(), [line(WORDS.agent, "Done.")])

    async def test_every_later_delegation_carries_an_instruction_outside_the_size_cap(self):
        log, side = HeardLog(PROMPTS), AcceptingDelegation()
        log.caller("A latte, please.")
        log.instruction("Reply in Spanish from now on.")
        await log.delegate(side)
        log.agent(1, "Un momento.")
        log.caller("And a muffin.")
        await log.delegate(side)
        instruction = line(WORDS.instruction, "Reply in Spanish from now on.")
        newer = [line(WORDS.agent, "Un momento."), line(WORDS.caller, "And a muffin.")]
        self.assertEqual(text_of(side.requests[1]), "\n".join([WORDS.earlier, instruction, WORDS.header, *newer]))
        plain = HeardLog(PROMPTS)  # the same turns with no instruction
        for n in range(200):
            log.caller(f"Turn {n}: " + "words " * 10)
            plain.caller(f"Turn {n}: " + "words " * 10)
        await log.delegate(side, unheard=True)
        lines = text_of(side.requests[2]).splitlines()
        self.assertEqual(lines[:4], [WORDS.earlier, instruction, WORDS.header, WORDS.omitted])
        self.assertEqual(lines[-1], WORDS.unheard)
        # The cap keeps as many turns as with no instruction: the instruction is outside it.
        self.assertEqual(lines[4:-1], text_of(plain.delegation_input()[0]).splitlines()[2:])

    async def test_an_instruction_only_a_refused_request_carried_is_still_new(self):
        log, side = HeardLog(PROMPTS), HeldDelegation()
        log.instruction("Reply in Spanish from now on.")
        sending = asyncio.create_task(log.delegate(side))
        await side.arrived.wait()
        side.decided.set_exception(RuntimeError("Delegation queue is full"))
        with self.assertRaises(RuntimeError):
            await sending
        log.caller("A latte, please.")
        instruction = line(WORDS.instruction, "Reply in Spanish from now on.")
        self.assertEqual(
            text_of(log.delegation_input()[0]).splitlines(),
            [WORDS.header, instruction, line(WORDS.caller, "A latte, please.")],
        )
