# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The heard log carries the backend's input: one user message, a labeled transcript."""

import asyncio
import unittest

from examples.frontend_backend_live.heard_log import MAX_TRANSCRIPT_CHARS, Heard, HeardLog
from examples.frontend_backend_live.prompts import TranscriptText, load_prompts

PROMPTS = load_prompts().cascade
WORDS = PROMPTS.transcript
TURNS = [f"Turn {n}: " + "words " * 10 for n in range(200)]


class Side:
    """A delegation side: it accepts each request at once or, ``held``, as the test decides."""

    def __init__(self, held=False):
        self.requests: list[list[dict]] = []
        self.arrived = asyncio.Event()
        self.decided = asyncio.get_running_loop().create_future() if held else None

    async def delegate(self, input: list[dict]) -> str:
        self.requests.append(input)
        self.arrived.set()
        return await self.decided if self.decided else f"item_{len(self.requests)}"


def text_of(items):
    return items[0]["content"][0]["text"]


def line(label, text):
    return f"{label}: {text}"


def carried(log):
    return [text for _, text in log.lines()]


class TranscriptTests(unittest.TestCase):
    def test_cut_lines_are_marked_and_unheard_ones_are_gone(self):
        log = HeardLog(PROMPTS)
        log.agent(1, "Your latte is ready at the front counter.")
        log.agent(2, "Anything else?")
        log.agent(3, "Your total is five dollars.", result=True)  # heard in full: the backend holds its own answer
        log.agent(4, "Your order number is BB-42 and it will be ready soon.", result=True)
        log.cut(1, "Your latte is ready")
        log.unsay(2)
        log.cut(4, "Your order number is")  # a cut result is a delivery note
        cut = [line(WORDS.agent, "Your latte is ready" + PROMPTS.cut_mark)]
        self.assertEqual(carried(log), [*cut, line(WORDS.agent, "Your order number is" + PROMPTS.cut_mark)])

    def test_one_message_of_labeled_lines_in_the_prompts_words(self):
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
        log = HeardLog(PROMPTS.model_copy(update={"transcript": words, "cut_mark": " <CUT>"}))
        log.agent(1, "Welcome.")
        log.caller("Teas?")
        log.context("Regular.")
        log.instruction("Be brief.")
        items, _ = log.delegation_input()
        self.assertEqual([item["role"] for item in items], ["user"])
        self.assertEqual(text_of(items), "HEADER\nAGENT: Welcome.\nCALLER: Teas?\nCONTEXT: Regular.\nNOTE: Be brief.")
        log.cut(1, "Wel")
        kinds = [Heard.AGENT, Heard.CALLER, Heard.CONTEXT, Heard.INSTRUCTION]
        texts = ["AGENT: Wel <CUT>", "CALLER: Teas?", "CONTEXT: Regular.", "NOTE: Be brief."]
        self.assertEqual(log.lines(), list(zip(kinds, texts, strict=True)))
        self.assertEqual(text_of(log.delegation_input()[0]).splitlines()[0], "HEADER")
        self.assertEqual(text_of(log.delegation_input(unheard=True)[0]).splitlines()[-1], "UNHEARD")
        for _ in range(200):
            log.caller("words " * 10)
        self.assertEqual(text_of(log.delegation_input()[0]).splitlines()[:2], ["HEADER", "OMITTED"])
        log.sent(log.delegation_input()[1])
        log.caller("Hours?")
        earlier = ["EARLIER", "NOTE: Be brief.", "HEADER", "CALLER: Hours?"]
        self.assertEqual(text_of(log.delegation_input()[0]).splitlines(), earlier)

    def test_the_unheard_line_and_the_cap_on_old_lines(self):
        log = HeardLog(PROMPTS)
        log.agent(1, "Let me check.")
        log.caller("Okay.")
        heard = [WORDS.header, line(WORDS.agent, "Let me check."), line(WORDS.caller, "Okay.")]
        self.assertEqual(text_of(log.delegation_input(unheard=True)[0]), "\n".join([*heard, WORDS.unheard]))
        self.assertEqual(text_of(log.delegation_input()[0]), "\n".join(heard))
        for turn in TURNS:
            log.caller(turn)
        self.assertEqual(text_of(log.delegation_input(unheard=True)[0]).splitlines()[-1], WORDS.unheard)
        text = text_of(log.delegation_input()[0])
        self.assertLessEqual(len(text), MAX_TRANSCRIPT_CHARS + len(WORDS.header) + len(WORDS.omitted) + 100)
        self.assertTrue(text.startswith(f"{WORDS.header}\n{WORDS.omitted}\n"))
        self.assertTrue(text.endswith(line(WORDS.caller, TURNS[-1])))


class DelegationTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_requests_in_flight_remove_what_they_carried(self):
        log, first, second = HeardLog(PROMPTS), Side(held=True), Side(held=True)
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
        self.assertEqual(carried(log), [line(WORDS.agent, "Still working.")])
        log.unsay(2)
        log.agent(3, "Done.")
        first.decided.set_result("item_1")
        await sending_first
        self.assertEqual(carried(log), [line(WORDS.agent, "Done.")])

    async def test_an_instruction_is_carried_outside_the_size_cap(self):
        log, side = HeardLog(PROMPTS), Side()
        log.caller("A latte, please.")
        log.instruction("Reply in Spanish from now on.")
        await log.delegate(side)
        log.agent(1, "Un momento.")
        log.caller("And a muffin.")
        await log.delegate(side)
        instruction = line(WORDS.instruction, "Reply in Spanish from now on.")
        newer = [line(WORDS.agent, "Un momento."), line(WORDS.caller, "And a muffin.")]
        self.assertEqual(text_of(side.requests[1]), "\n".join([WORDS.earlier, instruction, WORDS.header, *newer]))
        plain = HeardLog(PROMPTS)
        for turn in TURNS:
            log.caller(turn)
            plain.caller(turn)
        await log.delegate(side, unheard=True)
        lines = text_of(side.requests[2]).splitlines()
        self.assertEqual(lines[:4], [WORDS.earlier, instruction, WORDS.header, WORDS.omitted])
        self.assertEqual(lines[-1], WORDS.unheard)
        self.assertEqual(lines[4:-1], text_of(plain.delegation_input()[0]).splitlines()[2:])

    async def test_a_refused_request_keeps_everything_noted(self):
        log, side = HeardLog(PROMPTS), Side(held=True)
        log.caller("What teas do you have?")
        log.instruction("Reply in Spanish from now on.")
        sending = asyncio.create_task(log.delegate(side))
        await side.arrived.wait()
        side.decided.set_exception(RuntimeError("Delegation queue is full"))
        with self.assertRaises(RuntimeError):
            await sending
        asked = line(WORDS.caller, "What teas do you have?")
        instruction = line(WORDS.instruction, "Reply in Spanish from now on.")
        self.assertEqual(carried(log), [asked, instruction])
        log.caller("A latte, please.")
        new = [WORDS.header, asked, instruction, line(WORDS.caller, "A latte, please.")]
        self.assertEqual(text_of(log.delegation_input()[0]).splitlines(), new)
