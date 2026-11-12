# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103
"""The questions block: every request after the first lists the backend's newest questions with their counts."""

import asyncio
import json
import unittest

from examples.frontend_backend_live.delegation_core.backend.port import Start, Steer
from examples.frontend_backend_live.delegation_core.config import QuestionsSettings
from examples.frontend_backend_live.delegation_core.im.conversation import AskedQuestions
from examples.shared.delegation_contract import ReplyUnheard
from tests.unit.delegation_core.helpers import (
    MENU,
    TEXTS,
    IMCase,
    ScriptedAdapter,
    answer,
    app_call,
    ask,
    placing,
    planned,
    read_all,
    wait,
)

APP = {"tools": [MENU]}
HEADER = TEXTS.questions_block.split("{")[0]


def block_lines(history: list[dict]) -> list[dict]:
    """The JSON lines of the newest questions block in a task's history; [] when there is none."""
    texts = [
        part["text"]
        for item in history
        if item.get("type") == "message" and item.get("role") == "user"
        for part in item.get("content") or []
        if isinstance(part, dict) and part.get("text", "").startswith(HEADER)
    ]
    return [json.loads(line) for line in texts[-1][len(HEADER) :].splitlines() if line] if texts else []


class QuestionsBlockTests(IMCase):
    async def test_the_questions_block_lists_the_newest_with_their_counts(self):
        scripts = [[answer(text)] for text in ("Is the address on file still correct?", "Anything else?")]
        scripts.append([answer("Is the address on file still correct?")])
        adapter = ScriptedAdapter(*scripts, [answer()])
        im = self.session(adapter, questions=QuestionsSettings(shown=2))
        for number, level in ((1, None), (2, ReplyUnheard.NONE), (3, ReplyUnheard.QUESTION), (4, ReplyUnheard.NONE)):
            stream = im.admit(ask("Hm."), delegation_id=f"D{number}", reply_unheard=level).stream
            await read_all(stream)
        self.assertEqual(block_lines(adapter.runs[0].histories[0]), [])  # the first request: no block
        expected = [
            {"question": "Anything else?", "asked": 1},
            {"question": "Is the address on file still correct?", "asked": 2, "heard_in_full": 2},
        ]
        self.assertEqual(block_lines(adapter.runs[3].histories[0]), expected)

    async def test_a_continuation_does_not_credit_a_reply(self):
        adapter = ScriptedAdapter([answer("Would you like the first option or the second?")], [app_call(), answer()])
        im = self.session(adapter)
        await read_all(im.admit(ask("Options?"), delegation_id="D1", reply_unheard=ReplyUnheard.NONE).stream)
        events = await read_all(im.admit(ask("The menu."), delegation_id="D2", settings=APP).stream)
        await self.answer_calls(im, events, delegation_id="D2", settings=APP)
        (entry,) = im.questions.entries.values()
        self.assertEqual(entry.heard_in_full, None)  # D2 stated no fact; its continuation is no reply

    async def test_a_steering_batch_carries_the_questions_block(self):
        gate = asyncio.Event()
        conductor = planned(placing(Start(("D1",))), placing(Start(("D2",))), placing(Steer("W2", ("D3",))))
        adapter = ScriptedAdapter([answer("Anything else?")], [wait(gate), answer("Done.")], conductor=conductor)
        im = self.session(adapter)
        await read_all(im.admit(ask("Hi."), delegation_id="D1").stream)
        d2 = im.admit(ask("A latte."), delegation_id="D2")
        await self.until(lambda: len(adapter.runs) == 2, "the second Work")
        im.admit(ask("Make it oat."), delegation_id="D3")
        work = im.works["W2"]
        await self.until(lambda: work.batches, "the steering batch")
        items = work.steering_items(work.batches[0])  # what the task reads when the batch lands
        gate.set()
        await read_all(d2.stream)
        self.assertEqual(block_lines(items), [{"question": "Anything else?", "asked": 1}])


class RepliedTests(unittest.TestCase):
    def test_a_question_level_adds_nothing_to_heard_in_full(self):
        questions = AskedQuestions()
        questions.answer_written("Is the address on file still correct?")
        self.assertEqual(questions.replied(ReplyUnheard.NONE), 1)
        questions.answer_written("Is the address on file still correct?")
        self.assertEqual(questions.replied(ReplyUnheard.QUESTION), 1)
        (entry,) = questions.entries.values()
        self.assertEqual((entry.asked, entry.heard_in_full, entry.awaiting), (2, 1, False))

    def test_no_fact_leaves_heard_in_full_unknown(self):
        questions = AskedQuestions()
        questions.answer_written("Anything else?")
        questions.replied(None)
        (entry,) = questions.entries.values()
        self.assertEqual((entry.heard_in_full, entry.awaiting), (None, False))
