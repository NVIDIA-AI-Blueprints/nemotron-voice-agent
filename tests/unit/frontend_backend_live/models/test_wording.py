# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""One wording for every ready result: ``WriterWording`` on the ``ResultWording`` port."""

import asyncio
import inspect
import unittest

from examples.frontend_backend_live.cascade.ports import ReadyResult, ResultWording, WordingNotes
from examples.frontend_backend_live.cascade.writer.wording import WriterWording, as_written
from examples.frontend_backend_live.cascade.writer.writer import CommentaryWriter
from examples.frontend_backend_live.models.decision import TalkerDecision
from examples.frontend_backend_live.prompts import get_prompt_set
from tests.unit.cascade_helpers import ScriptedFrontend

V2 = get_prompt_set("v2")
INSTRUCTIONS = "You are Remy."
HISTORY = [{"role": "user", "content": "A latte and a croissant, please."}, {"role": "assistant", "content": "Sure."}]
LATTE = ReadyResult("item_d1", "msg_1", "Added one large Latte ($5.50).")
CROISSANT = ReadyResult("item_d2", "msg_2", "Added one Butter Croissant ($3.75). Total: $9.25.")
NO_NOTES = WordingNotes(told={}, newer_requests=())


class Recorder:
    def __init__(self):
        self.events = []

    def __call__(self, kind, **fields):
        self.events.append((kind, fields))


def wording(script, *, timeout_seconds=1.0, prompts=V2):
    frontend = ScriptedFrontend(script)
    observe = Recorder()
    words = WriterWording(
        CommentaryWriter(frontend, prompts),
        lambda: (INSTRUCTIONS, list(HISTORY)),
        timeout_seconds=timeout_seconds,
        observe=observe,
    )
    return words, frontend, observe


class OneCallTests(unittest.IsolatedAsyncioTestCase):
    def test_it_has_the_ports_one_member_with_its_signature(self):
        port = inspect.signature(ResultWording.words)
        self.assertEqual(inspect.signature(WriterWording.words).parameters.keys(), port.parameters.keys())

    async def test_one_result_sends_the_request_the_presenter_sent(self):
        words, frontend, _ = wording([TalkerDecision("speak", "Your latte is in.")])
        notes = WordingNotes(told={"item_d1": "Let me add that."}, newer_requests=())
        self.assertEqual(await words.words([LATTE], notes), "Your latte is in.")
        (call,) = frontend.calls
        self.assertEqual(call["instructions"], INSTRUCTIONS + "\n" + V2.commentary + V2.answer)
        self.assertEqual(call["history"][:-1], HISTORY)
        self.assertEqual(
            call["history"][-1]["content"],
            "Commentary update (reference data):\nAdded one large Latte ($5.50)."
            '\n(you already told the caller: "Let me add that." -- do not repeat that, report only the result)',
        )
        self.assertEqual((call["actions"], call["role"]), (("speak",), "writer"))

    async def test_several_results_make_one_call_and_one_line_in_order(self):
        words, frontend, _ = wording([TalkerDecision("speak", "Both are in; your total is nine twenty-five.")])
        notes = WordingNotes(
            told={"item_d1": "Let me add that.", "item_d2": "And the croissant.", "item_d9": "Unrelated."},
            newer_requests=("Make it a large.",),
        )
        line = await words.words([LATTE, CROISSANT], notes)
        self.assertEqual(line, "Both are in; your total is nine twenty-five.")
        self.assertEqual(len(frontend.calls), 1)
        update = frontend.calls[0]["history"][-1]["content"]
        self.assertTrue(
            update.startswith("Commentary update (reference data):\n" + LATTE.speak + "\n\n" + CROISSANT.speak)
        )
        self.assertIn('(you already told the caller: "Let me add that. And the croissant."', update)
        self.assertNotIn("Unrelated.", update)
        self.assertIn('still in progress: "Make it a large."', update)

    async def test_results_of_one_delegation_carry_its_told_note_once(self):
        words, frontend, _ = wording([TalkerDecision("speak", "Done.")])
        second = ReadyResult("item_d1", "msg_3", "Seat 12A is held.")
        await words.words([LATTE, second], WordingNotes(told={"item_d1": "One moment."}, newer_requests=()))
        self.assertEqual(frontend.calls[0]["history"][-1]["content"].count("One moment."), 1)

    async def test_the_one_attempt_is_the_only_attempt(self):
        words, frontend, observe = wording([TimeoutError("slow"), TalkerDecision("speak", "Never asked for.")])
        self.assertEqual(await words.words([LATTE], NO_NOTES), "Added one large Latte ($5.50).")
        self.assertEqual(len(frontend.calls), 1)
        self.assertEqual(observe.events[0][0], "commentary.recovered")

    async def test_nothing_ready_says_nothing_and_calls_nothing(self):
        words, frontend, _ = wording([])
        self.assertEqual(await words.words([], NO_NOTES), "")
        self.assertEqual(frontend.calls, [])


class AsWrittenTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_failed_call_says_every_result_as_written(self):
        words, _, observe = wording([RuntimeError("hub down")])
        results = [ReadyResult("d1", "m1", "**Total:** $12.50"), CROISSANT]
        self.assertEqual(
            await words.words(results, NO_NOTES), "Total: $12.50. Added one Butter Croissant ($3.75). Total: $9.25."
        )
        self.assertEqual(observe.events, [("commentary.recovered", {"error_type": "RuntimeError", "results": 2})])

    async def test_a_late_call_is_cut_at_the_deadline_and_said_as_written(self):
        late = asyncio.get_running_loop().create_future()
        words, _, _ = wording([late], timeout_seconds=0.05)
        self.assertEqual(await words.words([LATTE], NO_NOTES), "Added one large Latte ($5.50).")

    async def test_a_silent_answer_is_said_as_written(self):
        words, _, observe = wording([TalkerDecision("speak", "  ")])
        self.assertEqual(await words.words([LATTE, CROISSANT], NO_NOTES), as_written([LATTE, CROISSANT]))
        self.assertEqual(observe.events[0], ("commentary.answer_forced", {"talker_action": "speak", "results": 2}))

    async def test_a_writer_that_delegates_speaks_the_verified_text(self):
        words, _, _ = wording([TalkerDecision("delegate", "")])
        self.assertEqual(await words.words([LATTE], NO_NOTES), LATTE.speak)

    def test_as_written_skips_empty_results(self):
        self.assertEqual(as_written([ReadyResult("d", "m", "  "), LATTE]), "Added one large Latte ($5.50).")


class SetTests(unittest.IsolatedAsyncioTestCase):
    async def test_v3_words_several_results_under_its_own_frame_and_label(self):
        v3 = get_prompt_set("v3")
        words, frontend, _ = wording([TalkerDecision("speak", "Both are in.")], prompts=v3)
        await words.words([LATTE, CROISSANT], NO_NOTES)
        call = frontend.calls[0]
        self.assertTrue(call["instructions"].startswith(v3.writer_frame_before + INSTRUCTIONS))
        self.assertIn("When it holds several results", call["instructions"])
        self.assertTrue(call["history"][-1]["content"].startswith(v3.update_label))


if __name__ == "__main__":
    unittest.main()
