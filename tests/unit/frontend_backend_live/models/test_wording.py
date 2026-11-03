# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""One wording for every ready result: ``WriterWording`` on the ``ResultWording`` port."""

import asyncio
import unittest

from examples.frontend_backend_live.cascade.ports import ReadyResult, WordingNotes
from examples.frontend_backend_live.cascade.writer.plain_speech import plain_speech
from examples.frontend_backend_live.cascade.writer.wording import Instruction, WriterWording, as_written
from examples.frontend_backend_live.cascade.writer.writer import CommentaryWriter
from examples.frontend_backend_live.models.chat_completions import ChatCompletionsFrontend
from examples.frontend_backend_live.models.decision import TalkerDecision
from examples.frontend_backend_live.prompts import get_prompt_set
from tests.unit.cascade_helpers import CASCADE_ENDPOINT, FAST, ScriptedFrontend, _decision_chunks
from tests.unit.core_helpers import FakeChat

DEFAULT = get_prompt_set("default")
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


def wording(script, *, timeout_seconds=1.0, prompts=DEFAULT):
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
    async def test_one_result_sends_one_writer_request_with_the_frame_the_history_and_the_told_note(self):
        words, frontend, _ = wording([TalkerDecision("speak", "Your latte is in.")])
        notes = WordingNotes(told={"item_d1": "Let me add that."}, newer_requests=())
        self.assertEqual(await words.words([LATTE], notes), "Your latte is in.")
        (call,) = frontend.calls
        expected = DEFAULT.writer_framed(INSTRUCTIONS) + "\n" + DEFAULT.commentary + DEFAULT.answer
        self.assertEqual(call["instructions"], expected)
        self.assertEqual(call["history"][:-1], HISTORY)
        self.assertEqual(
            call["history"][-1]["content"],
            "Update for the caller:\nAdded one large Latte ($5.50).\n"
            '(the caller already heard: "Let me add that."; do not say that again)',
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
        self.assertTrue(update.startswith("Update for the caller:\n" + LATTE.speak + "\n\n" + CROISSANT.speak))
        self.assertIn(
            '(the caller already heard: "Let me add that. And the croissant."; do not say that again)', update
        )
        self.assertNotIn("Unrelated.", update)
        self.assertTrue(update.endswith(DEFAULT.writer_notes.newer))  # the newer request's note, without its words
        self.assertNotIn("Make it a large.", update)

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

    async def test_a_line_the_voice_already_said_is_no_wording_and_the_results_are_said_as_written(self):
        words, _, observe = wording([TalkerDecision("speak", "Sure.")])  # the voice's own line in HISTORY
        self.assertEqual(await words.words([LATTE], NO_NOTES), LATTE.speak)
        self.assertEqual(observe.events[0], ("commentary.answer_forced", {"talker_action": "speak", "results": 1}))


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    """The writer's unusable output: the results said as written, cleaned for speech, and never instructions said."""

    async def test_unusable_output_says_the_results_as_written_cleaned_for_speech(self):
        words, _, observe = wording([TalkerDecision("delegate", "", {"recovery": "invalid_json"})])
        order = ReadyResult("d1", "m1", "# Order\n- **Latte**\n- Scone")
        self.assertEqual(await words.words([order], NO_NOTES), "Order. Latte. Scone.")
        self.assertEqual(observe.events, [("commentary.answer_forced", {"talker_action": "delegate", "results": 1})])

    async def test_unusable_output_for_instructions_says_their_fallback_never_the_instructions(self):
        greeting = Instruction("", "c1", "Greet the caller as Remy", "Hi, I'm Remy. What can I get you?")
        words, _, observe = wording([TalkerDecision("delegate", "", {"recovery": "invalid_json"})])
        self.assertEqual(await words.words([greeting], NO_NOTES), "Hi, I'm Remy. What can I get you?")
        self.assertEqual(observe.events, [("commentary.answer_forced", {"talker_action": "delegate", "results": 1})])

    async def test_instructions_with_nested_list_markers_and_an_invalid_json_reply_say_the_fallback(self):
        # The reply echoes the instructions with no JSON, so the parser reads it as unusable and the action says so.
        instructions = "## 1. Greet the caller\n- - Greet the caller as Remy"
        greeting = Instruction("", "c1", instructions, "Hi, I'm Remy. What can I get you?")
        client = FakeChat(_decision_chunks(instructions))
        observe = Recorder()
        words = WriterWording(
            CommentaryWriter(ChatCompletionsFrontend(CASCADE_ENDPOINT, FAST, client), DEFAULT),
            lambda: (INSTRUCTIONS, list(HISTORY)),
            timeout_seconds=1.0,
            observe=observe,
        )
        self.assertEqual(await words.words([greeting], NO_NOTES), "Hi, I'm Remy. What can I get you?")
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(observe.events, [("commentary.answer_forced", {"talker_action": "delegate", "results": 1})])

    async def test_a_reply_that_repeats_the_instructions_cleaned_for_speech_says_the_fallback(self):
        instructions = "## 1. Greet the caller\n- - Greet the caller as Remy\n1. - Ask what they would like"
        greeting = Instruction("", "c1", instructions, "Hi, I'm Remy. What can I get you?")
        words, _, observe = wording([TalkerDecision("speak", plain_speech(instructions))])
        self.assertEqual(await words.words([greeting], NO_NOTES), "Hi, I'm Remy. What can I get you?")
        self.assertEqual(observe.events, [("commentary.answer_forced", {"talker_action": "speak", "results": 1})])

    async def test_a_reply_that_repeats_instructions_whose_markers_emphasis_hid_says_the_fallback(self):
        # One cleaning leaves such a marker ("1.", "-") and a second drops it: each cleaned once, the two differ.
        hidden = ("**1.** Greet the caller as Remy\n**2.** Ask what they want", "`- Greet the caller as Remy`")
        for instructions in hidden:
            with self.subTest(instructions=instructions):
                greeting = Instruction("", "c1", instructions, "Hi, I'm Remy. What can I get you?")
                words, _, observe = wording([TalkerDecision("speak", plain_speech(instructions))])
                self.assertEqual(await words.words([greeting], NO_NOTES), "Hi, I'm Remy. What can I get you?")
                self.assertEqual(
                    observe.events, [("commentary.answer_forced", {"talker_action": "speak", "results": 1})]
                )


if __name__ == "__main__":
    unittest.main()
