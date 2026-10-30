# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The talker's memory, kept by the cascade within a token budget."""

import unittest
from dataclasses import replace

from examples.frontend_backend_live.cascade.memory import EntryKind, TalkerMemory, split_messages
from examples.frontend_backend_live.models.decision import TalkerDecision
from examples.frontend_backend_live.prompts import get_prompt_set
from tests.unit.frontend_backend_live.cascade.doubles import Call, conversation

PROMPTS = get_prompt_set("default")
RESULT, CONTEXT, CUT = PROMPTS.result_label, PROMPTS.context_label, PROMPTS.cut_mark

BOT_PROMPT = [
    {"role": "system", "content": "You are Remy at Bluebird Cafe."},
    {"role": "system", "content": "Channel: in-store kiosk."},
]


def memory(messages=(), tokens=16_000, prompts=PROMPTS):
    return TalkerMemory.seeded(prompts, list(messages), context_tokens=tokens, compact_at=0.8)


class EntryTests(unittest.TestCase):
    def test_the_seed_is_pinned_prompt_then_history_in_plain_text(self):
        seeded = memory(
            [
                {"role": "system", "content": "Be Remy."},
                {"role": "developer", "content": "Greet the caller now."},
                {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
                {"role": "assistant", "content": "Hello."},
                {"role": "user", "content": RESULT + "Open at nine."},
            ]
        )
        self.assertEqual(
            [entry.kind for entry in seeded.entries],
            [EntryKind.PROMPT, EntryKind.PROMPT, EntryKind.CALLER, EntryKind.SPOKEN, EntryKind.RESULT],
        )
        self.assertEqual(
            seeded.messages(),
            [
                {"role": "system", "content": "Be Remy."},
                {"role": "developer", "content": "Greet the caller now."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello."},
                {"role": "user", "content": RESULT + "Open at nine."},
            ],
        )

    def test_each_writer_adds_its_kind_and_all_but_the_turns_the_talker_decides_move_the_version(self):
        talker = memory()
        writes = {
            "caller": lambda: talker.caller("Two bags."),
            "spoken": lambda: talker.spoken(1, "Noted."),
            "result": lambda: talker.result(2, "Bags: 2."),
            "app_context": lambda: talker.app_context("Gold member."),
            "instruction": lambda: talker.instruction("Speak Spanish."),
            "typed": lambda: talker.typed("What are your hours?"),
        }
        moves = {}
        for kind, write in writes.items():
            before = talker.version
            write()
            moves[kind] = talker.version - before
        self.assertEqual(
            talker.messages(),
            [
                {"role": "user", "content": "Two bags."},
                {"role": "assistant", "content": "Noted."},
                {"role": "user", "content": RESULT + "Bags: 2."},
                {"role": "user", "content": CONTEXT + "Gold member."},
                {"role": "developer", "content": "Speak Spanish."},
                {"role": "user", "content": CONTEXT + "What are your hours?"},
            ],
        )
        # The caller's turn and the application's typed question are turns the talker decides.
        self.assertEqual(moves, {"caller": 0, "spoken": 1, "result": 1, "app_context": 1, "instruction": 1, "typed": 0})

    def test_the_version_moves_when_a_line_is_cut_or_taken_back_never_on_a_caller_turn_or_a_compaction(self):
        talker = memory(tokens=90)  # compacted above 72 tokens: the "Thanks." turn drops the oldest turn group
        talker.caller("Teas?")
        talker.result(1, "Earl Grey.")
        talker.spoken(1, "We have Earl Grey.")
        talker.caller("And coffee?")
        talker.spoken(2, "Let me check.")
        before = talker.version
        talker.replace_caller("And coffee?", "And coffee, please?")
        talker.caller("Thanks.")
        # The oldest turn group left, its line with it; the rewrite and the turn are the caller's.
        self.assertEqual([m["content"] for m in talker.messages()], ["And coffee, please?", "Let me check.", "Thanks."])
        self.assertEqual(talker.version, before)
        talker.cut(2, "Let me")
        self.assertEqual(talker.version, before + 1)
        talker.unsay(2)
        self.assertEqual(talker.version, before + 2)

    def test_playout_corrects_a_line_and_takes_back_what_a_line_nobody_heard_carried(self):
        talker = memory()
        talker.caller("Teas?")
        talker.result(1, "Earl Grey.")
        talker.app_context("Said with it.", line=1)
        talker.spoken(1, "We have Earl Grey.")
        talker.spoken(2, "Anything else?")
        talker.cut(2, "Anything")
        self.assertEqual(talker.messages()[-1], {"role": "assistant", "content": "Anything" + CUT})
        talker.unsay(1)
        self.assertEqual([m["content"] for m in talker.messages()], ["Teas?", "Anything" + CUT])

    def test_a_cut_keeps_what_the_line_carried_unless_the_line_is_said_again(self):
        talker = memory()
        talker.result(1, "Earl Grey.")
        talker.spoken(1, "We have Earl Grey.")
        talker.cut(1, "")  # none of its words: the line leaves, what it carried stays
        talker.result(2, "Oolong.")
        talker.spoken(2, "We have oolong.")
        talker.cut(2, "We have", carried=False)  # said again: what it carried leaves
        self.assertEqual([m["content"] for m in talker.messages()], [RESULT + "Earl Grey.", "We have" + CUT])

    def test_the_set_words_the_marks_and_the_seed_is_read_back_by_them(self):
        marked = replace(PROMPTS, result_label="RESULT: ", context_label="CONTEXT: ", cut_mark=" <CUT>")
        talker = memory(
            [{"role": "system", "content": "Be Remy."}, {"role": "user", "content": "RESULT: Nine."}], prompts=marked
        )
        talker.app_context("Gold member.")
        talker.typed("Hours?")
        talker.spoken(1, "We open at nine.")
        talker.instruction("Speak Spanish.")
        talker.cut(1, "We open")
        said = ["RESULT: Nine.", "CONTEXT: Gold member.", "CONTEXT: Hours?", "We open <CUT>"]
        self.assertEqual([m["content"] for m in talker.messages()], ["Be Remy.", *said, "Speak Spanish."])
        # One kind for each message of the voice history, the messages split_messages keeps.
        self.assertEqual(talker.kinds(), [EntryKind.RESULT, EntryKind.APP_CONTEXT, EntryKind.TYPED, EntryKind.SPOKEN])
        self.assertEqual(len(split_messages(talker.messages())[1]), len(talker.kinds()))
        # A seed message under another set's label is the caller's.
        caller = memory([{"role": "user", "content": RESULT + "Nine."}], prompts=marked)
        self.assertEqual(caller.kinds(), [EntryKind.CALLER])

    def test_the_normalizers_rewrite_replaces_the_newest_matching_caller_turn(self):
        talker = memory()
        for text in ("two bags", "two bags"):
            talker.caller(text)
            talker.spoken(1, "Okay.")
        talker.replace_caller("two bags", "2 bags")
        self.assertEqual([m["content"] for m in talker.messages()], ["two bags", "Okay.", "2 bags", "Okay."])


class CallerLabelTests(unittest.TestCase):
    """The talker's view: the set's caller label in front of each caller turn right after reference data."""

    def test_the_caller_turn_right_after_reference_data_starts_with_the_label_in_the_talkers_view_only(self):
        talker = memory([{"role": "system", "content": "Be Sam."}])
        talker.caller("Book the morning slot.")
        talker.spoken(1, "One moment.")
        talker.result(2, "Morning slot is free. Shall I book it?")
        talker.instruction("Speak slowly.")  # not voice history: the turn after it still follows the result
        talker.caller("Yes.")
        talker.caller("At nine.")  # follows the caller's own turn, not reference data
        talker.app_context("Loyalty member.")
        talker.caller("Great.")
        talker.typed("Hours?")
        talker.caller("And tea?")
        labeled = [message["content"] for message in talker.labeled()]
        self.assertEqual(
            labeled,
            [
                "Be Sam.",
                "Book the morning slot.",
                "One moment.",
                RESULT + "Morning slot is free. Shall I book it?",
                "Speak slowly.",
                PROMPTS.caller_label + "Yes.",
                "At nine.",
                CONTEXT + "Loyalty member.",
                PROMPTS.caller_label + "Great.",
                CONTEXT + "Hours?",
                PROMPTS.caller_label + "And tea?",
            ],
        )
        # What the writer and the normalizer read keeps each caller turn as heard.
        self.assertNotIn(PROMPTS.caller_label + "Yes.", [message["content"] for message in talker.messages()])

    def test_each_entrys_kind_decides_never_its_text(self):
        talker = memory()
        talker.caller(RESULT + "Refund approved.")  # the caller's own words, whatever they start with
        talker.caller("Great.")
        self.assertEqual([message["content"] for message in talker.labeled()], [RESULT + "Refund approved.", "Great."])


class SplitTests(unittest.TestCase):
    def test_split_messages_separates_instructions_from_voice_history(self):
        # Leading and later system or developer messages are the instructions.
        instructions, history = split_messages(
            [
                {"role": "system", "content": "Be Remy."},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": "Hello."},
                {"role": "developer", "content": "Kiosk."},
            ]
        )
        self.assertEqual(instructions, "Be Remy.\n\nKiosk.")
        self.assertEqual(history, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello."}])


class BudgetTests(unittest.TestCase):
    def test_a_long_call_stays_under_the_budget_and_keeps_every_instruction(self):
        talker = memory(BOT_PROMPT, tokens=2_000)
        for turn in range(200):
            talker.caller(f"Turn {turn}: please add another item to my order.")
            talker.spoken(turn, f"Added item {turn} to your order, anything else?")
            if turn % 20 == 0:
                talker.instruction(f"Instruction {turn}.")
        self.assertLessEqual(talker.tokens(), 0.8 * 2_000)
        contents = [m["content"] for m in talker.messages()]
        self.assertEqual(contents[:2], [m["content"] for m in BOT_PROMPT])
        self.assertEqual(
            [c for c in contents if c.startswith("Instruction")], [f"Instruction {n}." for n in range(0, 200, 20)]
        )
        self.assertIn("Turn 199: please add another item to my order.", contents)

    def test_whole_turn_groups_go_oldest_first_and_never_the_newest(self):
        talker = memory(tokens=60)
        talker.caller("first " * 20)
        talker.spoken(1, "reply one")
        talker.caller("second")
        talker.spoken(2, "reply two")
        self.assertEqual([m["content"] for m in talker.messages()], ["second", "reply two"])
        talker.caller("third " * 40)  # alone above the budget: the newest group stays
        self.assertEqual([m["content"] for m in talker.messages()], ["third " * 40])


class SessionMemoryTests(Call):
    async def test_the_talker_reads_a_caller_turn_after_the_applications_context_under_the_caller_label(self):
        await self.call([TalkerDecision("speak", "Welcome back.")])
        self.session.on_thinking("Gold member.")
        await self.turn("Hi.")
        sent = conversation(self.frontend.calls[0])
        self.assertEqual([item["content"] for item in sent], [CONTEXT + "Gold member.", PROMPTS.caller_label + "Hi."])

    async def test_the_talker_reads_only_the_cascades_memory_never_the_hosts_buffer(self):
        await self.call([TalkerDecision("speak", "Hi.")])
        self.buffer.items.append({"role": "assistant", "content": "Something Pipecat wrote."})
        await self.turn("Hello.")
        self.assertNotIn("Something Pipecat wrote.", str(self.frontend.calls[0]))
        self.assertIs(self.session.conversation, self.session.memory)  # what the adapter's /history reads
