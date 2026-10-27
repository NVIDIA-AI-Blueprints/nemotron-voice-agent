# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The talker's memory, kept by the cascade within a token budget."""

import unittest

from examples.frontend_backend_live.cascade.memory import (
    BACKEND_RESULT_PREFIX,
    CLIENT_CONTEXT_PREFIX,
    EntryKind,
    TalkerMemory,
    split_messages,
)
from examples.frontend_backend_live.models.decision import TalkerDecision
from tests.unit.frontend_backend_live.cascade.doubles import Call

BOT_PROMPT = [
    {"role": "system", "content": "You are Remy at Bluebird Cafe."},
    {"role": "system", "content": "Channel: in-store kiosk."},
]


def memory(messages=(), tokens=16_000):
    return TalkerMemory.seeded(list(messages), context_tokens=tokens, compact_at=0.8)


class EntryTests(unittest.TestCase):
    def test_the_seed_is_pinned_prompt_then_history_rendered_as_today(self):
        seeded = memory(
            [
                {"role": "system", "content": "Be Remy."},
                {"role": "developer", "content": "Greet the caller now."},
                {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
                {"role": "assistant", "content": "Hello."},
                {"role": "user", "content": BACKEND_RESULT_PREFIX + "Open at nine."},
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
                {"role": "user", "content": BACKEND_RESULT_PREFIX + "Open at nine."},
            ],
        )

    def test_each_writer_adds_its_kind_and_moves_the_version(self):
        talker = memory()
        before = talker.version
        talker.caller("Two bags.")
        talker.spoken(1, "Noted.")
        talker.result(2, "Bags: 2.")
        talker.app_context("Gold member.")
        talker.instruction("Speak Spanish.")
        talker.typed("What are your hours?")
        self.assertEqual(
            talker.messages(),
            [
                {"role": "user", "content": "Two bags."},
                {"role": "assistant", "content": "Noted."},
                {"role": "user", "content": BACKEND_RESULT_PREFIX + "Bags: 2."},
                {"role": "user", "content": CLIENT_CONTEXT_PREFIX + "Gold member."},
                {"role": "developer", "content": "Speak Spanish."},
                {"role": "user", "content": CLIENT_CONTEXT_PREFIX + "What are your hours?"},
            ],
        )
        self.assertEqual(talker.version, before + 6)

    def test_playout_corrects_a_line_and_takes_back_what_a_line_nobody_heard_carried(self):
        talker = memory()
        talker.caller("Teas?")
        talker.result(1, "Earl Grey.")
        talker.app_context("Said with it.", line=1)
        talker.spoken(1, "We have Earl Grey.")
        talker.spoken(2, "Anything else?")
        talker.cut(2, "Anything")
        self.assertEqual(talker.messages()[-1], {"role": "assistant", "content": "Anything [cut off]"})
        talker.unsay(1)
        self.assertEqual([m["content"] for m in talker.messages()], ["Teas?", "Anything [cut off]"])

    def test_the_normalizers_rewrite_replaces_the_newest_matching_caller_turn(self):
        talker = memory()
        for text in ("two bags", "two bags"):
            talker.caller(text)
            talker.spoken(1, "Okay.")
        talker.replace_caller("two bags", "2 bags")
        self.assertEqual([m["content"] for m in talker.messages()], ["two bags", "Okay.", "2 bags", "Okay."])


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
    async def test_the_ui_bots_two_message_prompt_survives_thirty_turns(self):
        # The bot's prompt is pinned, so trimming old turns never drops it.
        await self.call([TalkerDecision("speak", f"Reply {n}.") for n in range(30)], seed=BOT_PROMPT[1:])
        for n in range(30):
            await self.turn(f"Turn {n}.")
        instructions = self.frontend.calls[-1]["instructions"]
        self.assertIn("You are Remy.", instructions)
        self.assertIn("Channel: in-store kiosk.", instructions)
        self.assertEqual(self.frontend.calls[-1]["history"][-1], {"role": "user", "content": "Turn 29."})

    async def test_the_talker_reads_only_the_cascades_memory_never_the_hosts_buffer(self):
        await self.call([TalkerDecision("speak", "Hi.")])
        self.buffer.items.append({"role": "assistant", "content": "Something Pipecat wrote."})
        await self.turn("Hello.")
        self.assertNotIn("Something Pipecat wrote.", str(self.frontend.calls[0]))
        self.assertIs(self.session.conversation, self.session.memory)  # what the adapter's /history reads
