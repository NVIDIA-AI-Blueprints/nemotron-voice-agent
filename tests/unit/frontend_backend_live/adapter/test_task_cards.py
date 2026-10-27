# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The display path's Pipecat half: where each delegation's display part goes."""

import asyncio
import unittest

from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame

from examples.frontend_backend_live.adapter.task_cards import ENDED_STATUS, RtviTaskCards
from examples.frontend_backend_live.cascade.ports import DisplaySink, Ending, ReadyResult

SHOW_ITEM = {
    "type": "mcp_call",
    "id": "mcp_1",
    "server_label": "voiceclaw",
    "name": "show",
    "arguments": '{"display": "Latte, oat milk, large: $5.40"}',
    "output": "Latte, oat milk, large: $5.40",
    "status": "completed",
}


def members(protocol) -> set[str]:
    return {name for name in vars(protocol) if not name.startswith("_")}


class TaskCardTests(unittest.IsolatedAsyncioTestCase):
    def cards(self):
        sent: list[dict] = []

        async def queue_frame(frame):
            self.assertIsInstance(frame, RTVIServerMessageFrame)
            sent.append(frame.data)

        return RtviTaskCards(queue_frame), sent

    async def settled(self):
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    async def test_a_card_carries_the_display_part(self):
        cards, sent = self.cards()
        cards.started("item_d1", "A large oat latte, please.")
        cards.shown("item_d1", SHOW_ITEM)
        await self.settled()
        self.assertEqual(sent[-1]["type"], "agent-task-update")
        self.assertEqual(sent[-1]["task_id"], "item_d1")
        self.assertEqual(sent[-1]["detail"], "Latte, oat milk, large: $5.40")
        self.assertEqual(sent[-1]["query"], "A large oat latte, please.")
        self.assertEqual(sent[-1]["status"], "running")

    async def test_the_card_follows_its_delegation_to_the_end_in_call_order(self):
        cards, sent = self.cards()
        cards.started("item_d1", "Book the Denver flight.")
        cards.progress("item_d1", "Checking flights to Denver.")
        cards.answered(ReadyResult("item_d1", "msg_1", "Your flight to Denver is booked."))
        cards.answered(ReadyResult("item_d1", "msg_2", "Your seat is 14C."))
        cards.ended("item_d1", Ending.ANSWERED)
        cards.spoken(["item_d1"], "You're booked to Denver, seat 14C.")
        await self.settled()
        self.assertEqual([card["status"] for card in sent], ["running"] * 4 + ["done"] * 2)
        self.assertEqual(sent[1]["stage"], "Checking flights to Denver.")
        self.assertEqual(sent[3]["response"], "Your flight to Denver is booked.\n\nYour seat is 14C.")
        # The words said come after the end here: the last card is done and carries them, so the UI shows the turn.
        self.assertEqual(sent[-1]["spoken_response"], "You're booked to Denver, seat 14C.")
        self.assertEqual(sent[-1]["stage"], "answered")
        self.assertEqual({card["agent"] for card in sent}, {"backend"})

    async def test_one_line_said_for_two_delegations_reaches_both_cards(self):
        cards, sent = self.cards()
        cards.spoken(["item_d1", "item_d2"], "Both are done.")
        await self.settled()
        self.assertEqual(
            [(card["task_id"], card["spoken_response"]) for card in sent],
            [
                ("item_d1", "Both are done."),
                ("item_d2", "Both are done."),
            ],
        )

    async def test_each_ending_has_a_card_status(self):
        self.assertEqual(set(ENDED_STATUS), set(Ending))
        cards, sent = self.cards()
        cards.ended("item_d1", Ending.FAILED)
        await self.settled()
        self.assertEqual((sent[0]["status"], sent[0]["stage"]), ("failed", "failed"))


class SinkContractTests(unittest.TestCase):
    def test_the_cards_take_every_call_of_the_display_port(self):
        self.assertLessEqual(members(DisplaySink), set(dir(RtviTaskCards)))


if __name__ == "__main__":
    unittest.main()
