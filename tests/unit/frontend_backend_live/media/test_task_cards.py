# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The display path's Pipecat half: each delegation's task card on the RTVI channel."""

import asyncio
import unittest

from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame

from examples.frontend_backend_live.delegations import Ending, ReadyResult
from examples.frontend_backend_live.media.task_cards import RtviTaskCards


class TaskCardTests(unittest.IsolatedAsyncioTestCase):
    def cards(self):
        sent: list[dict] = []

        async def queue_frame(frame):
            self.assertIsInstance(frame, RTVIServerMessageFrame)
            sent.append(frame.data)

        return RtviTaskCards(queue_frame), sent

    async def test_the_card_follows_its_delegation_to_the_end_in_call_order_with_its_display_part(self):
        cards, sent = self.cards()
        cards.started("item_d1", "Book the Denver flight.")
        cards.progress("item_d1", "Checking flights to Denver.")
        cards.shown(
            "item_d1", {"type": "mcp_call", "name": "show", "output": "Flight DL 42, 9:05", "status": "completed"}
        )
        cards.answered(ReadyResult("item_d1", "msg_1", "Your flight to Denver is booked."))
        cards.answered(ReadyResult("item_d1", "msg_2", "Your seat is 14C."))
        cards.ended("item_d1", Ending.ANSWERED)
        await asyncio.sleep(0)
        self.assertEqual([card["status"] for card in sent], ["running"] * 5 + ["done"])
        self.assertEqual(
            {(card["type"], card["task_id"], card["agent"]) for card in sent},
            {("agent-task-update", "item_d1", "backend")},
        )
        self.assertEqual((sent[1]["stage"], sent[2]["detail"]), ("Checking flights to Denver.", "Flight DL 42, 9:05"))
        self.assertEqual(sent[4]["response"], "Your flight to Denver is booked.\n\nYour seat is 14C.")
        self.assertEqual((sent[-1]["stage"], sent[-1]["query"]), ("answered", "Book the Denver flight."))
        # The card leaves the words said to the bot's transcript, so the UI shows each line once.
        fields = {"type", "task_id", "agent", "status", "stage", "query", "detail", "response"}
        self.assertEqual({frozenset(card) for card in sent}, {frozenset(fields)})

    async def test_a_failure_shows_as_failed(self):
        cards, sent = self.cards()
        cards.ended("item_d3", Ending.FAILED)
        await asyncio.sleep(0)
        self.assertEqual((sent[0]["task_id"], sent[0]["status"], sent[0]["stage"]), ("item_d3", "failed", "failed"))
