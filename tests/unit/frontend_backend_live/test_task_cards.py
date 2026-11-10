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
    async def test_a_card_follows_its_delegation_to_its_end(self):
        sent: list[dict] = []

        async def queue_frame(frame):
            self.assertIsInstance(frame, RTVIServerMessageFrame)
            sent.append(frame.data)

        cards = RtviTaskCards(queue_frame)
        cards.started("item_d1", "Order a large latte.")
        cards.progress("item_d1", "Checking the menu.")
        shown = {"type": "mcp_call", "name": "show", "output": "Latte, large, $5.50", "status": "completed"}
        cards.shown("item_d1", shown)
        cards.answered(ReadyResult("item_d1", "msg_1", "Your latte is in the cart."))
        cards.answered(ReadyResult("item_d1", "msg_2", "Your total is $5.97."))
        cards.ended("item_d1", Ending.ANSWERED)
        cards.ended("item_d3", Ending.FAILED)
        await asyncio.sleep(0)
        self.assertEqual([card["status"] for card in sent], ["running"] * 5 + ["done", "failed"])
        self.assertEqual({(card["type"], card["agent"]) for card in sent}, {("agent-task-update", "backend")})
        self.assertEqual([card["task_id"] for card in sent], ["item_d1"] * 6 + ["item_d3"])
        self.assertEqual((sent[1]["stage"], sent[2]["detail"]), ("Checking the menu.", "Latte, large, $5.50"))
        self.assertEqual(sent[4]["response"], "Your latte is in the cart.\n\nYour total is $5.97.")
        self.assertEqual((sent[5]["stage"], sent[5]["query"]), ("answered", "Order a large latte."))
        self.assertEqual(sent[6]["stage"], "failed")
        # The card leaves the words said to the bot's transcript, so the UI shows each line once.
        fields = {"type", "task_id", "agent", "status", "stage", "query", "detail", "response"}
        self.assertEqual({frozenset(card) for card in sent[:6]}, {frozenset(fields)})
