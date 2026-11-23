# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The display path's Pipecat half: each delegation's task card on the RTVI channel."""

import asyncio
import unittest

from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegations import Ending, ReadyResult
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.media.task_cards import RtviTaskCards


class TaskCardTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_card_follows_its_delegation_to_its_end(self):
        sent: list[dict] = []

        async def queue_frame(frame):
            self.assertIsInstance(frame, RTVIServerMessageFrame)
            sent.append(frame.data)

        cards = RtviTaskCards(queue_frame)
        cards.started("item_d1", "Order a large latte.")
        cards.observe("im.routing.started", delegation_ids=["item_d1"])
        cards.observe("im.routing.applying", delegation_ids=["item_d1"])
        worker = dict(source_delegation_id="item_d1", task_id="W1.1", brief="Add the latte")
        cards.observe("im.task.state", **worker, state="queued")
        cards.observe("im.task.state", **worker, state="running")
        lookup = dict(source_delegation_id="item_d1", task_id="W2.1", brief="Check pickup options")
        cards.observe("im.task.state", **lookup, state="queued")
        cards.observe("im.task.state", **lookup, state="running")
        cards.started("item_d2", "Cancel the pickup lookup.")
        cards.observe("im.routing.started", delegation_ids=["item_d2"])
        cards.observe("im.routing.applying", delegation_ids=["item_d2"])
        cards.observe("im.task.state", **lookup, state="stopping")
        cards.observe("im.decision.accepted", sources=["item_d2"])
        cards.observe("im.task.state", **lookup, state="cancelled")
        cards.ended("item_d2", Ending.ANSWERED)
        cards.progress("item_d1", "Checking the menu.")
        shown = {"type": "mcp_call", "name": "show", "output": "Latte, large, $5.50", "status": "completed"}
        cards.shown("item_d1", shown)
        cards.answered(ReadyResult("item_d1", "msg_1", "Your latte is in the cart."))
        cards.answered(ReadyResult("item_d1", "msg_2", "Your total is $5.97."))
        cards.observe("im.task.state", **worker, state="answered")
        cards.ended("item_d1", Ending.ANSWERED)
        cards.started("item_d3", "Check the menu.")
        cards.observe("im.routing.unresolved", delegation_ids=["item_d3"])
        refusal = ReadyResult("item_d3", "msg_3", "I could not start that request.")
        cards.answered(refusal)
        cards.ended("item_d3", Ending.ANSWERED)
        await asyncio.sleep(0)
        order = [card["status"] for card in sent if card["task_id"] == "item_d1"]
        self.assertEqual(
            list(dict.fromkeys(order)), ["queued", "planning", "applying", "running", "waiting_result", "done"]
        )
        cancellation = [card for card in sent if card["task_id"] == "item_d2"]
        self.assertEqual(
            [card["status"] for card in cancellation], ["queued", "planning", "applying", "waiting_result", "done"]
        )
        self.assertNotIn("workers", cancellation[-1])
        self.assertTrue(
            any([worker["status"] for worker in card.get("workers", [])] == ["running", "stopping"] for card in sent)
        )
        self.assertIn("Checking the menu.", [card["stage"] for card in sent])
        final = next(card for card in reversed(sent) if card["task_id"] == "item_d1")
        self.assertEqual((final["type"], final["agent"], final["task_id"]), ("agent-task-update", "backend", "item_d1"))
        self.assertEqual((final["stage"], final["query"]), ("answered", "Order a large latte."))
        self.assertEqual(final["detail"], "Latte, large, $5.50")
        self.assertEqual(final["response"], "Your latte is in the cart.\n\nYour total is $5.97.")
        self.assertEqual(
            final["workers"],
            [
                {"id": "W1.1", "label": "Add the latte", "status": "answered", "status_label": "Completed"},
                {"id": "W2.1", "label": "Check pickup options", "status": "cancelled", "status_label": "cancelled"},
            ],
        )
        failed = sent[-1]
        self.assertEqual(
            (failed["task_id"], failed["status"], failed["stage"]), ("item_d3", "failed", "Couldn't start")
        )
        self.assertEqual(failed["response"], refusal.speak)
        cards.started("item_d4", "Stop checking delivery.")
        cards.observe("im.routing.applying", delegation_ids=["item_d4"])
        cards.observe("im.decision.accepted", sources=["item_d4"])
        stopping = dict(source_delegation_id="item_d4", task_id="W4.1", brief="Check delivery")
        sibling = dict(source_delegation_id="item_d4", task_id="W4.2", brief="Check availability")
        cards.observe("im.task.state", **stopping, state="stop_unconfirmed")
        cards.observe("im.task.state", **sibling, state="running")
        cards.observe("im.task.state", **sibling, state="paused")
        await asyncio.sleep(0)
        self.assertEqual(sent[-2]["status"], "running")
        self.assertEqual((sent[-1]["status"], sent[-1]["status_label"]), ("stop_unconfirmed", "Stop unconfirmed"))
