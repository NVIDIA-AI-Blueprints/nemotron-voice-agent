# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The display path, cascade side: the sink hears each delegation at once; display never enters memory."""

from examples.frontend_backend_live.cascade.display import NullDisplay
from examples.frontend_backend_live.cascade.ports import DisplaySink, Ending, ReadyResult
from examples.frontend_backend_live.models.decision import TalkerDecision
from tests.unit.frontend_backend_live.cascade.doubles import (
    Call,
    completed,
    created,
    in_progress,
    mcp_call,
    message,
    queued,
)


class DisplayTests(Call):
    async def test_the_sink_sees_shown_before_answered_and_the_spoken_line_at_hand_off(self):
        await self.call([TalkerDecision("delegate", "One moment."), TalkerDecision("speak", "Your latte is ready.")])
        await self.turn("Order a latte.")
        self.emit("item_1", created("r1"), queued("r1"), in_progress("r1"))
        self.emit("item_1", message("Placing the order.", phase="commentary", item_id="c1"))
        self.emit("item_1", mcp_call("delegation_core", "show", output='{"order": "BB-42"}'))
        self.emit("item_1", message("Order BB-42 is placed.", item_id="m1"), completed("r1"))
        await self.said(2)
        self.assertEqual(
            self.display.calls,
            [
                ("started", "item_1", "Order a latte."),
                ("progress", "item_1", "Placing the order."),
                ("shown", "item_1", "show"),
                ("answered", "item_1", "Order BB-42 is placed."),
                ("ended", "item_1", Ending.ANSWERED.value),
                ("spoken", ("item_1",), "Your latte is ready."),
            ],
        )
        self.assertNotIn('BB-42"}', str(self.memory()))  # the display item never enters the talker's memory

    async def test_the_null_display_takes_every_call(self):
        sink: DisplaySink = NullDisplay()
        sink.started("D1", "Teas?")
        sink.progress("D1", "Looking.")
        sink.shown("D1", {"type": "mcp_call"})
        sink.answered(ReadyResult("D1", "m1", "Tea."))
        sink.spoken(["D1"], "Tea.")
        sink.ended("D1", Ending.ANSWERED)
