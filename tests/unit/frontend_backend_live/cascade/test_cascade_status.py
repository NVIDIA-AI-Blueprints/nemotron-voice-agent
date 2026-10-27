# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Status from progress notes."""

import asyncio

from examples.frontend_backend_live.cascade.settings import CascadeSettings
from examples.frontend_backend_live.models.decision import TalkerDecision
from tests.unit.frontend_backend_live.cascade.doubles import Call, created, in_progress, message, settle


def quiet(after=0.1, cap=1):
    return CascadeSettings.parse({"delegation": {"progress_after_seconds": after, "progress_max_per_delegation": cap}})


class StatusTests(Call):
    async def test_the_newest_progress_line_wins_in_the_work_line(self):
        await self.call([TalkerDecision("delegate", "One moment."), TalkerDecision("speak", "Checking flights now.")])
        await self.turn("Change my flight.")
        self.emit("item_1", created("r1"), in_progress("r1"))
        self.emit("item_1", message("Looking up your booking.", phase="commentary", item_id="c1"))
        self.emit("item_1", message("Searching flights.", phase="commentary", item_id="c2"))
        await self.turn("Any updates?")  # no shortcut: the talker reads the line and answers
        instructions = self.frontend.calls[1]["instructions"]
        self.assertIn('latest update: "Searching flights."', instructions)
        self.assertNotIn("Looking up your booking.", instructions)
        self.assertEqual(self.spoken[-1], "Checking flights now.")

    async def test_after_a_quiet_spell_the_newest_unsaid_line_is_said_once(self):
        await self.call([TalkerDecision("delegate", "One moment.")], settings=quiet())
        await self.turn("Change my flight.")
        self.emit("item_1", created("r1"), in_progress("r1"))
        self.emit("item_1", message("Searching flights.", phase="commentary", item_id="c1"))
        await self.said(2, timeout=3)
        await asyncio.sleep(0.4)  # several quiet spells: the cap of one holds
        self.assertEqual(self.spoken, ["One moment.", "Searching flights."])
        self.assertEqual(self.session.utterances.told_text("item_1"), "One moment. Searching flights.")

    async def test_with_no_progress_line_the_still_working_line_is_said(self):
        await self.call([TalkerDecision("delegate", "One moment.")], settings=quiet(cap=2))
        await self.turn("Change my flight.")
        await settle(lambda: len(self.spoken) == 3, timeout=3)
        self.assertEqual(self.spoken, ["One moment.", self.lines.still_working, self.lines.still_working])

    async def test_no_line_is_offered_while_the_caller_speaks_or_when_it_is_off(self):
        await self.call([TalkerDecision("delegate", "One moment.")], settings=quiet())
        await self.turn("Change my flight.")
        self.session.caller_started()
        await asyncio.sleep(0.3)
        self.assertEqual(self.spoken, ["One moment."])
        await self.call([TalkerDecision("delegate", "One moment.")], settings=quiet(after=None))
        await self.turn("Change my flight.")
        await asyncio.sleep(0.2)
        self.assertEqual(self.spoken, ["One moment."])
