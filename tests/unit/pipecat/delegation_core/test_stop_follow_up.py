# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

"""New requests survive an older Work's deadline, while a conductor cancellation stays final."""

import asyncio

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import (
    Cancel,
    Start,
    Steer,
    WorkTarget,
)
from tests.unit.pipecat.delegation_core import helpers


class StopFollowUpTests(helpers.IMCase):
    async def test_conductor_cancel_does_not_replay_unapplied_requests(self):
        conductor = helpers.planned(
            Start(("D1",)),
            Steer(("D2",), (WorkTarget("W1", None),), clarifications=()),
            Cancel(("D3",), (WorkTarget("W1", None),), clarifications=()),
        )
        im, adapter = self.scripted_session(
            [helpers.wait(asyncio.Event()), helpers.checkpoint(), helpers.answer("The old job finished.")],
            [helpers.answer("The cancelled request was incorrectly replayed.")],
            conductor=conductor,
            max_running=3,
            delivery="not_applied",
        )
        im.admit(helpers.ask("Do the long job."), delegation_id="D1")
        await self.until(lambda: bool(adapter.runs), "the first worker started")
        newer = im.admit(helpers.ask("Change that job."), delegation_id="D2")
        await self.until(lambda: im.ledger.delegations["D2"].work == "W1", "the unapplied steer")
        stop = im.admit(helpers.ask("Stop that job."), delegation_id="D3")
        await self.until(lambda: "W1" not in im.works, "the conductor cancelled the Work")

        events = await helpers.read_all(newer.stream, timeout=3)

        self.assertEqual(im.ledger.delegations["D2"].state, "cancelled")
        self.assertEqual(im.ledger.steerings["S1"].state, "cancelled")
        self.assertEqual(list(im.ledger.works), ["W1"])
        self.assertEqual(events[-1]["type"], "response.incomplete")
        self.assertEqual(len(adapter.runs), 1)
        self.assertEqual(self.traced("im.follow_up"), [])
        self.assertEqual((await helpers.read_all(stop.stream))[-1]["type"], "response.completed")
        self.assertEqual(im.ledger.delegations["D3"].state, "answered")
