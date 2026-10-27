# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The floor: a result waits while the caller speaks or a turn is decided, and goes once the line in flight played."""

import asyncio
import unittest

from examples.frontend_backend_live.cascade.floor import Floor
from examples.frontend_backend_live.cascade.ports import ReadyResult
from examples.frontend_backend_live.models.decision import TalkerDecision
from tests.unit.frontend_backend_live.cascade.doubles import Call, settle


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


class FloorTests(unittest.IsolatedAsyncioTestCase):
    def test_it_may_prepare_when_the_caller_is_silent_and_no_turn_waits_and_is_idle_with_nothing_in_flight(self):
        floor = Floor(Clock())
        self.assertTrue((floor.may_prepare, floor.idle) == (True, True))
        floor.set_caller_speaking(True)
        self.assertEqual((floor.may_prepare, floor.idle), (False, False))
        floor.set_caller_speaking(False)
        floor.set_turns_busy(True)
        self.assertEqual((floor.may_prepare, floor.idle), (False, False))
        floor.set_turns_busy(False)
        floor.line_handed()
        self.assertEqual((floor.may_prepare, floor.idle), (True, False))
        floor.line_settled()
        floor.line_settled()  # never below nothing in flight
        self.assertEqual((floor.in_flight, floor.idle), (0, True))

    def test_the_quiet_counts_from_when_it_last_became_idle(self):
        clock = Clock()
        floor = Floor(clock)
        floor.line_handed()
        clock.now = 15.0
        self.assertEqual(floor.quiet_for(), 0.0)
        floor.line_settled()
        clock.now = 18.0
        self.assertEqual(floor.quiet_for(), 3.0)

    async def test_waiting_for_idle_returns_when_the_line_ends(self):
        floor = Floor(Clock())
        floor.line_handed()
        waiting = asyncio.create_task(floor.wait_idle())
        await asyncio.sleep(0.01)
        self.assertFalse(waiting.done())
        floor.line_settled()
        await asyncio.wait_for(waiting, 1)


class HoldTests(Call):
    async def test_a_result_waits_while_the_caller_speaks_with_no_time_cap(self):
        await self.call()
        self.session.caller_started()
        self.session.presenter.ready(ReadyResult("D1", "m1", "Latte added."))
        await asyncio.sleep(0.3)
        self.assertEqual(self.spoken, [])  # held as long as the caller speaks
        self.session.caller_stopped()
        await self.said(1)

    async def test_a_result_waits_while_a_turn_is_decided(self):
        slow = asyncio.get_running_loop().create_future()
        await self.call([slow])
        self.buffer.turn("Hello?")
        self.session.caller_stopped()
        deciding = asyncio.create_task(self.session.on_turn())
        await settle(lambda: self.frontend.calls)
        self.session.presenter.ready(ReadyResult("D1", "m1", "Latte added."))
        await asyncio.sleep(0.05)
        self.assertEqual(self.spoken, [])
        slow.set_result(TalkerDecision("speak", "Hi there."))
        await deciding
        await self.said(2)
        self.assertEqual(self.spoken[0], "Hi there.")

    async def test_with_playout_reported_the_result_waits_for_the_line_in_flight_to_end(self):
        await self.call([TalkerDecision("speak", "We open at seven.")], reports_playout=True)
        await self.turn("When do you open?")
        self.session.presenter.ready(ReadyResult("D1", "m1", "Latte added."))
        await asyncio.sleep(0.05)
        self.assertEqual(self.spoken, ["We open at seven."])  # the reply is still playing
        self.session.playout.started()
        self.session.playout.finished()
        await self.said(2)
        self.assertEqual(self.spoken, ["We open at seven.", "Latte added."])
