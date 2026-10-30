# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The floor: a result waits while the caller speaks or a turn is decided, and goes once the line in flight played."""

import asyncio
import unittest

from examples.frontend_backend_live.cascade.floor import Floor
from examples.frontend_backend_live.cascade.ports import ReadyResult
from examples.frontend_backend_live.models.decision import TalkerDecision
from tests.unit.frontend_backend_live.cascade.doubles import Call


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

    async def test_a_wait_for_a_change_counts_one_made_before_the_wait_first_runs(self):
        floor = Floor(Clock())
        waiting = asyncio.ensure_future(floor.wait_change())  # it has not run yet
        floor.line_handed()
        await asyncio.wait_for(waiting, 1)

    async def test_it_counts_the_callers_starts_and_a_wait_for_their_silence_ends_when_they_stop(self):
        floor = Floor(Clock())
        await asyncio.wait_for(floor.wait_caller_silent(), 1)  # silent: at once
        floor.set_caller_speaking(True)
        waiting = asyncio.ensure_future(floor.wait_caller_silent())
        floor.set_turns_busy(True)  # another fact moves; the caller still speaks
        await asyncio.sleep(0.01)
        self.assertFalse(waiting.done())
        floor.set_caller_speaking(False)
        await asyncio.wait_for(waiting, 1)
        floor.set_caller_speaking(True)
        self.assertEqual(floor.caller_starts, 2)

    def test_a_call_for_the_callers_silence_runs_at_once_or_once_when_they_stop(self):
        floor, calls = Floor(Clock()), []
        floor.when_caller_silent(lambda: calls.append("silent"))  # silent: at once
        floor.set_caller_speaking(True)
        floor.when_caller_silent(lambda: calls.append("stopped"))
        floor.set_turns_busy(True)  # another fact moves; the caller still speaks
        self.assertEqual(calls, ["silent"])
        floor.set_caller_speaking(False)
        floor.set_caller_speaking(True)
        floor.set_caller_speaking(False)
        self.assertEqual(calls, ["silent", "stopped"])


class HoldTests(Call):
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
