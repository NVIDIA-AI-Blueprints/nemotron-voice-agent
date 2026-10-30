# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Tests for the backend LLM's rule conductors, ``steer`` and ``start_each``, on the conductor contract."""

import unittest

from delegation_core.backend.conductors import StartEach, SteerOpenWork
from delegation_core.backend.port import ConductorInput, Start, Steer, TaskSummary, WorkSummary
from tests.unit.contracts.backend import ConductorContract, FakeDecisionTarget, a_request, an_ended_work, an_open_work


class SteerOpenWorkContractTests(ConductorContract, unittest.IsolatedAsyncioTestCase):
    def make_conductor(self):
        return SteerOpenWork()


class StartEachContractTests(ConductorContract, unittest.IsolatedAsyncioTestCase):
    def make_conductor(self):
        return StartEach()


def a_work(work_id: str, state: str, seconds: float, acted: bool = False) -> WorkSummary:
    task = TaskSummary(f"{work_id}.1", state, None, (), acted, None, seconds)
    return WorkSummary(work_id, state, f"d_{work_id}", (a_request(f"d_{work_id}", "Earlier."),), (task,), seconds)


async def decide(conductor, *works: WorkSummary, arrivals=("d_new",)) -> list:
    target = FakeDecisionTarget()
    given = ConductorInput(tuple(a_request(d, "Also this.") for d in arrivals), works, (), 0)
    await conductor.decide(given, target.place)
    return target.decisions


class SteerRuleTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_newest_open_work_takes_the_join_and_an_ended_one_never_does(self):
        works = (an_ended_work("W0"), a_work("W1", "running", 9.0), a_work("W2", "queued", 0.0), an_ended_work("W3"))
        self.assertEqual(await decide(SteerOpenWork(), *works), [Steer("W2", ("d_new",))])
        tie = (a_work("W1", "queued", 0.0), a_work("W2", "queued", 0.0))
        self.assertEqual(await decide(SteerOpenWork(), *tie), [Steer("W2", ("d_new",))])
        # A Work that waits for the app's outputs is open too.
        waiting = a_work("W1", "waiting_client", 4.0, acted=True)
        self.assertEqual(await decide(SteerOpenWork(), waiting), [Steer("W1", ("d_new",))])
        # A Work that ended after the open one started is newer, and still never joined; with none open, a Start.
        just_ended = a_work("W5", "answered", 0.5)
        self.assertEqual(
            await decide(SteerOpenWork(), a_work("W4", "running", 9.0), just_ended), [Steer("W4", ("d_new",))]
        )
        self.assertEqual(await decide(SteerOpenWork(), just_ended, an_ended_work("W0")), [Start(("d_new",))])

    async def test_arrivals_together_are_one_decision(self):
        self.assertEqual(await decide(SteerOpenWork(), arrivals=("d_1", "d_2")), [Start(("d_1", "d_2"))])
        open_work = an_open_work("W1")
        self.assertEqual(
            await decide(SteerOpenWork(), open_work, arrivals=("d_1", "d_2")), [Steer("W1", ("d_1", "d_2"))]
        )
        self.assertEqual(await decide(SteerOpenWork(), open_work, arrivals=()), [])


class StartEachRuleTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_arrival_starts_its_own_work(self):
        decisions = await decide(StartEach(), an_open_work("W1"), arrivals=("d_1", "d_2"))
        self.assertEqual(decisions, [Start(("d_1",)), Start(("d_2",))])


if __name__ == "__main__":
    unittest.main()
