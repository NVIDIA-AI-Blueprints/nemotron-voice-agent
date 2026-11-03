# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Works, tasks, the scheduler and placement."""

import asyncio
import unittest

from delegation_core.backend.port import Answer, Cancel, Start, Steer
from tests.unit.contracts.backend import ConductorContract
from tests.unit.delegation_core.helpers import (
    MENU,
    TEXTS,
    IMCase,
    ScriptedAdapter,
    ScriptedConductor,
    StartEach,
    SteerNewest,
    answer,
    app_call,
    app_outputs,
    ask,
    fail,
    planned,
    read_all,
    wait,
)


class SchedulerTests(IMCase):
    async def test_two_works_run_at_once_up_to_max_running(self):
        # A task waiting for the app's outputs holds no slot, so two more Works run; once its outputs are in, it waits
        # for a slot until one of them ends.
        gates = [asyncio.Event(), asyncio.Event()]
        scripts = [[app_call(), answer("One.")], *([wait(gate), answer()] for gate in gates)]
        adapter = ScriptedAdapter(*scripts, conductor=StartEach(), max_running=2)
        im = self.session(adapter)
        first = await read_all(im.admit(ask("Request 1."), delegation_id="D1", settings={"tools": [MENU]}).stream)
        streams = [im.admit(ask(f"Request {n}."), delegation_id=f"D{n}").stream for n in (2, 3)]
        await self.turns()
        self.assertEqual([work.record.state for work in im.works.values()], ["waiting_client", "running", "running"])
        resumed = im.admit_client_request(app_outputs(first), delegation_id="D1", settings={"tools": [MENU]}).stream
        await self.turns()
        self.assertEqual(im.ledger.tasks["W1.1"].state, "waiting_client")  # its outputs are in: it waits for a slot
        gates[0].set()
        await read_all(streams[0])
        await self.turns()
        self.assertEqual(im.ledger.tasks["W1.1"].state, "answered")
        gates[1].set()
        for stream in (resumed, streams[1]):
            await read_all(stream)
        self.assertEqual([im.ledger.works[w].state for w in ("W1", "W2", "W3")], ["answered"] * 3)

    async def test_a_raising_work_ends_alone_and_the_scheduler_goes_on(self):
        adapter = ScriptedAdapter([fail()], [answer("Fine.")], conductor=StartEach())
        im = self.session(adapter)
        first = im.admit(ask("Break."), delegation_id="D1")
        second = im.admit(ask("Work."), delegation_id="D2")
        failed = await read_all(first.stream)
        answered = await read_all(second.stream)
        self.assertEqual(failed[-1]["type"], "response.failed")
        self.assertEqual(answered[-1]["type"], "response.completed")
        self.assertEqual((im.ledger.works["W1"].state, im.ledger.works["W2"].state), ("failed", "answered"))

    async def test_each_task_reads_its_own_brief(self):
        async def split(given, place):
            await place(Start(("D1",), ("teas", "hours")))

        adapter = ScriptedAdapter([answer()], [answer()], conductor=planned(split), max_running=2)
        im = self.session(adapter)
        await read_all(im.admit(ask("Teas and hours."), delegation_id="D1").stream)
        briefs = [run.histories[0][-1]["content"][0]["text"] for run in adapter.runs]
        self.assertEqual(briefs, [TEXTS.task_line.format(task="teas"), TEXTS.task_line.format(task="hours")])


class PlacementTests(IMCase):
    async def test_the_conductor_is_asked_once_per_session(self):
        adapter = ScriptedAdapter([answer()], [answer()])
        im = self.session(adapter)
        stream = im.admit(ask("One."), delegation_id="D1").stream
        self.assertTrue(im.busy)
        await read_all(stream)
        await read_all(im.admit(ask("Two."), delegation_id="D2").stream)
        self.assertEqual(adapter.conductors, 1)
        self.assertFalse(im.busy)

    async def test_an_arrival_restarts_a_decide_that_placed_nothing_once(self):
        thinking, calls = asyncio.Event(), []

        async def slow(given, place):
            calls.append([arrival.delegation_id for arrival in given.arrivals])
            await thinking.wait()
            for arrival in given.arrivals:
                await place(Start((arrival.delegation_id,)))

        conductor = ScriptedConductor(slow)
        im = self.session(ScriptedAdapter(conductor=conductor, max_running=3))
        im.admit(ask("One."), delegation_id="D1")
        await self.turns()
        im.admit(ask("Two."), delegation_id="D2")
        await self.turns()
        im.admit(ask("Three."), delegation_id="D3")  # the restarted call is not restarted again
        await self.turns()
        thinking.set()
        await self.turns()
        self.assertEqual(calls, [["D1"], ["D1", "D2"], ["D3"]])
        self.assertIn("im.decide.restarted", [kind for kind, _ in self.traces])

    async def test_an_arrival_after_a_placement_waits_for_the_next_decide(self):
        placed, release, calls = asyncio.Event(), asyncio.Event(), []

        async def plan(given, place):
            calls.append([arrival.delegation_id for arrival in given.arrivals])
            await place(Start(tuple(arrival.delegation_id for arrival in given.arrivals)))
            placed.set()
            await release.wait()

        im = self.session(ScriptedAdapter(conductor=ScriptedConductor(plan), max_running=2))
        im.admit(ask("One."), delegation_id="D1")
        await placed.wait()
        im.admit(ask("Two."), delegation_id="D2")
        await self.turns()
        self.assertEqual(calls, [["D1"]])
        release.set()
        await self.turns()
        self.assertEqual(calls, [["D1"], ["D2"]])
        self.assertEqual(im.ledger.delegations["D2"].work, "W2")

    async def test_a_split_over_max_running_is_refused_and_the_steer_rule_starts_one_task(self):
        conductor = planned(lambda given, place: place(Start(("D1",), ("teas", "hours", "days"))))
        im = self.session(ScriptedAdapter([answer()], conductor=conductor, max_running=2))
        await read_all(im.admit(ask("Teas, hours and days."), delegation_id="D1").stream)
        self.assertEqual(conductor.outcomes, [TEXTS.refusal("over_limit", max_running="2")])
        self.assertEqual([task.task_id for task in im.ended_works[-1].tasks], ["W1.1"])

    async def test_a_conductor_reads_the_open_and_ended_works_and_the_conversation_since_it_last_read(self):
        gate = asyncio.Event()
        conductor = ScriptedConductor(lambda given, place: place(Start(tuple(a.delegation_id for a in given.arrivals))))
        adapter = ScriptedAdapter([answer("Teas.")], [wait(gate), answer()], [answer()], conductor=conductor)
        adapter.max_running = 2
        im = self.session(adapter)
        await read_all(im.admit(ask("Teas?"), delegation_id="D1").stream)
        im.admit(ask("Hours?"), delegation_id="D2")
        await self.turns()
        im.admit(ask("Coffee?"), delegation_id="D3")
        await self.turns()
        first, second, third = conductor.given
        self.assertEqual((len(first.conversation), first.new_from), (0, 0))
        self.assertEqual(second.new_from, 0)
        self.assertTrue(second.conversation)
        self.assertEqual(third.new_from, len(third.conversation))
        self.assertEqual([(work.work_id, work.state) for work in third.works], [("W2", "running"), ("W1", "answered")])
        self.assertEqual([work.tasks[0].state for work in third.works], ["running", "answered"])
        gate.set()

    async def test_overlapping_places_apply_in_order_and_a_late_place_is_refused(self):
        kept = []

        async def parallel(given, place):
            kept.append(place)
            await asyncio.gather(place(Start(("D1",))), place(Steer("W1", ("D2",))))

        conductor = ScriptedConductor(parallel)
        im = self.session(ScriptedAdapter([answer()], conductor=conductor))
        first = im.admit(ask("Teas."), delegation_id="D1")
        im.admit(ask("Green only."), delegation_id="D2")
        await read_all(first.stream)
        self.assertEqual(conductor.outcomes[0], TEXTS.outcomes["started"].format(work="W1", tasks="W1.1"))
        self.assertEqual(conductor.outcomes[1], TEXTS.outcomes["steered"].format(work="W1", tasks="W1.1"))
        late = await kept[0](Start(("D2",)))
        self.assertTrue(late.startswith(TEXTS.outcomes["refused"].split("{")[0]))

    async def test_a_place_kept_from_a_restarted_decide_is_refused_while_the_next_one_runs(self):
        thinking, kept = asyncio.Event(), []

        async def slow(given, place):
            kept.append(place)
            await thinking.wait()
            for arrival in given.arrivals:
                await place(Start((arrival.delegation_id,)))

        im = self.session(ScriptedAdapter(conductor=ScriptedConductor(slow), max_running=2))
        im.admit(ask("One."), delegation_id="D1")
        await self.turns()
        im.admit(ask("Two."), delegation_id="D2")  # restarts the first decide(), which placed nothing
        await self.turns()
        stale = await kept[0](Start(("D1",)))  # a runtime that kept the cancelled turn places through it
        self.assertTrue(stale.startswith(TEXTS.outcomes["refused"].split("{")[0]), stale)
        self.assertEqual(im.arrivals, ["D1", "D2"])
        thinking.set()
        await self.turns()
        self.assertEqual([im.ledger.delegations[d].work for d in ("D1", "D2")], ["W1", "W2"])

    async def test_unplaced_arrivals_start_also_when_decide_raises(self):
        async def skip(given, place):
            return None

        async def boom(given, place):
            raise RuntimeError("conductor down")

        im = self.session(ScriptedAdapter([answer("One.")], [answer("Two.")], conductor=planned(skip, boom)))
        skipped = await read_all(im.admit(ask("One."), delegation_id="D1").stream)
        raised = await read_all(im.admit(ask("Two."), delegation_id="D2").stream)
        self.assertEqual([events[-1]["type"] for events in (skipped, raised)], ["response.completed"] * 2)
        self.assertEqual([im.ledger.delegations[d].work for d in ("D1", "D2")], ["W1", "W2"])

    async def test_what_a_decide_leaves_unplaced_joins_the_newest_open_work_by_the_steer_rule(self):
        # A request that a failed, timed-out, cancelled or skipping decide() leaves unplaced joins the newest open
        # Work, as the steer rule places, and starts a Work only when none is open.
        async def start(given, place):
            await place(Start(("D1",)))

        async def skip(given, place):
            return None

        async def boom(given, place):
            raise RuntimeError("conductor down")

        async def cancelled(given, place):
            raise asyncio.CancelledError

        for name, leave in (("placed nothing", skip), ("raised", boom), ("cancelled", cancelled)):
            with self.subTest(name):
                gate = asyncio.Event()
                adapter = ScriptedAdapter([wait(gate), answer("Green teas.")], conductor=planned(start, leave))
                im = self.session(adapter)
                im.admit(ask("Teas."), delegation_id="D1")
                await self.turns()
                second = im.admit(ask("Green only."), delegation_id="D2")
                await self.turns()
                self.assertEqual(im.ledger.delegations["D2"].work, "W1")  # it joined W1: no second Work beside it
                self.assertEqual(list(im.ledger.works), ["W1"])
                gate.set()
                taken = await asyncio.wait_for(read_all(second.stream), 2)
                self.assertEqual(taken[-1]["response"]["output"][-1]["content"][0]["text"], "Green teas.")
                self.assertEqual(im.ledger.delegations["D1"].handed_to, "D2")
                self.assertEqual(len(adapter.runs), 1)


class RecheckTests(IMCase):
    async def check(self, decisions, *, script=None):
        """Place ``decisions`` for the arrival D2 while W1 runs; return the outcome lines."""
        gate = asyncio.Event()
        conductor = ScriptedConductor(lambda given, place: None)

        async def plan(given, place):
            if given.arrivals[0].delegation_id == "D1":
                await place(Start(("D1",)))
                return
            for decision in decisions:
                await place(decision)

        conductor.plan = plan
        adapter = ScriptedAdapter(script or [wait(gate), answer()], [answer()], conductor=conductor, max_running=2)
        im = self.session(adapter)
        self.first = im.admit(ask("Teas."), delegation_id="D1")
        await self.turns()
        self.second = im.admit(ask("Hours."), delegation_id="D2")
        await self.turns()
        gate.set()
        self.im = im
        return conductor.outcomes[1:]

    async def test_a_delegation_is_placed_once_and_its_first_placement_stays(self):
        outcomes = await self.check([Start(("D2",)), Steer("W1", ("D2",)), Answer("D2", "Hi.")])
        refused = TEXTS.refusal("unplaced", delegations="D2")
        self.assertEqual(outcomes[1:], [refused, refused])
        self.assertEqual(self.im.ledger.delegations["D2"].work, "W2")

    async def test_unknown_works_tasks_and_requests_are_refused(self):
        outcomes = await self.check([Steer("W9", ("D2",)), Cancel("W9"), Steer("W1", ("D9",)), Cancel("W1", "W1.7")])
        self.assertTrue(all(outcome.startswith("Not applied") for outcome in outcomes), outcomes)

    async def test_a_steer_to_an_ended_work_starts_a_new_one(self):
        outcomes = await self.check([Steer("W1", ("D2",))], script=[answer("Done early.")])
        self.assertEqual(outcomes, [TEXTS.outcomes["restarted"].format(work="W1", new="W2")])
        await read_all(self.second.stream)
        self.assertEqual(self.im.ledger.delegations["D2"].state, "answered")

    async def test_a_cancel_of_an_ended_work_does_nothing(self):
        outcomes = await self.check([Cancel("W1"), Start(("D2",))], script=[answer("Done early.")])
        self.assertEqual(outcomes[0], TEXTS.outcomes["ended"].format(work="W1"))
        self.assertEqual(self.im.ledger.works["W1"].state, "answered")

    async def test_a_start_over_max_running_is_refused_and_the_steer_rule_joins_its_request_to_the_open_work(self):
        outcomes = await self.check([Start(("D2",), ("hours", "days"))])  # one task runs: two more make three of two
        self.assertEqual(outcomes, [TEXTS.refusal("over_limit", max_running="2")])
        self.assertEqual((self.im.ledger.delegations["D2"].work, sorted(self.im.ledger.works)), ("W1", ["W1"]))

    async def test_a_start_within_max_running_runs_beside_the_open_work(self):
        outcomes = await self.check([Start(("D2",))])
        self.assertEqual(outcomes, [TEXTS.outcomes["started"].format(work="W2", tasks="W2.1")])

    async def test_a_steer_into_an_ended_work_over_max_running_is_refused_and_its_request_joins_the_open_work(self):
        # W1 ended, and W2's two tasks run, as many as run at once: the new Work the steer would start makes three.
        gate = asyncio.Event()
        placements = {"D0": Start(("D0",)), "D1": Start(("D1",), ("teas", "hours")), "D2": Steer("W1", ("D2",))}
        conductor = ScriptedConductor(lambda given, place: place(placements[given.arrivals[0].delegation_id]))
        scripts = [[answer("Done early.")], [wait(gate), answer()], [wait(gate), answer()]]
        im = self.session(ScriptedAdapter(*scripts, conductor=conductor, max_running=2))
        await read_all(im.admit(ask("Menu."), delegation_id="D0").stream)
        first = im.admit(ask("Teas and hours."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("And the days."), delegation_id="D2")
        await self.turns()
        self.assertEqual(conductor.outcomes[-1], TEXTS.refusal("over_limit", max_running="2"))
        self.assertEqual((im.ledger.delegations["D2"].work, sorted(im.ledger.works)), ("W2", ["W1", "W2"]))
        gate.set()
        for stream in (first.stream, second.stream):
            await read_all(stream)


class SteerNewestContractTests(ConductorContract, unittest.IsolatedAsyncioTestCase):
    def make_conductor(self):
        return SteerNewest()
