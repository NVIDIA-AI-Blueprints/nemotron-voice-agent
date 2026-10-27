# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Steering: landing, delivery reports, follow-ups and settings per response."""

import asyncio

from delegation_core.backend.port import Start, Steer
from tests.unit.delegation_core.helpers import (
    MENU,
    IMCase,
    ScriptedAdapter,
    ScriptedConductor,
    answer,
    app_call,
    app_outputs,
    ask,
    checkpoint,
    finished_items,
    read_all,
    wait,
)


def model_of(run) -> list[str]:
    return [request["model"] for request in run.requests]


class SteeringTests(IMCase):
    async def test_a_running_task_is_woken_and_takes_the_steering_at_its_next_safe_point(self):
        gate = asyncio.Event()
        adapter = ScriptedAdapter([wait(gate), checkpoint(), answer()])
        im = self.session(adapter)
        im.admit(ask("Book Denver."), delegation_id="D1")
        await self.turns()
        im.admit(ask("Make it Tuesday."), delegation_id="D2")
        await self.turns()
        run = adapter.runs[0]
        self.assertEqual((run.steers, len(run.applied)), (1, 0))
        self.assertEqual(im.ledger.delegations["D2"].state, "queued")
        gate.set()
        await self.turns()
        self.assertEqual(len(run.applied), 1)
        self.assertEqual(im.ledger.delegations["D2"].state, "answered")

    async def test_not_applied_steering_waits_for_the_next_point_then_becomes_a_follow_up_work(self):
        gate = asyncio.Event()
        adapter = ScriptedAdapter([wait(gate), checkpoint(), checkpoint(), answer("Old.")], [answer("New.")])
        adapter.delivery = "not_applied"
        im = self.session(adapter)
        first = im.admit(ask("Book Denver."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("Make it Tuesday."), delegation_id="D2")
        await self.turns()
        gate.set()
        older, newer = await read_all(first.stream), await read_all(second.stream)
        self.assertEqual(len([b for b in adapter.runs[0].boundaries if b.steering is not None]), 3)
        self.assertEqual(older[-1]["response"]["output"][-1]["content"][0]["text"], "Old.")
        self.assertEqual(newer[-1]["response"]["output"][-1]["content"][0]["text"], "New.")
        self.assertEqual(im.ledger.delegations["D2"].work, "W2")
        self.assertNotIn("Make it Tuesday.", str(adapter.runs[0].histories[-1]))  # never applied, never in history
        self.assertIn("Make it Tuesday.", str(adapter.runs[1].histories[0]))

    async def test_a_steer_named_for_one_task_reaches_only_that_task(self):
        flights, hotels = asyncio.Event(), asyncio.Event()

        async def plan(given, place):
            ids = tuple(arrival.delegation_id for arrival in given.arrivals)
            if ids == ("D1",):
                await place(Start(ids, ("flights", "hotels")))
            else:
                await place(Steer("W1", ids, "W1.2"))

        adapter = ScriptedAdapter(
            [wait(flights), checkpoint(), answer("Flights.")],
            [wait(hotels), checkpoint(), answer("Hotels.")],
            conductor=ScriptedConductor(plan),
            max_running=2,
        )
        im = self.session(adapter)
        first = im.admit(ask("Flights and hotels."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("Hotels near the airport."), delegation_id="D2")
        await self.turns()
        hotels.set()
        await read_all(first.stream)  # handed over when the hotels task applied the steering
        flights.set()
        taken = await read_all(second.stream)
        self.assertEqual([len(run.applied) for run in adapter.runs], [0, 1])
        self.assertEqual([item["content"][0]["text"] for item in finished_items(taken)], ["Hotels.", "Flights."])
        self.assertNotIn("Hotels near the airport.", str(adapter.runs[0].histories))

    async def test_a_join_keeps_the_works_settings(self):
        gate = asyncio.Event()
        adapter = ScriptedAdapter([wait(gate), checkpoint(), answer()])
        im = self.session(adapter)
        im.admit(ask("Book Denver."), delegation_id="D1", settings={"model": "model/one"})
        await self.turns()
        im.admit(ask("Make it Tuesday."), delegation_id="D2", settings={"model": "model/two"})
        await self.turns()
        gate.set()
        await self.turns()
        self.assertEqual(model_of(adapter.runs[0]), ["model/one"])

    async def test_a_typed_run_steered_in_brings_its_settings(self):
        gate = asyncio.Event()
        adapter = ScriptedAdapter([wait(gate), checkpoint(), answer()])
        im = self.session(adapter)
        im.admit(ask("Book Denver."), delegation_id="D1", settings={"model": "model/one"})
        await self.turns()
        im.admit_client_request(ask("Typed: Tuesday."), delegation_id="D2", settings={"model": "model/two"})
        await self.turns()
        gate.set()
        await self.turns()
        self.assertEqual(model_of(adapter.runs[0]), ["model/two"])

    async def test_a_settings_update_before_a_continuation_reaches_the_next_round(self):
        adapter = ScriptedAdapter([app_call("get_menu"), answer()])
        im = self.session(adapter)
        before = {"tools": [MENU], "model": "model/one"}
        first = await read_all(im.admit(ask("Menu?"), delegation_id="D1", settings=before).stream)
        after = {"tools": [MENU], "model": "model/two", "instructions": "Answer in French."}
        await read_all(im.admit_client_request(app_outputs(first), delegation_id="D1", settings=after).stream)
        requests = adapter.runs[0].requests
        self.assertEqual([request["model"] for request in requests], ["model/one", "model/two"])
        self.assertTrue(requests[1]["instructions"].startswith("Answer in French."))
