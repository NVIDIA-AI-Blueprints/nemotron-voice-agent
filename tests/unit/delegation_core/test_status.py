# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The status path: conductor answers from the records, and status lines as commentary."""

import asyncio

from delegation_core.backend.port import Answer, Start
from tests.unit.delegation_core.helpers import (
    TEXTS,
    IMCase,
    ScriptedAdapter,
    answer,
    ask,
    finished_items,
    own_call,
    planned,
    read_all,
    say,
    wait,
)


class StatusTests(IMCase):
    async def test_an_answer_ends_the_question_and_leaves_the_work_running(self):
        gate = asyncio.Event()

        async def start(given, place):
            await place(Start(("D1",)))

        async def status(given, place):
            (work,) = given.works
            await place(Answer("D2", f"Still on it: {work.tasks[0].steps or 'no steps yet'}."))

        script = [own_call("search_flights"), wait(gate), answer("Booked.")]
        im = self.session(ScriptedAdapter(script, conductor=planned(start, status)))
        first = im.admit(ask("Book Denver."), delegation_id="D1")
        await self.turns()
        events = await read_all(im.admit(ask("How is it going?"), delegation_id="D2").stream)
        self.assertEqual(finished_items(events)[0]["content"][0]["text"], "Still on it: ('search_flights: done',).")
        self.assertEqual((im.ledger.works["W1"].state, im.ledger.works["W1"].carrier), ("running", "D1"))
        self.assertEqual(im.ledger.delegations["D2"].state, "answered")
        gate.set()
        done = await read_all(first.stream)
        self.assertEqual(done[-1]["response"]["output"][-1]["content"][0]["text"], "Booked.")

    async def test_a_joined_status_question_answered_beside_a_call_keeps_the_work_running(self):
        gate = asyncio.Event()
        script = [wait(gate), say("I am still searching flights."), own_call("search_flights"), answer("Booked.")]
        adapter = ScriptedAdapter(script)
        im = self.session(adapter)
        first = im.admit(ask("Book Denver."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("Any update?"), delegation_id="D2")
        await self.turns()
        gate.set()
        await read_all(first.stream)
        events = await read_all(second.stream)
        said = [(item.get("phase"), item.get("content", [{}])[0].get("text")) for item in finished_items(events)]
        self.assertEqual(said[0], ("commentary", "I am still searching flights."))
        self.assertEqual(said[-1], ("final_answer", "Booked."))
        self.assertIn(TEXTS.closing_line, str(adapter.runs[0].histories[-1]))
