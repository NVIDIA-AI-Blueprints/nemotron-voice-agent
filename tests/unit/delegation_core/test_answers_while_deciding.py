# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""An answer a task gives while the conductor decides on newer input is written at once; the endings decisions give.

The frontend words a result that newer requests may change, so nothing waits for the decision: a correction steered
into the answered Work starts a new one. The endings fixture's scenarios of a decision are checked on the IM's wire.
"""

import asyncio
import unittest

from examples.frontend_backend_live.delegation_core.backend.conductors import SteerOpenWork
from examples.frontend_backend_live.delegation_core.backend.port import Answer, Cancel, Start, Steer
from tests.unit.contracts.endings import load
from tests.unit.delegation_core.helpers import (
    TEXTS,
    IMCase,
    ScriptedAdapter,
    answer,
    ask,
    checkpoint,
    finished_items,
    planned,
    read_all,
    wait,
)

SCENARIOS = load()
OLD, NEW = "Booked for 6 pm.", "Booked for 7 pm."
# A response that does not end within this many seconds fails its test instead of hanging the suite.
READ_LIMIT_SECONDS = 2


def said(events: list[dict], phase: str) -> list[str]:
    """The texts of a response's messages of ``phase``, in order."""
    return [
        item["content"][0]["text"]
        for item in finished_items(events)
        if item["type"] == "message" and item["phase"] == phase
    ]


async def read(admission) -> list[dict]:
    return await asyncio.wait_for(read_all(admission.stream), READ_LIMIT_SECONDS)


def placing(*decisions):
    """A decision round that places ``decisions`` in order."""

    async def place_all(place):
        for decision in decisions:
            await place(decision)

    return place_all


async def failing(place):
    """A decision round whose model is down: it places nothing and raises."""
    raise RuntimeError("The conductor's model is down.")


class AnswerWhileDecidingTests(IMCase):
    async def answer_while_deciding(self, decide, *later, **settings):
        """W1.1 answers OLD while the conductor decides on D2 ("Make it 7."); ``decide`` places once it is let go.

        ``later`` are the scripts of the runs that start after W1.1's first one.
        """
        gate, self.deciding = asyncio.Event(), asyncio.Event()

        async def start(given, place):
            await place(Start(("D1",)))

        async def once_let_go(given, place):
            await self.deciding.wait()  # a model conductor's call: the newer request waits for it
            await decide(place)

        self.adapter = ScriptedAdapter([wait(gate), answer(OLD)], *later, conductor=planned(start, once_let_go))
        self.im = self.session(self.adapter, **settings)
        self.first = self.im.admit(ask("Book a table for 6 pm."), delegation_id="D1")
        await self.turns()
        self.second = self.im.admit(ask("Make it 7."), delegation_id="D2")
        await self.turns()
        gate.set()  # the round answers while the conductor still decides
        await self.turns()

    async def decided(self) -> tuple[list[dict], list[dict]]:
        """The conductor decides: both responses, read to their ends."""
        self.deciding.set()
        return await read(self.first), await read(self.second)

    def answers_so_far(self) -> list[str]:
        """Every final answer on the wire so far, in order."""
        return [
            event["item"]["content"][0]["text"]
            for _, event in self.wire.events
            if event["type"] == "response.output_item.done" and event["item"].get("phase") == "final_answer"
        ]

    def answer_traces(self) -> list[str]:
        """The traces of answers and decisions, in order, as the core writes them to its log."""
        kinds = ("backend.queued", "im.decided", "im.answer.written")
        return [kind for kind, _ in self.traces if kind in kinds]

    async def while_running(self, decide):
        """W1.1 runs until it is let go; the conductor places D2 with ``decide`` while it runs."""
        gate = asyncio.Event()

        async def start(given, place):
            await place(Start(("D1",)))

        async def on_d2(given, place):
            await decide(place)

        self.gate = gate
        self.adapter = ScriptedAdapter([wait(gate), answer(OLD)], conductor=planned(start, on_d2))
        self.im = self.session(self.adapter, progress_note_seconds=0)
        self.first = self.im.admit(ask("Book a table for 6 pm."), delegation_id="D1")
        await self.turns()
        self.second = self.im.admit(ask("Make it 7."), delegation_id="D2")
        await self.turns()

    # ------------------------------------------------------------------------------------- written at once
    async def test_the_answer_is_written_at_once_and_a_correction_steered_into_its_work_starts_a_new_one(self):
        await self.answer_while_deciding(placing(Steer("W1", ("D2",))), [answer(NEW)])
        self.assertEqual(self.answers_so_far(), [OLD])  # said while the conductor still decides
        older, newer = await self.decided()
        self.assertEqual((said(older, "final_answer"), said(newer, "final_answer")), ([OLD], [NEW]))
        self.assertEqual(self.im.ledger.delegations["D2"].work, "W2")
        outcomes = [fields["outcome"] for kind, fields in self.traces if kind == "im.placed"]
        self.assertEqual(outcomes[-1], TEXTS.outcomes["restarted"].format(work="W1", new="W2"))
        traces = ["backend.queued", "im.decided"]  # D1 starts W1
        traces += ["backend.queued", "im.answer.written", "im.decided"]  # written while D2 waits
        traces += ["im.answer.written"]  # W2's answer
        self.assertEqual(self.answer_traces(), traces)

    async def test_every_other_placement_finds_the_answer_written_already(self):
        decisions = {
            "a new work": placing(Start(("D2",))),
            "a status answer": placing(Answer("D2", "It is booked.")),
            "a stop": placing(Cancel("W1"), Answer("D2", "Nothing to stop.")),
        }
        for name, decide in decisions.items():
            with self.subTest(name):
                self.wire.events.clear()
                await self.answer_while_deciding(decide, [answer(NEW)])
                self.assertEqual(self.answers_so_far(), [OLD])
                older, _ = await self.decided()
                self.assertEqual(said(older, "final_answer"), [OLD])

    # ------------------------------------------------------------------------------------- the endings
    async def test_with_no_decision_waiting_an_answer_is_written_as_it_comes(self):
        im = self.session(ScriptedAdapter([answer()]))
        await read(im.admit(ask("Find teas."), delegation_id="D1"))
        self.assertEqual(self.timeline(), SCENARIOS["answered"]["timeline"])

    async def test_a_request_the_steer_rule_places_in_the_running_work_takes_it_over(self):
        gate = asyncio.Event()
        adapter = ScriptedAdapter([wait(gate), checkpoint(), answer(NEW)], conductor=SteerOpenWork())
        im = self.session(adapter, progress_note_seconds=0)
        first = im.admit(ask("Book a table for 6 pm."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("Make it 7."), delegation_id="D2")
        await self.turns()  # placed at once: it waits in the running task as steering
        gate.set()
        await read(first)
        self.assertEqual(said(await read(second), "final_answer"), [NEW])
        self.assertEqual(self.timeline(), SCENARIOS["handed_over_with_the_taker_named"]["timeline"])

    async def test_answered_by_the_conductor_the_work_it_asked_about_runs_on(self):
        await self.while_running(placing(Answer("D2", "It is booking the table.")))
        self.assertEqual(said(await read(self.second), "final_answer"), ["It is booking the table."])
        self.gate.set()
        self.assertEqual(said(await read(self.first), "final_answer"), [OLD])
        self.assertEqual(self.timeline(), SCENARIOS["answered_by_the_conductor"]["timeline"])

    async def test_a_work_cancelled_for_a_newer_request_answers_with_the_cancelled_line(self):
        await self.while_running(placing(Cancel("W1"), Answer("D2", "Stopped the booking.")))
        older, newer = await read(self.first), await read(self.second)
        self.assertEqual(said(older, "final_answer"), [TEXTS.cancelled_line])
        self.assertEqual(said(newer, "final_answer"), ["Stopped the booking."])
        self.assertNotIn(OLD, str(older))
        self.assertEqual(self.timeline(), SCENARIOS["cancelled_for_a_newer_request_stopped"]["timeline"])
        self.assertEqual(self.im.ledger.works["W1"].state, "cancelled")


if __name__ == "__main__":
    unittest.main()
