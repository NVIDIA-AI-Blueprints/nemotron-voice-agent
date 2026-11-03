# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""An answer a task gives while a model conductor decides on newer input (``delegation.hold_answers_while_deciding``).

With the setting on, the answer waits for the decision: steered into its task, it is progress and the task runs one
more round with the newer request; placed elsewhere, it is written once the decision is applied; a stop drops it.
"""

import asyncio
import re
import unittest

from delegation_core.backend.conductors import SteerOpenWork
from delegation_core.backend.port import Answer, Cancel, Start, Steer
from delegation_core.config import DelegationSettings
from tests.unit.contracts.endings import load
from tests.unit.delegation_core.helpers import (
    CORE_SETTINGS,
    CORE_YAML,
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


class HeldAnswerTests(IMCase):
    async def answer_while_deciding(self, decide, *later, hold=True, **settings):
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
        self.im = self.session(self.adapter, hold_answers_while_deciding=hold, **settings)
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

    def released(self) -> list[str]:
        """How each held answer left the hold: written, commentary or dropped."""
        return [fields["how"] for kind, fields in self.traces if kind == "im.answer.released"]

    def held(self) -> int:
        return [kind for kind, _ in self.traces].count("im.answer.held")

    def answer_traces(self) -> list[str]:
        """The traces of answers and decisions, in order, as the core writes them to its log."""
        kinds = ("backend.queued", "im.decided", "im.answer.held", "im.answer.released", "im.answer.written")
        return [kind for kind, _ in self.traces if kind in kinds]

    # ------------------------------------------------------------------------------------- steered into its task
    async def test_steered_into_its_work_the_answer_is_progress_and_the_task_runs_on_with_the_newer_request(self):
        await self.answer_while_deciding(placing(Steer("W1", ("D2",))), [answer(NEW)])
        self.assertEqual(self.answers_so_far(), [])  # held while the conductor decides
        self.assertEqual((self.held(), self.im.ledger.tasks["W1.1"].state), (1, "running"))
        older, newer = await self.decided()
        self.assertEqual(said(older, "final_answer"), [])
        handover = TEXTS.handover_line.format(delegation="D2")
        self.assertEqual(said(older, "commentary"), [OLD, handover])  # a progress note, then the handover
        self.assertEqual(said(newer, "final_answer"), [NEW])
        first, again = self.adapter.runs
        self.assertIs(again.task, first.task)  # one more round of the same task
        history = [str(item) for item in again.histories[0]]
        held_at = next(n for n, item in enumerate(history) if OLD in item)
        self.assertLess(held_at, next(n for n, item in enumerate(history) if "Make it 7." in item))
        self.assertEqual(self.im.ledger.delegations["D1"].handed_to, "D2")
        self.assertEqual(list(self.im.ledger.works), ["W1"])
        self.assertEqual(self.im.ledger.tasks["W1.1"].result.items[-1]["content"][0]["text"], NEW)
        self.assertEqual(self.released(), ["commentary"])
        traces = ["backend.queued", "im.decided"]  # D1 starts W1
        traces += ["backend.queued", "im.answer.held", "im.decided"]  # W1.1's answer waits with D2
        traces += ["im.answer.released", "im.answer.written"]  # progress, then the answer of one more round
        self.assertEqual(self.answer_traces(), traces)

    async def test_steered_into_its_work_the_hold_ends_on_the_wire_as_any_handover(self):
        await self.answer_while_deciding(placing(Steer("W1", ("D2",))), [answer(NEW)], progress_note_seconds=0)
        await self.decided()
        self.assertEqual(self.timeline(), SCENARIOS["handed_over_with_the_taker_named"]["timeline"])

    async def test_a_steer_that_names_its_task_runs_it_on_the_same_way(self):
        await self.answer_while_deciding(placing(Steer("W1", ("D2",), "W1.1")), [answer(NEW)])
        older, newer = await self.decided()
        self.assertEqual((said(older, "final_answer"), said(newer, "final_answer")), ([], [NEW]))
        self.assertEqual([run.task.task_id for run in self.adapter.runs], ["W1.1", "W1.1"])

    async def test_the_steer_rule_after_a_failed_decision_runs_it_on_too(self):
        await self.answer_while_deciding(failing, [answer(NEW)])
        older, newer = await self.decided()
        self.assertEqual((said(older, "final_answer"), said(newer, "final_answer")), ([], [NEW]))
        self.assertEqual(self.im.ledger.delegations["D2"].work, "W1")

    # ------------------------------------------------------------------------------------- placed elsewhere
    async def test_placed_in_a_new_work_the_answer_is_written_once_decided_and_before_its_end(self):
        await self.answer_while_deciding(placing(Start(("D2",))), [answer("A new booking for 7 pm.")])
        self.assertEqual(self.answers_so_far(), [])
        older, newer = await self.decided()
        self.assertEqual(said(older, "final_answer"), [OLD])
        self.assertEqual([event["type"] for event in older[-2:]], ["response.output_item.done", "response.completed"])
        self.assertEqual(said(newer, "final_answer"), ["A new booking for 7 pm."])
        self.assertEqual(self.im.ledger.delegations["D2"].work, "W2")
        self.assertEqual([run.task.task_id for run in self.adapter.runs], ["W1.1", "W2.1"])
        self.assertEqual(self.released(), ["written"])

    async def test_answered_by_the_conductor_the_held_answer_follows_the_status_answer(self):
        await self.answer_while_deciding(placing(Answer("D2", "It is booking the table.")), progress_note_seconds=0)
        older, newer = await self.decided()
        self.assertEqual(said(newer, "final_answer"), ["It is booking the table."])
        self.assertEqual(said(older, "final_answer"), [OLD])
        self.assertEqual(self.timeline(), SCENARIOS["answered_by_the_conductor"]["timeline"])
        self.assertEqual(len(self.adapter.runs), 1)

    async def test_a_stop_of_another_work_writes_the_held_answer_at_once(self):
        never, gate, deciding = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def start(given, place):
            await place(Start(tuple(arrival.delegation_id for arrival in given.arrivals)))

        async def stop_the_other(given, place):
            await deciding.wait()
            await place(Cancel("W1"))
            await place(Answer("D2", "Stopped the menu search."))

        adapter = ScriptedAdapter(
            [wait(never)], [wait(gate), answer(OLD)], conductor=planned(start, start, stop_the_other), max_running=2
        )
        im = self.session(adapter, hold_answers_while_deciding=True)
        other = im.admit(ask("Find the menu."), delegation_id="D0")
        await self.turns()
        first = im.admit(ask("Book a table for 6 pm."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("Stop the menu search."), delegation_id="D2")
        await self.turns()
        gate.set()  # W2 answers while the conductor decides
        await self.turns()
        self.assertEqual(self.answers_so_far(), [])
        deciding.set()
        self.assertEqual(said(await read(second), "final_answer"), ["Stopped the menu search."])
        self.assertEqual(said(await read(other), "final_answer"), [TEXTS.cancelled_line])
        self.assertEqual(said(await read(first), "final_answer"), [OLD])
        self.assertEqual((im.ledger.works["W1"].state, im.ledger.works["W2"].state), ("cancelled", "answered"))

    async def test_a_steer_into_another_task_of_its_work_writes_the_held_answer_at_once(self):
        flights, hotels, deciding = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def split(given, place):
            await place(Start(("D1",), ("flights", "hotels")))

        async def steer_the_hotels(given, place):
            await deciding.wait()
            await place(Steer("W1", ("D2",), "W1.2"))

        adapter = ScriptedAdapter(
            [wait(flights), answer("Flights booked.")],
            [wait(hotels), checkpoint(), answer("A hotel by the airport.")],
            conductor=planned(split, steer_the_hotels),
            max_running=2,
        )
        im = self.session(adapter, hold_answers_while_deciding=True)
        first = im.admit(ask("Book flights and a hotel."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("A hotel by the airport."), delegation_id="D2")
        await self.turns()
        flights.set()  # the flights task answers while the conductor decides
        await self.turns()
        self.assertEqual(self.answers_so_far(), [])
        deciding.set()
        await self.turns()
        self.assertEqual(self.answers_so_far(), ["Flights booked."])  # the other task took the newer request
        hotels.set()
        await read(first)
        self.assertEqual(said(await read(second), "final_answer"), ["A hotel by the airport."])
        self.assertEqual([run.task.task_id for run in adapter.runs], ["W1.1", "W1.2"])

    # ------------------------------------------------------------------------------------- stopped
    async def test_a_stop_of_its_work_drops_the_answer_and_the_stop_says_its_line(self):
        decide = placing(Cancel("W1"), Answer("D2", "Stopped the booking."))
        await self.answer_while_deciding(decide, progress_note_seconds=0)
        older, newer = await self.decided()
        self.assertEqual(said(older, "final_answer"), [TEXTS.cancelled_line])
        self.assertEqual(said(newer, "final_answer"), ["Stopped the booking."])
        self.assertNotIn(OLD, str(older))
        self.assertEqual(self.timeline(), SCENARIOS["cancelled_for_a_newer_request_stopped"]["timeline"])
        self.assertEqual((self.released(), self.im.ledger.works["W1"].state), (["dropped"], "cancelled"))

    async def test_a_stop_of_its_task_drops_the_answer_the_same_way(self):
        await self.answer_while_deciding(placing(Cancel("W1", "W1.1"), Answer("D2", "Stopped the booking.")))
        older, _ = await self.decided()
        self.assertEqual(said(older, "final_answer"), [TEXTS.cancelled_line])
        self.assertEqual(self.released(), ["dropped"])

    # ------------------------------------------------------------------------------------- unchanged
    async def test_with_no_decision_waiting_an_answer_is_written_as_it_comes(self):
        im = self.session(ScriptedAdapter([answer()]), hold_answers_while_deciding=True)
        await read(im.admit(ask("Find teas."), delegation_id="D1"))
        self.assertEqual(self.timeline(), SCENARIOS["answered"]["timeline"])
        self.assertEqual(self.held(), 0)

    async def test_off_the_answer_is_written_at_once_and_the_correction_starts_a_new_work(self):
        await self.answer_while_deciding(placing(Steer("W1", ("D2",))), [answer(NEW)], hold=False)
        self.assertEqual(self.answers_so_far(), [OLD])  # said while the conductor still decides
        older, newer = await self.decided()
        self.assertEqual((said(older, "final_answer"), said(newer, "final_answer")), ([OLD], [NEW]))
        self.assertEqual(self.im.ledger.delegations["D2"].work, "W2")
        outcomes = [fields["outcome"] for kind, fields in self.traces if kind == "im.placed"]
        self.assertEqual(outcomes[-1], TEXTS.outcomes["restarted"].format(work="W1", new="W2"))
        traces = ["backend.queued", "im.decided"]  # D1 starts W1
        traces += ["backend.queued", "im.answer.written", "im.decided"]  # written while D2 waits: stale
        traces += ["im.answer.written"]  # W2's answer
        self.assertEqual(self.answer_traces(), traces)

    async def test_off_every_other_placement_finds_the_answer_written_already(self):
        decisions = {
            "a new work": placing(Start(("D2",))),
            "a status answer": placing(Answer("D2", "It is booked.")),
            "a stop": placing(Cancel("W1"), Answer("D2", "Nothing to stop.")),
        }
        for name, decide in decisions.items():
            with self.subTest(name):
                self.wire.events.clear()
                await self.answer_while_deciding(decide, [answer(NEW)], hold=False)
                self.assertEqual(self.answers_so_far(), [OLD])
                older, _ = await self.decided()
                self.assertEqual(said(older, "final_answer"), [OLD])
        self.assertEqual(self.held(), 0)

    async def test_with_a_rule_conductor_that_decides_at_once_nothing_changes(self):
        timelines = []
        for hold in (False, True):
            self.wire.events.clear()
            gate = asyncio.Event()
            adapter = ScriptedAdapter([wait(gate), checkpoint(), answer(NEW)], conductor=SteerOpenWork())
            im = self.session(adapter, hold_answers_while_deciding=hold)
            first = im.admit(ask("Book a table for 6 pm."), delegation_id="D1")
            await self.turns()
            second = im.admit(ask("Make it 7."), delegation_id="D2")
            await self.turns()  # placed at once: it waits in the running task as steering
            gate.set()
            await read(first)
            self.assertEqual(said(await read(second), "final_answer"), [NEW])
            timelines.append(self.timeline())
        self.assertEqual(timelines[0], timelines[1])
        self.assertEqual(self.held(), 0)


class SettingTests(unittest.TestCase):
    def test_it_is_off_in_core_yaml_and_takes_true_or_false_only(self):
        line = re.compile(r"^  hold_answers_while_deciding: false +# \S", re.MULTILINE)
        self.assertRegex(CORE_YAML.read_text(), line)  # written out, with its comment
        self.assertFalse(CORE_SETTINGS.delegation.hold_answers_while_deciding)
        self.assertTrue(DelegationSettings.parse({"hold_answers_while_deciding": True}).hold_answers_while_deciding)
        for value in ("yes", 1, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DelegationSettings.parse({"hold_answers_while_deciding": value})


if __name__ == "__main__":
    unittest.main()
