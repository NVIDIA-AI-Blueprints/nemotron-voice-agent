# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""An IM session's Works: the scheduler, placement and its checks, steering, handovers, stops and the endings."""

import asyncio
from itertools import pairwise

from examples.frontend_backend_live.delegation_core.backend.conductors import SteerOpenWork
from examples.frontend_backend_live.delegation_core.backend.port import Answer, Cancel, Start, Steer
from tests.unit.core_helpers import load
from tests.unit.delegation_core.helpers import (
    MENU,
    TEXTS,
    Fail,
    IMCase,
    OwnCall,
    ScriptedAdapter,
    ScriptedConductor,
    ScriptedRun,
    StartEach,
    answer,
    answers,
    app_call,
    app_outputs,
    ask,
    checkpoint,
    finished_items,
    function_calls,
    is_handover_line,
    placing,
    planned,
    progress,
    read_all,
    said,
    texts,
    wait,
)

APP = {"tools": [MENU]}
REFUSED = TEXTS.outcomes["refused"].split("{")[0]
HANDOVER, CANCELLED, OWED = TEXTS.handover_line.format(delegation="D2"), TEXTS.cancelled_line, TEXTS.owed_output
LATE = ["response.created", "response.queued", "response.in_progress", "response.completed"]
ENDINGS = load()
OLD, NEW = "Booked for 6 pm.", "Booked for 7 pm."


def starts(given, place):
    return place(Start(tuple(arrival.delegation_id for arrival in given.arrivals)))


def split(*briefs: str):
    return placing(Start(("D1",), briefs))


def then(*decisions):
    """Build a conductor that starts D1's Work, then places ``decisions`` for the next arrival."""
    return planned(placing(Start(("D1",))), placing(*decisions))


def named(events: list[dict]) -> str | None:
    """Return the response a taker's in_progress names as the one it took over."""
    return next(e for e in events if e["type"] == "response.in_progress")["response"].get("previous_response_id")


def backend_calls(events: list[dict]) -> list[tuple[str, str]]:
    """Return the name and status of each backend call item a response's events finished, in order."""
    return [(item["name"], item["status"]) for item in finished_items(events) if item["type"] == "mcp_call"]


async def read(admission) -> list[dict]:
    return await asyncio.wait_for(read_all(admission.stream), 2)


class WorksCase(IMCase):
    def assertHandedOver(self, older: list[dict]) -> None:
        """Assert one taker's in_progress names ``older`` before it ends with only the handover line, then completed."""
        wire = [(response_id, event["type"], event.get("response", {})) for response_id, event in self.wire.events]
        older_id = older[0]["response"]["id"]
        takers = [n for n, (_, kind, response) in enumerate(wire) if response.get("previous_response_id") == older_id]
        line = wire.index((older_id, "response.output_item.done", {}))
        self.assertEqual(len({wire[n][0] for n in takers}), 1)
        self.assertLess(takers[0], line)
        (text,) = texts(older)
        self.assertEqual((said(older)[0][0], is_handover_line(text)), ("commentary", True))
        self.assertEqual(older[-1]["type"], "response.completed")


class PlacementTests(WorksCase):
    async def test_a_task_waiting_for_the_app_holds_no_slot(self):
        gates = [asyncio.Event(), asyncio.Event()]
        scripts = [[app_call(), answer("One.")], [wait(gates[0]), Fail()], [wait(gates[1]), answer()]]
        adapter = ScriptedAdapter(*scripts, conductor=StartEach(), max_running=2)
        im = self.session(adapter)
        first = await read_all(im.admit(ask("Request 1."), delegation_id="D1", settings=APP).stream)
        streams = [im.admit(ask(f"Request {n}."), delegation_id=f"D{n}").stream for n in (2, 3)]
        await self.turns()
        self.assertEqual(len(adapter.runs), 3)
        resumed = im.admit_client_request(app_outputs(first), delegation_id="D1", settings=APP).stream
        await self.turns()
        self.assertEqual(adapter.runs[0].resumes, 0)
        gates[0].set()
        self.assertEqual((await read_all(streams[0]))[-1]["type"], "response.failed")
        self.assertEqual(texts(await read_all(resumed)), ["One."])
        gates[1].set()
        await read_all(streams[1])

    async def test_an_arrival_restarts_a_decide_that_placed_nothing(self):
        thinking, calls, kept = asyncio.Event(), [], []

        async def slow(given, place):
            calls.append([arrival.delegation_id for arrival in given.arrivals])
            kept.append(place)
            await thinking.wait()
            for arrival in given.arrivals:
                await place(Start((arrival.delegation_id,)))

        im = self.session(ScriptedAdapter(conductor=ScriptedConductor(slow), max_running=3))
        await self.admit_each(im, "One.", "Two.")
        self.assertTrue((await kept[0](Start(("D1",)))).startswith(REFUSED))
        im.admit(ask("Three."), delegation_id="D3")
        await self.turns()
        thinking.set()
        await self.turns()
        self.assertEqual(calls, [["D1"], ["D1", "D2"], ["D3"]])

    async def test_an_arrival_after_a_placement_waits_for_the_next_decide(self):
        placed, release, calls = asyncio.Event(), asyncio.Event(), []

        async def plan(given, place):
            calls.append([arrival.delegation_id for arrival in given.arrivals])
            await starts(given, place)
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

    async def test_a_conductor_reads_what_is_new_since_it_last_read(self):
        gate, conductor = asyncio.Event(), ScriptedConductor(starts)
        scripts = ([answer()], [OwnCall("lookup"), progress("Checking stock."), wait(gate), answer()])
        im = self.session(ScriptedAdapter(*scripts, conductor=conductor, max_running=2))
        await read_all(im.admit(ask("Teas?"), delegation_id="D1").stream)
        for text, delegation in (("Hours?", "D2"), ("Coffee?", "D3")):
            im.admit(ask(text), delegation_id=delegation)
            await self.turns()
        first, second, third = conductor.given
        self.assertEqual((first.conversation, first.new_from, second.new_from), ((), 0, 0))
        self.assertTrue(second.conversation)
        self.assertEqual(third.new_from, len(third.conversation))
        works = [(work.work_id, work.state, work.carrier) for work in third.works]
        self.assertEqual(works, [("W2", "running", "D2"), ("W1", "answered", "D1")])
        (task,) = third.works[0].tasks
        self.assertEqual((task.state, task.steps, task.acted), ("running", ("lookup: done",), True))
        self.assertEqual(task.progress, "Checking stock.")
        gate.set()

    async def test_overlapping_places_apply_in_order(self):
        kept = []

        async def parallel(given, place):
            kept.append(place)
            await asyncio.gather(place(Start(("D1",))), place(Steer("W1", ("D2",))))

        conductor = ScriptedConductor(parallel)
        im = self.session(ScriptedAdapter([answer()], conductor=conductor))
        first = im.admit(ask("Teas."), delegation_id="D1")
        im.admit(ask("Green only."), delegation_id="D2")
        await read_all(first.stream)
        outcomes = [TEXTS.outcomes[kind].format(work="W1", tasks="W1.1") for kind in ("started", "steered")]
        self.assertEqual(conductor.outcomes, outcomes)
        self.assertTrue((await kept[0](Start(("D2",)))).startswith(REFUSED))

    async def test_an_unplaced_request_starts_or_joins_a_work(self):
        async def boom(given, place):
            raise RuntimeError("conductor down")

        async def cancelled(given, place):
            raise asyncio.CancelledError

        for name, leave in (("placed nothing", placing()), ("raised", boom), ("cancelled", cancelled)):
            with self.subTest(name):
                gate = asyncio.Event()
                adapter = ScriptedAdapter([wait(gate), answer("Green teas.")], conductor=planned(leave, leave))
                first, second = await self.admit_each(self.session(adapter), "Teas.", "Green only.")
                gate.set()
                older, taker = await read_all(first), await read_all(second)
                self.assertEqual((texts(older), texts(taker), len(adapter.runs)), ([HANDOVER], ["Green teas."], 1))
                self.assertHandedOver(older)

    async def test_each_decision_is_checked_against_what_it_names(self):
        refusal, outcome = TEXTS.refusal, TEXTS.outcomes
        started, ended = outcome["started"].format(work="W2", tasks="W2.1"), [answer("Done early.")]
        unknown = [Steer("W9", ("D2",)), Cancel("W9"), Steer("W1", ("D9",)), Cancel("W1", "W1.7"), Start(("D2",))]
        unknown_lines = [refusal("no_work", work="W9")] * 2 + [refusal("unplaced", delegations="D9")]
        unknown_lines += [refusal("no_task", task="W1.7", work="W1"), started]
        once = [Start(("D2",)), Steer("W1", ("D2",)), Answer("D2", "Hi.")]
        cases = {
            "placed once": (once, None, [started, *[refusal("unplaced", delegations="D2")] * 2]),
            "unknown names": (unknown, None, unknown_lines),
            "an ended work": ([Steer("W1", ("D2",))], ended, [outcome["restarted"].format(work="W1", new="W2")]),
            "its cancel": ([Cancel("W1"), Start(("D2",))], ended, [outcome["ended"].format(work="W1"), started]),
            "over max_running": ([Start(("D2",), ("hours", "days"))], None, [refusal("over_limit", max_running="2")]),
        }
        for name, (decisions, script, lines) in cases.items():
            with self.subTest(name):
                gate, conductor = asyncio.Event(), then(*decisions)
                scripts = (script or [wait(gate), answer()], [answer()])
                adapter = ScriptedAdapter(*scripts, conductor=conductor, max_running=2)
                _, second = await self.admit_each(self.session(adapter), "Teas.", "Hours.")
                gate.set()
                self.assertEqual(conductor.outcomes[1:], lines)
                self.assertEqual(texts(await read_all(second)), ["Done."])
        self.assertEqual(len(adapter.runs), 1)

    async def test_a_restart_over_max_running_is_refused(self):
        gate, splits = asyncio.Event(), placing(Start(("D2",), ("teas", "hours")))
        conductor = planned(placing(Start(("D1",))), splits, placing(Steer("W1", ("D3",))))
        scripts = [[answer("Done early.")], [wait(gate), answer()], [wait(gate), checkpoint(), answer()]]
        adapter = ScriptedAdapter(*scripts, conductor=conductor, max_running=2)
        im = self.session(adapter)
        await read_all(im.admit(ask("Menu."), delegation_id="D1").stream)
        im.admit(ask("Teas and hours."), delegation_id="D2")
        await self.turns()
        third = im.admit(ask("And the days."), delegation_id="D3").stream
        await self.turns()
        self.assertEqual(conductor.outcomes[-1], TEXTS.refusal("over_limit", max_running="2"))
        gate.set()
        self.assertEqual((len(texts(await read_all(third))), len(adapter.runs)), (2, 3))


class SteeringTests(WorksCase):
    async def test_not_applied_steering_becomes_a_follow_up_work(self):
        gate = asyncio.Event()
        script = [wait(gate), checkpoint(), checkpoint(), answer("Old.")]
        adapter = ScriptedAdapter(script, [answer("New.")], delivery="not_applied")
        first, second = await self.admit_each(self.session(adapter), "Order a latte.", "Make it large.")
        gate.set()
        self.assertEqual((texts(await read_all(first)), texts(await read_all(second))), (["Old."], ["New."]))
        self.assertEqual(len([b for b in adapter.runs[0].boundaries if b.steering is not None]), 3)
        self.assertNotIn("Make it large.", str(adapter.runs[0].histories[-1]))
        self.assertIn("Make it large.", str(adapter.runs[1].histories[0]))

    async def test_three_requests_hand_over_as_a_chain_to_the_newest(self):
        adapter = ScriptedAdapter([answer()], conductor=ScriptedConductor(starts))
        im = self.session(adapter)
        streams = [im.admit(ask(f"Part {n}."), delegation_id=f"D{n}").stream for n in (1, 2, 3)]
        events = [await read_all(stream) for stream in streams]
        for older, taker in pairwise(events):
            self.assertEqual(named(taker), older[0]["response"]["id"])
            self.assertHandedOver(older)
        self.assertEqual(texts(events[2]), ["Done."])
        self.assertIn("Part 1.", str(adapter.runs[0].histories[0]))
        self.assertIn("Part 3.", str(adapter.runs[0].histories[0]))

    async def test_a_handover_waits_for_the_carrier_to_reopen(self):
        gate = asyncio.Event()
        scripts = ([app_call("get_menu"), answer("Menu.")], [wait(gate), checkpoint(), answer("Hours, for green tea.")])
        conductor = planned(split("menu", "hours"), placing(Steer("W1", ("D2",), "W1.2")))
        im = self.session(ScriptedAdapter(*scripts, conductor=conductor, max_running=2))
        first = await read_all(im.admit(ask("Menu and hours."), delegation_id="D1", settings=APP).stream)
        second = im.admit(ask("Green tea only."), delegation_id="D2")
        await self.turns()
        gate.set()
        await self.turns()
        resumed = await self.answer_calls(im, first, settings=APP)
        taker = await read_all(second.stream)
        self.assertEqual(named(taker), resumed[0]["response"]["id"])
        self.assertHandedOver(resumed)
        self.assertEqual(texts(taker), ["Hours, for green tea.", "Menu."])

    async def test_requests_landing_out_of_order_hand_over_in_turn(self):
        milk, pastry = asyncio.Event(), asyncio.Event()
        latte, croissant = Steer("W1", ("D2",), "W1.1"), Steer("W1", ("D3",), "W1.2")
        conductor = planned(split("latte", "croissant"), placing(latte), placing(croissant))
        scripts = (
            [OwnCall("check_milk", gate=milk), checkpoint(), answer("Oat latte.")],
            [OwnCall("check_pastry", gate=pastry), checkpoint(), answer("Warm croissant.")],
        )
        im = self.session(ScriptedAdapter(*scripts, conductor=conductor, max_running=2))
        streams = await self.admit_each(im, "A latte and a croissant.", "Oat milk.", "Warm it.")
        reading = asyncio.ensure_future(read_all(streams[0]))
        pastry.set()
        await self.until(lambda: im.ledger.tasks["W1.2"].ended, "the croissant task's answer")
        await self.turns()
        self.assertEqual((im.ledger.works["W1"].carrier, reading.done()), ("D3", False))
        milk.set()
        first = await asyncio.wait_for(reading, 2)
        second, third = [await asyncio.wait_for(read_all(stream), 2) for stream in streams[1:]]
        work = im.ledger.works["W1"]
        self.assertEqual((work.carrier, work.requests), ("D2", ["D1", "D3", "D2"]))
        self.assertEqual((named(third), named(second)), (first[0]["response"]["id"], third[0]["response"]["id"]))
        self.assertEqual(texts(third), ["Warm croissant.", TEXTS.handover_line.format(delegation="D2")])
        self.assertEqual(texts(second), ["Oat latte."])
        self.assertEqual(backend_calls(first), [("check_pastry", "completed"), ("check_milk", "completed")])
        self.assertEqual(said(first)[-1], ("commentary", TEXTS.handover_line.format(delegation="D3")))
        self.assertEqual(finished_items(first)[-1]["type"], "message")
        self.assertEqual(backend_calls(third), [])


class EndingsTests(IMCase):
    """The endings table's scenarios of a decision, as the IM writes them; an answer never waits for a decision."""

    async def booking(self, decide, *later, deciding=False, **settings):
        """Admit D1, whose W1.1 runs until ``self.gate`` lets it go, then D2 ("Make it 7."), placed with ``decide``.

        With ``deciding``, ``decide`` places only once ``self.deciding`` is set.
        """
        self.gate, self.deciding = asyncio.Event(), asyncio.Event()
        if not deciding:
            self.deciding.set()

        async def once_let_go(given, place):
            # A model conductor's call: the newer request waits for it.
            await self.deciding.wait()
            await decide(given, place)

        conductor = planned(placing(Start(("D1",))), once_let_go)
        adapter = ScriptedAdapter([wait(self.gate), answer(OLD)], *later, conductor=conductor)
        self.im = self.session(adapter, **settings)
        self.first = self.im.admit(ask("Book a table for 6 pm."), delegation_id="D1")
        await self.turns()
        self.second = self.im.admit(ask("Make it 7."), delegation_id="D2")
        await self.turns()

    async def test_an_answer_given_while_deciding_is_written_at_once(self):
        # The frontend words a result that newer requests may change: a correction steered into it starts a new Work.
        await self.booking(placing(Steer("W1", ("D2",))), [answer(NEW)], deciding=True)
        self.gate.set()
        await self.turns()
        self.assertEqual(answers([event for _, event in self.wire.events]), [OLD])
        self.deciding.set()
        older, newer = await read(self.first), await read(self.second)
        self.assertEqual((answers(older), answers(newer)), ([OLD], [NEW]))
        outcomes = [fields["outcome"] for kind, fields in self.traces if kind == "im.placed"]
        self.assertEqual(outcomes[-1], TEXTS.outcomes["restarted"].format(work="W1", new="W2"))

    async def test_ending_answered(self):
        im = self.session(ScriptedAdapter([answer()]))
        await read(im.admit(ask("Find teas."), delegation_id="D1"))
        self.assertEqual(self.timeline(), ENDINGS["answered"]["timeline"])

    async def test_ending_handed_over_with_the_taker_named(self):
        gate = asyncio.Event()
        adapter = ScriptedAdapter([wait(gate), checkpoint(), answer(NEW)], conductor=SteerOpenWork())
        im = self.session(adapter, progress_note_seconds=0)
        first = im.admit(ask("Book a table for 6 pm."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("Make it 7."), delegation_id="D2")
        await self.turns()
        gate.set()
        await read(first)
        self.assertEqual(answers(await read(second)), [NEW])
        self.assertEqual(self.timeline(), ENDINGS["handed_over_with_the_taker_named"]["timeline"])

    async def test_ending_answered_by_the_conductor(self):
        await self.booking(placing(Answer("D2", "It is booking the table.")), progress_note_seconds=0)
        self.assertEqual(answers(await read(self.second)), ["It is booking the table."])
        self.gate.set()
        self.assertEqual(answers(await read(self.first)), [OLD])
        self.assertEqual(self.timeline(), ENDINGS["answered_by_the_conductor"]["timeline"])

    async def test_ending_cancelled_for_a_newer_request(self):
        await self.booking(placing(Cancel("W1"), Answer("D2", "Stopped the booking.")), progress_note_seconds=0)
        older, newer = await read(self.first), await read(self.second)
        self.assertEqual(answers(older), [TEXTS.cancelled_line])
        self.assertEqual(answers(newer), ["Stopped the booking."])
        self.assertNotIn(OLD, str(older))
        self.assertEqual(self.timeline(), ENDINGS["cancelled_for_a_newer_request_stopped"]["timeline"])


class AnswersWhileStopping(ScriptedRun):
    """A run that still answers while its stop is being confirmed."""

    async def stop(self):
        self.stops += 1
        await asyncio.sleep(0.05)
        self.stopping.set()
        return "stopped"


class StopTests(IMCase):
    async def test_a_cancel_ends_the_response_with_its_stops_line(self):
        cases = (
            ({}, {}, CANCELLED),
            ({"stop_outcome": "unconfirmed"}, {}, TEXTS.unconfirmed_line),
            ({"stop_delay": 1.0}, {"stop_timeout_seconds": 0.05}, TEXTS.unconfirmed_line),
        )
        for options, settings, line in cases:
            with self.subTest(line=line, **options):
                conductor = then(Cancel("W1"), Start(("D2",)))
                adapter = ScriptedAdapter([wait(asyncio.Event())], [answer()], conductor=conductor, **options)
                first, _ = await self.admit_each(self.session(adapter, **settings), "Book it.", "Never mind.")
                older = await read_all(first)
                self.assertEqual((said(older), older[-1]["type"]), ([("final_answer", line)], "response.completed"))
                self.assertEqual(adapter.runs[0].stops, 1)

    async def test_a_stopped_tasks_held_calls_never_reach_the_app(self):
        for cancelled, sent, last in (("W1.2", [], "Menu."), ("W1.1", ["call_prices"], "Prices.")):
            with self.subTest(cancelled=cancelled):
                gate = asyncio.Event()
                menu = [app_call(call_id="call_menu"), answer("Menu.")]
                prices = [wait(gate), app_call(call_id="call_prices"), answer("Prices.")]
                conductor = planned(split("menu", "prices"), placing(Cancel("W1", cancelled), Start(("D2",))))
                im = self.session(ScriptedAdapter(menu, prices, [answer("OK.")], conductor=conductor, max_running=3))
                first = await read_all(im.admit(ask("Menu and prices."), delegation_id="D1", settings=APP).stream)
                gate.set()
                await self.turns()
                await read_all(im.admit(ask("Skip one."), delegation_id="D2").stream)
                await self.turns()
                events = await self.answer_calls(im, first, settings=APP)
                self.assertEqual(function_calls(events), sent)
                events = await self.answer_calls(im, events, settings=APP) if sent else events
                self.assertEqual(texts(events)[-1], last)

    async def cancelled_while_waiting(self, then, **settings) -> tuple:
        """Let a Cancel stop the drink task after its app call ended D1's response, then let the food task answer.

        ``then`` places D2. Return the session, D1's events and D2's admission.
        """
        gate = asyncio.Event()
        conductor = planned(split("drink", "food"), placing(Cancel("W1", "W1.1"), then))
        scripts = ([app_call(call_id="call_drink"), answer("Drink.")], [wait(gate), answer("Food.")])
        im = self.session(ScriptedAdapter(*scripts, conductor=conductor, max_running=2), **settings)
        first = await read_all(im.admit(ask("Drink and food."), delegation_id="D1", settings=APP).stream)
        second = im.admit(ask("Skip the drink."), delegation_id="D2")
        await self.turns()
        self.assertEqual([task.state for task in im.ledger.works["W1"].tasks], ["cancelled", "running"])
        gate.set()
        await self.turns()
        self.assertFalse(im.works["W1"].ended)
        return im, first, second

    async def test_an_answer_after_a_cancel_goes_out_with_the_outputs(self):
        im, first, _ = await self.cancelled_while_waiting(Answer("D2", "OK."))
        events = await asyncio.wait_for(self.answer_calls(im, first, settings=APP), 2)
        self.assertEqual((texts(events), events[-1]["type"]), (["Food."], "response.completed"))
        self.assertEqual((im.ledger.works["W1"].state, im.ledger.delegations["D1"].state), ("answered", "answered"))

    async def test_at_the_deadline_a_work_of_ended_tasks_fails_silently(self):
        im, _, _ = await self.cancelled_while_waiting(Answer("D2", "OK."), client_tool_timeout_seconds=0.1)
        written = len(self.wire.events)
        await self.until(lambda: not im.works, "the deadline")
        self.assertEqual((im.ledger.works["W1"].state, im.ledger.delegations["D1"].state), ("failed", "failed"))
        self.assertEqual(len(self.wire.events), written)

    async def test_the_close_ends_a_work_of_ended_tasks(self):
        im, _, _ = await self.cancelled_while_waiting(Answer("D2", "OK."))
        await asyncio.wait_for(im.close(), 2)
        self.assertEqual((im.ledger.works["W1"].state, im.works), ("cancelled", {}))

    async def test_at_the_deadline_a_taker_carries_the_held_answer(self):
        then = Steer("W1", ("D2",), "W1.2")
        im, _, second = await self.cancelled_while_waiting(then, client_tool_timeout_seconds=0.1)
        self.assertEqual(im.works["W1"].deferred, ["D2"])
        taker = await asyncio.wait_for(read_all(second.stream), 2)
        self.assertEqual((texts(taker), taker[-1]["type"]), (["Food."], "response.completed"))
        self.assertEqual((im.ledger.works["W1"].state, im.ledger.delegations["D1"].state), ("answered", "handed_over"))

    async def test_a_whole_stop_hands_the_work_to_the_landed_request(self):
        gate = asyncio.Event()
        steer_food, cancel_all = placing(Steer("W1", ("D2",), "W1.2")), placing(Cancel("W1"), Start(("D3",)))
        scripts = (
            [app_call(call_id="call_drink"), answer("Drink.")],
            [wait(gate), checkpoint(), wait(asyncio.Event()), answer("Food.")],
            [answer("Stopped.")],
        )
        conductor = planned(split("drink", "food"), steer_food, cancel_all)
        im = self.session(ScriptedAdapter(*scripts, conductor=conductor, max_running=3))
        await read_all(im.admit(ask("A latte and a croissant."), delegation_id="D1", settings=APP).stream)
        second = im.admit(ask("Make the croissant warmed."), delegation_id="D2")
        await self.turns()
        gate.set()
        await self.turns()
        self.assertEqual(im.works["W1"].deferred, ["D2"])
        await read_all(im.admit(ask("Cancel everything."), delegation_id="D3").stream)
        events = await asyncio.wait_for(read_all(second.stream), 2)
        self.assertEqual((texts(events), events[-1]["type"]), ([CANCELLED] * 2, "response.completed"))
        works = {work_id: (work.state, work.requests) for work_id, work in im.ledger.works.items()}
        self.assertEqual(works, {"W1": ("cancelled", ["D1", "D2"]), "W2": ("answered", ["D3"])})
        _, conversation = im.conversation.window()
        self.assertEqual(sum("warmed" in str(item) for item in conversation), 1)

    async def test_a_landed_request_takes_over_when_the_app_never_answers(self):
        may_answer = asyncio.Event()
        scripts = ([app_call(call_id="call_menu"), answer("Menu.")], [wait(may_answer), answer("Open Sundays.")])
        conductor = planned(split("menu", "hours"), placing(Steer("W1", ("D2",), task_id="W1.2")))
        adapter = ScriptedAdapter(*scripts, conductor=conductor, max_running=2)
        im = self.session(adapter, client_tool_timeout_seconds=0.1)
        first = await read_all(im.admit(ask("Menu and hours?"), delegation_id="D1", settings=APP).stream)
        second = im.admit(ask("Open on Sunday?"), delegation_id="D2", settings=APP)
        await self.turns()
        may_answer.set()
        await self.turns()
        taker = await asyncio.wait_for(read_all(second.stream), 2)
        self.assertEqual((named(taker), texts(first)), (first[0]["response"]["id"], []))
        self.assertEqual((texts(taker), taker[-1]["type"]), (["Open Sundays.", OWED], "response.completed"))
        late = await self.answer_calls(im, first, settings=APP)
        self.assertEqual(([e["type"] for e in late], len(adapter.runs)), (LATE, 2))

    async def test_the_backstop_fails_a_stuck_work(self):
        adapter = ScriptedAdapter([wait(asyncio.Event()), answer()])
        im = self.session(adapter, work_timeout_seconds=0.05)
        ended = (await read_all(im.admit(ask("Stuck."), delegation_id="D1").stream))[-1]
        self.assertEqual((ended["type"], ended["response"]["error"]["code"]), ("response.failed", "server_error"))
        self.assertEqual(adapter.runs[0].stops, 1)

    async def test_nothing_given_while_a_stop_is_confirmed_is_written(self):
        gate = asyncio.Event()
        script, conductor = [wait(gate), answer("Too late.")], then(Cancel("W1"))
        adapter = ScriptedAdapter(script, conductor=conductor, run=AnswersWhileStopping)
        first, _ = await self.admit_each(self.session(adapter), "Book it.", "Stop.")
        gate.set()
        self.assertEqual(texts(await read_all(first)), [CANCELLED])
