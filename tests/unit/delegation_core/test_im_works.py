# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""An IM session's Works: the scheduler, placement and its checks, steering, handovers and stops."""

import asyncio
from itertools import pairwise

from examples.frontend_backend_live.delegation_core.backend.port import Answer, Cancel, Start, Steer
from tests.unit.delegation_core.helpers import (
    MENU,
    TEXTS,
    CancelOwnWork,
    Fail,
    IMCase,
    OwnCall,
    Progress,
    ScriptedAdapter,
    ScriptedConductor,
    ScriptedRun,
    StartEach,
    answer,
    app_call,
    app_outputs,
    ask,
    checkpoint,
    function_calls,
    is_handover_line,
    placing,
    planned,
    read_all,
    records_of,
    said,
    texts,
    wait,
)

APP = {"tools": [MENU]}
REFUSED = TEXTS.outcomes["refused"].split("{")[0]
HANDOVER, CANCELLED, OWED = TEXTS.handover_line.format(delegation="D2"), TEXTS.cancelled_line, TEXTS.owed_output
# A late continuation's response: it opens, goes in progress like every response, and ends empty.
LATE = ["response.created", "response.queued", "response.in_progress", "response.completed"]
RUNNING = ["W1", "running", ["search_flights: done"]]  # a running Work's record, after one step


def starts(given, place):
    """A plan that starts one Work for all the arrivals."""
    return place(Start(tuple(arrival.delegation_id for arrival in given.arrivals)))


def split(*briefs: str):
    """A plan that splits D1 into one task per brief."""
    return placing(Start(("D1",), briefs))


def then(*decisions):
    """A conductor that starts D1's Work, then places ``decisions`` for the next arrival."""
    return planned(placing(Start(("D1",))), placing(*decisions))


def named(events: list[dict]) -> str | None:
    """The response a taker's in_progress names as the one it took over."""
    return next(e for e in events if e["type"] == "response.in_progress")["response"].get("previous_response_id")


class WorksCase(IMCase):
    def assertHandedOver(self, older: list[dict]) -> None:
        """One taker's in_progress names ``older`` before it ends with only the handover line, then completed."""
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
    async def test_works_run_up_to_max_running_a_task_waiting_for_the_app_holds_no_slot_and_a_failure_ends_alone(self):
        gates = [asyncio.Event(), asyncio.Event()]
        scripts = [[app_call(), answer("One.")], [wait(gates[0]), Fail()], [wait(gates[1]), answer()]]
        adapter = ScriptedAdapter(*scripts, conductor=StartEach(), max_running=2)
        im = self.session(adapter)
        first = await read_all(im.admit(ask("Request 1."), delegation_id="D1", settings=APP).stream)
        streams = [im.admit(ask(f"Request {n}."), delegation_id=f"D{n}").stream for n in (2, 3)]
        await self.turns()
        self.assertEqual(len(adapter.runs), 3)  # W1 waits for the app, so W2 and W3 run
        resumed = im.admit_client_request(app_outputs(first), delegation_id="D1", settings=APP).stream
        await self.turns()
        self.assertEqual(adapter.runs[0].resumes, 0)  # its outputs are in: it waits for a slot
        gates[0].set()  # W2 raises: it ends alone, and the scheduler goes on with W1
        self.assertEqual((await read_all(streams[0]))[-1]["type"], "response.failed")
        self.assertEqual(texts(await read_all(resumed)), ["One."])
        gates[1].set()
        await read_all(streams[1])

    async def test_an_arrival_restarts_a_decide_that_placed_nothing_once_and_its_old_place_is_refused(self):
        thinking, calls, kept = asyncio.Event(), [], []

        async def slow(given, place):
            calls.append([arrival.delegation_id for arrival in given.arrivals])
            kept.append(place)
            await thinking.wait()
            for arrival in given.arrivals:
                await place(Start((arrival.delegation_id,)))

        im = self.session(ScriptedAdapter(conductor=ScriptedConductor(slow), max_running=3))
        await self.admit_each(im, "One.", "Two.")
        self.assertTrue((await kept[0](Start(("D1",)))).startswith(REFUSED))  # a place kept from the cancelled decide
        im.admit(ask("Three."), delegation_id="D3")  # the restarted decide is not restarted again
        await self.turns()
        thinking.set()
        await self.turns()
        self.assertEqual(calls, [["D1"], ["D1", "D2"], ["D3"]])

    async def test_an_arrival_after_a_placement_waits_for_the_next_decide_of_the_one_conductor(self):
        placed, release, calls = asyncio.Event(), asyncio.Event(), []

        async def plan(given, place):
            calls.append([arrival.delegation_id for arrival in given.arrivals])
            await starts(given, place)
            placed.set()
            await release.wait()

        adapter = ScriptedAdapter(conductor=ScriptedConductor(plan), max_running=2)
        im = self.session(adapter)
        im.admit(ask("One."), delegation_id="D1")
        await placed.wait()
        im.admit(ask("Two."), delegation_id="D2")
        await self.turns()
        self.assertEqual(calls, [["D1"]])
        release.set()
        await self.turns()
        self.assertEqual((calls, adapter.conductors), ([["D1"], ["D2"]], 1))  # one conductor per session

    async def test_a_conductor_reads_the_works_their_tasks_and_the_conversation_since_it_last_read(self):
        gate, conductor = asyncio.Event(), ScriptedConductor(starts)
        scripts = ([answer()], [OwnCall("lookup"), Progress("Checking stock."), wait(gate), answer()])
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
        self.assertEqual(works, [("W2", "running", "D2"), ("W1", "answered", "D1")])  # the open, then the ended
        (task,) = third.works[0].tasks
        self.assertEqual((task.state, task.steps, task.acted), ("running", ("lookup: done",), True))
        self.assertEqual(task.progress, "Checking stock.")  # its newest note
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
        outcomes = [TEXTS.outcomes[kind].format(work="W1", tasks="W1.1") for kind in ("started", "steered")]
        self.assertEqual(conductor.outcomes, outcomes)
        self.assertTrue((await kept[0](Start(("D2",)))).startswith(REFUSED))

    async def test_what_a_decide_leaves_unplaced_starts_a_work_or_joins_the_newest_open_one(self):
        async def boom(given, place):
            raise RuntimeError("conductor down")

        async def cancelled(given, place):
            raise asyncio.CancelledError

        for name, leave in (("placed nothing", placing()), ("raised", boom), ("cancelled", cancelled)):
            with self.subTest(name):
                gate = asyncio.Event()
                adapter = ScriptedAdapter([wait(gate), answer("Green teas.")], conductor=planned(leave, leave))
                first, second = await self.admit_each(self.session(adapter), "Teas.", "Green only.")
                gate.set()  # D1 started W1, as no Work was open; D2 joined it
                older, taker = await read_all(first), await read_all(second)
                self.assertEqual((texts(older), texts(taker), len(adapter.runs)), ([HANDOVER], ["Green teas."], 1))
                self.assertHandedOver(older)

    async def test_each_decision_is_checked_against_the_works_and_requests_it_names(self):
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
        self.assertEqual(len(adapter.runs), 1)  # refused, the steer rule joined D2 to the open W1

    async def test_a_steer_into_an_ended_work_over_max_running_is_refused_and_steered_by_the_rule(self):
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
        self.assertEqual((len(texts(await read_all(third))), len(adapter.runs)), (2, 3))  # it joined W2


class SteeringTests(WorksCase):
    async def test_a_running_task_is_woken_and_takes_the_steering_with_the_records_at_its_next_safe_point(self):
        gate = asyncio.Event()
        adapter = ScriptedAdapter([OwnCall("search_flights"), wait(gate), checkpoint(), answer()])
        _, second = await self.admit_each(self.session(adapter), "Book Denver.", "Make it Tuesday.")
        run = adapter.runs[0]
        self.assertEqual((run.steers, run.applied), (1, []))
        gate.set()
        await self.turns()
        (steering,) = run.applied
        parts = [part["text"] for part in steering.items[-1]["content"]]
        self.assertEqual((parts[0], parts[-1]), ("Make it Tuesday.", TEXTS.closing_line))
        (record,) = records_of(steering.items[-1])
        self.assertEqual([record[key] for key in ("work", "state", "steps")], RUNNING)
        self.assertEqual(texts(await read_all(second)), ["Done."])

    async def test_not_applied_steering_waits_for_the_next_safe_point_then_becomes_a_follow_up_work(self):
        gate = asyncio.Event()
        script = [wait(gate), checkpoint(), checkpoint(), answer("Old.")]
        adapter = ScriptedAdapter(script, [answer("New.")], delivery="not_applied")
        first, second = await self.admit_each(self.session(adapter), "Book Denver.", "Make it Tuesday.")
        gate.set()
        self.assertEqual((texts(await read_all(first)), texts(await read_all(second))), (["Old."], ["New."]))
        self.assertEqual(len([b for b in adapter.runs[0].boundaries if b.steering is not None]), 3)
        self.assertNotIn("Make it Tuesday.", str(adapter.runs[0].histories[-1]))  # never applied, never in history
        self.assertIn("Make it Tuesday.", str(adapter.runs[1].histories[0]))

    async def test_each_task_reads_its_own_brief_and_a_steer_named_for_one_task_reaches_only_that_task(self):
        flights, hotels = asyncio.Event(), asyncio.Event()
        conductor = planned(split("flights", "hotels"), placing(Steer("W1", ("D2",), "W1.2")))
        scripts = ([wait(flights), checkpoint(), answer("Flights.")], [wait(hotels), checkpoint(), answer("Hotels.")])
        adapter = ScriptedAdapter(*scripts, conductor=conductor, max_running=2)
        first, second = await self.admit_each(self.session(adapter), "Flights and hotels.", "Hotels near the airport.")
        hotels.set()
        await read_all(first)  # handed over when the hotels task applied the steering
        flights.set()
        self.assertEqual(texts(await read_all(second)), ["Hotels.", "Flights."])
        self.assertEqual([len(run.applied) for run in adapter.runs], [0, 1])
        self.assertNotIn("Hotels near the airport.", str(adapter.runs[0].histories))
        briefs = [TEXTS.task_line.format(task=brief) for brief in ("flights", "hotels")]
        seen = [[brief for brief in briefs if brief in str(run.histories[0])] for run in adapter.runs]
        self.assertEqual(seen, [[brief] for brief in briefs])  # each task reads its own brief

    async def test_a_join_keeps_the_works_settings_and_a_typed_run_brings_its_own(self):
        for admit, model in (("admit", "model/one"), ("admit_client_request", "model/two")):
            with self.subTest(admit):
                gate = asyncio.Event()
                adapter = ScriptedAdapter([wait(gate), checkpoint(), answer()])
                im = self.session(adapter)
                await self.admit_each(im, "Book Denver.", settings={"model": "model/one"})
                getattr(im, admit)(ask("Tuesday."), delegation_id="D2", settings={"model": "model/two"})
                await self.turns()
                gate.set()
                await self.turns()
                self.assertEqual([request["model"] for request in adapter.runs[0].requests], [model])

    async def test_a_merge_into_a_queued_work_names_the_taker_at_once(self):
        async def plan(given, place):
            queued = [work.work_id for work in given.works if work.state == "queued"]
            joins = given.arrivals[0].delegation_id == "D2"
            await (place(Steer(queued[-1], ("D2",))) if joins else starts(given, place))

        conductor = ScriptedConductor(plan)
        im = self.session(ScriptedAdapter([wait(asyncio.Event())], [answer()], conductor=conductor))
        im.admit(ask("Busy."), delegation_id="D0")
        await self.turns()
        first, _ = await self.admit_each(im, "Teas.", "Green only.")
        older = await read_all(first)
        kinds = ["response.created", "response.queued", "response.in_progress", "response.completed"]
        self.assertEqual([e["type"] for e in older if "response" in e], kinds)
        self.assertHandedOver(older)

    async def test_a_steer_into_work_waiting_for_the_app_lands_after_the_outputs(self):
        im = self.session(ScriptedAdapter([app_call("get_menu"), answer("Green: two.")]))
        first = await read_all(im.admit(ask("Find teas."), delegation_id="D1", settings=APP).stream)
        second = im.admit(ask("Green only."), delegation_id="D2", settings=APP)
        await self.turns()
        resumed = await self.answer_calls(im, first, settings=APP)
        taker = await read_all(second.stream)
        self.assertEqual((named(taker), texts(taker)), (resumed[0]["response"]["id"], ["Green: two."]))
        self.assertHandedOver(resumed)

    async def test_a_start_of_three_requests_hands_over_as_a_chain_to_the_newest(self):
        adapter = ScriptedAdapter([answer()], conductor=ScriptedConductor(starts))
        im = self.session(adapter)
        streams = [im.admit(ask(f"Part {n}."), delegation_id=f"D{n}").stream for n in (1, 2, 3)]
        events = [await read_all(stream) for stream in streams]
        for older, taker in pairwise(events):
            self.assertEqual(named(taker), older[0]["response"]["id"])
            self.assertHandedOver(older)
        self.assertEqual(texts(events[2]), ["Done."])
        self.assertIn("Part 1.", str(adapter.runs[0].histories[0]))  # its one task read every request
        self.assertIn("Part 3.", str(adapter.runs[0].histories[0]))

    async def test_steering_from_one_task_while_another_waits_hands_over_when_the_carrier_reopens(self):
        gate = asyncio.Event()
        scripts = ([app_call("get_menu"), answer("Menu.")], [wait(gate), checkpoint(), answer("Hours, for green tea.")])
        conductor = planned(split("menu", "hours"), placing(Steer("W1", ("D2",), "W1.2")))
        im = self.session(ScriptedAdapter(*scripts, conductor=conductor, max_running=2))
        first = await read_all(im.admit(ask("Menu and hours."), delegation_id="D1", settings=APP).stream)
        second = im.admit(ask("Green tea only."), delegation_id="D2")
        await self.turns()
        gate.set()
        await self.turns()  # landed, but D1's response waits for the app
        resumed = await self.answer_calls(im, first, settings=APP)
        taker = await read_all(second.stream)
        self.assertEqual(named(taker), resumed[0]["response"]["id"])
        self.assertHandedOver(resumed)
        self.assertEqual(texts(taker), ["Hours, for green tea.", "Menu."])


class AnswersWhileStopping(ScriptedRun):
    """A run that still answers while its stop is being confirmed."""

    async def stop(self):
        self.stops += 1
        await asyncio.sleep(0.05)
        self.stopping.set()
        return "stopped"


class StopTests(IMCase):
    async def test_a_cancel_for_a_newer_request_ends_the_older_response_with_the_line_its_stop_earned(self):
        cases = (
            ({}, {}, CANCELLED),
            ({"stop_outcome": "unconfirmed"}, {}, TEXTS.unconfirmed_line),
            ({"stop_delay": 1.0}, {"stop_timeout_seconds": 0.05}, TEXTS.unconfirmed_line),  # no answer in time
        )
        for options, settings, line in cases:
            with self.subTest(line=line, **options):
                conductor = then(Cancel("W1"), Start(("D2",)))
                adapter = ScriptedAdapter([wait(asyncio.Event())], [answer()], conductor=conductor, **options)
                first, _ = await self.admit_each(self.session(adapter, **settings), "Book it.", "Never mind.")
                older = await read_all(first)
                self.assertEqual((said(older), older[-1]["type"]), ([("final_answer", line)], "response.completed"))
                self.assertEqual(adapter.runs[0].stops, 1)

    async def test_a_queued_work_is_dequeued_and_its_cancel_still_says_so(self):
        conductor = planned(placing(Start(("D1",)), Start(("D2",))), placing(Cancel("W2"), Start(("D3",))))
        adapter = ScriptedAdapter([wait(asyncio.Event())], [answer()], conductor=conductor)
        im = self.session(adapter)
        im.admit(ask("Busy."), delegation_id="D1")
        second = im.admit(ask("Teas."), delegation_id="D2")
        await self.turns()
        im.admit(ask("Forget the teas."), delegation_id="D3")
        self.assertEqual((texts(await read_all(second.stream)), len(adapter.runs)), ([CANCELLED], 1))

    async def test_a_cancel_of_one_task_keeps_the_others_running(self):
        gates = [asyncio.Event(), asyncio.Event()]
        conductor = planned(split("flights", "hotels"), placing(Cancel("W1", "W1.2"), Start(("D2",))))
        scripts = ([wait(gates[0]), answer("Flights.")], [wait(gates[1]), answer("Hotels.")], [answer("OK.")])
        adapter = ScriptedAdapter(*scripts, conductor=conductor, max_running=3)
        first, _ = await self.admit_each(self.session(adapter), "Flights and hotels.", "Skip the hotels.")
        self.assertEqual(adapter.runs[1].stops, 1)
        gates[0].set()
        events = await read_all(first)
        self.assertEqual((texts(events), events[-1]["type"]), ([CANCELLED, "Flights."], "response.completed"))

    async def test_a_stopped_work_writes_nothing_more_and_the_apps_outputs_for_it_come_late(self):
        # A Cancel of a Work that waits for the app, and the client-tool deadline while its stop is being confirmed.
        cancel = ScriptedAdapter([app_call(), answer()], [answer("OK.")], conductor=then(Cancel("W1"), Start(("D2",))))
        deadline = ScriptedAdapter([app_call(), answer()], stop_delay=0.1)
        cases = ((cancel, {}, ["Never mind."]), (deadline, {"client_tool_timeout_seconds": 0.02}, []))
        for adapter, settings, newer in cases:
            with self.subTest(**settings):
                im = self.session(adapter, **settings)
                first = await read_all(im.admit(ask("Menu?"), delegation_id="D1", settings=APP).stream)
                for text in newer:
                    await read_all(im.admit(ask(text), delegation_id="D2").stream)
                await asyncio.sleep(0.05)
                self.assertEqual([e["type"] for e in await self.answer_calls(im, first, settings=APP)], LATE)
                self.assertEqual((first[-1]["type"], adapter.runs[0].stops), ("response.completed", 1))

    async def test_at_the_deadline_waiting_steering_becomes_a_follow_up_work(self):
        adapter = ScriptedAdapter([app_call("get_menu"), answer()], [answer("Green teas.")])
        im = self.session(adapter, client_tool_timeout_seconds=0.05)
        first = await read_all(im.admit(ask("Menu?"), delegation_id="D1", settings=APP).stream)
        taken = await read_all(im.admit(ask("Only green ones."), delegation_id="D2").stream)
        self.assertEqual((first[-1]["type"], texts(taken), named(taken)), ("response.completed", ["Green teas."], None))
        self.assertEqual([record["state"] for record in records_of(adapter.runs[1].histories[0][-1])], ["failed"])

    async def test_a_stopped_tasks_held_calls_stay_back_and_another_tasks_go_out_once_its_carrier_reopens(self):
        for cancelled, sent, last in (("W1.2", [], "Menu."), ("W1.1", ["call_prices"], "Prices.")):
            with self.subTest(cancelled=cancelled):
                gate = asyncio.Event()
                menu = [app_call(call_id="call_menu"), answer("Menu.")]
                prices = [wait(gate), app_call(call_id="call_prices"), answer("Prices.")]
                conductor = planned(split("menu", "prices"), placing(Cancel("W1", cancelled), Start(("D2",))))
                im = self.session(ScriptedAdapter(menu, prices, [answer("OK.")], conductor=conductor, max_running=3))
                first = await read_all(im.admit(ask("Menu and prices."), delegation_id="D1", settings=APP).stream)
                gate.set()
                await self.turns()  # the prices task's call is held: D1's response already ended with the menu call
                await read_all(im.admit(ask("Skip one."), delegation_id="D2").stream)
                await self.turns()
                events = await self.answer_calls(im, first, settings=APP)
                self.assertEqual(function_calls(events), sent)
                events = await self.answer_calls(im, events, settings=APP) if sent else events
                self.assertEqual(texts(events)[-1], last)

    async def test_a_newer_request_landed_in_another_task_takes_the_work_over_when_the_app_never_answers(self):
        # At the deadline the taker takes the Work over, the held answer is delivered, the stuck task ends with the
        # outcome-unknown line, and nothing runs twice.
        may_answer = asyncio.Event()
        scripts = ([app_call(call_id="call_menu"), answer("Menu.")], [wait(may_answer), answer("Open Sundays.")])
        conductor = planned(split("menu", "hours"), placing(Steer("W1", ("D2",), task_id="W1.2")))
        adapter = ScriptedAdapter(*scripts, conductor=conductor, max_running=2)
        im = self.session(adapter, client_tool_timeout_seconds=0.1)
        first = await read_all(im.admit(ask("Menu and hours?"), delegation_id="D1", settings=APP).stream)
        second = im.admit(ask("Open on Sunday?"), delegation_id="D2", settings=APP)
        await self.turns()
        may_answer.set()
        await self.turns()  # the hours task applied D2 and answered: its answer is held, D1's response being closed
        taker = await asyncio.wait_for(read_all(second.stream), 2)
        self.assertEqual((named(taker), texts(first)), (first[0]["response"]["id"], []))
        self.assertEqual((texts(taker), taker[-1]["type"]), (["Open Sundays.", OWED], "response.completed"))
        late = await self.answer_calls(im, first, settings=APP)
        self.assertEqual(([e["type"] for e in late], len(adapter.runs)), (LATE, 2))  # late, and nothing ran twice

    async def test_a_stop_line_follows_the_answers_held_while_the_response_was_closed(self):
        gate = asyncio.Event()
        conductor = planned(split("menu", "hours"), starts, placing(Cancel("W1", "W1.1")))
        scripts = ([app_call(), answer("Menu.")], [wait(gate), answer("Hours.")], [wait(asyncio.Event())])
        im = self.session(ScriptedAdapter(*scripts, conductor=conductor))
        first = await read_all(im.admit(ask("Menu and hours."), delegation_id="D1", settings=APP).stream)
        gate.set()
        await self.turns()  # the hours task answered while D1's response waited for the app: its answer is held
        im.admit(ask("And coffee."), delegation_id="D2")
        await self.turns()  # W2 takes the one slot
        continuation = im.admit_client_request(app_outputs(first), delegation_id="D1", settings=APP)
        await self.turns()  # the menu task waits for a slot
        im.admit(ask("Skip the menu."), delegation_id="D3")
        events = await asyncio.wait_for(read_all(continuation.stream), 2)
        self.assertEqual((texts(events), events[-1]["type"]), (["Hours.", CANCELLED], "response.completed"))

    async def test_the_backstop_fails_a_stuck_work_through_the_stop_door(self):
        adapter = ScriptedAdapter([wait(asyncio.Event()), answer()])
        im = self.session(adapter, work_timeout_seconds=0.05)
        ended = (await read_all(im.admit(ask("Stuck."), delegation_id="D1").stream))[-1]
        self.assertEqual((ended["type"], ended["response"]["error"]["code"]), ("response.failed", "server_error"))
        self.assertEqual(adapter.runs[0].stops, 1)

    async def test_nothing_a_task_gives_while_its_stop_is_being_confirmed_is_written(self):
        gate = asyncio.Event()
        script, conductor = [wait(gate), answer("Too late.")], then(Cancel("W1"))
        adapter = ScriptedAdapter(script, conductor=conductor, run=AnswersWhileStopping)
        first, _ = await self.admit_each(self.session(adapter), "Book it.", "Stop.")
        gate.set()  # the task answers while its stop is being confirmed
        self.assertEqual(texts(await read_all(first)), [CANCELLED])

    async def test_a_stop_request_left_beside_its_cancel_becomes_a_follow_up_work_once_the_stop_is_confirmed(self):
        conductor = then(Cancel("W1"))
        adapter = ScriptedAdapter([wait(asyncio.Event())], [answer("Stopped.")], conductor=conductor, stop_delay=0.05)
        first, second = await self.admit_each(self.session(adapter), "Book it.", "Stop.")
        stopped, answered = await read_all(first), await read_all(second)
        self.assertEqual((texts(stopped), texts(answered)), ([CANCELLED], ["Stopped."]))
        follow_up = adapter.runs[1].histories[0][-1]  # the follow-up's request: "Stop.", then the records
        self.assertEqual((follow_up["content"][0]["text"], records_of(follow_up)[0]["state"]), ("Stop.", "cancelled"))

    async def test_a_landed_stop_lets_the_task_end_its_own_work_with_cancel_work(self):
        gate = asyncio.Event()
        script = [OwnCall("search_flights"), wait(gate), checkpoint(), CancelOwnWork(), answer("Stopped, none booked.")]
        adapter = ScriptedAdapter(script)  # the shipped settings: every task is offered cancel_work
        first, second = await self.admit_each(self.session(adapter), "Book Denver.", "Stop, never mind.")
        gate.set()
        events = await read_all(second)
        await read_all(first)
        run = adapter.runs[0]
        self.assertEqual([tool["name"] for tool in run.task.tools().tools()], ["cancel_work"])
        (steering,) = run.applied
        self.assertEqual(steering.items[-1]["content"][-1]["text"], TEXTS.closing_line)
        self.assertEqual(run.routes, ["backend", "backend"])  # search_flights, then the IM's cancel_work
        self.assertEqual(texts(events), ["Stopped, none booked."])  # the stop's own answer: no item for the call
        self.assertEqual(len(adapter.runs), 1)  # the stop started no work of its own
