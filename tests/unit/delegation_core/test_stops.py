# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Stops: Cancel, the client-tool deadline, the backstop, and a task that ends its own Work."""

import asyncio

from delegation_core.backend.port import Cancel, Start, Steer
from delegation_core.im.stop import ForNewerRequest
from tests.unit.delegation_core.helpers import (
    MENU,
    TEXTS,
    IMCase,
    ScriptedAdapter,
    ScriptStep,
    answer,
    app_call,
    app_outputs,
    ask,
    call_item,
    checkpoint,
    finished_items,
    own_call,
    planned,
    read_all,
    wait,
)

# A late continuation's response: it opens, goes in progress like every response, and ends empty.
LATE = ["response.created", "response.queued", "response.in_progress", "response.completed"]


def texts(events: list[dict]) -> list[str]:
    return [item["content"][0]["text"] for item in finished_items(events) if item["type"] == "message"]


class StopTests(IMCase):
    async def test_a_queued_work_is_dequeued_and_its_cancel_still_says_so(self):
        never = asyncio.Event()

        async def start_both(given, place):
            for arrival in given.arrivals:
                await place(Start((arrival.delegation_id,)))

        async def cancel(given, place):
            await place(Cancel("W2"))
            await place(Start(("D3",)))

        adapter = ScriptedAdapter([wait(never)], [answer()], conductor=planned(start_both, cancel))
        im = self.session(adapter)
        im.admit(ask("Busy."), delegation_id="D1")
        second = im.admit(ask("Teas."), delegation_id="D2")
        await self.turns()
        im.admit(ask("Forget the teas."), delegation_id="D3")
        events = await read_all(second.stream)
        self.assertEqual(texts(events), [TEXTS.cancelled_line])
        self.assertEqual(len(adapter.runs), 1)  # W2 never started
        self.assertEqual(im.ledger.works["W2"].state, "cancelled")

    async def test_a_cancel_of_one_task_keeps_the_others_running(self):
        gates = [asyncio.Event(), asyncio.Event()]

        async def split(given, place):
            await place(Start(("D1",), ("flights", "hotels")))

        async def cancel_hotels(given, place):
            await place(Cancel("W1", "W1.2"))
            await place(Start(("D2",)))

        adapter = ScriptedAdapter(
            [wait(gates[0]), answer("Flights.")],
            [wait(gates[1]), answer("Hotels.")],
            [answer("OK.")],
            conductor=planned(split, cancel_hotels),
            max_running=3,
        )
        im = self.session(adapter)
        first = im.admit(ask("Flights and hotels."), delegation_id="D1")
        await self.turns()
        im.admit(ask("Skip the hotels."), delegation_id="D2")
        await self.turns()
        self.assertEqual(adapter.runs[1].stops, 1)
        gates[0].set()
        events = await read_all(first.stream)
        self.assertEqual(texts(events), [TEXTS.cancelled_line, "Flights."])
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(im.ledger.works["W1"].state, "answered")

    async def test_a_waiting_work_stops_writing_nothing_and_its_outputs_come_late(self):
        async def start(given, place):
            await place(Start(("D1",)))

        async def cancel(given, place):
            await place(Cancel("W1"))
            await place(Start(("D2",)))

        adapter = ScriptedAdapter([app_call("get_menu"), answer()], [answer("OK.")], conductor=planned(start, cancel))
        im = self.session(adapter)
        menu = {"tools": [MENU]}
        first = await read_all(im.admit(ask("Menu?"), delegation_id="D1", settings=menu).stream)
        await read_all(im.admit(ask("Never mind."), delegation_id="D2").stream)
        await self.turns()
        self.assertEqual(adapter.runs[0].stops, 1)
        self.assertEqual(im.ledger.works["W1"].state, "cancelled")
        self.assertEqual(im.ledger.actions[app_outputs(first)[0]["call_id"]].state, "unclear")
        late = await read_all(im.admit_client_request(app_outputs(first), delegation_id="D1", settings=menu).stream)
        self.assertEqual(
            [e["type"] for e in late],
            LATE,
        )

    async def test_the_deadline_writes_nothing_and_waiting_steering_becomes_a_follow_up_work(self):
        adapter = ScriptedAdapter([app_call("get_menu"), answer()], [answer("Green teas.")])
        im = self.session(adapter, client_tool_timeout_seconds=0.05)
        menu = {"tools": [MENU]}
        first = await read_all(im.admit(ask("Menu?"), delegation_id="D1", settings=menu).stream)
        second = im.admit(ask("Only green ones."), delegation_id="D2")
        await self.turns()
        self.assertEqual(im.works["W1"].batches[0].delegations, ["D2"])
        taken = await read_all(second.stream)
        self.assertEqual(first[-1]["type"], "response.completed")  # nothing after the deadline
        self.assertEqual(texts(taken), ["Green teas."])
        self.assertEqual((im.ledger.works["W1"].state, im.ledger.works["W2"].state), ("failed", "answered"))
        self.assertEqual(im.ledger.tasks["W1.1"].outcome, "unclear")
        self.assertNotIn("previous_response_id", str(taken))  # a follow-up takes nothing over

    async def test_outputs_that_come_while_the_deadlines_stop_is_confirmed_are_late(self):
        adapter = ScriptedAdapter([app_call("get_menu"), answer()], stop_delay=0.1)
        im = self.session(adapter, client_tool_timeout_seconds=0.02)
        menu = {"tools": [MENU]}
        first = await read_all(im.admit(ask("Menu?"), delegation_id="D1", settings=menu).stream)
        await asyncio.sleep(0.05)  # the deadline asked the stop; the run takes 0.1 s to confirm it
        self.assertFalse(im.ledger.works["W1"].ended)
        continuation = im.admit_client_request(app_outputs(first), delegation_id="D1", settings=menu)
        late = await asyncio.wait_for(read_all(continuation.stream), 2)
        self.assertEqual(
            [e["type"] for e in late],
            LATE,
        )
        await asyncio.sleep(0.1)
        self.assertEqual(im.ledger.works["W1"].state, "failed")
        self.assertEqual(im.book.open_delegations(), [])

    async def test_a_stopped_tasks_held_calls_never_reach_the_app(self):
        gate = asyncio.Event()

        async def split(given, place):
            await place(Start(("D1",), ("menu", "prices")))

        async def cancel_prices(given, place):
            await place(Cancel("W1", "W1.2"))
            await place(Start(("D2",)))

        adapter = ScriptedAdapter(
            [app_call("get_menu", call_id="call_menu"), answer("Menu.")],
            [wait(gate), app_call("get_menu", call_id="call_prices"), answer("Prices.")],
            [answer("OK.")],
            conductor=planned(split, cancel_prices),
            max_running=3,
        )
        im = self.session(adapter)
        menu = {"tools": [MENU]}
        first = await read_all(im.admit(ask("Menu and prices."), delegation_id="D1", settings=menu).stream)
        gate.set()
        await self.turns()  # the prices task's calls are held: D1's response already ended with the menu call
        await read_all(im.admit(ask("Skip the prices."), delegation_id="D2").stream)
        await self.turns()
        continuation = im.admit_client_request(app_outputs(first), delegation_id="D1", settings=menu)
        events = await asyncio.wait_for(read_all(continuation.stream), 2)
        self.assertEqual(texts(events), ["Menu."])
        self.assertNotIn("call_prices", str(events))
        self.assertEqual(im.ledger.works["W1"].state, "answered")

    async def test_a_cancel_of_the_task_waiting_for_the_app_lets_the_other_tasks_held_calls_go_out(self):
        # A Cancel of the task waiting for the app keeps its response waiting: only the app's outputs open D1's next
        # response, which the other task's held calls go into.
        gate = asyncio.Event()

        async def split(given, place):
            await place(Start(("D1",), ("menu", "prices")))

        async def cancel_menu(given, place):
            await place(Cancel("W1", "W1.1"))
            await place(Start(("D2",)))

        adapter = ScriptedAdapter(
            [app_call("get_menu", call_id="call_menu"), answer("Menu.")],
            [wait(gate), app_call("get_menu", call_id="call_prices"), answer("Prices.")],
            [answer("OK.")],
            conductor=planned(split, cancel_menu),
            max_running=3,
        )
        im = self.session(adapter)
        menu = {"tools": [MENU]}
        first = await read_all(im.admit(ask("Menu and prices."), delegation_id="D1", settings=menu).stream)
        gate.set()
        await self.turns()  # the prices task's calls are held: D1's response already ended with the menu call
        await read_all(im.admit(ask("Skip the menu."), delegation_id="D2").stream)
        await self.turns()
        self.assertEqual(adapter.runs[0].stops, 1)
        continuation = im.admit_client_request(app_outputs(first), delegation_id="D1", settings=menu)
        events = await asyncio.wait_for(read_all(continuation.stream), 2)
        calls = [item["call_id"] for item in finished_items(events) if item["type"] == "function_call"]
        self.assertEqual(calls, ["call_prices"])  # the held calls go out on the next response
        last = im.admit_client_request(app_outputs(events), delegation_id="D1", settings=menu)
        self.assertEqual(texts(await asyncio.wait_for(read_all(last.stream), 2)), ["Prices."])
        self.assertEqual(im.ledger.works["W1"].state, "answered")

    async def test_a_newer_request_landed_in_another_task_takes_the_work_over_when_the_app_never_answers(self):
        # The menu task waits for an app that never answers, and the hours task's answer to a newer request waits for
        # D1's next response. At the deadline the taker takes the Work over, named in previous_response_id, the held
        # answer is delivered, the stuck task ends with the outcome-unknown line, and nothing runs twice.
        may_answer = asyncio.Event()

        async def split(given, place):
            await place(Start(("D1",), ("menu", "hours")))

        async def steer_hours(given, place):
            await place(Steer("W1", ("D2",), task_id="W1.2"))

        adapter = ScriptedAdapter(
            [app_call("get_menu", call_id="call_menu"), answer("Menu.")],
            [wait(may_answer), answer("We open at seven, Sundays too.")],
            [answer("A second run of the request.")],
            conductor=planned(split, steer_hours),
            max_running=2,
        )
        im = self.session(adapter, client_tool_timeout_seconds=0.1)
        menu = {"tools": [MENU]}
        first = await read_all(im.admit(ask("Menu and hours?"), delegation_id="D1", settings=menu).stream)
        second = im.admit(ask("Open on Sunday?"), delegation_id="D2", settings=menu)
        await self.turns()
        may_answer.set()
        await self.turns()  # the hours task applied D2 and answered: its answer is held, D1's response being closed
        self.assertEqual(texts(first), [])
        taker = await asyncio.wait_for(read_all(second.stream), 2)
        named = next(event for event in taker if event["type"] == "response.in_progress")
        self.assertEqual(named["response"]["previous_response_id"], first[0]["response"]["id"])
        self.assertEqual(texts(taker), ["We open at seven, Sundays too.", TEXTS.owed_output])
        self.assertEqual(taker[-1]["type"], "response.completed")
        self.assertEqual(len(adapter.runs), 2)  # nothing ran twice: no follow-up Work
        self.assertEqual(list(im.ledger.works), ["W1"])
        d1 = im.ledger.delegations["D1"]
        self.assertEqual((d1.state, d1.handed_to, im.ledger.delegations["D2"].state), ("handed_over", "D2", "answered"))
        self.assertEqual(im.ledger.works["W1"].state, "answered")
        self.assertEqual(
            (im.ledger.tasks["W1.1"].outcome, im.ledger.actions["call_menu"].state), ("unclear", "unclear")
        )
        late = await read_all(im.admit_client_request(app_outputs(first), delegation_id="D1", settings=menu).stream)
        self.assertEqual([e["type"] for e in late], LATE)  # the app's answer after the deadline is late
        self.assertEqual(len(adapter.runs), 2)

    async def test_a_stop_line_follows_the_answers_held_while_the_response_was_closed(self):
        gate, never = asyncio.Event(), asyncio.Event()

        async def split(given, place):
            await place(Start(("D1",), ("menu", "hours")))

        async def start(given, place):
            await place(Start(("D2",)))

        adapter = ScriptedAdapter(
            [app_call("get_menu"), answer("Menu.")],
            [wait(gate), answer("Hours.")],
            [wait(never)],
            conductor=planned(split, start),
        )
        im = self.session(adapter)
        menu = {"tools": [MENU]}
        first = await read_all(im.admit(ask("Menu and hours."), delegation_id="D1", settings=menu).stream)
        gate.set()
        await self.turns()  # the hours task answered while D1's response waited for the app: its answer is held
        im.admit(ask("And coffee."), delegation_id="D2")
        await self.turns()  # W2 takes the one slot
        work = im.works["W1"]
        continuation = im.admit_client_request(app_outputs(first), delegation_id="D1", settings=menu)
        await self.turns()  # the menu task waits for a slot
        work.stop_soon(ForNewerRequest(), work.tasks[0])
        events = await asyncio.wait_for(read_all(continuation.stream), 2)
        self.assertEqual(texts(events), ["Hours.", TEXTS.cancelled_line])
        self.assertEqual(events[-1]["type"], "response.completed")

    async def test_the_backstop_fails_a_stuck_work_through_the_stop_door(self):
        never = asyncio.Event()
        adapter = ScriptedAdapter([wait(never), answer()])
        im = self.session(adapter, work_timeout_seconds=0.05)
        events = await read_all(im.admit(ask("Stuck."), delegation_id="D1").stream)
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertEqual(events[-1]["response"]["error"]["code"], "server_error")
        self.assertEqual(adapter.runs[0].stops, 1)
        work = im.ledger.works["W1"]
        self.assertEqual((work.state, im.ledger.tasks["W1.1"].stop.state), ("failed", "stopped"))

    async def test_nothing_of_a_stopping_task_is_written(self):
        gate = asyncio.Event()

        async def start(given, place):
            await place(Start(("D1",)))

        async def cancel(given, place):
            await place(Cancel("W1"))

        adapter = ScriptedAdapter([wait(gate), answer()], conductor=planned(start, cancel), stop_delay=0.05)
        im = self.session(adapter)
        first = im.admit(ask("Book it."), delegation_id="D1")
        await self.turns()
        im.admit(ask("Stop."), delegation_id="D2")
        await self.turns()
        work = im.works["W1"]
        work.answer(work.tasks[0], "Too late.")  # an answer that arrives while the stop is being confirmed
        events = await read_all(first.stream)
        self.assertEqual(texts(events), [TEXTS.cancelled_line])

    async def test_a_stop_request_left_beside_its_cancel_waits_for_the_stop_and_becomes_a_follow_up_work(self):
        # A Cancel places no request: the steer rule joins "Stop." to the stopping Work, the newest open one, whose task
        # never takes it; once the stop is confirmed it starts a follow-up Work, which reads the stopped Work's record.
        gate = asyncio.Event()

        async def start(given, place):
            await place(Start(("D1",)))

        async def cancel_only(given, place):
            await place(Cancel("W1"))

        adapter = ScriptedAdapter(
            [wait(gate), answer()], [answer("Stopped.")], conductor=planned(start, cancel_only), stop_delay=0.05
        )
        im = self.session(adapter)
        first = im.admit(ask("Book it."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("Stop."), delegation_id="D2")
        stopped, answered = await read_all(first.stream), await read_all(second.stream)
        self.assertEqual((texts(stopped), texts(answered)), ([TEXTS.cancelled_line], ["Stopped."]))
        [steering] = im.ledger.steerings.values()
        self.assertEqual((steering.task, steering.delegations, steering.state), ("W1.1", ["D2"], "follow_up"))
        works, request = im.ledger.works, im.ledger.delegations["D2"]
        self.assertEqual((request.work, works["W1"].state, works["W2"].state), ("W2", "cancelled", "answered"))
        records = adapter.runs[1].histories[0][-1]["content"][1]["text"]  # the follow-up's request: "Stop.", records
        self.assertIn('{"work": "W1", "state": "cancelled"', records)


class EndsOwnWork(ScriptStep):
    """A round that calls the work tool cancel_work, run through the task's tools, as a task that read a stop does."""

    async def play(self, run):
        call = call_item("cancel_work", {}, "call_cancel")
        for event in run.round_events([call]):
            yield event
        run.routes.append(run.task.route(call))
        tools = run.task.tools()
        run.task.tool_started(call, tools.label)
        run.task.tool_outcome("call_cancel", await tools.execute("call_cancel", "cancel_work", "{}"), "done")


class OwnStopTests(IMCase):
    """The steer rule never stops work: a caller's stop lands in the running task, which ends its own Work."""

    async def test_a_landed_stop_lets_the_task_end_its_own_work_with_cancel_work(self):
        gate = asyncio.Event()
        script = [own_call("search_flights"), wait(gate), checkpoint(), EndsOwnWork(), answer("Stopped, none booked.")]
        adapter = ScriptedAdapter(script)
        im = self.session(adapter)  # the shipped settings: every task is offered cancel_work
        first = im.admit(ask("Book Denver."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("Stop, never mind."), delegation_id="D2")
        await self.turns()
        gate.set()
        events = await read_all(second.stream)
        await read_all(first.stream)
        run = adapter.runs[0]
        self.assertEqual([tool["name"] for tool in run.task.tools().tools()], ["cancel_work"])
        (steering,) = run.applied
        self.assertEqual(steering.items[-1]["content"][-1]["text"], TEXTS.closing_line)
        self.assertEqual(run.routes, ["backend", "backend"])  # search_flights, then the IM's cancel_work
        self.assertEqual(im.ledger.actions["call_cancel"].state, "done")
        self.assertEqual(texts(events), ["Stopped, none booked."])  # the stop's own answer: no item for the call
        self.assertEqual((im.ledger.works["W1"].state, im.ledger.delegations["D2"].work), ("answered", "W1"))
        self.assertEqual(len(adapter.runs), 1)  # the stop started no work of its own
