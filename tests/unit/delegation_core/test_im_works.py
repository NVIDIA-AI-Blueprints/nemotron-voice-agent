# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""An IM session's Works: the scheduler, placement and its checks, steering, handovers, stops and the endings."""

import asyncio
import contextvars
import time
from itertools import pairwise

from examples.frontend_backend_live.delegation_core.backend.conductors import SteerOpenWork
from examples.frontend_backend_live.delegation_core.backend.port import Answer, Cancel, Start, Steer
from examples.frontend_backend_live.delegation_core.im.run import RECORD_STEPS, RECORD_TEXT_CHARS
from examples.frontend_backend_live.delegation_core.im.session import DelegationError
from examples.frontend_backend_live.delegation_core.items import message_text
from tests.unit.core_helpers import load
from tests.unit.delegation_core.helpers import (
    MARK,
    MENU,
    MESSAGE,
    QUESTION,
    TEXTS,
    ClosesWhenLetGo,
    Fail,
    IMCase,
    OwnCall,
    ScriptedAdapter,
    ScriptedConductor,
    ScriptedRun,
    StartEach,
    WorkTool,
    answer,
    answers,
    app_call,
    app_outputs,
    ask,
    asking,
    checkpoint,
    finished_items,
    function_calls,
    is_handover_line,
    marks,
    placing,
    planned,
    progress,
    read_all,
    records_of,
    said,
    texts,
    wait,
    without_ids,
)

APP = {"tools": [MENU]}
REFUSED = TEXTS.outcomes["refused"].split("{")[0]
HANDOVER, CANCELLED, OWED = TEXTS.handover_line.format(delegation="D2"), TEXTS.cancelled_line, TEXTS.owed_output
LATE = ["response.created", "response.queued", "response.in_progress", "response.completed"]
ENDINGS = load()
OLD, NEW = "Booked for 6 pm.", "Booked for 7 pm."
HOT_OR_ICED, REPORT = "Hot or iced?", "The comparison is ready."
# The call a task's run starts in, as a call's own context names it.
CALL = contextvars.ContextVar("call", default=None)


def starts(given, place):
    return place(Start(tuple(arrival.delegation_id for arrival in given.arrivals)))


def split(*briefs: str):
    return placing(Start(("D1",), briefs))


def then(*decisions):
    """Build a conductor that starts D1's Work, then places ``decisions`` for the next arrival."""
    return planned(placing(Start(("D1",))), placing(*decisions))


async def undecided(given, place):
    """Place nothing, ever: the requests wait for a decision."""
    await asyncio.Event().wait()


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
        placed = TEXTS.outcomes["already_placed"].format(delegation="D2", work="W1", tasks="W1.1")
        self.assertEqual(await kept[0](Start(("D2",))), placed)

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
        placed = outcome["already_placed"].format(delegation="D2", work="W2", tasks="W2.1")
        unknown = [Steer("W9", ("D2",)), Cancel("W9"), Steer("W1", ("D9",)), Cancel("W1", "W1.7"), Start(("D2",))]
        unknown_lines = [refusal("no_work", work="W9")] * 2 + [refusal("unplaced", delegations="D9")]
        unknown_lines += [refusal("no_task", task="W1.7", work="W1"), started]
        once = [Start(("D2",)), Steer("W1", ("D2",)), Answer("D2", "Hi.")]
        cases = {
            "placed once": (once, None, [started, placed, placed]),
            "unknown names": (unknown, None, unknown_lines),
            "an ended work": ([Steer("W1", ("D2",))], ended, [outcome["restarted"].format(work="W1", new="W2")]),
            "a task outside an ended work": (
                [Steer("W1", ("D2",), "W1.7"), Start(("D2",))],
                ended,
                [refusal("no_task", task="W1.7", work="W1"), started],
            ),
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

    async def test_a_request_placed_again_is_named_with_the_tasks_it_joined(self):
        gate = asyncio.Event()
        again = placing(Steer("W1", ("D2",), "W1.2"), Start(("D2",)), Steer("W1", ("D2", "D9")))
        scripts = ([wait(gate), answer()], [wait(gate), checkpoint(), answer()])
        conductor = planned(split("teas", "hours"), again)
        adapter = ScriptedAdapter(*scripts, conductor=conductor, max_running=2)
        _, second = await self.admit_each(self.session(adapter), "Teas and hours.", "Only the weekend hours.")
        gate.set()
        steered = TEXTS.outcomes["steered"].format(work="W1", tasks="W1.2")
        placed = TEXTS.outcomes["already_placed"].format(delegation="D2", work="W1", tasks="W1.2")
        unknown = TEXTS.refusal("unplaced", delegations="D9")
        self.assertEqual(conductor.outcomes[1:], [steered, placed, f"{placed} {unknown}"])
        await read_all(second)

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

    async def test_a_new_work_records_the_thread_it_continues(self):
        cases = {
            "a start": (Start(("D2",)), None),
            "a steer into the ended work": (Steer("W1", ("D2",)), "W1"),
            "a steer naming its ended task": (Steer("W1", ("D2",), "W1.1"), "W1.1"),
        }
        for name, (decision, continues) in cases.items():
            with self.subTest(name):
                self.traces.clear()
                adapter = ScriptedAdapter([answer()], [answer()], conductor=then(decision))
                im = self.session(adapter)
                await read(im.admit(ask("Book a table."), delegation_id="D1"))
                await read(im.admit(ask("And for two."), delegation_id="D2"))
                linked = (im.ledger.works["W2"].continues, adapter.runs[1].task.continues)
                self.assertEqual(linked, (continues, continues))
                dequeued = [fields["continues"] for fields in self.traced("backend.dequeue")]
                self.assertEqual(dequeued, [None, continues])

    async def test_a_follow_up_continues_the_ended_work(self):
        gate = asyncio.Event()
        conductor = then(Steer("W1", ("D2",)))
        adapter = ScriptedAdapter([wait(gate), answer()], [answer()], conductor=conductor, delivery="not_applied")
        im = self.session(adapter)
        _, second = await self.admit_each(im, "Book a table.", "For two.")
        await self.until(lambda: im.ledger.delegations["D2"].work == "W1", "the steer")
        gate.set()
        await asyncio.wait_for(read_all(second), 2)
        self.assertEqual((im.ledger.works["W2"].continues, adapter.runs[1].task.continues), ("W1", "W1"))

    async def test_an_ended_tasks_question_is_cleared_once_a_decision_read_it(self):
        conductor = ScriptedConductor(SteerOpenWork().decide)
        scripts = ([asking(QUESTION), answer("The food is in.")], [answer("A latte it is.")])
        im = self.session(ScriptedAdapter(*scripts, conductor=conductor))
        await read(im.admit(ask("Food and a coffee."), delegation_id="D1"))
        record = im.ledger.tasks["W1.1"]
        self.assertEqual((record.state, record.question), ("answered", QUESTION))
        await read(im.admit(ask("A latte."), delegation_id="D2"))
        (read_by_conductor,) = next(work for work in conductor.given[1].works if work.work_id == "W1").tasks
        self.assertEqual(read_by_conductor.question, QUESTION)
        await self.until(lambda: record.question is None, "the decision to end")


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

    async def test_a_landed_reply_clears_the_question_and_a_new_ask_sets_it_again(self):
        landed, between, last = asyncio.Event(), asyncio.Event(), asyncio.Event()
        script = [
            asking(QUESTION),
            wait(landed),
            checkpoint(),
            wait(between),
            asking(HOT_OR_ICED),
            wait(last),
            answer("One iced latte."),
        ]
        im = self.session(ScriptedAdapter(script))
        first = im.admit(ask("A coffee."), delegation_id="D1")
        await self.until(lambda: self.traced("im.question.written"), "the question")
        second = im.admit(ask("A latte."), delegation_id="D2")
        await self.until(lambda: im.ledger.steerings, "the reply to wait in the task")
        landed.set()
        await self.until(lambda: im.ledger.steerings["S1"].state == "applied", "the reply to land")
        record = im.ledger.tasks["W1.1"]
        self.assertIsNone(record.question)
        between.set()
        await self.until(lambda: len(self.traced("im.question.written")) == 2, "the second question")
        self.assertEqual(record.question, HOT_OR_ICED)
        last.set()
        older, newer = await read(first), await read(second)
        self.assertEqual(
            (said(older), answers(newer)),
            ([("final_answer", QUESTION), ("commentary", HANDOVER)], [HOT_OR_ICED, "One iced latte."]),
        )

    async def test_a_reply_steered_into_the_asking_task_clears_only_its_question(self):
        food, drink, croissant = asyncio.Event(), asyncio.Event(), "Warm or cold croissant?"
        scripts = (
            [asking(croissant), wait(food), checkpoint(), answer("A warm croissant.")],
            [asking(QUESTION), wait(drink), checkpoint(), answer("A latte.")],
        )
        conductor = planned(split("food", "drink"), placing(Steer("W1", ("D2",), "W1.2")))
        im = self.session(ScriptedAdapter(*scripts, conductor=conductor, max_running=2))
        first = im.admit(ask("A croissant and a coffee."), delegation_id="D1")
        await self.until(lambda: len(self.traced("im.question.written")) == 2, "both questions")
        second = im.admit(ask("A latte."), delegation_id="D2")
        await self.until(lambda: im.ledger.steerings, "the reply to wait in the drink task")
        read_by_conductor = {task.task_id: task.question for task in conductor.given[1].works[0].tasks}
        self.assertEqual(read_by_conductor, {"W1.1": croissant, "W1.2": QUESTION})
        drink.set()
        tasks = im.ledger.tasks
        await self.until(lambda: tasks["W1.2"].state == "answered", "the drink task's answer")
        self.assertEqual([steering.task for steering in im.ledger.steerings.values()], ["W1.2"])
        self.assertEqual((tasks["W1.1"].question, tasks["W1.2"].question), (croissant, None))
        food.set()
        await read(first)
        self.assertEqual(answers(await read(second)), ["A latte.", "A warm croissant."])

    async def test_the_records_list_an_open_question_until_the_reply_lands(self):
        landed, done = asyncio.Event(), asyncio.Event()
        asker = [asking(QUESTION), wait(landed), checkpoint(), wait(done), answer()]
        plans = (Start(("D1",)), Start(("D2",)), Steer("W1", ("D3",), "W1.1"), Start(("D4",)))
        adapter = ScriptedAdapter(asker, [answer()], [answer()], conductor=planned(*map(placing, plans)), max_running=3)
        im = self.session(adapter)
        im.admit(ask("A coffee."), delegation_id="D1")
        await self.until(lambda: self.traced("im.question.written"), "the question")
        await read(im.admit(ask("When do you open?"), delegation_id="D2"))
        im.admit(ask("A latte."), delegation_id="D3")
        await self.until(lambda: im.ledger.steerings, "the reply to wait in the task")
        landed.set()
        await self.until(lambda: im.ledger.steerings["S1"].state == "applied", "the reply to land")
        await read(im.admit(ask("Thanks."), delegation_id="D4"))
        batch = next(given.steering for given in adapter.runs[0].boundaries if given.steering is not None)
        # Another Work's input, the reply's batch, and a later Work's input, once the reply landed.
        inputs = (adapter.runs[1].histories[0][-1], batch.items[-1], adapter.runs[2].histories[0][-1])
        seen = [next(line for line in records_of(item) if line["work"] == "W1").get("questions") for item in inputs]
        self.assertEqual(seen, [[QUESTION], [QUESTION], None])
        done.set()

    async def test_the_records_list_a_steered_request_once_under_the_work_it_joins(self):
        food, drink = asyncio.Event(), asyncio.Event()
        scripts = ([answer()], [wait(food), checkpoint(), answer()], [wait(drink), checkpoint(), answer()])
        plans = (Start(("D1",)), Start(("D2",), ("food", "drink")), Steer("W2", ("D3",)))
        adapter = ScriptedAdapter(*scripts, conductor=planned(*map(placing, plans)), max_running=2)
        im = self.session(adapter)
        await read(im.admit(ask("Book a table."), delegation_id="D1"))
        im.admit(ask("Food and a drink."), delegation_id="D2")
        await self.until(lambda: len(adapter.runs) == 3, "both tasks to start")
        third = im.admit(ask("Make it vegan."), delegation_id="D3")
        await self.until(lambda: len(im.ledger.steerings) == 2, "the steer to wait in both tasks")
        # The food task lands it first; the drink task takes it after.
        food.set()
        await self.until(lambda: "D3" in im.works["W2"].requests, "the food task to land it")
        drink.set()
        await read(third)
        for run in adapter.runs[1:]:
            with self.subTest(run.task.task_id):
                (batch,) = [given.steering for given in run.boundaries if given.steering is not None]
                lines = {line["work"]: line["requests"] for line in records_of(batch.items[-1])}
                self.assertEqual(lines, {"W1": ["Book a table."], "W2": ["Food and a drink.", "Make it vegan."]})


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


class StartsInCall(ScriptedAdapter):
    """A scripted adapter that notes the call each task's run starts in."""

    def __init__(self, *scripts, **options):
        super().__init__(*scripts, **options)
        self.calls = {}

    def start(self, task):
        self.calls[task.task_id] = CALL.get()
        return super().start(task)


class AfterCallTests(WorksCase):
    """Work a task marks to go on after the call: the mark, the hang-up that keeps it, the job, and a later call."""

    async def marked(self, im) -> None:
        await self.until(lambda: "W1" in im.ledger.works and im.ledger.works["W1"].after_call, "the mark")

    async def test_a_mark_writes_its_message_once_and_lifts_the_time_limit(self):
        gate = asyncio.Event()
        again = WorkTool("continue_after_call", {"message": "Still at it."})
        adapter = ScriptedAdapter([MARK, wait(gate), again, answer(REPORT)], max_running=2)
        im = self.session(adapter, continuable=True, work_timeout_seconds=0.2)
        first = im.admit(ask("Compare every supplier."), delegation_id="D1")
        await self.marked(im)
        # Past the Work's time limit.
        await asyncio.sleep(0.3)
        self.assertEqual((im.ledger.works["W1"].state, im.ledger.tasks["W1.1"].stop.state), ("running", "none"))
        self.assertEqual(said([event for _, event in self.wire.events]), [("final_answer", MESSAGE)])
        self.assertIsNone(im.works["W1"].summary().tasks[0].question)
        gate.set()
        events = await read(first)
        self.assertEqual((answers(events), events[-1]["type"]), ([MESSAGE, REPORT], "response.completed"))
        self.assertEqual((marks(im), len(self.traced("im.work.after_call"))), ([TEXTS.continued_output] * 2, 1))

    async def test_a_mark_is_refused_while_its_message_would_wait(self):
        parked, resumed = asyncio.Event(), asyncio.Event()
        menu = [app_call(call_id="call_menu"), answer("Menu read.")]
        report = [wait(parked), MARK, wait(resumed), MARK, answer(REPORT)]
        adapter = ScriptedAdapter(menu, report, conductor=planned(split("menu", "report")), max_running=3)
        im = self.session(adapter, continuable=True)
        first = await read(im.admit(ask("Read the menu and start the report."), delegation_id="D1", settings=APP))
        parked.set()
        await self.until(lambda: marks(im), "the mark while the app works")
        self.assertEqual((marks(im), im.ledger.works["W1"].after_call), ([TEXTS.not_continued_output], False))
        continuation = im.admit_client_request(app_outputs(first), delegation_id="D1", settings=APP)
        resumed.set()
        rest = await read(continuation)
        self.assertEqual(marks(im), [TEXTS.not_continued_output, TEXTS.continued_output])
        self.assertEqual(texts([event for _, event in self.wire.events]).count(MESSAGE), 1)
        self.assertEqual((MESSAGE in texts(rest), im.ledger.works["W1"].after_call), (True, True))

    async def test_marked_tasks_leave_a_slot_free_for_the_call(self):
        gate = asyncio.Event()
        jobs = ([MARK, wait(gate), answer("A.")], [MARK, wait(gate), answer("B.")])
        im = self.session(ScriptedAdapter(*jobs, conductor=StartEach(), max_running=2), continuable=True)
        _, second = await self.admit_each(im, "Compare the suppliers.", "Compare the shippers.")
        await self.until(lambda: len(marks(im)) == 2, "both marks")
        self.assertEqual(marks(im), [TEXTS.continued_output, TEXTS.not_continued_output])
        self.assertEqual([im.ledger.works[work].after_call for work in ("W1", "W2")], [True, False])
        gate.set()
        self.assertEqual(answers(await asyncio.wait_for(read_all(second), 2)), ["B."])

    async def test_a_mark_is_refused_once_the_call_has_ended(self):
        job, other = asyncio.Event(), asyncio.Event()
        scripts = ([MARK, wait(job), answer(REPORT)], [wait(other), MARK, answer("Free all day.")])
        adapter = ScriptedAdapter(*scripts, conductor=StartEach(), max_running=3)
        im = self.session(adapter, continuable=True, close_grace_seconds=5)
        im.admit(ask("Compare every supplier."), delegation_id="D1")
        await self.marked(im)
        im.admit(ask("What is on my calendar?"), delegation_id="D2")
        await self.until(lambda: im.ledger.tasks.get("W2.1") and im.ledger.tasks["W2.1"].state == "running", "W2")
        leaving = asyncio.ensure_future(im.leave())
        await self.until(lambda: im.detached, "the hang-up")
        # W2 marks its Work in the hang-up's close grace.
        other.set()
        await self.until(lambda: im.ledger.works["W2"].ended, "W2's end")
        told = [TEXTS.continued_output, TEXTS.not_continued_output]
        self.assertEqual((marks(im), im.ledger.works["W2"].after_call), (told, False))
        job.set()
        self.assertTrue(await asyncio.wait_for(leaving, 2))

    async def test_the_fallback_rule_starts_work_beside_a_marked_work(self):
        async def fails(given, place):
            raise RuntimeError("conductor down")

        gate = asyncio.Event()
        scripts = ([MARK, wait(gate), answer(REPORT)], [answer("We open at nine.")])
        adapter = ScriptedAdapter(*scripts, conductor=planned(placing(Start(("D1",))), fails), max_running=2)
        im = self.session(adapter, continuable=True)
        first = im.admit(ask("Compare every supplier."), delegation_id="D1")
        await self.marked(im)
        events = await read(im.admit(ask("When do you open?"), delegation_id="D2"))
        self.assertEqual(answers(events), ["We open at nine."])
        started = TEXTS.outcomes["started"].format(work="W2", tasks="W2.1")
        self.assertEqual([fields["outcome"] for fields in self.traced("im.placed_by_rule")], [started])
        gate.set()
        self.assertEqual(answers(await read(first)), [MESSAGE, REPORT])

    async def hung_up(self) -> tuple:
        """Hang up while W1, which asked a question and was marked, runs with D3 steered in and not landed.

        Meanwhile W2, unmarked, runs and D4 waits for a decision. Return the session, its adapter, W1's gate and the
        four requests' streams.
        """
        job = asyncio.Event()
        script = [asking(QUESTION), MARK, wait(job), checkpoint(), answer(REPORT)]
        plans = (placing(Start(("D1",))), placing(Start(("D2",))), placing(Steer("W1", ("D3",))), undecided)
        adapter = ScriptedAdapter(script, [wait(asyncio.Event()), answer()], conductor=planned(*plans), max_running=3)
        im = self.session(adapter, continuable=True, renew=lambda: ScriptedAdapter(max_running=3))
        streams = [im.admit(ask("Compare every supplier."), delegation_id="D1").stream]
        await self.marked(im)
        for number, text in ((2, "What is on my calendar?"), (3, "Make it shorter."), (4, "And book a table.")):
            streams.append(im.admit(ask(text), delegation_id=f"D{number}").stream)
            await self.turns()
        await self.until(lambda: im.waits_for_decision and im.ledger.delegations["D3"].work == "W1", "the placements")
        self.assertTrue(await self.hang_up(im))
        return im, adapter, job, streams

    async def test_a_hang_up_keeps_the_conversation_while_its_marked_work_runs(self):
        im, adapter, _, streams = await self.hung_up()
        self.assertEqual(im.ledger.session.state, "detached")
        works = {work_id: work.state for work_id, work in im.ledger.works.items()}
        self.assertEqual(works, {"W1": "running", "W2": "cancelled"})
        ended = [(await asyncio.wait_for(read_all(stream), 2))[-1]["response"] for stream in streams]
        self.assertEqual([(each["status"], each["incomplete_details"]) for each in ended], [("cancelled", None)] * 4)
        states = {delegation: record.state for delegation, record in im.ledger.delegations.items()}
        self.assertEqual(states, {"D1": "running", "D2": "cancelled", "D3": "queued", "D4": "cancelled"})
        self.assertEqual((im.waits_for_decision, adapter.released, adapter.closed), (False, 1, False))
        self.assertEqual(im.ledger.tasks["W1.1"].question, QUESTION)
        with self.assertRaises(DelegationError):
            im.admit(ask("Hello?"), delegation_id="D5")

    async def test_after_the_hang_up_the_job_runs_unheard_to_its_end(self):
        im, adapter, job, _ = await self.hung_up()
        written = len(self.wire.events)
        job.set()
        await self.until(lambda: not im.works, "the job's end")
        self.assertEqual((im.ledger.works["W1"].state, im.ledger.tasks["W1.1"].answer), ("answered", REPORT))
        # The request steered in before the hang-up landed after it, and carried the job to its end.
        self.assertEqual([im.ledger.delegations[d].state for d in ("D1", "D3")], ["handed_over", "answered"])
        self.assertEqual(len(self.wire.events), written)
        await self.until(lambda: adapter.closed, "the backend's close")
        self.assertTrue(im.detached)

    async def test_what_the_job_writes_after_the_hang_up_is_dropped(self):
        part_a, part_b = asyncio.Event(), asyncio.Event()
        scripts = ([MARK, wait(part_a), answer("Part A.")], [wait(part_b), checkpoint(), answer("Part B, for Europe.")])
        conductor = planned(split("a", "b"), placing(Steer("W1", ("D2",), "W1.2")))
        im = self.session(ScriptedAdapter(*scripts, conductor=conductor, max_running=3), continuable=True)
        im.admit(ask("Compare the suppliers, in two parts."), delegation_id="D1")
        await self.marked(im)
        self.assertTrue(await self.hang_up(im))
        part_a.set()
        await self.until(lambda: im.ledger.tasks["W1.1"].state == "answered", "part A's answer")
        await self.call_again(im)
        later = im.admit(ask("Only Europe for part B."), delegation_id="D2")
        await self.until(lambda: im.ledger.delegations["D2"].work == "W1", "the steer into part B")
        part_b.set()
        # Part A answered while nobody heard the job: the later call that takes it over hears only part B.
        self.assertEqual(answers(await read(later)), ["Part B, for Europe."])

    async def steered_then_hung_up(self, *scripts) -> tuple:
        """Mark D1's W1, steer D2 into it, which its task never applies, and hang up; ``scripts`` run later Works.

        Return the session and W1's gate.
        """
        gate = asyncio.Event()
        conductor = planned(*map(placing, (Start(("D1",)), Steer("W1", ("D2",)), Steer("W1", ("D3",)))))
        job = [MARK, wait(gate), answer(REPORT)]
        adapter = ScriptedAdapter(job, *scripts, conductor=conductor, max_running=3, delivery="not_applied")
        im = self.session(adapter, continuable=True)
        im.admit(ask("Compare every supplier."), delegation_id="D1")
        await self.marked(im)
        im.admit(ask("Make it shorter."), delegation_id="D2")
        await self.until(lambda: im.ledger.delegations["D2"].work == "W1", "the steer into the job")
        self.assertTrue(await self.hang_up(im))
        return im, gate

    async def test_a_leftover_request_is_cancelled_while_detached(self):
        im, gate = await self.steered_then_hung_up()
        gate.set()
        await self.until(lambda: not im.works, "the job's end")
        self.assertEqual((im.ledger.delegations["D2"].state, list(im.ledger.works)), ("cancelled", ["W1"]))
        self.assertEqual(self.traced("im.follow_up"), [])

    async def test_a_leftover_whose_call_ended_is_cancelled_in_a_later_call(self):
        im, gate = await self.steered_then_hung_up([answer("Shorter, and only Europe.")])
        await self.call_again(im)
        later = im.admit(ask("And only Europe."), delegation_id="D3")
        await self.until(lambda: im.ledger.delegations["D3"].work == "W1", "the steer into the job")
        gate.set()
        self.assertEqual(answers(await read(later)), ["Shorter, and only Europe."])
        self.assertEqual(im.ledger.delegations["D2"].state, "cancelled")
        self.assertEqual(self.traced("im.follow_up"), [{"work_id": "W1", "delegation_ids": ["D3"]}])

    async def test_attach_waits_for_the_hang_up_and_serves_the_next_call_on_a_new_backend(self):
        other = asyncio.Event()
        plans = (placing(Start(("D1",))), placing(Start(("D2",))), undecided)
        scripts = ([MARK, answer(REPORT)], [wait(other), answer()])
        first = ScriptedAdapter(*scripts, conductor=planned(*plans), max_running=3)
        later = ScriptedConductor(starts)
        second = ScriptedAdapter([answer("Yes, it is ready.")], conductor=later, max_running=3)
        im = self.session(first, continuable=True, renew=lambda: second, close_grace_seconds=5)
        await read(im.admit(ask("Compare every supplier."), delegation_id="D1"))
        im.admit(ask("What is on my calendar?"), delegation_id="D2")
        await self.until(lambda: "W2" in im.works, "W2")
        unplaced = im.admit(ask("And book a table."), delegation_id="D3")
        leaving = asyncio.ensure_future(im.leave())
        await self.until(lambda: im.detached, "the hang-up")
        attaching = asyncio.ensure_future(im.attach())
        await self.turns()
        self.assertFalse(attaching.done())
        # W2 ends in the hang-up's close grace; then no Work is open, and the backend closes.
        other.set()
        await asyncio.wait_for(attaching, 2)
        self.assertEqual((leaving.result(), im.ledger.session.state, first.closed), (True, "active", True))
        self.assertIs(im.backend, second)
        self.assertEqual(self.traced("im.attached"), [{"after_call": {"W1": "D1"}}])
        cancelled = (await read(unplaced))[-1]["response"]["status"]
        self.assertEqual((cancelled, im.ledger.delegations["D3"].state), ("cancelled", "cancelled"))
        events = await read(im.admit(ask("Is my research ready?"), delegation_id="D5"))
        self.assertEqual(answers(events), ["Yes, it is ready."])
        (given,) = later.given
        self.assertEqual(([arrival.delegation_id for arrival in given.arrivals], given.new_from), (["D5"], 0))
        self.assertIn("Compare every supplier.", str(given.conversation))
        with self.assertRaisesRegex(DelegationError, "another call"):
            await self.call_again(im)

    async def test_a_later_calls_request_takes_the_job_over_at_once(self):
        gate = asyncio.Event()
        conductor = planned(placing(Start(("D1",))), placing(Steer("W1", ("D2",))))
        job = [MARK, wait(gate), checkpoint(), answer("Shorter.")]
        im = self.session(ScriptedAdapter(job, conductor=conductor, max_running=2), continuable=True)
        first = im.admit(ask("Write the long report."), delegation_id="D1")
        await self.marked(im)
        self.assertTrue(await self.hang_up(im))
        await self.call_again(im)
        second = im.admit(ask("Make it shorter."), delegation_id="D2")
        await self.until(lambda: im.ledger.delegations["D2"].work == "W1", "the steer into the job")
        gate.set()
        older, taker = await read(first), await read(second)
        self.assertEqual((named(taker), answers(taker)), (older[0]["response"]["id"], ["Shorter."]))
        self.assertEqual((im.ledger.delegations["D1"].state, taker[-1]["type"]), ("handed_over", "response.completed"))
        *requests, said_last = im.history
        self.assertEqual(without_ids(requests), [*ask("Write the long report."), *ask("Make it shorter.")])
        self.assertEqual((said_last["role"], message_text(said_last)), ("assistant", "Shorter."))

    async def test_a_request_whose_call_ended_never_takes_the_work_from_a_later_call(self):
        part_a, part_b, europe = asyncio.Event(), asyncio.Event(), asyncio.Event()
        scripts = (
            [MARK, wait(part_a), checkpoint(), answer("Part A, shorter.")],
            [wait(part_b), checkpoint(), wait(europe), answer("Part B, for Europe.")],
        )
        steers = (placing(Steer("W1", ("D2",), "W1.1")), placing(Steer("W1", ("D3",), "W1.2")))
        adapter = ScriptedAdapter(*scripts, conductor=planned(split("a", "b"), *steers), max_running=3)
        im = self.session(adapter, continuable=True)
        im.admit(ask("Compare the suppliers, in two parts."), delegation_id="D1")
        await self.marked(im)
        im.admit(ask("Make part A shorter."), delegation_id="D2")
        await self.until(lambda: im.ledger.delegations["D2"].work == "W1", "the steer into part A")
        self.assertTrue(await self.hang_up(im))
        await self.call_again(im)
        later = im.admit(ask("Only Europe for part B."), delegation_id="D3")
        await self.until(lambda: im.ledger.delegations["D3"].work == "W1", "the steer into part B")
        part_b.set()
        await self.until(lambda: im.ledger.works["W1"].carrier == "D3", "the later call's takeover")
        part_a.set()
        await self.until(lambda: im.ledger.delegations["D2"].ended, "the earlier call's request to land")
        self.assertEqual(im.ledger.works["W1"].carrier, "D3")
        europe.set()
        self.assertEqual(answers(await read(later)), ["Part A, shorter.", "Part B, for Europe."])
        states = [im.ledger.delegations[d].state for d in ("D1", "D2", "D3")]
        self.assertEqual(states, ["handed_over", "handed_over", "answered"])

    async def test_the_records_show_a_kept_tasks_answer_start_and_newest_steps(self):
        steps = [OwnCall("lookup", call_id=f"call_step_{n}") for n in range(RECORD_STEPS + 2)]
        long = "Suppliers ranked by price. " + "x" * RECORD_TEXT_CHARS
        conductor = ScriptedConductor(starts)
        scripts = ([MARK, *steps, answer(long)], [answer("We open at nine.")], [answer()])
        im = self.session(ScriptedAdapter(*scripts, conductor=conductor, max_running=2), continuable=True)
        for number, text in enumerate(("Compare every supplier.", "When do you open?", "Thanks."), 1):
            await read(im.admit(ask(text), delegation_id=f"D{number}"))
        kept, plain = conductor.given[2].works
        (task,) = kept.tasks
        self.assertEqual((kept.after_call, task.answer), (True, long[:RECORD_TEXT_CHARS]))
        self.assertEqual(task.steps, ("lookup: done",) * RECORD_STEPS)
        self.assertEqual((plain.after_call, plain.tasks[0].answer), (False, None))
        lines = {line["work"]: line for line in im.records()}
        records = ([long[:RECORD_TEXT_CHARS]], ["lookup: done"] * RECORD_STEPS)
        self.assertEqual((lines["W1"]["answers"], lines["W1"]["steps"]), records)
        self.assertEqual(sorted(lines["W2"]), ["requests", "state", "steps", "work"])

    async def test_the_keep_time_starts_again_when_the_job_ends(self):
        gate = asyncio.Event()
        adapter = ScriptedAdapter([MARK, wait(gate), answer(REPORT)], max_running=2)
        im = self.session(adapter, continuable=True, after_call_seconds=0.3)
        im.admit(ask("Compare every supplier."), delegation_id="D1")
        await self.marked(im)
        self.assertTrue(await self.hang_up(im))
        # The job ends a third of the way through the keep time.
        await asyncio.sleep(0.1)
        gate.set()
        ended = time.monotonic()
        await self.until(lambda: im.ledger.session.ended, "the keep time's end")
        self.assertGreaterEqual(time.monotonic() - ended, 0.29)
        self.assertEqual((im.ledger.works["W1"].state, adapter.closed), ("answered", True))

    async def test_the_keep_times_end_stops_a_job_still_running(self):
        adapter = ScriptedAdapter([MARK, wait(asyncio.Event()), answer(REPORT)], max_running=2)
        im = self.session(adapter, continuable=True, after_call_seconds=0.05)
        im.admit(ask("Compare every supplier."), delegation_id="D1")
        await self.marked(im)
        self.assertTrue(await self.hang_up(im))
        await self.until(lambda: im.ledger.session.ended, "the keep time's end")
        await asyncio.wait_for(im.close(), 2)
        self.assertEqual((im.ledger.works["W1"].state, adapter.runs[0].stops, adapter.closed), ("cancelled", 1, True))

    async def test_a_hang_up_leaves_the_backend_closing_and_attach_and_close_wait_for_it(self):
        for then in ("attach", "close"):
            with self.subTest(then):
                first, second = ClosesWhenLetGo([MARK, answer(REPORT)], max_running=2), ScriptedAdapter(max_running=2)
                im = self.session(first, continuable=True, renew=lambda second=second: second)
                # Before the session's own cleanup: a close that never ends must not hold the test.
                self.addCleanup(first.may_close.set)
                await read(im.admit(ask("Compare every supplier."), delegation_id="D1"))
                self.assertTrue(await self.hang_up(im))
                await self.until(lambda first=first: first.closes, "the backend's close")
                waiting = asyncio.ensure_future(im.attach() if then == "attach" else im.close())
                await self.turns()
                self.assertEqual((waiting.done(), first.closed), (False, False))
                first.may_close.set()
                await asyncio.wait_for(waiting, 2)
                kept = second if then == "attach" else first
                self.assertEqual((first.closes, first.closed, im.backend), (1, True, kept))

    async def test_a_follow_up_in_a_later_call_starts_in_that_call(self):
        gate = asyncio.Event()
        conductor = planned(placing(Start(("D1",))), placing(Steer("W1", ("D2",))))
        scripts = ([MARK, wait(gate), answer(REPORT)], [answer("Shorter.")])
        adapter = StartsInCall(*scripts, conductor=conductor, max_running=3, delivery="not_applied")

        async def first_call():
            CALL.set("call 1")
            im = self.session(adapter, continuable=True)
            im.admit(ask("Write the long report."), delegation_id="D1")
            await self.marked(im)
            self.assertTrue(await self.hang_up(im))
            return im

        async def second_call(im):
            CALL.set("call 2")
            await self.call_again(im)
            stream = im.admit(ask("Make it shorter."), delegation_id="D2").stream
            await self.until(lambda: im.ledger.delegations["D2"].work == "W1", "the steer into the job")
            gate.set()
            return await asyncio.wait_for(read_all(stream), 2)

        im = await asyncio.create_task(first_call())
        self.assertEqual(answers(await asyncio.create_task(second_call(im))), ["Shorter."])
        self.assertEqual(adapter.calls, {"W1.1": "call 1", "W2.1": "call 2"})
