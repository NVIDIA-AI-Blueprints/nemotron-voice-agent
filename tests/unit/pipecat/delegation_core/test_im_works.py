# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""An IM session's Works: the scheduler, placement and its checks, steering, handovers, stops and the endings."""

import asyncio
import contextvars
import json
import time
from itertools import pairwise, product

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.conductors import SteerOpenWork
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import (
    CallerMessage,
    Cancel,
    CompleteDecision,
    DecisionRejected,
    Start,
    Steer,
    TaskSpec,
    WorkTarget,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.run import (
    RECORD_STEPS,
    RECORD_TEXT_CHARS,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.session import DelegationError
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.items import WorkerData, message_text
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.ledger.records import MAX_KEPT_CLOSED
from tests.unit.pipecat.core_helpers import load
from tests.unit.pipecat.delegation_core.helpers import (
    MARK,
    MENU,
    MESSAGE,
    QUESTION,
    TEXTS,
    Act,
    ClosesWhenLetGo,
    Fail,
    IMCase,
    OwnCall,
    ScriptedAdapter,
    ScriptedConductor,
    ScriptedRun,
    ScriptStep,
    StartEach,
    WorkTool,
    answer,
    answers,
    app_call,
    app_outputs,
    ask,
    asking,
    call_item,
    checkpoint,
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
HANDOVER, OWED = TEXTS.handover_line.format(delegation="D2"), TEXTS.owed_output
LATE = ["response.created", "response.queued", "response.in_progress", "response.completed"]
ENDINGS = load()
OLD, NEW = "Booked for 6 pm.", "Booked for 7 pm."
HOT_OR_ICED, REPORT = "Hot or iced?", "The comparison is ready."
# The call a task's run starts in, as a call's own context names it.
CALL = contextvars.ContextVar("call", default=None)


def starts(given, place):
    return place(Start(tuple(arrival.delegation_id for arrival in given.arrivals)))


def split(*briefs: str):
    return placing(Start(("D1",), tuple(TaskSpec(brief) for brief in briefs)))


def then(*decisions):
    """Build a conductor that starts D1's Work, then places ``decisions`` for the next arrival."""
    return planned(Start(("D1",)), *decisions)


async def undecided(given, place):
    """Place nothing, ever: the requests wait for a decision."""
    await asyncio.Event().wait()


def named(events: list[dict]) -> str | None:
    """Return the response a taker's in_progress names as the one it took over."""
    return next(e for e in events if e["type"] == "response.in_progress")["response"].get("previous_response_id")


async def read(admission) -> list[dict]:
    return await read_all(admission.stream, timeout=2)


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
    async def test_operational_status_without_work_is_rejected_before_effects_and_can_be_clarified(self):
        rejected = []

        async def route(given, place):
            status = CompleteDecision((CallerMessage(("D1",), "status", "The search is running."),))
            with self.assertRaisesRegex(DecisionRejected, "requires open or retained") as refusal:
                await place(status)
            rejected.append(str(refusal.exception))
            self.assertEqual((list(im.ledger.works), list(im.ledger.controls), im.arrivals), ([], [], ["D1"]))
            self.assertEqual(im.ledger.delegations["D1"].state, "queued")
            await place(CompleteDecision((CallerMessage(("D1",), "question", "Which job should I check?"),)))

        im, adapter = self.scripted_session(conductor=ScriptedConductor(route), max_running=3)
        events = await self.request(im, "Any update?", "D1", timeout=2)
        self.assertEqual(texts(events), ["Which job should I check?"])
        self.assertEqual((len(rejected), len(adapter.runs), len(im.ledger.controls)), (1, 0, 1))
        self.assertEqual(next(iter(im.clarifications.values()))["state"], "waiting")

    async def test_status_distinguishes_jobs_with_missing_or_shared_briefs(self):
        for briefs, labels in (
            ((), ("W1.1", "W2.1")),
            (("Lookup", "Lookup"), ("W1.1", "W2.1")),
            (("   ", "Second lookup"), ("W1.1", "Second lookup")),
        ):
            with self.subTest(briefs=briefs):
                first, second = ((TaskSpec(brief),) for brief in briefs) if briefs else ((), ())
                finish = asyncio.Event()
                im, adapter = self.scripted_session(
                    [answer()],
                    [wait(finish), answer()],
                    conductor=planned(
                        Start(("D1",), first),
                        Start(("D2",), second),
                        CallerMessage(("D3",), "status", targets=(WorkTarget("W1"), WorkTarget("W2"))),
                    ),
                    max_running=2,
                )
                await self.request(im, "First lookup.", "D1", timeout=2)
                lookup = im.admit(ask("Second lookup."), delegation_id="D2")
                await self.until(lambda adapter=adapter: len(adapter.runs) == 2, "second task starts")
                status = texts(await self.request(im, "Status of both?", "D3", timeout=2))
                answered, running = labels
                self.assertEqual(
                    status,
                    [
                        f"Status for “{answered}”. The job has provided an update.\n"
                        f"Status for “{running}”. The job is still running."
                    ],
                )
                finish.set()
                await read(lookup)

    async def test_a_fully_continued_target_repairs_once_without_consuming_its_source(self):
        rejected, attempts = [], []

        async def route(given, place):
            source = given.arrivals[0].delegation_id
            if source == "D1":
                return await place(Start((source,), (TaskSpec("other order"), TaskSpec("this order"))))
            if source == "D2":
                return await place(Steer((source,), (WorkTarget("W1", ("W1.2",)),)))
            attempts.append(given.attempt)
            if given.attempt == 1:
                self.assertIsNone(given.decision_rejection)
                before = len(im.ledger.controls), list(im.ledger.works)
                try:
                    return await place(Steer((source,), (WorkTarget("W1", ("W1.2",)),)))
                except DecisionRejected as error:
                    rejected.append(json.loads(str(error)))
                    self.assertEqual((len(im.ledger.controls), list(im.ledger.works)), before)
                    self.assertEqual(im.ledger.delegations[source].state, "queued")
                    self.assertIn(source, im.arrivals)
                    self.assertNotIn(im._decision_epoch, im.executor.submissions)
                    raise
            current = next(task for work in given.works for task in work.tasks if task.task_id == "W2.1")
            self.assertEqual(json.loads(given.decision_rejection), rejected[0])
            self.assertEqual((current.continues, current.steering), ("W1.2", "continuable"))
            return await place(Steer((source,), (WorkTarget("W2", ("W2.1",)),)))

        im, adapter = self.scripted_session(
            [answer("Other order.")],
            [answer("Original status.")],
            [answer("Updated status.")],
            [answer("Latest status.")],
            conductor=ScriptedConductor(route),
            max_running=3,
        )
        await self.request(im, "Check both orders.", "D1", timeout=2)
        await self.request(im, "Update the second order.", "D2", timeout=2)
        events = await self.request(im, "Continue with that order.", "D3", timeout=2)
        self.assertEqual((attempts, texts(events), len(adapter.runs)), ([1, 2], ["Latest status."], 4))
        self.assertEqual((rejected[0]["reason"], rejected[0]["executed"]), ("continued_targets", False))
        self.assertEqual(self.traced("im.decision.rejected"), rejected)
        self.assertEqual(
            rejected[0]["actions"][0]["targets"],
            [
                {
                    "reason": "continued",
                    "work_id": "W1",
                    "task_id": "W1.2",
                    "successor_work_id": "W2",
                    "successor_task_id": "W2.1",
                }
            ],
        )
        self.assertEqual(im.ledger.tasks["W1.2"].successor_task_id, "W2.1")
        self.assertEqual(im.ledger.tasks["W2.1"].successor_task_id, "W3.1")
        self.assertEqual(len([row for row in self.traced("im.decide.retry") if row["delegation_ids"] == ["D3"]]), 1)

    async def test_a_continued_target_does_not_replay_a_valid_independent_stop(self):
        attempts = []
        proceed = asyncio.Event()

        async def route(given, place):
            source = given.arrivals[0].delegation_id
            if source == "D1":
                return await place(Start((source,), (TaskSpec("running order"), TaskSpec("finished order"))))
            if source == "D2":
                return await place(Steer((source,), (WorkTarget("W1", ("W1.2",)),)))
            attempts.append(given.attempt)
            return await place(
                CompleteDecision(
                    (
                        Steer(("D4",), (WorkTarget("W1", ("W1.2",)),)),
                        Cancel(("D4",), (WorkTarget("W1", ("W1.1",)),)),
                        Start((source,)),
                    )
                )
            )

        im, adapter = self.scripted_session(
            [wait(asyncio.Event())],
            [answer("Initial status.")],
            [wait(proceed), app_call(call_id="stale_order"), answer("Updated status.")],
            [app_call(call_id="menu_lookup"), answer("Menu checked.")],
            conductor=ScriptedConductor(route),
            max_running=3,
        )
        first = im.admit(ask("Check both orders."), delegation_id="D1")
        await self.until(
            lambda: im.ledger.tasks.get("W1.2") and im.ledger.tasks["W1.2"].ended, "the second order's initial answer"
        )
        continued = im.admit(ask("Continue the finished order."), delegation_id="D2", settings=APP)
        await self.until(lambda: len(adapter.runs) == 3, "the continuation started")
        menu_request = im.admit(ask("Check the current menu."), delegation_id="D3", settings=APP)
        control_request = im.admit(ask("Update that order and stop the other."), delegation_id="D4")
        result, menu = await asyncio.gather(read(control_request), read(menu_request))
        proceed.set()
        self.assertEqual(function_calls(await read(continued)), [])
        self.assertEqual(function_calls(menu), ["menu_lookup"])
        await read(first)
        controls = [row for row in im.ledger.controls.values() if row.sources == ("D4",)]
        self.assertEqual([(row.kind, row.state) for row in controls], [("Steer", "refused"), ("Cancel", "applied")])
        self.assertEqual(
            (controls[0].receipt["outcome"], controls[0].receipt["executed"]), ("target_unavailable", False)
        )
        self.assertEqual((attempts, adapter.runs[0].stops, len(adapter.runs)), ([1], 1, 4))
        self.assertEqual(result[-1]["type"], "response.completed")
        self.assertIn(im.texts.routing_failed_output, texts(result))
        self.assertEqual(im.ledger.tasks["W1.2"].successor_task_id, "W2.1")

    async def test_a_retained_predecessor_keeps_its_successor_after_child_pruning(self):
        async def route(given, place):
            source = given.arrivals[0].delegation_id
            if source == "D1":
                return await place(Start((source,), (TaskSpec("open question"), TaskSpec("other order"))))
            if source == "D2":
                return await place(Steer((source,), (WorkTarget("W1", ("W1.2",)),)))
            return await place(Start((source,)))

        im, adapter = self.scripted_session(
            [asking(QUESTION), answer("Waiting for a later answer.")],
            [answer("Other order.")],
            [answer("Its continuation.")],
            conductor=ScriptedConductor(route),
            max_running=3,
        )
        await self.request(im, "Check these separate matters.", "D1", timeout=2)
        await self.request(im, "Continue the other order.", "D2", timeout=2)
        for number in range(3, MAX_KEPT_CLOSED + 5):
            await self.request(im, "A separate job.", f"D{number}", timeout=2)
        self.assertNotIn("W2", im.ledger.works)
        self.assertNotIn("W2.1", im.thread_history)
        predecessor = im.ledger.tasks["W1.2"]
        self.assertEqual((predecessor.successor_work_id, predecessor.successor_task_id), ("W2", "W2.1"))
        self.assertEqual(im.task_steering_state(predecessor), "continued")
        before = list(im.ledger.works), len(adapter.runs)
        with self.assertRaisesRegex(ValueError, '"reason": "continued"'):
            im.executor._continuation(predecessor.id)
        with self.assertRaisesRegex(ValueError, '"reason": "continued"'):
            im.start_work(["D1"], (TaskSpec("first matter", "W1.1"), TaskSpec("other order", "W1.2")))
        self.assertIsNone(im.ledger.tasks["W1.1"].successor_task_id)
        self.assertEqual(im.ledger.tasks["W1.1"].questions[0].owner_task_id, "W1.1")
        self.assertEqual((list(im.ledger.works), len(adapter.runs)), before)

    async def test_a_task_waiting_for_the_app_holds_no_slot(self):
        gates = [asyncio.Event(), asyncio.Event()]
        scripts = [[app_call(), answer("One.")], [wait(gates[0]), Fail()], [wait(gates[1]), answer()]]
        im, adapter = self.scripted_session(*scripts, conductor=StartEach(), max_running=2)
        first = await self.request(im, "Request 1.", "D1", settings=APP)
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
            await place(CompleteDecision(tuple(Start((arrival.delegation_id,)) for arrival in given.arrivals)))

        im = self.session(ScriptedAdapter(conductor=ScriptedConductor(slow), max_running=3))
        await self.admit_each(im, "One.", "Two.")
        self.assertEqual(await kept[0](Start(("D1",))), TEXTS.refusal("ended"))
        im.admit(ask("Three."), delegation_id="D3")
        await self.turns()
        thinking.set()
        await self.turns()
        self.assertEqual(calls, [["D1"], ["D1", "D2"], ["D3"]])

    async def test_a_finished_conductor_cannot_place_from_a_queued_callback(self):
        callbacks, batches = [], []

        async def plan(given, place):
            sources = tuple(arrival.delegation_id for arrival in given.arrivals)
            batches.append(sources)
            if len(batches) == 1:
                callbacks.append(asyncio.create_task(place(Start(sources))))
                return
            await place(Start(sources))

        adapter = ScriptedAdapter([answer("Current result.")], conductor=ScriptedConductor(plan), max_running=2)
        im = self.session(adapter)
        result = await self.request(im, "Check the reservation.", "D1", timeout=2)

        self.assertEqual(await asyncio.wait_for(callbacks[0], 2), TEXTS.refusal("ended"))
        self.assertEqual(batches, [("D1",), ("D1",)])
        self.assertEqual(answers(result), ["Current result."])
        self.assertEqual(len(adapter.runs), 1)

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
        await self.request(im, "Teas?", "D1")
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

    async def test_an_omitted_bound_source_is_rejected_before_effects_and_corrected_in_the_same_call(self):
        receipts, callbacks = [], []
        stop = Cancel(("D2",), (), scope="all")
        decision = CompleteDecision((stop, Start(("D3",))))

        async def complete(given, place):
            callbacks.append(place)
            with self.assertRaisesRegex(DecisionRejected, "missing: D3"):
                await place(stop)
            with self.assertRaisesRegex(DecisionRejected, "A stopped queued request cannot also be used"):
                await place(CompleteDecision((*decision.actions, Start(("D1",)))))
            self.assertEqual((im.arrivals, im.ledger.works, im.ledger.controls), (["D1", "D2", "D3"], {}, {}))
            self.assertFalse(adapter.runs)
            receipts.extend(await asyncio.gather(place(decision), place(decision)))

        im, adapter = self.scripted_session(
            [answer("The shop opens at nine.")],
            conductor=planned(complete, Cancel(("D4",), (), scope="all")),
            max_running=3,
        )
        first = im.admit(ask("Find available rooms."), delegation_id="D1")
        second = im.admit(ask("Cancel all current tasks."), delegation_id="D2")
        third = im.admit(ask("What time does the shop open?"), delegation_id="D3")
        cancelled, confirmation, independent = await asyncio.gather(read(first), read(second), read(third))
        self.assertEqual(
            (cancelled[-1]["type"], cancelled[-1]["response"]["status"]), ("response.incomplete", "cancelled")
        )
        self.assertEqual(answers(confirmation), [TEXTS.queued_cancelled_line])
        self.assertEqual(answers(independent), ["The shop opens at nine."])
        await self.until(lambda: len(receipts) == 2, "both duplicate submission receipts")
        original, duplicate = receipts
        self.assertEqual(original, duplicate)
        self.assertEqual(json.loads(original)["actions"][0]["receipt"]["cancelled_delegation_ids"], ["D1"])
        self.assertEqual(await callbacks[0](decision), TEXTS.refusal("ended"))
        no_work = await self.request(im, "Cancel all current tasks.", "D4", timeout=2)
        self.assertEqual((answers(no_work), no_work[-1]["type"]), ([TEXTS.no_active_work_line], "response.completed"))
        self.assertEqual(len(adapter.runs), 1)

    async def test_each_decision_is_checked_against_what_it_names(self):
        receipts = []

        async def invalid_then_partial(given, place):
            invalid = CompleteDecision((Start(("D2",)), Steer(("D9",), (WorkTarget("W1"),))))
            with self.assertRaisesRegex(ValueError, "bound pending caller sources"):
                await place(invalid)
            self.assertEqual((list(im.ledger.works), len(adapter.runs), len(im.ledger.controls)), (["W1"], 1, 1))
            self.assertEqual(im.ledger.delegations["D2"].state, "queued")
            receipts.append(
                json.loads(
                    await place(
                        CompleteDecision(
                            (
                                Start(("D2",)),
                                Cancel(("D3",), (WorkTarget("W9"),)),
                                Cancel(("D3",), (WorkTarget("W1"),)),
                                Start(("D4",)),
                            )
                        )
                    )
                )
            )

        im, adapter = self.scripted_session(
            [wait(asyncio.Event())],
            [app_call(call_id="send_order"), answer()],
            [app_call(call_id="check_menu"), answer()],
            conductor=planned(Start(("D1",)), invalid_then_partial),
            max_running=3,
        )
        first = im.admit(ask("Check the tea order."), delegation_id="D1")
        await self.until(lambda: len(adapter.runs) == 1, "tea order started")
        second = im.admit(ask("Send the coffee order."), delegation_id="D2", settings=APP)
        third = im.admit(ask("Stop both orders."), delegation_id="D3")
        fourth = im.admit(ask("Check the menu."), delegation_id="D4", settings=APP)
        tea, coffee, stop_result, menu = await asyncio.gather(*(read(r) for r in (first, second, third, fourth)))
        _, refused, stopped, _ = receipts[0]["actions"]
        self.assertEqual((refused["state"], stopped["kind"]), ("refused", "Cancel"))
        self.assertEqual((texts(tea), stop_result[-1]["type"]), ([], "response.completed"))
        self.assertEqual((function_calls(coffee), function_calls(menu)), ([], ["check_menu"]))

    async def test_a_request_cannot_be_rebound_after_its_complete_decision(self):
        gate = asyncio.Event()
        receipts = []

        async def again(given, place):
            decision = CompleteDecision((Steer(("D2",), (WorkTarget("W1", ("W1.2",)),)),))
            receipts.append(await place(decision))
            self.assertEqual(await place(decision), receipts[0])
            for changed in (Start(("D2",)), Steer(("D2", "D9"), (WorkTarget("W1"),))):
                with self.assertRaisesRegex(ValueError, "already accepted a different complete decision"):
                    await place(CompleteDecision((changed,)))

        scripts = ([wait(gate), answer()], [wait(gate), checkpoint(), answer()])
        conductor = planned(split("teas", "hours"), again)
        im, adapter = self.scripted_session(*scripts, conductor=conductor, max_running=2)
        _, second = await self.admit_each(im, "Teas and hours.", "Only the weekend hours.")
        (action,) = json.loads(receipts[0])["actions"]
        self.assertEqual(
            (action["kind"], action["receipt"]["targets"], action["state"]),
            ("Steer", [{"work_id": "W1", "task_ids": ["W1.2"]}], "pending"),
        )
        gate.set()
        await read_all(second)
        self.assertEqual([steering.task for steering in im.ledger.steerings.values()], ["W1.2"])
        self.assertEqual((len(adapter.runs), len(im.ledger.controls)), (2, 2))

    async def test_a_restart_over_max_running_is_refused(self):
        gate, splits = asyncio.Event(), placing(Start(("D2",), (TaskSpec("teas"), TaskSpec("hours"))))
        conductor = planned(Start(("D1",)), splits, Steer(("D3",), (WorkTarget("W1", None),), clarifications=()))
        scripts = [[answer("Done early.")], [wait(gate), answer()], [wait(gate), checkpoint(), answer()]]
        im, adapter = self.scripted_session(*scripts, conductor=conductor, max_running=2)
        await self.request(im, "Menu.", "D1")
        im.admit(ask("Teas and hours."), delegation_id="D2")
        await self.turns()
        third = im.admit(ask("And the days."), delegation_id="D3").stream
        await self.turns()
        (refused,) = json.loads(conductor.outcomes[-1])["actions"]
        self.assertEqual((refused["kind"], refused["state"]), ("Steer", "refused"))
        self.assertEqual(
            (refused["receipt"]["outcome"], refused["receipt"]["executed"]), ("capacity_unavailable", False)
        )
        result = await read_all(third)
        self.assertEqual(
            (texts(result), result[-1]["type"]), ([TEXTS.capacity_unavailable_output], "response.completed")
        )
        self.assertEqual(len(im.running), 2)
        gate.set()
        await self.turns()
        self.assertEqual((len(adapter.runs), list(im.ledger.works)), (3, ["W1", "W2"]))

    async def test_a_new_work_records_the_thread_it_continues(self):
        cases = {
            "a start": (Start(("D2",)), None),
            "a steer into the ended work": (Steer(("D2",), (WorkTarget("W1", None),), clarifications=()), "W1.1"),
            "a steer naming its ended task": (
                Steer(("D2",), (WorkTarget("W1", ("W1.1",)),), clarifications=()),
                "W1.1",
            ),
        }
        for name, (decision, continues) in cases.items():
            with self.subTest(name):
                self.traces.clear()
                im, adapter = self.scripted_session([answer()], [answer()], conductor=then(decision))
                await self.request(im, "Book a table.", "D1", timeout=2)
                await self.request(im, "And for two.", "D2", timeout=2)
                linked = (im.ledger.works["W2"].continues, adapter.runs[1].task.continues)
                self.assertEqual(linked, (continues, continues))
                dequeued = [fields["continues"] for fields in self.traced("backend.dequeue")]
                self.assertEqual(dequeued, [None, continues])

    async def test_a_follow_up_continues_the_ended_work(self):
        for confirmed in (True, False):
            with self.subTest(confirmed=confirmed):
                gate = asyncio.Event()
                conductor = then(Steer(("D2",), (WorkTarget("W1", None),), clarifications=()))
                boundary = checkpoint() if confirmed else Act(lambda run: run.task.take_boundary_input())
                adapter = ScriptedAdapter(
                    [wait(gate), boundary, answer(boundary=False)],
                    [answer()],
                    conductor=conductor,
                    delivery="not_applied",
                )
                im = self.session(adapter)
                first, second = await self.admit_each(im, "Book a table.", "For two.")
                await self.until(lambda im=im: im.ledger.delegations["D2"].work == "W1", "the steer")
                gate.set()
                _, events = await asyncio.gather(read_all(first, timeout=2), read_all(second, timeout=2))
                [steering] = im.ledger.steerings.values()
                if confirmed:
                    self.assertEqual(steering.state, "follow_up")
                    self.assertEqual(im.ledger.works["W2"].continues, "W1.1")
                    self.assertEqual(adapter.runs[1].task.continues, "W1.1")
                else:
                    self.assertEqual(steering.state, "unknown")
                    self.assertEqual(len(adapter.runs), 1)
                    self.assertEqual(events[-1]["type"], "response.failed")


class SteeringTests(WorksCase):
    async def test_not_applied_steering_becomes_a_follow_up_work(self):
        for uncertain, typed in ((False, False), (True, False), (True, True)):
            with self.subTest(uncertain=uncertain, typed=typed):
                gate = asyncio.Event()
                checkpoints = [] if uncertain else [checkpoint(), checkpoint()]
                script = [wait(gate), *checkpoints, answer("Old.", boundary=not uncertain)]
                follow_up = [app_call(call_id="call_followup")] if uncertain else [answer("New.")]
                adapter = ScriptedAdapter(script, follow_up, delivery="not_applied")
                im = self.session(adapter)
                first, second = await self.admit_each(im, "Order a latte.", "Make it large.", settings=APP)
                task = adapter.runs[0].task
                if uncertain:
                    older_input = task.take_boundary_input().steering
                    im.admit(ask("Make it small."), delegation_id="D3", settings=APP)
                    await self.until(lambda task=task: bool(task.steering), "the newer correction")
                    task.settle(task.take_boundary_input().steering, "unknown")
                    task.settle(older_input, "not_applied")
                    if typed:
                        task.work.take_typed([task], ask("Yes, the small size."))
                gate.set()
                older, newer = await read_all(first, timeout=2), await read_all(second, timeout=2)
                self.assertNotIn("Make it large.", str(adapter.runs[0].histories[-1]))
                self.assertIn("Make it large.", str(adapter.runs[1].histories[0]))
                if uncertain:
                    action = im.ledger.actions["call_followup"]
                    self.assertEqual(action.app_published, typed)
                    self.assertEqual(action.state, "started" if typed else "not_run")
                else:
                    self.assertNotIn("Old.", texts(older))
                    self.assertEqual(texts(newer), ["New."])
                    self.assertEqual(len([b for b in adapter.runs[0].boundaries if b.steering is not None]), 3)

    async def test_three_requests_hand_over_as_a_chain_to_the_newest(self):
        im, adapter = self.scripted_session([answer()], conductor=ScriptedConductor(starts))
        streams = [im.admit(ask(f"Part {n}."), delegation_id=f"D{n}").stream for n in (1, 2, 3)]
        events = [await read_all(stream) for stream in streams]
        for older, taker in pairwise(events):
            self.assertEqual(named(taker), older[0]["response"]["id"])
            self.assertHandedOver(older)
        self.assertEqual(texts(events[2]), ["Done."])
        self.assertIn("Part 1.", str(adapter.runs[0].histories[0]))
        self.assertIn("Part 3.", str(adapter.runs[0].histories[0]))

    async def test_the_records_list_an_open_question_until_the_reply_lands(self):
        landed, done = asyncio.Event(), asyncio.Event()
        asker = [asking(QUESTION), wait(landed), checkpoint(), wait(done), answer()]
        plans = (
            Start(("D1",)),
            Start(("D2",)),
            Steer(("D3",), (WorkTarget("W1", ("W1.1",)),), clarifications=()),
            Start(("D4",)),
        )
        adapter = ScriptedAdapter(asker, [answer()], [answer()], conductor=planned(*map(placing, plans)), max_running=3)
        im = self.session(adapter)
        im.admit(ask("A coffee."), delegation_id="D1")
        await self.until(lambda: self.traced("im.question.written"), "the question")
        await self.request(im, "When do you open?", "D2", timeout=2)
        im.admit(ask("A latte."), delegation_id="D3")
        await self.until(lambda: im.ledger.steerings, "the reply to wait in the task")
        landed.set()
        await self.until(lambda: im.ledger.steerings["S1"].state == "applied", "the reply to land")
        await self.request(im, "Thanks.", "D4", timeout=2)
        batch = next(given.steering for given in adapter.runs[0].boundaries if given.steering is not None)
        self.assertNotIn(QUESTION, str(adapter.runs[1].histories[0]))
        self.assertEqual(im.ledger.tasks["W1.1"].question, QUESTION)
        self.assertIn(QUESTION, str(batch.items))
        self.assertNotIn(QUESTION, str(adapter.runs[2].histories[0]))
        done.set()

    async def test_the_records_list_a_steered_request_once_under_the_work_it_joins(self):
        food, drink = asyncio.Event(), asyncio.Event()
        scripts = ([answer()], [wait(food), checkpoint(), answer()], [wait(drink), checkpoint(), answer()])
        plans = (
            Start(("D1",)),
            Start(("D2",), (TaskSpec("food"), TaskSpec("drink"))),
            Steer(("D3",), (WorkTarget("W2", None),), clarifications=()),
        )
        im, adapter = self.scripted_session(*scripts, conductor=planned(*map(placing, plans)), max_running=2)
        await self.request(im, "Book a table.", "D1", timeout=2)
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
                self.assertEqual(lines, {"W2": ["Food and a drink.", "Make it vegan."]})


class EndingsTests(IMCase):
    async def test_a_delayed_continuation_refreshes_exact_late_facts_without_replacing_its_context(self):
        for focused in (False, True):
            for previously_unclear, represented in product((False, True), repeat=2):
                with self.subTest(focused=focused, previously_unclear=previously_unclear, represented=represented):
                    sibling, continuation = asyncio.Event(), asyncio.Event()
                    call = call_item("refund_order", {"order_id": "order-42"}, "call_refund")

                    class CommittedToolCall(ScriptStep):
                        async def play(self, run, call=call):
                            for event in run.round_events([call]):
                                yield event

                    def started(run, unclear=previously_unclear, call=call):
                        run.task.tool_started(call, "orders")
                        if unclear:
                            run.task.tool_outcome(call["call_id"], "The reply is still pending.", "unclear")

                    def unrelated_attempt(run):
                        call = call_item("unrelated_lookup", {}, "call_unrelated")
                        run.task.tool_started(call, "lookup")
                        run.task.tool_outcome(call["call_id"], "The reply is still pending.", "unclear")

                    conductor = planned(
                        split("Handle the order.", "An independent task."),
                        Steer(("D2",), (WorkTarget("W1", ("W1.1",)),)),
                        Start(("D3",)),
                    )
                    conductor.model_placement = focused
                    original = [CommittedToolCall()] if represented else []
                    im, adapter = self.scripted_session(
                        [*original, Act(started), Fail()],
                        [wait(sibling), answer("Independent task done.")],
                        [wait(continuation), answer("The refund receipt is recorded.")],
                        [Act(unrelated_attempt), answer("Separate job done.")],
                        conductor=conductor,
                        max_running=3,
                    )
                    first = im.admit(
                        ask("Handle order 42."), delegation_id="D1", settings={"instructions": "Original policy."}
                    )
                    await self.until(
                        lambda adapter=adapter: len(adapter.runs) == 2 and adapter.runs[0].task.ended,
                        "the failed native-like task with its independent sibling still running",
                    )
                    predecessor = adapter.runs[0].task
                    assert (im.ledger.actions[call["call_id"]].state) == (
                        "unclear" if previously_unclear else "started"
                    )
                    second = im.admit(ask("What happened to that order?"), delegation_id="D2")
                    await self.until(lambda adapter=adapter: len(adapter.runs) == 3, "the delayed exact continuation")
                    current = adapter.runs[2].task
                    own, settings, receipts = current.view.own(), current.request(), dict(current.input_receipts)
                    await self.request(im, "An unrelated secret request.", "D3", timeout=2)
                    adapter.runs[3].task.tool_outcome("call_unrelated", "Unrelated secret result.", "done")

                    current.refresh_history()
                    pending = json.loads(current.backend_state()[0]["content"][0]["text"])["private_backend_state"]
                    self.assertTrue(all(not row["has_result"] for row in pending.get("execution_facts", [])))
                    self.assertNotIn("Refund accepted: receipt 42.", str(current.history()))
                    predecessor.tool_outcome(call["call_id"], "Refund accepted: receipt 42.", "done")
                    actual = json.loads(current.backend_state()[0]["content"][0]["text"])["private_backend_state"]
                    self.assertEqual(actual["execution_facts"][0]["output"], "Refund accepted: receipt 42.")
                    self.assertTrue(actual["execution_facts"][0]["has_result"])
                    self.assertEqual(actual["execution_facts"][0].get("late_evidence", False), previously_unclear)
                    self.assertIn("Refund accepted: receipt 42.", str(current.history()))
                    current.refresh_history()
                    history = current.history()
                    current.refresh_history()
                    self.assertEqual(current.history(), history)
                    self.assertEqual(
                        (current.view.own(), current.request(), current.input_receipts), (own, settings, receipts)
                    )
                    self.assertEqual(current.history_mode, "focused" if focused else "shared")
                    state = json.loads(current.backend_state()[0]["content"][0]["text"])["private_backend_state"]
                    if represented and not previously_unclear and focused:
                        # A refreshed ordinary output already contains the receipt. No private replacement is needed.
                        self.assertNotIn("execution_facts", state)
                        expected = []
                    else:
                        expected = [("W1.1", "call_refund", True, "Refund accepted: receipt 42.")]
                    self.assertEqual(
                        [
                            (row["task_id"], row["call_id"], row["has_result"], row["output"])
                            for row in state.get("execution_facts", [])
                        ],
                        expected,
                    )
                    self.assertNotIn("unrelated secret", str(history).lower())
                    self.assertEqual(
                        any(item.get("type") == "function_call_output" for item in im.thread_history["W1.1"]),
                        represented,
                    )
                    self.assertEqual(len(im.ledger.actions), 2)
                    with self.assertRaisesRegex(ValueError, "exact task thread"):
                        current.refresh_history(unconfirmed_calls=(next(iter(adapter.runs[3].task.steps)),))
                    current.history(extra=ask("Required pending input."))
                    self.assertEqual(current.input_receipts, receipts)
                    with self.assertRaisesRegex(RuntimeError, "context budget"):
                        current.history(extra=ask("Required pending input. " * im.settings.context_tokens))
                    sibling.set()
                    continuation.set()
                    await asyncio.gather(read(first), read(second))

    """Publication waits for source routing; lifecycle endings stay explicit on the wire."""

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

        conductor = planned(Start(("D1",)), once_let_go)
        adapter = ScriptedAdapter([wait(self.gate), answer(OLD)], *later, conductor=conductor)
        self.im = self.session(adapter, **settings)
        self.first = self.im.admit(ask("Book a table for 6 pm."), delegation_id="D1")
        await self.turns()
        self.second = self.im.admit(ask("Make it 7."), delegation_id="D2")
        await self.turns()

    async def test_an_answer_given_while_deciding_waits_and_a_correction_replaces_it(self):
        await self.booking(
            placing(Steer(("D2",), (WorkTarget("W1", None),), clarifications=())), [answer(NEW)], deciding=True
        )
        self.gate.set()
        await self.turns()
        self.assertEqual(answers([event for _, event in self.wire.events]), [])
        self.deciding.set()
        older, newer = await read(self.first), await read(self.second)
        self.assertNotIn(OLD, answers(older))
        self.assertEqual(answers(newer), [NEW])
        self.assertEqual(self.im.ledger.works["W2"].continues, "W1.1")

    async def test_ending_answered(self):
        im = self.session(ScriptedAdapter([answer()]))
        await self.request(im, "Find teas.", "D1", timeout=2)
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
        await self.booking(
            placing(CallerMessage(("D2",), "status", "It is booking the table.")), progress_note_seconds=0
        )
        self.assertEqual(answers(await read(self.second)), ["It is booking the table."])
        self.gate.set()
        self.assertEqual(answers(await read(self.first)), [OLD])
        self.assertEqual(self.timeline(), ENDINGS["answered_by_the_conductor"]["timeline"])

    async def test_ending_cancelled_for_a_newer_request(self):
        await self.booking(placing(Cancel(("D2",), (WorkTarget("W1"),))), progress_note_seconds=0)
        older, newer = await read(self.first), await read(self.second)
        self.assertEqual((answers(older), older[-1]["response"]["status"]), ([], "cancelled"))
        self.assertEqual(answers(newer), [TEXTS.stop_confirmed_line])
        self.assertNotIn(OLD, str(older))
        (stop,) = [action for action in self.im.ledger.controls.values() if action.kind == "Cancel"]
        self.assertEqual((stop.sources, stop.state), (("D2",), "applied"))


class StopTests(IMCase):
    async def test_a_new_request_waits_for_the_only_workers_unconfirmed_stop(self):
        quiescent = []

        class LateStop(ScriptedRun):
            def on_quiescent(self, callback):
                quiescent.append(callback)
                return lambda: None

        im, adapter = self.scripted_session(
            [WorkTool("cancel_work", {}), wait(asyncio.Event())],
            [answer("We open at eight.")],
            run=LateStop,
            stop_outcome="unconfirmed",
        )
        await self.request(im, "Stop this work.", "D1", timeout=2)
        await self.until(lambda: quiescent, "the unconfirmed worker's release callback")
        self.assertEqual(im.available_worker_slots(), 0)

        newer = im.admit(ask("What time do you open?"), delegation_id="D2")
        await self.until(lambda: "W2" in im.ledger.works, "the queued request")
        self.assertEqual(im.ledger.works["W2"].state, "queued")
        self.assertEqual(len(adapter.runs), 1)
        (release,) = quiescent
        release()
        self.assertEqual(texts(await read(newer)), ["We open at eight."])
        self.assertEqual(len(adapter.runs), 2)

    async def test_a_cancel_reports_the_outcome_on_its_own_request(self):
        cases = (
            ({}, {}, True),
            ({"stop_outcome": "unconfirmed"}, {}, False),
            ({"stop_delay": 1.0}, {"stop_timeout_seconds": 0.05}, False),
        )
        for options, settings, confirmed in cases:
            with self.subTest(confirmed=confirmed, **options):
                conductor = planned(
                    Start(("D1",), (TaskSpec("Menu"), TaskSpec("Hours"))), Cancel(("D2",), (), scope="all")
                )
                gate = asyncio.Event()
                adapter = ScriptedAdapter([wait(gate)], [wait(gate)], conductor=conductor, max_running=2, **options)
                im = self.session(adapter, **settings)
                first, second = await self.admit_each(im, "Check the menu and hours.", "Cancel all tasks.")
                older = await read_all(first)
                self.assertEqual((answers(older), older[-1]["response"]["status"]), ([], "cancelled"))
                stop = await read_all(second)
                reply = TEXTS.stop_confirmed_line if confirmed else TEXTS.stop_unconfirmed_line
                self.assertEqual(answers(stop), [reply])
                outcome = "stopped" if confirmed else "unconfirmed"
                self.assertEqual([task.stop.state for task in im.ledger.works["W1"].tasks], [outcome, outcome])
                self.assertEqual([run.stops for run in adapter.runs], [1, 1])

    async def test_handover_preserves_an_unconfirmed_stop_on_the_older_request(self):
        steer_boundary, publish = asyncio.Event(), asyncio.Event()
        im, adapter = self.scripted_session(
            [wait(asyncio.Event()), answer("Old work.")],
            [wait(steer_boundary), checkpoint(), wait(publish), answer("Updated work.")],
            conductor=then(
                placing(Cancel(("D2",), (WorkTarget("W1"),)), Start(("D2",))),
                Steer(("D3",), (WorkTarget("W2"),)),
            ),
            max_running=3,
            stop_outcome="unconfirmed",
        )
        im.admit(ask("Check the menu."), delegation_id="D1")
        await self.until(lambda: len(adapter.runs) == 1, "original task starts")
        former = im.admit(ask("Stop that and check opening hours."), delegation_id="D2")
        await self.until(lambda: len(adapter.runs) == 2, "new task starts")
        newer = im.admit(ask("Check weekend hours instead."), delegation_id="D3")
        await self.until(lambda: adapter.runs[1].task.steering, "steer reaches the new task")
        steer_boundary.set()
        former_events = await read(former)
        self.assertEqual(texts(former_events), [TEXTS.stop_unconfirmed_line])
        self.assertEqual(former_events[-1]["type"], "response.failed")
        publish.set()
        newer_events = await read(newer)
        self.assertEqual(texts(newer_events), ["Updated work."])
        self.assertEqual(newer_events[-1]["type"], "response.completed")

    async def test_a_stopped_tasks_held_calls_never_reach_the_app(self):
        for cancelled, sent, last in (("W1.2", [], "Menu."), ("W1.1", ["call_prices"], "Prices.")):
            with self.subTest(cancelled=cancelled):
                gate = asyncio.Event()
                menu = [app_call(call_id="call_menu"), answer("Menu.")]
                prices = [wait(gate), app_call(call_id="call_prices"), answer("Prices.")]
                conductor = planned(
                    split("menu", "prices"),
                    placing(Cancel(("D2",), (WorkTarget("W1", (cancelled,)),)), Start(("D2",))),
                )
                im = self.session(ScriptedAdapter(menu, prices, [answer("OK.")], conductor=conductor, max_running=3))
                first = await self.request(im, "Menu and prices.", "D1", settings=APP)
                gate.set()
                await self.turns()
                await self.request(im, "Skip one.", "D2")
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
        conductor = planned(split("drink", "food"), placing(Cancel(("D2",), (WorkTarget("W1", ("W1.1",)),)), then))
        scripts = ([app_call(call_id="call_drink"), answer("Drink.")], [wait(gate), answer("Food.")])
        im = self.session(ScriptedAdapter(*scripts, conductor=conductor, max_running=2), **settings)
        first = await self.request(im, "Drink and food.", "D1", settings=APP)
        second = im.admit(ask("Skip the drink."), delegation_id="D2")
        await self.turns()
        self.assertEqual([task.state for task in im.ledger.works["W1"].tasks], ["cancelled", "running"])
        gate.set()
        await self.turns()
        self.assertFalse(im.works["W1"].ended)
        return im, first, second

    async def test_an_answer_after_a_cancel_goes_out_with_the_outputs(self):
        im, first, _ = await self.cancelled_while_waiting(CallerMessage(("D2",), "status", "OK."))
        events = await asyncio.wait_for(self.answer_calls(im, first, settings=APP), 2)
        self.assertEqual((texts(events), events[-1]["type"]), (["Food."], "response.completed"))
        self.assertEqual((im.ledger.works["W1"].state, im.ledger.delegations["D1"].state), ("answered", "answered"))

    async def test_at_the_deadline_a_work_of_ended_tasks_fails_silently(self):
        im, _, _ = await self.cancelled_while_waiting(
            CallerMessage(("D2",), "status", "OK."), client_tool_timeout_seconds=None
        )
        await self.until(lambda: im.ledger.works["W1"].tasks[1].state == "answered", "the food task ending")
        written = len(self.wire.events)
        parked = im.parked["call_drink"]
        parked.work.deadline(parked)
        await self.until(lambda: not im.works, "the deadline")
        self.assertEqual((im.ledger.works["W1"].state, im.ledger.delegations["D1"].state), ("failed", "failed"))
        self.assertEqual(len(self.wire.events), written)

    async def test_the_close_ends_a_work_of_ended_tasks(self):
        im, _, _ = await self.cancelled_while_waiting(CallerMessage(("D2",), "status", "OK."))
        await asyncio.wait_for(im.close(), 2)
        self.assertEqual((im.ledger.works["W1"].state, im.works), ("cancelled", {}))

    async def test_at_the_deadline_a_taker_carries_the_held_answer(self):
        then = Steer(("D2",), (WorkTarget("W1", ("W1.2",)),), clarifications=())
        im, _, second = await self.cancelled_while_waiting(then, client_tool_timeout_seconds=0.1)
        self.assertEqual(im.works["W1"].deferred, ["D2"])
        taker = await read_all(second.stream, timeout=2)
        assert (texts(taker), taker[-1]["type"]) == ([TEXTS.stop_confirmed_line, "Food."], "response.completed")
        self.assertEqual((im.ledger.works["W1"].state, im.ledger.delegations["D1"].state), ("answered", "handed_over"))

    async def test_a_whole_stop_hands_the_work_to_the_landed_request(self):
        gate = asyncio.Event()
        steer_food, cancel_all = (
            placing(Steer(("D2",), (WorkTarget("W1", ("W1.2",)),), clarifications=())),
            placing(Cancel(("D3",), (WorkTarget("W1"),))),
        )
        scripts = (
            [app_call(call_id="call_drink"), answer("Drink.")],
            [wait(gate), checkpoint(), wait(asyncio.Event()), answer("Food.")],
        )
        conductor = planned(split("drink", "food"), steer_food, cancel_all)
        im = self.session(ScriptedAdapter(*scripts, conductor=conductor, max_running=3))
        await self.request(im, "A latte and a croissant.", "D1", settings=APP)
        second = im.admit(ask("Make the croissant warmed."), delegation_id="D2")
        await self.turns()
        gate.set()
        await self.turns()
        self.assertEqual(im.works["W1"].deferred, ["D2"])
        stop_events = await self.request(im, "Cancel everything.", "D3")
        self.assertEqual(texts(stop_events), [TEXTS.stop_confirmed_line])
        events = await read_all(second.stream, timeout=2)
        self.assertEqual((texts(events), events[-1]["type"]), ([], "response.incomplete"))
        assert (events[-1]["response"]["status"], im.ledger.delegations["D2"].state) == ("cancelled", "cancelled")
        works = {work_id: (work.state, work.requests) for work_id, work in im.ledger.works.items()}
        self.assertEqual(works, {"W1": ("cancelled", ["D1", "D2"])})
        _, conversation = im.conversation.window()
        callers = [item for item in conversation if item.get("role") == "user" and item.get("id") == "D2"]
        self.assertEqual(without_ids(callers), ask("Make the croissant warmed."))

    async def test_a_landed_request_takes_over_when_the_app_never_answers(self):
        may_answer = asyncio.Event()
        scripts = ([app_call(call_id="call_menu"), answer("Menu.")], [wait(may_answer), answer("Open Sundays.")])
        conductor = planned(split("menu", "hours"), Steer(("D2",), (WorkTarget("W1", ("W1.2",)),), clarifications=()))
        adapter = ScriptedAdapter(*scripts, conductor=conductor, max_running=2)
        im = self.session(adapter, client_tool_timeout_seconds=0.1)
        first = await self.request(im, "Menu and hours?", "D1", settings=APP)
        second = im.admit(ask("Open on Sunday?"), delegation_id="D2", settings=APP)
        await self.turns()
        may_answer.set()
        await self.turns()
        taker = await read_all(second.stream, timeout=2)
        self.assertEqual((named(taker), texts(first)), (None, []))
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
        gate, stopping, settle_stop = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class HeldStop(ScriptedRun):
            async def stop(self):
                stopping.set()
                await settle_stop.wait()
                return await super().stop()

        script, conductor = (
            [wait(gate), answer("Too late.")],
            then(Cancel(("D2",), (WorkTarget("W1"),))),
        )
        adapter = ScriptedAdapter(script, conductor=conductor, run=HeldStop)
        im = self.session(adapter)
        first, stop = await self.admit_each(im, "Book it.", "Stop.")
        await asyncio.wait_for(stopping.wait(), 2)
        detaching = asyncio.create_task(im.set_attachment(False))
        try:
            await self.until(lambda: not im.caller_attached, "the caller detaches")
            gate.set()
            await self.turns()
        finally:
            settle_stop.set()
        await asyncio.wait_for(detaching, 2)
        self.assertEqual(texts(await read_all(first)), [])
        self.assertEqual(texts(await read_all(stop, timeout=2)), [TEXTS.stop_confirmed_line])


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

    async def test_shared_and_kept_history_excludes_the_exact_superseded_final(self):
        for kept in (False, True):
            with self.subTest(kept=kept):
                await self.check_superseded_history(kept)

    async def check_superseded_history(self, kept):
        may_answer, deciding, may_decide = (asyncio.Event() for _ in range(3))
        old, new = "A six o'clock slot is available.", "A seven o'clock slot is available."
        caller_words, brief = "Check availability at six.", "Check availability for the evening."
        initial_history = []

        async def route(given, place):
            source = given.arrivals[0].delegation_id
            if source == "D1":
                return await place(CompleteDecision((Start((source,), (TaskSpec(brief),)),)))
            deciding.set()
            await may_decide.wait()
            return await place(CompleteDecision((Steer((source,), (WorkTarget("W1", ("W1.1",)),)),)))

        def refresh(run):
            initial_history.extend(run.task.history())
            run.task.refresh_history()

        adapter = ScriptedAdapter(
            [*([MARK] if kept else []), wait(may_answer), answer(old)],
            [Act(refresh), answer(new)],
            max_running=2,
            conductor=ScriptedConductor(route),
        )
        im = self.session(adapter, continuable=True)
        first = im.admit(ask(caller_words), delegation_id="D1")
        if kept:
            await self.marked(im)
        else:
            await self.until(lambda: bool(adapter.runs), "the original worker")
        later = im.admit(ask("Make it seven instead."), delegation_id="D2")
        await asyncio.wait_for(deciding.wait(), 2)
        may_answer.set()
        await self.until(lambda: im.ledger.tasks["W1.1"].ended, "the held answer to the old request")
        self.assertEqual(im.ledger.tasks["W1.1"].answer_publication, "pending")
        may_decide.set()
        self.assertEqual(answers(await read(first)), [MESSAGE] if kept else [])
        self.assertEqual(answers(await read(later)), [new])
        stale = im.ledger.tasks["W1.1"]
        self.assertEqual((stale.answer_publication, stale.answer), ("superseded", old))
        self.assertTrue(initial_history)
        self.assertNotIn(stale.answer_item_id, {item.get("id") for item in initial_history})
        self.assertNotIn(stale.answer_item_id, {item.get("id") for item in adapter.runs[1].histories[0]})
        for perspective in (adapter.runs[0].histories[0], initial_history):
            assignment = next(item for item in perspective if isinstance(item, WorkerData) and brief in str(item))
            context = json.loads(assignment["content"][0]["text"])["conductor_routing_context"]
            self.assertEqual(context["task_input"], TEXTS.task_line.format(task=brief))
            self.assertFalse(assignment.starts_request)
            self.assertIn(caller_words, str(perspective))
        self.assertIn(caller_words, str(im.history))
        self.assertNotIn(brief, str(im.history))
        history = [message_text(item) for item in im.history if item.get("role") == "assistant"]
        self.assertNotIn(old, history)
        self.assertIn(new, history)

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
        first = await self.request(im, "Read the menu and start the report.", "D1", timeout=2, settings=APP)
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
        self.assertEqual(answers(await read_all(second, timeout=2)), ["B."])

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

    async def hung_up(self) -> tuple:
        """Hang up while W1, which asked a question and was marked, runs with D3 steered in and not landed.

        Meanwhile W2, unmarked, runs and D4 waits for a decision. Return the session, its adapter, W1's gate and the
        four requests' streams.
        """
        job = asyncio.Event()
        script = [asking(QUESTION), MARK, wait(job), checkpoint(), answer(REPORT)]
        plans = (
            placing(Start(("D1",))),
            placing(Start(("D2",))),
            placing(Steer(("D3",), (WorkTarget("W1", None),), clarifications=())),
            undecided,
        )
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
        ended = [(await read_all(stream, timeout=2))[-1]["response"] for stream in streams]
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
        conductor = planned(split("a", "b"), Steer(("D2",), (WorkTarget("W1", ("W1.2",)),), clarifications=()))
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
        conductor = planned(
            *map(
                placing,
                (
                    Start(("D1",)),
                    Steer(("D2",), (WorkTarget("W1", None),), clarifications=()),
                    Steer(("D3",), (WorkTarget("W1", None),), clarifications=()),
                ),
            )
        )
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
        self.assertEqual(self.traced("im.task.follow_up"), [{"task_id": "W1.1", "delegation_ids": ["D3"]}])
        self.assertEqual(list(im.ledger.works), ["W1", "W2"])
        self.assertEqual((im.ledger.works["W2"].carrier, im.ledger.works["W2"].continues), ("D3", "W1.1"))

    async def test_attach_waits_for_the_hang_up_and_serves_the_next_call_on_a_new_backend(self):
        other = asyncio.Event()
        plans = (placing(Start(("D1",))), placing(Start(("D2",))), undecided)
        scripts = ([MARK, answer(REPORT)], [wait(other), answer()])
        first = ScriptedAdapter(*scripts, conductor=planned(*plans), max_running=3)
        later = ScriptedConductor(starts)
        second = ScriptedAdapter([answer("Yes, it is ready.")], conductor=later, max_running=3)
        im = self.session(first, continuable=True, renew=lambda: second, close_grace_seconds=5)
        await self.request(im, "Compare every supplier.", "D1", timeout=2)
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
        events = await self.request(im, "Is my research ready?", "D5", timeout=2)
        self.assertEqual(answers(events), ["Yes, it is ready."])
        (given,) = later.given
        self.assertEqual(([arrival.delegation_id for arrival in given.arrivals], given.new_from), (["D5"], 0))
        self.assertIn("Compare every supplier.", str(given.conversation))
        with self.assertRaisesRegex(DelegationError, "another call"):
            await self.call_again(im)

    async def test_a_later_calls_request_takes_the_job_over_at_once(self):
        gate = asyncio.Event()
        conductor = planned(Start(("D1",)), Steer(("D2",), (WorkTarget("W1", None),), clarifications=()))
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
        *history, said_last = im.history
        requests = [item for item in history if item.get("role") == "user"]
        self.assertEqual(without_ids(requests), [*ask("Write the long report."), *ask("Make it shorter.")])
        self.assertEqual((said_last["role"], message_text(said_last)), ("assistant", "Shorter."))

    async def test_a_request_whose_call_ended_never_takes_the_work_from_a_later_call(self):
        part_a, part_b, europe = asyncio.Event(), asyncio.Event(), asyncio.Event()
        scripts = (
            [MARK, wait(part_a), checkpoint(), answer("Part A, shorter.")],
            [wait(part_b), checkpoint(), wait(europe), answer("Part B, for Europe.")],
        )
        steers = (
            placing(Steer(("D2",), (WorkTarget("W1", ("W1.1",)),), clarifications=())),
            placing(Steer(("D3",), (WorkTarget("W1", ("W1.2",)),), clarifications=())),
        )
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
            await self.request(im, text, f"D{number}", timeout=2)
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
                await self.request(im, "Compare every supplier.", "D1", timeout=2)
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
        conductor = planned(Start(("D1",)), Steer(("D2",), (WorkTarget("W1", None),), clarifications=()))
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
            return await read_all(stream, timeout=2)

        im = await asyncio.create_task(first_call())
        self.assertEqual(answers(await asyncio.create_task(second_call(im))), ["Shorter."])
        self.assertEqual(adapter.calls, {"W1.1": "call 1", "W2.1": "call 2"})
