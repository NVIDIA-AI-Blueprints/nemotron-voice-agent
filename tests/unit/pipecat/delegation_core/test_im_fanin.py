# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

"""A canonical caller reply can atomically join independent live Work branches."""

import asyncio
import json
from unittest.mock import patch

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import (
    CallerLine,
    CallerMessage,
    Cancel,
    DecisionRejected,
    Start,
    Steer,
    TaskSpec,
    TaskUpdate,
    WorkTarget,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.input import InputScope
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.session import DelegationError
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.stop import AtBackstop
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.items import WorkerData
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.work_provenance import decode_provenance
from tests.unit.pipecat.delegation_core.helpers import (
    MARK,
    TEXTS,
    IMCase,
    OwnCall,
    ScriptedAdapter,
    ScriptedRun,
    answer,
    app_call,
    app_outputs,
    ask,
    checkpoint,
    function_calls,
    placing,
    planned,
    read_all,
    steer_action,
    texts,
    wait,
    wire_decision,
)
from tests.unit.pipecat.delegation_core.test_im_calls import MENU_APP


def multi_steer():
    """Decode one complete decision that steers both live branches atomically."""
    return wire_decision(
        steer_action(
            "D3",
            {"work_id": "W1"},
            {"work_id": "W2"},
            updates={"W1.1": "Apply the first update.", "W2.1": "Apply the second update."},
        )
    )


def _start_then_steer(*later):
    return planned(Start(("D1",)), Start(("D2",)), multi_steer(), *later)


class FanInTests(IMCase):
    async def start_and_update_two_workers(self, adapter, *, settings=None, **options):
        im = self.session(adapter, **options)
        im.admit(ask("Do A."), delegation_id="D1", settings=settings)
        await self.until(lambda: len(adapter.runs) == 1, "A started")
        im.admit(ask("Do B."), delegation_id="D2", settings=settings)
        await self.until(lambda: len(adapter.runs) == 2, "B started")
        response = im.admit(ask("Update both."), delegation_id="D3")
        await self.until(lambda: len(im.ledger.steerings) == 2, "both steering targets placed")
        return im, response

    async def test_model_assigns_only_scoped_details_from_a_mixed_turn_to_each_task(self):
        original = "Find a flight under  $500 and book dinner for four, both in Boston."
        correction = "Make the flight Sat urday and dinner for six."
        both_running = asyncio.Event()

        class Conductor:
            model_placement = True

            async def decide(_, given, place):
                source = given.arrivals[0].delegation_id
                if source == "D1":
                    line = {"source_message_id": source, "sequence": 0}
                    self.assertEqual(given.arrivals[0].caller_lines, ({**line, "text": original},))
                    tasks = [
                        {
                            "brief": "Find a flight to Boston under $500.",
                            "caller_lines": [{**line, "excerpt": "under $500"}],
                        },
                        {"brief": "Book dinner for four in Boston."},
                    ]
                    if given.attempt == 1:
                        tasks[0]["caller_lines"] = [line]
                    else:
                        self.assertIn("whole caller line or source", given.decision_rejection)
                        self.assertFalse(im.ledger.works)
                        self.assertFalse(im.ledger.controls)
                    action = {"action": "start", "delegations": ["D1"], "tasks": tasks}
                else:
                    action = {
                        "action": "steer",
                        "delegations": ["D2"],
                        "targets": [{"work_id": "W1"}],
                        "updates": [
                            {
                                "task_id": "W1.1",
                                "text": "Find the flight for Saturday.",
                                "caller_lines": [
                                    {"source_message_id": source, "sequence": 5, "excerpt": "flight Saturday"}
                                ],
                            },
                            {
                                "task_id": "W1.1",
                                "text": "The caller also replied to the flight question.",
                                "caller_lines": [{"source_message_id": source, "sequence": 4}],
                            },
                            {"task_id": "W1.2", "text": "Change the dinner party size to six."},
                        ],
                    }
                    if given.attempt == 2:
                        self.assertIn("applicable caller lines", given.decision_rejection)
                        self.assertFalse(im.ledger.steerings)
                        action["updates"][2]["caller_lines"] = [
                            {"source_message_id": source, "sequence": 5, "excerpt": "dinner for six"}
                        ]
                await place(wire_decision(action))

        im, adapter = self.scripted_session(
            [wait(both_running), checkpoint(), answer("Flight checked.")],
            [wait(both_running), checkpoint(), answer("Dinner checked.")],
            conductor=Conductor(),
            max_running=2,
        )
        first = im.admit(ask(original), delegation_id="D1")
        await self.until(lambda: len(adapter.runs) == 2, "both scoped tasks")
        flight_item = {"task_ref": "W1.1", "item_id": "flight_question", "branch_ref": "shared_branch"}
        dinner_item = {"task_ref": "W1.2", "item_id": "dinner_question", "branch_ref": "shared_branch"}
        context = {
            "caller_turns": [(3, "yes yes yes"), (4, "Yes."), (5, correction)],
            "playback": [
                {
                    "sequence": 0,
                    "results": [flight_item],
                    "playback_state": "cut",
                    "recorded_sentence_prefix": "Friday was available.",
                },
                {
                    "sequence": 1,
                    "results": [dinner_item],
                    "playback_state": "complete",
                    "recorded_sentence_prefix": "Your table for four is reserved.",
                },
                {
                    "sequence": 2,
                    "results": [flight_item, dinner_item],
                    "playback_state": "cut",
                    "recorded_sentence_prefix": "Both were ready.",
                },
            ],
        }
        transcript = "\n".join(
            [
                *(receipt["recorded_sentence_prefix"] for receipt in context["playback"]),
                *(text for _, text in context["caller_turns"]),
            ]
        )
        second = im.admit(ask(transcript), delegation_id="D2", caller_context=context)
        with self.assertRaisesRegex(DecisionRejected, "selected caller line"):
            InputScope.bind("Find a flight", im.request_items("D2"), {"D2": context}, selections=(CallerLine("D2", 0),))
        with self.assertRaisesRegex(DecisionRejected, "sources bound"):
            InputScope.bind("Find a flight", im.request_items("D2"), {"D2": context}, selections=(CallerLine("D1", 0),))
        with self.assertRaisesRegex(DecisionRejected, "needs a selection"):
            InputScope.bind(
                "Confirm the caller checked in",
                im.request_items("D2"),
                {"D2": context},
                require_lines=True,
            )
        exclusive = InputScope.bind(
            "Confirm the caller checked in", im.request_items("D2"), {"D2": context}, include_bound_lines=True
        )
        note = json.loads(exclusive.item["content"][0]["text"])["conductor_routing_context"]
        self.assertEqual(
            note["source_excerpts"],
            [
                {"source_message_id": "D2", "sequence": sequence, "text": text}
                for sequence, text in context["caller_turns"]
            ],
        )
        self.assertNotIn("Friday was available.", str(exclusive.item))
        await self.until(lambda: len(im.ledger.steerings) == 2, "both scoped updates")
        both_running.set()
        streams = await asyncio.gather(read_all(first.stream), read_all(second.stream))
        flight_input = next(item for item in adapter.runs[0].histories[0] if isinstance(item, WorkerData))
        flight_note = json.loads(flight_input["content"][0]["text"])["conductor_routing_context"]
        self.assertEqual(flight_note["source_excerpts"], [{"source_message_id": "D1", "text": "under $500"}])
        flight, dinner = [json.dumps(run.histories) for run in adapter.runs]
        for history, positive, excluded in (
            (
                flight,
                (
                    "Boston under $500",
                    "flight for Saturday",
                    "flight Saturday",
                    "Friday was available.",
                    "replied to the flight question",
                    "Yes.",
                ),
                ("dinner for four", "size to six", "Your table for four is reserved."),
            ),
            (
                dinner,
                ("dinner for four in Boston", "size to six", "Your table for four is reserved."),
                ("under $500", "Saturday", "Friday was available.", "Yes."),
            ),
        ):
            for text in positive:
                self.assertIn(text, history)
            for text in (*excluded, original, correction, "both in Boston", "Both were ready."):
                self.assertNotIn(text, history)
        publications = {
            decode_provenance(event).item_attribution.task_ref
            for events in streams
            for event in events
            if event["type"] == "response.output_item.done" and event["item"]["type"] == "message"
        }
        self.assertEqual(publications, {"W1.1", "W1.2"})
        for run in adapter.runs:
            scopes = [item for history in run.histories for item in history if isinstance(item, WorkerData)]
            self.assertTrue(scopes)
            self.assertTrue(all(item["role"] == "user" for item in scopes))
        for task in im.ledger.tasks.values():
            assert [receipt.source_message_ids for receipt in task.input_receipts.values()] == [("D1",), ("D2",)]
        shared = str(im.conversation.window())
        self.assertIn(original, shared)
        self.assertIn(correction, shared)
        self.assertNotIn("conductor_routing_context", shared)

    async def test_one_worker_gets_only_explicit_caller_excerpts(self):
        gate = asyncio.Event()

        async def start(given, place):
            def decision(source):
                spec = TaskSpec("Check the price", caller_lines=(CallerLine(source, 0, "coffee prices"),))
                return Start(("D1",), (spec,))

            with self.assertRaisesRegex(DecisionRejected, "source"):
                await place(decision("unknown"))
            self.assertEqual(im.arrivals, ["D1"])
            await place(decision("D1"))

        async def steer(given, place):
            def decision(source):
                update = TaskUpdate("W1.1", "Change to decaf", caller_lines=(CallerLine(source, 0, "Make it decaf"),))
                return Steer(("D2",), (WorkTarget("W1"),), updates=(update,))

            with self.assertRaisesRegex(DecisionRejected, "source"):
                await place(decision("D1"))
            self.assertEqual(im.arrivals, ["D2"])
            await place(decision("D2"))

        im, adapter = self.scripted_session(
            [wait(gate), checkpoint(), answer("Decaf is available.")], conductor=planned(start, steer)
        )
        first = im.admit(ask("Check coffee prices. Unrelated detail A."), delegation_id="D1")
        await self.until(lambda: adapter.runs, "the scoped worker")
        second = im.admit(ask("Make it decaf. Unrelated detail B."), delegation_id="D2")
        await self.until(lambda: im.ledger.steerings, "the scoped update")
        gate.set()
        await read_all(first.stream)
        self.assertEqual(texts(await read_all(second.stream)), ["Decaf is available."])
        history = json.dumps(adapter.runs[0].histories)
        self.assertIn("coffee prices", history)
        self.assertIn("Make it decaf", history)
        self.assertNotIn("Unrelated detail", history)

    async def test_scoped_attachment_repairs_to_a_single_task_with_the_complete_input(self):
        message = ask("Read this label.")[0]
        image = {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
        message["content"].append(image)

        async def split(given, place):
            await place(
                wire_decision(
                    {"action": "start", "delegations": ["D1"], "tasks": [{"brief": "Read"}, {"brief": "Check"}]}
                )
            )

        async def repair(given, place):
            self.assertIn("Route the complete message to one eligible task", given.decision_rejection)
            self.assertFalse(im.ledger.works or im.ledger.controls)
            await place(Start(("D1",), (TaskSpec("Read the label"),)))

        im, adapter = self.scripted_session([answer("Read.")], conductor=planned(split, repair), max_running=3)
        response = im.admit([message], delegation_id="D1")
        self.assertEqual(texts(await read_all(response.stream)), ["Read."])
        self.assertEqual(len(adapter.runs), 1)
        self.assertTrue(any(image in item.get("content", ()) for item in adapter.runs[0].histories[0]))

    async def test_bundled_start_repairs_into_separately_scoped_tasks_before_admission(self):
        class Conductor:
            model_placement = True

            async def decide(_, given, place):
                sources = tuple(arrival.delegation_id for arrival in given.arrivals)
                self.assertEqual(sources, ("D1", "D2"))
                if given.attempt == 1:
                    await place(Start(sources, (TaskSpec("flight"), TaskSpec("dinner"))))
                    self.fail("A bundled Start with independent sources was accepted")
                self.assertIn("one start action per task", given.decision_rejection)
                self.assertFalse(im.ledger.works)
                self.assertFalse(im.ledger.controls)
                self.assertFalse(adapter.runs)
                await placing(Start(("D1",), (TaskSpec("flight"),)), Start(("D2",), (TaskSpec("dinner"),)))(
                    given, place
                )

        im, adapter = self.scripted_session(
            [answer("Flight checked.")], [answer("Dinner checked.")], conductor=Conductor(), max_running=3
        )
        flight = im.admit(ask("FLIGHT-SOURCE"), delegation_id="D1")
        dinner = im.admit(ask("DINNER-SOURCE"), delegation_id="D2")
        self.assertEqual(
            [texts(events) for events in await asyncio.gather(read_all(flight.stream), read_all(dinner.stream))],
            [["Flight checked."], ["Dinner checked."]],
        )
        for run, own, other in zip(
            adapter.runs, ("FLIGHT-SOURCE", "DINNER-SOURCE"), ("DINNER-SOURCE", "FLIGHT-SOURCE"), strict=True
        ):
            caller_words = [item["content"][0]["text"] for item in run.histories[0] if item.get("role") == "user"]
            self.assertIn(own, caller_words)
            self.assertNotIn(other, caller_words)

    async def test_atomic_budget_charges_retained_clarification_before_any_placement_or_consumption(self):
        gate = asyncio.Event()
        outcomes = []

        class Conductor:
            async def decide(_, given, place):
                for arrival in given.arrivals:
                    source = arrival.delegation_id
                    if source in ("D1", "D2"):
                        await place(Start((source,)))
                    elif source == "D3":
                        await place(
                            CallerMessage((source,), "question", "Which of the two live tasks?", clarifications=())
                        )
                    else:
                        clarification = next(iter(im.clarifications))
                        before = len(im.ownership.edges), len(im.ledger.steerings), tuple(im.arrivals)
                        decision = wire_decision(
                            steer_action(
                                source,
                                {"work_id": "W1"},
                                {"work_id": "W2"},
                                updates={
                                    "W1.1": "Apply the scoped update to A.",
                                    "W2.1": "Apply the scoped update to B.",
                                },
                                clarifications=[clarification],
                            )
                        )
                        try:
                            result = await place(decision)
                        except DecisionRejected as error:
                            after = len(im.ownership.edges), len(im.ledger.steerings), tuple(im.arrivals)
                            outcomes.append((str(error), before, after))
                            raise
                        else:
                            after = len(im.ownership.edges), len(im.ledger.steerings), tuple(im.arrivals)
                            outcomes.append((result, before, after))

        adapter = ScriptedAdapter(
            [wait(gate), answer("A.")], [wait(gate), answer("B.")], conductor=Conductor(), max_running=3
        )
        im = self.session(adapter, context_tokens=5000)
        im.admit(ask("A."), delegation_id="D1")
        await self.until(lambda: len(adapter.runs) == 1, "A")
        im.admit(ask("B."), delegation_id="D2")
        await self.until(lambda: len(adapter.runs) == 2, "B")
        await self.request(im, "Retain this original input: " + "x" * 12000, "D3")
        self.assertEqual(next(iter(im.clarifications.values()))["state"], "waiting")
        rejected_events = await self.request(im, "Both tasks need this additional context: " + "y" * 12000, "D4")
        self.assertEqual(rejected_events[-1]["type"], "response.completed")
        self.assertIn(im.texts.routing_rejected_output, texts(rejected_events))
        self.assertTrue(outcomes)
        for result, before, after in outcomes:
            self.assertIn("context budget", result)
            self.assertEqual(before, after, "refused atomic placement cannot take, queue or consume any part")
        self.assertEqual(next(iter(im.clarifications.values()))["state"], "waiting")
        self.assertFalse(im.ledger.delegations["D4"].placements)
        self.assertFalse(im.ledger.delegations["D4"].work)
        accepted = im.admit(ask("Apply it to both."), delegation_id="D5")
        await self.until(lambda im=im: len(im.ledger.steerings) == 2, "smaller combined input fits unchanged budget")
        self.assertEqual(next(iter(im.clarifications.values()))["state"], "consumed")
        self.assertEqual(len(im.ledger.delegations["D5"].placements), 2)
        self.assertEqual(len(self.traced("im.clarification.consumed")), 1)
        gate.set()
        self.assertCountEqual(texts(await read_all(accepted.stream)), ["A.", "B."])

    def assert_owner_revisions_consistent(self, events):
        snapshots = {}
        for event in events:
            proof = decode_provenance(event)
            if proof is None:
                continue
            key = (proof.owner_ref, proof.owner_revision)
            facts = proof.bindings, proof.obligations, proof.owner_settlements, proof.request_sources
            self.assertEqual(snapshots.setdefault(key, facts), facts, key)

    @staticmethod
    def overlapping_conductor():
        shared = wire_decision(steer_action("D3", {"work_id": "W1", "task_ids": ["W1.2"]}, {"work_id": "W2"}))
        return planned(
            Start(("D1",), (TaskSpec("A one"), TaskSpec("A two"))),
            Start(("D2",)),
            shared,
            Steer(("D4",), (WorkTarget("W1", ("W1.1",)),), clarifications=()),
        )

    async def overlapping_requests(self, adapter):
        im = self.session(adapter)
        im.admit(ask("A initial."), delegation_id="D1")
        await self.until(lambda: len(adapter.runs) == 2, "A tasks started")
        im.admit(ask("B initial."), delegation_id="D2")
        await self.until(lambda: len(adapter.runs) == 3, "B task started")
        shared = im.admit(ask("Correct A two and B."), delegation_id="D3")
        await self.until(lambda: len(im.ledger.steerings) == 2, "shared placement")
        newer = im.admit(ask("Newer correction for A one."), delegation_id="D4")
        await self.until(lambda: len(im.ledger.steerings) == 3, "newer placement")
        return im, shared, newer

    async def test_unconfirmed_backstop_does_not_execute_an_older_pending_input_in_an_overlapping_worker(self):
        a_new, a_done, a_old, b, b_done = [asyncio.Event() for _ in range(5)]
        adapter = ScriptedAdapter(
            [wait(a_new), checkpoint(), wait(a_done), answer("Newer first-task answer.")],
            [wait(a_old), answer("Old second-task answer.", boundary=False)],
            [wait(b), checkpoint(), wait(b_done), answer("B done.")],
            [OwnCall("actual_followup_effect"), answer("Older pending input handled.")],
            conductor=self.overlapping_conductor(),
            max_running=3,
            stop_outcome="unconfirmed",
        )
        im, shared, newer = await self.overlapping_requests(adapter)
        a_new.set()
        b.set()
        await self.until(
            lambda im=im: im.works["W1"].carrier == "D4" and im.works["W2"].carrier == "D3", "real inputs applied"
        )
        b_done.set()
        await self.until(lambda im=im: im.ledger.tasks["W2.1"].ended, "B ended before old input applies")
        self.assertFalse(im.ledger.delegations["D3"].ended, "the unapplied source still owes a real disposition")
        im.works["W1"].stop_soon(AtBackstop())
        a_events, b_events = await asyncio.wait_for(asyncio.gather(read_all(newer.stream), read_all(shared.stream)), 3)
        self.assertEqual(len(adapter.runs), 3, "an uncertain old worker must not acquire an executing successor")
        self.assertEqual(a_events[-1]["type"], "response.failed")
        self.assertIn("unconfirmed", {row.stop_outcome for row in decode_provenance(a_events[-1]).obligations})
        self.assertEqual(texts(b_events), ["B done."])
        self.assertFalse(any(action.name == "actual_followup_effect" for action in im.ledger.actions.values()))
        self.assertEqual(im.admitted, 4, "refusing unsafe execution must not fabricate another caller source")
        self.assert_owner_revisions_consistent([*a_events, *b_events])

    async def test_ending_a_call_releases_a_cancelled_shared_owner_after_its_held_result_is_discarded(self):
        a, b, kept = asyncio.Event(), asyncio.Event(), asyncio.Event()
        count = 36
        adapter = ScriptedAdapter(
            [wait(a), app_call(call_id="call_A"), answer("A.")],
            [wait(b), answer("B unheard final.")],
            [MARK, wait(kept), answer("Separate kept work.")],
            *([answer("Independent.")] for _ in range(count)),
            conductor=planned(
                Start(("D1",)),
                Start(("D2",)),
                multi_steer(),
                Start(("D4",)),
                *(placing(Start((f"D{index + 5}",))) for index in range(count)),
            ),
            max_running=3,
        )
        im, shared = await self.start_and_update_two_workers(adapter, settings=MENU_APP, continuable=True)
        a.set()
        await read_all(shared.stream)
        b.set()
        await self.until(
            lambda im=im: im.ledger.tasks["W2.1"].ended and bool(im.ownership.members("D3")[1].held), "B held"
        )
        im.admit(ask("Keep this separate work."), delegation_id="D4")
        await self.until(
            lambda im=im: im.ledger.works.get("W3") and im.ledger.works["W3"].after_call, "separate kept job"
        )
        self.assertTrue(await self.hang_up(im))
        self.assertEqual(im.ledger.delegations["D3"].state, "cancelled")
        self.assertEqual(im.ownership.owner("D3").disposition, "cancelled")
        await self.call_again(im)
        for index in range(count):
            await self.request(im, "Independent.", f"D{index + 5}")
        self.assertNotIn("W2", im.ledger.works, "discarded publication cannot pin an ended Work past retention")
        self.assertNotIn("W2", im.ownership.protected_works())
        self.assertFalse(any("B unheard final." in texts([event]) for _, event in self.wire.events))

    async def test_ending_a_call_settles_an_answered_shared_owner_with_an_outstanding_tool(self):
        a, b = asyncio.Event(), asyncio.Event()
        adapter = ScriptedAdapter(
            [wait(a), app_call(call_id="call_A"), answer("A.")],
            [wait(b), answer("B.")],
            conductor=_start_then_steer(),
            max_running=2,
        )
        im, shared = await self.start_and_update_two_workers(adapter, settings=MENU_APP)
        b.set()
        await self.until(lambda: im.ledger.tasks["W2.1"].ended, "B answered")
        a.set()
        events = await read_all(shared.stream)
        self.assertEqual(texts(events), ["B."])
        self.assertEqual(function_calls(events), ["call_A"])
        self.assertEqual(im.ledger.delegations["D3"].state, "waiting_client")
        self.assertFalse(await self.hang_up(im))
        self.assertEqual(im.ledger.delegations["D3"].state, "answered")
        self.assertEqual(im.ledger.tasks["W1.1"].state, "cancelled")

    async def case_with_held_app_call(self, *plans, sibling_result="B.", extra=(), capacity=3):
        later = asyncio.Event()
        im, adapter = self.scripted_session(
            [app_call(call_id="call_first"), answer("A.")],
            [wait(later), app_call(call_id="call_held"), answer(sibling_result)],
            *extra,
            conductor=planned(Start(("D1",), (TaskSpec("A"), TaskSpec("B"))), *plans),
            max_running=capacity,
        )
        first = await self.request(im, "Both menus.", "D1", timeout=3, settings=MENU_APP)
        self.assertEqual(function_calls(first), ["call_first"])
        later.set()
        await self.until(lambda: bool(im.works["W1"].held), "sibling app call held")
        return im, adapter.runs[1].task, first

    async def test_held_app_global_wait_uses_original_call_once_after_foreign_placement(self):
        deciding, release = asyncio.Event(), asyncio.Event()

        async def foreign(given, place):
            deciding.set()
            await release.wait()
            await place(Start(("D2",)))

        im, task, first = await self.case_with_held_app_call(foreign, extra=([answer("Foreign C.")],))
        premature = {"type": "function_call_output", "call_id": "call_held", "output": "{}"}
        with self.assertRaisesRegex(DelegationError, "published application call"):
            im.admit_client_request([premature])
        with self.assertRaisesRegex(DelegationError, "published application call"):
            im.reconcile_tool_outputs([premature])
        self.assertEqual(im.ledger.actions["call_held"].state, "started")
        im.admit(ask("Independent C."), delegation_id="D2")
        reading = None
        try:
            await self.until(deciding.is_set, "the conductor to start placing the independent request", timeout=2)
            response = im.admit_client_request(app_outputs(first), settings=MENU_APP)
            reading = asyncio.create_task(read_all(response.stream))
            await self.until(lambda im=im: bool(im.ownership._placement_waits), "actual pre-dispatch GLOBAL wait")
            self.assertNotIn("call_held", im.parked)
            self.assertEqual(task.routes["call_held"], "app")
            release.set()
            events = await asyncio.wait_for(reading, 3)
        finally:
            release.set()
            if reading is not None:
                reading.cancel()
                await asyncio.gather(reading, return_exceptions=True)
        self.assertEqual(function_calls(events), ["call_held"])
        self.assertNotIn("call_held", task.refusals)
        await read_all(im.admit_client_request(app_outputs(events), settings=MENU_APP).stream)
        self.assertEqual(im.ledger.actions["call_held"].state, "done")

    async def test_held_app_rechecks_routing_and_tool_restrictions_before_publication(self):
        restricted = {**MENU_APP, "tool_choice": {"type": "allowed_tools", "tools": []}}
        for restriction in ("unlisted", "routing_unresolved"):
            with self.subTest(reason=restriction):
                rejected = restriction == "routing_unresolved"
                plans = (Start(("unbound",)), Start(("unbound",))) if rejected else ()
                im, task, first = await self.case_with_held_app_call(*plans, sibling_result="B replanned.")
                if rejected:
                    await self.request(im, "Wait before sending the second request.", "D2", timeout=2)
                events = await read_all(
                    im.admit_client_request(app_outputs(first), settings=MENU_APP if rejected else restricted).stream
                )
                self.assertFalse(function_calls(events))
                action = im.ledger.actions["call_held"]
                self.assertEqual((action.state, action.app_published), ("not_run", False))
                self.assertEqual(task.refusal_details("call_held").reason, restriction)
                self.assertEqual(im.ledger.actions["call_first"].state, "done")

    async def test_stop_records_unpublished_app_call_while_a_sibling_still_waits(self):
        stop = placing(Cancel(("D2",), (WorkTarget("W1", ("W1.2",)),)))
        im, task, first = await self.case_with_held_app_call(stop, capacity=2)
        await self.request(im, "Stop the second lookup.", "D2", timeout=2)
        action = im.ledger.actions["call_held"]
        self.assertEqual((action.state, action.app_published), ("not_run", False))
        self.assertEqual(task.outputs["call_held"]["output"], TEXTS.stopped_output)
        self.assertIn("call_first", im.parked)
        with self.assertRaisesRegex(DelegationError, "published application call"):
            im.admit_client_request([{"type": "function_call_output", "call_id": "call_held", "output": "{}"}])
        self.assertEqual(texts(await self.answer_calls(im, first, settings=MENU_APP)), ["A."])

    async def test_stopping_unlanded_primary_retires_only_its_branch(self):
        a, b, b_done = asyncio.Event(), asyncio.Event(), asyncio.Event()
        adapter = ScriptedAdapter(
            [wait(a), answer("A must not restart.")],
            [wait(b), checkpoint(), wait(b_done), answer("B survives.")],
            conductor=planned(
                Start(("D1",)),
                Start(("D2",)),
                multi_steer(),
                Cancel(("D4",), (WorkTarget("W1", None),), clarifications=()),
            ),
            max_running=3,
        )
        im, shared = await self.start_and_update_two_workers(adapter)
        b.set()
        await self.until(lambda im=im: im.works["W2"].carrier == "D3", "B applied")
        await self.request(im, "Stop A.", "D4")
        await self.until(lambda im=im: im.ledger.tasks["W1.1"].ended, "A stopped")
        self.assertFalse(im.ledger.delegations["D3"].ended)
        self.assertEqual(len(adapter.runs), 2)
        b_done.set()
        events = await read_all(shared.stream, timeout=3)
        self.assertEqual(texts(events), ["B survives."])
        self.assertEqual(im.ledger.delegations["D3"].state, "answered")
        self.assertEqual(im.ledger.tasks["W1.1"].state, "cancelled")
        self.assertFalse(any(work.record.continues == "W1" for work in im.works.values()))

    async def test_scoped_steering_stop_proof_comes_from_the_selected_task(self):
        for applied in (False, True):
            with self.subTest(applied=applied):
                uncertain = "W1.1" if applied else "W1.2"

                class UnevenStop(ScriptedRun):
                    async def stop(run, uncertain=uncertain):
                        outcome = await super().stop()
                        return "unconfirmed" if run.task.task_id == uncertain else outcome

                first, deliver, finish = (asyncio.Event() for _ in range(3))
                adapter = ScriptedAdapter(
                    [wait(first), answer("First.")],
                    [wait(deliver), checkpoint(), wait(finish), answer("Second.")],
                    conductor=planned(
                        Start(("D1",), (TaskSpec("First"), TaskSpec("Second"))),
                        Steer(("D2",), (WorkTarget("W1", ("W1.2",)),)),
                        Cancel(("D3",), (WorkTarget("W1", ("W1.1",)),)),
                        Cancel(("D4",), (WorkTarget("W1", ("W1.2",)),)),
                    ),
                    run=UnevenStop,
                    max_running=2,
                )
                im = self.session(adapter)
                initial = im.admit(ask("Check both options."), delegation_id="D1")
                await self.until(lambda adapter=adapter: len(adapter.runs) == 2, "both tasks start")
                updated = im.admit(ask("Use the other date for the second."), delegation_id="D2")
                await self.until(lambda im=im: im.ledger.steerings, "the second task receives its update")
                if applied:
                    deliver.set()
                    await self.until(lambda im=im: im.works["W1"].carrier == "D2", "the second task applies its update")
                await self.request(im, "Stop the first.", "D3")
                if applied:
                    finish.set()
                else:
                    await self.request(im, "Stop the second.", "D4")
                events = await read_all(updated.stream, timeout=2)
                [obligation] = decode_provenance(events[-1]).obligations
                self.assertEqual(obligation.stop_outcome, None if applied else "unconfirmed")
                self.assertEqual(texts(events), ["Second."] if applied else [])
                await read_all(initial.stream, timeout=2)

    async def test_two_kept_live_works_with_no_old_writer_can_rejoin_one_response(self):
        a, b = asyncio.Event(), asyncio.Event()
        adapter = ScriptedAdapter(
            [MARK, wait(a), answer("A kept result.")],
            [MARK, wait(b), answer("B kept result.")],
            conductor=_start_then_steer(),
            max_running=3,
        )
        im = self.session(adapter, continuable=True)
        im.admit(ask("A."), delegation_id="D1")
        await self.until(lambda im=im: im.ledger.works.get("W1") and im.ledger.works["W1"].after_call, "A marked")
        im.admit(ask("B."), delegation_id="D2")
        await self.until(lambda im=im: im.ledger.works.get("W2") and im.ledger.works["W2"].after_call, "B marked")
        self.assertTrue(await self.hang_up(im))
        await self.call_again(im)
        self.assertIsNone(im.book.writer("D1"))
        self.assertIsNone(im.book.writer("D2"))
        shared = im.admit(ask("Resume both."), delegation_id="D3")
        await self.until(lambda im=im: len(im.ledger.steerings) == 2, "both kept steers")
        a.set()
        b.set()
        events = await read_all(shared.stream, timeout=3)
        self.assertCountEqual(texts(events), ["A kept result.", "B kept result."])
        self.assertEqual(len(adapter.runs), 2)
        for run in adapter.runs:
            self.assertFalse(any("_nva_work" in item for history in run.histories for item in history))

    async def test_rejected_or_refused_atomic_plans_never_fall_back_to_one_target(self):
        for invalid in ("unknown", "duplicate", "budget"):
            with self.subTest(invalid=invalid):
                a, b = asyncio.Event(), asyncio.Event()
                attempts = []

                async def refused(given, place, invalid=invalid, attempts=attempts):
                    attempts.append(given.attempt)
                    additional = "W1" if invalid == "duplicate" else "W404" if invalid == "unknown" else "W2"
                    decision = wire_decision(
                        steer_action(
                            "D3",
                            {"work_id": "W1"},
                            {"work_id": additional},
                        )
                    )
                    await place(decision)

                im, adapter = self.scripted_session(
                    [wait(a), answer("A.")],
                    [wait(b), answer("B.")],
                    [answer("MUST NOT FALL BACK")],
                    conductor=planned(Start(("D1",)), Start(("D2",)), refused, refused),
                    max_running=3,
                )
                im.admit(ask("Do A."), delegation_id="D1")
                await self.until(lambda im=im: len(im.running) == 1, "A running")
                im.admit(ask("Do B."), delegation_id="D2")
                await self.until(lambda im=im: len(im.running) == 2, "B running")
                if invalid == "budget":
                    im.settings = im.settings.model_copy(update={"context_tokens": 1})
                events = await self.request(im, "Update both.", "D3", timeout=3)
                self.assertEqual(events[-1]["type"], "response.completed")
                self.assertIn(
                    im.texts.routing_failed_output if invalid == "unknown" else im.texts.routing_rejected_output,
                    texts(events),
                )
                self.assertEqual(attempts, [1, 2])
                controls = [control for control in im.ledger.controls.values() if control.sources == ("D3",)]
                self.assertEqual(controls, [], "rejected plans must admit no action")
                self.assertEqual(im.ledger.delegations["D3"].state, "answered")
                self.assertEqual((len(adapter.runs), len(im.ledger.steerings)), (2, 0))
                self.assertEqual((im.works["W1"].carrier, im.works["W2"].carrier), ("D1", "D2"))
                self.assertFalse(im.waits_for_decision)

    async def test_mixed_start_and_steer_checks_budget_and_settles_exact_delivery(self):
        for budget, delivery in ((900, "applied"), (6000, "applied"), (6000, "failed"), (6000, "unknown")):
            with self.subTest(budget=budget, delivery=delivery):
                gate = asyncio.Event()
                decision = wire_decision(
                    {
                        "action": "start",
                        "delegations": ["D2"],
                        "tasks": [
                            {
                                "brief": "Look up the shipment",
                                "caller_lines": [
                                    {"source_message_id": "D2", "sequence": 0, "excerpt": "Track shipping."}
                                ],
                            }
                        ],
                    },
                    {
                        "action": "steer",
                        "delegations": ["D2"],
                        "targets": [{"work_id": "W1", "task_ids": ["W1.1"]}],
                        "updates": [
                            {
                                "task_id": "W1.1",
                                "text": "Change the booking",
                                "caller_lines": [
                                    {"source_message_id": "D2", "sequence": 0, "excerpt": "Change the booking."}
                                ],
                            }
                        ],
                    },
                )
                adapter = ScriptedAdapter(
                    [wait(gate), checkpoint(), answer("Dinner result.")],
                    [answer("Shipping result.")],
                    conductor=planned(Start(("D1",), (TaskSpec("First job"),)), decision, decision),
                    max_running=3,
                    delivery="unknown" if delivery == "unknown" else "applied",
                )
                im = self.session(adapter, context_tokens=budget)
                first = im.admit(ask("Book me a dinner."), delegation_id="D1")
                await self.until(lambda adapter=adapter: len(adapter.runs) == 1, "the existing booking worker")
                if delivery == "failed":
                    with patch.object(im.ownership, "reserve", side_effect=ValueError("Test delivery failure")):
                        response = im.admit(ask("Track shipping. Change the booking."), delegation_id="D2")
                        await im.decided()
                else:
                    response = im.admit(ask("Track shipping. Change the booking."), delegation_id="D2")
                    await im.decided()
                if delivery == "unknown":
                    await self.until(
                        lambda adapter=adapter: len(adapter.runs) == 2 and adapter.runs[1].task.ended,
                        "the new job to finish before the existing job takes the steer",
                    )
                gate.set()
                second = await read_all(response.stream, timeout=2)
                older = await read_all(first.stream, timeout=2)
                if delivery == "unknown":
                    worker = adapter.runs[0]
                    steering_id = next(boundary.steering.id for boundary in worker.boundaries if boundary.steering)
                    state, evidence = worker.histories[-1][-2:]
                    self.assertEqual((state["role"], evidence["role"]), ("developer", "user"))
                    self.assertIsInstance(evidence, WorkerData)
                    self.assertNotIn("Change the booking.", str(state))
                    self.assertIn("Change the booking.", str(evidence))
                    self.assertNotIn("Track shipping.", str(evidence))
                    self.assertEqual(
                        json.loads(state["content"][0]["text"])["private_backend_state"]["uncertain_inputs"],
                        [{"steering_id": steering_id, "source_delegation_ids": ["D2"]}],
                    )
                    self.assertEqual(
                        [receipt.source_message_ids for receipt in worker.task.input_receipts.values()], [("D1",)]
                    )
                    self.assertEqual((texts(older), older[-1]["type"]), (["Dinner result."], "response.completed"))
                    self.assertEqual((texts(second), second[-1]["type"]), (["Shipping result."], "response.failed"))
                    continue
                controls = {control.kind: control for control in im.ledger.controls.values() if "D2" in control.sources}
                if budget == 900:
                    self.assertEqual((len(adapter.runs), list(controls)), (1, []))
                    self.assertEqual(texts(second), [TEXTS.routing_rejected_output])
                    self.assertEqual(second[-1]["type"], "response.completed")
                    continue
                self.assertEqual(controls["Start"].state, "applied")
                self.assertIn("Shipping result.", texts(second))
                if delivery == "failed":
                    steer = controls["Steer"]
                    self.assertEqual((steer.state, steer.delivery), ("failed", {"W1.1": "failed"}))
                    self.assertEqual(second[-1]["type"], "response.failed")
                else:
                    self.assertEqual(controls["Steer"].state, "applied")
                    self.assertIn("Dinner result.", texts(second))
                    self.assertEqual(second[-1]["type"], "response.completed")

    async def test_held_app_call_rechecks_owned_input_before_its_first_emission(self):
        steer = placing(Steer(("D2",), (WorkTarget("W1", ("W1.2",)),), clarifications=()))
        im, task, first = await self.case_with_held_app_call(steer, sibling_result="B replanned.")
        im.admit(ask("Use the corrected second menu."), delegation_id="D2")
        await self.until(lambda: bool(task.steering), "owned correction queued")
        self.assertNotIn("call_held", im.parked)
        rest = await read_all(
            im.admit_client_request(app_outputs(first), delegation_id="D1", settings=MENU_APP).stream, timeout=3
        )
        self.assertNotIn("call_held", function_calls(rest), "a stale held app call crossed the OWN commit fence")
        self.assertEqual(task.routes["call_held"], "refused")
        self.assertEqual(im.ledger.actions["call_held"].state, "not_run")
