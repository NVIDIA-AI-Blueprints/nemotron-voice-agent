# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D107

"""A composite app frame survives rejection and split recovery across the real private boundaries."""

import asyncio
import json
from copy import deepcopy
from dataclasses import asdict
from unittest import mock

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.api.local import LocalResponsesClient
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.api.service import ResponsesService
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import (
    Start,
    Steer,
    TaskSpec,
    WorkTarget,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.session import (
    DelegationError,
    IMSession,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.stop import ForNewerRequest
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.work_provenance import decode_provenance
from nemotron_voice_agent.pipecat.runtime.gateways.live.protocol import BackendConfig, ProtocolError
from nemotron_voice_agent.pipecat.runtime.gateways.live.server_delegation import ServerDelegation
from tests.unit.pipecat.delegation_core.helpers import (
    CORE_SETTINGS,
    DEFAULTS,
    MENU,
    TEXTS,
    IMCase,
    OwnCall,
    ScriptedAdapter,
    StartEach,
    answer,
    app_call,
    ask,
    asking,
    checkpoint,
    resolving,
    steer_action,
    until,
    wait,
    wire_decision,
)
from tests.unit.pipecat.frontend_backend_live.test_delegations import Reading, Results


class Rules:
    async def check_model(self, model):
        pass

    def unsupported(self, settings):
        return []


class View:
    id, input_ms, store, acknowledgement_delay_ms = "test-call", 0, False, 0

    def __init__(self):
        self.responses = BackendConfig(model="test/model", tools=[MENU])
        self.events, self.refusals = [], []

    async def emit_paced(self, kind, *, before_publish=None, **fields):
        if before_publish is not None and not before_publish():
            return False
        self.events.append({"type": kind, **deepcopy(fields)})
        return True

    def report_refusal(self, message, code, client_event_id=None, param=None):
        self.refusals.append((message, code, client_event_id, param))


class ReadyResults(Results):
    def __init__(self):
        super().__init__()
        self.items = []

    def ready(self, result):
        self.items.append(result)
        super().ready(result)


class FanInBoundaryTests(IMCase):
    async def open_chain(self, adapter=None, *, after_app=None, independent_jobs=0):
        """Use actual IM, JSON client, ordered Live sender and frontend reader; only model rounds are scripted."""
        self.a, self.b = asyncio.Event(), asyncio.Event()
        self.calls = 0

        class Conductor:
            async def decide(_, given, place):
                self.calls += 1
                for arrival in given.arrivals:
                    if self.calls == 3:
                        decision = wire_decision(
                            steer_action(
                                arrival.delegation_id,
                                {"work_id": "W1"},
                                {"work_id": "W2"},
                                updates={"W1.1": "Use red.", "W2.1": "Use blue."},
                            )
                        )
                    else:
                        decision = Start((arrival.delegation_id,))
                    await place(decision)

        self.adapter = adapter or ScriptedAdapter(
            [
                asking("Which A option?"),
                wait(self.a),
                app_call(call_id="call-a"),
                *(
                    [wait(after_app), answer("A accepted result.")]
                    if after_app is not None
                    else [OwnCall("forbidden_after_expiry"), answer("A must not resume.")]
                ),
            ],
            [
                asking("Which B option?"),
                wait(self.b),
                *([resolving("W2.1")] if after_app is not None else []),
                answer("B accepted result."),
            ],
            *([answer("Independent.")] for _ in range(independent_jobs)),
            [answer("Fresh C result.")],
            conductor=Conductor(),
            max_running=3,
        )
        settings = CORE_SETTINGS.delegation.model_copy(
            update={"close_grace_seconds": 0.01, "client_tool_timeout_seconds": 5.0}
        )

        def record(kind, **fields):
            self.traces.append((kind, fields))

        def new_session(**hooks):
            self.im = IMSession(self.adapter, DEFAULTS, settings, TEXTS, observe=record, **hooks)
            return self.im

        service = ResponsesService(new_session, stateless_session=new_session, backend=Rules())
        self.client = LocalResponsesClient(service, close_seconds=0.5, close_core=service.aclose)
        self.addAsyncCleanup(self.client.aclose)
        self.view = View()
        self.gateway = ServerDelegation(self.client, observe=lambda *args, **kwargs: None)
        self.addAsyncCleanup(self.gateway.close)
        self.reading, self.results = Reading(), ReadyResults()
        self.reading.reader.results = self.results
        self.addCleanup(self.reading.reader.close)
        self.gateway.retain_attachments()
        self.gateway.listen(self.reading.reader)
        await self.gateway.open(self.view)

    async def delegate(self, text):
        note = self.reading.reader.worknote
        reservation = note.reserve(text)
        delegation_id = await self.gateway.delegate(ask(text))
        note.confirm(reservation, delegation_id)
        return delegation_id

    async def start_shared_update(self):
        first = await self.delegate("Do A.")
        await until(lambda: self.im.ledger.tasks.get("W1.1") and self.im.ledger.tasks["W1.1"].question, "A question")
        second = await self.delegate("Do B.")
        await until(lambda: self.im.ledger.tasks.get("W2.1") and self.im.ledger.tasks["W2.1"].question, "B question")
        combined = await self.delegate("A uses red; B uses blue.")
        await until(lambda: len(self.im.ledger.steerings) == 2, "atomic reply placement")
        return first, second, combined

    async def hold_sibling_behind_app_call(self):
        first, second, combined = await self.start_shared_update()
        self.a.set()
        await until(lambda: self.reading.reader.open[combined].waits, "A frontend app-wait frame")
        self.assertIn("call-a", self.gateway.routes.routes)
        self.b.set()
        await until(lambda: self.im.ledger.tasks["W2.1"].ended, "B answer held")
        self.assertFalse(any(item.speak == "B accepted result." for item in self.results.items))
        return first, second, combined

    def pace_app_call(self):
        reached, release = asyncio.Event(), asyncio.Event()
        emit = self.view.emit_paced

        async def pace(kind, **fields):
            if fields.get("event", {}).get("item", {}).get("call_id") == "call-a":
                reached.set()
                await release.wait()
            return await emit(kind, **fields)

        return reached, release, mock.patch.object(self.view, "emit_paced", pace)

    async def test_cancelling_an_unsent_app_call_releases_the_sibling_answer(self):
        await self.open_chain()
        reached, release, pacing = self.pace_app_call()
        with pacing:
            _, _, combined = await self.start_shared_update()
            self.a.set()
            await asyncio.wait_for(reached.wait(), 2)
            try:
                self.b.set()
                await until(lambda: self.im.ledger.tasks["W2.1"].ended, "B answer held")
                self.assertFalse(any(item.speak == "B accepted result." for item in self.results.items))
                await self.im.parked["call-a"].work.stop(ForNewerRequest())
            finally:
                release.set()
            await until(lambda: combined not in self.reading.reader.open, "the shared response completes")

        self.assertEqual(self.im.ledger.actions["call-a"].state, "not_run")
        self.assertNotIn("call-a", self.gateway.mailbox.emitted)
        self.assertEqual(self.im.ledger.delegations[combined].state, "answered")
        answers = [item.speak for item in self.results.items if item.delegation_id == combined]
        self.assertEqual(answers, ["B accepted result."])
        self.assertFalse(any(call[0] == "notice" for call in self.results.calls))

    async def expire_app_call(self, owner):
        parked = self.im.parked["call-a"]
        parked.work.deadline(parked)
        await until(lambda: self.im.ledger.tasks["W1.1"].ended, "existing app deadline")
        self.reading.reader._app_deadline(owner)

    def assert_app_call_stays_expired(self):
        self.assertEqual(self.adapter.runs[0].resumes, 0)
        self.assertFalse(any(action.name == "forbidden_after_expiry" for action in self.im.ledger.actions.values()))
        self.assertEqual(self.im.ledger.works["W1"].state, "failed")
        self.assertNotEqual(self.im.ledger.tasks["W1.1"].state, "answered")
        self.assertEqual(self.im.ledger.actions["call-a"].state, "unclear")

    async def test_expired_composite_rejects_typed_output_atomically_then_split_requests_recover_b(self):
        await self.open_chain()
        first, second, combined = await self.hold_sibling_behind_app_call()
        await self.expire_app_call(combined)
        self.assertIn(combined, self.reading.reader.open, "A frontend deadline cannot retire B's shared owner")
        a_run, b_run = self.adapter.runs[:2]

        def snapshot():
            return deepcopy(
                (self.im.history, self.im.admitted, {key: asdict(row) for key, row in self.im.ledger.actions.items()})
            )

        before = snapshot()
        output = {"type": "function_call_output", "call_id": "call-a", "output": '{"accepted": true}'}
        with self.assertRaisesRegex(ValueError, "outputs-only"):
            self.im.admit_client_request([output, *ask("Fresh C.")])
        self.assertEqual(snapshot(), before)
        await self.gateway.item_create(output, "late-output")
        await self.gateway.item_create(ask("Fresh C.")[0], "fresh-c")
        await self.gateway.response_create("composite-invalid")
        await until(lambda: self.view.refusals, "explicit mixed expired-frame refusal")
        self.assertEqual(self.view.refusals[-1][1], "invalid_value")
        self.assertIn("outputs-only", self.view.refusals[-1][0])
        self.assertIn("separately", self.view.refusals[-1][0])
        self.assertEqual(snapshot(), before)
        self.assertEqual(self.gateway.routes.routes["call-a"].delegation_id, combined)
        self.assertNotIn("call-a", self.gateway.routes.answered)
        self.assertEqual((self.gateway.routes.results, self.gateway.routes.messages), ([], []))
        self.assertEqual((a_run.resumes, b_run.resumes), (0, 0))

        await self.gateway.item_create(output, "outputs-only")
        await self.gateway.response_create("recover-b")
        await until(
            lambda: any(item.speak == "B accepted result." for item in self.results.items), "actual B ReadyResult"
        )
        self.assertEqual(sum(item.speak == "B accepted result." for item in self.results.items), 1)
        self.assertEqual(a_run.resumes, 0, "late evidence must not resume the expired task")
        self.assertEqual(sum(action.id == "call-a" for action in self.im.ledger.actions.values()), 1)

        action = self.im.ledger.actions["call-a"]
        self.assertEqual((action.state, action.late_outcome, action.late_output), ("unclear", "done", output["output"]))
        (status,), omitted = self.im.task_results(action.task)
        self.assertEqual(
            (status["outcome"], status["late_evidence"], status["output"], omitted),
            ("done", True, output["output"], 0),
        )
        self.assertEqual(self.im.reconcile_tool_outputs([output]), {"recorded_call_ids": ["call-a"], "resumed": False})
        with self.assertRaisesRegex(DelegationError, "conflicts"):
            self.im.reconcile_tool_outputs([{**output, "output": '{"accepted": false}'}])
        self.assertEqual(len(self.traced("tools.late_evidence")), 1)

        fresh = await self.delegate("Fresh C.")
        await until(lambda: any(item.speak == "Fresh C result." for item in self.results.items), "separate C result")
        self.assertNotEqual(fresh, combined)
        self.assertEqual(sum(item.speak == "B accepted result." for item in self.results.items), 1)
        self.assert_app_call_stays_expired()
        self.assertNotIn("Fresh C.", json.dumps(b_run.requests), "fresh input cannot be broadcast to B")
        self.assertEqual(len(self.adapter.runs), 3)
        self.assertEqual({first, second, combined, fresh}, set(self.im.ledger.delegations))
        self.assertTrue(all("_nva_work" not in event.get("event", {}) for event in self.view.events))
        history = await self.gateway.history()
        self.assertTrue(history["items"])
        self.assertTrue(all("_nva_work" not in item for item in history["items"]))

    async def test_legacy_single_late_output_and_typed_request_reopens_the_existing_transport_id(self):
        adapter = ScriptedAdapter(
            [app_call(call_id="call-a"), OwnCall("forbidden_after_expiry"), answer("A must not resume.")],
            [answer("Fresh C result.")],
            conductor=StartEach(),
            max_running=2,
        )
        await self.open_chain(adapter)
        first = await self.delegate("Do A.")
        await until(lambda: "call-a" in self.gateway.routes.routes, "single public app call")
        await until(lambda: self.reading.reader.open[first].waits, "single frontend app-wait frame")
        await self.expire_app_call(first)
        self.assertIn(first, self.reading.reader.ended)
        await self.gateway.item_create(
            {"type": "function_call_output", "call_id": "call-a", "output": '{"accepted": true}'}, "legacy-late"
        )
        await self.gateway.item_create(ask("Fresh C.")[0], "legacy-typed")
        await self.gateway.response_create("legacy-combined")
        await until(
            lambda: any(item.speak == "Fresh C result." for item in self.results.items), "legacy fresh C ReadyResult"
        )
        fresh = set(self.im.ledger.delegations) - {first}
        self.assertEqual(len(fresh), 1, "legacy late-plus-typed admission creates one real canonical request")
        created = [
            entry
            for entry in self.gateway.mailbox.entries
            if entry.event.get("type") == "response.created"
            and entry.event.get("_nva_work", {}).get("owner_ref") in fresh
        ]
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].delegation_id, first)
        self.assertEqual(created[0].event["_nva_work"]["response_id"], created[0].event["response"]["id"])
        self.assertEqual(self.view.refusals, [])
        self.assert_app_call_stays_expired()
        self.assertEqual(len(self.adapter.runs), 2)
        ready = [item for item in self.results.items if item.speak == "Fresh C result."]
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].delegation_id, first)
        self.assertEqual({row["delegation_id"] for row in self.view.events if row["type"] == "response.event"}, {first})
        announced = [row["delegation"]["id"] for row in self.view.events if row["type"] == "session.delegation.created"]
        self.assertEqual(announced, [first], "private canonical owner changes cannot invent a public delegation alias")
        self.assertTrue(all("_nva_work" not in event.get("event", {}) for event in self.view.events))

    async def test_actual_private_core_journal_restores_the_real_reader_without_replaying_a_ready_result(self):
        await self.open_chain(ScriptedAdapter([answer("Previously presented result.")], conductor=StartEach()))
        first = await self.delegate("Answer once.")
        await until(lambda: first in self.reading.reader.ended, "original frontend completed")
        self.assertEqual(sum(item.speak == "Previously presented result." for item in self.results.items), 1)
        entries = self.gateway.mailbox.entries
        self.assertTrue(entries)
        responses = [entry for entry in entries if entry.request_state is None]
        requests = [entry for entry in entries if entry.request_state is not None]
        self.assertTrue(responses)
        self.assertTrue(all("_nva_work" in entry.event for entry in responses))
        self.assertEqual(
            [(entry.request_ref, entry.request_state, entry.delegation_id) for entry in requests],
            [(first, "accepted", first)],
        )
        self.assertIsNone(self.gateway.mailbox.record_request(first, "submitted", first))
        self.assertTrue(all(entry.event == {} for entry in requests))
        self.assertLess(entries.index(requests[-1]), entries.index(responses[0]))
        before = deepcopy([entry.event for entry in entries])
        public_count = len(self.view.events)
        await self.gateway.suspend_attachment()
        restored, results = Reading(first), ReadyResults()
        restored.reader.results = results
        self.addCleanup(restored.reader.close)
        self.gateway.listen(restored.reader)
        await self.gateway.resume_attachment()
        self.assertIn(first, restored.reader.ended)
        self.assertEqual(results.items, [], "a private ownership restore is not permission to speak an answer again")
        self.assertEqual([entry.event for entry in self.gateway.mailbox.entries], before)
        self.assertEqual(len(self.view.events), public_count, "already forwarded public events are not re-emitted")

    async def restore_waiting_app_call(self, *, surviving):
        """Reattach the actual gateway and Core around an already published app call."""
        adapter = ScriptedAdapter([app_call(call_id="call-a"), answer("A result received.")])
        adapter.can_run_without_attachment = lambda: surviving
        await self.open_chain(adapter)
        first = await self.delegate("Look up A.")

        def sent_calls():
            return [
                frame
                for frame in self.view.events
                if frame.get("event", {}).get("type") == "response.output_item.done"
                and frame["event"].get("item", {}).get("call_id") == "call-a"
            ]

        await until(lambda: bool(sent_calls()), "the original app-facing call")
        await asyncio.gather(*self.gateway.readers)
        deadline = self.im.parked["call-a"].timer.when()
        before = sent_calls()
        await self.gateway.suspend_attachment()
        restored = Reading(first, client_tool_timeout_seconds=self.im.settings.client_tool_timeout_seconds)
        results = ReadyResults()
        restored.reader.results = results
        self.addCleanup(restored.reader.close)
        self.gateway.listen(restored.reader)
        current = await self.gateway.resume_attachment()
        await self.gateway.relay_pending_calls(current)
        self.assertEqual(sent_calls(), before)
        self.assertEqual(self.gateway.mailbox.snapshot()["unclear_calls"], [{"call_id": "call-a", "name": "get_menu"}])
        self.assertEqual(current["pending_calls"], ["call-a"] if surviving else [])
        if surviving:
            self.assertEqual(self.im.parked["call-a"].timer.when(), deadline)
            self.assertEqual(restored.reader.deadlines[first].when(), deadline)
        return {"type": "function_call_output", "call_id": "call-a", "output": "Original app result."}, results

    async def test_reconnect_waits_for_delivery_or_cancellation_before_starting_the_result_window(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                await self.check_result_window_after_reconnect(cancel=cancel)

    async def check_result_window_after_reconnect(self, *, cancel):
        """Use the actual frontend and Core to settle a tool call held across reconnect."""
        adapter = ScriptedAdapter([app_call(call_id="call-a"), answer("A result received.")])
        adapter.can_run_without_attachment = lambda: True
        await self.open_chain(adapter)
        reached, release, pacing = self.pace_app_call()
        with pacing:
            first = await self.delegate("Look up A.")
            await asyncio.wait_for(reached.wait(), 2)
            await self.gateway.suspend_attachment()
            release.set()
            await asyncio.gather(*self.gateway.readers)
            reached.clear()
            release.clear()
            restored = Reading(first, client_tool_timeout_seconds=self.im.settings.client_tool_timeout_seconds)
            self.addCleanup(restored.reader.close)
            self.gateway.listen(restored.reader)
            current = await self.gateway.resume_attachment()
            await self.gateway.relay_pending_calls(current)
            await asyncio.wait_for(reached.wait(), 2)
            try:
                parked = self.im.parked["call-a"]
                self.assertTrue(parked.response_ended)
                self.assertIsNone(parked.timer)
                self.assertNotIn(first, restored.reader.deadlines)
                if cancel:
                    await parked.work.stop(ForNewerRequest())
            finally:
                release.set()
            await self.gateway._retained_replay
        if cancel:
            self.assertEqual(self.im.ledger.actions["call-a"].state, "not_run")
            self.assertNotIn("call-a", self.gateway.mailbox.emitted)
            self.assertNotIn(first, restored.reader.open)
            self.assertIn(("ended", first, "cancelled"), restored.display.calls)
            self.assertFalse(any(call[0] == "notice" for call in restored.results.calls))
            return
        self.assertEqual(restored.reader.deadlines[first].when(), parked.timer.when())
        output = {"type": "function_call_output", "call_id": "call-a", "output": "A result."}
        await self.gateway.item_create(output, "new-result")
        await self.gateway.response_create("continue")
        await until(lambda: self.im.ledger.actions["call-a"].state == "done", "the real output is recorded")

    async def test_reconnect_accepts_original_output_for_a_still_waiting_core_frame(self):
        output, results = await self.restore_waiting_app_call(surviving=True)
        await self.gateway.item_create(output, "recovered-result")
        await self.gateway.item_create(output, "same-result")
        with self.assertRaisesRegex(ProtocolError, "different result"):
            await self.gateway.item_create({**output, "output": "Conflicting result."}, "conflicting-pending-result")
        self.assertIn("call-a", self.im.parked)
        self.assertEqual(len(self.gateway.routes.results), 1)
        self.assertEqual(self.adapter.runs[0].resumes, 0)

        await self.gateway.response_create("complete-original-frame")
        await until(
            lambda: any(item.speak == "A result received." for item in results.items), "resumed worker's result"
        )
        action = self.im.ledger.actions["call-a"]
        self.assertEqual((action.state, action.output), ("done", output["output"]))
        self.assertEqual(self.adapter.runs[0].resumes, 1)
        self.assertEqual(self.gateway.mailbox.snapshot()["unclear_calls"], [])
        await self.gateway.item_create(output, "repeated-recorded-result")
        self.assertEqual(self.gateway.routes.results, [])
        self.assertEqual(len(self.adapter.runs), 1)

    async def test_reconnect_keeps_the_returned_result_as_evidence_after_the_task_stops(self):
        output, results = await self.restore_waiting_app_call(surviving=False)
        await self.gateway.item_create(output, "recovered-late-result")
        action = self.im.ledger.actions["call-a"]
        self.assertEqual((action.state, action.late_outcome, action.late_output), ("unclear", "done", output["output"]))
        self.assertEqual(self.im.ledger.tasks["W1.1"].state, "cancelled")
        self.assertEqual(self.gateway.mailbox.snapshot()["unclear_calls"], [])
        with self.assertRaisesRegex(ProtocolError, "conflicts"):
            await self.gateway.item_create({**output, "output": "Conflicting result."}, "conflicting-late-result")
        await self.gateway.response_create("after-evidence-only")
        await self.turns()
        self.assertEqual(self.adapter.runs[0].resumes, 0)
        self.assertEqual(len(self.adapter.runs), 1)
        self.assertEqual(len(self.im.ledger.works), 1)
        self.assertEqual(results.items, [])

        await self.gateway.item_create(output, "repeat-terminal-evidence")
        await self.gateway.item_create(ask("A separate lookup.")[0], "fresh-input")
        await self.gateway.response_create("start-fresh-input")
        await until(lambda: len(self.adapter.runs) == 2, "the actual fresh input starts")
        self.assertIn("A separate lookup.", str(self.adapter.runs[1].task.initial_requested))
        self.assertEqual(self.adapter.runs[0].resumes, 0)

    async def test_held_and_flushed_sibling_survives_pruning_and_keeps_its_wire_revision(self):
        sweep = 36
        after_app = asyncio.Event()
        await self.open_chain(after_app=after_app, independent_jobs=2 * sweep)
        _, _, combined = await self.hold_sibling_behind_app_call()
        im = self.im

        async def prune_with_closed_jobs(start):
            for index in range(start, start + sweep):
                await self.request(im, "Independent.", f"independent-{index}")
            self.assertIn("W2", im.ledger.works, "the open owner retains its sibling's exact outcome")

        await prune_with_closed_jobs(0)
        self.assertIn("W2.1", im.thread_history)
        self.assertEqual([question.resolution for question in im.ledger.tasks["W2.1"].questions], ["answered"])
        self.assertTrue(next(work for work in im.ownership.members(combined) if work.id == "W2").held)
        await self.gateway.item_create(
            {"type": "function_call_output", "call_id": "call-a", "output": '{"accepted": true}'}, "app-output"
        )
        await self.gateway.response_create("flush-b")
        await until(lambda: any(item.speak == "B accepted result." for item in self.results.items), "B ReadyResult")
        self.assertFalse(im.ledger.tasks["W1.1"].ended)
        await prune_with_closed_jobs(sweep)
        after_app.set()
        await until(lambda: combined in self.reading.reader.ended, "A result closes the shared response")
        self.assertCountEqual(
            [item.speak for item in self.results.items if item.delegation_id == combined],
            ["A accepted result.", "B accepted result."],
        )
        self.assertFalse(any(kind == "delegation.provenance_refused" for kind, _ in self.reading.events))
        snapshots = {}
        for entry in self.gateway.mailbox.entries:
            if entry.delegation_id != combined or (proof := decode_provenance(entry.event)) is None:
                continue
            key = proof.owner_ref, proof.owner_revision
            facts = proof.bindings, proof.obligations, proof.owner_settlements, proof.request_sources
            self.assertEqual(snapshots.setdefault(key, facts), facts)
        self.assertTrue(snapshots)
        im._prune()
        self.assertLessEqual(len(im.ownership.branches), len(im.ledger.works))
        self.assertTrue(set(im.ownership.owners) <= set(im.ledger.delegations))

    async def test_older_unapplied_input_follows_up_under_the_newer_publication_owner_through_the_reader(self):
        for applied in (False, True):
            with self.subTest(applied=applied):
                await self.check_older_scoped_input(applied)

    async def check_older_scoped_input(self, applied):
        a_new, a_done, a_old, old_done, b, b_done = [asyncio.Event() for _ in range(6)]
        effect = OwnCall("actual_followup_effect")
        old_task = (
            [checkpoint(), effect, wait(old_done), answer("Older pending input handled.")]
            if applied
            else [answer("Old second-task answer.", boundary=False)]
        )

        class Conductor:
            round = 0

            async def decide(self, given, place):
                self.round += 1
                for arrival in given.arrivals:
                    source = arrival.delegation_id
                    if self.round == 1:
                        action = Start((source,), (TaskSpec("A one"), TaskSpec("A two")))
                    elif self.round == 3:
                        action = wire_decision(
                            steer_action(
                                source,
                                {"work_id": "W1", "task_ids": ["W1.2"]},
                                {"work_id": "W2"},
                                updates={
                                    "W1.2": "Apply the correction to A two.",
                                    "W2.1": "Apply the correction to B.",
                                },
                            )
                        )
                    elif self.round == 4:
                        action = Steer((source,), (WorkTarget("W1", ("W1.1",)),))
                    else:
                        action = Start((source,))
                    await place(action)

        adapter = ScriptedAdapter(
            [wait(a_new), checkpoint(), wait(a_done), answer("Newer first-task answer.")],
            [wait(a_old), *old_task],
            [wait(b), checkpoint(), wait(b_done), answer("B done.")],
            [effect, answer("Older pending input handled.")],
            conductor=Conductor(),
            max_running=3,
        )
        await self.open_chain(adapter)
        first = await self.delegate("A initial.")
        await until(lambda: len(adapter.runs) == 2, "A tasks")
        second = await self.delegate("B initial.")
        await until(lambda: len(adapter.runs) == 3, "B task")
        shared = await self.delegate("Older correction for A two and B.")
        await until(lambda: len(self.im.ledger.steerings) == 2, "shared placement")
        old_branch = self.im.works["W1"].branch_ref
        b_branch = self.im.works["W2"].branch_ref
        edge = next(edge for edge in self.im.ownership.owner(shared).edges.values() if edge.branch == old_branch)
        identity = edge.ref
        newer = await self.delegate("Newer correction for A one.")
        await until(lambda: len(self.im.ledger.steerings) == 3, "newer placement")
        a_new.set()
        await until(lambda: self.im.works["W1"].carrier == newer, "newer input applied first")
        self.assertEqual((edge.state, edge.pending), ("transferred", {"W1.2"}))
        b.set()
        await until(lambda: self.im.works["W2"].carrier == shared, "B input applied")
        self.assertEqual(self.im.ownership.pending_sources(), {shared})
        self.assertFalse(self.im.waits_for_decision)

        b_done.set()
        await until(lambda: any(item.speak == "B done." for item in self.results.items), "B real ReadyResult")
        self.assertFalse(self.im.ledger.delegations[shared].ended)
        self.assertIn(shared, self.reading.reader.open)
        if not applied:
            a_done.set()
        a_old.set()
        await until(
            lambda: any(action.name == "actual_followup_effect" for action in self.im.ledger.actions.values()),
            "actual application of the older input",
        )
        actual = adapter.runs[1 if applied else 3]
        self.assertEqual(actual.task.work.carrier, newer)
        if not applied:
            self.assertEqual(actual.task.request_ids, [shared])
            self.assertEqual(actual.task.continues, "W1.2")
            self.assertNotEqual(edge.branch, old_branch)
            predecessor = adapter.runs[1].task.record
            self.assertEqual((predecessor.answer, predecessor.result.state), ("Old second-task answer.", "final"))
        context = json.dumps(actual.histories)
        self.assertIn("Apply the correction to A two.", context)
        for foreign in ("Apply the correction to B.", "Older correction for A two and B."):
            self.assertNotIn(foreign, context)
        receipts = [
            receipt
            for identity, receipt in actual.task.input_receipts.items()
            if identity.startswith(f"{actual.task.task_id}.I") and (not applied or receipt.revision == 2)
        ]
        self.assertEqual([receipt.source_message_ids for receipt in receipts], [(shared,)])
        if applied:
            self.assertEqual(edge.branch, old_branch)
        self.assertEqual((edge.ref, edge.branch, edge.state), (identity, actual.task.work.branch_ref, "transferred"))
        self.assertFalse(edge.pending)
        self.assertEqual(self.im.ownership.pending_sources(), set())
        a_done.set()
        old_done.set()
        await until(lambda: len(self.results.items) == 3, "all three ReadyResults")
        self.assertCountEqual(
            [(item.delegation_id, item.speak) for item in self.results.items],
            [(shared, "B done."), (newer, "Newer first-task answer."), (newer, "Older pending input handled.")],
        )
        self.assertEqual(sum(action.name == "actual_followup_effect" for action in self.im.ledger.actions.values()), 1)
        self.assertEqual((self.im.admitted, len(adapter.runs)), (4, 3 if applied else 4))
        sources = self.reading.reader.ownership.request_sources
        self.assertEqual(sources[old_branch], {first, shared, newer} if applied else {first, newer})
        self.assertEqual(sources[b_branch], {second, shared})
        self.assertFalse(any(kind == "delegation.provenance_refused" for kind, _ in self.reading.events))
