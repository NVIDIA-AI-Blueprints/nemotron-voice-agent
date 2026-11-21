# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Question ownership, routing epochs and focused task histories under adversarial interleavings."""

import asyncio
import json
from unittest.mock import patch

from examples.frontend_backend_live.delegation_core.backend.port import (
    CallerMessage,
    Cancel,
    Start,
    Steer,
    TaskSpec,
    WorkTarget,
)
from examples.frontend_backend_live.delegation_core.im.conversation import item_size
from examples.frontend_backend_live.delegation_core.im.session import DelegationError
from examples.frontend_backend_live.delegation_core.im.stop import ForNewerRequest
from examples.frontend_backend_live.delegation_core.im.tools import Invocation
from tests.unit.delegation_core.helpers import (
    QUESTION,
    IMCase,
    ScriptedAdapter,
    ScriptedConductor,
    ScriptedProvider,
    ScriptedRun,
    WorkTool,
    answer,
    ask,
    asking,
    checkpoint,
    gated,
    placing,
    planned,
    read_all,
    texts,
    wait,
)
from tests.unit.delegation_core.test_agent import answer_status, cancel, conductor_on, decisions, start, steer


def resolution(call_id, question_id, disposition="no_longer_needed", input_ref=None):
    selected = {"question_id": question_id, "disposition": disposition}
    if input_ref is not None:
        selected["input_ref"] = input_ref
    return Invocation(call_id, "resolve_questions", {"resolutions": [selected]})


class QuestionLifecycleTests(IMCase):
    async def test_pending_reply_is_reported_and_only_resolves_after_application(self):
        from examples.frontend_backend_live.delegation_core.backend.agent.render import work as render_work
        from examples.frontend_backend_live.delegation_core.backend.harness.render import Renderer
        from tests.unit.delegation_core.harness.doubles import settings

        im, task, _, finish = await self.live_question()
        im.admit(ask("Large."), delegation_id="D2")
        await self.until(lambda: task.steering, "pending steering")
        native = Renderer(settings().prompts)

        def check(delivery, carrier):
            summary = task.work.summary()
            self.assertEqual(
                [(row.delegation_id, row.delivery) for row in summary.requests], [("D1", "landed"), ("D2", delivery)]
            )
            self.assertEqual(summary.carrier, carrier)
            rendered = render_work(summary)
            self.assertEqual(rendered["requests"][1].get("delivery"), "pending" if delivery == "pending" else None)
            self.assertIn(
                json.dumps({"delegation_id": "D2", "delivery": delivery, "text": "Large."}),
                native._work(summary),
            )

        check("pending", "D1")
        steering = task.take_boundary_input().steering
        self.assertIsNotNone(steering)
        self.assertTrue(task.taken)
        check("pending", "D1")
        envelope = steering.items[-1]
        data = json.loads(envelope["content"][0]["text"])["private_backend_data"]
        [source] = data["source_message_ids"]
        self.assertEqual(envelope["role"], "developer")
        self.assertFalse(data["proves_delivery"])
        invocation = resolution("resolve", task.record.questions[0].id, "answered", source)
        with self.assertRaisesRegex(ValueError, "newer applied"):
            task.resolve_questions(invocation)
        task.settle(steering, "applied")
        check("landed", "D2")
        self.assertEqual(json.loads(task.resolve_questions(invocation))["resolutions"][0]["disposition"], "answered")
        self.assertEqual(task.record.questions[0].input_ref, "W1.1.I2")
        self.assertEqual(task.input_receipts["W1.1.I2"].source_message_ids, (source,))
        self.assertNotIn(envelope["id"], task.input_receipts["W1.1.I2"].source_message_ids)
        finish.set()

    async def live_question(self, question=QUESTION, **settings):
        land, finish = asyncio.Event(), asyncio.Event()
        adapter = ScriptedAdapter([asking(question), wait(land), checkpoint(), wait(finish), answer("Finished.")])
        im = self.session(adapter, **settings)
        im.admit(ask("Coffee."), delegation_id="D1")
        await self.until(lambda: self.traced("im.question.written"), "written question")
        return im, adapter.runs[0].task, land, finish

    async def test_unrelated_targeted_input_does_not_resolve_a_question(self):
        im, task, land, finish = await self.live_question()
        im.admit(ask("Also a muffin."), delegation_id="D2")
        await self.until(lambda: task.steering, "steering")
        land.set()
        await self.until(lambda: task.input_revision == 2, "actual landing")
        self.assertEqual(task.record.question, QUESTION)
        q = task.record.questions[0]
        await task.tools().execute("status", "caller_message", '{"kind":"status","message":"The menu is ready."}')
        self.assertEqual((q.resolution, task.record.progress, task.answered), ("open", "The menu is ready.", False))
        with self.assertRaisesRegex(ValueError, "newer applied"):
            task.resolve_questions(resolution("r1", q.id, "answered", "W1.1.I1"))
        self.assertTrue(q.open)
        invalid = {
            "resolutions": [
                {"question_id": q.id, "disposition": "no_longer_needed"},
                {"question_id": "unknown", "disposition": "no_longer_needed"},
            ]
        }
        with self.assertRaises(ValueError):
            await task.tools().execute("invalid", "resolve_questions", json.dumps(invalid))
        self.assertTrue(q.open)
        output = await task.tools().execute(
            "r2",
            "resolve_questions",
            json.dumps(resolution("r2", q.id).arguments),
        )
        self.assertEqual(
            json.loads(output)["resolutions"],
            [{"question_id": q.id, "disposition": "no_longer_needed", "input_ref": None}],
        )
        self.assertEqual((q.resolution, q.input_ref, task.record.question), ("no_longer_needed", None, None))
        [written] = self.traced("im.question.written")
        self.assertEqual(
            (written["question_id"], written["call_id"], written["input_revision"]), (q.id, q.call_id, q.input_revision)
        )
        resolved = self.traced("im.question.resolved")
        self.assertEqual(len(resolved), 1)  # The rejected old reference produced no resolution event.
        self.assertEqual((resolved[0]["question_id"], resolved[0]["input_ref"]), (q.id, None))
        landed = self.traced("im.input.landed")[-1]
        self.assertEqual((landed["input_ref"], landed["input_revision"]), ("W1.1.I2", 2))
        self.assertTrue(all(identity.startswith("D2") for identity in landed["source_message_ids"]))
        self.assertFalse(task.completed)
        for name, message in (("day", "Which day?"), ("milk", "Which milk?")):
            await task.tools().execute(name, "caller_message", json.dumps({"kind": "question", "message": message}))
        self.assertEqual([row.text for row in task.record.questions if row.open], ["Which day?", "Which milk?"])
        finish.set()

    async def test_replayed_clarification_source_is_not_a_new_answer_alias(self):
        apply, finish = asyncio.Event(), asyncio.Event()

        async def route_reply(given, place):
            [clarification] = given.clarifications
            await place(Steer(("D2",), (WorkTarget("W1"),), clarifications=(clarification["id"],)))

        conductor = planned(
            placing(Start(("D1",)), CallerMessage(("D1",), "question", "Which booking should change?")),
            route_reply,
        )
        im, adapter = self.scripted_session(
            [asking("What time?"), asking("What day?"), wait(apply), checkpoint(), wait(finish), answer()],
            conductor=conductor,
            max_running=2,
        )
        first = im.admit(ask("Find dinner options and change that booking."), delegation_id="D1")
        await self.until(lambda: len(self.traced("im.question.written")) == 2, "two written worker questions")
        task = adapter.runs[0].task
        when, day = task.record.questions
        reply = im.admit(ask("The dinner booking, tomorrow at eight."), delegation_id="D2")
        await self.until(lambda: task.steering, "retained clarification and new reply")
        apply.set()
        await self.until(lambda: task.input_revision == 2, "applied update")
        receipt = task.input_receipts[f"{task.task_id}.I2"]
        self.assertEqual(receipt.source_message_ids, ("D1", "D2"))

        with self.assertRaisesRegex(ValueError, "newer applied"):
            task.resolve_questions(resolution("old", when.id, "answered", "D1"))
        self.assertTrue(when.open)
        task.resolve_questions(resolution("time", when.id, "answered", "D2"))
        task.resolve_questions(resolution("day", day.id, "answered", receipt.id))
        self.assertEqual((when.input_ref, day.input_ref), (receipt.id, receipt.id))
        finish.set()
        await asyncio.gather(read_all(first.stream), read_all(reply.stream))

    async def test_question_replay_and_selected_resolution_preserve_independent_questions(self):
        _, task, _, finish = await self.live_question()
        tools = task.tools()
        q = task.record.questions[0]
        reply = await tools.execute(q.call_id, "caller_message", json.dumps({"kind": "question", "message": QUESTION}))
        self.assertEqual(json.loads(reply)["question_id"], q.id)
        second = await tools.execute(
            "another", "caller_message", json.dumps({"kind": "question", "message": "Hot or iced?"})
        )
        self.assertNotEqual(json.loads(second)["question_id"], q.id)
        with self.assertRaisesRegex(ValueError, "different arguments"):
            await tools.execute(q.call_id, "caller_message", json.dumps({"kind": "question", "message": "Changed?"}))
        self.assertEqual(len(task.record.questions), 2)
        self.assertEqual(len(self.traced("im.question.written")), 2)
        task.resolve_questions(resolution("selected", q.id))
        self.assertFalse(q.open)
        self.assertTrue(task.record.questions[1].open)
        self.assertEqual(task.record.question, "Hot or iced?")
        await tools.execute("third", "caller_message", '{"kind":"question","message":"Would you like milk?"}')
        selected = [
            {"question_id": item.id, "disposition": "no_longer_needed"} for item in task.record.questions if item.open
        ]
        result = await tools.execute("remaining", "resolve_questions", json.dumps({"resolutions": selected}))
        self.assertEqual(len(json.loads(result)["resolutions"]), 2)
        self.assertFalse(any(item.open for item in task.record.questions))
        finish.set()

    async def test_stop_fences_question_resolution_and_delayed_asks(self):
        _, task, _, finish = await self.live_question()
        task.cancel_own()
        self.assertIsNone(task.record.question)
        self.assertEqual(task.record.questions[0].resolution, "cancelled")
        with self.assertRaisesRegex(ValueError, "ended or is stopping"):
            task.resolve_questions(resolution("r", "W1.1.Q1"))
        finish.set()

    async def test_question_that_cannot_fit_required_state_is_rejected_without_publication(self):
        finish = asyncio.Event()
        question = ("Which option? " + "detail " * 1200).strip()
        adapter = ScriptedAdapter([wait(finish), answer()])
        im = self.session(adapter, context_tokens=1000)
        im.admit(ask("Find an option."), delegation_id="D1")
        await self.until(lambda: adapter.runs, "running task")
        task = adapter.runs[0].task
        self.assertLess(sum(map(item_size, task.view.history())), im.settings.context_tokens)
        with self.assertRaisesRegex(ValueError, "context budget"):
            await task.tools().execute(
                "too_large", "caller_message", json.dumps({"kind": "question", "message": question})
            )
        self.assertEqual(task.record.questions, [])
        self.assertFalse(self.traced("im.question.written"))
        self.assertLessEqual(sum(map(item_size, task.history())), im.settings.context_tokens)
        finish.set()

    async def test_old_publication_receipts_cannot_crowd_out_open_questions_or_the_answer_input(self):
        im, task, land, finish = await self.live_question(context_tokens=1000)
        await task.tools().execute("another", "caller_message", '{"kind":"question","message":"Which milk?"}')
        before = task.history()
        self.assertLessEqual(sum(map(item_size, before)), im.settings.context_tokens)
        self.assertEqual(task.history()[-1], before[-1])
        for number in range(25):
            await task.tools().execute(
                f"status_{number}", "caller_message", '{"kind":"status","message":"Still checking the menu."}'
            )
        im.admit(ask("A large one, please."), delegation_id="D2")
        await self.until(lambda: task.steering, "reply waiting")
        land.set()
        await self.until(lambda: task.input_revision == 2, "landed reply")

        history = task.history()
        self.assertLessEqual(sum(map(item_size, history)), im.settings.context_tokens)
        self.assertIn("A large one, please.", str(history))
        state = json.loads(history[-1]["content"][0]["text"])["private_backend_state"]
        questions = task.record.questions
        self.assertEqual([row["question_id"] for row in state["questions"]], [q.id for q in questions])
        self.assertLess(len(state["publications"]), len(task.publications))
        input_ref = state["applied_inputs"][-1]["input_ref"]
        self.assertEqual(input_ref, f"{task.task_id}.I2")
        task.resolve_questions(resolution("size", questions[0].id, "answered", input_ref))
        self.assertEqual((questions[0].input_ref, questions[1].resolution), (input_ref, "open"))
        after = task.history()
        self.assertNotEqual(before[-1]["id"], after[-1]["id"])
        self.assertLessEqual(sum(map(item_size, after)), im.settings.context_tokens)
        metadata = json.loads(after[-1]["content"][0]["text"])["private_backend_state"]
        self.assertEqual([item["question_id"] for item in metadata["questions"]], [questions[1].id])
        self.assertEqual(
            metadata["recent_resolutions"][-1],
            {"question_id": questions[0].id, "disposition": "answered", "input_ref": input_ref},
        )
        self.assertEqual(
            metadata["applied_inputs"][-1],
            {"input_ref": input_ref, "revision": 2, "source_message_ids": ["D2"], "caller_order": [["D2", 2]]},
        )
        self.assertEqual(task.history()[-1], after[-1])
        finish.set()

    async def test_follow_up_transfers_canonical_question_ownership(self):
        end = asyncio.Event()
        im, adapter = self.scripted_session(
            [asking(), wait(end), answer("Found it.")], [wait(asyncio.Event())], delivery="not_applied"
        )
        im.admit(ask("Coffee."), delegation_id="D1")
        await self.until(lambda: self.traced("im.question.written"), "question")
        im.admit(ask("Large."), delegation_id="D2")
        await self.until(lambda: im.ledger.steerings, "unlanded reply")
        end.set()
        await self.until(lambda: len(adapter.runs) == 2, "follow-up")
        old, new = im.ledger.tasks["W1.1"], adapter.runs[1].task
        self.assertEqual((old.questions[0].resolution, old.question), ("open", None))
        self.assertTrue(old.questions[0].open)
        self.assertEqual(old.questions[0].owner_task_id, new.task_id)
        self.assertIs(new.record.questions[0], old.questions[0])
        self.assertIn(QUESTION, str(new.history()))
        self.assertEqual(new.record.question, QUESTION)
        self.assertEqual(old.questions[0].dispatch, "written")
        self.assertIsNone(old.questions[0].input_ref)
        with self.assertRaisesRegex(ValueError, "ended"):
            adapter.runs[0].task.resolve_questions(resolution("stale_owner", old.questions[0].id))
        new.resolve_questions(resolution("r", old.questions[0].id))
        await new.tools().execute("next_ask", "caller_message", '{"kind":"question","message":"Which day?"}')
        self.assertEqual(new.record.question, "Which day?")
        self.assertEqual(new.record.questions[1].creator_task_id, new.task_id)
        self.assertEqual(old.questions[0].resolution, "no_longer_needed")


class PlacementLifecycleTests(IMCase):
    async def test_status_uses_live_or_retained_work_without_steering_it(self):
        for state, count in (("live", 1), ("ended", 1), ("ended", 2)):
            with self.subTest(state=state, count=count):
                self.traces.clear()
                await self.check_conductor_status(state, count)

    async def check_conductor_status(self, state, count):
        finish = asyncio.Event()
        live = state == "live"
        line = (
            "The job is waiting for your reply."
            if live
            else "The job has provided an update."
            if count == 1
            else "Status for “Search 1”. The job has provided an update.\n"
            "Status for “Search 2”. The job has provided an update."
        )
        model = ScriptedProvider(
            *(decisions(start(f"D{number}", tasks=[f"Search {number}"])) for number in range(1, count + 1)),
            decisions(answer_status("Dstatus", "W404")),
            decisions(answer_status("Dstatus", *(f"W{number}" for number in range(1, count + 1)))),
        )
        scripts = [[answer(f"Confirmed result {number}.")] for number in range(count)]
        if live:
            scripts = [
                [
                    asking("Which day?"),
                    wait(finish),
                    WorkTool("caller_message", {"kind": "status", "message": "Checking."}),
                ]
            ]
        im, adapter = self.scripted_session(*scripts, max_running=3, conductor=conductor_on(model))
        if live:
            work = im.admit(ask("Find a table."), delegation_id="D1")
            await self.until(lambda: self.traced("im.question.written"), "live day question")
        else:
            for number in range(1, count + 1):
                await self.request(im, f"Search {number}.", f"D{number}")
                await self.until(lambda: not im.works, "ended search")
            self.assertEqual(len(im.ended_works), count)
        task = adapter.runs[0].task
        revision = task.input_revision
        events = await self.request(im, "Where does the search stand?", "Dstatus")
        self.assertEqual(texts(events), [line])
        self.assertEqual(events[-1]["type"], "response.completed")
        source = im.ledger.delegations["Dstatus"]
        self.assertEqual(source.state, "answered")
        self.assertFalse(source.work)
        self.assertEqual((len(im.ledger.works), len(adapter.runs), len(model.payloads)), (count, count, count + 2))
        self.assertEqual([row["reason"] for row in self.traced("im.decide.retry")], ["decision_rejected"])
        self.assertFalse(self.traced("im.placed_by_rule"))
        if live:
            self.assertEqual((task.record.question, task.record.questions[0].resolution), ("Which day?", "open"))
            self.assertEqual(task.input_revision, revision)
            self.assertFalse(task.ended or task.steering)
            finish.set()
            ongoing = await read_all(work.stream)
            self.assertEqual(texts(ongoing), ["Which day?", "Checking."])
            self.assertEqual(ongoing[-1]["type"], "response.failed")
            self.assertEqual(im.ledger.tasks["W1.1"].state, "failed")

    async def test_real_conductor_submits_steering_with_stop_or_status_in_one_completed_response(self):
        for operation in ("stop_after", "stop_before", "status"):
            with self.subTest(operation=operation):
                report_status = operation == "status"
                movie, dinner, control = asyncio.Event(), asyncio.Event(), asyncio.Event()
                update_dinner = steer("W2", "D3") | {
                    "updates": [
                        {
                            "task_id": "W2.1",
                            "text": "The requested dinner time is six.",
                            "caller_lines": [
                                {"source_message_id": "D3", "sequence": 0, "excerpt": "Yes dinner at six"}
                            ],
                        }
                    ]
                }
                movie_action = answer_status("D3", "W1") if report_status else cancel("W1", delegations=("D3",))
                actions = [update_dinner, movie_action]
                if operation == "stop_before":
                    actions.reverse()
                model = ScriptedProvider(
                    decisions(start("D1")), decisions(start("D2")), gated(control, decisions(*actions))
                )
                im, adapter = self.scripted_session(
                    [wait(movie), answer()],
                    [wait(dinner), checkpoint(), answer("Dinner at six.")],
                    max_running=2,
                    conductor=conductor_on(model),
                )
                im.admit(ask("Movie tickets."), delegation_id="D1")
                await self.until(lambda adapter=adapter: len(adapter.runs) == 1, "movie")
                im.admit(ask("Dinner table."), delegation_id="D2")
                await self.until(lambda adapter=adapter: len(adapter.runs) == 2, "dinner")
                movie_request = "check movie progress" if report_status else "stop movie work"
                admission = im.admit(ask(f"Yes dinner at six. {movie_request}."), delegation_id="D3")
                await self.until(lambda model=model: len(model.payloads) == 3, "complete control response")
                self.assertIn("D3", im.arrivals)
                self.assertTrue(im.waits_for_decision)
                self.assertEqual(im.ledger.tasks["W1.1"].stop.state, "none")
                control.set()
                if report_status:
                    await self.until(lambda im=im: not im.waits_for_decision, "mixed status accepted")
                    self.assertEqual(im.ledger.works["W1"].state, "running")
                    self.assertEqual(im.ledger.tasks["W1.1"].input_revision, 1)
                else:
                    await self.until(lambda im=im: im.ledger.works["W1"].state == "cancelled", "movie stopped")
                dinner.set()
                output = texts(await read_all(admission.stream))
                self.assertIn("Dinner at six.", output)
                if report_status:
                    self.assertIn("The job is still running.", output)
                    movie.set()
                self.assertIn("The requested dinner time is six.", str(adapter.runs[1].histories))
                self.assertNotIn(movie_request, str(adapter.runs[1].histories))
                self.assertEqual(im.ledger.works["W2"].carrier, "D3")
                self.assertFalse(im.waits_for_decision)
                self.assertEqual(len(model.payloads), 3)
                self.assertEqual(len(im.ledger.works), 2)
                self.assertFalse(self.traced("im.placed_by_rule"))

    async def test_pending_start_reacts_to_other_slot_release_or_its_own_stop(self):
        for stop_queued in (False, True):
            with self.subTest(stop_queued=stop_queued):
                await self.check_pending_start(stop_queued)

    async def check_pending_start(self, stop_queued):
        stopping, settle_stop, finish_peer, finish_new = (asyncio.Event() for _ in range(4))

        class HeldStop(ScriptedRun):
            async def stop(self):
                if self.task.task_id == "W1.1":
                    stopping.set()
                    await settle_stop.wait()
                return await super().stop()

        target = WorkTarget("W2", ("W2.1",))
        third = Cancel(("D3",), (target,)) if stop_queued else Steer(("D3",), (target,))
        im, adapter = self.scripted_session(
            [wait(asyncio.Event()), answer()],
            [wait(finish_peer), answer("Other lookup done.")],
            [wait(finish_new), answer("Dinner at eight.")],
            conductor=planned(
                Start(("D1",), (TaskSpec("movie"), TaskSpec("unrelated lookup"))),
                placing(Cancel(("D2",), (WorkTarget("W1", ("W1.1",)),)), Start(("D2",), (TaskSpec("dinner"),))),
                third,
            ),
            max_running=2,
            run=HeldStop,
        )
        im.admit(ask("Find movies and do the lookup."), delegation_id="D1")
        await self.until(lambda: len(adapter.runs) == 2, "two occupied slots")
        start = im.admit(ask("Stop movies and find dinner at seven."), delegation_id="D2")
        try:
            await asyncio.wait_for(stopping.wait(), 2)
            (receipt,) = [row for row in im.ledger.controls.values() if row.kind == "Start" and row.sources == ("D2",)]
            self.assertEqual((receipt.state, receipt.receipt["work_id"]), ("pending", "W2"))
            self.assertEqual(len(adapter.runs), 2)
            newer = im.admit(ask("Stop dinner." if stop_queued else "Make dinner at eight."), delegation_id="D3")
            if stop_queued:
                await read_all(newer.stream, timeout=2)
                await self.until(lambda: receipt.state == "cancelled", "the pending start settles", timeout=2)
                self.assertEqual(len(adapter.runs), 2)
            else:
                await self.until(lambda: im.ledger.delegations["D3"].work == "W2", "the exact queued update")
                finish_peer.set()
                await self.until(lambda: len(adapter.runs) == 3, "dinner uses the free slot", timeout=2)
                self.assertEqual(receipt.state, "applied")
                self.assertEqual(len(im.running), 2)
            self.assertEqual(im.ledger.tasks["W1.1"].stop.state, "stop_requested")
        finally:
            settle_stop.set()
            finish_peer.set()
            finish_new.set()
        await read_all(start.stream, timeout=2)
        if not stop_queued:
            self.assertIn("Make dinner at eight.", str(adapter.runs[-1].histories[0]))
            self.assertIn("Dinner at eight.", texts(await read_all(newer.stream, timeout=2)))
        await im.close()

    async def test_real_single_call_conductor_finishes_a_stop_only_source_without_new_work(self):
        model = ScriptedProvider(
            decisions(start("D1")),
            decisions(cancel("W1")),
        )
        adapter = ScriptedAdapter([wait(asyncio.Event()), answer()], conductor=conductor_on(model), max_running=2)
        im = self.session(adapter)
        im.admit(ask("Movie tickets."), delegation_id="D1")
        await self.until(lambda: adapter.runs, "movie")
        result = await self.request(im, "Stop the movie work.", "D2")
        self.assertEqual(texts(result), [im.texts.control_requested_line])
        self.assertEqual(len(im.ledger.works), 1)
        self.assertEqual(len(model.payloads), 2)
        self.assertFalse(self.traced("im.placed_by_rule"))

    async def test_clarification_releases_the_gate_and_uses_the_newest_source(self):
        for requests in (("Change it to six.",), ("Change it.", "To six.", "For everyone.")):
            with self.subTest(requests=requests):
                self.traces.clear()
                sources = [f"D{n}" for n in range(1, len(requests) + 1)]
                reply = f"D{len(sources) + 1}"

                async def choose(given, place, reply=reply):
                    [row] = given.clarifications
                    await place(Start((reply,), (TaskSpec("chosen booking"),), (row["id"],)))

                clarify = CallerMessage(tuple(reversed(sources)), "question", "Which booking?", clarifications=())
                adapter = ScriptedAdapter([answer()], max_running=2, conductor=planned(clarify, choose))
                im = self.session(adapter)
                admissions = [
                    im.admit(ask(text), delegation_id=source) for source, text in zip(sources, requests, strict=True)
                ]
                streams = await asyncio.gather(*(read_all(a.stream) for a in admissions))
                [question] = [
                    row["item"]
                    for stream in streams
                    for row in stream
                    if row["type"] == "response.output_item.done" and row["item"].get("phase") == "final_answer"
                ]
                self.assertEqual(texts(streams[-1]), ["Which booking?"])
                self.assertFalse(im.works)
                self.assertFalse(im.waits_for_decision)
                for index, source in enumerate(sources[:-1]):
                    self.assertEqual(im.ledger.delegations[source].state, "handed_over")
                    self.assertEqual(
                        texts(streams[index]), [im.texts.handover_line.format(delegation=sources[index + 1])]
                    )
                self.assertEqual(im.ledger.delegations[sources[-1]].state, "answered")
                [row] = im.clarifications.values()
                self.assertEqual((row["carrier"], row["source_delegation_ids"]), (sources[-1], sources))
                self.assertEqual(row["input"].count(question), 1)
                await self.request(im, "The movie.", reply)
                self.assertEqual(im.ledger.works["W1"].carrier, reply)
                self.assertEqual(next(iter(im.clarifications.values()))["state"], "consumed")
                history = str(adapter.runs[0].histories[0])
                for text in (*requests, "Which booking?", "The movie."):
                    self.assertIn(text, history)
                self.assertNotIn("Handled together", history)
                [created], [consumed] = (
                    self.traced("im.clarification.created"),
                    self.traced("im.clarification.consumed"),
                )
                self.assertEqual(created["clarification_id"], consumed["clarification_id"])
                self.assertEqual(created["delegation_ids"], sources)
                self.assertEqual((consumed["source_delegation_ids"], consumed["delegation_ids"]), (sources, [reply]))

    async def test_explicit_reclarification_supersedes_only_after_fresh_valid_placement(self):
        refused = []

        async def again(given, place):
            [old] = given.clarifications
            with self.assertRaises(ValueError) as error:
                await place(CallerMessage(("D1",), "question", "Again?", clarifications=(old["id"],)))
            refused.append(error.exception)
            self.assertEqual(im.clarifications[old["id"]]["state"], "waiting")
            for identities in (("unknown",), (old["id"], old["id"])):
                with self.assertRaises(ValueError) as error:
                    await place(CallerMessage(("D2",), "question", "Again?", clarifications=identities))
                refused.append(error.exception)
            self.assertIn("D2", im.arrivals)
            self.assertEqual(im.clarifications[old["id"]]["state"], "waiting")
            await place(CallerMessage(("D2",), "question", "The movie or the cafe?", clarifications=(old["id"],)))

        async def choose(given, place):
            [new] = given.clarifications
            [old_id] = new["supersedes"]
            with self.assertRaises(ValueError) as error:
                await place(Start(("D3",), (TaskSpec("selected movie"),), (old_id,)))
            refused.append(error.exception)
            self.assertIn("D3", im.arrivals)
            await place(Start(("D3",), (TaskSpec("selected movie"),), (new["id"],)))

        adapter = ScriptedAdapter(
            [answer()],
            max_running=2,
            conductor=planned(
                CallerMessage(("D1",), "question", "Which booking?", clarifications=()),
                again,
                choose,
            ),
        )
        im = self.session(adapter, max_pending_delegations=1)
        await self.request(im, "Change it to six.", "D1")
        [old] = im.clarifications.values()
        events = await self.request(im, "That one.", "D2")
        self.assertEqual(texts(events), ["The movie or the cafe?"])
        old, new = im.clarifications.values()
        self.assertEqual((old["state"], old["superseded_by"]), ("superseded", new["id"]))
        self.assertEqual(new["source_delegation_ids"], ["D1", "D2"])
        self.assertEqual(new["reply_delegation_ids"], ["D2"])
        self.assertFalse(im.works)
        await self.request(im, "The movie.", "D3")
        self.assertEqual(new["state"], "consumed")
        self.assertEqual(im.ledger.works["W1"].carrier, "D3")
        history = str(adapter.runs[0].histories[0])
        for text in ("Change it to six.", "That one.", "The movie.", "Which booking?", "The movie or the cafe?"):
            self.assertIn(text, history)
        self.assertEqual(len(refused), 4)
        self.assertTrue(all(isinstance(error, ValueError) for error in refused))

    async def test_repeated_clarification_prunes_closed_rows_without_waiting_for_a_work_to_end(self):
        async def reclarify(given, place):
            ids = tuple(row["id"] for row in given.clarifications)
            await place(CallerMessage((given.arrivals[0].delegation_id,), "question", "Which job?", clarifications=ids))

        adapter = ScriptedAdapter(conductor=ScriptedConductor(reclarify), max_running=2)
        im = self.session(adapter, max_pending_delegations=1)
        for number in range(36):
            await self.request(im, f"Unclear reply {number}.", f"D{number + 1}")
            self.assertLessEqual(len(im.clarifications), 33)
            self.assertEqual(sum(row["state"] == "waiting" for row in im.clarifications.values()), 1)
            self.assertFalse(im.waits_for_decision)
        self.assertEqual(sum(row["state"] == "superseded" for row in im.clarifications.values()), 32)
        self.assertFalse(im.works)
        self.assertEqual(len(self.traced("im.clarification.created")), 36)
        consumed = self.traced("im.clarification.consumed")
        self.assertEqual(len(consumed), 35)
        [current] = [row for row in im.clarifications.values() if row["state"] == "waiting"]
        self.assertNotIn(current["id"], {row["clarification_id"] for row in consumed})
        self.assertLessEqual(len(im.ledger.delegations), 32)
        self.assertLessEqual(len(im.ledger.responses), 32)
        self.assertNotIn("D1", im.ledger.delegations)
        with self.assertRaisesRegex(DelegationError, "already exists"):
            im.admit(ask("Reuse the old source."), delegation_id="D1")
        self.assertFalse(im.arrivals)

    async def test_zero_record_window_still_continues_at_one(self):
        adapter = ScriptedAdapter([answer()], [answer()])
        im = self.session(adapter, records_max_works=0)
        await self.request(im, "Coffee.", "D1")
        await self.request(im, "Large.", "D2")
        self.assertEqual(im.ledger.tasks["W2.1"].continues, "W1.1")


class ControlledAdapter(ScriptedAdapter):
    def can_run_without_attachment(self):
        return True


class AttachmentFenceTests(IMCase):
    async def test_a_reserved_worker_slot_does_not_override_a_stop_or_detach(self):
        for outcome in ("stop", "detach"):
            with self.subTest(outcome=outcome):
                adapter = ControlledAdapter([answer("The search finished.")])
                im = self.session(adapter)
                with patch.object(im, "_schedule", return_value=None):
                    admission = im.admit(ask("Find a table."), delegation_id="D1")
                    await self.until(lambda im=im: im.ready, "the queued worker")
                work = im.works["W1"]
                (task,) = work.tasks
                if outcome == "detach":
                    detached = asyncio.create_task(im.set_attachment(False))
                im._schedule()
                self.assertIn(task, im.running)
                self.assertTrue(task.grant.done())

                if outcome == "stop":
                    work.stop_soon(ForNewerRequest())
                    await read_all(admission.stream, timeout=2)
                    self.assertEqual(task.record.state, "cancelled")
                    self.assertFalse(adapter.runs)
                else:
                    await detached
                    await self.turns()
                    try:
                        self.assertFalse(adapter.runs)
                        self.assertEqual(task.record.state, "queued")
                        self.assertNotIn(task, im.running)
                    finally:
                        await im.set_attachment(True)
                    events = await read_all(admission.stream, timeout=2)
                    self.assertEqual(texts(events), ["The search finished."])
                    self.assertEqual(len(adapter.runs), 1)
                await im.close()

    async def test_late_attachment_activation_cannot_open_the_gate_for_a_newer_attempt(self):
        im = self.session(ControlledAdapter())
        first, second = asyncio.Event(), asyncio.Event()
        release_first, release_second = asyncio.Event(), asyncio.Event()
        attempts = iter(((first, release_first), (second, release_second)))

        async def set_conductor_attachment(attached):
            if attached:
                entered, release = next(attempts)
                entered.set()
                await release.wait()

        await im.set_attachment(False)
        with patch.object(im.conductor, "set_attachment", new=set_conductor_attachment, create=True):
            older = asyncio.create_task(im.set_attachment(True))
            await asyncio.wait_for(first.wait(), 2)
            await im.set_attachment(False)
            newer = asyncio.create_task(im.set_attachment(True))
            try:
                await asyncio.wait_for(second.wait(), 2)
                release_first.set()
                self.assertEqual(await older, {"attached": False})
                self.assertFalse(im.attachment_ready)
                self.assertFalse(im._attachment.is_set())
            finally:
                release_first.set()
                release_second.set()
                await asyncio.gather(older, newer)
        self.assertEqual(newer.result(), {"attached": True, "pending_calls": []})
        self.assertTrue(im.attachment_ready)

    async def test_disconnect_fences_new_app_calls_with_or_without_pending_input(self):
        for correction in (False, True):
            with self.subTest(correction=correction):
                self.traces.clear()
                await self.check_new_app_call_during_disconnect(correction=correction)

    async def check_new_app_call_during_disconnect(self, *, correction):
        from tests.unit.delegation_core.helpers import MENU, app_call, call_item, function_calls

        decide, finish = asyncio.Event(), asyncio.Event()

        async def later(given, place):
            await decide.wait()
            await place(Steer(("D2",), (WorkTarget("W1"),), clarifications=()))

        script = [wait(finish), checkpoint() if correction else app_call(), answer()]
        conductor = planned(Start(("D1",)), later) if correction else None
        adapter = ControlledAdapter(script, conductor=conductor, max_running=2 if correction else 1)
        im = self.session(adapter)
        first = im.admit(ask("Menu?"), delegation_id="D1", settings={"tools": [MENU]})
        await self.until(lambda: adapter.runs, "task")
        if correction:
            reply = im.admit(ask("Only decaf."), delegation_id="D2")
            await self.until(lambda: len(conductor.given) == 2, "pending decision")
        await im.set_attachment(False)
        task = adapter.runs[0].task
        if correction:
            self.assertEqual(task.route(call_item("get_menu", {}, "pending")), "refused")
            self.assertEqual(task.refusal("pending"), im.texts.not_run_output)
        else:
            finish.set()
            await self.until(lambda: self.traced("tools.refused"), "attachment refusal")
            self.assertEqual(self.traced("tools.refused")[-1]["reason"], "attachment")
            self.assertEqual(im.ledger.works["W1"].state, "running")
            self.assertFalse(im._attachment.is_set())
        await im.set_attachment(True)
        if correction:
            decide.set()
            await self.until(lambda: task.steering, "steering")
            finish.set()
            await read_all(reply.stream)
        self.assertEqual(function_calls(await read_all(first.stream)), [])
        self.assertEqual(len(adapter.runs), 1)

    async def test_unqualified_native_backend_stops_with_uncertainty(self):
        im, adapter = self.scripted_session([wait(asyncio.Event()), answer()])
        admission = im.admit(ask("Change the booking."), delegation_id="D1")
        await self.until(lambda: adapter.runs, "task")
        self.assertEqual(await im.set_attachment(False), {"attached": False})
        events = await read_all(admission.stream)
        self.assertEqual(im.ledger.works["W1"].state, "failed")
        self.assertIn(im.texts.owed_output, texts(events))
        self.assertEqual(adapter.runs[0].stops, 1)
        self.assertFalse(im.ledger.works["W1"].after_call)

    async def test_after_call_permission_does_not_reenable_caller_tools_while_away(self):
        from tests.unit.delegation_core.helpers import MARK, MENU, call_item

        adapter = ScriptedAdapter([MARK, wait(asyncio.Event()), answer()], max_running=2)
        im = self.session(adapter, continuable=True)
        im.admit(ask("Long comparison."), delegation_id="D1", settings={"tools": [MENU]})
        await self.until(
            lambda: im.ledger.works.get("W1") and im.ledger.works["W1"].after_call, "after-call permission"
        )
        await im.set_attachment(False)
        task = adapter.runs[0].task
        for name in ("get_menu", "caller_message"):
            self.assertEqual(task.route(call_item(name, {}, name)), "refused")
        self.assertEqual(task.route(call_item("backend_read", {}, "backend_read")), "backend")
        self.assertFalse(task.stop_reason)


class StopAndQuestionRaceTests(IMCase):
    async def test_dropped_unpublished_question_can_be_reissued_with_its_canonical_identity(self):
        from tests.unit.delegation_core.helpers import app_outputs
        from tests.unit.delegation_core.test_im_calls import CallTests

        go, finish = asyncio.Event(), asyncio.Event()
        correction = placing(Steer(("D2",), (WorkTarget("W1", ("W1.2",)),)))
        im, first = await CallTests.asking_while_the_app_works(
            self, go, correction, before_answer=(checkpoint(), wait(finish))
        )
        task = im.works["W1"].tasks[1]
        question = task.record.questions[0]
        self.assertEqual(question.dispatch, "pending")
        second = im.admit(ask("Also check pastries."), delegation_id="D2")
        await self.until(lambda: im.ledger.steerings, "reply waiting")
        go.set()
        await self.until(lambda: im.ledger.steerings["S1"].state == "applied", "reply landing")
        self.assertEqual((question.dispatch, question.resolution), ("dropped", "open"))
        metadata = json.loads(task.backend_state()[0]["content"][0]["text"])["private_backend_state"]
        self.assertEqual(metadata["questions"][0]["dispatch"], "dropped")
        receipt = await task.tools().execute(
            "reissue",
            "caller_message",
            json.dumps({"kind": "question", "message": QUESTION, "question_id": question.id}),
        )
        self.assertEqual(json.loads(receipt)["question_id"], question.id)
        self.assertEqual(len(task.record.questions), 1)
        with self.assertRaises(ValueError):
            await task.tools().execute(
                "changed",
                "caller_message",
                json.dumps({"kind": "question", "message": "Changed?", "question_id": question.id}),
            )
        older = im.admit_client_request(app_outputs(first), delegation_id="D1")
        finish.set()
        a, b = await asyncio.gather(read_all(older.stream), read_all(second.stream))
        self.assertEqual((texts(a) + texts(b)).count(QUESTION), 1)
        self.assertEqual(question.dispatch, "written")
