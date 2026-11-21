# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Terminal task ownership must not turn historical questions into unrelated tasks' locks."""

import asyncio
import json

from examples.frontend_backend_live.delegation_core.backend import port
from examples.frontend_backend_live.delegation_core.backend.llm import LlmSettings
from tests.unit.delegation_core import helpers


class QuestionRetirementTests(helpers.IMCase):
    async def test_default_llm_can_ask_on_a_later_request_with_or_without_explicit_resolution(self):
        for resolves in (False, True):
            with self.subTest(explicit_resolution=resolves):
                self.traces.clear()
                held = asyncio.Event()
                rounds = [
                    helpers.call_round("caller_message", {"kind": "question", "message": helpers.QUESTION}, "ask_old"),
                    helpers.gated(held, helpers.text_round("Before the caller's reply.")),
                    helpers.text_round("A latte it is."),
                    helpers.call_round(
                        "caller_message", {"kind": "question", "message": "For how many people?"}, "ask_new"
                    ),
                    helpers.text_round("The table request is ready for its party size."),
                ]
                if resolves:
                    rounds.insert(
                        2,
                        helpers.call_round(
                            "resolve_questions",
                            {
                                "resolutions": [
                                    {"question_id": "W1.1.Q1", "disposition": "answered", "input_ref": "W1.1.I2"}
                                ]
                            },
                            "resolve_old",
                        ),
                    )
                provider = helpers.ScriptedProvider(*rounds)
                backend = helpers.adapter(provider, settings=LlmSettings())
                self.assertEqual(backend.max_running, 1)
                im = self.session(backend)
                first = im.admit(helpers.ask("A coffee please."), delegation_id="D1")
                await self.until(lambda provider=provider: provider.opened == 2, "the original task's next model round")
                second = im.admit(helpers.ask("A latte."), delegation_id="D2")
                earlier, updated = await asyncio.gather(helpers.read_all(first.stream), helpers.read_all(second.stream))
                self.assertEqual(helpers.texts(updated), ["A latte it is."])
                self.assertIn(helpers.QUESTION, helpers.texts(earlier))
                self.assertIn(im.texts.handover_line.format(delegation="D2"), helpers.texts(earlier))
                events = await self.request(im, "Also book a table tonight.", "D3")
                self.assertIn("For how many people?", helpers.texts(events))
                self.assertEqual(len(self.traced("im.question.written")), 2)
                old, new = im.ledger.tasks["W1.1"], im.ledger.tasks["W2.1"]
                self.assertEqual(old.questions[0].resolution, "answered" if resolves else "open")
                self.assertEqual(old.questions[0].open, not resolves)
                if not resolves:
                    self.assertIs(old.questions[0], new.questions[0])
                    self.assertEqual(old.questions[0].owner_task_id, new.id)
                self.assertEqual(new.questions[-1].text, "For how many people?")
                self.assertEqual(new.questions[-1].creator_task_id, new.id)
                self.assertNotIn("owner_active", json.dumps(events))

    async def test_stop_closes_only_the_selected_ended_threads_questions_and_prevents_continuation(self):
        target = port.WorkTarget("W1", ("W1.1",))
        stop = port.Cancel(("D2",), (target,))

        async def stop_coffee(_given, place):
            conflicting = port.Start(("D2",), (port.TaskSpec("coffee", "W1.1"),))
            with self.assertRaisesRegex(port.DecisionRejected, "continued and stopped"):
                await place(port.CompleteDecision((stop, conflicting)))
            self.assertTrue(coffee.questions[0].open)
            await place(port.CompleteDecision((stop,)))

        conductor = helpers.planned(
            port.Start(("D1",), (port.TaskSpec("coffee"), port.TaskSpec("movie"))),
            stop_coffee,
            port.Steer(("D3",), (target,)),
            port.Steer(("D3",), (target,)),
            port.Start(("D4",), (port.TaskSpec("coffee", "W1.1"),)),
            port.Start(("D4",), (port.TaskSpec("coffee", "W1.1"),)),
        )
        im, backend = self.scripted_session(
            [helpers.asking(), helpers.answer("Coffee choices checked.")],
            [helpers.asking("Which movie?"), helpers.answer("Movies checked.")],
            conductor=conductor,
            max_running=2,
        )
        await self.request(im, "Coffee and a movie.", "D1")
        coffee, movie = (im.ledger.tasks[identity] for identity in ("W1.1", "W1.2"))
        self.assertEqual((coffee.state, movie.state), ("answered", "answered"))
        self.assertTrue(coffee.questions[0].open)
        self.assertTrue(movie.questions[0].open)

        stopped = await self.request(im, "Stop the coffee request.", "D2")
        action = next(row for row in im.ledger.controls.values() if row.sources == ("D2",))
        self.assertEqual(action.receipt["outcome"], "already_ended")
        self.assertEqual(action.receipt["closed_questions"], {coffee.id: [coffee.questions[0].id]})
        self.assertEqual(helpers.texts(stopped), [im.texts.control_requested_line])
        assert (coffee.state, coffee.stop.state, coffee.questions[0].resolution) == ("answered", "none", "cancelled")
        self.assertTrue(movie.questions[0].open)
        self.assertEqual(im.task_steering_state(movie), "continuable")
        self.assertEqual(im.task_steering_state(coffee), "unavailable")

        for source in ("D3", "D4"):
            events = await self.request(im, "A large one.", source)
            self.assertFalse(any(row.sources == (source,) for row in im.ledger.controls.values()))
            self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(len(backend.runs), 2)
        self.assertIsNone(coffee.successor_task_id)

    async def test_normal_end_drops_unpublished_question_without_resolving_it(self):
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled):
                await self.check_unpublished_question(cancelled)

    async def check_unpublished_question(self, cancelled):
        from tests.unit.delegation_core.test_im_calls import CallTests

        finish = asyncio.Event()
        successor = helpers.placing(port.Start(("D2",), (port.TaskSpec("coffee", "W1.2"),)))
        plans = () if cancelled else (successor,)
        im, first = await CallTests.asking_while_the_app_works(self, finish, *plans)
        task = im.works["W1"].tasks[1]
        question = task.record.questions[0]
        self.assertEqual(question.dispatch, "pending")
        status_args = '{"kind":"status","message":"Checking the order."}'
        status = await task.tools().execute("held_status", "caller_message", status_args)
        self.assertEqual(json.loads(status)["publication"], "pending")
        if cancelled:
            task.cancel_own()
        finish.set()
        await self.until(lambda: task.ended, "the asking task ended")
        self.assertEqual(question.dispatch, "dropped")
        self.assertEqual(question.resolution, "cancelled" if cancelled else "open")
        self.assertEqual(task.publications["held_status"]["state"], "dropped")
        continuation = im.admit_client_request(helpers.app_outputs(first), delegation_id="D1")
        events = await helpers.read_all(continuation.stream)
        self.assertNotIn(helpers.QUESTION, helpers.texts(events))
        self.assertNotIn("Checking the order.", helpers.texts(events))
        self.assertEqual(question.open, not cancelled)
        if cancelled:
            return
        self.assertEqual(question.owner_task_id, task.task_id)
        await self.request(im, "Continue the coffee.", "D2")
        state = json.loads(im.backend.runs[-1].task.backend_state()[0]["content"][0]["text"])["private_backend_state"]
        received = {row["call_id"]: row for row in state["publications"]}
        self.assertEqual(received["held_status"]["state"], "dropped")
        self.assertEqual(received["held_status"]["task_id"], task.task_id)
        self.assertEqual(received[question.publication_call_id]["state"], "dropped")
        self.assertEqual(await task.tools().execute("held_status", "caller_message", status_args), status)
