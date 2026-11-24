# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101,D102,D103,D107

"""A stop retires tool ownership without claiming an app handoff never ran."""

import asyncio

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.harness.protocol import (
    RefusedReply,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import RefusalDetails
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.stop import ForNewerRequest
from tests.unit.pipecat.delegation_core.harness.doubles import FakeHost, settings, until
from tests.unit.pipecat.delegation_core.harness.test_harness_worker import Harness, Runs
from tests.unit.pipecat.delegation_core.helpers import IMCase, app_outputs, ask, items_of, read_all


class ForwardedStopOutcome(Harness):
    async def queued(self, **options):
        session = await self.open(**options)
        held = asyncio.create_task(self.run.door.call("get_menu", {}))
        await until(lambda: bool(self.run.door.held))
        return session, held

    async def forwarded(self, **options):
        session, held = await self.queued(**options)
        [call] = items_of(await self.rest(), "function_call")
        self.assertEqual(self.run.foreground.forwarded[0].call_id, call["call_id"])
        return session, held, call

    async def stopped(self, session):
        ending = asyncio.create_task(self.run.stop())
        await until(lambda: session.cancels == 1)
        session.end(1, "cancelled")
        await self.events.aclose()
        self.assertEqual(await ending, "stopped")

    async def test_release_preserves_uncertainty_of_a_forwarded_call(self):
        session, held, _ = await self.forwarded()
        await self.run.release()
        session.end(1, "cancelled")
        await self.events.aclose()
        self.assertEqual(await held, self.task.texts.owed_output)

    async def test_mixed_forwarded_and_queued_calls_keep_distinct_outcomes(self):
        session, first, original = await self.forwarded()
        second = asyncio.create_task(self.run.door.call("get_menu", {}))
        await until(lambda: len(self.run.door.held) == 2)
        self.assertEqual([held.call_id for held in self.run.foreground.forwarded], [original["call_id"]])
        await self.stopped(session)
        self.assertEqual(await first, self.task.texts.owed_output)
        self.assertEqual(await second, self.task.texts.stopped_output)
        self.assertEqual(len(items_of(self.seen, "function_call")), 1)

    async def test_committed_app_outputs_survive_stop_with_and_without_resume(self):
        outputs = ("Accepted once.", "Error: rejected by the app.", "", "Not run: this work was stopped.")
        for resumed in (False, True):
            for output in outputs:
                with self.subTest(resumed=resumed, output=output):
                    await self.check_committed_output(output, resumed=resumed)

    async def check_committed_output(self, output, *, resumed):
        session, held, call = await self.forwarded()
        self.task.give_outputs([{"type": "function_call_output", "call_id": call["call_id"], "output": output}])
        if resumed:
            await self.run.resume()
            self.assertEqual(await held, output)
            await self.stopped(session)
            self.assertEqual(await held, output)
            return

        pending = self.task.steer_with([{"id": "D2", "type": "message", "role": "user", "content": "New detail."}])
        original_settle = self.task.settle

        def settle(steering, delivery):
            original_settle(steering, delivery)
            if delivery == "not_applied":
                self.task._steering.appendleft(steering)

        self.task.settle = settle
        await self.stopped(session)
        self.assertEqual(await held, output)
        self.assertEqual(self.task.settled, [(pending.id, "not_applied")])
        self.assertEqual(list(self.task._steering), [pending])
        self.assertFalse(any(item.get("id") == "D2" for item in self.task.items))
        self.assertEqual(len(session.sent), 1)
        self.assertEqual(self.run.foreground.next_prompts, [])
        await self.run.release()
        self.assertEqual(self.task.settled, [(pending.id, "not_applied")])

    async def test_existing_bound_refusal_keeps_its_original_reply_and_metadata(self):
        session, held, call = await self.forwarded()
        original = "The earlier call returned: Accepted once."
        details = RefusalDetails(self.task.work_id, self.task.task_id, call["call_id"], 7, "repeated", "none", True)
        self.task.refusal_details = lambda call_id: details if call_id == call["call_id"] else None
        self.task.refusal = lambda call_id: original
        await self.stopped(session)
        reply = await held
        self.assertIsInstance(reply, RefusedReply)
        self.assertEqual((reply.text, reply.details), (original, details))

    async def test_repeated_concurrent_stop_does_not_replace_first_reply_or_cancel_twice(self):
        session, held, _call = await self.forwarded()
        [forwarded] = self.run.foreground.forwarded
        first, second = asyncio.create_task(self.run.stop()), asyncio.create_task(self.run.stop())
        await until(lambda: session.cancels == 1)
        session.end(1, "cancelled")
        self.assertEqual(await asyncio.gather(first, second), ["stopped", "stopped"])
        self.assertEqual(await self.run.stop(), "stopped")
        forwarded.reply("Late result: Accepted once.")
        self.assertEqual(await held, self.task.texts.owed_output)
        self.assertEqual(session.cancels, 1)
        await self.events.aclose()

    async def test_boundary_failure_still_answers_forwarded_and_retires_queued_calls(self):
        for stage in ("boundary", "bridge"):
            with self.subTest(stage=stage):
                await self.check_close_error(stage)

    async def check_close_error(self, stage):
        from unittest.mock import patch

        session, first, _ = await self.forwarded()
        door = self.run.door
        second = None
        if stage == "boundary":
            second = asyncio.create_task(door.call("get_menu", {}))
            await until(lambda: len(door.held) == 2)
        [forwarded] = self.run.foreground.forwarded
        target, method = (self.task, "take_boundary_input") if stage == "boundary" else (self.host.bridge, "close")
        failure = RuntimeError(f"{stage} failed")
        session.end(1, "cancelled")
        try:
            with patch.object(target, method, side_effect=failure), self.assertRaises(RuntimeError) as caught:
                await self.run.stop()
            self.assertIs(caught.exception, failure)
            self.assertTrue(forwarded.answer.done())
            if second is not None:
                self.assertTrue(door.retired)
        finally:
            await self.run.release()
            await self.events.aclose()
        self.assertEqual(await first, self.task.texts.owed_output)
        if second is not None:
            self.assertEqual(await second, self.task.texts.stopped_output)


class PendingCoreOutput(IMCase):
    async def test_real_im_output_arriving_before_slot_resume_survives_stop(self):
        host = FakeHost(settings())
        runs = Runs(host)
        im = self.session(runs)
        first = im.admit(
            ask("Read the menu."),
            delegation_id="D1",
            settings={
                "tools": [{"type": "function", "name": "get_menu", "parameters": {"type": "object", "properties": {}}}]
            },
        )
        await self.until(lambda: runs.runs and runs.runs[0].door is not None, "native double opened")
        run = runs.runs[0]
        native = host.runtime.client.sessions[-1]
        held = asyncio.create_task(run.door.call("get_menu", {}))
        first_events = await read_all(first.stream)
        [call] = items_of(first_events, "function_call")
        task = im.works["W1"].tasks[0]
        # Admission gives the actual TaskRun its output synchronously. Stop before its slot/resume coroutine runs.
        rest = im.admit_client_request(app_outputs(first_events, "Accepted before slot."), delegation_id="D1")
        self.assertEqual(task.outputs[call["call_id"]]["output"], "Accepted before slot.")
        asked = task.work.ask_stop(ForNewerRequest())
        stopping = asyncio.create_task(task.work.confirm_stop(asked))
        await self.until(lambda: native.cancels == 1, "stop requested")
        native.end(1, "cancelled")
        await stopping
        await read_all(rest.stream)
        self.assertEqual(await held, "Accepted before slot.")
        self.assertEqual(task.outputs, {})
        self.assertTrue(
            any(
                item.get("call_id") == call["call_id"] and item.get("output") == "Accepted before slot."
                for item in task.thread_items
            )
        )
        self.assertEqual(len(native.sent), 1)
        self.assertEqual(task.input_revision, 1)
