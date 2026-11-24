# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Own cancellation uses the shared immediate fence and retains uncertain native ownership."""

from tests.unit.pipecat.delegation_core.harness.doubles import FakeHost, settings
from tests.unit.pipecat.delegation_core.harness.test_harness_worker import Runs
from tests.unit.pipecat.delegation_core.helpers import MENU, IMCase, answers, ask, call_item, read_all


class OwnCancelBoundaryTests(IMCase):
    async def started(self):
        self.host = FakeHost(settings())
        self.runs = Runs(self.host)
        im = self.session(self.runs)
        first = im.admit(ask("Check the options."), delegation_id="D1", settings={"tools": [MENU]})
        await self.until(lambda: self.runs.runs and self.runs.runs[0].door is not None, "native opened")
        run = self.runs.runs[0]
        return im, first, run, run.session

    async def stop_own(self, run):
        await run.door.call("cancel_work", {})
        await self.until(lambda: run.stopped and run.session.cancels, "immediate native stop requested")

    async def test_own_cancel_fences_calls_and_requests_native_stop_without_a_report(self):
        im, first, run, native = await self.started()
        await self.stop_own(run)
        self.assertIsNotNone(run.task.stop_reason)
        self.assertEqual(run.task.route(call_item("get_menu", {}, "late-call")), "refused")
        await run.door.call("cancel_work", {})
        self.assertEqual((native.cancels, len(native.texts("prompt"))), (1, 1))
        self.assertTrue(native.outstanding)
        self.assertEqual((self.host.runtime.places, self.host.pool.idle), (1, []))
        native.say("stale", "An obsolete report must stay private.")
        native.end(1, "cancelled")
        events = await read_all(first.stream)
        self.assertNotIn("An obsolete report must stay private.", answers(events))
        self.assertEqual(im.ledger.tasks[run.task.task_id].state, "cancelled")
        await self.until(lambda: self.host.runtime.places == 0, "confirmed native place released")

        later = im.admit(ask("Find a different option."), delegation_id="D2", settings={"tools": [MENU]})
        await self.until(lambda: len(self.runs.runs) == 2 and self.runs.runs[-1].session is not None, "new task opened")
        fresh = self.runs.runs[-1]
        self.assertIsNot(fresh.session, native)
        self.assertNotEqual(fresh.task.work_id, run.task.work_id)
        fresh.session.answer(1, "fresh", "The different option is available.")
        self.assertEqual(answers(await read_all(later.stream)), ["The different option is available."])
