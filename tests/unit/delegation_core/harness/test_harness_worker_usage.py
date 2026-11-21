# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""A started native prompt remains measurable when stop/drop prevents its terminal from being read."""

import asyncio
import unittest
from unittest import mock

from examples.frontend_backend_live.delegation_core.backend.harness.protocol import PromptEnded
from examples.frontend_backend_live.delegation_core.backend.harness.worker import WorkerRun
from tests.unit.delegation_core.harness.doubles import FakeHost, FakeTaskContext, settings, until
from tests.unit.delegation_core.helpers import finished_items, read_all

USAGE = {"inputTokens": 80, "outputTokens": 20, "totalTokens": 100}


class WorkerUsageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.host = FakeHost(settings())
        self.task = FakeTaskContext()
        self.run = WorkerRun(self.task, self.host)

    async def open(self):
        events = self.run.events()
        self.assertEqual((await anext(events))["type"], "response.created")
        return events, self.host.runtime.client.sessions[-1]

    async def stopped(self, native):
        stopping = asyncio.create_task(self.run.stop())
        await until(lambda: native.cancels)
        native.end(native.last_prompt(), stop="cancelled", usage=USAGE)
        self.assertEqual(await stopping, "stopped")

    def usage(self):
        return self.task.traced("harness.round.usage")

    async def test_stopped_prompts_keep_one_unknown_usage_record(self):
        for stage in ("before_response", "first_response", "reused_session"):
            with self.subTest(stage=stage):
                self.setUp()
                await self.check_stopped_prompt(stage)

    async def check_stopped_prompt(self, stage):
        if stage == "before_response":
            await self.run._open()
            native = self.host.runtime.client.sessions[-1]
            self.assertEqual(self.run.round, 0)
            events = None
        else:
            events, native = await self.open()
        if stage == "reused_session":
            native.answer(1, "first-answer", "Done.", usage=USAGE)
            history = [*self.task.history(), *finished_items(await read_all(events))]
            history.append({"id": "D2", "role": "user", "content": "One more."})
            self.task = FakeTaskContext(history=history, work_id="W2", task_id="W2.1", continues="W1")
            self.run = WorkerRun(self.task, self.host)
            events, reused = await self.open()
            self.assertIs(reused, native)
            self.assertEqual((self.run.turn.total_before, self.run.turn.usage_before), (100, USAGE))

        prompt = 2 if stage == "reused_session" else 1
        await self.stopped(native)
        [row] = self.usage()
        self.assertEqual((row["prompt"], row["session"]), (prompt, native.id))
        if stage == "before_response":
            self.assertEqual(row["round"], 0)
        self.assertEqual((row["ending"], row["native_terminal_observed"]), ("stop_requested", False))
        self.assertTrue(row["usage_missing"])
        self.assertIsNone(row["reported_counters"])

        self.run.prompt_ended(PromptEnded(prompt, "cancelled", USAGE))
        self.run._turn_usage(self.run.turn)
        if events is not None:
            await events.aclose()
        self.assertEqual(self.host.runtime.places, 0)
        await self.run.release()
        await self.run.stop()
        self.assertEqual(self.usage(), [row])
        self.assertIsNone(row["reported_counters"])

    async def test_no_native_usage_before_a_prompt_is_admitted(self):
        for failed_open in (False, True):
            with self.subTest(failed_open=failed_open):
                self.setUp()
                if failed_open:
                    client = self.host.runtime.client
                    client.new_session = mock.AsyncMock(side_effect=RuntimeError("startup failed"))
                    with self.assertRaisesRegex(RuntimeError, "startup failed"):
                        await anext(self.run.events())
                else:
                    self.assertEqual(await self.run.stop(), "stopped")
                self.assertEqual(self.usage(), [])
                self.assertIsNone(self.run.turn)
                self.assertEqual(self.host.runtime.client.sessions, [])
                self.assertEqual(self.host.runtime.places, 0)

    async def test_reader_failure_after_prompt_keeps_an_unknown_observation(self):
        events, native = await self.open()
        with (
            mock.patch.object(self.run, "_next", side_effect=RuntimeError("reader failed")),
            self.assertRaisesRegex(RuntimeError, "reader failed"),
        ):
            await anext(events)
        [row] = self.usage()
        self.assertIsNone(row["reported_counters"])
        self.assertEqual(row["ending"], "incomplete")
        native.end(1, stop="cancelled", usage=USAGE)
        await self.run.release()
        self.assertEqual(len(self.usage()), 1)

    async def test_metadata_lookup_and_trace_failures_do_not_break_physical_cleanup(self):
        for broken_trace in (False, True):
            with self.subTest(broken_trace=broken_trace):
                self.setUp()
                events, native = await self.open()
                self.host.usage_identity = mock.Mock(side_effect=RuntimeError("metadata lookup failed"))
                trace = self.task.trace

                def observed(kind, broken=broken_trace, record=trace, **fields):
                    if kind == "harness.round.usage" and broken:
                        raise RuntimeError("observer failed")
                    record(kind, **fields)

                self.task.trace = observed
                await self.stopped(native)
                await events.aclose()
                self.assertEqual(self.host.runtime.places, 0)
                self.assertTrue(native.retired)
                self.assertEqual(len(self.usage()), 0 if broken_trace else 1)
