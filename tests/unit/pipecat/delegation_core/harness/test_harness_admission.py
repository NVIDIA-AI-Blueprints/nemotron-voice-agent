# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""First admission waits for placement; lifecycle changes never turn a waiting call into an orphan effect."""

import asyncio
import unittest
from types import SimpleNamespace

import httpx

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.harness.bridge import (
    OPERATION_SCOPE,
    McpBridge,
    Operation,
    WorkerDoor,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.harness.protocol import (
    RefusedReply,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.harness.settings import Bridge
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.harness.worker import (
    HarnessTurnError,
    WorkerRun,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import RefusalDetails
from tests.unit.pipecat.delegation_core.harness.doubles import FakeHost, FakeTaskContext, settings, until
from tests.unit.pipecat.delegation_core.harness.test_harness_bridge import McpClient, Tools
from tests.unit.pipecat.delegation_core.helpers import TEXTS, read_all


class BarrierTask(FakeTaskContext):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pending = True
        self.owned = False
        self.placed = asyncio.Event()
        self.routed = []
        self.waiting = 0

    def placement_pending(self, call):
        return self.pending and not self.owned

    async def wait_for_placement(self):
        self.waiting += 1
        try:
            await self.placed.wait()
        finally:
            self.waiting -= 1

    def route(self, call):
        self.routed.append(call)
        return "refused" if self.owned else super().route(call)

    def refusal_details(self, call_id):
        if not self.owned:
            raise KeyError(call_id)
        return RefusalDetails(self.work_id, self.task_id, call_id, 2, "not_run", "owned", True)

    def foreign(self):
        self.pending = False
        self.placed.set()


class DoorAdmissionTests(unittest.IsolatedAsyncioTestCase):
    def door(self):
        task = BarrierTask()
        door = WorkerDoor("w1", task, (Tools(),), asyncio.Queue(), tool_timeout_seconds=1)
        self.addAsyncCleanup(self.close, door)
        return task, door

    async def close(self, door):
        door.retire(TEXTS.stopped_output)
        drain = getattr(door, "drain_admissions", None)
        if drain is not None:
            await drain()

    async def held_backend(self, task, door):
        call = asyncio.create_task(door.call("lookup", {}))
        self.addAsyncCleanup(self.join, call)
        await until(lambda: task.waiting or task.routed)
        self.assertEqual(task.routed, [], "The call must remain unadmitted, not receive a sticky refusal.")
        self.assertEqual(task.started, [])
        return call

    async def join(self, task):
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def test_foreign_placement_runs_the_original_backend_call_once(self):
        task, door = self.door()
        call = await self.held_backend(task, door)
        self.assertFalse(door.serves(FakeTaskContext(), (Tools(),)))
        self.assertFalse(door.rebind(FakeTaskContext(), (Tools(),)))
        task.foreign()
        self.assertEqual(await call, '{"found": "lookup"}')
        self.assertEqual(len(task.routed), 1)
        self.assertEqual(task.started[0][0], task.routed[0]["call_id"])
        self.assertEqual(len(task.outcomes), 1)
        self.assertEqual((door.admitting, task.waiting), (set(), 0))

    async def test_noncallable_required_hooks_do_not_silently_bypass_admission(self):
        for field in ("placement_pending", "wait_for_placement"):
            with self.subTest(field=field):
                task, door = self.door()
                setattr(task, field, True)
                with self.assertRaises(TypeError):
                    await door.call("lookup", {})
                self.assertEqual(len(task.routed), 0)
                self.assertEqual(task.waiting, 0)

    async def test_owned_placement_wakes_before_other_global_placement_finishes(self):
        task, door = self.door()
        call = await self.held_backend(task, door)
        task.owned = True
        door.admission_changed()
        reply = await asyncio.wait_for(call, 1)
        self.assertIsInstance(reply, RefusedReply)
        self.assertEqual(reply.details.input_scope, "owned")
        self.assertFalse(task.placed.is_set())
        self.assertEqual((len(task.routed), task.started, task.waiting), (1, [], 0))

    async def test_withdrawn_backend_request_never_executes(self):
        for retired in (False, True):
            with self.subTest(retired=retired):
                task, door = self.door()
                call = await self.held_backend(task, door)
                if retired:
                    door.retire("Stopped before admission.")
                    await door.drain_admissions()
                    self.assertEqual(await call, "Stopped before admission.")
                else:
                    call.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await call
                task.foreign()
                await asyncio.sleep(0)
                self.assertEqual((task.routed, task.started, door.admitting, task.waiting), ([], [], set(), 0))

    async def test_rebound_identity_is_rechecked_before_routing(self):
        task, door = self.door()
        call = await self.held_backend(task, door)
        later = FakeTaskContext(work_id="W2", task_id="W2.1")
        # The public rebind above refuses this; also fence stale callbacks if the owner identity changes.
        door.task = later
        door.admission_changed()
        await call
        self.assertEqual((task.routed, task.started, later.started, task.waiting), ([], [], [], 0))

    async def test_new_global_placement_is_rechecked_after_wakeup(self):
        task, door = self.door()
        call = await self.held_backend(task, door)
        waiting_again = asyncio.Event()
        wait_for_placement = task.wait_for_placement

        async def next_placement():
            waiting_again.set()
            await wait_for_placement()

        task.wait_for_placement = next_placement
        task.placed.set()
        task.placed = asyncio.Event()
        door.admission_changed()
        await asyncio.wait_for(waiting_again.wait(), 1)
        self.assertFalse(call.done())
        self.assertEqual((task.routed, task.started), ([], []))
        task.foreign()
        self.assertEqual(await call, '{"found": "lookup"}')
        self.assertEqual(len(task.routed), 1)


class McpAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bridge = McpBridge(Bridge(bind="127.0.0.1:0"))
        await self.bridge.start()
        self.addAsyncCleanup(self.bridge.aclose)
        self.task = BarrierTask()
        self.door = WorkerDoor("w1", self.task, (Tools(),), asyncio.Queue(), tool_timeout_seconds=1)
        server = self.bridge.open(self.door)
        self.client = McpClient(server.url, server.token)
        self.http = httpx.AsyncClient(timeout=2)
        self.addAsyncCleanup(self.http.aclose)
        await self.client.connect(self.http)

    async def test_actual_mcp_result_waits_without_negative_reply_then_executes_once(self):
        call = asyncio.create_task(self.client.call(self.http, "lookup", {}))
        await until(lambda: self.task.waiting or self.task.routed)
        self.assertEqual(self.task.routed, [])
        self.assertFalse(call.done())
        self.task.foreign()
        result = await call
        self.assertEqual(result, {"content": [{"type": "text", "text": '{"found": "lookup"}'}], "isError": False})
        self.assertEqual((len(self.task.routed), len(self.task.started), self.task.waiting), (1, 1, 0))

    async def test_cancelled_http_waiter_leaves_the_admitted_operation_owned(self):
        context = SimpleNamespace(
            request=SimpleNamespace(
                headers={"authorization": f"Bearer {self.door.token}"},
                scope={OPERATION_SCOPE: Operation.capture(self.door)},
            )
        )
        params = SimpleNamespace(name="lookup", arguments={})
        call = asyncio.create_task(self.bridge._call_tool(context, params))
        await until(lambda: self.task.waiting or self.task.routed)
        self.assertEqual(self.task.routed, [])
        call.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await call
        self.task.foreign()
        await until(lambda: not self.door.pending_calls)
        self.assertEqual((len(self.task.routed), len(self.task.started), self.task.waiting), (1, 1, 0))
        self.assertEqual(self.door.admitting, set())


class WorkerAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def open(self, *, seconds=1):
        self.task = BarrierTask()
        self.host = FakeHost(settings(turn_timeout_seconds=seconds), tools=(Tools(),))
        self.run = WorkerRun(self.task, self.host)
        self.events = self.run.events()
        await anext(self.events)
        self.native = self.host.runtime.client.sessions[0]
        self.addAsyncCleanup(self.cleanup)

    async def cleanup(self):
        if 1 in self.native.outstanding:
            self.native.end(1, "cancelled", {"totalTokens": 1})
        if self.run.door is not None:
            self.run.door.retire(TEXTS.stopped_output)
        if hasattr(self, "reader"):
            if not self.reader.done():
                self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)
        await self.events.aclose()
        await self.run.release()

    async def app_wait(self):
        # The native request arrives before placement; the actual first app route happens on the reader later.
        self.task.pending = False
        call = asyncio.create_task(self.run.door.call("get_menu", {}))
        await until(lambda: self.run.door.held)
        self.task.pending = True
        self.reader = asyncio.create_task(read_all(self.events))
        await until(lambda: self.task.waiting or self.task.routed)
        self.assertEqual(self.task.routed, [], "App enqueue is not its first route: check again at that route.")
        return call

    async def test_app_enqueue_race_waits_then_forwards_the_original_call_once(self):
        await self.open()
        call = await self.app_wait()
        original = self.run.door.held[0]
        self.task.foreign()
        await until(lambda: self.run.foreground.forwarded)
        self.assertIs(self.run.foreground.forwarded[0], original)
        self.assertEqual([item["call_id"] for item in self.task.routed], [original.call_id])
        await self.reader
        self.task.give_outputs([{"type": "function_call_output", "call_id": original.call_id, "output": "Tea."}])
        await self.run.resume()
        self.assertEqual(await call, "Tea.")
        self.assertEqual(self.native.cancels, 0)

    async def test_cancelled_app_is_removed_without_first_route_or_late_forwarding(self):
        await self.open()
        call = await self.app_wait()
        call.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await call
        await until(lambda: not self.run.foreground.app_queue and not self.run.door.app_admissions)
        self.task.foreign()
        await asyncio.sleep(0)
        self.assertEqual(
            (self.task.routed, self.run.foreground.forwarded, self.task.waiting, self.native.cancels), ([], [], 0, 0)
        )

    async def test_door_admission_fence_alone_releases_queued_app_without_dispatch(self):
        await self.open()
        call = await self.app_wait()
        self.run.door.stop_admissions(TEXTS.stopped_output)
        self.assertEqual(await asyncio.wait_for(call, 1), TEXTS.stopped_output)
        await until(lambda: not self.run.door.app_admissions)
        self.task.foreign()
        await asyncio.sleep(0)
        self.assertFalse(self.run.stopped)
        self.assertFalse(self.run.door.retired)
        assert (self.task.routed, self.task.started, self.run.foreground.forwarded, self.native.cancels) == (
            [],
            [],
            [],
            0,
        )

    async def test_stop_fences_an_app_wait_without_waiting_for_global_placement(self):
        await self.open()
        call = await self.app_wait()
        stopped = asyncio.create_task(self.run.stop())
        await until(lambda: self.native.cancels)
        self.native.end(1, "cancelled", {"totalTokens": 1})
        self.assertEqual(await stopped, "stopped")
        self.assertEqual(await call, TEXTS.stopped_output)
        self.assertEqual((self.task.routed, self.task.waiting, self.run.door.app_admissions), ([], 0, {}))
        self.assertFalse(self.task.placed.is_set())

    async def test_native_failure_still_ends_a_placement_paused_run_and_joins_admission(self):
        await self.open()
        self.reader = asyncio.create_task(read_all(self.events))
        call = asyncio.create_task(self.run.door.call("lookup", {}))
        await until(lambda: self.task.waiting)
        self.native.end(1, "refusal", {"totalTokens": 10})
        with self.assertRaisesRegex(HarnessTurnError, "ended the turn refusal"):
            await self.reader
        self.assertEqual(await call, TEXTS.stopped_output)
        self.assertEqual((self.task.routed, self.task.started, self.task.waiting), ([], [], 0))
        self.assertFalse(self.task.placed.is_set())
        self.assertTrue(self.native.retired)

    async def test_global_wait_preserves_reading_budget_then_budget_resumes(self):
        await self.open(seconds=0.08)
        self.reader = asyncio.create_task(read_all(self.events))
        calls = [asyncio.create_task(self.run.door.call("lookup", {})) for _ in range(2)]
        await until(lambda: self.task.waiting == 2 or self.task.routed)
        self.assertEqual(self.task.routed, [])
        await asyncio.sleep(0.15)
        self.assertFalse(self.reader.done(), "Global-placement waiting used to be outside the reading budget.")
        self.assertEqual(self.native.cancels, 0)
        self.assertLess(self.run.foreground.turn.used, 0.04)
        self.task.foreign()
        self.assertEqual(await asyncio.gather(*calls), ['{"found": "lookup"}'] * 2)
        with self.assertRaisesRegex(HarnessTurnError, "ran past"):
            await asyncio.wait_for(self.reader, 0.4)
        self.assertEqual((len(self.task.routed), self.native.cancels, self.task.waiting), (2, 1, 0))
