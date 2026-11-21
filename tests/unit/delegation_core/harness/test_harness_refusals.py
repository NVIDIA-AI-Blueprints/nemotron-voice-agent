# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Native refusal control belongs to the runtime; business output remains opaque data."""

import asyncio
import json
import unittest

import httpx
import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from examples.frontend_backend_live.delegation_core.backend import port
from examples.frontend_backend_live.delegation_core.backend.harness.bridge import (
    REFUSAL_META,
    McpBridge,
    WorkerDoor,
)
from examples.frontend_backend_live.delegation_core.backend.harness.protocol import (
    AppCall,
    MessageChunk,
    RefusedReply,
    ToolCard,
    ToolCardEnd,
    refused_reply,
)
from examples.frontend_backend_live.delegation_core.backend.harness.settings import Bridge
from examples.frontend_backend_live.delegation_core.backend.harness.worker import WorkerRun
from examples.frontend_backend_live.delegation_core.im.stop import AtClose, ForNewerRequest
from tests.unit.delegation_core import helpers
from tests.unit.delegation_core.harness.doubles import FakeHost, cooperative, until
from tests.unit.delegation_core.harness.doubles import FakeTaskContext as NativeTaskContext
from tests.unit.delegation_core.harness.test_harness_bridge import McpClient, Tools
from tests.unit.delegation_core.harness.test_harness_worker import Runs, close_events


class RefusingTask(helpers.FakeTaskContext):
    scope = "owned"
    reason = "not_run"

    def route(self, call):
        self.last_call = call
        return "refused"

    def refusal(self, call_id):
        return helpers.TEXTS.not_run_output

    def refusal_details(self, call_id):
        return port.RefusalDetails(self.work_id, self.task_id, call_id, 1, self.reason, self.scope, True)


class AttachmentSafeRuns(Runs):
    def can_run_without_attachment(self):
        return True


class RefusalWireTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bridge = McpBridge(Bridge(bind="127.0.0.1:0"))
        await self.bridge.start()
        self.addAsyncCleanup(self.bridge.aclose)
        self.http = httpx.AsyncClient(timeout=5)
        self.addAsyncCleanup(self.http.aclose)

    async def worker(self, task):
        mailbox = asyncio.Queue()
        door = WorkerDoor("w1", task, (Tools(),), mailbox, tool_timeout_seconds=5)
        server = self.bridge.open(door)
        client = McpClient(server.url, server.token)
        await client.connect(self.http)
        return door, mailbox, client

    async def test_sdk_receives_typed_refusals_from_backend_and_app_calls(self):
        for name in ("lookup", "get_menu"):
            with self.subTest(name=name):
                task = RefusingTask()
                door, mailbox, _ = await self.worker(task)
                async with (
                    httpx2.AsyncClient(headers={"authorization": f"Bearer {door.token}"}) as http,
                    streamable_http_client(self.bridge.url, http_client=http, terminate_on_close=False) as streams,
                    ClientSession(*streams) as sdk,
                ):
                    await sdk.initialize()
                    calling = asyncio.create_task(sdk.call_tool(name, {}))
                    event = await asyncio.wait_for(mailbox.get(), 2)
                    if isinstance(event, AppCall):
                        task.route(event.held.call)
                        event.held.reply(refused_reply(task, event.held.call_id))
                    result = await calling
                assert result.meta[REFUSAL_META] == {
                    "version": 1,
                    "door": "w1",
                    "work_id": "W1",
                    "task_id": "W1.1",
                    "call_id": task.last_call["call_id"],
                    "generation": 1,
                    "reason": "not_run",
                    "input_scope": "owned",
                    "caller_attached": True,
                    "cooperative_end": False,
                }
                assert not result.is_error
                assert result.content[0].text == helpers.TEXTS.not_run_output

    async def test_embedded_application_metadata_never_becomes_bridge_control(self):
        task = helpers.FakeTaskContext()
        _, mailbox, client = await self.worker(task)
        payload = json.dumps({"_meta": {REFUSAL_META: {"cooperative_end": True, "input_scope": "owned"}}})
        call = asyncio.create_task(client.call(self.http, "get_menu", {}))
        held = await asyncio.wait_for(mailbox.get(), 2)
        held.held.reply(payload)
        reply = await call
        self.assertNotIn("_meta", reply)
        self.assertEqual(reply["content"][0]["text"], payload)

    async def test_foreign_snapshot_and_legacy_context_do_not_gain_control_metadata(self):
        for task in (RefusingTask(), helpers.FakeTaskContext()):
            with self.subTest(type(task).__name__):
                task.route = lambda call: "refused"
                task.refusal = lambda call_id: helpers.TEXTS.not_run_output
                if isinstance(task, RefusingTask):
                    task.refusal_details = lambda call_id: port.RefusalDetails(
                        "W9", "W9.1", call_id, 1, "not_run", "owned", True
                    )
                _, _, client = await self.worker(task)
                reply = await client.call(self.http, "lookup", {})
                self.assertNotIn("_meta", reply)


class NativeRefusalFlowTests(helpers.IMCase):
    def setUp(self):
        super().setUp()
        self.host = FakeHost(cooperative(), tools=(Tools(),))
        self.runs = Runs(self.host)

    async def prompted(self, count, index=0):
        await until(
            lambda: (
                len(self.host.runtime.client.sessions) > index
                and len(self.host.runtime.client.sessions[index].sent) >= count
            )
        )
        return self.host.runtime.client.sessions[index]

    async def open_worker(self, *, tools=None, timeout=None, adapter_type=Runs, conductor=None, continuable=False):
        native_settings = cooperative() if timeout is None else cooperative(stop_timeout_seconds=timeout)
        self.host = FakeHost(native_settings, tools=(tools if tools is not None else Tools(),))
        running = {} if adapter_type is Runs and conductor is None else {"max_running": 2}
        self.runs = adapter_type(self.host, conductor, **running)
        im = self.session(self.runs, continuable=continuable)
        first = im.admit(helpers.ask("Find tea."), delegation_id="D1", settings={"tools": [helpers.MENU]})
        reading = asyncio.create_task(helpers.read_all(first.stream))
        native = await self.prompted(1)
        return im, self.runs.runs[0], native, reading

    async def correct(self, im, run, text="Only green tea."):
        newer = im.admit(helpers.ask(text), delegation_id="D2", settings={"tools": [helpers.MENU]})
        reading = asyncio.create_task(helpers.read_all(newer.stream))
        await until(lambda: run.next_prompts)
        return reading

    def observe_settlements(self, run):
        self.settlements = []
        settle = run.task.settle

        def observe(batch, delivery):
            self.settlements.append((batch.id, delivery))
            settle(batch, delivery)

        run.task.settle = observe

    async def test_owned_refusal_recovers_without_a_cooperative_native_end_or_replaying_the_call(self):
        for name in ("get_menu", "lookup"):
            with self.subTest(name=name):
                im, run, native, first_read = await self.open_worker()
                original_task, old_door = run.task, run.door
                newer_read = await self.correct(im, run)
                reply = await run.door.call(name, {})
                self.assertIsInstance(reply, RefusedReply)
                self.assertTrue(reply.cooperative_end)
                await until(lambda native=native: native.cancels == 1)
                self.assertEqual(original_task.input_revision, 1)
                self.assertEqual(len(self.host.runtime.client.sessions), 1)
                last = im.admit(helpers.ask("And no sugar."), delegation_id="D3", settings={"tools": [helpers.MENU]})
                last_read = asyncio.create_task(helpers.read_all(last.stream))
                await until(lambda im=im: "D3" not in im.arrivals)
                native.put(MessageChunk("obsolete", "This old answer must not survive.", prompt=1))
                native.put(ToolCard("old-card", "lookup", ended=ToolCardEnd("old-card", False), prompt=1))
                native.end(1, stop="cancelled", usage={"totalTokens": 100})
                fresh = await self.prompted(2)
                self.assertIs(fresh, native)
                self.assertIs(run.task, original_task)
                self.assertFalse(original_task.ended)
                self.assertFalse(native.retired)
                self.assertFalse(old_door.retired)
                await until(lambda original_task=original_task: original_task.input_revision == 3)
                self.assertIn("Only green tea.", fresh.sent[-1][2])
                self.assertIn("And no sugar.", fresh.sent[-1][2])
                self.assertNotIn("This old answer must not survive.", fresh.sent[-1][2])
                self.assertIsNone(run.pending)
                self.assertFalse(run.cards_open)
                self.assertEqual(native.cancels, 1)
                self.assertFalse(old_door.qualify_refusal(reply).cooperative_end)
                self.assertEqual(fresh.cancels, 1)
                self.assertEqual(len(original_task.routes), 1)
                fresh.answer(2, "final", "Green tea is available.", usage={"totalTokens": 200})
                results = await asyncio.gather(first_read, newer_read, last_read)
                self.assertNotIn("This old answer must not survive.", json.dumps(results))
                await im.close()

    async def recovering(self, *, seconds=0.5):
        im, run, native, first_read = await self.open_worker(timeout=seconds)
        newer_read = await self.correct(im, run)
        self.observe_settlements(run)
        reply = await run.door.call("lookup", {})
        self.assertTrue(reply.cooperative_end)
        await until(lambda: native.cancels == 1)
        return im, run, native, (first_read, newer_read)

    async def test_an_old_unadmitted_call_cannot_execute_or_cancel_a_newer_invocation_on_the_same_session(self):
        for pending in (False, True):
            with self.subTest(pending=pending):
                im, run, native, first_read = await self.open_worker()
                reads = [first_read]
                door = run.door
                admitting, release = asyncio.Event(), asyncio.Event()
                self.addCleanup(release.set)
                wait_for_admission = door.wait_for_admission

                async def delayed(*args, admitting=admitting, release=release, wait=wait_for_admission, **kwargs):
                    admitting.set()
                    await release.wait()
                    return await wait(*args, **kwargs)

                door.wait_for_admission = delayed
                old = asyncio.create_task(door.call("lookup", {}))
                await admitting.wait()
                second = im.admit(
                    helpers.ask("Green tea only."), delegation_id="D2", settings={"tools": [helpers.MENU]}
                )
                reads.append(asyncio.create_task(helpers.read_all(second.stream)))
                await until(lambda run=run: run.next_prompts)
                native.end(1, usage={"totalTokens": 100})
                self.assertIs(await self.prompted(2), native)
                if pending:
                    third = im.admit(
                        helpers.ask("And no sugar."), delegation_id="D3", settings={"tools": [helpers.MENU]}
                    )
                    reads.append(asyncio.create_task(helpers.read_all(third.stream)))
                    await until(lambda run=run: run.next_prompts)
                release.set()
                self.assertEqual(await old, im.texts.not_run_output)
                self.assertEqual(run.task.routes, {})
                self.assertIsNone(run._input_recovery)
                self.assertEqual(native.cancels, 0)
                if pending:
                    native.end(2, usage={"totalTokens": 200})
                    self.assertIs(await self.prompted(3), native)
                native.answer(3 if pending else 2, "final", "The updated tea is available.", usage={"totalTokens": 300})
                await asyncio.gather(*reads)
                self.assertEqual(len(self.host.runtime.client.sessions), 1)
                await im.close()

    async def finish_successor(self, run, reads):
        await until(lambda: run.task.ended)
        self.assertTrue(run.done)
        self.assertEqual(run.task.input_revision, 1)
        self.assertEqual([delivery for _, delivery in self.settlements], ["not_applied"])
        self.assertFalse(self.traced("harness.input_recovery_ready"))
        successor = await self.prompted(1, index=1)
        self.assertIsNot(self.runs.runs[-1], run)
        successor.answer(
            1,
            "final",
            "The caller's still-owed request was handled by a separate successor.",
            usage={"totalTokens": 200},
        )
        return await asyncio.wait_for(asyncio.gather(*reads), 2)

    async def test_a_missing_native_terminal_fails_the_old_run_bounded_and_keeps_its_physical_place(self):
        im, run, native, reads = await self.recovering(seconds=0.1)
        try:
            await until(lambda: len(self.runs.runs) == 2)
            self.assertTrue(run.task.ended)
            self.assertEqual(len(self.host.runtime.client.sessions), 1)
            self.assertIsNone(self.runs.runs[-1].session)
            self.assertEqual(self.host.runtime.places, 1)
            # Core may create the successor immediately; its native dispatch must wait for the actual old owner.
            native.end(1, stop="cancelled")
            events = await self.finish_successor(run, reads)
            self.assertTrue(native.retired)
            self.assertEqual(self.host.runtime.places, 0)
            self.assertTrue(any(event[-1]["type"] == "response.failed" for event in events))
            failure = self.traced("harness.turn_refused")[0]["reason"]
            self.assertIn("TimeoutError", failure)
        finally:
            if 1 in native.outstanding:
                native.end(1, stop="cancelled")
            await im.close()
        await until(lambda: self.host.runtime.places == 0)

    async def test_renewed_adapter_and_stopped_intermediate_keep_all_exact_work_owners_fenced(self):
        old_host = FakeHost(cooperative(stop_timeout_seconds=0.02))
        old_runs = [WorkerRun(NativeTaskContext(task_id=f"W1.{number}"), old_host) for number in (1, 2)]
        for run in old_runs:
            events = run.events()
            await anext(events)
            self.addAsyncCleanup(close_events, events)
            self.assertEqual(await run.stop(), "unconfirmed")

        renewed = FakeHost(cooperative())
        # Distinct adapter runtime, same source-owned unfinished owners.
        renewed.retirements = old_host.retirements
        middle = WorkerRun(NativeTaskContext(work_id="W2", task_id="W2.1", continues="W1"), renewed)
        middle_events = middle.events()
        middle_read = asyncio.create_task(helpers.read_all(middle_events))
        await asyncio.sleep(0)
        self.assertEqual(renewed.runtime.client.sessions, [])
        self.assertEqual(await middle.stop(), "stopped")
        await middle_read

        latest = WorkerRun(NativeTaskContext(work_id="W3", task_id="W3.1", continues="W2.1"), renewed)
        latest_events = latest.events()
        latest_read = asyncio.create_task(helpers.read_all(latest_events))
        self.addAsyncCleanup(latest.release)
        self.addAsyncCleanup(close_events, latest_events)

        other_host = FakeHost(cooperative())
        other_host.retirements = old_host.retirements
        other_task = NativeTaskContext(work_id="W3", task_id="W3.1", continues="W1")
        other_task.conversation_ref = "a different conversation with the same local task names"
        independent = WorkerRun(other_task, other_host)
        independent_events = independent.events()
        await anext(independent_events)
        other_native = other_host.runtime.client.sessions[0]
        other_native.answer(1, "independent", "Independent work completed.")
        await helpers.read_all(independent_events)

        first, sibling = old_host.runtime.client.sessions
        first.end(1, stop="cancelled")
        await until(lambda: old_host.runtime.places == 1)
        self.assertEqual(renewed.runtime.client.sessions, [])
        self.assertFalse(latest_read.done())
        sibling.end(1, stop="cancelled")
        await until(lambda: bool(renewed.runtime.client.sessions))
        native = renewed.runtime.client.sessions[0]
        native.answer(1, "continued", "The exact continued work completed.")
        await latest_read
        self.assertEqual(old_host.runtime.places, 0)
        self.assertEqual(old_host.retirements.owners, {})

    async def test_stop_during_owned_input_recovery_wins_and_shares_the_single_cancel(self):
        im, run, native, reads = await self.recovering()
        self.assertTrue(run.task.request_stop(ForNewerRequest()))
        stopping = asyncio.create_task(run.task.confirm_stop())
        native.end(1, stop="cancelled", usage={"totalTokens": 100})
        self.assertEqual(await stopping, "stopped")
        await asyncio.gather(*reads)
        self.assertEqual(native.cancels, 1)
        self.assertEqual(run.task.input_revision, 1)
        self.assertEqual(len(self.host.runtime.client.sessions), 1)
        await im.close()

    async def test_recovery_never_turns_provider_failure_or_token_cap_into_a_fresh_prompt(self):
        for stop, error in (("max_tokens", None), ("cancelled", "provider failed")):
            with self.subTest(stop=stop, error=error):
                im, run, native, reads = await self.recovering()
                native.end(1, stop=stop, error=error, usage={"totalTokens": 100})
                events = await self.finish_successor(run, reads)
                self.assertTrue(native.retired)
                self.assertTrue(any(event[-1]["type"] == "response.failed" for event in events))
                await im.close()

    async def test_input_stays_unsent_when_replacement_open_fails_or_stop_wins_during_open(self):
        for stop in (False, True):
            with self.subTest(stop=stop):
                im, run, native, reads = await self.recovering()
                opening, release = asyncio.Event(), asyncio.Event()
                client = self.host.runtime.client
                open_session = client.new_session

                async def replacement(
                    *args, opening=opening, release=release, stop=stop, open_session=open_session, **kwargs
                ):
                    opening.set()
                    await release.wait()
                    if not stop:
                        raise RuntimeError("The replacement session could not open")
                    return await open_session(*args, **kwargs)

                client.new_session = replacement
                # A normal terminal racing our session-scoped cancellation cannot prove the cancel was consumed.
                native.end(1, stop="end_turn", usage={"totalTokens": 100})
                await opening.wait()
                if stop:
                    self.assertTrue(run.task.request_stop(ForNewerRequest()))
                    self.assertEqual(await run.task.confirm_stop(), "stopped")
                release.set()
                await asyncio.wait_for(asyncio.gather(*reads), 2)
                await until(lambda: self.host.runtime.places == 0)
                self.assertEqual([delivery for _, delivery in self.settlements], ["not_applied"])
                self.assertEqual(run.task.input_revision, 1)
                self.assertFalse(self.traced("harness.steer_applied"))
                self.assertTrue(all(not session.sent for session in client.sessions[1:]))
                self.assertTrue(all(session.retired for session in client.sessions))
                await im.close()

    async def test_recovery_waits_for_started_backend_effect_and_preserves_its_result_without_replay(self):
        for deadline_expired in (False, True):
            with self.subTest(deadline_expired=deadline_expired):
                self.traces.clear()
                await self.check_running_effect(deadline_expired)

    async def check_running_effect(self, deadline_expired):
        executing, finish = asyncio.Event(), asyncio.Event()
        self.addCleanup(finish.set)
        output = "The original lookup completed."

        class BlockingTools(Tools):
            calls = 0

            async def execute(self, call_id, name, arguments):
                self.calls += 1
                executing.set()
                await finish.wait()
                return output

        tools = BlockingTools()
        im, run, native, first_read = await self.open_worker(tools=tools, timeout=0.1 if deadline_expired else None)
        door = run.door
        old = asyncio.create_task(door.call("lookup", {}))
        if deadline_expired:
            # The HTTP/MCP client cannot release the bridge's shielded serve task.
            door.pending_calls.add(old)
            old.add_done_callback(door.pending_calls.discard)
        await executing.wait()
        await until(lambda: run.running)
        newer_read = await self.correct(im, run)
        self.observe_settlements(run)
        self.assertTrue((await door.call("lookup", {})).cooperative_end)
        await until(lambda: native.cancels == 1)
        native.end(1, "cancelled", {"totalTokens": 100})
        if deadline_expired:
            await until(lambda: len(self.runs.runs) == 2)
            self.assertTrue(native.quiescent)
            self.assertIsNone(self.runs.runs[-1].session)
        self.assertFalse(old.done())
        self.assertEqual((len(self.host.runtime.client.sessions), self.host.runtime.places, tools.calls), (1, 1, 1))
        finish.set()
        self.assertEqual(await old, output)
        number, index = (1, 1) if deadline_expired else (2, 0)
        current = await self.prompted(number, index=index)
        self.assertEqual(current is native, not deadline_expired)
        self.assertIn(output, current.sent[-1][2])
        self.assertIn("Only green tea.", current.sent[-1][2])
        reads = (first_read, newer_read)
        if deadline_expired:
            await self.finish_successor(run, reads)
        else:
            current.answer(number, "final", "The updated lookup is complete.", usage={"totalTokens": 200})
            await asyncio.gather(*reads)
        await until(lambda: self.host.runtime.places == 0)
        self.assertEqual(tools.calls, 1)
        await im.close()

    async def hold_placement(self, decision, *, text="Other request.", adapter_type=Runs):
        deciding, entered = asyncio.Event(), asyncio.Event()

        async def place_after_release(given, place):
            entered.set()
            await deciding.wait()
            await place(port.CompleteDecision((decision,)))

        conductor = helpers.planned(port.Start(("D1",)), place_after_release)
        im, _, native, first_read = await self.open_worker(conductor=conductor, adapter_type=adapter_type)
        newer = im.admit(helpers.ask(text), delegation_id="D2")
        await entered.wait()
        return im, native, first_read, newer, deciding

    async def test_global_arriving_after_app_enqueue_is_checked_before_the_actual_first_route(self):
        executing, finish = asyncio.Event(), asyncio.Event()
        deciding, entered = asyncio.Event(), asyncio.Event()
        self.addCleanup(finish.set)
        self.addCleanup(deciding.set)

        class BlockingTools(Tools):
            async def execute(self, call_id, name, arguments):
                executing.set()
                await finish.wait()
                return "Lookup completed."

        async def other(given, place):
            entered.set()
            await deciding.wait()
            await place(port.CompleteDecision((port.CallerMessage(("D2",), "status", "Other request handled."),)))

        self.host = FakeHost(cooperative(), tools=(BlockingTools(),))
        self.runs = Runs(self.host, helpers.planned(port.Start(("D1",)), other), max_running=2)
        im = self.session(self.runs)
        first = im.admit(helpers.ask("Find tea."), delegation_id="D1", settings={"tools": [helpers.MENU]})
        first_read = asyncio.create_task(helpers.read_all(first.stream))
        native = await self.prompted(1)
        run = self.runs.runs[0]
        backend = asyncio.create_task(run.door.call("lookup", {}))
        await executing.wait()
        await until(lambda: run.running)
        app = asyncio.create_task(run.door.call("get_menu", {}))
        await until(lambda: run.app_queue)
        original = run.app_queue[0]
        other_response = im.admit(helpers.ask("Other request."), delegation_id="D2")
        await entered.wait()
        finish.set()
        self.assertEqual(await backend, "Lookup completed.")
        await until(lambda: run.door.placement_waiting)
        self.assertNotIn(original.call_id, run.task.routes)
        self.assertEqual(run.task.refusals, {})
        self.assertFalse(app.done())
        deciding.set()
        await helpers.read_all(other_response.stream)
        events = await first_read
        completion = im.admit_client_request(
            helpers.app_outputs(events, "Tea."), delegation_id="D1", settings={"tools": [helpers.MENU]}
        )
        completion_read = asyncio.create_task(helpers.read_all(completion.stream))
        self.assertEqual(await app, "Tea.")
        self.assertEqual(run.task.routes[original.call_id], "app")
        self.assertEqual(run.task.refusals, {})
        self.assertEqual((len(native.sent), native.cancels), (1, 0))
        native.answer(1, "final", "Tea is available.", usage={"totalTokens": 100})
        await completion_read

    async def check_held_backend_wait(self, disposition):
        stopped = disposition == "work stopped"
        status = port.CallerMessage(("D2",), "status", "Other request handled.")
        runtime = Runs if stopped else AttachmentSafeRuns
        im, native, first_read, newer, deciding = await self.hold_placement(status, adapter_type=runtime)
        run = self.runs.runs[0]
        door = run.door
        held = asyncio.create_task(door.call("lookup", {}))
        await until(lambda: door.placement_waiting)
        self.assertFalse(held.done())
        if stopped:
            self.assertTrue(run.task.request_stop(AtClose()))
            confirmed = asyncio.create_task(run.task.confirm_stop())
            await until(lambda: native.cancels)
            native.end(1, stop="cancelled", usage={"totalTokens": 100})
            self.assertEqual(await confirmed, "stopped")
            self.assertEqual(await held, helpers.TEXTS.stopped_output)
            self.assertFalse(deciding.is_set())
            self.assertEqual(len(native.sent), 1)
        else:
            await im.set_attachment(False)
            self.assertFalse(run.task.pending_input_ready())
            self.assertFalse(held.done())
            if disposition == "caller withdrew":
                held.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await held
                self.assertEqual(door.held, [])
        deciding.set()
        if not stopped:
            await im.set_attachment(True)
        await helpers.read_all(newer.stream)
        if not stopped:
            if disposition == "resumed":
                self.assertEqual(await held, '{"found": "lookup"}')
            self.assertEqual(await door.call("lookup", {}), '{"found": "lookup"}')
            self.assertEqual(native.cancels, 0)
            native.answer(1, "final", "The original lookup is complete.", usage={"totalTokens": 100})
        await first_read
        self.assertEqual(door.held, [])

    async def test_only_existing_after_call_eligibility_allows_a_detached_owned_input_hint(self):
        for kept in (False, True):
            with self.subTest(kept=kept):
                im, run, native, first_read = await self.open_worker(continuable=True, adapter_type=AttachmentSafeRuns)
                if kept:
                    self.assertEqual(
                        await run.door.call("continue_after_call", {"message": "I will finish checking."}),
                        im.texts.continued_output,
                    )
                self.assertEqual(run.task.after_call, kept)
                newer_read = await self.correct(im, run, "Green tea only.")
                await im.set_attachment(False)
                self.assertEqual(run.task.pending_input_ready(), kept)
                reply = await run.door.call("lookup", {})
                original = run.task.refusal_details(reply.details.call_id)
                self.assertFalse(reply.details.caller_attached)
                self.assertTrue(reply.details.owned_input_pending)
                self.assertEqual(reply.cooperative_end, kept)
                if not kept:
                    await im.set_attachment(True)
                    native.answer(1, "yield", "I will use the pending input.", usage={"totalTokens": 100})
                    current = await self.prompted(2)
                else:
                    await until(lambda native=native: native.cancels == 1)
                    native.end(1, stop="cancelled", usage={"totalTokens": 100})
                    current = await self.prompted(2)
                self.assertEqual(run.task.input_revision, 2)
                if kept:
                    refused = await run.door.call("get_menu", {})
                    self.assertEqual(refused.details.reason, "attachment")
                    self.assertFalse(refused.cooperative_end)
                    later_call = helpers.call_item("lookup", {}, "new-after-call-lookup")
                    self.assertFalse(run.task.placement_pending(later_call))
                    self.assertEqual(await run.door.call("lookup", {}), '{"found": "lookup"}')
                    await im.set_attachment(True)
                self.assertIs(run.task.refusal_details(original.call_id), original)
                self.assertFalse(original.caller_attached)
                self.assertEqual(native.cancels, int(kept))
                self.assertEqual(current.cancels, int(kept))
                current.answer(2, "final", "The updated lookup is complete.", usage={"totalTokens": 200})
                await asyncio.gather(first_read, newer_read)
