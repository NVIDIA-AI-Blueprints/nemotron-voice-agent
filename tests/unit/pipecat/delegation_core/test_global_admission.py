# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Global placement waits before first admission; owned input and prior effects keep their normal gates."""

import asyncio
import json
from types import SimpleNamespace

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.llm import LlmSettings
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import (
    CallerMessage,
    CompleteDecision,
    Start,
    Steer,
    WorkTarget,
)
from nemotron_voice_agent.pipecat.runtime.gateways.live.server_delegation import ServerDelegation
from tests.unit.pipecat.delegation_core.helpers import (
    MENU,
    TEXTS,
    IMCase,
    RecordingTools,
    ScriptedProvider,
    a_round,
    adapter,
    answers,
    ask,
    call_events,
    call_item,
    call_round,
    function_calls,
    gated,
    planned,
    read_all,
    text_round,
)


class HeldTools(RecordingTools):
    """An actual LlmRun tool source that records admission, then waits before reporting its sole outcome."""

    def __init__(self):
        super().__init__()
        self.started, self.release = asyncio.Event(), asyncio.Event()

    async def execute(self, call_id, name, arguments):
        self.calls.append((call_id, name, arguments))
        self.started.set()
        await self.release.wait()
        return json.dumps({"ok": True, "tool": name})


class GlobalAdmissionTests(IMCase):
    async def pending(self, backend, *, decision=None, settings=None, **session_options):
        deciding = asyncio.Event()

        async def place_newer(given, place):
            await deciding.wait()
            await place(decision or CallerMessage(("D2",), "status", "The unrelated request is handled."))

        backend.conductor = lambda: planned(Start(("D1",)), place_newer)
        im = self.session(backend, **session_options)
        first = im.admit(ask("Read the actual tool result."), delegation_id="D1", settings=settings)
        await self.until(lambda: "W1" in im.works and im.works["W1"].tasks[0].run is not None, "the task")
        second = im.admit(ask("Handle this newer request."), delegation_id="D2")
        await self.turns()
        self.assertTrue(im.waits_for_decision)
        return im, im.works["W1"].tasks[0], first, second, deciding

    async def test_pending_placement_keeps_the_entire_tool_batch_unadmitted(self):
        for mixed in (False, True):
            with self.subTest(mixed=mixed):
                self.traces.clear()
                await self.check_batch(mixed)

    async def check_batch(self, mixed):
        streamed = asyncio.Event()
        other = "get_menu" if mixed else "stock"
        batch = a_round(
            call_events(other, {}, call_id="call_other"), call_events("lookup", {}, call_id="call_lookup", index=1)
        )
        tools = RecordingTools(("lookup", "stock"))
        settings = (
            {
                "tools": [MENU],
                "tool_choice": {"type": "allowed_tools", "tools": [{"type": "function", "name": "lookup"}]},
            }
            if mixed
            else None
        )
        limits = LlmSettings(
            first_event_timeout_seconds=0.05,
            idle_timeout_seconds=0.05,
            server_tool_timeout_seconds=0.05,
            provider_retries=0,
        )
        provider = ScriptedProvider(gated(streamed, batch), text_round("The lookup completed."), limits=limits.limits)
        im, task, first, second, deciding = await self.pending(
            adapter(provider, tools=[tools], settings=limits), settings=settings
        )
        streamed.set()
        await self.until(lambda: self.traced("backend.completed"), "the completed provider response")
        await asyncio.sleep(0.08)
        self.assertEqual((task.routes, task.refusals, tools.calls), ({}, {}, []))
        self.assertEqual(len(provider.payloads), 1)
        self.assertFalse(task.driver.done())
        deciding.set()
        older, newer = await asyncio.gather(read_all(first.stream), read_all(second.stream))
        called = ["call_lookup"] if mixed else ["call_other", "call_lookup"]
        self.assertEqual([call[0] for call in tools.calls], called)
        self.assertEqual(task.routes, {"call_other": "refused" if mixed else "backend", "call_lookup": "backend"})
        if mixed:
            self.assertEqual(task.refusal_details("call_other").reason, "unlisted")
        else:
            self.assertEqual(self.traced("tools.refused"), [])
        self.assertEqual(answers(older), ["The lookup completed."])
        self.assertEqual(answers(newer), ["The unrelated request is handled."])
        self.assertEqual(im.ledger.tasks[task.task_id].state, "answered")

    async def test_global_becoming_owned_refuses_the_unadmitted_call_without_running_it(self):
        streamed = asyncio.Event()
        provider = ScriptedProvider(gated(streamed, call_round("lookup", {}, "call_old")), text_round("Updated."))
        tools = RecordingTools()
        im, task, first, second, deciding = await self.pending(
            adapter(provider, tools=[tools]), decision=Steer(("D2",), (WorkTarget("W1"),))
        )
        streamed.set()
        await self.until(lambda: self.traced("backend.completed"), "the pending batch")
        await self.turns()
        self.assertEqual((task.routes, tools.calls), ({}, []))
        deciding.set()
        await asyncio.gather(read_all(first.stream), read_all(second.stream))
        self.assertEqual(tools.calls, [])
        details = task.refusal_details("call_old")
        self.assertEqual((details.reason, details.input_scope), ("not_run", "owned"))
        self.assertEqual(task.routes, {"call_old": "refused"})
        self.assertIn("Handle this newer request.", str(provider.payloads[-1]["input"]))
        self.assertFalse(task.steering or task.taken)
        self.assertEqual(im.ledger.tasks[task.task_id].state, "answered")

    async def test_failed_routing_fences_effects_until_applicable_input_is_delivered(self):
        cases = ("rejected", "late_rejected", "partial", "frontend", "frontend_reconnect")
        for case in cases:
            late, mixed = case == "late_rejected", case == "partial"
            frontend = case in {"frontend", "frontend_reconnect"}
            with self.subTest(case=case):
                self.traces.clear()
                streamed, deciding, wait_for_input = asyncio.Event(), asyncio.Event(), asyncio.Event()
                question = "Which day did you mean?"
                proposal = (
                    CompleteDecision(
                        (Steer(("D2",), (WorkTarget("W404"),)), CallerMessage(("D2",), "question", question))
                    )
                    if mixed
                    else Start(("unbound",))
                )

                async def submit(given, place, ready=deciding, decision=proposal):
                    await ready.wait()
                    await place(decision)

                tools = RecordingTools()
                tools.read_only_tools = frozenset({"lookup"})
                retry_with_read = a_round(
                    call_events("get_menu", {}, call_id="call_retry"),
                    call_events("lookup", {}, call_id="call_read", index=1),
                )
                provider = ScriptedProvider(
                    gated(streamed, call_round("get_menu", {}, "call_old")),
                    retry_with_read,
                    gated(wait_for_input, text_round("Waiting for input.")),
                    call_round("get_menu", {}, "call_current"),
                    text_round("Current request completed."),
                )
                backend = adapter(provider, tools=[tools])
                backend.max_running = 3
                routing = () if frontend else (submit,) if mixed else (submit, submit)
                backend.conductor = lambda routing=routing: planned(
                    Start(("D1",)), *routing, Steer(("D3",), (WorkTarget("W1"),))
                )
                im = self.session(backend)
                first = im.admit(ask("Please send the order."), delegation_id="D1", settings={"tools": [MENU]})
                await self.until(lambda provider=provider: len(provider.payloads) == 1, "the original worker round")
                task = im.works["W1"].tasks[0]
                if frontend:
                    core = SimpleNamespace(begin_caller_turn=lambda _conversation, im=im: im.begin_caller_turn())
                    gateway = ServerDelegation(core)
                    gateway.conversation = "conversation"
                    if case == "frontend_reconnect":
                        await im.set_attachment(False)
                    normal = gateway.begin_caller_turn()
                    settle = gateway.begin_caller_turn()
                    normal()
                    self.assertTrue(im.caller_turn_pending)
                    if case == "frontend_reconnect":
                        await im.set_attachment(True)
                second = None if frontend else im.admit(ask("Wait before submitting it."), delegation_id="D2")
                if not late:
                    streamed.set()
                    await self.until(lambda: self.traced("backend.completed"), "the held call proposal")
                    self.assertNotIn("call_old", im.ledger.actions)
                if frontend:
                    settle(routing_failed=True)
                    self.assertFalse(im.caller_turn_pending)
                else:
                    deciding.set()
                    expected = [question, TEXTS.routing_failed_output] if mixed else [TEXTS.routing_rejected_output]
                    self.assertEqual(answers(await read_all(second.stream)), expected)
                self.assertEqual(len(self.traced("im.decide.retry")), 0 if mixed or frontend else 1)
                streamed.set()
                await self.until(lambda provider=provider: len(provider.payloads) == 3, "the replan after refusals")
                for call_id in ("call_old", "call_retry"):
                    record = im.ledger.actions[call_id]
                    self.assertEqual((record.state, record.app_published), ("not_run", False))
                    self.assertEqual(task.refusal_details(call_id).reason, "routing_unresolved")
                self.assertEqual([call[0] for call in tools.calls], ["call_read"])
                third = im.admit(ask("Please go ahead with the order now."), delegation_id="D3")
                older, current = await asyncio.gather(read_all(first.stream), read_all(third.stream))
                self.assertEqual(function_calls(older + current), ["call_current"])
                final = await self.answer_calls(im, current, output="Sent the order.")
                self.assertEqual(answers(final), ["Current request completed."])

    async def late_app(self, *, decision=None, detach_after_placement=False, unclear=False, max_running=1):
        tools = HeldTools()
        batch = a_round(
            call_events("lookup", {}, call_id="call_lookup"),
            call_events("get_menu", {}, call_id="call_menu", index=1),
        )
        provider = ScriptedProvider(batch, text_round("The actual menu was received."))
        limits = LlmSettings(server_tool_timeout_seconds=0.02) if unclear else LlmSettings()
        backend = adapter(provider, tools=[tools], settings=limits)
        backend.max_running = max_running
        # Let the server call start first. The other call has a route but has not reached the app.
        deciding = asyncio.Event()

        async def place_newer(given, place):
            await deciding.wait()
            await place(decision or CallerMessage(("D2",), "status", "Independent status."))
            if detach_after_placement:
                await im.set_attachment(False)

        backend.conductor = lambda: planned(Start(("D1",)), place_newer)
        im = self.session(backend)
        first = im.admit(ask("Read lookup and the menu."), delegation_id="D1", settings={"tools": [MENU]})
        await tools.started.wait()
        task = im.works["W1"].tasks[0]
        self.assertEqual(task.routes, {"call_lookup": "backend", "call_menu": "app"})
        earlier = im.begin_caller_turn()
        settle = im.begin_caller_turn()
        earlier()
        if not unclear:
            tools.release.set()
        await self.until(lambda: task.run.waits, "the unforwarded app boundary")
        await self.turns()
        self.assertFalse(im.waits_for_decision)
        self.assertFalse(im.ledger.actions["call_menu"].app_published)
        second = im.admit(ask("The newer request."), delegation_id="D2")
        settle()
        self.assertTrue(im.waits_for_decision)
        return im, task, tools, provider, first, second, deciding

    async def test_late_global_foreign_placement_forwards_the_original_app_call_once(self):
        for unclear in (False, True):
            workers = 3 if unclear else 1
            with self.subTest(backend_outcome_unknown=unclear, workers=workers):
                self.traces.clear()
                im, task, tools, provider, first, second, deciding = await self.late_app(
                    unclear=unclear, max_running=workers
                )
                self.assertEqual(task.routes, {"call_lookup": "backend", "call_menu": "app"})
                self.assertEqual(task.refusals, {})
                self.assertEqual(task.record.state, "running")
                self.assertFalse(task.placement_pending(call_item("get_menu", {}, "call_menu")))
                deciding.set()
                older, newer = await asyncio.gather(read_all(first.stream), read_all(second.stream))
                self.assertEqual((function_calls(older), function_calls(newer)), (["call_menu"], []))
                final = await self.answer_calls(im, older, output="The real menu.")
                self.assertEqual(answers(final), ["The actual menu was received."])
                self.assertEqual([call[0] for call in tools.calls], ["call_lookup"])
                self.assertEqual(task.refusals, {})
                if unclear:
                    self.assertEqual([trace["call_id"] for trace in self.traced("tools.unclear")], ["call_lookup"])
                    self.assertEqual(im.ledger.actions["call_lookup"].state, "unclear")
                    output = {"type": "function_call_output", "call_id": "call_lookup", "output": TEXTS.owed_output}
                    self.assertIn(output, provider.payloads[-1]["input"])

    async def test_detach_before_late_commit_never_forwards_the_app_call(self):
        im, task, tools, _, first, second, deciding = await self.late_app(detach_after_placement=True)
        deciding.set()
        await self.until(lambda: task.refusals or task.ended, "the detached commit gate")
        self.assertEqual(task.refusal_details("call_menu").reason, "attachment")
        self.assertEqual([call[0] for call in tools.calls], ["call_lookup"])
        await im.set_attachment(True)
        older, newer = await asyncio.gather(read_all(first.stream), read_all(second.stream))
        self.assertEqual(function_calls(older + newer), [])
        self.assertEqual(task.route(call_item("get_menu", {}, "call_menu")), "refused")

    async def test_detached_pending_placement_stays_held_until_the_existing_reattach_lifecycle(self):
        im, task, tools, _, first, second, deciding = await self.late_app()
        await im.set_attachment(False)
        deciding.set()
        await self.turns()
        self.assertTrue(im.waits_for_decision)
        self.assertEqual(task.routes["call_menu"], "app")
        self.assertIsNone(task.outputs_in)
        self.assertEqual([call[0] for call in tools.calls], ["call_lookup"])
        await im.set_attachment(True)
        older, newer = await asyncio.gather(read_all(first.stream), read_all(second.stream))
        self.assertEqual(function_calls(older + newer), ["call_menu"])
        await self.answer_calls(im, older)
