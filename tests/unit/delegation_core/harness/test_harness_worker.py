# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The harness's worker run and conductor on a session double: turns, app calls, steering, stops and sessions."""

import asyncio
import itertools
import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

from examples.frontend_backend_live.delegation_core.backend.conductors import SteerOpenWork
from examples.frontend_backend_live.delegation_core.backend.harness.acp import AcpAgent, SessionUpdates
from examples.frontend_backend_live.delegation_core.backend.harness.conductor import (
    HarnessConductor,
)
from examples.frontend_backend_live.delegation_core.backend.harness.protocol import (
    NativeSteerReceipt,
    PlanUpdate,
    RefusedReply,
    ThoughtChunk,
    ToolCard,
    ToolCardEnd,
)
from examples.frontend_backend_live.delegation_core.backend.harness.worker import HarnessTurnError, WorkerRun
from examples.frontend_backend_live.delegation_core.backend.port import (
    CallerMessage,
    ConductorInput,
    DecisionRejected,
    PlacementExpired,
    Start,
    Steer,
    TaskSpec,
    WorkTarget,
)
from tests.unit.delegation_core.harness.doubles import (
    FakeHost,
    FakeTaskContext,
    cooperative,
    plain,
    settings,
    until,
    wire,
)
from tests.unit.delegation_core.helpers import (
    MENU,
    STEERING,
    FakeDecisionTarget,
    IMCase,
    RecordingTools,
    a_request,
    an_ended_work,
    an_open_work,
    answers,
    app_outputs,
    ask,
    check_events,
    finished_items,
    items_of,
    logged,
    placing,
    planned,
    read_all,
    steer_action,
    turns,
)

ORDER_FUNCTION = {
    "type": "function",
    "name": "get_order",
    "description": "A cafe order's status.",
    "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
}
USAGE = {"inputTokens": 80, "outputTokens": 20, "totalTokens": 100}
LATER_USAGE = {"inputTokens": 150, "outputTokens": 30, "totalTokens": 180}
NEXT = {"id": "d_next", "type": "message", "role": "user", "content": "And black tea?"}
BETWEEN = {"id": "d_between", "type": "message", "role": "user", "content": "Hi."}
DECIDED = {"totalTokens": 50}
EARLIER = ({"type": "message", "role": "user", "content": "Earlier: book a table."},)
PARTS = [
    {"type": "input_text", "text": "Teas."},
    {"type": "input_image", "image_url": "https://x/t.png"},
    {"type": "input_text", "text": "Hot."},
]
OLDER = [{"type": "message", "role": "user", "content": "Oat milk."}]
NEWER = [{"type": "message", "role": "user", "content": "Warm the croissant."}]
MENU_APP = {"tools": [MENU]}
MILK = {"type": "message", "role": "user", "content": "With milk, please."}
IMAGE = {"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,"}]}


def texts(events):
    return [item["content"][0]["text"] for item in items_of(events, "message")]


def brief(part):
    """Return a conductor's brief as the IM writes it: a user message with an id of its own."""
    return {"id": f"msg_{part}", "type": "message", "role": "user", "content": f"Task: {part}"}


def request(delegation_id, text):
    """Return a request's input as the IM records it: its message carries the request's id."""
    return [{**item, "id": delegation_id} for item in ask(text)]


async def close_events(events):
    """Close a run's events, as the IM does once it stops reading them."""
    await events.aclose()


class Harness(unittest.IsolatedAsyncioTestCase):
    def make(self, task=None, harness=None, **block):
        self.host = FakeHost(harness or settings(**block))
        self.task = task or FakeTaskContext()
        self.run = WorkerRun(self.task, self.host)
        return self.run

    async def open(self, **block):
        run = self.make(**block)
        self.events = run.events()
        first = await anext(self.events)
        self.assertEqual(first["type"], "response.created")
        self.session = self.host.runtime.client.sessions[-1]
        self.seen = [first]
        return self.session

    def collecting(self):
        async def collect():
            async for event in self.events:
                self.seen.append(event)
            return self.seen

        return asyncio.ensure_future(collect())

    async def opened(self, **block):
        """Open the run and collect its events from there on; return the session and the collector."""
        return await self.open(**block), self.collecting()

    async def rest(self):
        return await self.collecting()

    async def failure(self):
        with self.assertRaises(HarnessTurnError) as raised:
            await self.collecting()
        return raised.exception


class TurnTests(Harness):
    async def test_failed_native_turn_keeps_reported_usage_without_publishing_its_text(self):
        session = await self.open()
        session.say("m1", "Never publish this.")
        session.end(1, "refusal", usage=USAGE, invocation_usage=USAGE)
        await self.failure()
        assert self.seen[-1]["response"]["usage"] == {"input_tokens": 80, "output_tokens": 20, "total_tokens": 100}
        self.assertEqual(items_of(self.seen, "message"), [])
        self.assert_reported_terminal_usage(session)

    def assert_reported_terminal_usage(self, session):
        [reported] = self.task.traced("harness.round.usage")
        self.assertEqual(reported["reported_counters"], USAGE)
        self.assertEqual((reported["prompt"], reported["conversation_ref"]), (1, self.task.conversation_ref))
        self.assertEqual(reported["session"], session.id)
        self.assertTrue(reported["native_terminal_observed"])

    async def test_exact_invocation_usage_overrides_stale_cumulative_fields_even_when_empty(self):
        partial = {"totalTokens": 100, "thoughtTokens": 4, "cachedReadTokens": 12, "cachedWriteTokens": 3}
        cases = (
            ({}, None, None),
            ({"outputTokens": 7}, {"output_tokens": 7}, {"outputTokens": 7}),
            (LATER_USAGE, {"input_tokens": 150, "output_tokens": 30, "total_tokens": 180}, LATER_USAGE),
            (
                {**partial, "inputTokens": True, "outputTokens": -2},
                {
                    "total_tokens": 100,
                    "input_tokens_details": {"cached_tokens": 12},
                    "output_tokens_details": {"reasoning_tokens": 4},
                },
                partial,
            ),
        )
        for invocation, expected, observation in cases:
            with self.subTest(invocation=invocation):
                session = await self.open()
                session.answer(1, "m1", "Done.", usage=USAGE, invocation_usage=invocation)
                self.assertEqual((await self.rest())[-1]["response"].get("usage"), expected)
                [reported] = self.task.traced("harness.round.usage")
                self.assertEqual(reported["reported_counters"], observation)
                self.assertEqual(reported["raw_reported_counters"], USAGE)
                self.assertEqual(reported["usage_scope"], "invocation")

    async def test_unlisted_tools_retry_the_session_once_then_fail(self):
        run = self.make()
        self.host.bridge.lists = False
        with self.assertRaisesRegex(RuntimeError, "did not list the session's tools"):
            await anext(run.events())
        self.assertEqual([server.name for server in self.host.runtime.client.servers], ["w1", "w2"])
        self.assertTrue(all(session.retired for session in self.host.runtime.client.sessions))

    async def test_the_first_prompt_reads_an_app_output_as_its_text(self):
        call = {"type": "function_call", "call_id": "call_1", "name": "get_menu", "arguments": "{}"}
        cases = (
            (PARTS, "Teas.\nHot."),
            ([{"id": "o1"}], '[{"id": "o1"}]'),
            ([], "[]"),
            ({"stock": 3}, '{"stock": 3}'),
            (0, "0"),
            (False, "false"),
            (None, "null"),
        )
        for output, text in cases:
            with self.subTest(output=output):
                result = {"type": "function_call_output", "call_id": "call_1", "output": output}
                session = await self.open(task=FakeTaskContext(history=[*FakeTaskContext().history(), call, result]))
                line = self.host.renderer.prompts.output.format(name="get_menu", output=text)
                self.assertIn(line, session.texts("prompt")[0])

    async def test_a_streamed_answer_ends_the_turn_and_sends_no_cancel(self):
        session = await self.open()
        session.answer(1, "m1", "Do", "ne.", usage=USAGE, invocation_usage=USAGE)
        events = await self.rest()
        self.assertEqual(check_events(events), [])
        self.assertEqual(answers(events), ["Done."])
        usage = {"input_tokens": 80, "output_tokens": 20, "total_tokens": 100}
        self.assertEqual(events[-1]["response"]["usage"], usage)
        self.assertEqual(self.host.idle(), [session])
        self.assertEqual((session.retired, self.host.bridge.closed), (False, []))
        self.assertEqual(await self.run.stop(), "stopped")
        self.assert_reported_terminal_usage(session)
        self.assertEqual(session.cancels, 0)

    async def test_the_runtimes_own_text_with_no_usage_is_unsaid(self):
        for scenario in ("provider_error", "idle_cancel_poison"):
            with self.subTest(scenario):
                session = await self.open()
                steps = wire(scenario)
                for step in steps:
                    if "update" in step:
                        session.replay(step["update"]["update"])
                if scenario == "provider_error":
                    session.end(1, error="RPC: admitted_turn_failed")
                    reason = "admitted_turn_failed"
                else:
                    session.end(1, "cancelled")
                    reason = "cancelled"
                self.assertIn(reason, str(await self.failure()))
                self.assertEqual(self.seen[-1]["type"], "response.failed")
                self.assertEqual(self.seen[-1]["response"]["error"]["code"], "server_error")
                self.assertEqual(items_of(self.seen, "message"), [])

    async def test_a_turn_with_no_answer_fails_with_its_reason(self):
        tool = (ToolCard("tc1", "mcp__w1__lookup", {}), ToolCardEnd("tc1", False, "ok"))
        for reason, chunks, events, end in (
            ("without a message", (), (), {"usage": {"totalTokens": 0}}),
            ("without a message after its last tool call", ("Let me ", "look."), tool, {"usage": USAGE}),
            ("cancelled", (), (), {"stop": "cancelled", "usage": USAGE}),
            ("Connection closed", (), (), {"stop": None, "error": "ConnectionError: Connection closed"}),
        ):
            with self.subTest(reason):
                session = await self.open()
                if chunks:
                    session.say("m1", *chunks)
                for event in events:
                    session.put(event)
                session.end(1, **end)
                self.assertIn(reason, str(await self.failure()))

    async def test_a_plain_agents_cards_are_activity_only(self):
        session = await self.open(harness=plain())
        session.say("m1", "Looking.", thought=False)
        session.put(ToolCard("tc1", "Looking up the menu", {}))
        session.put(ToolCardEnd("tc1", False, "ok"))
        session.say("m2", "Found it.", thought=False)
        session.end(1)
        events = await self.rest()
        self.assertEqual(self.task.started, [])
        self.assertEqual(answers(events, "commentary"), ["Looking."])
        self.assertEqual(answers(events), ["Found it."])
        self.assertIsNone(events[-1]["response"].get("usage"))

    async def test_a_card_announced_over_ends_at_once(self):
        # ACP lets a tool_call carry its final status: an agent may report a call in one update after it returned.
        session = await self.open(harness=plain(turn_timeout_seconds=1))
        reader = SessionUpdates(AcpAgent(), "w1")
        for call_id, status in (("t1", "completed"), ("t2", "failed"), ("t3", "pending")):
            update = {"sessionUpdate": "tool_call", "toolCallId": call_id, "title": "get_menu", "status": status}
            session.put(reader.read(update))
        call = asyncio.ensure_future(self.run.door.call("get_menu", {}))
        events = await self.rest()
        [forwarded] = items_of(events, "function_call")
        self.assertEqual((forwarded["name"], self.run.cards_open), ("get_menu", {"t3"}))
        self.task.give_outputs([{"type": "function_call_output", "call_id": forwarded["call_id"], "output": "teas"}])
        await self.run.resume()
        self.assertEqual(await call, "teas")
        session.end(1)

    async def test_a_resumed_app_call_gets_its_output_as_text_even_empty(self):
        for output, answered in (
            ("", ""),
            (None, FakeTaskContext().texts.owed_output),
            (PARTS, "Teas.\nHot."),
            ([{"id": "o1"}], '[{"id": "o1"}]'),
            ([], "[]"),
            ({"stock": 3}, '{"stock": 3}'),
        ):
            with self.subTest(output=output):
                session = await self.open()
                call = asyncio.ensure_future(self.run.door.call("get_menu", {}))
                [forwarded] = items_of(await self.rest(), "function_call")
                if output is not None:
                    outputs = [{"type": "function_call_output", "call_id": forwarded["call_id"], "output": output}]
                    self.task.give_outputs(outputs)
                await self.run.resume()
                self.assertEqual(await call, answered)
                session.end(1, usage=USAGE)

    async def test_a_misrouted_app_function_fails_with_its_output(self):
        task = FakeTaskContext()
        task.texts = task.texts.model_copy(update={"misrouted_output": "{name} belongs to the app."})
        # The IM routes even the app's function to the backend, which it never should.
        task.route = lambda call: "backend"
        session, collector = await self.opened(task=task)
        misrouted = json.dumps({"error": "get_menu belongs to the app."})
        self.assertEqual(await self.run.door.call("get_menu", {}), misrouted)
        session.answer(1, "m2", "I could ", "not.", usage=USAGE)
        await collector
        [(call_id, _)] = self.task.started
        self.assertEqual(self.task.outcomes[call_id], (misrouted, "failed"))


class SteeringTests(Harness):
    async def queue(self, items=STEERING, identity="steer_1"):
        self.task.steer_with(items, identity)
        self.run.steer()
        await until(
            lambda: (
                self.run.next_prompts
                or getattr(self.run, "sent", {})
                or getattr(self.run, "reprompts", [])
                or getattr(self.run.turn, "early", [])
            )
        )

    async def test_queued_inputs_wait_for_the_next_prompt_despite_native_tool_activity(self):
        for tool in ("already_ended", "still_running", None):
            with self.subTest(tool=tool):
                session, collector = await self.opened(harness=cooperative())
                if tool:
                    session.put(ThoughtChunk("m0"))
                    session.put(ToolCard("tc1", "lookup", {}))
                    await until(lambda: bool(self.run.cards_open))
                if tool == "already_ended":
                    session.put(ToolCardEnd("tc1", False, "found"))
                    await until(lambda: self.run.tool_rows)
                await self.queue()
                if tool == "still_running":
                    session.put(ToolCardEnd("tc1", False, "found"))
                if tool:
                    session.say("old", "Old result.", thought=False)
                    await until(lambda: "old" in self.run.messages)
                else:
                    session.put(ThoughtChunk("m0"))
                    await until(session.mailbox.empty)
                self.assertEqual((self.task.settled, session.texts("steer"), session.cancels), ([], [], 0))
                session.end(1, usage=USAGE)
                await until(lambda session=session: len(session.texts("prompt")) == 2)
                self.assertEqual(self.task.settled, [("steer_1", "applied")])
                session.answer(2, "m2", "Changed result.", usage=LATER_USAGE)
                events = await collector
                self.assertEqual((answers(events, "commentary"), answers(events)), ([], ["Changed result."]))
                self.assertEqual(self.host.runtime.client.sessions, [session])

    async def test_verified_native_steering_releases_only_the_corrected_reply(self):
        session, collector = await self.opened()
        receipt = asyncio.get_running_loop().create_future()
        second = asyncio.get_running_loop().create_future()
        submit = mock.Mock(side_effect=[receipt, second])
        session.native_steering = SimpleNamespace(submit=submit)
        self.task.steer_with(STEERING, "steer_1")
        self.run.steer()
        await until(lambda: submit.called)
        state = session.prompt_state(1)
        submit.assert_called_once_with(1, state.invocation_id, "steer_1", mock.ANY)

        self.task.steer_with(NEWER, "steer_2")
        self.run.steer()
        await until(lambda: self.run.next_prompts)
        self.assertEqual(submit.call_count, 1)
        session.say("obsolete", "Monday.", thought=False)
        await until(lambda: session.mailbox.empty())
        self.assertEqual(self.task.settled, [])
        self.assertNotIn("obsolete", self.run.messages)
        receipt.set_result(NativeSteerReceipt(state.invocation_id, "steer_1", "applied", "api-attempt-2"))
        await until(lambda: submit.call_count == 2)
        self.assertEqual(submit.call_args.args[:3], (1, state.invocation_id, "steer_2"))
        session.say("obsolete-first-then-second", "Tuesday.", thought=False)
        await until(lambda: session.mailbox.empty())
        second.set_result(NativeSteerReceipt(state.invocation_id, "steer_2", "applied", "api-attempt-3"))
        await until(lambda: len(self.task.settled) == 2)
        session.answer(1, "corrected", "Wednesday.", usage=USAGE)

        events = await collector
        self.assertEqual(self.task.settled, [("steer_1", "applied"), ("steer_2", "applied")])
        self.assertEqual(texts(events), ["Wednesday."])
        self.assertEqual((len(session.texts("prompt")), session.cancels), (1, 0))

    async def test_native_steering_receipt_cannot_settle_input_on_another_invocation(self):
        session, collector = await self.opened()
        receipt = asyncio.get_running_loop().create_future()
        submit = mock.Mock(return_value=receipt)
        session.native_steering = SimpleNamespace(submit=submit)
        self.task.steer_with(STEERING, "steer_1")
        self.run.steer()
        await until(lambda: submit.called)
        receipt.set_result(NativeSteerReceipt("a-different-invocation", "steer_1", "applied", "api-attempt-2"))
        await until(lambda: self.task.settled)
        session.answer(1, "obsolete", "Monday.", usage=USAGE)

        with self.assertRaisesRegex(HarnessTurnError, "native steering delivery is unknown"):
            await collector
        self.assertEqual(self.task.settled, [("steer_1", "unknown")])
        self.assertEqual(texts(self.seen), [])
        self.assertEqual(len(session.texts("prompt")), 1)

    async def test_hermes_fallback_reprompts_the_same_session_with_unapplied_input(self):
        for native_steer in (False, True):
            with self.subTest(native_steer=native_steer):
                await self.check_fallback(native_steer)

    async def check_fallback(self, native_steer):
        session, collector = await self.opened()
        if native_steer:
            receipt = asyncio.get_running_loop().create_future()
            submit = mock.Mock(return_value=receipt)
            session.native_steering = SimpleNamespace(submit=submit)
            self.task.steer_with(STEERING, "steer_1")
            self.run.steer()
            await until(lambda: submit.called)
            self.assertEqual(session.cancels, 0)
            state = session.prompt_state(1)
            receipt.set_result(NativeSteerReceipt(state.invocation_id, "steer_1", "not_applied"))
        else:
            await self.queue()
        await until(lambda: session.cancels == 1)
        self.assertEqual(self.task.settled, [])
        session.say("obsolete", "Old result.")
        session.end(1, "cancelled")
        await self.assert_reprompt_answer(session, collector)

    async def assert_reprompt_answer(self, session, collector):
        await until(lambda: len(session.texts("prompt")) == 2)
        self.assertIn("Make it two.", session.texts("prompt")[1])
        self.assertEqual(self.host.runtime.client.sessions, [session])
        session.answer(2, "corrected", "Updated result.")
        self.assertEqual(texts(await collector), ["Updated result."])
        self.assertEqual(self.task.settled, [("steer_1", "applied")])

    async def test_bridge_does_not_adopt_a_tool_request_received_before_follow_up_dispatch(self):
        tools = RecordingTools()
        session, collector = await self.opened(task=FakeTaskContext(tools=tools), harness=cooperative())
        await self.queue()
        session.hold_next_prompt = True
        session.end(1, usage=USAGE)
        await until(lambda: session.dispatch_pending is not None)
        door = self.run.door
        arrived_early = door.capture_call()

        session.dispatch_pending()
        await until(lambda: self.task.settled == [("steer_1", "applied")])
        self.assertEqual(await arrived_early("lookup", {}), self.task.texts.not_run_output)
        self.assertEqual(tools.calls, [])

        current = door.capture_call()
        self.assertEqual(json.loads(await current("lookup", {})), {"ok": True, "tool": "lookup"})
        self.assertEqual(len(tools.calls), 1)
        session.answer(2, "m2", "Updated result.", usage=LATER_USAGE)
        self.assertEqual(answers(await collector), ["Updated result."])

    async def test_a_failed_native_turn_does_not_prompt_queued_input_on_the_failed_session(self):
        for stop, error in (("refusal", None), (None, "RPC: admitted_turn_failed")):
            with self.subTest(stop=stop, error=error):
                session, collector = await self.opened()
                session.put(ThoughtChunk("m0"))
                await self.queue()
                if error:
                    session.say("m0", "The old response.", thought=False)
                session.end(1, stop, USAGE, error=error)
                with self.assertRaisesRegex(HarnessTurnError, error or stop):
                    await asyncio.wait_for(collector, 1)
                self.assertNotIn("The old response.", texts(self.seen))
                self.assertEqual(len(session.texts("prompt")), 1)
                self.assertEqual(self.task.settled, [("steer_1", "not_applied")])
                self.assertTrue(session.retired)

    async def test_explicit_stop_wins_over_queued_input(self):
        session, collector = await self.opened()
        await self.queue()
        stopping = asyncio.create_task(self.run.stop())
        await until(lambda: session.cancels == 1)
        session.end(1, "cancelled", USAGE)
        self.assertEqual(await stopping, "stopped")
        await collector
        self.assertEqual((len(session.texts("prompt")), self.task.settled), (1, [("steer_1", "not_applied")]))
        self.assertTrue(session.retired)

    async def test_a_stop_while_the_old_answer_closes_prevents_the_next_prompt(self):
        session = await self.open()
        self.task.steer_with(STEERING)
        self.run.steer()
        session.answer(1, "m1", "Old answer.", usage=USAGE)
        async for event in self.events:
            self.seen.append(event)
            if event["type"] == "response.completed" and not self.run.turn.main_open:
                self.assertEqual(await self.run.stop(), "stopped")
        self.assertEqual(
            (len(session.texts("prompt")), self.task.settled, session.cancels), (1, [("steer_1", "not_applied")], 0)
        )
        self.assertTrue(session.retired)

    async def test_steering_while_the_session_opens_joins_the_prompt(self):
        run = self.make()
        self.task.steer_with(STEERING)
        run.steer()
        self.events = run.events()
        self.seen = [await anext(self.events)]
        session = self.host.runtime.client.sessions[-1]
        [first] = session.texts("prompt")
        self.assertIn("Make it two.", first)
        self.assertEqual(self.task.settled, [])
        collector = self.collecting()
        session.answer(1, "m1", "Two it is.", usage=USAGE)
        self.assertEqual(answers(await collector), ["Two it is."])


class StopTests(Harness):
    async def test_release_finishes_the_reader_without_starting_another_round(self):
        native = await self.open()
        await self.run.release()
        self.assertIsNone(await anext(self.events, None))
        self.assertEqual(self.host.runtime.places, 1)
        native.end(1, "cancelled", USAGE)
        await until(lambda: self.host.runtime.places == 0)

    async def test_a_stop_cancels_the_prompt_and_retires_the_door(self):
        session = await self.open()
        held = asyncio.ensure_future(self.run.door.call("get_menu", {}))
        await until(lambda: self.run.door.held)
        self.assertEqual(self.run.forwarded, [])
        stopping = asyncio.ensure_future(self.run.stop())
        await until(lambda: session.cancels == 1)
        self.assertEqual(await held, self.task.texts.stopped_output)
        session.end(1, "cancelled", USAGE)
        self.assertEqual(await stopping, "stopped")
        self.assertTrue(session.retired)
        self.assertEqual(await self.rest(), self.seen)

    async def test_a_turn_past_its_limit_is_cancelled_and_fails(self):
        session = await self.open(turn_timeout_seconds=0.2)
        error = await self.failure()
        self.assertIn("ran past", str(error))
        self.assertEqual(session.cancels, 1)

    async def test_a_marked_tasks_turn_runs_past_the_turn_limit(self):
        await self.check_marked_turn(initially_marked=True)

    async def test_a_turn_whose_work_is_marked_while_it_waits_runs_past_the_turn_limit(self):
        await self.check_marked_turn(initially_marked=False)

    async def check_marked_turn(self, *, initially_marked):
        task = FakeTaskContext(after_call=initially_marked)
        session, collector = await self.opened(task=task, turn_timeout_seconds=0.2)
        # The marked run starts waiting first, so a time limit would expire before the later run's.
        await turns()
        task.after_call = True
        unmarked = WorkerRun(FakeTaskContext(work_id="W2", task_id="W2.1"), self.host)
        with self.assertRaisesRegex(HarnessTurnError, "ran past"):
            await read_all(unmarked.events())
        self.assertFalse(collector.done())
        session.answer(1, "m1", "Do", "ne.", usage=USAGE)
        self.assertEqual(answers(await collector), ["Done."])
        self.assertEqual(session.cancels, 0)

    async def test_a_turn_whose_task_call_raises_is_cancelled(self):
        task = FakeTaskContext(declared=())
        task.progress = mock.Mock(side_effect=RuntimeError("the IM failed"))
        session = await self.open(task=task)
        session.put(PlanUpdate(("search (in_progress)",)))
        with self.assertRaisesRegex(RuntimeError, "the IM failed"):
            await self.rest()
        self.assertEqual((session.outstanding, session.cancels, session.retired), (frozenset({1}), 1, True))

    async def test_an_unconfirmed_stop_holds_the_turn_until_it_ends(self):
        # Hermes runs at most turns_per_process turns in one process, and a turn holds its place until it ends: a stop
        # the runtime does not confirm leaves the turn running, so its place is not free yet.
        session = await self.open()
        runtime = self.host.runtime
        self.assertEqual(runtime.places, 1)
        self.assertEqual(await self.run.stop(), "unconfirmed")
        self.assertEqual(runtime.places, 1)
        session.end(1, "cancelled", USAGE)
        await until(lambda: runtime.places == 0)


async def submit_start(door, source, brief):
    return await door.call(
        "submit_decision", {"actions": [{"action": "start", "delegations": [source], "tasks": [{"brief": brief}]}]}
    )


class Runs:
    """The worker runs on ``host`` as an IM's backend, ``max_running`` tasks at a time.

    ``placer`` places each request; the steer rule does unless one is given.
    """

    def __init__(self, host, placer=None, max_running=1):
        self.host, self.runs, self.placer, self.max_running = host, [], placer, max_running

    def conductor(self):
        return self.placer or SteerOpenWork()

    def start(self, task):
        self.runs.append(WorkerRun(task, self.host))
        return self.runs[-1]

    async def close(self):
        pass


def one_thread(count):
    """Return a conductor that starts W1 with D1, then places each later request in the Work before it, ended by then.

    So each later Work continues the one before it, D2's W1 and D3's W2.
    """
    later = [placing(Steer((f"D{number + 1}",), (WorkTarget(f"W{number}"),))) for number in range(1, count)]
    return planned(Start(("D1",)), *later)


class Conducted(Runs):
    """The worker runs and the harness conductor on ``host`` as an IM's backend, three tasks at a time."""

    def __init__(self, host):
        super().__init__(host, max_running=3)
        self.harness = HarnessConductor(host)

    def conductor(self):
        return self.harness

    def release_idle(self):
        for kept in self.host.pool.close():
            kept.retire(self.host.bridge)

    async def close(self):
        self.host.closed = True
        await self.harness.close()


class UnderTheImTests(IMCase):
    """The worker runs under the real IM: its records, the commit gate and the session kept between Works."""

    def setUp(self):
        super().setUp()
        self.host = FakeHost(settings())
        self.client = self.host.runtime.client
        self.runs = Runs(self.host)

    async def prompted(self, count, index=0):
        """Return session ``index`` once it has been sent ``count`` prompts."""
        sessions = self.client.sessions
        await until(lambda: len(sessions) > index and len(sessions[index].texts("prompt")) >= count)
        return sessions[index]

    async def test_a_native_turn_without_a_decision_gets_one_core_repair_before_routing_fails(self):
        for disconnected, repaired in itertools.product((False, True), repeat=2):
            with self.subTest(disconnected=disconnected, repaired=repaired):
                self.host = FakeHost(settings())
                self.client, self.runs = self.host.runtime.client, Conducted(self.host)
                im = self.session(self.runs)

                def end_without_decision(session, disconnected):
                    if disconnected:
                        self.host.runtime.client.alive = False
                        replacement = type(self.host.runtime)()
                        replacement.client.sessions = self.client.sessions
                        self.host.runtime = replacement
                        session.end(1, None, error="connection lost", terminal_confirmed=False)
                    else:
                        session.answer(1, "ack", "I will keep working.", usage=DECIDED)

                admitted = im.admit(ask("Find a dinner table."), delegation_id="D1")
                first = await self.prompted(1)
                old_door = self.runs.harness.door
                end_without_decision(first, disconnected)
                fresh = await self.prompted(1, index=1)
                self.assertTrue(first.retired and old_door.retired)
                late = await submit_start(old_door, "D1", "Late decision.")
                self.assertEqual(late, old_door.vocabulary.idle)
                self.assertFalse(im.works)
                self.assertIn(json.dumps({"placement_attempt": 2}), fresh.texts("prompt")[0])
                reason = "connection closed before a decision" if disconnected else "one complete decision call"
                self.assertIn(reason, fresh.texts("prompt")[0])
                if repaired:
                    await submit_start(self.runs.harness.door, "D1", "Find a dinner table.")
                    fresh.end(1, usage=DECIDED)
                    worker = await self.prompted(1, index=2)
                    worker.answer(1, "answer", "A table is available.", usage=USAGE)
                else:
                    end_without_decision(fresh, disconnected)
                events = await read_all(admitted.stream, timeout=2)
                self.assertEqual(events[-1]["type"], "response.completed")
                if not repaired:
                    self.assertEqual(answers(events), [im.texts.routing_rejected_output])
                self.assertEqual(self.runs.harness.round, 2)
                self.assertEqual(len(self.runs.runs), int(repaired))
                await im.close()

    async def test_unqualified_native_control_loss_fails_only_its_dispatched_source_without_replay(self):
        self.runs = Conducted(self.host)
        im = self.session(self.runs)
        first = im.admit(ask("Find a dinner table."), delegation_id="D1")
        control = await self.prompted(1)
        await im.set_attachment(False)
        events = await read_all(first.stream)
        self.assertEqual(answers(events), [im.texts.owed_output])
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertEqual(im.ledger.delegations["D1"].state, "failed")
        self.assertTrue(control.retired)
        self.assertEqual(control.cancels, 1)
        self.assertFalse(im.arrivals)
        await im.set_attachment(True)
        await turns()
        self.assertFalse(self.runs.runs)
        self.assertFalse(im.works)
        self.assertEqual([r["delegation_id"] for r in self.traced("im.controls.uncertain")], ["D1"])

    async def test_native_next_prompt_learns_a_question_source_alias_without_early_delivery(self):
        from examples.frontend_backend_live.delegation_core.im.tools import Invocation

        self.host = FakeHost(cooperative())
        self.client, self.runs = self.host.runtime.client, Runs(self.host)
        im = self.session(self.runs)
        im.admit(ask("Find a coffee."), delegation_id="D1")
        native = await self.prompted(1)
        run = self.runs.runs[0]
        self.assertEqual(
            json.loads(await run.door.call("caller_message", {"kind": "question", "message": "What size?"}))[
                "question_id"
            ],
            "W1.1.Q1",
        )
        reply = im.admit(ask("Large."), delegation_id="D2")
        await until(lambda run=run: run.next_prompts)
        [pending] = run.next_prompts
        self.assertEqual((native.cancels, native.texts("steer")), (0, []))
        envelope = pending.items[-1]
        [source] = json.loads(envelope["content"][0]["text"])["private_backend_data"]["source_message_ids"]
        self.assertIn(source, self.host.renderer.steering(pending))
        self.assertEqual(run.task.input_revision, 1)
        resolution = {"resolutions": [{"question_id": "W1.1.Q1", "disposition": "answered", "input_ref": source}]}
        refused = await run.door.call("resolve_questions", resolution)
        self.assertIsInstance(refused, RefusedReply)
        self.assertEqual(refused.text, im.texts.not_run_output)
        self.assertTrue(refused.cooperative_end)
        with self.assertRaisesRegex(ValueError, "newer applied"):
            run.task.resolve_questions(Invocation("early", "resolve_questions", resolution))
        native.answer(1, "old-final", "The old small coffee result.", usage=USAGE)
        await self.prompted(2)
        self.assertEqual((len(self.runs.runs), len(self.client.sessions), native.cancels), (1, 1, 0))
        self.assertEqual(run.task.work.carrier, "D2")
        self.assertIn(source, native.texts("prompt")[1])
        self.assertIn(json.dumps(envelope["content"][0]["text"]), native.texts("prompt")[1])
        await until(lambda: run.task.input_revision == 2)
        self.assertEqual(run.task.input_revision, 2)
        resolved = json.loads(await run.door.call("resolve_questions", resolution))
        self.assertEqual(resolved["resolutions"][0]["input_ref"], "W1.1.I2")
        self.assertEqual(run.task.record.questions[0].input_ref, "W1.1.I2")
        native.answer(2, "m2", "A large coffee.", usage=LATER_USAGE)
        self.assertEqual(answers(await read_all(reply.stream)), ["A large coffee."])

    async def test_the_door_answers_a_repeat_from_the_record(self):
        im = self.session(self.runs)
        tools = {"tools": [ORDER_FUNCTION]}
        first = im.admit(ask("Find my order."), delegation_id="D1", settings=tools)
        await until(lambda: self.runs.runs and self.runs.runs[0].door is not None and self.client.sessions)
        door, session = self.runs.runs[0].door, self.client.sessions[-1]
        lookup = {"order_id": "BB-1042"}
        answered = asyncio.ensure_future(door.call("get_order", lookup))
        [call] = items_of(await read_all(first.stream), "function_call")
        output = [{"type": "function_call_output", "call_id": call["call_id"], "output": "Error: order not found"}]
        rest = im.admit_client_request(output, delegation_id="D1", settings=tools)
        self.assertEqual(await answered, "Error: order not found")
        repeated = await asyncio.wait_for(door.call("get_order", lookup), 5)
        self.assertIsInstance(repeated, RefusedReply)
        self.assertEqual(repeated.text, im.texts.repeated_call_output.format(output="Error: order not found"))
        self.assertFalse(repeated.cooperative_end)
        session.answer(1, "m2", "Please spell ", "your order number.", usage=USAGE)
        events = await read_all(rest.stream, timeout=5)
        self.assertEqual(items_of(events, "function_call"), [])
        self.assertEqual(answers(events), ["Please spell your order number."])

    async def test_an_own_tool_run_while_steering_waits_is_reported(self):
        self.host = FakeHost(cooperative())
        self.client, self.runs = self.host.runtime.client, Runs(self.host)
        im = self.session(self.runs)
        first = im.admit(ask("Find teas."), delegation_id="D1")
        await until(lambda: self.runs.runs and self.client.sessions)
        session = self.client.sessions[0]
        session.say("m1", "Searching ", "now.")
        await self.turns()
        newer = im.admit(ask("Only green ones."), delegation_id="D2")
        await until(lambda: self.runs.runs[0].next_prompts)
        self.assertEqual(session.cancels, 0)
        session.put(ToolCard("tc9", "web_search", {"query": "teas"}, bridged=False))
        session.put(ToolCardEnd("tc9", False, "green, black, oolong"))
        await self.turns()
        session.end(1, usage=USAGE)
        await self.prompted(2)
        session.answer(2, "m2", "Green ", "teas.", usage=LATER_USAGE)
        events = await read_all(first.stream, timeout=5)
        await read_all(newer.stream, timeout=5)
        await until(lambda: not im.works)
        [item] = items_of(events, "mcp_call")
        self.assertEqual((item["name"], item["status"]), ("web_search", "completed"))
        action = im.ledger.actions["tc9"]
        self.assertEqual((action.state, action.output), ("done", "green, black, oolong"))
        self.assertEqual(im.ended_works[-1].tasks[0].step_lines(), ("web_search: done",))

    async def test_the_next_works_task_takes_the_kept_session(self):
        im = self.session(self.runs)
        first = im.admit(ask("Find teas."), delegation_id="D1")
        session = await self.prompted(1)
        session.answer(1, "m1", "Green ", "tea.", usage=USAGE)
        self.assertEqual(answers(await read_all(first.stream)), ["Green tea."])
        await until(lambda: not im.works)
        second = im.admit(ask("And black?"), delegation_id="D2")
        await self.prompted(2)
        self.assertEqual(len(self.client.sessions), 1)
        prompt = session.texts("prompt")[1]
        self.assertTrue(prompt.startswith("And black?\n"))
        self.assertNotIn("Green tea.", prompt)
        session.answer(2, "m2", "Black ", "too.", usage=LATER_USAGE)
        self.assertEqual(answers(await read_all(second.stream)), ["Black too."])
        kinds = [kind for kind, _ in self.traces]
        self.assertEqual([kinds.count(kind) for kind in ("harness.session_opened", "harness.session_reused")], [1, 1])

    async def test_a_caller_who_says_yes_twice_keeps_one_session(self):
        # Under the steer rule each request continues the Work before it, and its id tells it from the other "Yes.":
        # the session that read the thread is sent only the newest request.
        im = self.session(self.runs)
        newest = []
        for number, text in enumerate(("Book a table.", "Yes.", "Yes."), 1):
            admitted = im.admit(ask(text), delegation_id=f"D{number}")
            session = await self.prompted(number)
            history = self.runs.runs[-1].task.history()
            source = next(index for index, item in enumerate(history) if item.get("id") == f"D{number}")
            newest.append(history[source:])
            session.answer(session.last_prompt(), f"m{number}", "Done.", usage={"totalTokens": 100 * number})
            await read_all(admitted.stream)
            await until(lambda: not im.works)
        self.assertEqual(len(self.client.sessions), 1)
        self.assertEqual(session.texts("prompt")[1:], [self.host.renderer.transcript(items) for items in newest[1:]])

    async def test_two_requests_of_the_same_words_are_told_apart(self):
        # W2's "Yes." and answer join the conversation before W1's: W1's session read only its own "Yes.".
        placer = planned(Start(("D1",)), Start(("D2",)), Steer(("D3",), (WorkTarget("W1"),)))
        self.runs = Runs(self.host, placer, max_running=3)
        im = self.session(self.runs)
        table = im.admit(ask("Yes."), delegation_id="D1")
        ours = await self.prompted(1)
        taxi = im.admit(ask("Yes."), delegation_id="D2")
        theirs = await self.prompted(1, index=1)
        for session, admitted, text in ((theirs, taxi, "Your taxi is booked."), (ours, table, "Your table is booked.")):
            session.answer(1, "m1", text, usage=USAGE)
            await read_all(admitted.stream)
        last = im.admit(ask("And the wifi password?"), delegation_id="D3")
        await self.prompted(2)
        history = self.runs.runs[-1].task.history()
        self.assertEqual([item.get("id") for item in history if item.get("role") == "user"], ["D1", "D3"])
        source = next(index for index, item in enumerate(history) if item.get("id") == "D3")
        self.assertEqual(ours.texts("prompt")[1], self.host.renderer.transcript(history[source:]))
        self.assertEqual(len(self.client.sessions), 2)
        ours.answer(2, "m2", "It is on the card.", usage=LATER_USAGE)
        await read_all(last.stream)

    async def test_a_follow_up_of_one_split_task_never_reads_its_siblings_brief_or_answer(self):
        # D2 continues W1.1, whose session read the request and its own brief: each brief has an id of its own.
        placer = planned(
            placing(
                Start(
                    ("D1",),
                    (
                        TaskSpec("teas"),
                        TaskSpec("hours"),
                    ),
                )
            ),
            Steer(("D2",), (WorkTarget("W1", ("W1.1",)),)),
        )
        self.runs = Runs(self.host, placer, max_running=2)
        im = self.session(self.runs)
        first = im.admit(ask("Teas and hours."), delegation_id="D1")
        await self.prompted(1, index=1)
        teas, hours = self.runs.runs
        sessions, hours_brief = [teas.session, hours.session], hours.task.history()[-1]
        for session, text in zip(sessions, ("Green tea.", "Open till six."), strict=True):
            session.answer(1, "m1", text, usage=USAGE)
        await read_all(first.stream)
        await until(lambda: not im.works)
        last = im.admit(ask("And black tea?"), delegation_id="D2")
        await until(lambda: len(sessions[0].texts("prompt")) == 2)
        history = self.runs.runs[-1].task.history()
        self.assertNotIn(hours_brief, history)
        self.assertNotIn("Open till six.", str(history))
        source = next(index for index, item in enumerate(history) if item.get("id") == "D2")
        self.assertEqual(sessions[0].texts("prompt")[1], self.host.renderer.transcript(history[source:]))
        self.assertEqual(len(self.client.sessions), 2)
        sessions[0].answer(2, "m2", "Black tea too.", usage=LATER_USAGE)
        await read_all(last.stream)

    async def typed_with_the_outputs(self, harness, typed):
        """Have D1's task ask the app for its menu, and send the app's outputs with ``typed``; a later D2 continues W1.

        Return the session, the held call's reply and D1's continuation.
        """
        self.host = FakeHost(harness)
        self.client, self.runs = self.host.runtime.client, Runs(self.host, one_thread(2))
        self.im = self.session(self.runs)
        first = self.im.admit(ask("Find teas."), delegation_id="D1", settings=MENU_APP)
        session = await self.prompted(1)
        session.put(ToolCard("tc1", "mcp__w1__get_menu", {}))
        call = asyncio.ensure_future(self.runs.runs[0].door.call("get_menu", {}))
        outputs = app_outputs(await read_all(first.stream), "teas")
        rest = self.im.admit_client_request([*outputs, typed], delegation_id="D1", settings=MENU_APP)
        return session, call, rest

    def steering_states(self):
        return [steering.state for steering in self.im.ledger.steerings.values()]

    async def test_typed_input_with_tool_outputs_waits_for_the_next_native_prompt(self):
        await self.check_typed_output(MILK, reply="Green tea with milk.", text="With milk, please.")

    async def test_a_typed_message_with_no_text_is_applied_at_once(self):
        await self.check_typed_output(IMAGE, reply="Green tea.")

    async def check_typed_output(self, item, *, reply, text=None):
        for name, harness in (("hermes", settings()), ("plain", plain())):
            with self.subTest(name=name):
                session, call, rest = await self.typed_with_the_outputs(harness, item)
                self.assertEqual(await asyncio.wait_for(call, 1), "teas")
                if text:
                    run = self.runs.runs[0]
                    await until(lambda run=run: run.next_prompts)
                    self.assertEqual(self.steering_states(), ["taken"])
                self.assertEqual((session.texts("steer"), session.cancels), ([], 0))
                session.put(ToolCardEnd("tc1", False, "teas"))
                if text:
                    session.answer(1, "old-final", "Old plain tea.", usage=USAGE, completion_calls=1)
                    await self.prompted(2)
                    self.assertEqual(self.steering_states(), ["applied"])
                    session.answer(2, "m2", reply, usage=LATER_USAGE, completion_calls=1)
                else:
                    session.answer(1, "m2", reply, usage=USAGE)
                self.assertEqual(answers(await read_all(rest.stream)), [reply])
                self.assertEqual(self.steering_states(), ["applied"])
                self.assertEqual(session.texts("prompt")[1:], [text] if text else [])
                self.assertEqual((len(self.client.sessions), session.texts("steer"), session.cancels), (1, [], 0))

    async def test_each_call_the_gate_refuses_is_traced_once(self):
        # The run routes a call as the agent makes it, and the IM again as it reads the round.
        self.host = FakeHost(cooperative(), tools=(RecordingTools(),))
        self.client, self.runs = self.host.runtime.client, Runs(self.host)
        im = self.session(self.runs)
        im.admit(ask("Find teas."), delegation_id="D1", settings=MENU_APP)
        session = await self.prompted(1)
        newer = im.admit(ask("Only green ones."), delegation_id="D2", settings=MENU_APP)
        await until(lambda: self.runs.runs[0].next_prompts)
        self.assertEqual(session.cancels, 0)
        door = self.runs.runs[0].door
        refused = await asyncio.gather(*(door.call(name, {}) for name in ("get_menu", "lookup")))
        self.assertTrue(all(isinstance(reply, RefusedReply) and reply.cooperative_end for reply in refused))
        self.assertEqual([reply.text for reply in refused], [im.texts.not_run_output] * 2)
        await until(lambda: session.cancels == 1)
        session.end(1, "cancelled", usage=USAGE)
        fresh = await self.prompted(2)
        self.assertIs(fresh, session)
        fresh.say("m2", "Green ", "teas.", thought=False)
        fresh.end(2, usage=LATER_USAGE)
        self.assertEqual(answers(await read_all(newer.stream, timeout=5)), ["Green teas."])
        refused = [(fields["name"], fields["reason"]) for kind, fields in self.traces if kind == "tools.refused"]
        self.assertCountEqual(refused, [("get_menu", "not_run"), ("lookup", "not_run")])

    async def test_a_hang_up_sends_no_cancel_to_the_conductors_running_turn(self):
        # The job runs on after the call, so the backend stays open. The agent may be compacting its memory: its turn
        # runs on, and its session waits for a later call.
        self.runs = Conducted(self.host)
        im = self.session(self.runs)
        im.continuable = True
        conductor = self.runs.harness
        im.admit(ask("Research every supplier."), delegation_id="D1")
        await until(lambda: conductor.session is not None and conductor.session.outstanding)
        session = conductor.session
        await conductor.door.call(
            "submit_decision",
            {"actions": [{"action": "start", "delegations": ["D1"], "tasks": [{"brief": "Caller request."}]}]},
        )
        session.end(1, usage=USAGE)
        await until(lambda: self.runs.runs and self.runs.runs[0].door is not None)
        worker = self.runs.runs[0]
        mark = {"message": "I'll keep working on it after we hang up."}
        self.assertEqual(await worker.door.call("continue_after_call", mark), im.texts.continued_output)
        im.admit(ask("What's on my calendar?"), delegation_id="D2")
        await until(lambda: len(session.texts("prompt")) == 2)
        self.assertIs(await im.leave(), True)
        self.assertEqual((session.cancels, session.retired, conductor.session), (0, False, session))
        self.assertIsNotNone(conductor.turn)
        # The decide() that prompted it ended with the call: what the turn places now is refused.
        late = await conductor.door.call(
            "submit_decision",
            {"actions": [{"action": "start", "delegations": ["D2"], "tasks": [{"brief": "Caller request."}]}]},
        )
        self.assertEqual(json.loads(late), {"status": "superseded", "executed": False})
        self.assertFalse(conductor.turn.control_completed)


class SessionReuseTests(unittest.IsolatedAsyncioTestCase):
    """The idle sessions of tasks that answered: a later task of their thread takes the first that fits.

    A task of another thread, or one that no idle session fits, opens a fresh one.
    """

    def setUp(self):
        # An agent's running token total, which grows with each model call.
        self.totals = itertools.count(200, 100)
        self.use(settings())

    def use(self, harness):
        self.host = FakeHost(harness, tools=(RecordingTools(),))
        self.client = self.host.runtime.client

    async def start(self, task):
        """Run ``task`` on the host up to its first event, its session opened or taken over and prompted."""
        run = WorkerRun(task, self.host)
        events = run.events()
        self.addAsyncCleanup(close_events, events)
        self.assertEqual((await anext(events))["type"], "response.created")
        return run, events

    async def answer(self, events, session, message_id="m1", usage=None):
        """Have ``session`` answer, its token total grown unless ``usage`` is given; return the rest of the events."""
        session.answer(
            session.last_prompt(), message_id, "Green ", "tea.", usage=usage or {"totalTokens": next(self.totals)}
        )
        return [event async for event in events]

    async def answered(self, task=None):
        """Answer a first task; return its session and the task's view as the IM keeps it, its run's items joined."""
        self.first = task or FakeTaskContext()
        _, events = await self.start(self.first)
        session = self.client.sessions[-1]
        rest = await self.answer(events, session, usage=USAGE)
        return session, [*self.first.history(), *finished_items(rest)]

    async def answered_together(self, *tasks):
        """Start ``tasks`` side by side, then answer them in that order; return their sessions and their items."""
        started = [await self.start(task) for task in tasks]
        sessions = [run.session for run, _ in started]
        items = [finished_items(await self.answer(events, run.session)) for run, events in started]
        return sessions, items

    def next_task(self, view, **kwargs):
        """Return W2's task, which continues W1 unless ``kwargs`` say otherwise: ``view``, then the next request."""
        return FakeTaskContext(history=[*view, NEXT], work_id="W2", task_id="W2.1", **{"continues": "W1", **kwargs})

    async def test_an_answered_session_takes_the_next_task(self):
        session, view = await self.answered()
        self.assertEqual((self.host.idle(), session.retired), ([session], False))
        _, events = await self.start(self.next_task(view))
        self.assertEqual((len(self.client.sessions), self.host.idle()), (1, []))
        self.assertEqual(session.texts("prompt")[1], "And black tea?")
        rest = await self.answer(events, session, "m2", usage=LATER_USAGE)
        self.assertEqual(answers(rest), ["Green tea."])
        self.assertNotIn("usage", rest[-1]["response"])

    async def test_a_focused_continuation_rejects_shared_native_context_then_reuses_focused_context(self):
        shared = FakeTaskContext()
        shared.history_mode = "shared"
        original, view = await self.answered(shared)
        focused = self.next_task(view)
        focused.history_mode = "focused"
        run, events = await self.start(focused)
        focused_session = run.session
        self.assertIsNot(focused_session, original)
        self.assertEqual(len(self.client.sessions), 2)
        self.assertIn(original, self.host.idle())
        rest = await self.answer(events, focused_session)
        successor = FakeTaskContext(
            history=[*focused.history(), *finished_items(rest), {**NEXT, "id": "third-request"}],
            work_id="W3",
            task_id="W3.1",
            continues="W2",
        )
        successor.history_mode = "focused"
        next_run, next_events = await self.start(successor)
        self.assertIs(next_run.session, focused_session)
        self.assertEqual(len(self.client.sessions), 2)
        await self.answer(next_events, next_run.session, message_id="m2")

    async def test_a_stopped_failed_or_timed_out_session_is_not_kept(self):
        run, events = await self.start(FakeTaskContext())
        stopping = asyncio.ensure_future(run.stop())
        await until(lambda: self.client.sessions[-1].cancels == 1)
        self.client.sessions[-1].end(1, "cancelled", USAGE)
        self.assertEqual(await stopping, "stopped")
        self.assertEqual(self.host.idle(), [])
        _, events = await self.start(FakeTaskContext())
        self.client.sessions[-1].end(1, "cancelled", USAGE)
        with self.assertRaises(HarnessTurnError):
            [event async for event in events]
        self.assertEqual(self.host.idle(), [])
        self.use(settings(turn_timeout_seconds=0.2))
        _, events = await self.start(FakeTaskContext())
        with self.assertRaisesRegex(HarnessTurnError, "ran past"):
            [event async for event in events]
        self.assertEqual(self.host.idle(), [])
        self.assertTrue(all(session.retired for session in self.client.sessions))

    async def test_an_item_inserted_or_trimmed_is_no_bar_to_reuse(self):
        # An item inserted before the session's answer, such as another Work's, is sent after what the session read;
        # one the budget trimmed is still in the agent's memory.
        for name, cut, sent in (
            ("an item inserted", lambda view: [view[0], BETWEEN, *view[1:]], [BETWEEN, NEXT]),
            ("an item trimmed", lambda view: view[1:], [NEXT]),
        ):
            with self.subTest(name):
                self.use(settings())
                session, view = await self.answered()
                task = self.next_task(cut(view))
                run, events = await self.start(task)
                self.assertEqual((run.session, len(self.client.sessions)), (session, 1))
                self.assertEqual(session.texts("prompt")[1], self.host.renderer.transcript(sent))
                if name == "an item trimmed":
                    await self.check_trimmed_reuse(session, task, events)

    async def check_trimmed_reuse(self, session, task, events):
        self.assertEqual(task.traced("harness.session_reused")[0]["absent"], 1)
        result = await self.answer(events, session, message_id="m2", usage=LATER_USAGE)
        newest = {**NEXT, "id": "D3", "content": "And a coffee?"}
        third = FakeTaskContext(
            history=[*task.history(), *finished_items(result), newest],
            work_id="W3",
            task_id="W3.1",
            continues="W2",
        )
        run, events = await self.start(third)
        self.assertIs(run.session, session)
        self.assertEqual(third.traced("harness.session_reused")[0]["absent"], 1)
        self.assertEqual(session.texts("prompt")[2], newest["content"])
        await self.answer(events, session, message_id="m3", usage={"totalTokens": 300})

    async def test_a_task_the_idle_session_does_not_fit_opens_a_fresh_one_and_leaves_it_untouched(self):
        for why, cut, changes in (
            ("other instructions", lambda view: view, {"instructions": "OTHER-INSTRUCTIONS"}),
            ("another toolset", lambda view: view, {"declared": [MENU, ORDER_FUNCTION]}),
            # The idle session holds W1's work as its own turns: it stays for W1's thread, which may hold a question.
            ("no idle session of its thread", lambda view: view, {"continues": None}),
            # The budget kept only the newest request: where the session's reads end, the history does not show.
            ("its last answer not in the history", lambda view: [], {}),
        ):
            with self.subTest(why):
                self.use(settings())
                session, view = await self.answered()
                [kept] = self.host.pool.idle
                task = self.next_task(cut(view), **changes)
                run, _ = await self.start(task)
                self.assertIsNot(run.session, session)
                [opened] = task.traced("harness.session_opened")
                self.assertEqual(opened["why"], [why])
                self.assertEqual((self.host.idle(), len(session.sent), self.host.runtime.places), ([session], 1, 1))
                self.assertEqual((kept.door.task, self.host.bridge.closed), (self.first, []))

    async def test_a_continuing_task_takes_its_threads_session(self):
        for continues, taken in (("W1", 0), ("W2", 1), ("W2.1", 1)):
            with self.subTest(continues=continues):
                self.use(settings())
                asked = [request("D1", "Book a table."), request("D2", "Wifi password?")]
                works = [
                    FakeTaskContext(history=asked[0]),
                    FakeTaskContext(history=asked[1], work_id="W2", task_id="W2.1"),
                ]
                sessions, items = await self.answered_together(*works)
                history = [*asked[0], *items[0], *asked[1], *items[1], NEXT]
                task = FakeTaskContext(history=history, work_id="W3", task_id="W3.1", continues=continues)
                run, _ = await self.start(task)
                self.assertIs(run.session, sessions[taken])
                self.assertEqual(self.host.idle(), [sessions[1 - taken]])

    async def test_a_task_whose_threads_session_is_gone_opens_a_fresh_one_beside_a_siblings(self):
        # The pool keeps two: W1.1's session, idle longest, is retired. Its siblings' sessions ran other parts of W1.
        self.use(settings(max_running=2))
        asked, briefs = request("D1", "Tea, coffee and juice."), [brief("tea"), brief("coffee"), brief("juice")]
        split = [FakeTaskContext(history=[*asked, part], task_id=f"W1.{n}") for n, part in enumerate(briefs, 1)]
        sessions, items = await self.answered_together(*split)
        self.assertEqual((self.host.idle(), sessions[0].retired), (sessions[1:], True))
        segment = [*asked, *(item for part, own in zip(briefs, items, strict=True) for item in (part, *own))]
        task = FakeTaskContext(history=[*segment, NEXT], work_id="W2", task_id="W2.1", continues="W1.1")
        run, _ = await self.start(task)
        self.assertNotIn(run.session, sessions)
        [opened] = task.traced("harness.session_opened")
        self.assertEqual(opened["why"], ["no idle session of its thread"])
        self.assertEqual(self.host.idle(), sessions[1:])

    async def test_a_split_works_follow_up_takes_the_named_tasks_session_else_the_newest(self):
        asked, briefs = request("D1", "Menu and hours."), [brief("menu"), brief("hours")]
        for continues, taken in (("W1.1", 0), ("W1", 1)):
            with self.subTest(continues=continues):
                self.use(settings())
                split = [FakeTaskContext(history=[*asked, part], task_id=f"W1.{n}") for n, part in enumerate(briefs, 1)]
                sessions, items = await self.answered_together(*split)
                segment = [*asked, briefs[0], *items[0], briefs[1], *items[1]]
                task = FakeTaskContext(history=[*segment, NEXT], work_id="W2", task_id="W2.1", continues=continues)
                run, _ = await self.start(task)
                self.assertIs(run.session, sessions[taken])

    async def test_the_pool_keeps_max_running_sessions_and_retires_the_idle_longest(self):
        self.use(settings(max_running=2))
        tasks = [FakeTaskContext(task_id=f"W1.{n}") for n in (1, 2, 3)]
        sessions, _ = await self.answered_together(*tasks)
        self.assertEqual(self.host.idle(), sessions[1:])
        self.assertTrue(sessions[0].retired)
        self.assertEqual([door.name for door, _ in self.host.bridge.closed], ["w1"])
        [retired] = tasks[2].traced("harness.session_retired")
        self.assertEqual((retired["why"], retired["last_task"]), ("evicted", "W1.1"))

    async def test_a_history_over_the_budget_fails_and_leaves_the_pool_as_it_was(self):
        # The history is read before the walk, and again once a place is taken for the session that fits.
        for walked in (False, True):
            with self.subTest(walked=walked):
                self.use(settings())
                session, view = await self.answered()
                task = self.next_task(view)
                over = RuntimeError("Backend context budget exceeded")
                task.history = mock.Mock(side_effect=[task.history(), over] if walked else over)
                with self.assertRaisesRegex(RuntimeError, "budget exceeded"):
                    await self.start(task)
                self.assertEqual((self.host.idle(), session.retired, len(session.sent)), ([session], False, 1))
                self.assertEqual((self.host.runtime.places, self.host.bridge.closed), (0, []))

    async def test_a_batch_landed_in_both_split_tasks_is_sent_to_neither_session_again(self):
        # The segment holds the batch twice, each copy with the batch's id: each session read it with its own task, and
        # is sent only the other task's part and the new request.
        asked, batch = request("D1", "Two lattes."), request("D2", "Oat milk.")
        briefs = [brief("first"), brief("second")]
        for continues, taken in (("W1.1", 0), ("W1.2", 1)):
            with self.subTest(continues=continues):
                self.use(settings())
                split = [FakeTaskContext(history=[*asked, part], task_id=f"W1.{n}") for n, part in enumerate(briefs, 1)]
                for number, task in enumerate(split, 1):
                    task.steer_with(batch, f"steer_{number}")
                sessions, items = await self.answered_together(*split)
                self.assertEqual([task.settled for task in split], [[("steer_1", "applied")], [("steer_2", "applied")]])
                parts = [[part, *batch, *own] for part, own in zip(briefs, items, strict=True)]
                task = FakeTaskContext(
                    history=[*asked, *parts[0], *parts[1], NEXT], work_id="W2", task_id="W2.1", continues=continues
                )
                run, _ = await self.start(task)
                self.assertIs(run.session, sessions[taken])
                other = 1 - taken
                sent = [briefs[other], *items[other], NEXT]
                self.assertEqual(sessions[taken].texts("prompt")[-1], self.host.renderer.transcript(sent))

    async def test_a_reused_sessions_failed_first_prompt_moves_the_task_to_a_fresh_session(self):
        for note in ("", "Keep correction and tool receipts tied to the active job."):
            with self.subTest(note=bool(note)):
                await self.check_failed_reused_prompt(note)

    async def check_failed_reused_prompt(self, note):
        harness = cooperative() if note else settings()
        prompts = harness.prompts.model_copy(update={"worker_protocol": note})
        self.use(harness.model_copy(update={"prompts": prompts}))
        original = FakeTaskContext()
        instructions = original.request()["instructions"]
        session, view = await self.answered(original)
        first_instructions = (self.client.folders[0] / ".hermes.md").read_text(encoding="utf-8")
        if note:
            self.assertEqual(first_instructions.strip(), instructions + "\n\n" + note)
        task = self.next_task(view)
        session.reject_next_prompt = True
        run, events = await self.start(task)
        self.assertIs(run.session, session)
        self.assertEqual((len(self.client.sessions), len(session.texts("prompt"))), (1, 1))
        collector = asyncio.create_task(read_all(events))
        await until(lambda: len(self.client.sessions) == 2 and self.client.sessions[1].sent)
        fresh = self.client.sessions[1]
        self.assertTrue(session.retired)
        self.assertEqual(fresh.texts("prompt"), [self.host.renderer.worker_input(task.history())])
        fresh_instructions = (self.client.folders[-1] / ".hermes.md").read_text(encoding="utf-8")
        self.assertEqual(fresh_instructions, first_instructions)
        self.assertEqual((original.request()["instructions"], task.request()["instructions"]), (instructions,) * 2)
        fresh.answer(1, "m2", "Black too.", usage={"totalTokens": next(self.totals)})
        self.assertEqual(answers(await collector), ["Black too."])
        [retired], [opened] = task.traced("harness.session_retired"), task.traced("harness.session_opened")
        self.assertEqual((retired["why"], opened["why"]), ("a failed first prompt", ["a failed first prompt"]))
        self.assertEqual((self.host.idle(), self.host.runtime.places), ([fresh], 0))

    async def test_a_lost_idle_session_is_retired_on_the_walk(self):
        session, view = await self.answered()
        session.lost = True
        task = self.next_task(view)
        run, _ = await self.start(task)
        self.assertIsNot(run.session, session)
        self.assertEqual((session.retired, self.host.idle()), (True, []))
        [retired] = task.traced("harness.session_retired")
        self.assertEqual(retired["why"], "session lost")

    async def test_a_marked_tasks_session_is_retired_not_pooled(self):
        # No later task's history holds the steps of a Work that went on after the call.
        task = FakeTaskContext(after_call=True)
        run, events = await self.start(task)
        session = run.session
        await self.answer(events, session)
        self.assertEqual((session.retired, self.host.idle(), self.host.runtime.places), (True, [], 0))
        [retired] = task.traced("harness.session_retired")
        self.assertEqual(retired["why"], "a marked task")

    async def test_a_task_start_warns_once_of_a_tool_name_offered_twice(self):
        # The tool source's get_menu collides with the app's: the walk's checks over the idle sessions log nothing.
        self.host = FakeHost(settings(), tools=(RecordingTools(["get_menu"]),))
        self.client = self.host.runtime.client
        await self.answered_together(FakeTaskContext(), FakeTaskContext(task_id="W1.2"))
        with logged() as lines:
            await self.start(FakeTaskContext(work_id="W2", task_id="W2.1", continues="W1"))
        self.assertEqual(len(self.host.idle()), 2)
        collisions = [line for line in lines if line.startswith("harness.tool_collision")]
        self.assertEqual(len(collisions), 1)


def given(*requests, works=(), conversation=(), new_from=0):
    conversation = (*conversation, *(item for request in requests for item in request.input))
    return ConductorInput(tuple(requests), tuple(works), conversation, new_from)


class Unreadable:
    """An event whose reading raises: the agent's events failed while a ``decide()`` read them."""

    def apply(self, reader):
        raise RuntimeError("an unreadable event")


class ConductorTests(unittest.IsolatedAsyncioTestCase):
    async def submit_steer(self, work, task):
        envelope = {"actions": [steer_action("d_new", {"work_id": work, "task_ids": [task]})]}
        return json.loads(await self.conductor.door.call("submit_decision", envelope))

    async def test_a_rejected_envelope_ends_the_native_turn_without_more_placement_calls(self):
        repeat = {"actions": [steer_action("d_new", {"work_id": "W1", "task_ids": ["W1.1"]})]}
        malformed = {
            "actions": [
                {
                    "action": "start",
                    "delegations": ["d_new"],
                    "tasks": [
                        {
                            "brief": "Find teas.",
                            "caller_lines": [{"source_message_id": "d_new", "sequence": 0, "text": "Find teas."}],
                        }
                    ],
                }
            ]
        }
        rejection = DecisionRejected("Task W0.1 already continued as W1.1 in W1")
        for payload, expected_placements, error in (
            (malformed, 0, "caller_lines.0.text"),
            ({"actions": [steer_action("d_new", {"work_id": "W0", "task_ids": ["W0.1"]})]}, 1, str(rejection)),
        ):
            with self.subTest(error=error):
                self.make()
                target = SimpleNamespace(place=mock.AsyncMock(side_effect=rejection))
                deciding, native = await self.deciding(given(a_request("d_new", "Use green tea.")), target)
                call = self.conductor.door.capture_call()
                refused = json.loads(await call("submit_decision", payload))
                self.assertEqual(refused["status"], "rejected")
                self.assertIn(error, refused["error"])
                self.assertEqual(json.loads(await call("submit_decision", repeat)), refused)
                with self.assertRaises(DecisionRejected) as ended:
                    await asyncio.wait_for(deciding, 1)
                self.assertEqual(str(ended.exception), refused["error"])
                self.assertEqual(target.place.await_count, expected_placements)
                self.assertEqual((len(native.texts("prompt")), native.retired, native.cancels), (1, True, 1))

    async def test_a_normal_native_terminal_preserves_the_typed_rejection_for_a_fresh_core_attempt(self):
        from examples.frontend_backend_live.delegation_core.backend.agent.render import round_input

        self.make()
        rejection = DecisionRejected("Task W0.1 already continued as W1.1 in W1")
        old, active = an_ended_work(), an_open_work()
        old = replace(
            old, tasks=(replace(old.tasks[0], successor_task_id="W1.1", successor_work_id="W1", steering="continued"),)
        )
        request = a_request("d_old", "Use black tea.\n" + "recorded playback " * 30)
        active = replace(
            active,
            requests=(replace(request, caller_excerpt="Use black tea."),),
            tasks=(replace(active.tasks[0], continues="W0.1", steering="active"),),
        )
        initial = given(a_request("d_new", "Use green tea."), works=(old, active))
        target = SimpleNamespace(place=mock.AsyncMock(side_effect=rejection))
        deciding, native = await self.deciding(initial, target)
        refused = await self.submit_steer("W0", "W0.1")
        self.assertEqual(refused, {"status": "rejected", "error": str(rejection)})
        native.end(1, usage=self.used())
        with self.assertRaises(DecisionRejected) as ended:
            await deciding
        self.assertIs(ended.exception, rejection)
        self.assertEqual(target.place.await_count, 1)
        self.assertTrue(native.retired)
        self.assertEqual(self.target.decisions, [])
        retry = replace(initial, attempt=2, decision_rejection=str(ended.exception))
        repairing, fresh = await self.deciding(retry)
        self.assertIsNot(fresh, native)
        [prompt] = fresh.texts("prompt")
        for fact in (
            json.dumps({"placement_attempt": 2}),
            json.dumps({"decision_rejection": str(rejection)}),
            '"successor_task_id": "W1.1"',
            '"successor_work_id": "W1"',
            '"steering": "continued"',
            '"continues": "W0.1"',
            '"steering": "active"',
            '"text": "Use black tea."',
        ):
            self.assertIn(fact, prompt)
        rendered = json.loads(round_input(retry)["content"][0]["text"])
        self.assertEqual((rendered["placement_attempt"], rendered["decision_rejection"]), (2, str(rejection)))
        self.assertEqual(
            rendered["routing_targets"], {"steer": [{"work_id": "W1", "task_ids": ["W1.1"]}], "continues": []}
        )
        previous, current = [work["tasks"][0] for work in rendered["works"]]
        self.assertEqual(
            {key: previous[key] for key in ("successor_task_id", "successor_work_id", "steering")},
            {"successor_task_id": "W1.1", "successor_work_id": "W1", "steering": "continued"},
        )
        self.assertEqual((current["continues"], current["steering"]), ("W0.1", "active"))
        await self.submit_steer("W1", "W1.1")
        fresh.end(1, usage=self.used())
        await repairing
        self.assertEqual(self.conductor.round, 2)
        self.assertEqual(self.target.decisions, [Steer(("d_new",), (WorkTarget("W1", ("W1.1",)),))])

    async def test_dropped_conductor_turn_reports_unknown_usage_once_with_its_own_correlation(self):
        for ending in ("timeout", "close", "detach", "catalog error"):
            with self.subTest(ending=ending):
                await self.check_dropped_usage(ending)

    async def check_dropped_usage(self, ending):
        self.make(turn_timeout_seconds=0.2 if ending == "timeout" else 300)
        endpoint = mock.Mock(return_value=SimpleNamespace(model="native-test-model"))
        self.host.source = SimpleNamespace(endpoint=endpoint)
        with logged("DEBUG") as lines:
            native = await self.placed(given(a_request("d_before", "Teas?")))
            task, same = await self.deciding(
                replace(given(a_request("d_next", "Hours?")), conversation_ref="conv-usage")
            )
            self.assertIs(same, native)
            turn, door = self.conductor.turn, self.conductor.door
            if ending == "timeout":
                await until(lambda: native.retired and native.cancels)
            elif ending == "detach":
                await self.conductor.set_attachment(False)
            else:
                if ending == "catalog error":
                    endpoint.side_effect = RuntimeError("private catalog detail")
                await self.close(self.host, self.conductor)
            await asyncio.gather(task, return_exceptions=True)
            native.end(2, usage={"totalTokens": 999})
            await turns()
            for _ in range(2):
                await self.close(self.host, self.conductor)
        _, row = [line for line in lines if line.startswith("conductor.round.usage ")]
        model = (
            "None model_source=unknown"
            if ending == "catalog error"
            else "native-test-model model_source=configured_endpoint"
        )
        for field in (
            f"model={model} ",
            "round=2 ",
            "conversation_ref=conv-usage ",
            "delegation_ids=('d_next',) ",
            "usage=None reported_counters=None usage_missing=True ",
            f"prompt=2 session={native.id} ",
            f"ending={'timeout' if ending == 'timeout' else 'incomplete'}",
        ):
            self.assertIn(field, row)
        self.assertEqual(native.cancels, 1)
        if ending == "catalog error":
            self.assertNotIn("private catalog detail", row)
            self.assertTrue(turn.timer.cancelled())
            self.assertTrue(native.retired and door.retired)
            self.assertIn((door, ""), self.host.bridge.closed)

    async def test_terminal_conductor_usage_is_once_and_preserves_reported_or_unknown_counters(self):
        for error in (None, "provider_failed"):
            with self.subTest(error=error):
                self.make()
                self.host.source = SimpleNamespace(endpoint=lambda: SimpleNamespace(model="native-test-model"))
                with logged("DEBUG") as lines:
                    task, native = await self.deciding(
                        replace(given(a_request("d_first", "Teas?")), conversation_ref="conv-terminal")
                    )
                    turn = self.conductor.turn
                    native.end(1, stop=None if error else "end_turn", usage=None if error else USAGE, error=error)
                    await asyncio.gather(task, return_exceptions=True)
                    self.conductor._report_usage(turn, None, ending="incomplete")
                    await self.close(self.host, self.conductor)
                [row] = [line for line in lines if line.startswith("conductor.round.usage ")]
                self.assertIn("model_source=configured_endpoint ", row)
                self.assertIn(f"prompt=1 session={native.id} ", row)
                self.assertIn("conversation_ref=conv-terminal ", row)
                self.assertIn("delegation_ids=('d_first',) ", row)
                if error:
                    self.assertIn("usage=None reported_counters=None usage_missing=True ", row)
                    self.assertIn("ending=error", row)
                else:
                    self.assertIn("usage=None reported_counters=None usage_missing=True", row)
                    self.assertIn("'inputTokens': 80", row)
                    self.assertIn("usage_missing=True ", row)
                    self.assertIn("ending=end_turn", row)

    async def test_refused_prompt_does_not_invent_a_conductor_usage_round(self):
        self.make()
        with logged("DEBUG") as lines:
            native = await self.placed(given(a_request("d_first", "Teas?")))
            with (
                mock.patch.object(native, "prompt", side_effect=RuntimeError("refused prompt")),
                self.assertRaisesRegex(RuntimeError, "refused prompt"),
            ):
                await self.conductor.decide(given(a_request("d_refused", "Hours?")), self.target.place)
            await self.close(self.host, self.conductor)
        self.assertEqual(self.conductor.round, 1)
        self.assertEqual(len(native.texts("prompt")), 1)
        self.assertEqual(sum(line.startswith("conductor.round.usage ") for line in lines), 1)

    async def test_an_unavailable_replacement_fails_once_without_creating_native_sessions_in_a_loop(self):
        self.make()
        old = await self.placed(given(a_request("d_1", "Teas?")))
        self.host.runtime.client.alive = False
        with mock.patch.object(self.host, "shared", wraps=self.host.shared) as reopening:
            with self.assertRaisesRegex(RuntimeError, "runtime closed"):
                await self.conductor.decide(given(a_request("d_2", "Hours?")), self.target.place)
            self.assertEqual(reopening.await_count, 1)
        self.assertTrue(old.retired)
        self.assertEqual(len(self.sessions), 1)
        self.assertEqual(self.host.runtime.places, 0)

    async def test_closing_the_host_never_requests_another_conductor_warm_open(self):
        self.make()
        native = await self.placed(given(a_request("d_1", "Teas?")))
        self.host.closed = True
        with mock.patch.object(self.conductor, "_open", wraps=self.conductor._open) as opened:
            await self.conductor.close()
            await turns()
        opened.assert_not_awaited()
        self.assertTrue(native.retired)

    async def test_attachment_loss_retires_the_control_door_without_warming_until_reattachment(self):
        self.make()
        with mock.patch.object(
            self.host.runtime.client, "new_session", wraps=self.host.runtime.client.new_session
        ) as opened:
            deciding, native = await self.deciding(given(a_request("d_1", "Find teas.")))
            self.assertEqual(opened.await_args.kwargs["role"], "conductor")
            old_door = self.conductor.door
            self.assertEqual(self.conductor.attachment_uncertain_sources(), ("d_1",))
            await self.conductor.set_attachment(False)
            await deciding
            self.assertTrue(old_door.retired)
            self.assertTrue(native.retired)
            self.assertEqual(native.cancels, 1)
            self.conductor.warm()
            await turns()
            self.assertEqual(len(self.sessions), 1)
            await self.conductor.set_attachment(True)
            await until(lambda: len(self.sessions) == 2)
            self.assertEqual(self.sessions[1].texts("prompt"), [])

    async def test_a_late_native_open_from_an_old_attachment_cannot_replace_the_new_control_session(self):
        self.make()
        entered, release = asyncio.Event(), asyncio.Event()
        client = self.host.runtime.client
        original = client.new_session
        first = True

        async def delayed(*args, **kwargs):
            nonlocal first
            if first:
                first = False
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    await release.wait()
            return await original(*args, **kwargs)

        with mock.patch.object(client, "new_session", side_effect=delayed):
            self.conductor.warm()
            await entered.wait()
            detaching = asyncio.create_task(self.conductor.set_attachment(False))
            await until(lambda: not self.conductor.attached)
            await self.conductor.set_attachment(True)
            await until(lambda: self.conductor.session is not None)
            fresh = self.conductor.session
            release.set()
            await detaching
            self.assertIs(self.conductor.session, fresh)
            [late] = [session for session in self.sessions if session is not fresh]
            self.assertTrue(late.retired)
            self.assertFalse(fresh.retired)
            self.assertEqual(self.host.runtime.places, 1)

    def make(self, harness=None, **block):
        self.host = FakeHost(harness or settings(**block))
        self.conductor = HarnessConductor(self.host)
        self.target = FakeDecisionTarget()
        self.sessions = self.host.runtime.client.sessions
        self.totals = itertools.count(100, 100)
        self.addAsyncCleanup(self.close, self.host, self.conductor)

    async def close(self, host, conductor):
        """Close the conductor as its adapter does at the call's end: a turn it left running is cancelled."""
        host.closed = True
        await conductor.close()

    def used(self):
        """Return the usage of a turn that made a model call: a larger running total."""
        return {"totalTokens": next(self.totals)}

    def prompted(self):
        """Return the session whose prompt is out: the newest one in use with a prompt that has not ended."""
        return next(
            (session for session in reversed(self.sessions) if session.outstanding and not session.retired), None
        )

    async def deciding(self, given_input, target=None):
        """Start a ``decide()`` placing into ``target`` or the test's; once it prompted, return it and its session."""
        task = asyncio.ensure_future(self.conductor.decide(given_input, (target or self.target).place))
        await until(self.prompted)
        return task, self.prompted()

    async def placed(self, given_input):
        """Run one ``decide()`` whose turn starts work for its first arrival; return the session it ran in."""
        task, session = await self.deciding(given_input)
        await submit_start(self.conductor.door, given_input.arrivals[0].delegation_id, "Caller request.")
        session.end(session.last_prompt(), usage=self.used())
        await task
        return session

    async def left_running(self, task):
        """Cancel a ``decide()`` twice, as a hang-up after a restart does: it ends, and its agent's turn runs on."""
        task.cancel()
        await turns()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(task.cancelled())

    def later(self, given_input):
        """Start a ``decide()`` placing into the test's target, with no wait; return its task."""
        return asyncio.ensure_future(self.conductor.decide(given_input, self.target.place))

    async def test_one_prompt_per_decide_placed_through_its_door(self):
        self.make()
        works = (an_open_work(), an_ended_work())
        line = {"source_message_id": "d_new", "sequence": 2, "text": "Find teas."}
        request = replace(a_request("d_new", "Find teas."), caller_lines=(line,))
        task, session = await self.deciding(given(request, works=works, conversation=EARLIER))
        [prompt] = session.texts("prompt")
        for part in (
            "Earlier: book a table.",
            "New requests to place:\n[d_new] Find teas.",
            json.dumps({"caller_lines": [line]}),
            '[W1] running, carried by d_old; requests: {"delegation_id": "d_old", '
            '"delivery": "landed", "text": "Find teas."}',
            'task W1.1: {"state": "running", "stop": "none", "answer_publication": "none", '
            '"tool_results": [], "tool_results_omitted": 0} for 3 s; '
            "steps: get_menu: done; newest note: Looking it up.",
            "[W0] answered, carried by d_done",
        ):
            self.assertIn(part, prompt)
        folder = self.host.runtime.client.folders[0]
        self.assertEqual((folder / ".hermes.md").read_text().strip(), self.host.vocabulary.instructions(3))
        self.assertEqual(self.host.runtime.client.servers[0].name, "c1")
        tools = [tool["name"] for tool in self.conductor.door.tools()]
        self.assertEqual(tools, ["submit_decision"])
        outcome = await self.conductor.door.call(
            "submit_decision",
            {"actions": [{"action": "steer", "delegations": ["d_new"], "targets": [{"work_id": "W1"}]}]},
        )
        self.assertEqual(json.loads(outcome)["actions"][0]["state"], "applied")
        malformed = await self.conductor.door.call("submit_decision", {"actions": []})
        self.assertEqual(json.loads(malformed), {"status": "refused", "reason": "decision_already_submitted"})
        session.answer(1, "m1", "Placed ", "it.", usage=DECIDED)
        await task
        self.assertEqual(self.target.decisions, [Steer(("d_new",), (WorkTarget("W1"),))])
        self.assertIs(await self.placed(given(a_request("d_2", "Hours?"), conversation=EARLIER, new_from=1)), session)
        self.assertNotIn("Earlier: book a table.", session.texts("prompt")[1])

    async def test_a_plain_conductor_reads_its_prompt_atop_the_first(self):
        self.make(plain())
        task, session = await self.deciding(given(a_request("d_1", "Hi.")))
        instructions = self.host.vocabulary.instructions(3)
        self.assertTrue(session.texts("prompt")[0].startswith(instructions + "\n\n"))
        self.assertEqual(list(self.host.runtime.client.folders[0].iterdir()), [])
        await submit_start(self.conductor.door, "d_1", "Caller request.")
        session.end(1)
        await task
        status_turn, reused = await self.deciding(given(a_request("d_2", "Any update?"), works=(an_open_work(),)))
        self.assertIs(reused, session)
        result = await self.conductor.door.call(
            "submit_decision",
            {
                "actions": [
                    {
                        "action": "caller_message",
                        "kind": "status",
                        "delegations": ["d_2"],
                        "targets": [{"work_id": "W1"}],
                    }
                ]
            },
        )
        self.assertEqual(json.loads(result)["actions"][0]["state"], "applied")
        session.end(2)
        await status_turn
        self.assertNotIn(instructions, session.texts("prompt")[1])
        self.assertEqual(
            self.target.decisions,
            [
                Start(("d_1",), (TaskSpec("Caller request."),)),
                CallerMessage(("d_2",), "status", targets=(WorkTarget("W1"),)),
            ],
        )

    async def test_a_failed_turn_or_lost_runtime_gives_a_fresh_session(self):
        self.make()
        task, session = await self.deciding(given(a_request("d_1", "Teas?")))
        session.end(1, "refusal")
        with self.assertRaises(RuntimeError):
            await task
        self.assertEqual((session.retired, self.target.decisions), (True, []))
        fresh = await self.placed(given(a_request("d_2", "Hours?")))
        self.assertIsNot(fresh, session)
        self.host.runtime.client.alive = False
        # A real adapter opens a healthy replacement runtime; the shared dummy does not do that itself.
        replacement = type(self.host.runtime)()
        replacement.client.sessions = self.sessions
        self.host.runtime = replacement
        await self.placed(given(a_request("d_3", "Teas?")))
        self.assertEqual((len(self.sessions), fresh.retired), (3, True))

    async def test_a_turn_is_read_to_its_end_however_long_it_runs(self):
        # Only the backstop at turn_timeout_seconds cuts a turn, which may be the agent compacting its memory.
        self.make()
        task, session = await self.deciding(given(a_request("d_1", "Teas?")))
        await submit_start(self.conductor.door, "d_1", "Caller request.")
        # The agent works on: half a minute passes on the loop's clock.
        loop = asyncio.get_running_loop()
        clock = loop.time
        loop.time = lambda: clock() + 30
        await turns()
        self.assertFalse(task.done())
        session.end(1, usage=self.used())
        await task
        self.assertEqual(
            (session.cancels, session.retired, self.target.decisions),
            (0, False, [Start(("d_1",), (TaskSpec("Caller request."),))]),
        )

    async def test_a_restarted_decide_cancels_and_fences_its_native_turn(self):
        self.make()
        session = await self.placed(given(a_request("d_0", "Book."), conversation=EARLIER))
        task, _ = await self.deciding(given(a_request("d_1", "Teas?"), conversation=EARLIER, new_from=1))
        turn = self.conductor.turn
        task.cancel(PlacementExpired)
        await turns()
        self.assertFalse(task.done())
        refused = await submit_start(self.conductor.door, "d_1", "Caller request.")
        self.assertEqual(json.loads(refused), {"status": "superseded", "executed": False})
        self.assertFalse(turn.control_completed)
        session.end(2, "cancelled", usage=self.used())
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(self.target.decisions[-1].delegations, ("d_0",))
        self.assertEqual(session.cancels, 1)
        self.assertIs(await self.placed(given(a_request("d_2", "Green."), conversation=EARLIER, new_from=1)), session)
        self.assertEqual(len(session.texts("prompt")), 3)
        self.assertNotIn("Earlier: book a table.", session.texts("prompt")[2])

    async def test_the_next_decide_waits_for_a_turn_a_cancelled_one_left(self):
        # A hang-up cancels a decide() once, and it reads its turn on; a hang-up after a restart cancels it twice, and
        # it leaves the turn running. Either way the next decide() never splits that turn.
        for cancels in (1, 2):
            with self.subTest(cancels=cancels):
                self.make()
                first = FakeDecisionTarget()
                task, session = await self.deciding(given(a_request("d_1", "Teas?")), first)
                for _ in range(cancels):
                    task.cancel()
                    await turns()
                self.assertEqual((task.done(), task.cancelled()), (cancels == 2, cancels == 2))
                later = self.later(given(a_request("d_2", "Hours?"), new_from=1))
                await turns()
                # Never a prompt into a running turn; the turn's own calls still reach the decision that prompted it.
                self.assertEqual(len(session.texts("prompt")), 1)
                await submit_start(self.conductor.door, "d_1", "Caller request.")
                self.assertEqual(
                    (first.decisions, self.target.decisions), ([Start(("d_1",), (TaskSpec("Caller request."),))], [])
                )
                session.end(1, usage=self.used())
                await until(lambda session=session: len(session.texts("prompt")) == 2)
                await submit_start(self.conductor.door, "d_2", "Caller request.")
                session.end(2, usage=self.used())
                await later
                decided = (self.target.decisions, session.cancels, session.retired)
                self.assertEqual(decided, ([Start(("d_2",), (TaskSpec("Caller request."),))], 0, False))

    async def test_a_wait_that_ends_retired_prompts_a_fresh_session_with_the_whole_history(self):
        self.make()
        task, session = await self.deciding(given(a_request("d_1", "Teas?"), conversation=EARLIER))
        await self.left_running(task)
        later = self.later(given(a_request("d_2", "Hours?"), conversation=EARLIER, new_from=1))
        await turns()
        self.assertEqual((len(self.sessions), len(session.texts("prompt"))), (1, 1))
        session.end(1, "refusal")
        await until(lambda: len(self.sessions) == 2 and self.sessions[1].sent)
        fresh = self.sessions[1]
        self.assertIn("Earlier: book a table.", fresh.texts("prompt")[0])
        self.assertEqual((session.retired, session.cancels), (True, 0))
        await submit_start(self.conductor.door, "d_2", "Caller request.")
        fresh.end(1, usage=self.used())
        await later
        self.assertEqual(self.target.decisions, [Start(("d_2",), (TaskSpec("Caller request."),))])

    async def test_a_turn_past_its_limit_is_cancelled_once_and_its_session_replaced(self):
        # The backstop holds whether a decide() reads the turn or none does, and conductor_timeout_seconds sets it.
        limits = ({"turn_timeout_seconds": 0.2}, {"conductor_timeout_seconds": 0.2})
        for reading, limit in itertools.product((True, False), limits):
            with self.subTest(reading=reading, limit=limit):
                self.make(**limit)
                with logged() as lines:
                    task, session = await self.deciding(given(a_request("d_1", "Teas?"), conversation=EARLIER))
                    if not reading:
                        await self.left_running(task)
                    await until(lambda session=session: session.retired and session.cancels)
                    await asyncio.gather(task, return_exceptions=True)
                overran = [line for line in lines if line.startswith("harness.conductor_turn_overran")]
                self.assertEqual((len(overran), session.cancels), (1, 1))
                fresh = await self.placed(given(a_request("d_2", "Hours?"), conversation=EARLIER, new_from=1))
                self.assertIsNot(fresh, session)
                self.assertIn("Earlier: book a table.", fresh.texts("prompt")[0])

    async def test_close_cancels_a_running_turn(self):
        self.make()
        task, session = await self.deciding(given(a_request("d_1", "Teas?")))
        self.host.closed = True
        await self.conductor.close()
        await task
        self.assertEqual((session.cancels, session.retired, len(self.sessions)), (1, True, 1))

    async def test_a_decide_that_fails_while_a_turn_runs_cancels_it_once_and_retires(self):
        self.make()
        task, session = await self.deciding(given(a_request("d_1", "Teas?"), conversation=EARLIER))
        await self.left_running(task)
        later = self.later(given(a_request("d_2", "Hours?")))
        await turns()
        self.assertEqual((len(self.sessions), len(session.texts("prompt"))), (1, 1))
        session.put(Unreadable())
        with self.assertRaisesRegex(RuntimeError, "unreadable event"):
            await later
        self.assertEqual((session.cancels, session.retired, self.target.decisions), (1, True, []))
        fresh = await self.placed(given(a_request("d_3", "Sunday?"), conversation=EARLIER, new_from=1))
        self.assertIsNot(fresh, session)
        self.assertIn("Earlier: book a table.", fresh.texts("prompt")[0])
