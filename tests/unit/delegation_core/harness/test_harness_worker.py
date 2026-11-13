# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The harness's worker run and conductor on a session double: turns, app calls, steering, stops and sessions."""

import asyncio
import contextlib
import itertools
import json
import unittest
from unittest import mock

from loguru import logger

from examples.frontend_backend_live.delegation_core.backend.conductors import SteerOpenWork
from examples.frontend_backend_live.delegation_core.backend.harness.acp import AcpAgent, SessionUpdates
from examples.frontend_backend_live.delegation_core.backend.harness.conductor import HarnessConductor
from examples.frontend_backend_live.delegation_core.backend.harness.protocol import (
    CallRunning,
    CallSettled,
    PlanUpdate,
    ThoughtChunk,
    ToolCard,
    ToolCardEnd,
)
from examples.frontend_backend_live.delegation_core.backend.harness.runtime import _CANCELS
from examples.frontend_backend_live.delegation_core.backend.harness.worker import HarnessTurnError, WorkerRun
from examples.frontend_backend_live.delegation_core.backend.port import ConductorInput, Start, Steer
from tests.unit.delegation_core.harness.doubles import FakeHost, FakeTaskContext, plain, settings, until, wire
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
    placing,
    planned,
    read_all,
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


@contextlib.contextmanager
def logged():
    """Collect the warning lines logged inside the block."""
    lines = []
    sink = logger.add(lambda message: lines.append(message.strip()), level="WARNING", format="{message}")
    try:
        yield lines
    finally:
        logger.remove(sink)


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
        session.say("m1", "Do", "ne.")
        session.end(1, usage=USAGE)
        events = await self.rest()
        self.assertEqual(check_events(events), [])
        self.assertEqual(answers(events), ["Done."])
        usage = {"input_tokens": 80, "output_tokens": 20, "total_tokens": 100}
        self.assertEqual(events[-1]["response"]["usage"], usage)
        self.assertEqual(self.host.idle(), [session])
        self.assertEqual((session.retired, self.host.bridge.closed), (False, []))
        self.assertEqual(await self.run.stop(), "stopped")
        # An idle cancel poisons Hermes's next prompt.
        self.assertEqual(session.cancels, 0)

    async def test_the_runtimes_own_text_with_no_usage_is_unsaid(self):
        for scenario in ("provider_error", "idle_cancel_poison"):
            with self.subTest(scenario):
                session = await self.open()
                steps = wire(scenario)
                for step in steps:
                    if "update" in step:
                        session.replay(step["update"]["update"])
                result = next(step["result"] for step in steps if "end" in step)
                session.end(1, result["stopReason"], result.get("usage"))
                self.assertIn("no usage", str(await self.failure()))
                self.assertEqual(self.seen[-1]["type"], "response.failed")
                self.assertEqual(self.seen[-1]["response"]["error"]["code"], "server_error")
                self.assertEqual(items_of(self.seen, "message"), [])

    async def test_a_turn_with_no_answer_fails_with_its_reason(self):
        tool = (ToolCard("tc1", "mcp__w1__lookup", {}), ToolCardEnd("tc1", False, "ok"))
        for reason, chunks, events, end in (
            ("did not move", ("Do", "ne."), (), {"usage": {"totalTokens": 0}}),
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
        session.say("m2", "I could ", "not.")
        session.end(1, usage=USAGE)
        await collector
        [(call_id, _)] = self.task.started
        self.assertEqual(self.task.outcomes[call_id], (misrouted, "failed"))


class SteeringTests(Harness):
    async def steer(self, session, *, begin=True):
        if begin:
            session.put(ThoughtChunk("m0"))
        self.steering = self.task.steer_with(STEERING)
        self.run.steer()
        await until(lambda: session.texts("steer"))
        [steer] = session.texts("steer")
        self.assertIn("Make it two.", steer)

    async def test_a_native_steer_waits_for_the_turns_first_event(self):
        session, collector = await self.opened()
        self.task.steer_with(STEERING)
        self.run.steer()
        await asyncio.sleep(0.05)
        self.assertEqual(session.texts("steer"), [])
        session.put(ThoughtChunk("m1"))
        await until(lambda: session.texts("steer"))
        session.end(2)
        session.say("m1", "Do", "ne.", thought=False)
        session.end(1, usage=USAGE)
        events = await collector
        self.assertEqual(answers(events), ["Done."])
        self.assertEqual(self.task.settled, [("steer_1", "not_applied")])

    async def test_a_native_steer_applies_at_a_model_call_after_a_tool(self):
        session, collector = await self.opened()
        await self.steer(session)
        # The ack: the command returns while the turn runs.
        session.end(2)
        session.put(ToolCard("tc1", "mcp__w1__lookup", {}))
        session.put(ToolCardEnd("tc1", False, "ok"))
        await asyncio.sleep(0.05)
        # A tool result alone proves nothing yet.
        self.assertEqual(self.task.settled, [])
        session.say("m2", "Two ", "coming.")
        await until(lambda: self.task.settled)
        self.assertEqual(self.task.settled, [("steer_1", "applied")])
        session.end(1, usage=USAGE)
        events = await collector
        self.assertEqual(answers(events), ["Two coming."])

    async def test_a_native_steer_applies_at_a_new_call_after_a_tool_row(self):
        session = await self.open()
        session.put(ToolCard("tc1", "mcp__w1__lookup", {}))
        session.put(ToolCardEnd("tc1", False, "ok"))
        session.say("m1", "Looking ", "further.")
        collector = self.collecting()
        await until(lambda: self.run.current_message == "m1")
        await self.steer(session, begin=False)
        session.end(2)
        session.say("m1", " still")
        await asyncio.sleep(0.05)
        self.assertEqual(self.task.settled, [])
        session.say("m2", "With ", "two.")
        await until(lambda: self.task.settled)
        session.end(1, usage=USAGE)
        events = await collector
        self.assertEqual((answers(events, "commentary"), answers(events)), (["Looking further. still"], ["With two."]))

    async def test_a_steer_not_taken_leaves_the_answer_and_goes_back(self):
        session, collector = await self.opened()
        await self.steer(session)
        session.end(2)
        session.say("m1", "Old ", "answer.")
        # The model answered: Hermes drops a steer still waiting (pending_steer).
        session.end(1, usage=USAGE)
        events = await collector
        self.assertEqual((answers(events), answers(events, "commentary")), (["Old answer."], []))
        self.assertEqual(self.task.settled, [("steer_1", "not_applied")])
        self.assertEqual(len(session.texts("prompt")), 1)

    async def test_the_runtimes_own_text_is_never_commentary(self):
        # A model call that reasoned and then failed: Hermes ends the turn with its own text under the reasoning's id,
        # and no usage. A steer waits, so that text would become commentary the voice speaks: it is dropped instead.
        session, collector = await self.opened()
        await self.steer(session)
        session.end(2)
        own = "Error: the provider refused the request."
        session.say("m0", own, thought=False)
        session.end(1)
        await until(lambda: len(session.texts("prompt")) == 2)
        session.say("m2", "Two ", "done.")
        session.end(3, usage={"totalTokens": 200})
        events = await collector
        self.assertNotIn(own, texts(events))
        self.assertEqual(answers(events), ["Two done."])

    async def assert_next_prompt_in_order(self, session, collector, number):
        """Assert the batches went as the next prompt (prompt ``number``), in the order the run took them."""
        await until(lambda: len(session.texts("prompt")) == 2)
        self.assertEqual(self.task.settled, [("steer_1", "applied"), ("steer_2", "applied")])
        prompt = session.texts("prompt")[1]
        self.assertLess(prompt.index("Oat milk."), prompt.index("Warm the croissant."))
        session.say("m2", "Done.")
        session.end(number, usage=USAGE)
        self.assertEqual(answers(await collector), ["Done."])

    async def test_a_steer_never_delivered_goes_ahead_of_a_later_batch(self):
        session, collector = await self.opened()
        session.put(ThoughtChunk("m0"))
        self.task.steer_with(OLDER, "steer_1")
        self.run.steer()
        await until(lambda: session.texts("steer"))
        session.end(1)
        await until(lambda: not self.run.turn.main_open)
        self.task.steer_with(NEWER, "steer_2")
        self.run.steer()
        await until(lambda: self.run.next_prompts)
        session.end(2, error="the steer failed")
        await self.assert_next_prompt_in_order(session, collector, 3)

    async def test_a_batch_held_for_the_turns_first_event_goes_first(self):
        session, collector = await self.opened()
        self.task.steer_with(OLDER, "steer_1")
        self.run.steer()
        await until(lambda: self.run.turn.early)
        self.task.steer_with(NEWER, "steer_2")
        session.end(1)
        await self.assert_next_prompt_in_order(session, collector, 2)

    async def test_a_delivered_steer_joins_after_earlier_outputs(self):
        # Hermes puts the steer after the tool batch's results, so the task's history must too: an output never
        # follows a newer request that came after its call.
        session, collector = await self.opened()
        session.say("m1", "Let me ", "check.")
        call = {"type": "function_call", "id": "fc_call_1", "call_id": "call_1", "name": "lookup", "arguments": "{}"}
        session.put(CallRunning(call))
        session.put(ToolCard("tc1", "mcp__w1__lookup", {}))
        await until(lambda: self.run.calls)
        await self.steer(session, begin=False)
        session.end(2)
        output = {"type": "function_call_output", "call_id": "call_1", "output": "found"}
        self.task.give_outputs([output])
        session.put(CallSettled("call_1"))
        session.put(ToolCardEnd("tc1", False, "found"))
        session.say("m2", "With ", "two.")
        await until(lambda: self.task.settled)
        session.end(1, usage=USAGE)
        await collector
        joined = self.task.items
        self.assertLess(joined.index(output), joined.index(STEERING[0]))

    async def test_steering_while_the_session_opens_joins_the_prompt(self):
        run = self.make()
        self.task.steer_with(STEERING)
        run.steer()
        self.events = run.events()
        self.seen = [await anext(self.events)]
        session = self.host.runtime.client.sessions[-1]
        [first] = session.texts("prompt")
        self.assertIn("Make it two.", first)
        self.assertEqual(self.task.settled, [("steer_1", "applied")])
        collector = self.collecting()
        session.say("m1", "Two ", "it is.")
        session.end(1, usage=USAGE)
        events = await collector
        self.assertEqual(answers(events), ["Two it is."])

    async def test_a_steer_read_after_the_answer_is_a_turn_of_its_own(self):
        for ran, answered, delivery in ((True, "Two now.", "applied"), (False, "Old answer.", "not_applied")):
            with self.subTest(ran=ran):
                session, collector = await self.opened()
                await self.steer(session)
                session.say("m1", "Old ", "answer.")
                session.end(1, usage=USAGE)
                if ran:
                    session.say("m2", "Two ", "now.")
                    session.end(2, usage={"totalTokens": 180})
                else:
                    session.end(2)
                self.assertEqual(answers(await collector), [answered])
                self.assertEqual(self.task.settled, [("steer_1", delivery)])
                self.assertEqual(len(session.texts("prompt")), 1)

    async def test_a_cancelled_turns_held_app_call_ends_not_run(self):
        session, collector = await self.opened(harness=plain())
        self.task.steer_with(STEERING)
        self.run.steer()
        await until(lambda: session.cancels == 1)
        session.put(ToolCard("tc1", "Looking up", {}))
        session.put(ToolCard("tc2", "Getting the menu", {}))
        stale = asyncio.ensure_future(self.run.door.call("get_menu", {"page": 1}))
        await until(lambda: self.run.app_queue)
        # The cards never ended: ACP leaves a cancelled turn's calls to the client.
        session.end(1, "cancelled", USAGE)
        self.assertEqual(await asyncio.wait_for(stale, 1), self.task.texts.not_run_output)
        session.put(ToolCard("tc3", "Getting the menu", {}))
        call = asyncio.ensure_future(self.run.door.call("get_menu", {"page": 2}))
        [forwarded] = items_of(await asyncio.wait_for(collector, 2), "function_call")
        self.assertEqual(forwarded["arguments"], '{"page": 2}')
        self.assertFalse(call.done())

    async def test_a_reprompts_cancel_is_kept_and_ends_quietly(self):
        session, _ = await self.opened(harness=plain())
        session.say("m1", "Let me ", "look.", thought=False)
        sent = asyncio.Event()

        async def unsent_cancel():
            sent.set()
            await asyncio.Event().wait()

        session.cancel = unsent_cancel
        errors = []
        asyncio.get_running_loop().set_exception_handler(lambda loop, context: errors.append(context["message"]))
        self.task.steer_with(STEERING)
        self.run.steer()
        await asyncio.wait_for(sent.wait(), 2)
        # Held until it ends, so that it is never collected before it is sent.
        (cancel,) = _CANCELS
        cancel.cancel()
        await asyncio.sleep(0.05)
        self.assertEqual((_CANCELS, errors), (set(), []))

    async def test_a_turn_ended_before_the_cancel_keeps_its_session(self):
        session, collector = await self.opened(harness=plain())
        steering = self.task.steer_with(STEERING)
        self.run.steer()
        session.end(1)
        await until(lambda: len(session.texts("prompt")) == 2)
        self.assertEqual((session.cancels, len(self.host.runtime.client.sessions)), (0, 1))
        self.assertEqual(session.texts("prompt")[1], self.host.renderer.steering(steering))
        session.say("m1", "Two ", "it is.", thought=False)
        session.end(2)
        self.assertEqual(answers(await collector), ["Two it is."])

    async def test_a_turn_ended_after_the_cancel_gets_a_fresh_session(self):
        # It may have ended before the cancel reached the agent, which then takes the cancel for its next prompt.
        session, collector = await self.opened(harness=plain())
        self.task.steer_with(STEERING)
        self.run.steer()
        await until(lambda: session.cancels == 1)
        session.end(1)
        client = self.host.runtime.client
        await until(lambda: len(client.sessions) == 2)
        fresh = client.sessions[1]
        self.assertTrue(session.retired)
        self.assertEqual(client.servers[1].name, "w2")
        self.assertEqual(self.task.settled, [("steer_1", "applied")])
        self.assertIn("Make it two.", fresh.texts("prompt")[0])
        fresh.say("m1", "Two ", "it is.")
        fresh.end(1, usage=USAGE)
        events = await collector
        self.assertEqual(answers(events), ["Two it is."])


class StopTests(Harness):
    async def test_a_stop_cancels_the_prompt_and_retires_the_door(self):
        session = await self.open()
        held = asyncio.ensure_future(self.run.door.call("get_menu", {}))
        await asyncio.sleep(0)
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
        # Its Work goes on after the call, bounded by the keep time and a stop: an unmarked turn fails meanwhile.
        session, collector = await self.opened(task=FakeTaskContext(after_call=True), turn_timeout_seconds=0.2)
        # The marked run reads first, so a limit on its turn would run out before the unmarked one's.
        await turns()
        unmarked = WorkerRun(FakeTaskContext(work_id="W2", task_id="W2.1"), self.host)
        with self.assertRaisesRegex(HarnessTurnError, "ran past"):
            await read_all(unmarked.events())
        self.assertFalse(collector.done())
        session.say("m1", "Do", "ne.")
        session.end(1, usage=USAGE)
        self.assertEqual(answers(await collector), ["Done."])
        self.assertEqual(session.cancels, 0)

    async def test_a_turn_whose_work_is_marked_while_it_waits_runs_past_the_turn_limit(self):
        # Another task of its Work marks the Work while this one waits on a quiet session.
        session, collector = await self.opened(turn_timeout_seconds=0.2)
        await turns()
        self.task.after_call = True
        # The marked run waits first, so the limit it began its wait with runs out before the unmarked one's.
        unmarked = WorkerRun(FakeTaskContext(work_id="W2", task_id="W2.1"), self.host)
        with self.assertRaisesRegex(HarnessTurnError, "ran past"):
            await read_all(unmarked.events())
        self.assertFalse(collector.done())
        session.say("m1", "Do", "ne.")
        session.end(1, usage=USAGE)
        self.assertEqual(answers(await collector), ["Done."])
        self.assertEqual(session.cancels, 0)

    async def test_a_turn_whose_task_call_raises_is_cancelled(self):
        # The task fails and no stop follows: the run cancels the turn, which would run on with no reader.
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
    later = [placing(Steer(f"W{number}", (f"D{number + 1}",))) for number in range(1, count)]
    return planned(placing(Start(("D1",))), *later)


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
        self.assertEqual(repeated, im.texts.repeated_call_output.format(output="Error: order not found"))
        session.say("m2", "Please spell ", "your order number.")
        session.end(1, usage=USAGE)
        events = await asyncio.wait_for(read_all(rest.stream), 5)
        self.assertEqual(items_of(events, "function_call"), [])
        self.assertEqual(answers(events), ["Please spell your order number."])

    async def test_an_own_tool_run_while_steering_waits_is_reported(self):
        im = self.session(self.runs)
        first = im.admit(ask("Find teas."), delegation_id="D1")
        await until(lambda: self.runs.runs and self.client.sessions)
        session = self.client.sessions[0]
        session.say("m1", "Searching ", "now.")
        await self.turns()
        im.admit(ask("Only green ones."), delegation_id="D2")
        await until(lambda: session.texts("steer"))
        session.put(ToolCard("tc9", "web_search", {"query": "teas"}, bridged=False))
        session.put(ToolCardEnd("tc9", False, "green, black, oolong"))
        await self.turns()
        session.end(2)
        session.say("m2", "Green ", "teas.")
        session.end(1, usage=USAGE)
        events = await asyncio.wait_for(read_all(first.stream), 5)
        [item] = items_of(events, "mcp_call")
        self.assertEqual((item["name"], item["status"]), ("web_search", "completed"))
        action = im.ledger.actions["tc9"]
        self.assertEqual((action.state, action.output), ("done", "green, black, oolong"))
        self.assertEqual(im.ended_works[-1].tasks[0].step_lines(), ("web_search: done",))

    async def test_the_next_works_task_takes_the_kept_session(self):
        # One task at a time: the steer rule places D2 in W1, which has ended, so D2's Work continues W1's thread.
        im = self.session(self.runs)
        first = im.admit(ask("Find teas."), delegation_id="D1")
        session = await self.prompted(1)
        session.say("m1", "Green ", "tea.")
        session.end(1, usage=USAGE)
        self.assertEqual(answers(await read_all(first.stream)), ["Green tea."])
        await until(lambda: not im.works)
        second = im.admit(ask("And black?"), delegation_id="D2")
        await self.prompted(2)
        self.assertEqual(len(self.client.sessions), 1)
        prompt = session.texts("prompt")[1]
        self.assertTrue(prompt.startswith("And black?\n"))
        self.assertNotIn("Green tea.", prompt)
        session.say("m2", "Black ", "too.")
        session.end(2, usage=LATER_USAGE)
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
            newest.append(self.runs.runs[-1].task.history()[-1])
            session.say(f"m{number}", "Done.")
            session.end(session.last_prompt(), usage={"totalTokens": 100 * number})
            await read_all(admitted.stream)
            await until(lambda: not im.works)
        self.assertEqual(len(self.client.sessions), 1)
        self.assertEqual(session.texts("prompt")[1:], [self.host.renderer.transcript([item]) for item in newest[1:]])

    async def test_two_requests_of_the_same_words_are_told_apart(self):
        # W2's "Yes." and answer join the conversation before W1's: W1's session read only its own "Yes.".
        placer = planned(placing(Start(("D1",))), placing(Start(("D2",))), placing(Steer("W1", ("D3",))))
        self.runs = Runs(self.host, placer, max_running=3)
        im = self.session(self.runs)
        table = im.admit(ask("Yes."), delegation_id="D1")
        ours = await self.prompted(1)
        taxi = im.admit(ask("Yes."), delegation_id="D2")
        theirs = await self.prompted(1, index=1)
        for session, admitted, text in ((theirs, taxi, "Your taxi is booked."), (ours, table, "Your table is booked.")):
            session.say("m1", text)
            session.end(1, usage=USAGE)
            await read_all(admitted.stream)
        last = im.admit(ask("And the wifi password?"), delegation_id="D3")
        await self.prompted(2)
        history = self.runs.runs[-1].task.history()
        self.assertEqual([item.get("id") for item in history if item.get("role") == "user"], ["D2", "D1", "D3"])
        # What it has not read: W2's request and answer, then D3.
        self.assertEqual(ours.texts("prompt")[1], self.host.renderer.transcript([*history[:2], history[-1]]))
        self.assertEqual(len(self.client.sessions), 2)
        ours.say("m2", "It is on the card.")
        ours.end(2, usage=LATER_USAGE)
        await read_all(last.stream)

    async def test_a_follow_up_of_one_split_task_is_sent_the_other_tasks_brief_and_answer(self):
        # D2 continues W1.1, whose session read the request and its own brief: each brief has an id of its own.
        placer = planned(placing(Start(("D1",), ("teas", "hours"))), placing(Steer("W1", ("D2",), "W1.1")))
        self.runs = Runs(self.host, placer, max_running=2)
        im = self.session(self.runs)
        first = im.admit(ask("Teas and hours."), delegation_id="D1")
        await self.prompted(1, index=1)
        teas, hours = self.runs.runs
        sessions, hours_brief = [teas.session, hours.session], hours.task.history()[-1]
        for session, text in zip(sessions, ("Green tea.", "Open till six."), strict=True):
            session.say("m1", text)
            session.end(1, usage=USAGE)
        await read_all(first.stream)
        await until(lambda: not im.works)
        last = im.admit(ask("And black tea?"), delegation_id="D2")
        await until(lambda: len(sessions[0].texts("prompt")) == 2)
        history = self.runs.runs[-1].task.history()
        sent = history[history.index(hours_brief) :]
        self.assertEqual(sessions[0].texts("prompt")[1], self.host.renderer.transcript(sent))
        self.assertEqual(len(self.client.sessions), 2)
        sessions[0].say("m2", "Black tea too.")
        sessions[0].end(2, usage=LATER_USAGE)
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

    async def test_a_typed_message_with_the_outputs_goes_as_a_steer(self):
        session, call, rest = await self.typed_with_the_outputs(settings(), MILK)
        await until(lambda: session.texts("steer"))
        self.assertEqual((session.texts("steer"), call.done()), (["With milk, please."], False))
        session.end(2)
        self.assertEqual(await call, "teas")
        session.put(ToolCardEnd("tc1", False, "teas"))
        session.say("m2", "Green tea ", "with milk.")
        session.end(1, usage=USAGE)
        self.assertEqual(answers(await read_all(rest.stream)), ["Green tea with milk."])
        self.assertEqual(self.steering_states(), ["applied"])
        await until(lambda: not self.im.works)
        self.im.admit(ask("And black?"), delegation_id="D2", settings=MENU_APP)
        await self.prompted(2)
        self.assertEqual((len(self.client.sessions), session.retired), (1, False))

    async def test_a_typed_message_reaches_a_plain_agent_as_a_prompt(self):
        session, call, rest = await self.typed_with_the_outputs(plain(), MILK)
        self.assertEqual(await call, "teas")
        await self.turns()
        self.assertEqual(session.cancels, 0)
        session.put(ToolCardEnd("tc1", False, "teas"))
        await until(lambda: session.cancels == 1)
        session.end(1, "cancelled", USAGE)
        await self.prompted(2)
        self.assertEqual(session.texts("prompt")[1], "With milk, please.")
        session.say("m2", "Green tea ", "with milk.", thought=False)
        session.end(2, usage=LATER_USAGE)
        self.assertEqual(answers(await read_all(rest.stream)), ["Green tea with milk."])
        self.assertEqual(self.steering_states(), ["applied"])

    async def test_a_typed_message_with_no_text_is_applied_at_once(self):
        for name, harness in (("hermes", settings()), ("plain", plain())):
            with self.subTest(name):
                session, call, rest = await self.typed_with_the_outputs(harness, IMAGE)
                self.assertEqual(await asyncio.wait_for(call, 1), "teas")
                session.put(ToolCardEnd("tc1", False, "teas"))
                session.say("m2", "Green ", "tea.")
                session.end(1, usage=USAGE)
                self.assertEqual(answers(await read_all(rest.stream)), ["Green tea."])
                self.assertEqual(self.steering_states(), ["applied"])
                sent = (session.texts("steer"), session.texts("prompt")[1:], session.cancels)
                self.assertEqual(sent, ([], [], 0))

    async def test_an_answer_across_the_cancel_gives_a_fresh_session(self):
        self.host = FakeHost(plain())
        self.client, self.runs = self.host.runtime.client, Runs(self.host)
        im = self.session(self.runs)
        menu = {"tools": [MENU]}
        first = im.admit(ask("Find teas."), delegation_id="D1", settings=menu)
        session = await self.prompted(1)
        call = asyncio.ensure_future(self.runs.runs[0].door.call("get_menu", {"page": 1}))
        outputs = app_outputs(await read_all(first.stream), "green, black")
        rest = im.admit_client_request(outputs, delegation_id="D1", settings=menu)
        self.assertEqual(await call, "green, black")
        newer = im.admit(ask("Make it two."), delegation_id="D2", settings=menu)
        await until(lambda: session.cancels == 1)
        session.say("m2", "Green ", "tea.", thought=False)
        session.end(1, usage=USAGE)
        self.assertEqual(answers(await read_all(rest.stream)), ["Green tea."])
        fresh = await self.prompted(1, index=1)
        self.assertTrue(session.retired)
        prompt = fresh.texts("prompt")[0]
        self.assertIn('get_menu with {"page": 1}', prompt)
        self.assertLess(prompt.index("green, black"), prompt.index("Make it two."))
        fresh.say("m3", "Two ", "teas.", thought=False)
        fresh.end(1, usage=LATER_USAGE)
        self.assertEqual(answers(await read_all(newer.stream)), ["Two teas."])

    async def test_each_call_the_gate_refuses_is_traced_once(self):
        # The run routes a call as the agent makes it, and the IM again as it reads the round.
        self.host = FakeHost(plain(), tools=(RecordingTools(),))
        self.client, self.runs = self.host.runtime.client, Runs(self.host)
        im = self.session(self.runs)
        im.admit(ask("Find teas."), delegation_id="D1", settings=MENU_APP)
        session = await self.prompted(1)
        newer = im.admit(ask("Only green ones."), delegation_id="D2", settings=MENU_APP)
        await until(lambda: session.cancels == 1)
        door = self.runs.runs[0].door
        self.assertEqual([await door.call(name, {}) for name in ("get_menu", "lookup")], [im.texts.not_run_output] * 2)
        session.end(1, "cancelled", USAGE)
        await self.prompted(2)
        session.say("m2", "Green ", "teas.", thought=False)
        session.end(2, usage=LATER_USAGE)
        self.assertEqual(answers(await asyncio.wait_for(read_all(newer.stream), 5)), ["Green teas."])
        refused = [(fields["name"], fields["reason"]) for kind, fields in self.traces if kind == "tools.refused"]
        self.assertEqual(refused, [("get_menu", "not_run"), ("lookup", "not_run")])

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
        await conductor.door.call("start_work", {"delegations": ["D1"]})
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
        late = await conductor.door.call("start_work", {"delegations": ["D2"]})
        self.assertEqual(late, im.texts.refusal("ended"))


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
        session.say(message_id, "Green ", "tea.")
        session.end(session.last_prompt(), usage=usage or {"totalTokens": next(self.totals)})
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
        # ACP counts usage across the session's turns: the second task's turn is what moved since the first.
        self.assertEqual(rest[-1]["response"]["usage"], {"input_tokens": 70, "output_tokens": 10, "total_tokens": 80})

    async def test_a_stopped_failed_or_timed_out_session_is_not_kept(self):
        run, events = await self.start(FakeTaskContext())
        stopping = asyncio.ensure_future(run.stop())
        await until(lambda: self.client.sessions[-1].cancels == 1)
        self.client.sessions[-1].end(1, "cancelled", USAGE)
        self.assertEqual(await stopping, "stopped")
        _, events = await self.start(FakeTaskContext())
        self.client.sessions[-1].end(1, "cancelled", USAGE)
        with self.assertRaises(HarnessTurnError):
            [event async for event in events]
        self.use(settings(turn_timeout_seconds=0.2))
        _, events = await self.start(FakeTaskContext())
        with self.assertRaisesRegex(HarnessTurnError, "ran past"):
            [event async for event in events]
        self.assertEqual(self.host.idle(), [])
        self.assertTrue(all(session.retired for session in self.client.sessions))

    async def test_a_session_whose_turn_took_the_cancel_is_kept(self):
        self.use(plain())
        task, steering = FakeTaskContext(), request("D2", "Make it two.")
        run, events = await self.start(task)
        session = self.client.sessions[-1]
        collector = asyncio.ensure_future(read_all(events))
        session.say("m1", "Let me ", "look.", thought=False)
        task.steer_with(steering)
        run.steer()
        await until(lambda: session.cancels == 1)
        session.end(1, "cancelled", USAGE)
        await until(lambda: len(session.texts("prompt")) == 2)
        session.say("m2", "Two ", "it is.")
        session.end(2, usage=LATER_USAGE)
        rest = await collector
        self.assertEqual((answers(rest, "commentary"), answers(rest)), (["Let me look."], ["Two it is."]))
        self.assertEqual(self.host.idle(), [session])
        # What it read and wrote, in order (the cancelled turn's text before the batch), is the task's view, so the next
        # task's prompt is only its new request.
        commentary, answer = finished_items(rest)
        await self.start(self.next_task([*task.history()[: -len(steering)], commentary, *steering, answer]))
        self.assertEqual((len(self.client.sessions), session.texts("prompt")[2]), (1, "And black tea?"))

    async def test_a_session_answered_across_the_cancel_is_not_kept(self):
        self.use(plain())
        task = FakeTaskContext()
        run, events = await self.start(task)
        session = self.client.sessions[-1]
        collector = asyncio.ensure_future(read_all(events))
        session.say("m1", "Green ", "tea.")
        task.steer_with(STEERING)
        run.steer()
        await until(lambda: session.cancels == 1)
        session.end(1, usage=USAGE)
        self.assertEqual(answers(await collector), ["Green tea."])
        self.assertEqual(task.settled, [("steer_1", "not_applied")])
        self.assertEqual(self.host.idle(), [])
        self.assertTrue(session.retired)

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
                run, _ = await self.start(task)
                self.assertEqual((run.session, len(self.client.sessions)), (session, 1))
                self.assertEqual(session.texts("prompt")[1], self.host.renderer.transcript(sent))

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
                # Untouched: still pooled, nothing more sent, no place taken, its door still its old task's.
                self.assertEqual((self.host.idle(), len(session.sent), self.host.runtime.places), ([session], 1, 1))
                self.assertEqual((kept.door.task, self.host.bridge.closed), (self.first, []))

    async def test_a_continuing_task_takes_its_threads_session(self):
        # W1's session went idle first: a task takes its own thread's, the idle longest or not.
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
        # The agent may have ended the session meanwhile: nothing of the task reached it.
        session, view = await self.answered()
        task = self.next_task(view)
        _, events = await self.start(task)
        collector = asyncio.ensure_future(read_all(events))
        session.end(session.last_prompt(), stop=None, error="RequestError: Session not found")
        await until(lambda: len(self.client.sessions) == 2 and self.client.sessions[1].sent)
        fresh = self.client.sessions[1]
        self.assertTrue(session.retired)
        self.assertEqual(fresh.texts("prompt"), [self.host.renderer.worker_input(task.history())])
        fresh.say("m2", "Black ", "too.")
        fresh.end(1, usage={"totalTokens": next(self.totals)})
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
    def make(self, harness=None, **block):
        self.host = FakeHost(harness or settings(**block))
        self.conductor = HarnessConductor(self.host)
        self.target = FakeDecisionTarget()
        self.sessions = self.host.runtime.client.sessions
        # The agent's running token total, which grows with each model call.
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
        await self.conductor.door.call("start_work", {"delegations": [given_input.arrivals[0].delegation_id]})
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
        task, session = await self.deciding(given(a_request("d_new", "Find teas."), works=works, conversation=EARLIER))
        [prompt] = session.texts("prompt")
        for part in (
            "Earlier: book a table.",
            "New requests to place:\n[d_new] Find teas.",
            "[W1] running, carried by d_old; requests: [d_old] Find teas.",
            "task W1.1: running for 3 s; steps: get_menu: done; newest note: Looking it up.",
            "[W0] answered, carried by d_done",
        ):
            self.assertIn(part, prompt)
        folder = self.host.runtime.client.folders[0]
        self.assertEqual((folder / ".hermes.md").read_text().strip(), self.host.vocabulary.instructions(3))
        self.assertEqual(self.host.runtime.client.servers[0].name, "c1")
        tools = [tool["name"] for tool in self.conductor.door.tools()]
        self.assertEqual(tools, ["start_work", "steer_work", "cancel_work", "answer_status"])
        malformed = await self.conductor.door.call("steer_work", {"delegations": ["d_new"]})
        self.assertEqual(json.loads(malformed), {"error": "steer_work: missing work_id"})
        outcome = await self.conductor.door.call("steer_work", {"work_id": "W1", "delegations": ["d_new"]})
        self.assertEqual(outcome, "Sent to W1; it applies at the next step.")
        session.say("m1", "Placed ", "it.")
        session.end(1, usage=DECIDED)
        await task
        self.assertEqual(self.target.decisions, [Steer("W1", ("d_new",))])
        self.assertIs(await self.placed(given(a_request("d_2", "Hours?"), conversation=EARLIER, new_from=1)), session)
        self.assertNotIn("Earlier: book a table.", session.texts("prompt")[1])

    async def test_a_plain_conductor_reads_its_prompt_atop_the_first(self):
        self.make(plain())
        task, session = await self.deciding(given(a_request("d_1", "Hi.")))
        instructions = self.host.vocabulary.instructions(3)
        self.assertTrue(session.texts("prompt")[0].startswith(instructions + "\n\n"))
        self.assertEqual(list(self.host.runtime.client.folders[0].iterdir()), [])
        await self.conductor.door.call("start_work", {"delegations": ["d_1"]})
        session.end(1)
        await task
        self.assertIs(await self.placed(given(a_request("d_2", "Hours?"))), session)
        self.assertNotIn(instructions, session.texts("prompt")[1])
        self.assertEqual(self.target.decisions, [Start(("d_1",)), Start(("d_2",))])

    async def test_a_failed_turn_or_lost_runtime_gives_a_fresh_session(self):
        self.make()
        task, session = await self.deciding(given(a_request("d_1", "Teas?")))
        session.end(1, "refusal")
        await task
        self.assertEqual((session.retired, self.target.decisions), (True, []))
        fresh = await self.placed(given(a_request("d_2", "Hours?")))
        self.assertIsNot(fresh, session)
        self.host.runtime.client.alive = False
        await self.placed(given(a_request("d_3", "Teas?")))
        self.assertEqual((len(self.sessions), fresh.retired), (3, True))

    async def test_a_turn_is_read_to_its_end_however_long_it_runs(self):
        # Only the backstop at turn_timeout_seconds cuts a turn, which may be the agent compacting its memory.
        self.make()
        task, session = await self.deciding(given(a_request("d_1", "Teas?")))
        await self.conductor.door.call("start_work", {"delegations": ["d_1"]})
        # The agent works on: half a minute passes on the loop's clock.
        loop = asyncio.get_running_loop()
        clock = loop.time
        loop.time = lambda: clock() + 30
        await turns()
        self.assertFalse(task.done())
        session.end(1, usage=self.used())
        await task
        self.assertEqual((session.cancels, session.retired, self.target.decisions), (0, False, [Start(("d_1",))]))

    async def test_a_restarted_decide_reads_its_turn_to_its_end(self):
        # The IM restarts a decide() that placed nothing when a request arrives: the agent's turn runs on uncut, and
        # what it places counts. The IM builds the next input once the agent is free, and it goes out as one prompt.
        self.make()
        session = await self.placed(given(a_request("d_0", "Book."), conversation=EARLIER))
        task, _ = await self.deciding(given(a_request("d_1", "Teas?"), conversation=EARLIER, new_from=1))
        task.cancel()
        await turns()
        self.assertFalse(task.done())
        await self.conductor.door.call("start_work", {"delegations": ["d_1"]})
        session.end(2, usage=self.used())
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual((self.target.decisions[-1], session.cancels), (Start(("d_1",)), 0))
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
                await self.conductor.door.call("start_work", {"delegations": ["d_1"]})
                self.assertEqual((first.decisions, self.target.decisions), ([Start(("d_1",))], []))
                session.end(1, usage=self.used())
                await until(lambda session=session: len(session.texts("prompt")) == 2)
                await self.conductor.door.call("start_work", {"delegations": ["d_2"]})
                session.end(2, usage=self.used())
                await later
                decided = (self.target.decisions, session.cancels, session.retired)
                self.assertEqual(decided, ([Start(("d_2",))], 0, False))

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
        await self.conductor.door.call("start_work", {"delegations": ["d_2"]})
        fresh.end(1, usage=self.used())
        await later
        self.assertEqual(self.target.decisions, [Start(("d_2",))])

    async def test_a_turn_past_turn_timeout_seconds_is_cancelled_once_and_its_session_replaced(self):
        # The backstop holds whether a decide() reads the turn or none does.
        for reading in (True, False):
            with self.subTest(reading=reading):
                self.make(turn_timeout_seconds=0.2)
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

    async def test_a_turn_with_no_model_call_retires_a_hermes_session_only(self):
        # Hermes ends a turn whose model call failed end_turn, with text of its own: only its usage tells.
        for name, harness, usages, retired in (
            ("hermes, no usage", settings(), [None], True),
            ("hermes, an unchanged total", settings(), [{"totalTokens": 100}, {"totalTokens": 100}], True),
            ("hermes, a first turn with no tokens", settings(), [{"totalTokens": 0}], True),
            ("a plain agent, no usage", plain(), [None], False),
        ):
            with self.subTest(name):
                self.make(harness)
                for number, usage in enumerate(usages, 1):
                    task, session = await self.deciding(given(a_request(f"d_{number}", "Teas?")))
                    session.end(session.last_prompt(), usage=usage)
                    await task
                self.assertEqual((session.retired, session.cancels), (retired, 0))

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
        await later
        self.assertEqual((session.cancels, session.retired, self.target.decisions), (1, True, []))
        fresh = await self.placed(given(a_request("d_3", "Sunday?"), conversation=EARLIER, new_from=1))
        self.assertIsNot(fresh, session)
        self.assertIn("Earlier: book a table.", fresh.texts("prompt")[0])
