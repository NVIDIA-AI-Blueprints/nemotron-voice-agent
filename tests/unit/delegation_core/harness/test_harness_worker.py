# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The harness's worker run and conductor on a session double: turns, app calls, steering, stops and sessions."""

import asyncio
import json
import unittest
from unittest import mock

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
from examples.frontend_backend_live.delegation_core.backend.harness.worker import _CANCELS, HarnessTurnError, WorkerRun
from examples.frontend_backend_live.delegation_core.backend.port import ConductorInput, Start, Steer
from tests.unit.delegation_core.harness.doubles import FakeHost, plain, settings, until, wire
from tests.unit.delegation_core.helpers import (
    MENU,
    STEERING,
    FakeDecisionTarget,
    FakeTaskContext,
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
    read_all,
)

ORDER_FUNCTION = {
    "type": "function",
    "name": "get_order",
    "description": "A cafe order's status.",
    "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
}
USAGE = {"inputTokens": 80, "outputTokens": 20, "totalTokens": 100}
LATER_USAGE = {"inputTokens": 150, "outputTokens": 30, "totalTokens": 180}
NEXT = {"type": "message", "role": "user", "content": "And black tea?"}
BETWEEN = {"type": "message", "role": "user", "content": "Hi."}
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
        self.assertIs(self.host.kept.session, session)
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
    """The worker runs on ``host`` as an IM's backend, one task at a time (the steer rule places each request)."""

    max_running = 1

    def __init__(self, host):
        self.host, self.runs = host, []

    def conductor(self):
        return SteerOpenWork()

    def start(self, task):
        self.runs.append(WorkerRun(task, self.host))
        return self.runs[-1]

    async def close(self):
        pass


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
        im = self.session(self.runs)
        first = im.admit(ask("Find teas."), delegation_id="D1")
        session = await self.prompted(1)
        session.say("m1", "Green ", "tea.")
        session.end(1, usage=USAGE)
        self.assertEqual(answers(await read_all(first.stream)), ["Green tea."])
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

    async def typed_with_the_outputs(self, harness, typed):
        """Have D1's task ask the app for its menu, and send the app's outputs with ``typed``.

        Return the session, the held call's reply and D1's continuation.
        """
        self.host = FakeHost(harness)
        self.client, self.runs = self.host.runtime.client, Runs(self.host)
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


class SessionReuseTests(unittest.IsolatedAsyncioTestCase):
    """The session of a task that answered takes the next task that fits; any other task opens a fresh session."""

    def setUp(self):
        self.use(settings())

    def use(self, harness):
        self.host = FakeHost(harness, tools=(RecordingTools(),))
        self.client = self.host.runtime.client

    async def start(self, task):
        """Run ``task`` on the host up to its first event, its session opened or taken over and prompted."""
        run = WorkerRun(task, self.host)
        events = run.events()
        self.assertEqual((await anext(events))["type"], "response.created")
        return run, events

    async def answer(self, events, session, message_id="m1", usage=USAGE):
        session.say(message_id, "Green ", "tea.")
        session.end(session.last_prompt(), usage=usage)
        return [event async for event in events]

    async def answered(self, task=None):
        """Answer a first task; return its session and the task's view as the IM keeps it, its run's items joined."""
        self.first = task or FakeTaskContext()
        _, events = await self.start(self.first)
        session = self.client.sessions[-1]
        rest = await self.answer(events, session)
        return session, [*self.first.history(), *finished_items(rest)]

    def next_task(self, view, **kwargs):
        return FakeTaskContext(history=[*view, NEXT], task_id="W2.1", **kwargs)

    async def test_an_answered_session_takes_the_next_task(self):
        session, view = await self.answered()
        self.assertIs(self.host.kept.session, session)
        self.assertFalse(session.retired)
        _, events = await self.start(self.next_task(view))
        self.assertEqual(len(self.client.sessions), 1)
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
        self.assertIsNone(self.host.kept)
        self.assertTrue(all(session.retired for session in self.client.sessions))

    async def test_a_session_whose_turn_took_the_cancel_is_kept(self):
        self.use(plain())
        task = FakeTaskContext()
        run, events = await self.start(task)
        session = self.client.sessions[-1]
        collector = asyncio.ensure_future(read_all(events))
        session.say("m1", "Let me ", "look.", thought=False)
        task.steer_with(STEERING)
        run.steer()
        await until(lambda: session.cancels == 1)
        session.end(1, "cancelled", USAGE)
        await until(lambda: len(session.texts("prompt")) == 2)
        session.say("m2", "Two ", "it is.")
        session.end(2, usage=LATER_USAGE)
        rest = await collector
        self.assertEqual((answers(rest, "commentary"), answers(rest)), (["Let me look."], ["Two it is."]))
        self.assertIs(self.host.kept.session, session)
        # What it read and wrote, in order (the cancelled turn's text before the batch), is the task's view, so the next
        # task's prompt is only its new request.
        commentary, answer = finished_items(rest)
        await self.start(self.next_task([*task.history()[: -len(STEERING)], commentary, *STEERING, answer]))
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
        self.assertIsNone(self.host.kept)
        self.assertTrue(session.retired)

    async def test_a_changed_history_or_settings_opens_a_fresh_session(self):
        for name, cut, changes in (
            ("an item inserted", lambda view: [view[0], BETWEEN, *view[1:]], {}),
            ("an item trimmed", lambda view: view[1:], {}),
            ("other instructions", lambda view: view, {"instructions": "OTHER-INSTRUCTIONS"}),
            ("another toolset", lambda view: view, {"declared": [MENU, ORDER_FUNCTION]}),
        ):
            with self.subTest(name):
                self.use(settings())
                session, view = await self.answered()
                run, _ = await self.start(self.next_task(cut(view), **changes))
                fresh = self.client.sessions[-1]
                self.assertEqual(len(self.client.sessions), 2)
                self.assertIsNot(fresh, session)
                self.assertTrue(session.retired)
                self.assertTrue(fresh.texts("prompt")[0].startswith("The conversation and your task:"))
                self.assertEqual((self.host.kept, self.host.runtime.places), (None, 1))
                self.assertEqual([door for door, _ in self.host.bridge.closed][-1].name, "w1")
                self.assertEqual(run.door.name, "w2")

    async def test_two_tasks_use_two_sessions_and_the_first_is_kept(self):
        _, first = await self.start(FakeTaskContext())
        _, second = await self.start(FakeTaskContext(task_id="W1.2"))
        one, two = self.client.sessions
        await self.answer(second, two)
        await self.answer(first, one)
        self.assertIs(self.host.kept.session, two)
        self.assertEqual((one.retired, two.retired), (True, False))

    async def test_a_failed_history_retires_the_kept_session(self):
        session, view = await self.answered()
        task = FakeTaskContext(history=[*view, NEXT])
        task.history = mock.Mock(side_effect=RuntimeError("Backend context budget exceeded"))
        with self.assertRaisesRegex(RuntimeError, "budget exceeded"):
            await self.start(task)
        self.assertTrue(session.retired)
        self.assertEqual((self.host.kept, self.host.runtime.places), (None, 0))


def given(*requests, works=(), conversation=(), new_from=0):
    conversation = (*conversation, *(item for request in requests for item in request.input))
    return ConductorInput(tuple(requests), tuple(works), conversation, new_from)


class ConductorTests(unittest.IsolatedAsyncioTestCase):
    def make(self, harness=None, **block):
        self.host = FakeHost(harness or settings(**block))
        self.conductor = HarnessConductor(self.host)
        self.target = FakeDecisionTarget()
        self.sessions = self.host.runtime.client.sessions

    def prompted(self):
        """Return the session whose prompt is out: the newest one in use with a prompt that has not ended."""
        return next(
            (session for session in reversed(self.sessions) if session.outstanding and not session.retired), None
        )

    async def deciding(self, given_input):
        """Start a ``decide()``; once its prompt went out, return its task and its session."""
        task = asyncio.ensure_future(self.conductor.decide(given_input, self.target.place))
        await until(self.prompted)
        return task, self.prompted()

    async def placed(self, given_input):
        """Run one ``decide()`` whose turn starts work for its first arrival; return the session it ran in."""
        task, session = await self.deciding(given_input)
        await self.conductor.door.call("start_work", {"delegations": [given_input.arrivals[0].delegation_id]})
        session.end(session.last_prompt(), usage=DECIDED)
        await task
        return session

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

    async def test_a_conductor_turn_past_its_limit_is_cancelled(self):
        # Its session is kept only when the turn ends cancelled: one that ends end_turn after its cancel went out may
        # have crossed it, and the agent got the cancel idle.
        ends = {
            "cancelled": {"stop": "cancelled"},
            "end_turn": {},
            "lost": {"stop": None, "error": "lost"},
            "never": None,
        }
        for name, end in ends.items():
            with self.subTest(name):
                self.make(conductor_timeout_seconds=0.2)
                task, session = await self.deciding(given(a_request("d_1", "Teas?")))
                await until(lambda session=session: session.cancels == 1)
                self.assertFalse(task.done())
                if end is not None:
                    session.end(1, **end)
                await task
                self.assertEqual((session.retired, self.target.decisions), (name != "cancelled", []))
                if name == "cancelled":
                    self.assertIs(await self.placed(given(a_request("d_2", "Hours?"))), session)

    async def test_a_cancelled_decide_waits_for_its_turn(self):
        self.make()
        task, session = await self.deciding(given(a_request("d_0", "Book.")))
        # As the turn ends: no cancel goes to a turn that has ended.
        task.cancel()
        session.end(1, usage=DECIDED)
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual((session.cancels, session.retired), (0, False))
        task, _ = await self.deciding(given(a_request("d_1", "Teas?")))
        # The IM's busy rule: a decide() that placed nothing restarts.
        task.cancel()
        await until(lambda: session.cancels == 1)
        self.assertFalse(task.done())
        session.end(2, "cancelled")
        await asyncio.gather(task, return_exceptions=True)
        self.assertIs(await self.placed(given(a_request("d_1", "Teas?"), a_request("d_2", "Green."))), session)

    async def test_a_cancelled_decide_whose_turn_hangs_retires_it(self):
        # A second cancel (the IM's close) ends the wait while the turn runs.
        for again in (False, True):
            with self.subTest(again=again):
                self.make()
                session = await self.placed(given(a_request("d_0", "Book."), conversation=EARLIER))
                newer = given(a_request("d_1", "Teas?"), conversation=EARLIER, new_from=1)
                task, _ = await self.deciding(newer)
                # The double's turn does not end within stop_timeout_seconds.
                task.cancel()
                if again:
                    await until(lambda session=session: session.cancels == 1)
                    task.cancel()
                    session.end(2, "cancelled")
                await asyncio.gather(task, return_exceptions=True)
                self.assertEqual((session.cancels, session.retired), (1, True))
                fresh = await self.placed(newer)
                self.assertIsNot(fresh, session)
                self.assertIn("Earlier: book a table.", fresh.texts("prompt")[0])
