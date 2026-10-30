# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Tests for the worker run on a session double: rounds, the turn parse, app calls, steering and stops."""

import asyncio
import dataclasses
import json
import unittest

from delegation_core.backend.conductors import SteerOpenWork
from delegation_core.backend.harness.protocol import CallRunning, CallSettled, ThoughtChunk, ToolCard, ToolCardEnd
from delegation_core.backend.harness.worker import HarnessTurnError, WorkerRun
from tests.unit.contracts.backend import MENU_FUNCTION, FakeTaskContext, check_events
from tests.unit.delegation_core.backend.harness.doubles import FakeHost, plain, settings, until, wire
from tests.unit.delegation_core.helpers import IMCase, app_outputs, ask, finished_items, read_all

STEERING = [{"type": "message", "role": "user", "content": "Make it two."}]
USER_FUNCTION = {
    "type": "function",
    "name": "get_user_details",
    "description": "A user's profile.",
    "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}},
}
USAGE = {"inputTokens": 80, "outputTokens": 20, "totalTokens": 100}


def items(events, kind="message"):
    return [e["item"] for e in events if e["type"] == "response.output_item.done" and e["item"]["type"] == kind]


def said(events, phase):
    return [i["content"][0]["text"] for i in items(events) if i.get("phase") == phase]


def a_call(name="lookup", call_id="call_1"):
    return {"type": "function_call", "id": f"fc_{call_id}", "call_id": call_id, "name": name, "arguments": "{}"}


class RefusingTask(FakeTaskContext):
    def __init__(self, refuse=(), **kwargs):
        super().__init__(**kwargs)
        self.refuse = set(refuse)

    def route(self, call):
        return "refused" if call.get("name") in self.refuse else super().route(call)

    def refusal(self, call_id):
        return self.texts.clash_output  # the IM refused it for a name the app and a work tool share


class MisroutingTask(FakeTaskContext):
    """A task whose IM routes the app's function to the backend, with a misrouted_output of its own."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.texts = dataclasses.replace(self.texts, misrouted_output="{name} belongs to the app.")

    def route(self, call):
        return "backend"  # the IM routes even the app's function to the backend (it never should)


class Harness(unittest.IsolatedAsyncioTestCase):
    def make(self, task=None, harness=None, **block):
        self.host = FakeHost(harness or settings(**block))
        self.task = task or FakeTaskContext(declared=[MENU_FUNCTION])
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

    async def rest(self):
        return await self.collecting()

    async def failure(self):
        with self.assertRaises(HarnessTurnError) as raised:
            await self.collecting()
        return raised.exception


class OpeningTests(Harness):
    async def test_the_session_opens_in_its_folder_with_the_composed_instructions_and_the_history(self):
        session = await self.open()
        client = self.host.runtime.client
        self.assertEqual(client.servers[0].name, "w1")
        folder = client.folders[0]
        self.assertEqual((folder / ".hermes.md").read_text().strip(), "COMPOSED-INSTRUCTIONS")
        self.assertTrue((folder / ".git").is_dir())  # the pin: the runtime's project search stops here
        [(number, kind, text)] = session.sent
        self.assertEqual((number, kind), (1, "prompt"))
        self.assertIn("Do the task.", text)
        self.assertNotIn("COMPOSED-INSTRUCTIONS", text)  # instructions are the session's own, not the prompt's
        self.assertIn("harness.session_opened", [kind for kind, _ in self.task.traces])

    async def test_a_session_whose_tools_are_never_listed_is_retried_once_then_the_task_fails(self):
        run = self.make()
        self.host.bridge.lists = False
        with self.assertRaisesRegex(RuntimeError, "did not list the session's tools"):
            await anext(run.events())
        self.assertEqual([server.name for server in self.host.runtime.client.servers], ["w1", "w2"])
        self.assertTrue(all(session.retired for session in self.host.runtime.client.sessions))

    async def test_a_plain_agent_gets_the_instructions_at_the_top_of_its_first_prompt(self):
        session = await self.open(harness=plain())
        self.assertTrue(session.texts("prompt")[0].startswith("COMPOSED-INSTRUCTIONS\n\n"))
        self.assertEqual(list(self.host.runtime.client.folders[0].iterdir()), [])  # nothing written, no pin


class TurnParseTests(Harness):
    async def test_a_streamed_answer_is_the_final_answer_with_the_turns_own_usage(self):
        session = await self.open()
        session.say("m1", "Do", "ne.")
        session.end(1, usage=USAGE)
        events = await self.rest()
        self.assertEqual(check_events(events), [])
        self.assertEqual(said(events, "final_answer"), ["Done."])
        self.assertEqual(
            events[-1]["response"]["usage"], {"input_tokens": 80, "output_tokens": 20, "total_tokens": 100}
        )
        self.assertTrue(session.retired)  # released once answered
        self.assertEqual(len(self.host.bridge.closed), 1)

    async def test_a_lone_chunk_after_reasoning_in_a_turn_that_streamed_no_text_is_never_spoken(self):
        # Hermes sends this when the model only reasoned (its reasoning as the reply) or a model call failed after
        # reasoning (its error copy, under the reasoning's id). A real one-chunk answer, as the "pong" of the wire
        # fixture, looks the same on the wire, so the run fails closed.
        session = await self.open()
        session.say("m1", "I should check the balance first, then answer.")  # one chunk, under the reasoning's id
        session.end(1, usage=USAGE)
        error = await self.failure()
        self.assertIn("not streamed", str(error))
        self.assertEqual(items(self.seen), [])

    async def test_a_lone_chunk_after_reasoning_once_the_turn_streamed_text_is_an_answer(self):
        session = await self.open()
        session.say("m1", "Let me ", "check.")  # text streamed: the runtime sends no text of its own in this turn
        session.put(ToolCard("tc1", "mcp__w1__lookup", {}))
        session.put(ToolCardEnd("tc1", False, "ok"))
        session.say("m2", "Done.")
        session.end(1, usage=USAGE)
        events = await self.rest()
        self.assertEqual(said(events, "final_answer"), ["Done."])

    async def assert_refused(self, scenario):
        session = await self.open()
        steps = wire(scenario)
        for step in steps:
            if "update" in step:
                session.replay(step["update"]["update"])
        result = next(step["result"] for step in steps if "end" in step)
        session.end(1, result["stopReason"], result.get("usage"))
        error = await self.failure()
        self.assertEqual(self.seen[-1]["type"], "response.failed")
        self.assertEqual(self.seen[-1]["response"]["error"]["code"], "server_error")
        self.assertEqual(items(self.seen), [])  # nothing of the runtime's own text reaches the IM
        return error

    async def test_the_runtimes_own_text_in_a_turn_with_no_usage_is_never_spoken(self):
        # Two turns of the real Hermes wire with no usage: a provider's error text ("NVIDIA NIM rejected your API
        # key...") and the text of a prompt an idle cancel poisoned ("Stopped waiting for another Hermes process...").
        for scenario in ("provider_error", "idle_cancel_poison"):
            with self.subTest(scenario):
                self.assertIn("no usage", str(await self.assert_refused(scenario)))

    async def test_the_runtimes_own_one_chunk_text_after_a_tool_is_never_spoken(self):
        session = await self.open()
        session.put(ToolCard("tc1", "mcp__w1__lookup", {}))
        session.put(ToolCardEnd("tc1", False, "ok"))
        session.say("m9", "No reply: the turn stopped while a tool result was still pending.", thought=False)
        session.end(1, usage=USAGE)
        error = await self.failure()
        self.assertIn("not streamed", str(error))

    async def test_a_turn_whose_usage_did_not_move_is_no_answer(self):
        session = await self.open()
        session.say("m1", "Do", "ne.")
        session.end(1, usage={"totalTokens": 0})
        error = await self.failure()
        self.assertIn("did not move", str(error))

    async def test_a_turn_that_ends_after_a_tool_without_a_message_is_no_answer(self):
        session = await self.open()
        session.say("m1", "Let me ", "look.")
        session.put(ToolCard("tc1", "mcp__w1__lookup", {}))
        session.put(ToolCardEnd("tc1", False, "ok"))
        session.end(1, usage=USAGE)
        error = await self.failure()
        self.assertIn("without a message after its last tool call", str(error))

    async def test_a_turn_the_runtime_cancelled_or_a_lost_connection_fails_the_task(self):
        session = await self.open()
        session.end(1, "cancelled", USAGE)
        self.assertIn("cancelled", str(await self.failure()))
        session = await self.open()
        session.end(1, None, None, "ConnectionError: Connection closed")
        self.assertIn("Connection closed", str(await self.failure()))


class PlainAgentTests(Harness):
    """A plain ACP agent: every message is its model's, and a turn that ended ``end_turn`` answered."""

    async def test_a_one_chunk_answer_with_no_usage_is_the_answer(self):
        session = await self.open(harness=plain())
        session.say("m1", "Done.", thought=False)  # one chunk, and the agent reports no usage
        session.end(1)
        events = await self.rest()
        self.assertEqual(check_events(events), [])
        self.assertEqual(said(events, "final_answer"), ["Done."])
        self.assertNotIn("usage", events[-1]["response"])

    async def test_its_tool_cards_are_activity_only_the_bridge_reports_its_calls(self):
        session = await self.open(harness=plain())
        session.say("m1", "Looking.", thought=False)
        session.put(ToolCard("tc1", "Looking up the menu", {}))  # a plain agent's cards are read as bridged
        session.put(ToolCardEnd("tc1", False, "ok"))
        session.say("m2", "Found it.", thought=False)
        session.end(1)
        events = await self.rest()
        self.assertEqual(self.task.started, [])
        self.assertEqual(said(events, "commentary"), ["Looking."])
        self.assertEqual(said(events, "final_answer"), ["Found it."])


class RoundTests(Harness):
    async def test_a_new_model_message_after_tool_activity_closes_the_round_as_commentary_and_calls(self):
        session = await self.open()
        session.say("m1", "Let me ", "check.")
        session.put(CallRunning(a_call()))
        session.put(ToolCard("tc1", "mcp__w1__lookup", {}))
        session.put(CallSettled("call_1"))
        session.put(ToolCardEnd("tc1", False, "found"))
        session.say("m2", "Found ", "it.")
        session.end(1, usage=USAGE)
        events = await self.rest()
        self.assertEqual(check_events(events), [])
        self.assertEqual(sum(e["type"] == "response.created" for e in events), 2)
        self.assertEqual(said(events, "commentary"), ["Let me check."])
        self.assertEqual(said(events, "final_answer"), ["Found it."])
        self.assertEqual([call["name"] for call in items(events, "function_call")], ["lookup"])

    async def test_the_runtimes_own_tools_are_reported_with_its_label_and_their_output_as_written(self):
        for output in ("three teas", ""):  # an empty output too
            with self.subTest(output=output):
                session = await self.open(runtime_label="hermes")
                session.say("m1", "Searching ", "now.")
                session.put(ToolCard("tc9", "web_search", {"query": "teas"}, bridged=False))
                session.put(ToolCardEnd("tc9", False, output))
                session.say("m2", "Three ", "teas.")
                session.end(1, usage=USAGE)
                events = await self.rest()
                self.assertEqual(self.task.started, [("tc9", "hermes")])
                self.assertEqual(self.task.outcomes["tc9"], (output, "done"))
                self.assertEqual([call["call_id"] for call in items(events, "function_call")], ["tc9"])


class AppCallTests(Harness):
    async def test_an_app_call_ends_the_round_and_resume_answers_it(self):
        session = await self.open()
        session.say("m1", "Let me ", "look.")
        call = asyncio.ensure_future(self.run.door.call("get_menu", {}))
        events = await self.rest()
        [forwarded] = items(events, "function_call")
        self.assertEqual(forwarded["name"], "get_menu")
        self.assertEqual(said(events, "commentary"), ["Let me look."])
        self.assertFalse(call.done())  # the runtime waits for the app
        self.task.give_outputs([{"type": "function_call_output", "call_id": forwarded["call_id"], "output": "teas"}])
        await self.run.resume()
        self.assertEqual(await call, "teas")
        self.events, self.seen = self.run.events(), []
        session.say("m2", "We have ", "teas.")
        session.end(1, usage=USAGE)
        rest = await self.rest()
        self.assertEqual(said(rest, "final_answer"), ["We have teas."])

    async def test_an_app_call_resumed_without_its_output_is_answered_owed(self):
        session = await self.open()
        call = asyncio.ensure_future(self.run.door.call("get_menu", {}))
        await self.rest()
        await self.run.resume()
        self.assertEqual(await call, self.task.texts.owed_output)
        session.end(1, usage=USAGE)

    async def test_a_second_app_call_waits_for_the_next_round(self):
        session = await self.open()
        first = asyncio.ensure_future(self.run.door.call("get_menu", {"page": 1}))
        second = asyncio.ensure_future(self.run.door.call("get_menu", {"page": 2}))
        events = await self.rest()
        [call_1] = items(events, "function_call")
        self.task.give_outputs([{"type": "function_call_output", "call_id": call_1["call_id"], "output": "one"}])
        await self.run.resume()
        self.assertEqual(await first, "one")
        self.events, self.seen = self.run.events(), []
        events = await self.rest()
        [call_2] = items(events, "function_call")
        self.assertEqual(call_2["arguments"], '{"page": 2}')
        self.assertFalse(second.done())
        session.end(1, usage=USAGE)

    async def test_a_refused_app_call_is_answered_with_the_ims_reason_and_the_round_goes_on(self):
        session = await self.open(task=RefusingTask(refuse={"get_menu"}, declared=[MENU_FUNCTION]))
        collector = self.collecting()
        self.assertEqual(await self.run.door.call("get_menu", {}), self.task.texts.clash_output)
        session.say("m2", "I could ", "not.")
        session.end(1, usage=USAGE)
        events = await collector
        self.assertEqual(said(events, "final_answer"), ["I could not."])
        self.assertEqual([call["name"] for call in items(events, "function_call")], ["get_menu"])

    async def test_an_app_function_the_im_routes_to_the_backend_fails_with_the_cores_misrouted_output(self):
        session = await self.open(task=MisroutingTask(declared=[MENU_FUNCTION]))
        collector = self.collecting()
        misrouted = json.dumps({"error": "get_menu belongs to the app."})  # the task's own template, not core.yaml's
        self.assertEqual(await self.run.door.call("get_menu", {}), misrouted)
        session.say("m2", "I could ", "not.")
        session.end(1, usage=USAGE)
        await collector
        [(call_id, _)] = self.task.started
        self.assertEqual(self.task.outcomes[call_id], (misrouted, "failed"))


class SteeringTests(Harness):
    async def steer(self, session, *, begin=True):
        if begin:
            session.put(ThoughtChunk("m0"))  # the turn has begun: the runtime's first model call reasons
        self.steering = self.task.steer_with(STEERING)
        self.run.steer()
        await until(lambda: session.texts("steer"))
        [steer] = session.texts("steer")  # the agent class words it (Hermes: "/steer ...")
        self.assertIn("Make it two.", steer)

    async def test_a_native_steer_waits_for_the_turns_first_event_so_it_never_races_the_prompt(self):
        session = await self.open()
        collector = self.collecting()
        self.task.steer_with(STEERING)
        self.run.steer()
        await asyncio.sleep(0.05)
        self.assertEqual(session.texts("steer"), [])  # nothing of the turn yet: the steer could run as a prompt
        session.put(ThoughtChunk("m1"))
        await until(lambda: session.texts("steer"))
        session.end(2)
        session.say("m1", "Do", "ne.", thought=False)
        session.end(1, usage=USAGE)
        events = await collector
        self.assertEqual(said(events, "final_answer"), ["Done."])
        self.assertEqual(self.task.settled, [("steer_1", "not_applied")])  # not taken in the turn: the next request

    async def test_a_native_steer_is_applied_when_a_model_call_follows_a_tool_result_after_its_ack(self):
        session = await self.open()
        collector = self.collecting()
        await self.steer(session)
        session.end(2)  # the ack: the command returns while the turn runs
        session.put(ToolCard("tc1", "mcp__w1__lookup", {}))
        session.put(ToolCardEnd("tc1", False, "ok"))
        await asyncio.sleep(0.05)
        self.assertEqual(self.task.settled, [])  # a tool result alone proves nothing yet
        session.say("m2", "Two ", "coming.")
        await until(lambda: self.task.settled)
        self.assertEqual(self.task.settled, [("steer_1", "applied")])
        session.end(1, usage=USAGE)
        events = await collector
        self.assertEqual(said(events, "final_answer"), ["Two coming."])

    async def test_a_native_steer_is_applied_at_a_new_model_call_once_the_transcript_holds_a_tool_row(self):
        session = await self.open()
        session.put(ToolCard("tc1", "mcp__w1__lookup", {}))
        session.put(ToolCardEnd("tc1", False, "ok"))
        session.say("m1", "Looking ", "further.")
        collector = self.collecting()
        await until(lambda: self.run.current_message == "m1")
        await self.steer(session, begin=False)
        session.end(2)
        session.say("m1", " still")  # the model call that ran at the ack goes on: no proof
        await asyncio.sleep(0.05)
        self.assertEqual(self.task.settled, [])
        session.say("m2", "With ", "two.")
        await until(lambda: self.task.settled)
        session.end(1, usage=USAGE)
        events = await collector
        self.assertEqual(said(events, "commentary"), ["Looking further. still"])
        self.assertEqual(said(events, "final_answer"), ["With two."])

    async def test_a_steer_the_turn_did_not_take_leaves_its_answer_the_answer_and_goes_back_to_the_im(self):
        session = await self.open()
        collector = self.collecting()
        await self.steer(session)
        session.end(2)  # acked
        session.say("m1", "Old ", "answer.")
        session.end(1, usage=USAGE)  # the model answered: Hermes drops a steer still waiting (pending_steer)
        events = await collector
        self.assertEqual((said(events, "final_answer"), said(events, "commentary")), (["Old answer."], []))
        self.assertEqual(self.task.settled, [("steer_1", "not_applied")])  # the IM starts it as the next request
        self.assertEqual(len(session.texts("prompt")), 1)

    async def test_the_runtimes_own_text_is_never_commentary_while_a_steer_waits(self):
        # A model call that reasoned and then failed: Hermes ends the turn with its own text (here the model's reasoning
        # as its reply) under the reasoning's id. A steer waits, so the turn's text would become commentary, which the
        # voice may speak as progress: it is dropped instead.
        session = await self.open()
        collector = self.collecting()
        await self.steer(session)  # the turn began with reasoning under m0
        session.end(2)  # acked
        own = "I should check the account balance first, then answer."
        session.say("m0", own, thought=False)  # one chunk, the turn's only text
        session.end(1, usage=USAGE)
        await until(lambda: len(session.texts("prompt")) == 2)  # the steer goes as the next prompt
        session.say("m2", "Two ", "done.")
        session.end(3, usage={"totalTokens": 200})
        events = await collector
        self.assertNotIn(own, [i["content"][0]["text"] for i in items(events)])
        self.assertEqual(said(events, "final_answer"), ["Two done."])

    async def test_a_delivered_steer_joins_after_the_outputs_of_the_calls_before_it(self):
        # Hermes puts the steer after the tool batch's results, so the task's history must too: an output never
        # follows a newer request that came after its call.
        session = await self.open()
        collector = self.collecting()
        session.say("m1", "Let me ", "check.")
        session.put(CallRunning(a_call()))
        session.put(ToolCard("tc1", "mcp__w1__lookup", {}))
        await until(lambda: self.run.calls)
        await self.steer(session, begin=False)  # the round with the call closes; the batch goes as /steer
        session.end(2)  # acked while the call runs
        output = {"type": "function_call_output", "call_id": "call_1", "output": "found"}
        self.task.give_outputs([output])  # the bridge reported the call: its output joins at the next safe point
        session.put(CallSettled("call_1"))
        session.put(ToolCardEnd("tc1", False, "found"))
        session.say("m2", "With ", "two.")  # a new model call after the result: the steer was delivered
        await until(lambda: self.task.settled)
        session.end(1, usage=USAGE)
        await collector
        joined = self.task.items
        self.assertLess(joined.index(output), joined.index(STEERING[0]))

    async def test_steering_that_comes_while_the_session_opens_goes_in_the_first_prompt(self):
        run = self.make()
        self.task.steer_with(STEERING)
        run.steer()  # the IM calls it once the task runs, before the run opened its session
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
        self.assertEqual(said(events, "final_answer"), ["Two it is."])

    async def test_a_command_that_runs_as_a_turn_of_its_own_after_the_turn_ended_is_applied(self):
        session = await self.open()
        collector = self.collecting()
        await self.steer(session)
        session.say("m1", "Old ", "answer.")
        session.end(1, usage=USAGE)  # the turn ended before the command was read: it runs as a prompt
        session.say("m2", "Two ", "now.")
        session.end(2, usage={"totalTokens": 180})
        events = await collector
        self.assertEqual(self.task.settled, [("steer_1", "applied")])
        self.assertEqual(said(events, "final_answer"), ["Two now."])
        self.assertEqual(session.texts("prompt")[1:], [])

    async def test_a_command_absorbed_after_the_turn_answered_goes_back_to_the_im(self):
        session = await self.open()
        collector = self.collecting()
        await self.steer(session)
        session.say("m1", "Old ", "answer.")
        session.end(1, usage=USAGE)
        session.end(2)  # returned with nothing run
        events = await collector
        self.assertEqual(said(events, "final_answer"), ["Old answer."])
        self.assertEqual(self.task.settled, [("steer_1", "not_applied")])
        self.assertEqual(len(session.texts("prompt")), 1)

    async def test_next_prompt_waits_for_the_turns_end_and_an_answer_leaves_it_with_the_im(self):
        session = await self.open(steer="next_prompt")
        collector = self.collecting()
        self.task.steer_with(STEERING)
        self.run.steer()
        await asyncio.sleep(0.05)
        self.assertEqual(session.texts("steer"), [])
        session.say("m1", "Old ", "answer.")
        session.end(1, usage=USAGE)
        events = await collector
        self.assertEqual(said(events, "final_answer"), ["Old answer."])
        self.assertEqual(self.task.settled, [("steer_1", "not_applied")])

    async def test_next_prompt_carries_on_a_turn_that_ended_without_an_answer(self):
        session = await self.open(steer="next_prompt")
        collector = self.collecting()
        self.task.steer_with(STEERING)
        self.run.steer()
        session.say("m1", "Let me ", "look.")
        session.put(ToolCard("tc1", "mcp__w1__lookup", {}))
        session.put(ToolCardEnd("tc1", False, "ok"))
        session.end(1, usage=USAGE)  # no message after its last tool call: no answer, so the steering goes on with it
        await until(lambda: len(session.texts("prompt")) == 2)
        self.assertEqual(self.task.settled, [("steer_1", "applied")])
        self.assertIn("Make it two.", session.texts("prompt")[1])
        session.say("m2", "Two ", "found.")
        session.end(2, usage={"totalTokens": 200})
        events = await collector
        self.assertEqual(said(events, "final_answer"), ["Two found."])

    async def test_cancel_reprompt_cancels_then_a_fresh_session_carries_the_history_with_the_batch(self):
        session = await self.open(steer="cancel_reprompt")
        collector = self.collecting()
        self.task.steer_with(STEERING)
        self.run.steer()
        await until(lambda: session.cancels == 1)
        session.end(1, "cancelled", USAGE)
        client = self.host.runtime.client
        await until(lambda: len(client.sessions) == 2)
        fresh = client.sessions[1]
        self.assertTrue(session.retired)
        self.assertEqual(client.servers[1].name, "w2")  # a new name: never the cancelled session's server again
        self.assertEqual(self.task.settled, [("steer_1", "applied")])
        self.assertIn("Make it two.", fresh.texts("prompt")[0])
        fresh.say("m1", "Two ", "it is.")
        fresh.end(1, usage=USAGE)
        events = await collector
        self.assertEqual(said(events, "final_answer"), ["Two it is."])


class StopTests(Harness):
    async def test_a_stop_cancels_the_running_prompt_retires_the_door_and_confirms(self):
        session = await self.open()
        held = asyncio.ensure_future(self.run.door.call("get_menu", {}))
        await asyncio.sleep(0)
        stopping = asyncio.ensure_future(self.run.stop())
        await until(lambda: session.cancels == 1)
        self.assertEqual(await held, self.task.texts.stopped_output)  # the held call is answered
        session.end(1, "cancelled", USAGE)
        self.assertEqual(await stopping, "stopped")
        self.assertTrue(session.retired)
        self.assertEqual(await self.rest(), self.seen)  # nothing follows the stop

    async def test_no_cancel_is_sent_to_an_idle_session(self):
        session = await self.open()
        session.say("m1", "Do", "ne.")
        session.end(1, usage=USAGE)
        await self.rest()
        self.assertEqual(await self.run.stop(), "stopped")
        self.assertEqual(session.cancels, 0)  # an idle cancel poisons Hermes's next prompt

    async def test_a_turn_past_its_limit_is_cancelled_and_fails(self):
        session = await self.open(turn_timeout_seconds=0.2)
        error = await self.failure()
        self.assertIn("ran past", str(error))
        self.assertEqual(session.cancels, 1)

    async def test_an_unconfirmed_stop_gives_the_sessions_turn_back_only_once_its_prompts_ended(self):
        # Hermes runs at most turns_per_process turns in one process, and a turn holds its place until it ends: a stop
        # the runtime does not confirm leaves the turn running, so its place is not free yet.
        session = await self.open()
        runtime = self.host.runtime
        self.assertEqual(runtime.places, 1)
        self.assertEqual(await self.run.stop(), "unconfirmed")
        self.assertEqual(runtime.places, 1)
        session.end(1, "cancelled", USAGE)
        await until(lambda: runtime.places == 0)


class RepeatedAppCallTests(IMCase):
    """Under the real IM, the agent's app call that repeats the newest call that ran is answered from the record."""

    async def test_the_door_answers_the_repeat_with_the_recorded_answer_and_it_never_reaches_the_app(self):
        host, runs = FakeHost(), []

        class Workers:
            max_running = 1

            def conductor(self):
                return SteerOpenWork()

            def start(self, task):
                runs.append(WorkerRun(task, host))
                return runs[-1]

            async def close(self):
                pass

        im = self.session(Workers())
        settings = {"tools": [USER_FUNCTION]}
        first = im.admit(ask("Find my account."), delegation_id="D1", settings=settings)
        await until(lambda: runs and runs[0].door is not None and host.runtime.client.sessions)
        door, session = runs[0].door, host.runtime.client.sessions[-1]
        lookup = {"user_id": "noh_mlr_9847"}
        answered = asyncio.ensure_future(door.call("get_user_details", lookup))
        [call] = items(await read_all(first.stream), "function_call")
        output = [{"type": "function_call_output", "call_id": call["call_id"], "output": "Error: user not found"}]
        rest = im.admit_client_request(output, delegation_id="D1", settings=settings)
        self.assertEqual(await answered, "Error: user not found")
        repeated = await asyncio.wait_for(door.call("get_user_details", lookup), 5)
        self.assertEqual(repeated, im.texts.repeated_call_output.format(output="Error: user not found"))
        session.say("m2", "Please spell ", "your id.")
        session.end(1, usage=USAGE)
        events = await asyncio.wait_for(read_all(rest.stream), 5)
        self.assertEqual(items(events, "function_call"), [])  # the repeat never went to the app
        self.assertEqual(said(events, "final_answer"), ["Please spell your id."])


# --------------------------------------------------------------------------------------------- one session per call
NEXT = {"type": "message", "role": "user", "content": "And black tea?"}
BETWEEN = {"type": "message", "role": "user", "content": "Hi."}
LATER_USAGE = {"inputTokens": 150, "outputTokens": 30, "totalTokens": 180}


class Lookup:
    """A tool source whose ``lookup`` answers."""

    label = "shop"

    def tools(self):
        return [{"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {}}}]

    async def execute(self, call_id, name, arguments):
        return "found"


class OverBudgetTask(FakeTaskContext):
    """A task whose newest request alone is over the IM's history budget."""

    def history(self):
        raise RuntimeError("Backend context budget exceeded")


class SessionReuseTests(unittest.IsolatedAsyncioTestCase):
    """``backend.harness.session: call``: the session of a task that answered takes the next task that fits."""

    def setUp(self):
        self.use(settings(session="call"))

    def use(self, harness):
        self.host = FakeHost(harness, tools=(Lookup(),))
        self.client = self.host.runtime.client

    async def start(self, task):
        """``task``'s run on the host up to its first event: its session opened or taken over, and prompted."""
        run = WorkerRun(task, self.host)
        events = run.events()
        self.assertEqual((await anext(events))["type"], "response.created")
        return run, events

    async def answer(self, events, session, message_id="m1", usage=USAGE):
        """The session's newest prompt answers; the rest of the run's events."""
        session.say(message_id, "Green ", "tea.")
        session.end(session.last_prompt(), usage=usage)
        return [event async for event in events]

    async def answered(self, task=None):
        """A first task, answered: its session, and the task's view as the IM keeps it (its run's items joined)."""
        self.first = task or FakeTaskContext(declared=[MENU_FUNCTION])
        _, events = await self.start(self.first)
        session = self.client.sessions[-1]
        rest = await self.answer(events, session)
        return session, [*self.first.history(), *finished_items(rest)]

    def next_task(self, view, **kwargs):
        """The next task: its history ``view``, then a new request."""
        return FakeTaskContext(history=[*view, NEXT], declared=[MENU_FUNCTION], task_id="W2.1", **kwargs)

    async def test_an_answered_tasks_session_takes_the_next_task_and_reads_only_what_follows(self):
        session, view = await self.answered()
        self.assertIs(self.host.kept.session, session)
        self.assertFalse(session.retired)
        _, events = await self.start(self.next_task(view))
        self.assertEqual(len(self.client.sessions), 1)  # no session/new
        self.assertEqual(session.texts("prompt")[1], "And black tea?")  # only the new request, as a transcript
        rest = await self.answer(events, session, "m2", usage=LATER_USAGE)
        self.assertEqual(said(rest, "final_answer"), ["Green tea."])
        # ACP counts usage across the session's turns: the second task's turn is what moved since the first.
        self.assertEqual(rest[-1]["response"]["usage"], {"input_tokens": 70, "output_tokens": 10, "total_tokens": 80})

    async def test_a_stopped_failed_or_timed_out_tasks_session_is_never_kept(self):
        run, events = await self.start(FakeTaskContext(declared=[MENU_FUNCTION]))
        stopping = asyncio.ensure_future(run.stop())
        await until(lambda: self.client.sessions[-1].cancels == 1)
        self.client.sessions[-1].end(1, "cancelled", USAGE)
        self.assertEqual(await stopping, "stopped")
        _, events = await self.start(FakeTaskContext(declared=[MENU_FUNCTION]))
        self.client.sessions[-1].end(1, "cancelled", USAGE)  # the runtime ended the turn itself: the task failed
        with self.assertRaises(HarnessTurnError):
            [event async for event in events]
        self.use(settings(session="call", turn_timeout_seconds=0.2))
        _, events = await self.start(FakeTaskContext(declared=[MENU_FUNCTION]))
        with self.assertRaisesRegex(HarnessTurnError, "ran past"):
            [event async for event in events]
        self.assertIsNone(self.host.kept)
        self.assertTrue(all(session.retired for session in self.client.sessions))

    async def test_a_session_cancelled_for_a_reprompt_is_never_kept(self):
        self.use(settings(session="call", steer="cancel_reprompt"))
        task = FakeTaskContext(declared=[MENU_FUNCTION])
        run, events = await self.start(task)
        cancelled = self.client.sessions[-1]
        collector = asyncio.ensure_future(read_all(events))
        task.steer_with(STEERING)
        run.steer()
        await until(lambda: cancelled.cancels == 1)
        cancelled.end(1, "cancelled", USAGE)
        await until(lambda: len(self.client.sessions) == 2)
        fresh = self.client.sessions[1]
        fresh.say("m1", "Two ", "it is.")
        fresh.end(1, usage=USAGE)
        self.assertEqual(said(await collector, "final_answer"), ["Two it is."])
        self.assertTrue(cancelled.retired)
        self.assertIs(self.host.kept.session, fresh)  # the session that read the history with the batch, and answered

    async def test_a_session_sent_a_cancel_is_never_kept_even_when_its_turn_answered(self):
        self.use(settings(session="call", steer="cancel_reprompt"))
        task = FakeTaskContext(declared=[MENU_FUNCTION])
        run, events = await self.start(task)
        session = self.client.sessions[-1]
        collector = asyncio.ensure_future(read_all(events))
        session.say("m1", "Green ", "tea.")
        task.steer_with(STEERING)
        run.steer()
        await until(lambda: session.cancels == 1)
        session.end(1, usage=USAGE)  # the turn answered before the cancel reached the agent, which holds the cancel
        self.assertEqual(said(await collector, "final_answer"), ["Green tea."])
        self.assertEqual(task.settled, [("steer_1", "not_applied")])
        self.assertIsNone(self.host.kept)
        self.assertTrue(session.retired)

    async def test_an_item_that_came_between_or_left_the_history_opens_a_fresh_session(self):
        for name, cut in {
            "an item inserted": lambda view: [view[0], BETWEEN, *view[1:]],
            "an item trimmed": lambda view: view[1:],
        }.items():
            with self.subTest(name):
                self.use(settings(session="call"))
                session, view = await self.answered()
                await self.start(self.next_task(cut(view)))
                fresh = self.client.sessions[-1]
                self.assertIsNot(fresh, session)
                self.assertTrue(session.retired)
                self.assertTrue(fresh.texts("prompt")[0].startswith("The conversation and your task:"))
                self.assertEqual((self.host.kept, self.host.runtime.places), (None, 1))  # the kept one's place is back

    async def test_other_instructions_or_another_toolset_open_a_fresh_session(self):
        for name, kwargs in {
            "other instructions": {"instructions": "OTHER-INSTRUCTIONS"},
            "another toolset": {"declared": [MENU_FUNCTION, USER_FUNCTION]},
        }.items():
            with self.subTest(name):
                self.use(settings(session="call"))
                session, view = await self.answered()
                task = FakeTaskContext(history=[*view, NEXT], task_id="W2.1", **{"declared": [MENU_FUNCTION], **kwargs})
                run, _ = await self.start(task)
                self.assertEqual(len(self.client.sessions), 2)
                self.assertTrue(session.retired)
                self.assertEqual([door for door, _ in self.host.bridge.closed][-1].name, "w1")  # its door closed
                self.assertEqual(run.door.name, "w2")

    async def test_a_rebound_door_keeps_its_name_and_token_and_runs_the_next_tasks_calls_through_that_task(self):
        session, view = await self.answered()
        door = self.host.kept.door
        name, token = door.name, door.token
        task = self.next_task(view)
        run, _ = await self.start(task)
        self.assertIs(run.door, door)
        self.assertEqual((door.name, door.token), (name, token))
        self.assertEqual(await door.call("lookup", {}), "found")
        self.assertEqual([label for _, label in task.started], ["shop"])
        self.assertEqual(self.first.started, [])

    async def test_two_tasks_at_once_use_two_sessions_and_the_first_that_answers_is_kept(self):
        _, first = await self.start(FakeTaskContext(declared=[MENU_FUNCTION]))
        _, second = await self.start(FakeTaskContext(declared=[MENU_FUNCTION], task_id="W1.2"))
        one, two = self.client.sessions
        await self.answer(second, two)
        await self.answer(first, one)
        self.assertIs(self.host.kept.session, two)
        self.assertEqual((one.retired, two.retired), (True, False))

    async def test_a_kept_session_holds_no_turn_and_a_full_process_gives_the_next_task_a_fresh_session(self):
        self.use(settings(session="call", max_running=1, turns_per_process=1))
        session, view = await self.answered()
        runtime = self.host.runtime
        self.assertEqual(runtime.places, 0)  # an idle session holds no turn of its process
        _, events = await self.start(self.next_task(view))
        self.assertEqual((len(self.client.sessions), runtime.places), (1, 1))  # taken over: its turn taken again
        rest = await self.answer(events, session, "m2", usage={"totalTokens": 180})
        self.assertEqual((self.host.kept.session, runtime.places), (session, 0))
        runtime.places = 1  # another session holds the process's one turn
        await self.start(self.next_task([*view, NEXT, *finished_items(rest)]))
        self.assertEqual(len(self.client.sessions), 2)
        self.assertTrue(session.retired)
        self.assertEqual(runtime.places, 2)  # the fresh session's turn: the kept one took none

    async def test_a_task_whose_history_fails_leaves_the_kept_session_retired_and_its_turn_free(self):
        session, view = await self.answered()
        with self.assertRaisesRegex(RuntimeError, "budget exceeded"):
            await self.start(OverBudgetTask(history=[*view, NEXT], declared=[MENU_FUNCTION]))
        self.assertTrue(session.retired)
        self.assertEqual((self.host.kept, self.host.runtime.places), (None, 0))

    async def test_steering_waits_when_the_kept_session_does_not_fit_and_the_fresh_one_fails_to_open(self):
        self.use(settings(session="call", ready_timeout_seconds=0.05))
        session, view = await self.answered()
        task = self.next_task(view[1:])  # an item left the history: the kept session does not fit
        steering = task.steer_with(STEERING)  # it came before the run opened a session
        self.host.bridge.lists = False  # the fresh session never lists its tools
        with self.assertRaisesRegex(RuntimeError, "did not list the session's tools"):
            await self.start(task)
        self.assertTrue(session.retired)
        self.assertEqual(task.settled, [])  # no session read it: it still waits, and the IM starts it as a follow-up
        self.assertIs(task.take_boundary_input().steering, steering)

    async def test_under_task_no_session_is_ever_kept(self):
        self.use(settings())
        session, view = await self.answered()
        self.assertIsNone(self.host.kept)
        self.assertTrue(session.retired)
        await self.start(self.next_task(view))
        self.assertEqual(len(self.client.sessions), 2)


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


class SessionReuseUnderTheImTests(IMCase):
    """Under the real IM, an ended Work's segment begins the next Work's view, so the kept session takes its task."""

    def setUp(self):
        super().setUp()
        self.host = FakeHost(settings(session="call"))
        self.client = self.host.runtime.client
        self.runs = Runs(self.host)

    async def prompted(self, count, index=0):
        """Session ``index`` once it has been sent ``count`` prompts."""
        sessions = self.client.sessions
        await until(lambda: len(sessions) > index and len(sessions[index].texts("prompt")) >= count)
        return sessions[index]

    async def test_the_next_works_task_takes_the_session_and_reads_only_its_request_with_the_records(self):
        im = self.session(self.runs)
        first = im.admit(ask("Find teas."), delegation_id="D1")
        session = await self.prompted(1)
        session.say("m1", "Green ", "tea.")
        session.end(1, usage=USAGE)
        self.assertEqual(said(await read_all(first.stream), "final_answer"), ["Green tea."])
        second = im.admit(ask("And black?"), delegation_id="D2")
        await self.prompted(2)
        self.assertEqual(len(self.client.sessions), 1)
        prompt = session.texts("prompt")[1]
        self.assertTrue(prompt.startswith("And black?\n"))  # the request, then the records block it carries
        self.assertNotIn("Green tea.", prompt)
        session.say("m2", "Black ", "too.")
        session.end(2, usage=LATER_USAGE)
        self.assertEqual(said(await read_all(second.stream), "final_answer"), ["Black too."])
        kinds = [kind for kind, _ in self.traces]
        self.assertEqual([kinds.count(kind) for kind in ("harness.session_opened", "harness.session_reused")], [1, 1])

    async def test_a_typed_message_the_task_never_sent_gives_the_next_work_a_fresh_session(self):
        im = self.session(self.runs)
        menu = {"tools": [MENU_FUNCTION]}
        first = im.admit(ask("Find teas."), delegation_id="D1", settings=menu)
        session = await self.prompted(1)
        call = asyncio.ensure_future(self.runs.runs[0].door.call("get_menu", {}))
        typed = {"type": "message", "role": "user", "content": "With milk, please."}
        events = await read_all(first.stream)
        rest = im.admit_client_request([*app_outputs(events, "teas"), typed], delegation_id="D1", settings=menu)
        self.assertEqual(await call, "teas")
        session.say("m2", "Green ", "tea.")
        session.end(1, usage=USAGE)
        await read_all(rest.stream)
        self.assertIs(self.host.kept.session, session)
        im.admit(ask("And black?"), delegation_id="D2", settings=menu)
        fresh = await self.prompted(1, index=1)
        self.assertTrue(session.retired)
        self.assertIn("With milk, please.", fresh.texts("prompt")[0])  # the whole history, the typed message in it

    async def test_a_request_that_crossed_the_answer_starts_a_follow_up_work_on_the_kept_session(self):
        im = self.session(self.runs)
        first = im.admit(ask("Find teas."), delegation_id="D1")
        session = await self.prompted(1)
        session.say("m1", "Green ", "tea.")
        newer = im.admit(ask("Make it two."), delegation_id="D2")
        await until(lambda: im.ledger.delegations["D2"].work == "W1")
        await until(lambda: session.texts("steer"))  # the native steer, which the answer then crosses
        session.end(2)
        session.end(1, usage=USAGE)
        self.assertEqual(said(await read_all(first.stream), "final_answer"), ["Green tea."])
        await self.prompted(2)
        self.assertEqual(len(self.client.sessions), 1)  # the follow-up Work's task took the kept session
        self.assertTrue(session.texts("prompt")[1].startswith("Make it two.\n"))
        session.say("m2", "Two ", "teas.")
        session.end(3, usage={"totalTokens": 180})
        self.assertEqual(said(await read_all(newer.stream), "final_answer"), ["Two teas."])
        self.assertEqual(im.ledger.delegations["D2"].work, "W2")


if __name__ == "__main__":
    unittest.main()
