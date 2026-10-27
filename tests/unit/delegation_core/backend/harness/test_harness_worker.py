# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Tests for the worker run on a session double: rounds, the turn parse, app calls, steering and stops."""

import asyncio
import unittest

from delegation_core.backend.harness.protocol import CallRunning, CallSettled, ThoughtChunk, ToolCard, ToolCardEnd
from delegation_core.backend.harness.worker import HarnessTurnError, WorkerRun
from tests.unit.contracts.backend import MENU_FUNCTION, FakeTaskContext, check_events
from tests.unit.delegation_core.backend.harness.doubles import FakeHost, plain, settings, until, wire

STEERING = [{"type": "message", "role": "user", "content": "Make it two."}]
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

    async def test_the_recorded_pong_turn_is_refused_as_a_lone_chunk_the_runtime_could_have_written(self):
        # The recorded "pong" came as one chunk after reasoning, in a turn that streamed no other text. On the wire that
        # is what Hermes sends when the model only reasoned (its reasoning as the reply) or a model call failed after
        # reasoning (its error copy, under the reasoning's id), so the run cannot tell them apart: it fails closed.
        error = await self.assert_refused("pong")
        self.assertIn("not streamed", str(error))

    async def test_a_lone_chunk_after_reasoning_in_a_turn_that_streamed_no_text_is_never_spoken(self):
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

    async def test_a_provider_error_text_is_never_spoken(self):
        error = await self.assert_refused("provider_error")  # "NVIDIA NIM rejected your API key...", no usage
        self.assertIn("no usage", str(error))

    async def test_an_idle_cancel_poisoned_prompt_is_never_spoken(self):
        error = await self.assert_refused("idle_cancel_poison")  # "Stopped waiting for another Hermes process..."
        self.assertIn("no usage", str(error))

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

    async def test_a_turn_it_ended_otherwise_still_fails_the_task(self):
        session = await self.open(harness=plain())
        session.say("m1", "Partial.", thought=False)
        session.end(1, "max_tokens")
        self.assertIn("max_tokens", str(await self.failure()))


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

    async def test_the_runtimes_own_tools_are_reported_with_its_label(self):
        session = await self.open(runtime_label="hermes")
        session.say("m1", "Searching ", "now.")
        session.put(ToolCard("tc9", "web_search", {"query": "teas"}, bridged=False))
        session.put(ToolCardEnd("tc9", False, "three teas"))
        session.say("m2", "Three ", "teas.")
        session.end(1, usage=USAGE)
        events = await self.rest()
        self.assertEqual(self.task.started, [("tc9", "hermes")])
        self.assertEqual(self.task.outcomes["tc9"], ("three teas", "done"))
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

    async def test_a_refused_app_call_is_answered_not_run_and_the_round_goes_on(self):
        session = await self.open(task=RefusingTask(refuse={"get_menu"}, declared=[MENU_FUNCTION]))
        collector = self.collecting()
        self.assertEqual(await self.run.door.call("get_menu", {}), self.host.settings.prompts.not_run)
        session.say("m2", "I could ", "not.")
        session.end(1, usage=USAGE)
        events = await collector
        self.assertEqual(said(events, "final_answer"), ["I could not."])
        self.assertEqual([call["name"] for call in items(events, "function_call")], ["get_menu"])


class SteeringTests(Harness):
    async def steer(self, session, *, begin=True):
        if begin:
            session.put(ThoughtChunk("m0"))  # the turn has begun: the runtime's first model call reasons
        self.steering = self.task.steer_with(STEERING)
        self.run.steer()
        await until(lambda: session.texts("steer"))
        [steer] = session.texts("steer")  # the agent class words it (Hermes: "/steer ..."; acp tests)
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
        await until(lambda: len(session.texts("prompt")) == 2)  # not taken in the turn: the next prompt
        session.say("m2", "Two ", "done.")
        session.end(3, usage={"totalTokens": 200})
        events = await collector
        self.assertEqual(said(events, "final_answer"), ["Two done."])

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

    async def test_a_steer_the_turn_did_not_take_makes_its_answer_commentary_and_goes_as_the_next_prompt(self):
        session = await self.open()
        collector = self.collecting()
        await self.steer(session)
        session.end(2)  # acked
        session.say("m1", "Old ", "answer.")
        session.end(1, usage=USAGE)  # the model answered: Hermes drops a steer still waiting (pending_steer)
        await until(lambda: len(session.texts("prompt")) == 2)
        self.assertEqual(self.task.settled, [("steer_1", "applied")])
        self.assertIn("Make it two.", session.texts("prompt")[1])
        session.say("m2", "New ", "answer.")
        session.end(3, usage={"totalTokens": 200})
        events = await collector
        self.assertEqual(said(events, "commentary"), ["Old answer."])
        self.assertEqual(said(events, "final_answer"), ["New answer."])

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

    async def test_a_command_absorbed_after_the_turn_ended_goes_as_the_next_prompt(self):
        session = await self.open()
        collector = self.collecting()
        await self.steer(session)
        session.say("m1", "Old ", "answer.")
        session.end(1, usage=USAGE)
        session.end(2)  # returned with nothing run
        await until(lambda: len(session.texts("prompt")) == 2)
        session.say("m2", "New ", "one.")
        session.end(3, usage={"totalTokens": 300})
        events = await collector
        self.assertEqual(said(events, "final_answer"), ["New one."])

    async def test_next_prompt_waits_for_the_turns_end(self):
        session = await self.open(steer="next_prompt")
        collector = self.collecting()
        self.task.steer_with(STEERING)
        self.run.steer()
        await asyncio.sleep(0.05)
        self.assertEqual(session.texts("steer"), [])
        session.say("m1", "Old ", "answer.")
        session.end(1, usage=USAGE)
        await until(lambda: len(session.texts("prompt")) == 2)
        self.assertEqual(self.task.settled, [("steer_1", "applied")])
        session.say("m2", "New ", "answer.")
        session.end(2, usage={"totalTokens": 200})
        events = await collector
        self.assertEqual(said(events, "final_answer"), ["New answer."])

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
        self.assertEqual(await held, self.host.settings.prompts.stopped)  # the held call is answered
        session.end(1, "cancelled", USAGE)
        self.assertEqual(await stopping, "stopped")
        self.assertTrue(session.retired)
        self.assertEqual(await self.rest(), self.seen)  # nothing follows the stop

    async def test_a_stop_the_runtime_does_not_confirm_is_unconfirmed(self):
        session = await self.open()
        self.assertEqual(await self.run.stop(), "unconfirmed")
        self.assertEqual(session.cancels, 1)

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

    async def test_the_sessions_turn_is_given_back_only_once_its_prompts_ended(self):
        # Hermes runs at most turns_per_process turns in one process, and a turn holds its place until it ends: a stop
        # the runtime does not confirm leaves the turn running, so its place is not free yet.
        session = await self.open()
        runtime = self.host.runtime
        self.assertEqual(runtime.places, 1)
        self.assertEqual(await self.run.stop(), "unconfirmed")
        self.assertEqual(runtime.places, 1)
        session.end(1, "cancelled", USAGE)
        await until(lambda: runtime.places == 0)


if __name__ == "__main__":
    unittest.main()
