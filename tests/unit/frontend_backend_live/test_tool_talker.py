# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""The tool-calling talker through the frontend's own build: its moves, its hand-offs, and the calls in its context.

Each call runs on a scripted model client that streams text and tool-call chunks, as an endpoint with automatic tool
choice does, through the same pipeline, voice and hand-offs as the JSON talker; Pipecat runs each routing tool's
handler.
"""

import asyncio
import json
import unittest

from pipecat.processors.aggregators import async_tool_messages
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator

from examples.frontend_backend_live.config_manager.schema import MemoryConfig
from examples.frontend_backend_live.heard_log import Heard
from examples.frontend_backend_live.talker import TalkerChat, TalkerTools
from tests.unit.core_helpers import chunk, tool_fragment
from tests.unit.frontend_helpers import CONFIG, PROMPTS, CallHarness, Port, decision, tool_call

TEXTS, LINES = PROMPTS.cascade, PROMPTS.lines
# The routing call's instructions, framed as the prompt set frames the seed's, before the rules of its format.
FRAMED = TEXTS.frame_before + "You are Remy." + TEXTS.frame_after + "\n" + TEXTS.routing
LEAD_IN = '{"lead_in": "Let me check."}'


def system(request: dict) -> str:
    return request["messages"][0]["content"]


def offered(request: dict) -> list[str]:
    """The names of the tools ``request`` offers; Pipecat's own "no tools" is no list."""
    tools = request.get("tools")
    return [tool["function"]["name"] for tool in tools] if isinstance(tools, list) else []


def observed(call: CallHarness) -> list[tuple[str, dict]]:
    """The voice's observations from now on, each as its kind and fields."""
    events, observe = [], call.voice.observe

    def record(kind, **fields):
        events.append((kind, fields))
        observe(kind, **fields)

    call.voice.observe = record
    return events


def completed(events: list[tuple[str, dict]]) -> list[dict]:
    """The fields of each routing decision the voice carried out."""
    return [fields for kind, fields in events if kind == "talker.completed"]


def calls_in(messages) -> list[dict]:
    """The routing tools' calls and their results among ``messages``, as Pipecat wrote them."""
    return [message for message in messages if message.get("tool_calls") or message.get("role") == "tool"]


def results(call: CallHarness) -> list[dict]:
    """Each routing tool call's result in the talker's context, settled in its place or deferred as a note of its own.

    A call still running has none: its placeholder is Pipecat's.
    """
    found = []
    for message in call.context.get_messages():
        payload = async_tool_messages.parse_message(message)
        if message.get("role") == "tool" and payload is None:
            found.append(json.loads(message["content"]))
        elif payload is not None and payload.kind == "final":
            found.append(json.loads(payload.result))
    return found


class ReplyTests(unittest.IsolatedAsyncioTestCase):
    async def test_plain_text_is_a_reply_spoken_as_it_streams(self):
        reply = [chunk("Hello there. "), chunk("How can I help?"), chunk(finish="stop")]
        async with CallHarness(reply, style="tools") as call:
            self.assertIs(type(call.talker), TalkerTools)
            await call.say("Hi there")
            await call.until(lambda: call.said() == ["Hello there. How can I help?"], "the reply")
            self.assertEqual(call.tts.spoken, ["Hello there.", "How can I help?"])
            request = call.request(0)
            self.assertEqual(system(request), FRAMED + "\n" + TEXTS.tool_rules)
            self.assertEqual(offered(request), ["delegate", "wait"])
            self.assertEqual(request["tool_choice"], "auto")
            self.assertNotIn("response_format", request)
            await call.quiet()
            self.assertEqual((len(call.chat.requests), call.port.inputs), (1, []))  # decided once, nothing handed over
            self.assertEqual(calls_in(call.context.get_messages()), [])

    async def test_white_space_before_the_first_word_is_never_said(self):
        reply = [chunk("\n\n"), chunk("Hello there."), chunk(finish="stop")]
        async with CallHarness(reply, style="tools") as call:
            await call.say("Hi there")
            await call.until(lambda: call.said() == ["Hello there."], "the reply")
            await call.quiet()
            # The next delegation's input carries the line as said, with no line break before its words.
            self.assertIn((Heard.AGENT, f"{TEXTS.transcript.agent}: Hello there."), call.voice.heard_log.lines())

    async def test_the_json_talker_offers_no_tools(self):
        async with CallHarness(decision("speak", "Hello.")) as call:
            await call.say("Hi")
            await call.until(lambda: call.said() == ["Hello."], "the reply")
            self.assertIs(type(call.talker), TalkerChat)
            request = call.request(0)
            self.assertEqual(system(request), FRAMED + "\n" + TEXTS.json_rules)
            self.assertTrue(request["response_format"]["json_schema"]["strict"])
            self.assertEqual(offered(request), [])
            self.assertFalse(isinstance(request.get("tool_choice"), str))


class DelegateTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_delegate_call_hands_the_turn_over_once_and_its_lead_in_plays_once_the_part_accepts(self):
        port = Port(answer=asyncio.Event())
        async with CallHarness(tool_call("delegate", LEAD_IN), style="tools", port=port) as call:
            events = observed(call)
            await call.say("What teas do you have?")
            await call.until(lambda: port.asked == 1, "the hand-off")
            await asyncio.sleep(0.3)
            self.assertEqual((call.said(), call.tts.spoken, results(call)), ([], [], []))  # the part has not answered
            port.answer_event.set()
            await call.until(lambda: call.said() == ["Let me check."], "the lead-in")
            await call.quiet()
            self.assertEqual(len(port.inputs), 1)
            self.assertIn("What teas do you have?", port.text())
            self.assertEqual(results(call), [{"status": "delegated"}])
            [decided] = completed(events)
            self.assertEqual((decided["action"], decided["tool_calls"]), ("delegate", ["delegate"]))
            self.assertEqual(len(call.chat.requests), 1)  # the call's result runs no model call
            # Pipecat wrote the call and its result into the context, and the lead-in after them.
            [made, result] = calls_in(call.context.get_messages())
            self.assertEqual(made["tool_calls"][0]["function"], {"name": "delegate", "arguments": LEAD_IN})
            self.assertEqual(result["tool_call_id"], made["tool_calls"][0]["id"])
            self.assertEqual(call.context.get_messages()[-1], {"role": "assistant", "content": "Let me check."})

    async def test_the_lead_in_waits_for_the_parts_acknowledgement_delay(self):
        port = Port(acknowledgement_delay_ms=400)
        async with CallHarness(tool_call("delegate", LEAD_IN), style="tools", port=port) as call:
            await call.say("What teas do you have?")
            await call.until(lambda: port.inputs, "the accepted hand-off")
            accepted = asyncio.get_running_loop().time()
            await call.until(lambda: results(call), "the call's result")
            await asyncio.sleep(0.2)
            self.assertEqual(call.tts.spoken, [])  # the call has settled, and the delay still runs
            await call.until(lambda: call.tts.spoken == ["Let me check."], "the lead-in")
            self.assertGreaterEqual(asyncio.get_running_loop().time() - accepted, 0.4)

    async def test_a_delegate_call_with_no_lead_in_hands_the_turn_over_and_says_nothing(self):
        for arguments in ("", "{}", '{"lead_in": " "}', '{"lead_in": 7}'):
            with self.subTest(arguments=arguments):
                async with CallHarness(tool_call("delegate", arguments), style="tools") as call:
                    await call.say("Book it.")
                    await call.until(lambda: call.port.inputs, "the delegation")
                    await call.until(lambda: results(call), "the call's result")
                    await call.quiet()
                    self.assertEqual((call.said(), call.tts.spoken, len(call.chat.requests)), ([], [], 1))
                    self.assertEqual(results(call), [{"status": "delegated"}])

    async def test_a_refused_hand_off_says_the_refusal_and_never_the_lead_in(self):
        async with CallHarness(tool_call("delegate", LEAD_IN), style="tools", port=Port(refuse=True)) as call:
            await call.say("What teas do you have?")
            await call.until(lambda: call.said() == [LINES.refusal], "the refusal")
            await call.until(lambda: results(call), "the call's result")
            await call.quiet()
            self.assertNotIn("Let me check.", call.tts.spoken)
            self.assertEqual(results(call), [{"status": "refused"}])
            self.assertEqual(call.voice.worknote.open(), [])

    async def test_words_beside_a_delegate_call_answer_the_turn_and_its_hand_off_says_no_lead_in(self):
        words = "Sure, let me look into that."
        for case, reply in {
            "before the call": [chunk(words), *tool_call("delegate", LEAD_IN)],
            "after the call": [*tool_call("delegate", LEAD_IN, finish=None), chunk(words), chunk(finish="tool_calls")],
        }.items():
            with self.subTest(case):
                async with CallHarness(reply, style="tools") as call:
                    events = observed(call)
                    await call.say("What teas?")
                    await call.until(lambda: call.port.inputs, "the delegation")
                    await call.until(lambda: results(call), "the call's result")
                    await call.quiet()
                    self.assertEqual((call.said(), call.tts.spoken), ([words], [words]))
                    self.assertIn(f"{TEXTS.transcript.agent}: {words}", call.port.text().splitlines())
                    self.assertEqual([fields["action"] for fields in completed(events)], ["delegate"])
                    self.assertEqual(results(call), [{"status": "delegated"}])

    async def test_text_in_a_calls_own_chunk_is_never_said_as_stock_pipecat_says_none(self):
        both = chunk("Let me check.", tool_calls=[tool_fragment(0, "call_0", "delegate", "{}")])
        async with CallHarness([both, chunk(finish="tool_calls")], style="tools") as call:
            await call.say("What teas?")
            await call.until(lambda: call.port.inputs, "the delegation")
            await call.quiet()
            self.assertEqual((call.said(), call.tts.spoken), ([], []))


class IdleTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_callers_idle_timer_still_fires_while_the_delegated_work_runs(self):
        # The delegate call settles once the part has accepted the hand-off, as Pipecat's own BackendConnector.delegate
        # settles its call once the request is sent; Pipecat arms no idle timer while a call is open.
        async with CallHarness(tool_call("delegate", LEAD_IN), style="tools") as call:
            idle = []  # the open work at each of the caller's idle spells
            user = next(processor for processor in call.processors if isinstance(processor, LLMUserAggregator))
            user.add_event_handler("on_user_turn_idle", lambda aggregator: idle.append(call.voice.worknote.open()))
            await call.say("What teas do you have?")
            await call.until(lambda: call.said() == ["Let me check."], "the lead-in")
            await call.until(lambda: idle, "the caller's idle spell")
            self.assertEqual(results(call), [{"status": "delegated"}])
            self.assertEqual([work.id for work in idle[0]], ["del_1"])  # the work runs, with no result yet


class ResultTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_ready_result_is_said_once_in_either_style(self):
        # The voice relays every ready result; the delegate call's own result is the hand-off's status, not the answer.
        delegates = {"json": decision("delegate", "Let me check."), "tools": tool_call("delegate", LEAD_IN)}
        for style, delegate in delegates.items():
            with self.subTest(style=style):
                async with CallHarness(delegate, decision("speak", "We have Earl Grey."), style=style) as call:
                    await call.say("What teas do you have?")
                    await call.until(lambda: call.said() == ["Let me check."], "the lead-in")
                    call.port.answer("del_1", "Teas: Earl Grey")
                    await call.until(lambda: len(call.said()) == 2, "the result's line")
                    await call.quiet(0.8)  # past the caller's idle spell, which runs whatever is still due
                    lines = ["Let me check.", "We have Earl Grey."]
                    self.assertEqual((call.said(), call.tts.spoken), (lines, lines))
                    self.assertEqual(len(call.chat.requests), 2)  # the routing call, and one wording of the result
                    self.assertEqual(results(call), [{"status": "delegated"}] if style == "tools" else [])


class WaitTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_wait_call_says_nothing_and_hands_nothing_over(self):
        async with CallHarness(tool_call("wait"), style="tools") as call:
            events = observed(call)
            await call.say("mm-hmm")
            await call.until(lambda: results(call), "the call's result")
            await call.quiet()
            self.assertEqual((call.said(), call.tts.spoken, call.port.inputs), ([], [], []))
            self.assertEqual([fields["action"] for fields in completed(events)], ["wait"])
            self.assertEqual(results(call), [{"status": "waited"}])
            self.assertEqual(len(call.chat.requests), 1)


class FallbackTests(unittest.IsolatedAsyncioTestCase):
    async def falls_back(self, reply: list, reason: str) -> None:
        """With no work open, ``reply`` is delegated with no lead-in and nothing is said, as unusable JSON is."""
        async with CallHarness(reply, style="tools") as call:
            events = observed(call)
            await call.say("Where is my order?")
            await call.until(lambda: call.port.inputs, "the delegation the fallback makes")
            await call.quiet()
            self.assertEqual((call.said(), call.tts.spoken), ([], []))
            [decided] = completed(events)
            self.assertEqual((decided["action"], decided["recovery"]), ("delegate", reason))
            self.assertIn("Where is my order?", call.port.text())
            self.assertEqual(calls_in(call.context.get_messages()), [])  # Pipecat got no call to run

    async def test_a_reply_that_makes_no_usable_move_delegates_the_callers_request(self):
        nameless = [chunk(tool_calls=[tool_fragment(0, "call_0", arguments='{"order": "W1"}')]), chunk(finish="stop")]
        cases = {
            "broken arguments": (tool_call("delegate", '{"lead_in": "Let'), "invalid_decision"),
            "an unknown tool": (tool_call("lookup_order", '{"order": "W1"}'), "invalid_decision"),
            "a call with no name": (nameless, "invalid_decision"),
            "no text and no call": ([chunk(finish="stop")], "empty_output"),
            "a reply cut by the token limit": (tool_call("delegate", LEAD_IN, finish="length"), "incomplete_output"),
        }
        for case, (reply, reason) in cases.items():
            with self.subTest(case):
                await self.falls_back(reply, reason)

    async def test_while_work_is_open_it_says_the_holding_line(self):
        script = (tool_call("delegate"), tool_call("lookup_order", "{}", call_id="call_1"))
        async with CallHarness(*script, style="tools") as call:
            await call.say("Book it.")
            await call.until(lambda: call.port.inputs, "the delegation")
            await call.say("Are you there?")
            await call.until(lambda: call.said() == [LINES.holding], "the holding line")
            self.assertEqual(len(call.port.inputs), 1)


class InterruptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_callers_words_while_a_hand_off_waits_never_cancel_it(self):
        port = Port(answer=asyncio.Event())
        waits = (tool_call("wait", call_id="call_1"), tool_call("wait", call_id="call_2"))
        script = (tool_call("delegate", LEAD_IN), *waits)
        async with CallHarness(*script, style="tools", port=port) as call:
            events = observed(call)
            await call.say("What teas do you have?")
            await call.until(lambda: port.asked == 1, "the hand-off")
            await call.say("mm-hmm")  # its start interrupts while the part has not answered
            await call.until(lambda: len(completed(events)) == 2, "the second turn's decision")
            port.answer_event.set()
            await call.until(lambda: len(results(call)) == 2, "both calls' results")
            await call.quiet()
            self.assertEqual(len(port.inputs), 1)  # handed over once, never cancelled
            self.assertEqual(call.voice.worknote.open()[0].id, "del_1")
            self.assertNotIn("Let me check.", call.tts.spoken)  # the caller spoke after the decision
            self.assertEqual(sorted(result["status"] for result in results(call)), ["delegated", "waited"])
            # The first call's result came after the caller's words: Pipecat deferred it, and the next run settles it
            # in the call's place, so the talker reads the call's own result.
            await call.say("Okay.")
            await call.until(lambda: len(call.chat.requests) == 3, "the third routing call")
            placeholder = '{"status": "delegated"}'
            result = {"role": "tool", "content": placeholder, "tool_call_id": "call_0"}
            self.assertIn(result, call.request(2)["messages"])
            self.assertFalse([m for m in call.request(2)["messages"] if async_tool_messages.parse_message(m)])


class ContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_next_routing_call_reads_the_call_and_its_result_and_the_writer_leaves_them_out(self):
        script = (
            tool_call("delegate", LEAD_IN),
            decision("speak", "We have Earl Grey."),
            [chunk("Sure, one Earl Grey."), chunk(finish="stop")],
        )
        async with CallHarness(*script, style="tools") as call:
            call.voice._word_early = lambda: None  # the relay words at idle, in the pipeline
            await call.say("What teas do you have?")
            await call.until(lambda: call.said() == ["Let me check."], "the lead-in")
            call.port.answer("del_1", "Teas: Earl Grey")
            await call.until(lambda: call.said()[-1:] == ["We have Earl Grey."], "the relay's line")
            await call.say("I will take the Earl Grey.")
            await call.until(lambda: call.said()[-1:] == ["Sure, one Earl Grey."], "the reply")
            relay, routing = call.request(1), call.request(2)
            self.assertEqual(calls_in(relay["messages"]), [])  # the writer offers no tools
            self.assertEqual(offered(relay), [])
            self.assertTrue(relay["response_format"]["json_schema"]["strict"])
            [made, result] = calls_in(routing["messages"])
            self.assertEqual(made["role"], "assistant")
            self.assertEqual(made["tool_calls"][0]["function"]["name"], "delegate")
            self.assertEqual(result, {"role": "tool", "content": '{"status": "delegated"}', "tool_call_id": "call_0"})
            after = routing["messages"][routing["messages"].index(result) + 1]
            self.assertEqual(after, {"role": "assistant", "content": "Let me check."})
            self.assertEqual(offered(routing), ["delegate", "wait"])

    async def test_the_view_shows_the_call_and_its_result_and_the_next_delegation_carries_neither(self):
        script = (tool_call("delegate", LEAD_IN), tool_call("delegate", "{}", call_id="call_1"))
        async with CallHarness(*script, style="tools") as call:
            await call.say("What teas do you have?")
            await call.until(lambda: call.said() == ["Let me check."], "the lead-in")
            await call.quiet()
            kinds = [entry["kind"] for entry in call.voice.conversation()]
            self.assertEqual(kinds, ["caller", "call", "call_result", "spoken"])
            made, result = call.voice.conversation()[1:3]
            self.assertEqual(made["tool_calls"][0]["function"]["name"], "delegate")
            self.assertEqual(result["tool_call_id"], made["tool_calls"][0]["id"])
            await call.say("And a scone.")
            await call.until(lambda: len(call.port.inputs) == 2, "the second delegation")
            transcript = call.port.text()
            self.assertIn("And a scone.", transcript)
            self.assertNotIn("delegate", transcript)
            self.assertNotIn("status", transcript)

    async def test_the_memory_bound_drops_a_call_with_its_result(self):
        config = CONFIG.model_copy(update={"memory": MemoryConfig(context_tokens=1000, compact_at=0.8)})
        script = (tool_call("wait"), tool_call("wait", call_id="call_1"))
        async with CallHarness(*script, style="tools", config=config) as call:
            await call.say("Long details. " * 300)
            await call.until(lambda: len(results(call)) == 1, "the first call's result")
            await call.quiet()
            self.assertEqual(len(calls_in(call.context.get_messages())), 2)
            await call.say("And one more thing.")
            await call.until(lambda: len(call.chat.requests) == 2, "the second routing call")
            await call.until(lambda: len(results(call)) == 1 and results(call)[0]["status"] == "waited", "the result")
            await call.quiet()
            messages = call.context.get_messages()
            self.assertEqual([m.get("content") for m in messages if m.get("role") == "user"], ["And one more thing."])
            [made, result] = calls_in(messages)  # the second turn's call; the first turn's left with its turn
            self.assertEqual((made["tool_calls"][0]["id"], result["tool_call_id"]), ("call_1", "call_1"))
            self.assertEqual(calls_in(call.request(1)["messages"]), [])  # the bound ran before the request


if __name__ == "__main__":
    unittest.main()
