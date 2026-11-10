# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""The talker as Pipecat's LLM service, in either style: each call's request and format, and what streams."""

import asyncio
import unittest
import unittest.mock

import yaml
from pipecat.frames.frames import (
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.nvidia.llm import NvidiaLLMService
from pipecat.tests.utils import SleepFrame, run_test

from examples.frontend_backend_live.heard_log import Heard
from examples.frontend_backend_live.models.decision import CallSpec
from examples.frontend_backend_live.talker import (
    RESPONSE_FORMATS,
    RESPONSE_KEY,
    Request,
    TalkerCall,
    TalkerChat,
    TalkerTools,
    routing_tools,
)
from tests.unit.cascade_helpers import scripted_talker
from tests.unit.core_helpers import FakeChat, chunk, tool_fragment
from tests.unit.frontend_helpers import ENDPOINT, PROMPTS, RELIABILITY, CallHarness, ScriptedChat, decision, tool_call

TEXTS = PROMPTS.cascade
ROUTING = CallSpec("talker", ("speak", "delegate", "wait"), "the hint")
RELAY = CallSpec("writer", ("speak",), "the answer hint")
DECIDE = Request("Decide.", "Runtime data.", "\n\n", "Caller: ", calls=True)
WORD = Request("Word it.", "Update: Earl Grey.")
CONTEXT = [
    {"role": "user", "content": "What teas?"},
    {"role": "assistant", "content": "Let me check."},
    {"role": "developer", "content": "Result: Earl Grey."},
    {"role": "user", "content": "Tea, please."},
    {"role": "user", "content": "And cake."},
]
FUNCTION_FRAMES = (FunctionCallsStartedFrame, FunctionCallInProgressFrame, FunctionCallResultFrame)


class Calls:
    """The voice's side of the talker, recording what the talker tells it; each run makes ``make()``'s call."""

    def __init__(self, make):
        self.make, self.events, self.decisions, self.number = make, [], [], 0

    async def next_call(self, context):
        call = self.make()
        if call is not None:
            call.deadline = asyncio.get_running_loop().time() + RELIABILITY.talker_attempt_timeout_seconds
        return call

    def speaking(self, call, text):
        self.events.append(("speaking", text))

    def finished(self, call, decision, held):
        self.events.append(("finished", decision.action if decision else None, held))
        self.decisions.append(decision)
        return call.text

    async def hand_off(self, call_id):
        self.events.append(("hand_off", call_id))
        return True

    def written(self):
        return []

    def dropped(self, call):
        self.events.append(("dropped",))

    def cancelled(self, call):
        self.events.append(("cancelled",))

    def failed(self, call, error):
        self.events.append(("failed", type(error).__name__))

    def response_started(self):
        self.number += 1
        return self.number

    def response_ended(self):
        return self.number

    def interrupted(self):
        self.events.append(("interrupted",))

    def caller_started(self):
        pass

    def caller_stopped(self):
        pass


def talker(*script, make, level="json_schema", chat_class=FakeChat, service=TalkerChat):
    chat = chat_class(*script)
    built = scripted_talker(service, chat).from_config(ENDPOINT, RELIABILITY, RESPONSE_FORMATS[level])
    built.bind(Calls(make))
    return built, chat


def routing(**changes) -> TalkerCall:
    return TalkerCall(spec=ROUTING, request=DECIDE, streams=True, **changes)


def tool_talker(*script):
    """Build the tool-calling talker on ``script``; each run makes a routing call in its routing format."""
    service, chat = talker(*script, make=lambda: None, service=TalkerTools)
    form = service.routing_format(TEXTS)
    service.calls.make = lambda: routing(format=form)
    return service, chat


async def run(service, messages=()) -> list:
    down, _ = await run_test(service, frames_to_send=[LLMContextFrame(context=LLMContext(list(messages)))])
    return down


async def run_tools(reply: list, settle: float = 0.2) -> tuple:
    """Run the tool-calling talker on ``reply``, with ``settle`` seconds for Pipecat to run the calls it read."""
    service, _ = tool_talker(reply)
    down, up = await run_test(service, frames_to_send=[LLMContextFrame(context=LLMContext()), SleepFrame(settle)])
    [decided] = service.calls.decisions
    return service, decided, down, up


def texts(frames) -> str:
    return "".join(frame.text for frame in frames if isinstance(frame, LLMTextFrame))


class FormatTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_call_adjusts_its_request_and_leaves_the_context(self):
        context = [dict(message) for message in CONTEXT]
        joined = "Result: Earl Grey.\nCaller: Tea, please.\nAnd cake.\n\nRuntime data."
        for call, system, last in (
            (routing(), "Decide.", joined),
            (
                TalkerCall(spec=RELAY, request=WORD, text="Said."),
                "Word it.",
                "Result: Earl Grey.\nTea, please.\nAnd cake.\nUpdate: Earl Grey.",
            ),
        ):
            with self.subTest(call=call.role):
                service, chat = talker(decision("wait"), make=lambda call=call: call)
                await run(service, context)
                said = [{"role": "user", "content": "What teas?"}, {"role": "assistant", "content": "Let me check."}]
                sent = [{"role": "system", "content": system}, *said, {"role": "user", "content": last}]
                self.assertEqual(chat.requests[0]["messages"], sent)
                self.assertEqual(context, CONTEXT)

    async def test_each_call_sends_its_strict_decision_schema(self):
        service, chat = talker(decision("wait"), make=routing)
        await run(service)
        [request] = chat.requests
        self.assertEqual(request["response_format"]["json_schema"]["name"], "voice_decision")
        self.assertTrue(request["response_format"]["json_schema"]["strict"])
        schema = request["response_format"]["json_schema"]["schema"]
        self.assertEqual(list(schema["properties"]), ["action", "speech"])  # the action decodes first
        self.assertEqual(schema["properties"]["action"]["enum"], ["speak", "delegate", "wait"])
        self.assertEqual(schema["properties"]["speech"]["description"], "the hint")
        self.assertEqual((request["temperature"], request["stream"]), (0.0, True))

    async def test_the_entrys_level_is_sent_and_nothing_steps_down(self):
        form = RESPONSE_FORMATS["none"]
        service, chat = talker(decision("wait"), make=lambda: routing(format=form), level="none")
        await run(service)
        self.assertIsNone(chat.requests[0].get("response_format"))

    async def test_one_shot_inference_sends_no_decision_format(self):
        service, chat = talker("A summary.", make=lambda: None, chat_class=ScriptedChat)
        reply = await service.run_inference(LLMContext([{"role": "developer", "content": "Sum up."}]))
        self.assertEqual(reply, "A summary.")
        self.assertEqual(chat.requests[0]["messages"], [{"role": "developer", "content": "Sum up."}])
        self.assertNotIn("response_format", chat.requests[0])

    async def test_the_client_has_the_sdks_timeout_and_set_retries(self):
        service = TalkerChat.from_config(ENDPOINT, RELIABILITY, RESPONSE_FORMATS["json_schema"])
        self.assertEqual(service._client.max_retries, RELIABILITY.provider_retries)
        generic = NvidiaLLMService(api_key="key", base_url=ENDPOINT.base_url)
        self.assertEqual(service._client.timeout, generic._client.timeout)

    def test_routing_keeps_the_calls_and_a_writer_leaves_them_out(self):
        call = {"id": "call_0", "type": "function", "function": {"name": "delegate", "arguments": "{}"}}
        result = {"role": "tool", "content": '{"status": "delegated"}', "tool_call_id": "call_0"}
        asked, checked = {"role": "user", "content": "What teas?"}, {"role": "assistant", "content": "Let me check."}
        made = {"role": "assistant", "tool_calls": [call]}
        context = [asked, {"role": "assistant", "content": "Sure."}, made, result, checked]
        context += [{"role": "user", "content": "Thanks."}]
        joined = {"role": "assistant", "content": "Sure.", "tool_calls": [call]}
        routed = [asked, joined, result, checked, {"role": "user", "content": "Thanks.\n\nRuntime data."}]
        self.assertEqual(DECIDE.messages(context)[1:], routed)
        alone = [asked, made, result, {"role": "user", "content": "Runtime data."}]
        self.assertEqual(DECIDE.messages([asked, made, result])[1:], alone)
        worded = [asked, {"role": "assistant", "content": "Sure.\nLet me check."}]
        self.assertEqual(
            WORD.messages(context)[1:], [*worded, {"role": "user", "content": "Thanks.\nUpdate: Earl Grey."}]
        )


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_speak_streams_its_words_in_a_numbered_response(self):
        service, _ = talker(decision("speak", "Sure, take your time.", size=4), make=routing)
        frames = await run(service)
        self.assertEqual(texts(frames), "Sure, take your time.")
        self.assertGreater(sum(isinstance(f, LLMTextFrame) for f in frames), 2)
        streamed = "".join(event[1] for event in service.calls.events if event[0] == "speaking")
        self.assertEqual(streamed, "Sure, take your time.")
        self.assertEqual(service.calls.events[-1], ("finished", "speak", ""))
        [start] = [frame for frame in frames if isinstance(frame, LLMFullResponseStartFrame)]
        [end] = [frame for frame in frames if isinstance(frame, LLMFullResponseEndFrame)]
        self.assertEqual((start.metadata[RESPONSE_KEY], end.metadata[RESPONSE_KEY]), (1, 1))

    async def test_speech_before_the_action_waits_for_the_action(self):
        reply = '{"speech": "Hi there.", "action": "speak"}'
        service, _ = talker([chunk(reply[:20]), chunk(reply[20:]), chunk(finish="stop")], make=routing)
        self.assertEqual(texts(await run(service)), "Hi there.")

    async def test_reasoning_before_the_object_is_never_said(self):
        reply = '<think>Plan it.</think>{"action": "speak", "speech": "Hi."}'
        pieces = [chunk(reply[:12]), chunk(reply[12:30]), chunk(reply[30:]), chunk(finish="stop")]
        service, _ = talker(pieces, make=routing)
        self.assertEqual(texts(await run(service)), "Hi.")

    async def test_output_cut_by_the_token_limit_is_unusable(self):
        service, _ = talker(decision("wait", finish="length"), make=routing)
        await run(service)
        self.assertEqual(service.calls.events, [("finished", "delegate", "")])  # unusable output parses as delegate

    async def test_a_result_before_anything_is_said_cuts_the_call(self):
        call = routing(cut=True)
        service, _ = talker(decision("delegate", "Let me check."), make=lambda: call)
        frames = await run(service)
        self.assertEqual((texts(frames), service.calls.events), ("", [("dropped",)]))

    async def test_a_call_past_its_deadline_fails(self):
        class Slow(FakeChat):
            async def create(self, **kwargs):
                await asyncio.sleep(1)
                return await super().create(**kwargs)

        call = routing()
        service, _ = talker(decision("wait"), make=lambda: call, chat_class=Slow)

        async def soon(context):
            call.deadline = asyncio.get_running_loop().time() + 0.05
            return call

        service.calls.next_call = soon
        await run(service)
        self.assertEqual(service.calls.events, [("failed", "TimeoutError")])


class ToolStyleTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_routing_call_offers_the_tools_and_no_format(self):
        service, chat = tool_talker(tool_call("wait"))
        await run(service)
        [request] = chat.requests
        self.assertEqual(request["messages"], DECIDE.messages([]))
        offered = {tool["function"]["name"]: tool for tool in request["tools"]}
        self.assertEqual(list(offered), ["delegate", "wait"])
        for name, text in TEXTS.tools.model_dump().items():
            self.assertEqual(offered[name], {"type": "function", "function": {"name": name, **yaml.safe_load(text)}})
        self.assertEqual(request["tool_choice"], "auto")
        self.assertNotIn("response_format", request)
        self.assertNotIn("parallel_tool_calls", request)

    async def test_plain_text_is_a_reply_spoken_as_it_streams(self):
        reply = [chunk("\n\n"), chunk("Hello there. "), chunk("How can I help?"), chunk(finish="stop")]
        async with CallHarness(reply, style="tools") as call:
            self.assertIsInstance(call.talker, TalkerTools)
            await call.say("Hi there")
            await call.until(lambda: call.said() == ["Hello there. How can I help?"], "the reply")
            self.assertEqual(call.tts.spoken, ["Hello there.", "How can I help?"])
            request = call.request(0)
            framed = TEXTS.tool_frame_before + "You are Remy." + TEXTS.frame_after + "\n" + TEXTS.tool_routing
            self.assertEqual(request["messages"][0]["content"], framed + "\n" + TEXTS.tool_rules)
            self.assertEqual([tool["function"]["name"] for tool in request["tools"]], ["delegate", "wait"])
            self.assertEqual(request["tool_choice"], "auto")
            self.assertNotIn("response_format", request)
            await call.quiet()
            self.assertEqual((len(call.chat.requests), call.port.inputs), (1, []))
            self.assertFalse([m for m in call.context.get_messages() if m.get("tool_calls") or m.get("role") == "tool"])
            heard = (Heard.AGENT, f"{TEXTS.transcript.agent}: Hello there. How can I help?")
            self.assertIn(heard, call.voice.heard_log.lines())

    async def test_an_unusable_call_makes_no_move(self):
        for case, (reply, reason, ran) in {
            "broken arguments": (tool_call("delegate", '{"lead_in": "Let'), "invalid_decision", []),
            "an unknown tool": (tool_call("lookup_order", '{"order": "BB-42"}'), "invalid_decision", ["lookup_order"]),
            "no text and no call": ([chunk(finish="stop")], "empty_output", []),
        }.items():
            with self.subTest(case):
                service, decided, down, _ = await run_tools(reply, settle=0.1)
                self.assertEqual((decided.action, decided.diagnostics["recovery"]), ("delegate", reason))
                settled = [frame for frame in down if isinstance(frame, FunctionCallResultFrame)]
                self.assertEqual([frame.function_name for frame in settled], ran)
                self.assertNotIn("hand_off", [event[0] for event in service.calls.events])

    async def test_arguments_of_any_json_kind_make_the_move(self):
        for case, (reply, action, status) in {
            "a list": (tool_call("wait", "[]"), "wait", "waited"),
            "a string": (tool_call("delegate", '"Earl Grey"'), "delegate", "delegated"),
        }.items():
            with self.subTest(case):
                _, decided, down, _ = await run_tools(reply)
                self.assertEqual((decided.action, decided.speech, decided.call_id), (action, "", "call_0"))
                [settled] = [frame for frame in down if isinstance(frame, FunctionCallResultFrame)]
                self.assertEqual(settled.result, {"status": status})

    async def test_a_whole_call_stands_past_the_token_limit(self):
        lead_in = '{"lead_in": "Let me check."}'
        _, decided, _, _ = await run_tools(tool_call("delegate", lead_in, finish="length"))
        self.assertEqual((decided.action, decided.speech, decided.call_id), ("delegate", "Let me check.", "call_0"))
        _, decided, down, up = await run_tools(tool_call("delegate", lead_in[:-4], finish="length"), settle=0.1)
        self.assertEqual((decided.action, decided.diagnostics["recovery"]), ("delegate", "incomplete_output"))
        self.assertEqual([frame for frame in [*down, *up] if isinstance(frame, FUNCTION_FRAMES)], [])

    async def test_pipecat_runs_a_call_under_the_id_the_voice_read(self):
        def fragments(*parts):
            return [*(chunk(tool_calls=[tool_fragment(0, *part)]) for part in parts), chunk(finish="tool_calls")]

        for case, reply in {
            "no id at all": fragments((None, "delegate", "{}")),
            "the id before the name": fragments(("call_A", None, ""), (None, "delegate", "{}")),
            "a name split, the id on the first": fragments(("call_A", "dele", ""), (None, "gate", "{}")),
        }.items():
            with self.subTest(case):
                service, decided, down, _ = await run_tools(reply)
                [running] = [frame for frame in down if isinstance(frame, FunctionCallInProgressFrame)]
                self.assertEqual((running.function_name, running.tool_call_id), ("delegate", decided.call_id))
                self.assertIn(("hand_off", decided.call_id), service.calls.events)
                self.assertTrue(decided.call_id.startswith("call_"))

    async def test_white_space_before_a_call_is_never_said(self):
        service, _ = tool_talker([chunk("\n"), *tool_call("wait")])
        frames = await run(service)
        # NvidiaLLMService holds leading white space while it looks for <think>, and flushes it at the stream's end.
        self.assertEqual(texts(frames).strip(), "")
        self.assertEqual([event[:2] for event in service.calls.events][0], ("finished", "wait"))

    async def test_a_call_first_reply_stops_ttfb_at_its_first_chunk(self):
        # The rest of the reply waits on a gate, so the time to first byte must stop at the call's first fragment.
        rest, started, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()
        items = [
            chunk(tool_calls=[tool_fragment(0, "call_0", "delegate", "")]),
            rest,
            chunk(tool_calls=[tool_fragment(0, arguments='{"lead_in": "One moment."}')]),
            chunk(finish="tool_calls"),
        ]
        start, stop = TalkerChat.start_ttfb_metrics, TalkerChat.stop_ttfb_metrics

        async def start_ttfb_metrics(service, **kwargs):
            started.set()
            return await start(service, **kwargs)

        async def stop_ttfb_metrics(service, **kwargs):
            if started.is_set():
                stopped.set()
            return await stop(service, **kwargs)

        with (
            unittest.mock.patch.object(TalkerChat, "start_ttfb_metrics", start_ttfb_metrics),
            unittest.mock.patch.object(TalkerChat, "stop_ttfb_metrics", stop_ttfb_metrics),
        ):
            async with CallHarness(items, style="tools") as call:
                await call.say("What teas do you have?")
                await asyncio.wait_for(stopped.wait(), 2)
                rest.set()
                await call.until(lambda: call.port.inputs, "the hand-off")

    def test_without_the_tools_texts_only_the_json_talker_routes(self):
        tools = TEXTS.tools.model_copy(update={"wait": None})
        prompts = TEXTS.model_copy(update={"tool_frame_before": None, "tool_routing": None, "tools": tools})
        json_talker, _ = talker(make=lambda: None)
        self.assertEqual(json_talker.routing_format(prompts).name, "json_schema")
        tools_talker, _ = talker(make=lambda: None, service=TalkerTools)
        lacking = r"^agent_prompts\.Cascade has no tool_frame_before, tool_routing, tools\.wait: "
        with self.assertRaisesRegex(ValueError, lacking):
            tools_talker.routing_format(prompts)

    def test_a_tool_text_that_is_no_schema_is_refused(self):
        for text in ("description: Hands it over.", "[unclosed", "description: x\nparameters: {type: array}"):
            with self.subTest(text=text):
                prompts = TEXTS.model_copy(update={"tools": TEXTS.tools.model_copy(update={"delegate": text})})
                with self.assertRaisesRegex(ValueError, "agent_prompts.Cascade.tools.delegate"):
                    routing_tools(prompts)
