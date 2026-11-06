# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""The talker as Pipecat's LLM service: each call's request and decision format, what streams, and calls out of band."""

import asyncio
import unittest
from dataclasses import asdict, replace

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
from tests.unit.core_helpers import FakeChat, chunk, tool_fragment
from tests.unit.frontend_helpers import ENDPOINT, PROMPTS, RELIABILITY, ScriptedChat, decision, tool_call

ROUTING = CallSpec("talker", ("speak", "delegate", "wait"), "the hint")
RELAY = CallSpec("writer", ("speak",), "the answer hint")
# A routing call's and a relay's adjustments, as the voice's layout gives them.
DECIDE = Request("Decide.", "Runtime data.", "\n\n", "Caller: ", calls=True)
WORD = Request("Word it.", "Update: Earl Grey.")
# The talker's context: a caller turn, a line that played, a backend result, then two caller turns in a row.
CONTEXT = [
    {"role": "user", "content": "What teas?"},
    {"role": "assistant", "content": "Let me check."},
    {"role": "developer", "content": "Result: Earl Grey."},
    {"role": "user", "content": "Tea, please."},
    {"role": "user", "content": "And cake."},
]
# The frames Pipecat pushes when it runs a function the model called.
FUNCTION_FRAMES = (FunctionCallsStartedFrame, FunctionCallInProgressFrame, FunctionCallResultFrame)


class Calls:
    """The voice's side of the talker, recording what the talker tells it; each run makes ``make()``'s call."""

    def __init__(self, make, accepted=True):
        self.make, self.accepted = make, accepted
        self.events: list[tuple] = []
        self.decisions: list = []  # each decision the talker finished with
        self.number = 0

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
        return self.accepted

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


def talker(*script, make, level="json_schema", chat_class=FakeChat, service=TalkerChat, accepted=True):
    chat = chat_class(*script)
    built = service.from_config(ENDPOINT, RELIABILITY, RESPONSE_FORMATS[level], client=chat)
    built.bind(Calls(make, accepted))
    return built, chat


def routing(**changes) -> TalkerCall:
    return TalkerCall(spec=ROUTING, request=DECIDE, streams=True, **changes)


def tool_talker(*script, accepted=True, **changes):
    """The tool-calling talker on ``script``: each run makes a routing call in its routing format."""
    service, chat = talker(*script, make=lambda: None, service=TalkerTools, accepted=accepted)
    form = service.routing_format(PROMPTS.cascade)
    service.calls.make = lambda: routing(format=form, **changes)
    return service, chat


async def run(service, messages=()) -> list:
    down, _ = await run_test(service, frames_to_send=[LLMContextFrame(context=LLMContext(list(messages)))])
    return down


def texts(frames) -> str:
    return "".join(frame.text for frame in frames if isinstance(frame, LLMTextFrame))


class FormatTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_call_adjusts_the_messages_pipecat_builds_from_the_context_only_in_its_request(self):
        context = [dict(message) for message in CONTEXT]
        for call, sent in (
            (
                routing(),
                [
                    {"role": "system", "content": "Decide."},
                    {"role": "user", "content": "What teas?"},
                    {"role": "assistant", "content": "Let me check."},
                    {
                        "role": "user",
                        "content": "Result: Earl Grey.\nCaller: Tea, please.\nAnd cake.\n\nRuntime data.",
                    },
                ],
            ),
            (
                TalkerCall(spec=RELAY, request=WORD, text="Said."),
                [
                    {"role": "system", "content": "Word it."},
                    {"role": "user", "content": "What teas?"},
                    {"role": "assistant", "content": "Let me check."},
                    {"role": "user", "content": "Result: Earl Grey.\nTea, please.\nAnd cake.\nUpdate: Earl Grey."},
                ],
            ),
        ):
            with self.subTest(call=call.role):
                service, chat = talker(decision("wait"), make=lambda call=call: call)
                await run(service, context)
                self.assertEqual(chat.requests[0]["messages"], sent)
                self.assertEqual(context, CONTEXT)  # the context stays as Pipecat holds it

    async def test_an_update_after_a_line_that_played_is_its_own_user_message(self):
        call = TalkerCall(spec=RELAY, request=WORD, text="Said.")
        service, chat = talker(decision("speak", "Tea."), make=lambda: call)
        await run(service, CONTEXT[:2])
        last_two = [
            {"role": "assistant", "content": "Let me check."},
            {"role": "user", "content": "Update: Earl Grey."},
        ]
        self.assertEqual(chat.requests[0]["messages"][-2:], last_two)

    async def test_each_call_sends_its_strict_decision_schema(self):
        call = routing()
        service, chat = talker(decision("wait"), make=lambda: call)
        await run(service)
        [request] = chat.requests
        self.assertEqual(
            request["messages"],
            [{"role": "system", "content": "Decide."}, {"role": "user", "content": "Runtime data."}],
        )
        self.assertEqual(request["response_format"]["json_schema"]["name"], "voice_decision")
        self.assertTrue(request["response_format"]["json_schema"]["strict"])
        schema = request["response_format"]["json_schema"]["schema"]
        self.assertEqual(list(schema["properties"]), ["action", "speech"])  # the action decodes first
        self.assertEqual(schema["properties"]["action"]["enum"], ["speak", "delegate", "wait"])
        self.assertEqual(schema["properties"]["speech"]["description"], "the hint")
        self.assertEqual((request["temperature"], request["stream"]), (0.0, True))

    async def test_the_level_the_services_entry_declares_is_sent_and_nothing_steps_down(self):
        form = RESPONSE_FORMATS["none"]
        service, chat = talker(decision("wait"), make=lambda: routing(format=form), level="none")
        await run(service)
        self.assertIsNone(chat.requests[0].get("response_format"))

    async def test_one_shot_inference_sends_its_own_request_and_no_decision_format(self):
        service, chat = talker("A summary.", make=lambda: None, chat_class=ScriptedChat)
        reply = await service.run_inference(LLMContext([{"role": "developer", "content": "Sum up."}]))
        self.assertEqual(reply, "A summary.")
        self.assertEqual(chat.requests[0]["messages"], [{"role": "developer", "content": "Sum up."}])
        self.assertNotIn("response_format", chat.requests[0])

    async def test_a_call_run_out_of_band_adjusts_its_context_and_sends_its_format_not_streamed(self):
        service, chat = talker('{"action": "speak", "speech": "Tea."}', make=lambda: None, chat_class=ScriptedChat)
        call = TalkerCall(spec=RELAY, request=WORD, format=RESPONSE_FORMATS["json_schema"])
        reply = await service.run_call(call, LLMContext(list(CONTEXT)))
        self.assertEqual(reply, '{"action": "speak", "speech": "Tea."}')
        [request] = chat.requests
        self.assertEqual(request["messages"], WORD.messages(CONTEXT))  # its system text once, first
        self.assertTrue(request["response_format"]["json_schema"]["strict"])
        self.assertEqual(request["response_format"]["json_schema"]["schema"]["properties"]["action"]["enum"], ["speak"])
        self.assertFalse(request["stream"])
        self.assertNotIn("stream_options", request)

    async def test_a_run_with_nothing_to_do_sends_no_request(self):
        service, chat = talker(make=lambda: None)
        frames = await run(service)
        self.assertEqual(chat.requests, [])
        responses = [type(f) for f in frames if isinstance(f, (LLMFullResponseStartFrame, LLMFullResponseEndFrame))]
        self.assertEqual(responses, [LLMFullResponseStartFrame, LLMFullResponseEndFrame])
        self.assertFalse([frame for frame in frames if isinstance(frame, LLMTextFrame)])

    async def test_a_line_said_with_no_model_call_goes_out_as_the_response(self):
        service, chat = talker(make=lambda: TalkerCall(text="One moment."))
        frames = await run(service)
        self.assertEqual((chat.requests, texts(frames)), ([], "One moment."))
        self.assertEqual(service.calls.events, [("finished", None, "")])

    async def test_the_client_is_pipecats_with_the_sdks_timeout_and_the_settings_retries(self):
        service = TalkerChat.from_config(ENDPOINT, RELIABILITY, RESPONSE_FORMATS["json_schema"])
        self.assertEqual(service._client.max_retries, RELIABILITY.provider_retries)
        generic = NvidiaLLMService(api_key="key", base_url=ENDPOINT.base_url)  # as generic builds its talker
        self.assertEqual(service._client.timeout, generic._client.timeout)


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_routing_speak_streams_its_words_and_the_response_is_numbered(self):
        service, _ = talker(decision("speak", "Sure, take your time.", size=4), make=routing)
        frames = await run(service)
        self.assertEqual(texts(frames), "Sure, take your time.")
        self.assertGreater(sum(isinstance(f, LLMTextFrame) for f in frames), 2)  # as it decoded, not at the end
        streamed = "".join(event[1] for event in service.calls.events if event[0] == "speaking")
        self.assertEqual(streamed, "Sure, take your time.")
        self.assertEqual(service.calls.events[-1], ("finished", "speak", ""))
        [start] = [frame for frame in frames if isinstance(frame, LLMFullResponseStartFrame)]
        [end] = [frame for frame in frames if isinstance(frame, LLMFullResponseEndFrame)]
        self.assertEqual((start.metadata[RESPONSE_KEY], end.metadata[RESPONSE_KEY]), (1, 1))

    async def test_a_delegates_speech_is_held_as_its_lead_in_and_a_wait_says_nothing(self):
        for action, speech in (("delegate", "Let me check."), ("wait", "")):
            with self.subTest(action=action):
                service, _ = talker(decision(action, speech), make=routing)
                frames = await run(service)
                self.assertEqual(texts(frames), "")
                self.assertEqual(service.calls.events, [("finished", action, speech)])

    async def test_a_relay_holds_its_line_for_the_voice(self):
        call = TalkerCall(spec=RELAY, request=WORD, text="Said.")
        service, _ = talker(decision("speak", "We have tea."), make=lambda: call)
        frames = await run(service)
        self.assertEqual(texts(frames), "Said.")  # the voice's words, at the end
        self.assertEqual(service.calls.events, [("finished", "speak", "We have tea.")])

    async def test_speech_written_before_the_action_goes_out_once_the_action_decodes(self):
        reply = '{"speech": "Hi there.", "action": "speak"}'
        service, _ = talker([chunk(reply[:20]), chunk(reply[20:]), chunk(finish="stop")], make=routing)
        frames = await run(service)
        self.assertEqual(texts(frames), "Hi there.")

    async def test_reasoning_before_the_object_never_reaches_the_decision(self):
        reply = '<think>Plan it.</think>{"action": "speak", "speech": "Hi."}'
        service, _ = talker(
            [chunk(reply[:12]), chunk(reply[12:30]), chunk(reply[30:]), chunk(finish="stop")], make=routing
        )
        frames = await run(service)
        self.assertEqual(texts(frames), "Hi.")

    async def test_output_cut_by_the_token_limit_is_unusable(self):
        service, _ = talker(decision("wait", finish="length"), make=routing)
        await run(service)
        self.assertEqual(service.calls.events, [("finished", "delegate", "")])  # unusable output parses as delegate

    async def test_a_result_that_arrives_before_anything_is_said_cuts_the_call(self):
        call = routing(cut=True)
        service, _ = talker(decision("delegate", "Let me check."), make=lambda: call)
        frames = await run(service)
        self.assertEqual((texts(frames), service.calls.events), ("", [("dropped",)]))


class FailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_failed_request_goes_to_pipecats_error_path_and_the_talker_stays_usable(self):
        service, _ = talker(RuntimeError("boom"), make=routing)
        _, up = await run_test(service, frames_to_send=[LLMContextFrame(context=LLMContext())])
        self.assertEqual(service.calls.events, [("failed", "RuntimeError")])
        self.assertTrue([frame for frame in up if type(frame).__name__ == "ErrorFrame"])
        self.assertTrue(service.is_usable)

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


class ToolCallTests(unittest.IsolatedAsyncioTestCase):
    """The tool-calling talker's routing call: its tools, what streams, its decision, and the call Pipecat runs."""

    async def test_a_routing_call_offers_the_routing_tools_of_the_prompt_set_and_no_decision_format(self):
        service, chat = tool_talker(tool_call("wait"))
        await run(service)
        [request] = chat.requests
        self.assertEqual(request["messages"], DECIDE.messages([]))
        offered = {tool["function"]["name"]: tool for tool in request["tools"]}
        self.assertEqual(list(offered), ["delegate", "wait"])
        for name, text in asdict(PROMPTS.cascade.tools).items():
            self.assertEqual(offered[name], {"type": "function", "function": {"name": name, **yaml.safe_load(text)}})
        self.assertEqual(request["tool_choice"], "auto")
        self.assertNotIn("response_format", request)
        self.assertNotIn("parallel_tool_calls", request)

    async def test_its_plain_text_streams_as_it_arrives_and_is_a_speak(self):
        service, _ = tool_talker([chunk("Sure, "), chunk("take "), chunk("your time."), chunk(finish="stop")])
        frames = await run(service)
        self.assertEqual(texts(frames), "Sure, take your time.")
        self.assertGreater(sum(isinstance(f, LLMTextFrame) for f in frames), 2)  # as it arrived, not at the end
        streamed = "".join(event[1] for event in service.calls.events if event[0] == "speaking")
        self.assertEqual(streamed, "Sure, take your time.")
        [decided] = service.calls.decisions
        self.assertEqual((decided.action, decided.speech, decided.call_id), ("speak", "Sure, take your time.", ""))
        self.assertEqual((decided.diagnostics["response_format"], decided.diagnostics["tool_calls"]), ("tools", []))

    async def test_a_delegate_call_goes_to_pipecat_once_the_voice_has_acted_and_its_handler_settles_it(self):
        service, _ = tool_talker(tool_call("delegate", '{"lead_in": "Let me check that."}', size=5))
        down, up = await run_test(service, frames_to_send=[LLMContextFrame(context=LLMContext()), SleepFrame(0.2)])
        self.assertEqual(texts(down), "")
        [decided] = service.calls.decisions
        self.assertEqual(
            (decided.action, decided.speech, decided.call_id), ("delegate", "Let me check that.", "call_0")
        )
        # The voice acted on the decision before Pipecat ran the call's handler, which asked how its hand-off went.
        self.assertEqual([event[0] for event in service.calls.events], ["finished", "hand_off"])
        self.assertEqual(service.calls.events[-1], ("hand_off", "call_0"))
        [running] = [frame for frame in down if isinstance(frame, FunctionCallInProgressFrame)]
        self.assertEqual((running.function_name, running.tool_call_id), ("delegate", "call_0"))
        self.assertEqual(
            (running.arguments, running.cancel_on_interruption), ({"lead_in": "Let me check that."}, False)
        )
        [settled] = [frame for frame in down if isinstance(frame, FunctionCallResultFrame)]
        self.assertEqual((settled.result, settled.properties.run_llm), ({"status": "delegated"}, False))

    async def test_each_handler_settles_with_what_happened(self):
        cases = {
            "an accepted hand-off": (tool_call("delegate"), True, "delegated"),
            "a refused hand-off": (tool_call("delegate"), False, "refused"),
            "a call that handed nothing over": (tool_call("delegate"), None, "nothing_to_delegate"),
            "a wait": (tool_call("wait"), True, "waited"),
        }
        for case, (reply, accepted, status) in cases.items():
            with self.subTest(case):
                service, _ = tool_talker(reply, accepted=accepted)
                down, _ = await run_test(
                    service, frames_to_send=[LLMContextFrame(context=LLMContext()), SleepFrame(0.2)]
                )
                [settled] = [frame for frame in down if isinstance(frame, FunctionCallResultFrame)]
                self.assertEqual(settled.result, {"status": status})

    async def test_a_reply_that_calls_delegate_delegates_whatever_else_it_calls_and_only_that_call_runs(self):
        calls = [
            chunk(tool_calls=[tool_fragment(0, "call_0", "wait", "{}")]),
            chunk(tool_calls=[tool_fragment(1, "call_1", "delegate", "{}")]),
            chunk(finish="tool_calls"),
        ]
        service, _ = tool_talker(calls)
        down, _ = await run_test(service, frames_to_send=[LLMContextFrame(context=LLMContext()), SleepFrame(0.2)])
        [decided] = service.calls.decisions
        self.assertEqual(
            (decided.action, decided.call_id, decided.diagnostics["tool_calls"]),
            ("delegate", "call_1", ["wait", "delegate"]),
        )
        self.assertEqual(
            [frame.function_name for frame in down if isinstance(frame, FunctionCallInProgressFrame)], ["delegate"]
        )

    async def test_a_call_it_cannot_use_goes_to_no_handler(self):
        for case, reply in {
            "broken arguments": tool_call("delegate", '{"lead_in": "Let'),
            "an unknown tool": tool_call("lookup_order", '{"order": "W1"}'),
            "arguments that hold no object": tool_call("wait", "[]"),
        }.items():
            with self.subTest(case):
                service, _ = tool_talker(reply)
                down, up = await run_test(
                    service, frames_to_send=[LLMContextFrame(context=LLMContext()), SleepFrame(0.1)]
                )
                [decided] = service.calls.decisions
                self.assertEqual((decided.action, decided.diagnostics["recovery"]), ("delegate", "invalid_decision"))
                self.assertEqual([frame for frame in [*down, *up] if isinstance(frame, FUNCTION_FRAMES)], [])

    async def test_white_space_before_a_call_is_never_said(self):
        service, _ = tool_talker([chunk("\n"), *tool_call("wait")])
        frames = await run(service)
        # NvidiaLLMService holds leading white space while it looks for <think>, and flushes it at the stream's end.
        self.assertEqual(texts(frames).strip(), "")
        self.assertEqual([event[:2] for event in service.calls.events][0], ("finished", "wait"))

    async def test_a_call_a_result_cuts_before_anything_is_said_runs_nothing(self):
        service, _ = tool_talker(tool_call("delegate", '{"lead_in": "One moment."}'), cut=True)
        down, up = await run_test(service, frames_to_send=[LLMContextFrame(context=LLMContext()), SleepFrame(0.1)])
        self.assertEqual(service.calls.events, [("dropped",)])
        self.assertEqual([frame for frame in [*down, *up] if isinstance(frame, FUNCTION_FRAMES)], [])

    def test_a_tool_text_that_is_not_a_schema_is_refused_with_its_entry(self):
        for text in ("description: Hands it over.", "[unclosed", "description: x\nparameters: {type: array}"):
            with self.subTest(text=text):
                prompts = replace(PROMPTS.cascade, tools=replace(PROMPTS.cascade.tools, delegate=text))
                with self.assertRaisesRegex(ValueError, "agent_prompts.Cascade.tools.delegate"):
                    routing_tools(prompts)


class CallsInRequestTests(unittest.TestCase):
    """The talker's tool calls in its context: a routing request keeps them as Pipecat wrote them, a writer's not."""

    CALL = {"id": "call_0", "type": "function", "function": {"name": "delegate", "arguments": "{}"}}
    CONTEXT = [
        {"role": "user", "content": "What teas?"},
        {"role": "assistant", "content": "Sure."},
        {"role": "assistant", "tool_calls": [CALL]},
        {"role": "tool", "content": '{"status": "delegated"}', "tool_call_id": "call_0"},
        {"role": "assistant", "content": "Let me check."},
        {"role": "user", "content": "Thanks."},
    ]

    def test_a_routing_request_keeps_each_call_and_its_result_a_call_joined_to_the_line_before_it(self):
        self.assertEqual(
            DECIDE.messages(self.CONTEXT)[1:],
            [
                {"role": "user", "content": "What teas?"},
                {"role": "assistant", "content": "Sure.", "tool_calls": [self.CALL]},
                {"role": "tool", "content": '{"status": "delegated"}', "tool_call_id": "call_0"},
                {"role": "assistant", "content": "Let me check."},
                {"role": "user", "content": "Thanks.\n\nRuntime data."},
            ],
        )

    def test_a_writers_request_leaves_them_out(self):
        self.assertEqual(
            WORD.messages(self.CONTEXT)[1:],
            [
                {"role": "user", "content": "What teas?"},
                {"role": "assistant", "content": "Sure.\nLet me check."},
                {"role": "user", "content": "Thanks.\nUpdate: Earl Grey."},
            ],
        )

    def test_a_call_after_a_caller_turn_stands_as_pipecat_wrote_it(self):
        context = [self.CONTEXT[0], *self.CONTEXT[2:4]]
        self.assertEqual(
            DECIDE.messages(context)[1:],
            [
                {"role": "user", "content": "What teas?"},
                {"role": "assistant", "tool_calls": [self.CALL]},
                {"role": "tool", "content": '{"status": "delegated"}', "tool_call_id": "call_0"},
                {"role": "user", "content": "Runtime data."},
            ],
        )


if __name__ == "__main__":
    unittest.main()
