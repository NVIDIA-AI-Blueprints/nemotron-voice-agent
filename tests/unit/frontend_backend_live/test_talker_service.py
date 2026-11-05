# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""The talker as Pipecat's LLM service: the decision format per call, what streams, and one-shot calls out of band."""

import asyncio
import unittest

from pipecat.frames.frames import (
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.tests.utils import run_test

from examples.frontend_backend_live.models.decision import CallSpec
from examples.frontend_backend_live.talker import RESPONSE_FORMATS, RESPONSE_KEY, TalkerCall, TalkerChat
from tests.unit.core_helpers import FakeChat, chunk
from tests.unit.frontend_helpers import ENDPOINT, RELIABILITY, ScriptedChat, decision

ROUTING = CallSpec("talker", ("speak", "delegate", "wait"), "the hint")
RELAY = CallSpec("writer", ("speak",), "the answer hint")


class Calls:
    """The voice's side of the talker, recording what the talker tells it; each run makes ``make()``'s call."""

    def __init__(self, make):
        self.make = make
        self.events: list[tuple] = []
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
        return call.text

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


def talker(*script, make, level="json_schema", chat_class=FakeChat):
    chat = chat_class(*script)
    service = TalkerChat.from_config(ENDPOINT, RELIABILITY, RESPONSE_FORMATS[level], client=chat)
    service.calls = Calls(make)
    return service, chat


def routing(**changes) -> TalkerCall:
    return TalkerCall(spec=ROUTING, messages=[{"role": "user", "content": "hi"}], streams=True, **changes)


async def run(service) -> list:
    down, _ = await run_test(service, frames_to_send=[LLMContextFrame(context=LLMContext())])
    return down


def texts(frames) -> str:
    return "".join(frame.text for frame in frames if isinstance(frame, LLMTextFrame))


class FormatTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_call_sends_its_request_and_its_strict_decision_schema(self):
        call = routing()
        service, chat = talker(decision("wait"), make=lambda: call)
        await run(service)
        [request] = chat.requests
        self.assertEqual(request["messages"], call.messages)
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

    async def test_one_shot_inference_sends_no_decision_format(self):
        service, chat = talker("A summary.", make=lambda: None, chat_class=ScriptedChat)
        reply = await service.run_inference(LLMContext([{"role": "user", "content": "Sum up."}]))
        self.assertEqual(reply, "A summary.")
        self.assertNotIn("response_format", chat.requests[0])

    async def test_a_call_run_out_of_band_sends_its_own_request_and_format_not_streamed(self):
        service, chat = talker('{"action": "speak", "speech": "Tea."}', make=lambda: None, chat_class=ScriptedChat)
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        call = TalkerCall(spec=RELAY, messages=messages, format=RESPONSE_FORMATS["json_schema"])
        reply = await service.run_call(call)
        self.assertEqual(reply, '{"action": "speak", "speech": "Tea."}')
        [request] = chat.requests
        self.assertEqual(request["messages"], call.messages)
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

    async def test_the_client_takes_the_call_timeout_and_the_sdks_retries(self):
        service = TalkerChat.from_config(ENDPOINT, RELIABILITY, RESPONSE_FORMATS["json_schema"])
        self.assertEqual(service._client.max_retries, RELIABILITY.provider_retries)
        self.assertEqual(service._client.timeout, RELIABILITY.client_timeout_seconds)


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
        call = TalkerCall(spec=RELAY, messages=[{"role": "user", "content": "u"}], text="Said.")
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


if __name__ == "__main__":
    unittest.main()
