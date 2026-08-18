# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D105

import asyncio
import unittest
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock

from pipecat.frames.frames import (
    LLMTextFrame,
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
    TranscriptionFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

from examples.omni_assistant.nvidia_omni_multimodal_service import (
    NvidiaOmniLLMService,
    NvidiaOmniSettings,
    _TranscriptResponseExtractor,
    audio_message_part,
)


def _chunk(content=None, *, tool_calls=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None)


async def _stream(*chunks):
    for chunk in chunks:
        yield chunk


def _context(messages):
    return SimpleNamespace(get_messages=lambda: messages)


def _user_context():
    """A context carrying one unanswered user turn, as an aggregator pushes."""
    return _context([{"role": "user", "content": "hi"}])


class _FakeTurn:
    """Stand-in for one in-flight Omni turn that can be released or cancelled."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.cancelled = False
        self.completed = False
        self.args: tuple = ()
        self.kwargs: dict = {}


class OmniTurnPreemptionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.service = NvidiaOmniLLMService(api_key="not-needed", base_url="http://localhost:8000/v1")

        # Run turns as plain asyncio tasks and bypass the pipecat task-manager /
        # metrics machinery, which would otherwise require a full pipeline setup.
        self.service.create_task = lambda coro, name=None: asyncio.create_task(coro, name=name)
        self.service.stop_all_metrics = AsyncMock()

        self.turns: list[_FakeTurn] = []

        async def fake_run_turn(*args, **kwargs) -> None:
            turn = _FakeTurn()
            turn.args = args
            turn.kwargs = kwargs
            self.turns.append(turn)
            try:
                await turn.release.wait()
                turn.completed = True
            except asyncio.CancelledError:
                turn.cancelled = True
                raise

        self.service._run_turn = fake_run_turn

    async def asyncTearDown(self) -> None:
        for turn in self.turns:
            turn.release.set()
        await self.service._cancel_pending_request()

    async def _wait_for(self, predicate: Callable[[], bool], timeout: float = 1.0) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not predicate():
            if loop.time() > deadline:
                self.fail("condition was not met within the timeout")
            await asyncio.sleep(0.005)

    def _fill_audio(self, seconds: float = 1.0) -> None:
        # PCM16 mono payload comfortably above the min_user_audio_secs gate (0.3s).
        nbytes = int(self.service._sample_rate * self.service._channels * 2 * seconds)
        self.service._audio_buffer = [b"\x00" * nbytes]

    async def test_audio_turn_preempts_in_flight_turn(self) -> None:
        self._fill_audio()
        await self.service._maybe_run_audio_turn()
        first_task = self.service._pending_request
        self.assertIsNotNone(first_task)
        await self._wait_for(lambda: len(self.turns) == 1)
        self.assertFalse(first_task.done())

        self._fill_audio()
        await self.service._maybe_run_audio_turn()
        second_task = self.service._pending_request

        # The previous turn must be preempted (cancelled), not skipped...
        self.assertTrue(first_task.cancelled())
        self.assertTrue(self.turns[0].cancelled)
        self.service.stop_all_metrics.assert_awaited()

        # ...and a brand-new turn must have started in its place.
        self.assertIsNotNone(second_task)
        self.assertIsNot(second_task, first_task)
        await self._wait_for(lambda: len(self.turns) == 2)
        self.assertFalse(second_task.done())

    async def test_text_turn_preempts_in_flight_turn(self) -> None:
        await self.service._maybe_run_text_turn(_user_context(), force=True)
        first_task = self.service._pending_request
        self.assertIsNotNone(first_task)
        await self._wait_for(lambda: len(self.turns) == 1)

        await self.service._maybe_run_text_turn(_user_context(), force=True)
        second_task = self.service._pending_request

        self.assertTrue(first_task.cancelled())
        self.assertTrue(self.turns[0].cancelled)
        self.assertIsNot(second_task, first_task)
        await self._wait_for(lambda: len(self.turns) == 2)
        self.assertFalse(second_task.done())

    async def test_context_turn_yields_to_in_flight_audio_turn(self) -> None:
        self._fill_audio()
        await self.service._maybe_run_audio_turn()
        audio_task = self.service._pending_request
        await self._wait_for(lambda: len(self.turns) == 1)

        # Context/run echo for the same spoken turn must yield, not preempt.
        await self.service._maybe_run_text_turn(_user_context(), force=True)

        self.assertIs(self.service._pending_request, audio_task)
        self.assertFalse(audio_task.cancelled())
        self.assertEqual(len(self.turns), 1)
        self.service.stop_all_metrics.assert_not_awaited()

    async def test_audio_turn_below_min_duration_does_not_preempt(self) -> None:
        self._fill_audio()
        await self.service._maybe_run_audio_turn()
        first_task = self.service._pending_request
        await self._wait_for(lambda: len(self.turns) == 1)

        self._fill_audio(seconds=0.05)
        await self.service._maybe_run_audio_turn()

        self.assertIs(self.service._pending_request, first_task)
        self.assertFalse(first_task.cancelled())
        self.assertEqual(len(self.turns), 1)

    async def test_text_turn_is_skipped_when_text_modality_is_disabled(self) -> None:
        self.service._settings.input_modalities = ("audio",)
        await self.service._maybe_run_text_turn(_user_context(), force=True)
        self.assertIsNone(self.service._pending_request)

    async def test_an_utterance_arriving_before_any_context_is_still_answered(self) -> None:
        # An audio-only pipeline may never send a context frame, and the user is
        # waiting either way, so the turn runs without history behind it.
        self.service._context = None
        self._fill_audio()

        await self.service._maybe_run_audio_turn()
        await self._wait_for(lambda: len(self.turns) == 1)

        self.assertEqual(self.turns[0].args[0].get_messages(), [])

    async def test_a_completion_asked_for_before_any_context_does_not_run(self) -> None:
        await self.service._maybe_run_text_turn(None, force=True)
        self.assertIsNone(self.service._pending_request)

    async def test_tool_result_is_answered_even_in_an_audio_only_pipeline(self) -> None:
        # The completion that asked for the call can only be finished by another
        # one, so an audio-only pipeline must not leave the result unspoken.
        self.service._settings.input_modalities = ("audio",)
        context = _context(
            [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            ]
        )

        await self.service._maybe_run_text_turn(context)

        self.assertIsNotNone(self.service._pending_request)
        await self._wait_for(lambda: len(self.turns) == 1)

    async def test_text_only_pipeline_answers_every_context_frame(self) -> None:
        # With no audio path to echo, this behaves like any other LLM service:
        # the aggregator asks for a completion and it runs.
        self.service._settings.input_modalities = ("text",)
        self.service._answered_transcript = "where is the tower?"
        context = _context([{"role": "user", "content": "where is the tower?"}, {"role": "assistant", "content": "hi"}])

        await self.service._maybe_run_text_turn(context)

        self.assertIsNotNone(self.service._pending_request)
        await self._wait_for(lambda: len(self.turns) == 1)

    async def test_context_without_pending_turn_does_not_run(self) -> None:
        context = _context([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
        await self.service._maybe_run_text_turn(context)
        self.assertIsNone(self.service._pending_request)

    async def test_transcript_echo_does_not_answer_the_same_turn_twice(self) -> None:
        self.service._answered_transcript = "where is the tower?"
        context = _context([{"role": "user", "content": "where is the tower?"}])

        await self.service._maybe_run_text_turn(context)

        self.assertIsNone(self.service._pending_request)

    async def test_tool_result_runs_the_follow_up_completion(self) -> None:
        context = _context(
            [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            ]
        )
        await self.service._maybe_run_text_turn(context)
        self.assertIsNotNone(self.service._pending_request)
        await self._wait_for(lambda: len(self.turns) == 1)

    async def test_tool_follow_up_carries_a_spoken_turn_the_context_lacks(self) -> None:
        # The aggregator writes a spoken turn once the assistant response starts,
        # which can be later than the follow-up a tool result asks for, so the
        # follow-up carries the request the user spoke itself.
        self.service._answered_transcript = "what is the weather?"
        context = _context(
            [
                {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            ]
        )

        await self.service._maybe_run_text_turn(context)
        await self._wait_for(lambda: len(self.turns) == 1)

        self.assertEqual(self.turns[0].kwargs["turn_parts"], [{"type": "text", "text": "what is the weather?"}])

    async def test_tool_follow_up_leaves_a_written_spoken_turn_alone(self) -> None:
        self.service._answered_transcript = "what is the weather?"
        context = _context(
            [
                {"role": "user", "content": "what is the weather?"},
                {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            ]
        )

        await self.service._maybe_run_text_turn(context)
        await self._wait_for(lambda: len(self.turns) == 1)

        self.assertIsNone(self.turns[0].kwargs["turn_parts"])

    async def test_tool_result_waits_for_the_turn_that_requested_it(self) -> None:
        # The tool-calling turn is still wrapping up when the follow-up arrives.
        await self.service._maybe_run_text_turn(_user_context(), force=True)
        tool_call_task = self.service._pending_request
        await self._wait_for(lambda: len(self.turns) == 1)

        context = _context(
            [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            ]
        )
        follow_up = asyncio.create_task(self.service._maybe_run_text_turn(context))
        await asyncio.sleep(0)

        # It must wait rather than cancel the turn that issued the tool call.
        self.assertFalse(follow_up.done())
        self.assertFalse(tool_call_task.cancelled())

        self.turns[0].release.set()
        await follow_up

        self.assertTrue(self.turns[0].completed)
        self.assertIsNot(self.service._pending_request, tool_call_task)
        await self._wait_for(lambda: len(self.turns) == 2)


class OmniStreamHandlingTests(unittest.IsolatedAsyncioTestCase):
    """Reasoning comes from NvidiaLLMService; these cover what Omni adds on top."""

    async def asyncSetUp(self) -> None:
        self.service = NvidiaOmniLLMService(api_key="not-needed", base_url="http://localhost:8000/v1")
        self.frames: list = []
        self.service.push_frame = AsyncMock(side_effect=lambda frame, *_a, **_kw: self.frames.append(frame))

    def _begin_turn(self, *, expect_transcript: bool) -> None:
        self.service._reset_response_state()
        self.service._transcript_extractor = _TranscriptResponseExtractor() if expect_transcript else None
        self.service._transcript_emitted = False

    async def _drain(self, *chunks) -> list:
        """Run chunks through the inherited wrapper and the base loop's text push."""
        out = []
        async for chunk in self.service._handle_reasoning_content(_stream(*chunks)):
            out.append(chunk)
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is not None and delta.content:
                await self.service._push_llm_text(delta.content)
        return out

    def _spoken(self) -> str:
        return "".join(f.text for f in self.frames if isinstance(f, LLMTextFrame))

    async def test_transcript_tags_split_user_speech_from_spoken_reply(self) -> None:
        self._begin_turn(expect_transcript=True)
        await self._drain(
            _chunk("<transcript>Where is the "),
            _chunk("Eiffel Tower?</transcript>"),
            _chunk("<response>It is in Paris."),
            _chunk("</response>"),
        )

        transcripts = [f for f in self.frames if isinstance(f, TranscriptionFrame)]
        self.assertEqual([f.text for f in transcripts], ["Where is the Eiffel Tower?"])
        self.assertEqual(self._spoken(), "It is in Paris.")

    async def test_untagged_response_still_reaches_tts(self) -> None:
        self._begin_turn(expect_transcript=True)
        await self._drain(_chunk("It is in Paris."))

        self.assertEqual([f for f in self.frames if isinstance(f, TranscriptionFrame)], [])
        self.assertEqual(self._spoken(), "It is in Paris.")

    async def test_response_held_at_stream_end_is_still_spoken(self) -> None:
        # The last chunk ends mid-response, so the split is holding text back
        # against a possible closing tag when the stream ends.
        self._begin_turn(expect_transcript=True)
        await self._drain(
            _chunk("<transcript>Hi</transcript><response>Hello the"),
            _chunk("re."),
        )

        self.assertEqual(self._spoken(), "Hello there.")

    async def test_reasoning_and_transcript_sections_compose(self) -> None:
        self._begin_turn(expect_transcript=True)
        await self._drain(
            _chunk("<think>They asked for a city.</think>"),
            _chunk("<transcript>Where is it?</transcript>"),
            _chunk("<response>In Paris.</response>"),
        )

        thoughts = "".join(f.text for f in self.frames if isinstance(f, LLMThoughtTextFrame))
        self.assertEqual(thoughts, "They asked for a city.")
        self.assertIsInstance(self.frames[0], LLMThoughtStartFrame)
        self.assertTrue(any(isinstance(f, LLMThoughtEndFrame) for f in self.frames))
        self.assertEqual([f.text for f in self.frames if isinstance(f, TranscriptionFrame)], ["Where is it?"])
        self.assertEqual(self._spoken(), "In Paris.")

    async def test_reasoning_content_field_stays_out_of_the_transcript_split(self) -> None:
        self._begin_turn(expect_transcript=True)
        reasoning_chunk = _chunk(None)
        reasoning_chunk.choices[0].delta.reasoning_content = "Thinking hard."
        await self._drain(reasoning_chunk, _chunk("<transcript>Hi</transcript><response>Hello.</response>"))

        self.assertIsInstance(self.frames[0], LLMThoughtStartFrame)
        self.assertEqual(self.frames[1].text, "Thinking hard.")
        self.assertEqual(self._spoken(), "Hello.")

    async def test_plain_content_is_not_buffered_or_rewritten(self) -> None:
        self._begin_turn(expect_transcript=False)
        out = await self._drain(_chunk("Hello"), _chunk(" there."))

        self.assertEqual([c.choices[0].delta.content for c in out], ["Hello", " there."])
        self.assertEqual(self._spoken(), "Hello there.")

    async def test_tool_call_chunks_pass_through_untouched(self) -> None:
        self._begin_turn(expect_transcript=True)
        tool_call = SimpleNamespace(index=0, id="call_1", function=SimpleNamespace(name="get_weather", arguments=""))
        out = await self._drain(_chunk(None, tool_calls=[tool_call]))

        self.assertEqual(out[0].choices[0].delta.tool_calls, [tool_call])
        self.assertEqual([f for f in self.frames if isinstance(f, TranscriptionFrame)], [])


class OmniTranscriptOwnershipTests(unittest.IsolatedAsyncioTestCase):
    """Who writes a spoken turn into the conversation, and who only reports it."""

    async def asyncSetUp(self) -> None:
        self.service = NvidiaOmniLLMService(api_key="not-needed", base_url="http://localhost:8000/v1")
        self.context = LLMContext([{"role": "system", "content": "You are helpful."}])
        self.service._context = self.context
        self.pushed: list[tuple] = []
        self.service.push_frame = AsyncMock(
            side_effect=lambda frame, direction=None: self.pushed.append((frame, direction))
        )

    async def test_a_spoken_turn_is_reported_for_the_user_aggregator_to_write(self) -> None:
        # The aggregator upstream writes what this frame carries, so writing it
        # here as well would leave the same turn in the conversation twice.
        await self.service._emit_user_transcript("where is the tower?")

        self.assertEqual([m["role"] for m in self.context.get_messages()], ["system"])
        frame, direction = self.pushed[-1]
        self.assertIsInstance(frame, TranscriptionFrame)
        self.assertEqual(frame.text, "where is the tower?")
        self.assertEqual(direction, FrameDirection.UPSTREAM)

    async def test_an_audio_pipeline_is_announced_as_a_realtime_service(self) -> None:
        # Realtime mode is what moves the aggregator's write late enough for a
        # transcript that only exists once the model has answered.
        self.assertTrue(self.service.service_metadata_frame().is_realtime_service)

    async def test_a_text_only_pipeline_is_announced_as_a_plain_service(self) -> None:
        self.service._settings.input_modalities = ("text",)

        self.assertFalse(self.service.service_metadata_frame().is_realtime_service)


class OmniRequestBuildingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.service = NvidiaOmniLLMService(api_key="not-needed", base_url="http://localhost:8000/v1")

    async def test_audio_turn_parts_are_appended_without_mutating_context(self) -> None:
        params_from_context = {"messages": [{"role": "user", "content": "hi"}]}
        self.service._active_turn_parts = [{"type": "text", "text": "listen"}]

        params = self.service.build_chat_completion_params(params_from_context)

        self.assertEqual(len(params["messages"]), 2)
        self.assertEqual(params["messages"][-1]["content"], [{"type": "text", "text": "listen"}])
        self.assertEqual(len(params_from_context["messages"]), 1)

    async def test_text_turns_send_context_messages_unchanged(self) -> None:
        params_from_context = {"messages": [{"role": "user", "content": "hi"}]}

        params = self.service.build_chat_completion_params(params_from_context)

        self.assertEqual(params["messages"], [{"role": "user", "content": "hi"}])

    async def test_a_universal_audio_message_is_named_the_way_the_endpoint_reads_it(self) -> None:
        # Callers build audio the standard way; the adapter renames it for NIM.
        message = {"role": "user", "content": [audio_message_part(b"\x00\x00", 16000, 1)]}
        context = LLMContext([message])

        params = self.service.get_llm_adapter().get_llm_invocation_params(
            context, system_instruction=None, convert_developer_to_user=False
        )

        part = params["messages"][0]["content"][0]
        self.assertEqual(part["type"], "audio_url")
        self.assertTrue(part["audio_url"]["url"].startswith("data:audio/wav;base64,"))
        # The context keeps the universal shape: only the request is rewritten.
        self.assertEqual(message["content"][0]["type"], "input_audio")

    async def test_a_buffered_utterance_is_named_the_same_way(self) -> None:
        self.service._active_turn_parts = [audio_message_part(b"\x00\x00", 16000, 1)]

        params = self.service.build_chat_completion_params({"messages": []})

        self.assertEqual(params["messages"][-1]["content"][0]["type"], "audio_url")

    async def test_only_the_configured_token_limit_reaches_the_endpoint(self) -> None:
        # Both fields at once leaves the endpoint free to honour either.
        service = NvidiaOmniLLMService(
            api_key="not-needed",
            base_url="http://localhost:8000/v1",
            settings=NvidiaOmniSettings(max_tokens=8192),
        )
        sent: dict = {}

        async def fake_create(**kwargs):
            sent.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

        service._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))

        await service.run_inference(LLMContext([{"role": "user", "content": "hi"}]), max_tokens=2048)

        self.assertEqual(sent["max_tokens"], 2048)
        self.assertNotIn("max_completion_tokens", sent)

    async def test_no_token_limit_is_sent_unless_one_is_configured(self) -> None:
        params = self.service.build_chat_completion_params({"messages": []})

        self.assertNotIn("max_tokens", params)
        self.assertNotIn("max_completion_tokens", params)

    async def test_media_modalities_are_rejected_for_pipeline_input(self) -> None:
        with self.assertRaises(ValueError):
            NvidiaOmniLLMService(
                api_key="not-needed",
                base_url="http://localhost:8000/v1",
                settings=NvidiaOmniSettings(input_modalities=("video",)),
            )


if __name__ == "__main__":
    unittest.main()
