# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103

"""Realtime responses finish whether or not the reply is spoken."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMTextFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.utils.text.base_text_aggregator import AggregationType

from realtime.observer import RealtimeLifecycleObserver
from realtime.serializer import RealtimeFrameSerializer

ITEM_EVENTS = {"conversation.item.created", "response.output_item.added", "response.content_part.added"}


def tool_calls_started(call_id: str = "call_1") -> FunctionCallsStartedFrame:
    return FunctionCallsStartedFrame(
        function_calls=[SimpleNamespace(tool_call_id=call_id, function_name="calculate_bmi", arguments={})]
    )


def tool_result(call_id: str = "call_1", **kwargs: Any) -> FunctionCallResultFrame:
    kwargs.setdefault("result", {"bmi": 22.86})
    return FunctionCallResultFrame(function_name="calculate_bmi", tool_call_id=call_id, arguments={}, **kwargs)


class RealtimeResponseLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.new_session()

    def new_session(self) -> None:
        self.emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            self.emitted.append(event)

        self.serializer = RealtimeFrameSerializer()
        self.serializer.set_emit(emit)
        self.serializer.conversation.open_client_text()
        self.observer = RealtimeLifecycleObserver(emit=emit, conversation=self.serializer.conversation)
        self.serializer.set_on_response_cancel(self.observer.on_response_cancelled)

    async def push(self, frame: Frame, *, source: Any = None) -> None:
        await self.observer.on_push_frame(
            FramePushed(
                source=source or MagicMock(),
                destination=MagicMock(),
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )

    async def push_output_stop(self) -> None:
        # The output transport re-pushes TTSStoppedFrame after its audio drains.
        await self.push(TTSStoppedFrame(), source=MagicMock(spec=BaseOutputTransport))

    async def response_create(self) -> Frame | None:
        return await self.serializer.deserialize(json.dumps({"type": "response.create"}))

    def types(self) -> list[str]:
        return [event["type"] for event in self.emitted]

    def done_events(self) -> list[dict[str, Any]]:
        return [event for event in self.emitted if event["type"] == "response.done"]

    async def test_response_create_announces_only_the_response(self) -> None:
        frame = await self.response_create()

        self.assertIsInstance(frame, LLMRunFrame)
        self.assertEqual(self.types(), ["response.created"])

    async def test_reply_without_speech_finishes_when_the_llm_run_ends(self) -> None:
        for text in (None, "", "  \n ", "..."):
            with self.subTest(text=text):
                self.new_session()
                await self.response_create()
                await self.push(LLMFullResponseStartFrame())
                if text is not None:
                    await self.push(LLMTextFrame(text=text))
                await self.push(LLMFullResponseEndFrame())

                done = self.done_events()
                self.assertEqual(len(done), 1)
                self.assertEqual(done[0]["response"]["status"], "completed")
                self.assertEqual(done[0]["response"]["output"], [])
                self.assertFalse(ITEM_EVENTS & set(self.types()))
                self.assertIsInstance(await self.response_create(), LLMRunFrame)

    async def test_spoken_reply_announces_item_on_first_speakable_text_and_waits_for_audio(self) -> None:
        await self.response_create()
        await self.push(LLMFullResponseStartFrame())
        await self.push(LLMTextFrame(text="\n\n"))
        self.assertEqual(self.types(), ["response.created"])

        await self.push(LLMTextFrame(text="Hello there."))
        self.assertEqual(
            self.types(),
            [
                "response.created",
                "conversation.item.created",
                "response.output_item.added",
                "response.content_part.added",
            ],
        )
        await self.push(LLMFullResponseEndFrame())
        self.assertEqual(self.done_events(), [])

        await self.push(TTSTextFrame(text="Hello there.", aggregated_by=AggregationType.SENTENCE))
        await self.push_output_stop()
        done = self.done_events()
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["response"]["status"], "completed")
        self.assertEqual(done[0]["response"]["output"][0]["content"][0]["transcript"], "Hello there.")

    async def test_server_tool_round_stays_in_one_response(self) -> None:
        await self.response_create()
        await self.push(LLMFullResponseStartFrame())
        await self.push(tool_calls_started())
        await self.push(LLMFullResponseEndFrame())
        await self.push(FunctionCallInProgressFrame(function_name="calculate_bmi", tool_call_id="call_1", arguments={}))
        await self.push(tool_result())
        self.assertEqual(self.done_events(), [])

        await self.push(LLMFullResponseStartFrame())
        await self.push(LLMTextFrame(text="Your BMI is 22.9."))
        await self.push(LLMFullResponseEndFrame())
        await self.push(TTSTextFrame(text="Your BMI is 22.9.", aggregated_by=AggregationType.SENTENCE))
        await self.push_output_stop()

        self.assertEqual(self.types().count("response.created"), 1)
        done = self.done_events()
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["response"]["status"], "completed")
        self.assertEqual(done[0]["response"]["output"][0]["content"][0]["transcript"], "Your BMI is 22.9.")

    async def test_tool_result_received_before_llm_run_ends_still_waits_for_follow_up(self) -> None:
        await self.response_create()
        await self.push(LLMFullResponseStartFrame())
        await self.push(tool_calls_started())
        await self.push(tool_result())
        await self.push(LLMFullResponseEndFrame())
        self.assertEqual(self.done_events(), [])

        await self.push(LLMFullResponseStartFrame())
        await self.push(LLMFullResponseEndFrame())
        done = self.done_events()
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["response"]["output"], [])

    async def test_tool_result_without_follow_up_finishes_reply_without_speech(self) -> None:
        cases = {
            "run_llm=False": {"run_llm": False},
            "properties.run_llm=False": {"properties": FunctionCallResultProperties(run_llm=False)},
            "empty result": {"result": None},
        }
        for case, result_kwargs in cases.items():
            with self.subTest(case=case):
                self.new_session()
                await self.response_create()
                await self.push(LLMFullResponseStartFrame())
                await self.push(tool_calls_started())
                await self.push(LLMFullResponseEndFrame())
                self.assertEqual(self.done_events(), [])

                await self.push(tool_result(**result_kwargs))
                done = self.done_events()
                self.assertEqual(len(done), 1)
                self.assertEqual(done[0]["response"]["status"], "completed")
                self.assertEqual(done[0]["response"]["output"], [])

    async def test_llm_error_fails_reply_without_speech(self) -> None:
        await self.response_create()
        await self.push(LLMFullResponseStartFrame())
        error = ErrorFrame(error="Error during completion: upstream returned 500")
        error.processor = MagicMock(spec=LLMService)
        await self.push(error)
        await self.push(LLMFullResponseEndFrame())

        self.assertIn("error", self.types())
        done = self.done_events()
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["response"]["status"], "failed")
        self.assertIsInstance(await self.response_create(), LLMRunFrame)

    async def test_error_from_another_processor_does_not_fail_the_reply(self) -> None:
        await self.response_create()
        await self.push(LLMFullResponseStartFrame())
        await self.push(ErrorFrame(error="ASR stream reconnecting"))
        await self.push(LLMFullResponseEndFrame())

        self.assertEqual(self.done_events()[0]["response"]["status"], "completed")

    async def test_voice_turn_reply_without_speech_opens_no_response(self) -> None:
        await self.push(LLMFullResponseStartFrame())
        await self.push(LLMTextFrame(text="  "))
        await self.push(TTSTextFrame(text="...", aggregated_by=AggregationType.SENTENCE))
        await self.push(LLMFullResponseEndFrame())

        self.assertEqual(self.emitted, [])
        self.assertIsNone(self.serializer.conversation.response_id)

    async def test_interruption_during_tool_round_cancels_and_resets_tracking(self) -> None:
        await self.response_create()
        await self.push(LLMFullResponseStartFrame())
        await self.push(tool_calls_started())
        await self.push(LLMFullResponseEndFrame())
        await self.push(InterruptionFrame())
        self.assertEqual(self.done_events()[0]["response"]["status"], "cancelled")

        self.emitted.clear()
        await self.response_create()
        await self.push(LLMFullResponseStartFrame())
        await self.push(LLMFullResponseEndFrame())
        done = self.done_events()
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["response"]["status"], "completed")

    async def test_response_cancel_before_speech_finishes_without_output(self) -> None:
        await self.response_create()
        frame = await self.serializer.deserialize(json.dumps({"type": "response.cancel"}))

        self.assertIsInstance(frame, InterruptionFrame)
        done = self.done_events()
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["response"]["status"], "cancelled")
        self.assertEqual(done[0]["response"]["output"], [])

    async def test_finished_turn_frames_at_later_edges_open_no_new_response(self) -> None:
        tts_text = TTSTextFrame(text="Hi there.", aggregated_by=AggregationType.SENTENCE)
        await self.push(LLMFullResponseStartFrame())
        await self.push(LLMTextFrame(text="Hi there."))
        await self.push(LLMFullResponseEndFrame())
        await self.push(tts_text)
        await self.push_output_stop()
        self.assertEqual(len(self.done_events()), 1)

        self.emitted.clear()
        # The output transport forwards the same frame to the next processor.
        await self.push(tts_text)
        self.assertEqual(self.emitted, [])
        self.assertIsNone(self.serializer.conversation.response_id)


if __name__ == "__main__":
    unittest.main()
