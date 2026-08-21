# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103

"""Realtime transcripts, response.done, cancel/truncate, and text items."""

from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any
from unittest.mock import MagicMock

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    TranscriptionFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.utils.text.base_text_aggregator import AggregationType

from realtime.conversation import ConversationState
from realtime.observer import RealtimeLifecycleObserver
from realtime.serializer import RealtimeFrameSerializer


class ConversationStateTests(unittest.TestCase):
    def test_begin_response_is_idempotent_while_in_progress(self) -> None:
        state = ConversationState()
        first, created_first = state.begin_response()
        second, created_second = state.begin_response()
        self.assertTrue(created_first)
        self.assertFalse(created_second)
        self.assertEqual(first, second)
        self.assertEqual(state.response_status, "in_progress")

    def test_complete_response_only_once(self) -> None:
        state = ConversationState()
        state.begin_response()
        snap = state.complete_response("completed")
        self.assertIsNotNone(snap)
        self.assertIsNone(state.complete_response("completed"))

    def test_reset_skips_when_newer_generation_owns_slot(self) -> None:
        state = ConversationState()
        state.begin_response()
        snap = state.complete_response("completed")
        assert snap is not None
        new_id, created = state.begin_response()
        self.assertTrue(created)
        state.reset_response_slot(generation=snap.generation)
        self.assertEqual(state.response_id, new_id)
        self.assertEqual(state.response_status, "in_progress")


class ObserverTranscriptTests(unittest.IsolatedAsyncioTestCase):
    async def test_user_interim_and_final_transcripts(self) -> None:
        from pipecat.frames.frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame

        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        state = ConversationState()
        observer = RealtimeLifecycleObserver(emit=emit, conversation=state)
        source = MagicMock()
        dest = MagicMock()

        await observer.on_push_frame(
            FramePushed(
                source=source,
                destination=dest,
                frame=UserStartedSpeakingFrame(),
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )
        interim = InterimTranscriptionFrame(text="hel", user_id="", timestamp="")
        await observer.on_push_frame(
            FramePushed(
                source=source,
                destination=dest,
                frame=interim,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )
        final = TranscriptionFrame(text="hello", user_id="", timestamp="")
        await observer.on_push_frame(
            FramePushed(
                source=source,
                destination=dest,
                frame=final,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=1,
            )
        )
        self.assertNotIn("conversation.item.created", [e["type"] for e in emitted])
        await observer.on_push_frame(
            FramePushed(
                source=source,
                destination=dest,
                frame=UserStoppedSpeakingFrame(),
                direction=FrameDirection.DOWNSTREAM,
                timestamp=2,
            )
        )

        types = [e["type"] for e in emitted]
        self.assertIn("conversation.item.input_audio_transcription.delta", types)
        self.assertIn("conversation.item.input_audio_transcription.completed", types)
        self.assertIn("conversation.item.created", types)
        completed = next(e for e in emitted if e["type"] == "conversation.item.input_audio_transcription.completed")
        self.assertEqual(completed["transcript"], "hello")

    async def test_late_interim_transcript_is_ignored_outside_active_turn(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        state = ConversationState()
        observer = RealtimeLifecycleObserver(emit=emit, conversation=state)
        interim = InterimTranscriptionFrame(text="late", user_id="", timestamp="")

        await observer._handle_frame(interim)
        state.begin_user_turn()
        state.stop_user_turn()
        await observer._handle_frame(interim)

        self.assertEqual(emitted, [])

    async def test_vad_events_share_item_id_and_sample_clock(self) -> None:
        from pipecat.frames.frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame

        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        state = ConversationState()
        state.add_input_audio(32000, 16000)
        observer = RealtimeLifecycleObserver(emit=emit, conversation=state)
        source = MagicMock()
        dest = MagicMock()

        for frame in (UserStartedSpeakingFrame(),):
            await observer.on_push_frame(
                FramePushed(
                    source=source,
                    destination=dest,
                    frame=frame,
                    direction=FrameDirection.DOWNSTREAM,
                    timestamp=0,
                )
            )
        state.add_input_audio(16000, 16000)
        await observer.on_push_frame(
            FramePushed(
                source=source,
                destination=dest,
                frame=UserStoppedSpeakingFrame(),
                direction=FrameDirection.DOWNSTREAM,
                timestamp=1,
            )
        )

        started = next(e for e in emitted if e["type"] == "input_audio_buffer.speech_started")
        stopped = next(e for e in emitted if e["type"] == "input_audio_buffer.speech_stopped")
        committed = next(e for e in emitted if e["type"] == "input_audio_buffer.committed")
        self.assertEqual(started["item_id"], stopped["item_id"])
        self.assertEqual(stopped["item_id"], committed["item_id"])
        self.assertEqual(started["audio_start_ms"], 1000)
        self.assertEqual(stopped["audio_end_ms"], 1500)

    async def test_final_transcript_before_vad_stop_keeps_same_item_id(self) -> None:
        from pipecat.frames.frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame

        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        state = ConversationState()
        observer = RealtimeLifecycleObserver(emit=emit, conversation=state)
        source = MagicMock()
        destination = MagicMock()
        for frame in (
            UserStartedSpeakingFrame(),
            TranscriptionFrame(text="hello", user_id="", timestamp=""),
            UserStoppedSpeakingFrame(),
        ):
            await observer.on_push_frame(
                FramePushed(
                    source=source,
                    destination=destination,
                    frame=frame,
                    direction=FrameDirection.DOWNSTREAM,
                    timestamp=0,
                )
            )

        started = next(e for e in emitted if e["type"] == "input_audio_buffer.speech_started")
        stopped = next(e for e in emitted if e["type"] == "input_audio_buffer.speech_stopped")
        created = next(e for e in emitted if e["type"] == "conversation.item.created")
        self.assertEqual(started["item_id"], stopped["item_id"])
        self.assertEqual(stopped["item_id"], created["item"]["id"])
        types = [e["type"] for e in emitted]
        self.assertLess(
            types.index("input_audio_buffer.speech_stopped"),
            types.index("input_audio_buffer.committed"),
        )
        self.assertLess(
            types.index("input_audio_buffer.committed"),
            types.index("conversation.item.created"),
        )

    async def test_vad_stop_before_final_transcript_preserves_protocol_order(self) -> None:
        from pipecat.frames.frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame

        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        observer = RealtimeLifecycleObserver(
            emit=emit,
            conversation=ConversationState(),
        )
        for frame in (
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            TranscriptionFrame(text="hello", user_id="", timestamp=""),
        ):
            await observer.on_push_frame(
                FramePushed(
                    source=MagicMock(),
                    destination=MagicMock(),
                    frame=frame,
                    direction=FrameDirection.DOWNSTREAM,
                    timestamp=0,
                )
            )

        types = [e["type"] for e in emitted]
        self.assertLess(
            types.index("input_audio_buffer.speech_stopped"),
            types.index("conversation.item.created"),
        )
        self.assertLess(
            types.index("conversation.item.created"),
            types.index("conversation.item.input_audio_transcription.completed"),
        )

    async def test_interruption_cancels_in_progress_response_upstream(self) -> None:
        """Pipeline barge-in must finish the response when TTSStopped never arrives."""
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        state = ConversationState()
        observer = RealtimeLifecycleObserver(emit=emit, conversation=state, max_frames=4)
        source = MagicMock()
        dest = MagicMock()

        await observer.on_push_frame(
            FramePushed(
                source=source,
                destination=dest,
                frame=BotStartedSpeakingFrame(),
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )
        await observer.on_push_frame(
            FramePushed(
                source=source,
                destination=dest,
                frame=TTSTextFrame(text="Hello", aggregated_by=AggregationType.SENTENCE),
                direction=FrameDirection.DOWNSTREAM,
                timestamp=1,
            )
        )
        self.assertEqual(state.response_status, "in_progress")

        # Irrelevant frames must not pollute dedupe (would evict with max_frames=4).
        from pipecat.frames.frames import OutputAudioRawFrame

        for i in range(20):
            audio = OutputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)
            await observer.on_push_frame(
                FramePushed(
                    source=source,
                    destination=dest,
                    frame=audio,
                    direction=FrameDirection.DOWNSTREAM,
                    timestamp=10 + i,
                )
            )

        await observer.on_push_frame(
            FramePushed(
                source=source,
                destination=dest,
                frame=InterruptionFrame(),
                direction=FrameDirection.UPSTREAM,
                timestamp=100,
            )
        )
        self.assertIsNone(state.response_id)
        done = next(e for e in emitted if e["type"] == "response.done")
        self.assertEqual(done["response"]["status"], "cancelled")

        # Replay of the same interrupt id is ignored; a new interrupt is a no-op when idle.
        emitted.clear()
        await observer.on_push_frame(
            FramePushed(
                source=source,
                destination=dest,
                frame=InterruptionFrame(),
                direction=FrameDirection.UPSTREAM,
                timestamp=101,
            )
        )
        self.assertNotIn("response.done", [e["type"] for e in emitted])

    async def test_nonfatal_pipeline_error_preserves_active_response(self) -> None:
        from pipecat.frames.frames import ErrorFrame

        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        state = ConversationState()
        state.begin_response()
        observer = RealtimeLifecycleObserver(emit=emit, conversation=state)
        await observer.on_push_frame(
            FramePushed(
                source=MagicMock(),
                destination=MagicMock(),
                frame=ErrorFrame(error="TTS unavailable"),
                direction=FrameDirection.UPSTREAM,
                timestamp=0,
            )
        )

        self.assertEqual([event["type"] for event in emitted], ["error"])
        self.assertEqual(state.response_status, "in_progress")

        await observer._handle_frame(TTSStoppedFrame())
        done = next(e for e in emitted if e["type"] == "response.done")
        self.assertEqual(done["response"]["status"], "completed")

    async def test_nonfatal_pipeline_error_releases_unannounced_response_request(self) -> None:
        from pipecat.frames.frames import ErrorFrame

        async def emit(_event: dict[str, Any]) -> None:
            pass

        state = ConversationState()
        self.assertTrue(state.request_response())
        observer = RealtimeLifecycleObserver(emit=emit, conversation=state)

        await observer.on_push_frame(
            FramePushed(
                source=MagicMock(),
                destination=MagicMock(),
                frame=ErrorFrame(error="LLM unavailable"),
                direction=FrameDirection.UPSTREAM,
                timestamp=0,
            )
        )

        self.assertFalse(state.response_requested)
        self.assertTrue(state.request_response())

    async def test_fatal_pipeline_error_finishes_active_response_as_failed(self) -> None:
        from pipecat.frames.frames import FatalErrorFrame

        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        state = ConversationState()
        state.begin_response()
        observer = RealtimeLifecycleObserver(emit=emit, conversation=state)
        await observer.on_push_frame(
            FramePushed(
                source=MagicMock(),
                destination=MagicMock(),
                frame=FatalErrorFrame(error="Pipeline unavailable"),
                direction=FrameDirection.UPSTREAM,
                timestamp=0,
            )
        )

        self.assertEqual(emitted[0]["type"], "error")
        done = next(event for event in emitted if event["type"] == "response.done")
        self.assertEqual(done["response"]["status"], "failed")

    async def test_function_call_emits_nvidia_observation_only(self) -> None:
        from pipecat.frames.frames import FunctionCallInProgressFrame

        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        state = ConversationState()
        observer = RealtimeLifecycleObserver(emit=emit, conversation=state)
        source = MagicMock()
        dest = MagicMock()

        frame = FunctionCallInProgressFrame(
            function_name="set_memory",
            tool_call_id="call_abc",
            arguments={"key": "intro", "value": "hi"},
        )
        await observer.on_push_frame(
            FramePushed(
                source=source,
                destination=dest,
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )

        types = [e["type"] for e in emitted]
        self.assertEqual(types, ["nvidia.tool.started"])
        started = emitted[0]
        self.assertEqual(started["tool_call_id"], "call_abc")
        self.assertEqual(started["name"], "set_memory")
        self.assertEqual(started["arguments"], {"key": "intro", "value": "hi"})
        self.assertIsNone(state.response_id)

    async def test_function_call_result_emits_nvidia_completion(self) -> None:
        from pipecat.frames.frames import FunctionCallResultFrame

        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        observer = RealtimeLifecycleObserver(emit=emit, conversation=ConversationState())
        frame = FunctionCallResultFrame(
            function_name="get_weather",
            tool_call_id="call_abc",
            arguments={"city": "SF"},
            result={"temp": 72},
            run_llm=True,
        )
        await observer.on_push_frame(
            FramePushed(
                source=MagicMock(),
                destination=MagicMock(),
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )
        self.assertEqual(emitted[0]["type"], "nvidia.tool.completed")
        self.assertEqual(emitted[0]["result"], {"temp": 72})

    async def test_bot_transcript_and_response_done(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        state = ConversationState()
        observer = RealtimeLifecycleObserver(emit=emit, conversation=state)
        source = MagicMock()
        dest = MagicMock()

        tts_text = TTSTextFrame(text="Hi there.", aggregated_by=AggregationType.SENTENCE)
        await observer.on_push_frame(
            FramePushed(
                source=source,
                destination=dest,
                frame=tts_text,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )
        stopped = TTSStoppedFrame()
        transport_source = MagicMock(spec=BaseOutputTransport)
        await observer.on_push_frame(
            FramePushed(
                source=source,
                destination=dest,
                frame=stopped,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=1,
            )
        )
        self.assertNotIn("response.done", [e["type"] for e in emitted])
        await observer.on_push_frame(
            FramePushed(
                source=transport_source,
                destination=dest,
                frame=stopped,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=2,
            )
        )
        await asyncio.sleep(0)

        types = [e["type"] for e in emitted]
        self.assertIn("response.created", types)
        self.assertIn("response.output_audio_transcript.delta", types)
        self.assertIn("response.output_audio_transcript.done", types)
        self.assertIn("response.done", types)
        created = next(e for e in emitted if e["type"] == "response.created")
        self.assertEqual(created["response"]["output"], [])
        done = next(e for e in emitted if e["type"] == "response.done")
        self.assertEqual(done["response"]["status"], "completed")
        transcript_deltas = [e for e in emitted if e["type"] == "response.output_audio_transcript.delta"]
        self.assertEqual(len(transcript_deltas), 1)
        self.assertEqual(transcript_deltas[0]["delta"], "Hi there.")

    async def test_magpie_sentence_stops_keep_single_response(self) -> None:
        """Transport BotStopped between sentences must not emit response.done; TTSStopped does."""
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        state = ConversationState()
        observer = RealtimeLifecycleObserver(emit=emit, conversation=state)
        source = MagicMock()
        dest = MagicMock()

        async def push(frame) -> None:
            frame_source = MagicMock(spec=BaseOutputTransport) if isinstance(frame, TTSStoppedFrame) else source
            await observer.on_push_frame(
                FramePushed(
                    source=frame_source,
                    destination=dest,
                    frame=frame,
                    direction=FrameDirection.DOWNSTREAM,
                    timestamp=0,
                )
            )

        await push(TTSTextFrame(text="One.", aggregated_by=AggregationType.SENTENCE))
        await push(BotStoppedSpeakingFrame())
        await push(BotStartedSpeakingFrame())
        await push(TTSTextFrame(text=" Two.", aggregated_by=AggregationType.SENTENCE))
        await push(BotStoppedSpeakingFrame())
        await asyncio.sleep(0.25)
        self.assertNotIn("response.done", [e["type"] for e in emitted])
        await push(TTSStoppedFrame())
        await asyncio.sleep(0)
        types = [e["type"] for e in emitted]
        self.assertEqual(types.count("response.created"), 1)
        self.assertIn("response.done", types)
        done = next(e for e in emitted if e["type"] == "response.done")
        self.assertEqual(done["response"]["output"][0]["content"][0]["transcript"], "One. Two.")

    async def test_llm_text_waits_for_tts_stopped_not_llm_end(self) -> None:
        """Buffered LLM text must not finish without TTSStopped."""
        from pipecat.frames.frames import LLMTextFrame

        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        state = ConversationState()
        observer = RealtimeLifecycleObserver(emit=emit, conversation=state)
        source = MagicMock()
        dest = MagicMock()

        async def push(frame) -> None:
            frame_source = MagicMock(spec=BaseOutputTransport) if isinstance(frame, TTSStoppedFrame) else source
            await observer.on_push_frame(
                FramePushed(
                    source=frame_source,
                    destination=dest,
                    frame=frame,
                    direction=FrameDirection.DOWNSTREAM,
                    timestamp=0,
                )
            )

        await push(LLMTextFrame(text="Hello."))
        await asyncio.sleep(0.05)
        self.assertNotIn("response.done", [e["type"] for e in emitted])
        await push(TTSTextFrame(text="Hello.", aggregated_by=AggregationType.SENTENCE))
        await push(TTSStoppedFrame())
        await asyncio.sleep(0)
        self.assertIn("response.done", [e["type"] for e in emitted])

    async def test_skip_tts_llm_end_completes_no_audio_response(self) -> None:
        from pipecat.frames.frames import LLMFullResponseEndFrame, LLMTextFrame

        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        observer = RealtimeLifecycleObserver(
            emit=emit,
            conversation=ConversationState(),
        )
        source = MagicMock()
        destination = MagicMock()
        await observer.on_push_frame(
            FramePushed(
                source=source,
                destination=destination,
                frame=LLMTextFrame(text="No speech."),
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )
        end = LLMFullResponseEndFrame()
        end.skip_tts = True
        await observer.on_push_frame(
            FramePushed(
                source=source,
                destination=destination,
                frame=end,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=1,
            )
        )
        done = next(e for e in emitted if e["type"] == "response.done")
        self.assertEqual(done["response"]["status"], "completed")
        self.assertEqual(
            done["response"]["output"][0]["content"][0]["transcript"],
            "No speech.",
        )

    async def test_function_call_batch_emits_two_nvidia_started_events(self) -> None:
        from types import SimpleNamespace

        from pipecat.frames.frames import FunctionCallsStartedFrame

        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        state = ConversationState()
        observer = RealtimeLifecycleObserver(emit=emit, conversation=state)
        source = MagicMock()
        dest = MagicMock()
        frame = FunctionCallsStartedFrame(
            function_calls=[
                SimpleNamespace(tool_call_id="c1", function_name="set_memory", arguments={"k": "v"}),
                SimpleNamespace(tool_call_id="c2", function_name="get_weather", arguments={"lat": 1}),
            ]
        )
        await observer.on_push_frame(
            FramePushed(
                source=source,
                destination=dest,
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )
        await asyncio.sleep(0)
        types = [e["type"] for e in emitted]
        self.assertEqual(types, ["nvidia.tool.started", "nvidia.tool.started"])
        self.assertEqual([event["tool_call_id"] for event in emitted], ["c1", "c2"])
        self.assertIsNone(state.response_id)
        observer.shutdown()

    async def test_llm_text_does_not_duplicate_tts_transcript(self) -> None:
        """Cascaded LLMText + TTSText must yield one output_audio_transcript.delta only."""
        from pipecat.frames.frames import LLMTextFrame

        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        state = ConversationState()
        observer = RealtimeLifecycleObserver(emit=emit, conversation=state)
        source = MagicMock()
        dest = MagicMock()

        async def push(frame) -> None:
            frame_source = MagicMock(spec=BaseOutputTransport) if isinstance(frame, TTSStoppedFrame) else source
            await observer.on_push_frame(
                FramePushed(
                    source=frame_source,
                    destination=dest,
                    frame=frame,
                    direction=FrameDirection.DOWNSTREAM,
                    timestamp=0,
                )
            )

        await push(LLMTextFrame(text="Hello there."))
        await push(TTSTextFrame(text="Hello there.", aggregated_by=AggregationType.SENTENCE))
        await push(TTSStoppedFrame())
        await asyncio.sleep(0)

        transcript_deltas = [e for e in emitted if e["type"] == "response.output_audio_transcript.delta"]
        self.assertEqual(len(transcript_deltas), 1)
        self.assertEqual(transcript_deltas[0]["delta"], "Hello there.")
        text_deltas = [e for e in emitted if e["type"] == "response.output_text.delta"]
        self.assertEqual(text_deltas, [])
        self.assertEqual(state.assistant_transcript, "")  # reset after done
        done = next(e for e in emitted if e["type"] == "response.done")
        self.assertEqual(done["response"]["output"][0]["content"][0]["transcript"], "Hello there.")
        self.assertEqual(done["response"]["output"][0]["content"][0]["type"], "output_audio")
        output_done = next(e for e in emitted if e["type"] == "response.output_item.done")
        self.assertEqual(output_done["item"]["content"][0]["type"], "output_audio")
        self.assertEqual(
            [e["type"] for e in emitted].count("response.output_audio_transcript.delta"),
            1,
        )
        self.assertEqual(
            [e["type"] for e in emitted].count("response.audio_transcript.delta"),
            1,
        )


class SerializerControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_response_cancel_emits_done_cancelled(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer()
        ser.set_emit(emit)
        ser.conversation.begin_response()
        ser.conversation.append_assistant_transcript("partial")

        frame = await ser.deserialize(json.dumps({"type": "response.cancel"}))
        self.assertIsInstance(frame, InterruptionFrame)
        types = [e["type"] for e in emitted]
        self.assertIn("response.done", types)
        done = next(e for e in emitted if e["type"] == "response.done")
        self.assertEqual(done["response"]["status"], "cancelled")

    async def test_response_cancel_drains_observer_llm_text_and_blocks_phantom(self) -> None:
        """Cancel must finish with buffered output_text and not revive the turn."""
        from pipecat.frames.frames import LLMTextFrame

        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer()
        ser.set_emit(emit)
        observer = RealtimeLifecycleObserver(
            emit=emit,
            conversation=ser.conversation,
        )
        ser.set_on_response_cancel(observer.on_response_cancelled)

        source = MagicMock()
        dest = MagicMock()

        async def push(frame) -> None:
            await observer.on_push_frame(
                FramePushed(
                    source=source,
                    destination=dest,
                    frame=frame,
                    direction=FrameDirection.DOWNSTREAM,
                    timestamp=0,
                )
            )

        await push(LLMTextFrame(text="Once upon a time"))
        self.assertEqual(observer._llm_text_buffer, "Once upon a time")

        frame = await ser.deserialize(json.dumps({"type": "response.cancel"}))
        self.assertIsInstance(frame, InterruptionFrame)
        self.assertEqual(observer._llm_text_buffer, "")
        self.assertIsNone(ser.conversation.response_id)

        self.assertNotIn("response.output_text.done", [e["type"] for e in emitted])
        transcript_done = next(e for e in emitted if e["type"] == "response.output_audio_transcript.done")
        self.assertEqual(transcript_done["transcript"], "Once upon a time")
        done = next(e for e in emitted if e["type"] == "response.done")
        self.assertEqual(done["response"]["status"], "cancelled")
        cancelled_id = done["response"]["id"]

        emitted.clear()
        # Late TTSStopped must not open a phantom completed response.
        await push(TTSStoppedFrame())
        await asyncio.sleep(0)
        self.assertEqual(emitted, [])
        self.assertIsNone(ser.conversation.response_id)
        self.assertNotEqual(cancelled_id, "")

    async def test_item_truncate_active_interrupts_without_truncated_event(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer()
        ser.set_emit(emit)
        ser.conversation.begin_response()
        item_id = ser.conversation.assistant_item_id
        assert item_id is not None

        frame = await ser.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "content_index": 0,
                    "audio_end_ms": 500,
                }
            )
        )
        self.assertIsInstance(frame, InterruptionFrame)
        types = [e["type"] for e in emitted]
        self.assertNotIn("conversation.item.truncated", types)
        self.assertIn("response.done", types)
        done = next(e for e in emitted if e["type"] == "response.done")
        self.assertEqual(done["response"]["status"], "cancelled")

    async def test_item_truncate_idle_is_noop(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer()
        ser.set_emit(emit)
        ser.conversation.begin_response()
        item_id = ser.conversation.assistant_item_id
        assert item_id is not None
        ser.conversation.item_transcripts[item_id] = "hello"
        ser.conversation.complete_response("completed")
        ser.conversation.reset_response_slot()

        frame = await ser.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "content_index": 0,
                    "audio_end_ms": 500,
                }
            )
        )
        self.assertIsNone(frame)
        self.assertEqual(emitted, [])

    async def test_response_cancel_idle_is_noop(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer()
        ser.set_emit(emit)
        frame = await ser.deserialize(json.dumps({"type": "response.cancel"}))
        self.assertIsNone(frame)
        self.assertEqual(emitted, [])

    async def test_response_create_returns_llm_run(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer()
        ser.set_emit(emit)
        ser.conversation.open_client_text()
        frame = await ser.deserialize(json.dumps({"type": "response.create"}))
        self.assertIsInstance(frame, LLMRunFrame)

    async def test_response_create_without_emitter_does_not_reserve_response(self) -> None:
        ser = RealtimeFrameSerializer()
        ser.conversation.open_client_text()

        frame = await ser.deserialize(json.dumps({"type": "response.create"}))

        self.assertIsNone(frame)
        self.assertFalse(ser.conversation.response_requested)
        self.assertIsNone(ser.conversation.response_status)

    async def test_response_create_empty_response_object_accepted(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer()
        ser.set_emit(emit)
        ser.conversation.open_client_text()
        frame = await ser.deserialize(json.dumps({"type": "response.create", "response": {}}))
        self.assertIsInstance(frame, LLMRunFrame)
        self.assertEqual(emitted[0]["type"], "response.created")

    async def test_response_create_rejects_duplicate_active_response(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer()
        ser.set_emit(emit)
        ser.conversation.open_client_text()
        first = await ser.deserialize(json.dumps({"type": "response.create"}))
        self.assertIsInstance(first, LLMRunFrame)
        frame = await ser.deserialize(json.dumps({"type": "response.create"}))
        self.assertIsNone(frame)
        self.assertEqual(
            emitted[-1]["error"]["code"],
            "conversation_already_has_active_response",
        )

    async def test_response_create_nonempty_override_rejected(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer()
        ser.set_emit(emit)
        ser.conversation.open_client_text()
        frame = await ser.deserialize(json.dumps({"type": "response.create", "response": {"instructions": "override"}}))
        self.assertIsNone(frame)
        self.assertEqual(emitted[0]["error"]["code"], "unsupported_response_override")

    async def test_item_create_text_appends_without_run(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer()
        ser.set_emit(emit)
        # Welcome window closed (intro done or welcome disabled).
        ser.conversation.open_client_text()
        frame = await ser.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Hello bot"}],
                    },
                }
            )
        )
        self.assertIsInstance(frame, LLMMessagesAppendFrame)
        assert isinstance(frame, LLMMessagesAppendFrame)
        self.assertFalse(frame.run_llm)
        self.assertEqual(frame.messages[0]["content"], "Hello bot")
        self.assertEqual(emitted[0]["type"], "conversation.item.created")

    async def test_pre_intro_user_text_and_response_create_rejected(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer()
        ser.set_emit(emit)
        self.assertFalse(ser.conversation.assistant_has_responded)

        dropped = await ser.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Anything at all"}],
                    },
                }
            )
        )
        self.assertIsNone(dropped)
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["type"], "error")
        self.assertEqual(emitted[0]["error"]["code"], "item_rejected_pre_intro")
        emitted.clear()

        # response.create uses the same welcome window and returns an error.
        rejected = await ser.deserialize(json.dumps({"type": "response.create"}))
        self.assertIsNone(rejected)
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["type"], "error")
        self.assertEqual(emitted[0]["error"]["code"], "response_create_rejected_pre_intro")
        emitted.clear()

        # After intro, text + response.create work normally.
        ser.conversation.open_client_text()
        kept = await ser.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Real turn"}],
                    },
                }
            )
        )
        self.assertIsInstance(kept, LLMMessagesAppendFrame)
        run = await ser.deserialize(json.dumps({"type": "response.create"}))
        self.assertIsInstance(run, LLMRunFrame)

    async def test_response_create_after_welcome_not_swallowed_without_prior_reject(self) -> None:
        """Opening the welcome gate must not leave a sticky response.create suppress."""

        async def emit(_event: dict[str, Any]) -> None:
            return None

        ser = RealtimeFrameSerializer()
        ser.set_emit(emit)
        ser.conversation.open_client_text()
        run = await ser.deserialize(json.dumps({"type": "response.create"}))
        self.assertIsInstance(run, LLMRunFrame)

    async def test_finish_does_not_reset_newer_response(self) -> None:
        from realtime.lifecycle import finish_response

        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)
            # Simulate a new response starting while finish awaits.
            if event.get("type") == "response.output_audio.done":
                state.begin_response()

        state = ConversationState()
        state.begin_response()
        state.append_assistant_transcript("hi")
        old_gen = state.response_generation
        await finish_response(state, emit, status="completed")
        self.assertNotEqual(state.response_generation, old_gen)
        self.assertEqual(state.response_status, "in_progress")
        self.assertTrue(any(e.get("type") == "response.done" for e in emitted))

    async def test_pre_intro_audio_append_not_blocked(self) -> None:
        import base64

        from pipecat.frames.frames import InputAudioRawFrame

        from realtime.audio import PIPELINE_PCM_RATE

        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer(
            session_view={
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": PIPELINE_PCM_RATE},
                        "turn_detection": {"type": "server_vad"},
                    }
                }
            }
        )
        ser.set_emit(emit)
        self.assertFalse(ser.conversation.assistant_has_responded)
        pcm = b"\x01\x00" * 40
        frame = await ser.deserialize(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode("ascii"),
                }
            )
        )
        self.assertIsInstance(frame, InputAudioRawFrame)

    async def test_function_call_output_is_rejected(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer()
        ser.set_emit(emit)
        frame = await ser.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": "call_abc",
                        "output": '{"temp": 72}',
                    },
                }
            )
        )
        self.assertIsNone(frame)
        self.assertEqual(emitted[0]["type"], "error")
        self.assertEqual(emitted[0]["error"]["code"], "unsupported_item")

    async def test_function_call_output_rejects_unknown_call_id(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer()
        ser.set_emit(emit)
        frame = await ser.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": "call_missing",
                        "output": "{}",
                    },
                }
            )
        )
        self.assertIsNone(frame)
        self.assertEqual(emitted[0]["type"], "error")
        self.assertEqual(emitted[0]["error"]["code"], "unsupported_item")

    async def test_truncate_rejects_unknown_item(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer()
        ser.set_emit(emit)
        frame = await ser.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.truncate",
                    "item_id": "item_unknown",
                    "content_index": 0,
                    "audio_end_ms": 100,
                }
            )
        )
        self.assertIsNone(frame)
        self.assertEqual(emitted[0]["type"], "error")
        self.assertEqual(emitted[0]["error"]["code"], "invalid_truncate")


class WelcomeGateAlignmentTests(unittest.IsolatedAsyncioTestCase):
    """Realtime text gate follows the shared welcome_enabled toggle."""

    async def test_open_client_text_accepts_first_user_item(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer()
        ser.set_emit(emit)
        self.assertFalse(ser.conversation.assistant_has_responded)
        ser.conversation.open_client_text()

        frame = await ser.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "First turn"}],
                    },
                }
            )
        )
        self.assertIsInstance(frame, LLMMessagesAppendFrame)
        self.assertEqual(emitted[0]["type"], "conversation.item.created")

    def test_register_handlers_opens_gate_when_welcome_disabled(self) -> None:
        from examples.shared.pipeline_utils import register_session_start_handlers
        from realtime.serializer import RealtimeFrameSerializer

        class _Transport:
            def __init__(self) -> None:
                self._realtime_serializer = RealtimeFrameSerializer()
                self.handlers: dict[str, Any] = {}

            def event_handler(self, name: str):
                def decorator(fn):
                    self.handlers[name] = fn
                    return fn

                return decorator

        transport = _Transport()
        self.assertFalse(transport._realtime_serializer.conversation.assistant_has_responded)

        runner_args = MagicMock()
        runner_args.body = {"protocol": "realtime"}
        register_session_start_handlers(
            transport=transport,
            task=MagicMock(),
            context=MagicMock(),
            runner_args=runner_args,
            welcome_enabled=False,
        )
        self.assertTrue(transport._realtime_serializer.conversation.assistant_has_responded)
        self.assertIn("on_client_connected", transport.handlers)

    def test_register_handlers_keeps_gate_when_welcome_enabled(self) -> None:
        from examples.shared.pipeline_utils import register_session_start_handlers
        from realtime.serializer import RealtimeFrameSerializer

        class _Transport:
            def __init__(self) -> None:
                self._realtime_serializer = RealtimeFrameSerializer()

            def event_handler(self, name: str):
                def decorator(fn):
                    return fn

                return decorator

        transport = _Transport()
        runner_args = MagicMock()
        runner_args.body = {"protocol": "realtime"}
        register_session_start_handlers(
            transport=transport,
            task=MagicMock(),
            context=MagicMock(),
            runner_args=runner_args,
            welcome_enabled=True,
        )
        self.assertFalse(transport._realtime_serializer.conversation.assistant_has_responded)


if __name__ == "__main__":
    unittest.main()
