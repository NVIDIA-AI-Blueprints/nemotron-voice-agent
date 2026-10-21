# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102

import asyncio
import copy
import unittest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMServiceMetadataFrame,
    StartFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    TTSTextFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager

from examples.generic.offline.aggregation import (
    ChunkPositionedFrame,
    SampleClock,
    SharedAssistantAggregation,
    SharedUserAggregation,
)
from examples.generic.offline.models import RATE


def params(timeout=0, *, watchdog=5):
    """Build fresh online strategies without loading VAD model weights."""
    return LLMUserAggregatorParams(
        user_turn_strategies=UserTurnStrategies(stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=timeout)]),
        user_turn_stop_timeout=watchdog,
        empty_user_turn=None,
    )


def final(text, *, finalized=True):
    """Create an ordinary final transcript with no ownership timestamps."""
    return TranscriptionFrame(text=text, user_id="user", timestamp="", finalized=finalized)


def interim(text):
    """Create a partial transcript; it must never enter committed text."""
    return InterimTranscriptionFrame(text=text, user_id="user", timestamp="")


def start():
    """Create the configured Silero start confirmation frame."""
    return VADUserStartedSpeakingFrame(start_secs=0.2)


def stop():
    """Create the configured Silero end confirmation frame."""
    return VADUserStoppedSpeakingFrame(stop_secs=0.5)


class CaptureStockAggregator(LLMUserAggregator):
    """Actual online aggregator with its original asyncio timers as an oracle."""

    def __init__(self):
        """Capture the stock context outputs while retaining original policy."""
        super().__init__(LLMContext(), params=params(), enable_direct_mode=True)
        self.contexts = []
        self.interruptions = 0

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMContextFrame):
            self.contexts.append(copy.deepcopy(frame.context.get_messages()))
        elif direction == FrameDirection.DOWNSTREAM and isinstance(frame, InterruptionFrame):
            self.interruptions += 1

    async def settle_zero_timers(self):
        for strategy in self._user_turn_controller.user_turn_strategies.stop:
            for task in (strategy._user_speech_timeout_task, strategy._stt_timeout_task):
                if task and task is not asyncio.current_task():
                    await task


class OfflineAggregationTests(unittest.IsolatedAsyncioTestCase):
    async def make(self, timeout=0, **kwargs):
        adapter = SharedUserAggregation(params=params(timeout, **kwargs))
        self.addAsyncCleanup(adapter.close)
        return adapter

    async def test_multiple_finals_commit_one_turn_without_timestamp_ownership(self):
        adapter = await self.make()
        await adapter.feed(ChunkPositionedFrame(512, start()))
        await adapter.feed(ChunkPositionedFrame(1024, final("First sentence.")))
        await adapter.feed(ChunkPositionedFrame(2048, final("Third statement.")))
        self.assertEqual(adapter.contexts, [])
        await adapter.feed(ChunkPositionedFrame(4096, stop()))
        self.assertEqual(adapter.contexts, [(4096, [{"role": "user", "content": "First sentence. Third statement."}])])

    async def test_quiet_interim_starts_and_interrupts_without_vad(self):
        adapter = await self.make()
        await adapter.feed(ChunkPositionedFrame(0, BotStartedSpeakingFrame()))
        await adapter.feed(ChunkPositionedFrame(512, interim("quiet")))
        self.assertEqual(adapter.contexts, [])
        self.assertEqual([event["sample"] for event in adapter.events if event["type"] == "interruption"], [512])
        await adapter.feed(ChunkPositionedFrame(1024, final("quiet words")))
        self.assertEqual(adapter.contexts[-1], (1024, [{"role": "user", "content": "quiet words"}]))

    async def test_late_final_uses_stock_transcription_fallback_as_new_turn(self):
        adapter = await self.make()
        await adapter.feed(ChunkPositionedFrame(512, start()))
        await adapter.feed(ChunkPositionedFrame(1024, final("First")))
        await adapter.feed(ChunkPositionedFrame(2048, stop()))
        await adapter.feed(ChunkPositionedFrame(3072, final("Third")))
        self.assertEqual(
            adapter.contexts[-1],
            (3072, [{"role": "user", "content": "First"}, {"role": "user", "content": "Third"}]),
        )
        self.assertEqual(len([event for event in adapter.events if event["type"] == "user_turn_started"]), 2)

    async def test_policy_floor_uses_media_samples_and_restarts_after_resumed_speech(self):
        adapter = await self.make(0.6)
        await adapter.feed(ChunkPositionedFrame(512, start()))
        await adapter.feed(ChunkPositionedFrame(1024, final("First")))
        await adapter.feed(ChunkPositionedFrame(2048, stop()))
        await adapter.feed(ChunkPositionedFrame(4096, start()))
        await adapter.feed(ChunkPositionedFrame(6144, final("Second")))
        await adapter.feed(ChunkPositionedFrame(8192, stop()))
        await adapter.advance_to(8192 + 9600 - 1)
        self.assertEqual(adapter.contexts, [])
        await adapter.advance_to(8192 + 9600)
        self.assertEqual(adapter.contexts[-1], (17792, [{"role": "user", "content": "First Second"}]))

    async def test_stt_safety_net_is_short_circuited_by_finalized_event(self):
        adapter = await self.make(0.6)
        await adapter.feed(ChunkPositionedFrame(0, STTMetadataFrame(service_name="test", ttfs_p99_latency=2)))
        await adapter.feed(ChunkPositionedFrame(512, start()))
        await adapter.feed(ChunkPositionedFrame(1024, final("First", finalized=False)))
        await adapter.feed(ChunkPositionedFrame(2048, stop()))
        await adapter.advance_to(12000)
        self.assertEqual(adapter.contexts, [])
        await adapter.feed(ChunkPositionedFrame(13000, final("Second", finalized=True)))
        self.assertEqual(adapter.contexts[-1], (13000, [{"role": "user", "content": "First Second"}]))

    async def test_stt_safety_net_expires_on_processed_audio_progress(self):
        adapter = await self.make(0.6)
        await adapter.feed(ChunkPositionedFrame(0, STTMetadataFrame(service_name="test", ttfs_p99_latency=2)))
        await adapter.feed(ChunkPositionedFrame(512, start()))
        await adapter.feed(ChunkPositionedFrame(1024, final("First", finalized=False)))
        await adapter.feed(ChunkPositionedFrame(2048, stop()))
        await adapter.advance_to(2048 + 24000 - 1)
        self.assertEqual(adapter.contexts, [])
        await adapter.advance_to(2048 + 24000)
        self.assertEqual(adapter.contexts[-1][0], 26048)

    async def test_silence_does_not_create_transcript_or_response(self):
        adapter = await self.make()
        await adapter.advance_to(300 * RATE)
        self.assertEqual(adapter.contexts, [])
        self.assertEqual(adapter.events, [])

    async def test_watchdog_keeps_upstream_empty_turn_behavior(self):
        adapter = await self.make(watchdog=2)
        await adapter.feed(ChunkPositionedFrame(512, interim("uncommitted")))
        await adapter.advance_to(512 + 2 * RATE)
        self.assertEqual(adapter.contexts, [])
        watchdogs = [event for event in adapter.events if event["type"] == "user_turn_stop_timeout"]
        self.assertEqual(watchdogs, [{"type": "user_turn_stop_timeout", "sample": 32512}])

    async def test_eof_flushes_pending_final_once_and_cleanup_cancels_deadlines(self):
        adapter = await self.make(0.6)
        await adapter.feed(ChunkPositionedFrame(512, final("At EOF")))
        await adapter.finish(1024)
        await adapter.finish(1024)
        self.assertEqual(adapter.contexts, [(1024, [{"role": "user", "content": "At EOF"}])])
        self.assertIsNone(adapter.next_deadline_sample)

    async def test_same_chunk_finals_are_delivered_before_zero_timeout(self):
        adapter = await self.make()
        await adapter.feed_batch(
            [ChunkPositionedFrame(512, final("First")), ChunkPositionedFrame(512, final("Second"))]
        )
        self.assertEqual(adapter.contexts, [(512, [{"role": "user", "content": "First Second"}])])

    async def test_assistant_context_and_bot_state_can_be_shared_with_outer_scheduler(self):
        adapter = await self.make()
        await adapter.feed(ChunkPositionedFrame(512, final("Question")))
        adapter.context.add_message({"role": "assistant", "content": "Answer"})
        await adapter.feed(ChunkPositionedFrame(1024, BotStartedSpeakingFrame()))
        await adapter.feed(ChunkPositionedFrame(1536, BotStoppedSpeakingFrame()))
        await adapter.feed(ChunkPositionedFrame(2048, final("Next question")))
        self.assertEqual(
            adapter.contexts[-1][1],
            [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
                {"role": "user", "content": "Next question"},
            ],
        )

    async def test_callbacks_execute_inline_with_sample_positions(self):
        calls = []

        async def context(sample, messages):
            calls.append(("context", sample, messages[-1]["content"]))

        async def event(value):
            calls.append((value["type"], value["sample"]))

        adapter = SharedUserAggregation(params=params(), on_context=context, on_event=event)
        self.addAsyncCleanup(adapter.close)
        await adapter.feed(ChunkPositionedFrame(512, final("Ready")))
        self.assertIn(("context", 512, "Ready"), calls)
        self.assertIn(("interruption", 512), calls)

    async def test_backward_progress_and_mixed_batch_are_rejected(self):
        adapter = await self.make()
        await adapter.advance_to(1000)
        with self.assertRaisesRegex(ValueError, "backwards"):
            await adapter.feed(ChunkPositionedFrame(512, final("Old")))
        with self.assertRaisesRegex(ValueError, "share one"):
            await adapter.feed_batch(
                [ChunkPositionedFrame(1024, final("One")), ChunkPositionedFrame(2048, final("Two"))]
            )

    async def test_idle_timers_are_rejected_instead_of_retaining_wall_clock(self):
        unsupported = params()
        unsupported.user_idle_timeout = 5
        with self.assertRaisesRegex(ValueError, "idle"):
            SharedUserAggregation(params=unsupported)

    async def test_unsupported_stop_policy_is_rejected_instead_of_substituted(self):
        unsupported = params()
        unsupported.user_turn_strategies.stop = [object()]
        with self.assertRaisesRegex(ValueError, "Silero"):
            SharedUserAggregation(params=unsupported)

    async def test_realtime_service_handoff_is_rejected_instead_of_arming_wall_clock_wait(self):
        adapter = await self.make()
        with self.assertRaisesRegex(ValueError, "realtime"):
            await adapter.feed(
                ChunkPositionedFrame(512, LLMServiceMetadataFrame(service_name="realtime", is_realtime_service=True))
            )

    async def test_outer_scheduler_callback_failure_is_not_silently_logged_and_dropped(self):
        async def fail(sample, messages):
            raise RuntimeError("Scheduler rejected request")

        adapter = SharedUserAggregation(params=params(), on_context=fail)
        self.addAsyncCleanup(adapter.close)
        with self.assertRaisesRegex(RuntimeError, "Scheduler rejected request"):
            await adapter.feed(ChunkPositionedFrame(512, final("Ready")))

    async def test_exact_stock_aggregation_for_ordered_trace_with_zero_policy_waits(self):
        stock = CaptureStockAggregator()
        await stock.setup(FrameProcessorSetup(clock=SampleClock(), task_manager=TaskManager(), pipeline_worker=None))
        self.addAsyncCleanup(stock.cleanup)
        await stock.queue_frame(StartFrame())
        adapter = await self.make()
        # Same ordered frames, including an interim, several finals per VAD
        # turn, late final fallback, and a new turn while the bot is speaking.
        batches = [
            [start()],
            [interim("First")],
            [final("First sentence."), final("Third statement.")],
            [stop()],
            [BotStartedSpeakingFrame()],
            [final("Quiet interruption")],
            [BotStoppedSpeakingFrame()],
            [start()],
            [final("Last question")],
            [stop()],
        ]
        for index, batch in enumerate(batches, 1):
            for frame in batch:
                await stock.queue_frame(copy.deepcopy(frame))
            await stock.settle_zero_timers()
            await adapter.feed_batch([ChunkPositionedFrame(index * 512, frame) for frame in batch])
        await stock.queue_frame(EndFrame())
        await adapter.finish((len(batches) + 1) * 512)
        self.assertEqual([messages for _, messages in adapter.contexts], stock.contexts)
        self.assertEqual(
            len([event for event in adapter.events if event["type"] == "interruption"]), stock.interruptions
        )


class OfflineAssistantAggregationTests(unittest.IsolatedAsyncioTestCase):
    async def make(self):
        clock = SampleClock()
        context = LLMContext(messages=[{"role": "user", "content": "Question"}])
        adapter = SharedAssistantAggregation(context, clock)
        self.addAsyncCleanup(adapter.close)
        return adapter, clock, context

    async def deliver(self, adapter, clock, sample, frame):
        await clock.advance_to(sample)
        await adapter.feed(ChunkPositionedFrame(sample, frame))

    async def test_completed_response_commits_at_output_ordered_response_end(self):
        adapter, clock, context = await self.make()
        await self.deliver(adapter, clock, 512, LLMFullResponseStartFrame())
        await self.deliver(adapter, clock, 1024, TTSTextFrame(text="The answer is four.", aggregated_by="sentence"))
        self.assertEqual(context.get_messages(), [{"role": "user", "content": "Question"}])
        await self.deliver(adapter, clock, 4096, LLMFullResponseEndFrame())
        self.assertEqual(context.get_messages()[-1], {"role": "assistant", "content": "The answer is four."})
        self.assertEqual(adapter.contexts[-1][0], 4096)
        self.assertEqual(
            adapter.events[-1],
            {
                "type": "assistant_turn_stopped",
                "sample": 4096,
                "text": "The answer is four.",
                "interrupted": False,
            },
        )

    async def test_interruption_commits_delivered_text_once_without_unreleased_remainder(self):
        adapter, clock, context = await self.make()
        await self.deliver(adapter, clock, 512, LLMFullResponseStartFrame())
        await self.deliver(adapter, clock, 1024, TTSTextFrame(text="The", aggregated_by="word"))
        await self.deliver(adapter, clock, 1536, TTSTextFrame(text="answer", aggregated_by="word"))
        await self.deliver(adapter, clock, 2048, InterruptionFrame())
        await self.deliver(adapter, clock, 2560, LLMFullResponseEndFrame())
        self.assertEqual(context.get_messages()[-1], {"role": "assistant", "content": "The answer"})
        self.assertEqual(len(adapter.contexts), 1)
        self.assertTrue(adapter.events[-1]["interrupted"])

    async def test_interruption_before_any_output_does_not_commit_generated_answer(self):
        adapter, clock, context = await self.make()
        await self.deliver(adapter, clock, 512, LLMFullResponseStartFrame())
        await self.deliver(adapter, clock, 1024, InterruptionFrame())
        self.assertEqual(context.get_messages(), [{"role": "user", "content": "Question"}])
        self.assertEqual(adapter.contexts, [])
        self.assertEqual(adapter.events[-1]["text"], "")

    async def test_assistant_adapter_never_advances_shared_clock(self):
        adapter, clock, _ = await self.make()
        with self.assertRaisesRegex(ValueError, "current sample"):
            await adapter.feed(ChunkPositionedFrame(512, LLMFullResponseStartFrame()))
        self.assertEqual(clock.now_sample, 0)

    async def test_assistant_output_matches_unmodified_stock_assistant(self):
        adapter, clock, context = await self.make()
        stock_context = LLMContext(messages=copy.deepcopy(context.get_messages()))
        stock = LLMAssistantAggregator(stock_context, enable_direct_mode=True)
        await stock.setup(FrameProcessorSetup(clock=SampleClock(), task_manager=TaskManager(), pipeline_worker=None))
        await stock.queue_frame(StartFrame())
        self.addAsyncCleanup(stock.cleanup)
        frames = [
            LLMFullResponseStartFrame(),
            BotStartedSpeakingFrame(),
            TTSTextFrame(text="First", aggregated_by="word"),
            TTSTextFrame(text="answer", aggregated_by="word"),
            InterruptionFrame(),
            BotStoppedSpeakingFrame(),
            LLMFullResponseStartFrame(),
            TTSTextFrame(text="Next answer.", aggregated_by="sentence"),
            LLMFullResponseEndFrame(),
        ]
        for index, frame in enumerate(frames, 1):
            await stock.queue_frame(copy.deepcopy(frame))
            await self.deliver(adapter, clock, index * 512, frame)
        self.assertEqual(context.get_messages(), stock_context.get_messages())


if __name__ == "__main__":
    unittest.main()
