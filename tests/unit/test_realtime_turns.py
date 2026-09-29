# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Tests for Realtime-only ASR input sequencing and terminal ownership."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import EndFrame, InputAudioRawFrame, StartFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import PipelineSource
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator, LLMUserAggregatorParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.turns.user_start.external_user_turn_start_strategy import ExternalUserTurnStartStrategy
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_turn_controller import UserTurnController
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager

from examples.shared.frames import USER_TRANSCRIPT_TURN_FRAME_ID_METADATA
from realtime.frames import (
    RealtimeASRTurnEndedFrame,
    RealtimeASRTurnReleaseFrame,
    RealtimeInputTranscriptionErrorFrame,
    RealtimeManualUserStartedSpeakingFrame,
    RealtimeManualUserStoppedSpeakingFrame,
)
from realtime.turns import (
    RealtimeASRInputSequencer,
    RealtimeLLMContextAggregatorPair,
    RealtimeManualTurnStopStrategy,
    RealtimeSemanticTurnStopStrategy,
    RealtimeServerVADTurnStopStrategy,
)
from realtime.vad import (
    RealtimeVADUserStartedSpeakingFrame,
    RealtimeVADUserStoppedSpeakingFrame,
)


def _owned_transcript(text: str, owner: int) -> TranscriptionFrame:
    frame = TranscriptionFrame(text=text, user_id="", timestamp="", finalized=True)
    frame.metadata[USER_TRANSCRIPT_TURN_FRAME_ID_METADATA] = owner
    return frame


class RealtimeASRInputSequencerTests(unittest.IsolatedAsyncioTestCase):
    """Verify one native turn drains before the next reaches ASR."""

    def test_failed_terminal_requires_owned_public_failure_detail(self) -> None:
        """Make an item-scoped failure self-contained at the typed boundary."""
        with self.assertRaisesRegex(ValueError, "non-empty message"):
            RealtimeASRTurnEndedFrame(1, 2, "failed", code="asr_provider_error")
        with self.assertRaisesRegex(ValueError, "cannot include failure detail"):
            RealtimeASRTurnEndedFrame(1, 2, "completed", code="unexpected", message="unexpected")
        with self.assertRaisesRegex(ValueError, "cannot include failure detail"):
            RealtimeASRTurnEndedFrame(1, 2, "empty", code="unexpected", message="unexpected")

    async def test_terminal_and_release_are_both_required_in_either_order(self) -> None:
        """Never expose turn B until both native and Pipecat barriers arrive."""
        for release_first in (False, True):
            with self.subTest(release_first=release_first):
                sequencer = RealtimeASRInputSequencer()
                start_a = RealtimeVADUserStartedSpeakingFrame(audio_start_sample=0, sample_rate=16_000)
                audio_a = InputAudioRawFrame(audio=b"\x01\x00" * 4, sample_rate=16_000, num_channels=1)
                stop_a = RealtimeVADUserStoppedSpeakingFrame(audio_stop_sample=4, sample_rate=16_000)
                start_b = RealtimeVADUserStartedSpeakingFrame(audio_start_sample=4, sample_rate=16_000)
                audio_b = InputAudioRawFrame(audio=b"\x02\x00" * 2, sample_rate=16_000, num_channels=1)
                stop_b = RealtimeVADUserStoppedSpeakingFrame(audio_stop_sample=6, sample_rate=16_000)
                pushed: list[tuple[object, FrameDirection]] = []

                async def record(frame, direction=FrameDirection.DOWNSTREAM, _pushed=pushed) -> None:
                    _pushed.append((frame, direction))

                terminal = RealtimeASRTurnEndedFrame(start_a.id, stop_a.id, "completed")
                release = RealtimeASRTurnReleaseFrame(start_a.id, stop_a.id)
                first, second = (release, terminal) if release_first else (terminal, release)
                with (
                    patch.object(FrameProcessor, "process_frame", AsyncMock()),
                    patch.object(sequencer, "push_frame", side_effect=record),
                ):
                    await sequencer.process_frame(StartFrame(audio_in_sample_rate=16_000), FrameDirection.DOWNSTREAM)
                    for frame in (start_a, audio_a, stop_a, start_b, audio_b, stop_b):
                        await sequencer.process_frame(frame, FrameDirection.DOWNSTREAM)

                    self.assertEqual([frame for frame, _ in pushed[1:]], [start_a, audio_a, stop_a])
                    await sequencer.process_frame(first, FrameDirection.UPSTREAM)
                    self.assertEqual([frame for frame, _ in pushed[1:]], [start_a, audio_a, stop_a])
                    await sequencer.process_frame(second, FrameDirection.UPSTREAM)

                self.assertEqual(
                    [frame for frame, _ in pushed[1:]],
                    [start_a, audio_a, stop_a, start_b, audio_b, stop_b],
                )
                self.assertTrue(all(direction == FrameDirection.DOWNSTREAM for _, direction in pushed))

    async def test_buffered_replay_cannot_be_overtaken_by_concurrent_input(self) -> None:
        """Keep newly arriving audio behind the complete retained turn replay."""
        sequencer = RealtimeASRInputSequencer()
        start_a = RealtimeVADUserStartedSpeakingFrame(audio_start_sample=0, sample_rate=16_000)
        stop_a = RealtimeVADUserStoppedSpeakingFrame(audio_stop_sample=0, sample_rate=16_000)
        start_b = RealtimeVADUserStartedSpeakingFrame(audio_start_sample=0, sample_rate=16_000)
        audio_b = InputAudioRawFrame(audio=b"\x02\x00", sample_rate=16_000, num_channels=1)
        stop_b = RealtimeVADUserStoppedSpeakingFrame(audio_stop_sample=1, sample_rate=16_000)
        audio_c = InputAudioRawFrame(audio=b"\x03\x00", sample_rate=16_000, num_channels=1)
        replay_started = asyncio.Event()
        continue_replay = asyncio.Event()
        concurrent_input_started = asyncio.Event()
        pushed: list[object] = []

        async def record(frame, _direction=FrameDirection.DOWNSTREAM) -> None:
            pushed.append(frame)
            if frame is start_b:
                replay_started.set()
                await continue_replay.wait()

        async def submit_concurrent_input() -> None:
            concurrent_input_started.set()
            await sequencer.process_frame(audio_c, FrameDirection.DOWNSTREAM)

        with (
            patch.object(FrameProcessor, "process_frame", AsyncMock()),
            patch.object(sequencer, "push_frame", side_effect=record),
        ):
            await sequencer.process_frame(StartFrame(audio_in_sample_rate=16_000), FrameDirection.DOWNSTREAM)
            for frame in (start_a, stop_a, start_b, audio_b, stop_b):
                await sequencer.process_frame(frame, FrameDirection.DOWNSTREAM)
            await sequencer.process_frame(
                RealtimeASRTurnEndedFrame(start_a.id, stop_a.id, "completed"),
                FrameDirection.UPSTREAM,
            )

            release_a = asyncio.create_task(
                sequencer.process_frame(
                    RealtimeASRTurnReleaseFrame(start_a.id, stop_a.id),
                    FrameDirection.UPSTREAM,
                )
            )
            async with asyncio.timeout(1):
                await replay_started.wait()
            concurrent_input = asyncio.create_task(submit_concurrent_input())
            await concurrent_input_started.wait()
            self.assertFalse(concurrent_input.done())
            continue_replay.set()
            await asyncio.gather(release_a, concurrent_input)

            self.assertEqual(pushed[1:], [start_a, stop_a, start_b, audio_b, stop_b])
            await sequencer.process_frame(
                RealtimeASRTurnEndedFrame(start_b.id, stop_b.id, "completed"),
                FrameDirection.UPSTREAM,
            )
            await sequencer.process_frame(
                RealtimeASRTurnReleaseFrame(start_b.id, stop_b.id),
                FrameDirection.UPSTREAM,
            )

        self.assertIs(pushed[-1], audio_c)

    async def test_foreign_or_duplicate_terminal_and_release_fail_closed(self) -> None:
        """Reject every barrier that is not the unique owner barrier."""
        for invalid_kind in ("foreign_terminal", "foreign_release", "duplicate_terminal", "duplicate_release"):
            with self.subTest(invalid_kind=invalid_kind):
                sequencer = RealtimeASRInputSequencer()
                start = RealtimeVADUserStartedSpeakingFrame(audio_start_sample=0, sample_rate=16_000)
                stop = RealtimeVADUserStoppedSpeakingFrame(audio_stop_sample=0, sample_rate=16_000)
                terminal = RealtimeASRTurnEndedFrame(start.id, stop.id, "completed")
                release = RealtimeASRTurnReleaseFrame(start.id, stop.id)
                pushed: list[object] = []

                async def record(frame, _direction=FrameDirection.DOWNSTREAM, _pushed=pushed) -> None:
                    _pushed.append(frame)

                with (
                    patch.object(FrameProcessor, "process_frame", AsyncMock()),
                    patch.object(sequencer, "push_frame", side_effect=record),
                ):
                    await sequencer.process_frame(StartFrame(audio_in_sample_rate=16_000), FrameDirection.DOWNSTREAM)
                    await sequencer.process_frame(start, FrameDirection.DOWNSTREAM)
                    await sequencer.process_frame(stop, FrameDirection.DOWNSTREAM)
                    if invalid_kind == "foreign_terminal":
                        invalid = RealtimeASRTurnEndedFrame(start.id + 1, stop.id, "completed")
                    elif invalid_kind == "foreign_release":
                        invalid = RealtimeASRTurnReleaseFrame(start.id, stop.id + 1)
                    elif invalid_kind == "duplicate_terminal":
                        await sequencer.process_frame(terminal, FrameDirection.UPSTREAM)
                        invalid = terminal
                    else:
                        await sequencer.process_frame(release, FrameDirection.UPSTREAM)
                        invalid = release
                    await sequencer.process_frame(invalid, FrameDirection.UPSTREAM)

                self.assertIsInstance(pushed[-1], RealtimeInputTranscriptionErrorFrame)
                self.assertEqual(pushed[-1].code, "asr_turn_owner_mismatch")

    async def test_item_scoped_failure_releases_next_turn_and_session_stays_usable(self) -> None:
        """A correlated provider failure retires only turn A, not later input."""
        sequencer = RealtimeASRInputSequencer()
        start_a = RealtimeVADUserStartedSpeakingFrame(audio_start_sample=0, sample_rate=16_000)
        stop_a = RealtimeVADUserStoppedSpeakingFrame(audio_stop_sample=0, sample_rate=16_000)
        start_b = RealtimeVADUserStartedSpeakingFrame(audio_start_sample=0, sample_rate=16_000)
        stop_b = RealtimeVADUserStoppedSpeakingFrame(audio_stop_sample=0, sample_rate=16_000)
        start_c = RealtimeVADUserStartedSpeakingFrame(audio_start_sample=0, sample_rate=16_000)
        pushed: list[object] = []

        async def record(frame, _direction=FrameDirection.DOWNSTREAM) -> None:
            pushed.append(frame)

        with (
            patch.object(FrameProcessor, "process_frame", AsyncMock()),
            patch.object(sequencer, "push_frame", side_effect=record),
        ):
            await sequencer.process_frame(StartFrame(audio_in_sample_rate=16_000), FrameDirection.DOWNSTREAM)
            for frame in (start_a, stop_a, start_b, stop_b):
                await sequencer.process_frame(frame, FrameDirection.DOWNSTREAM)
            await sequencer.process_frame(
                RealtimeASRTurnEndedFrame(
                    start_a.id,
                    stop_a.id,
                    "failed",
                    code="asr_provider_error",
                    message="provider rejected turn A",
                ),
                FrameDirection.UPSTREAM,
            )
            await sequencer.process_frame(
                RealtimeASRTurnReleaseFrame(start_a.id, stop_a.id),
                FrameDirection.UPSTREAM,
            )
            self.assertEqual(pushed[1:], [start_a, stop_a, start_b, stop_b])
            await sequencer.process_frame(start_c, FrameDirection.DOWNSTREAM)
            await sequencer.process_frame(
                RealtimeASRTurnReleaseFrame(start_b.id, stop_b.id),
                FrameDirection.UPSTREAM,
            )
            await sequencer.process_frame(
                RealtimeASRTurnEndedFrame(start_b.id, stop_b.id, "completed"),
                FrameDirection.UPSTREAM,
            )

        self.assertIs(pushed[-1], start_c)
        self.assertFalse(any(isinstance(frame, RealtimeInputTranscriptionErrorFrame) for frame in pushed))

    async def test_fatal_or_cancelled_terminal_discards_buffer_and_freezes_input(self) -> None:
        """Do not transfer later audio across a session-ending ASR terminal."""
        terminals = (
            (
                "fatal",
                lambda start, stop: RealtimeASRTurnEndedFrame(
                    start.id,
                    stop.id,
                    "failed",
                    code="asr_turn_owner_mismatch",
                    message="provider ownership was lost",
                    fatal=True,
                ),
            ),
            ("cancelled", lambda start, stop: RealtimeASRTurnEndedFrame(start.id, stop.id, "cancelled")),
        )
        for status, terminal_factory in terminals:
            with self.subTest(status=status):
                sequencer = RealtimeASRInputSequencer()
                start_a = RealtimeVADUserStartedSpeakingFrame(audio_start_sample=0, sample_rate=16_000)
                stop_a = RealtimeVADUserStoppedSpeakingFrame(audio_stop_sample=0, sample_rate=16_000)
                start_b = RealtimeVADUserStartedSpeakingFrame(audio_start_sample=0, sample_rate=16_000)
                pushed: list[object] = []

                async def record(frame, _direction=FrameDirection.DOWNSTREAM, _pushed=pushed) -> None:
                    _pushed.append(frame)

                with (
                    patch.object(FrameProcessor, "process_frame", AsyncMock()),
                    patch.object(sequencer, "push_frame", side_effect=record),
                ):
                    await sequencer.process_frame(StartFrame(audio_in_sample_rate=16_000), FrameDirection.DOWNSTREAM)
                    for frame in (start_a, stop_a, start_b):
                        await sequencer.process_frame(frame, FrameDirection.DOWNSTREAM)
                    await sequencer.process_frame(terminal_factory(start_a, stop_a), FrameDirection.UPSTREAM)
                    await sequencer.process_frame(
                        RealtimeASRTurnReleaseFrame(start_a.id, stop_a.id),
                        FrameDirection.UPSTREAM,
                    )
                    await sequencer.process_frame(
                        RealtimeVADUserStartedSpeakingFrame(audio_start_sample=0, sample_rate=16_000),
                        FrameDirection.DOWNSTREAM,
                    )

                self.assertEqual(pushed[1:], [start_a, stop_a])


class RealtimeAggregatorOrderingTests(unittest.IsolatedAsyncioTestCase):
    """Verify concurrent native writers share one ordered user-state lane."""

    async def test_audio_processing_completes_before_concurrent_typed_stop(self) -> None:
        """Block audio in the base aggregator and prove typed stop cannot enter."""
        user = RealtimeLLMContextAggregatorPair(LLMContext([])).user()
        self.assertTrue(user._enable_direct_mode)
        audio = InputAudioRawFrame(audio=b"\x01\x00", sample_rate=16_000, num_channels=1)
        stop = RealtimeVADUserStoppedSpeakingFrame(audio_stop_sample=1, sample_rate=16_000)
        audio_entered = asyncio.Event()
        release_audio = asyncio.Event()
        stop_submitted = asyncio.Event()
        processed: list[object] = []

        async def controlled_base(_self, frame, _direction) -> None:
            if frame is audio:
                audio_entered.set()
                await release_audio.wait()
            processed.append(frame)

        async def submit_stop() -> None:
            stop_submitted.set()
            await user.queue_frame(stop)

        with patch.object(LLMUserAggregator, "process_frame", new=controlled_base):
            audio_task = asyncio.create_task(user.queue_frame(audio))
            await audio_entered.wait()
            stop_task = asyncio.create_task(submit_stop())
            await stop_submitted.wait()
            self.assertEqual(processed, [])
            release_audio.set()
            await asyncio.gather(audio_task, stop_task)

        self.assertEqual(processed, [audio, stop])

    async def test_strategy_release_reenters_ordered_aggregator_without_deadlock(self) -> None:
        """Allow the real stop-strategy callback path to publish its release."""
        strategy = RealtimeServerVADTurnStopStrategy()
        user_params = LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[strategy],
            )
        )
        user = RealtimeLLMContextAggregatorPair(LLMContext([]), user_params=user_params).user()
        upstream: list[object] = []

        async def record_upstream(frame, _direction) -> None:
            upstream.append(frame)

        source = PipelineSource(record_upstream, name="realtime-turn-test-source")
        source.link(user)
        task_manager = TaskManager()
        setup = FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=task_manager,
            pipeline_worker=MagicMock(),
        )
        await source.setup(setup)
        await user.setup(setup)
        start = RealtimeVADUserStartedSpeakingFrame(audio_start_sample=0, sample_rate=16_000)
        stop = RealtimeVADUserStoppedSpeakingFrame(audio_stop_sample=0, sample_rate=16_000)

        try:
            await source.queue_frame(StartFrame(audio_in_sample_rate=16_000))
            for frame in (start, stop, _owned_transcript("ready", start.id)):
                await source.queue_frame(frame)
            async with asyncio.timeout(1):
                await source.queue_frame(RealtimeASRTurnEndedFrame(start.id, stop.id, "completed"))
            release = next(frame for frame in upstream if isinstance(frame, RealtimeASRTurnReleaseFrame))
            self.assertEqual(
                (release.turn_start_frame_id, release.turn_stop_frame_id),
                (start.id, stop.id),
            )
            await source.queue_frame(EndFrame())
        finally:
            await user.cleanup()
            await source.cleanup()
            for task in tuple(task_manager.current_tasks()):
                await task_manager.cancel_task(task)


class RealtimeTurnStopStrategyTests(unittest.IsolatedAsyncioTestCase):
    """Verify ASR producer terminals, rather than final-result bits, stop turns."""

    async def test_server_vad_requires_matching_completed_terminal(self) -> None:
        """Stop inference and release only the matching completed owner."""
        strategy = RealtimeServerVADTurnStopStrategy()
        strategy.trigger_user_turn_stopped = AsyncMock()
        strategy.trigger_user_turn_finalized = AsyncMock()
        strategy.push_frame = AsyncMock()
        start = RealtimeVADUserStartedSpeakingFrame(audio_start_sample=0, sample_rate=16_000)
        stop = RealtimeVADUserStoppedSpeakingFrame(audio_stop_sample=0, sample_rate=16_000)

        for frame in (start, stop, _owned_transcript("ready", start.id)):
            await strategy.process_frame(frame)
        strategy.trigger_user_turn_stopped.assert_not_awaited()

        await strategy.process_frame(RealtimeASRTurnEndedFrame(start.id + 1, stop.id, "completed"))
        strategy.trigger_user_turn_stopped.assert_not_awaited()
        strategy.push_frame.assert_not_awaited()

        await strategy.process_frame(RealtimeASRTurnEndedFrame(start.id, stop.id, "completed"))
        strategy.trigger_user_turn_stopped.assert_awaited_once()
        strategy.trigger_user_turn_finalized.assert_not_awaited()
        strategy.push_frame.assert_awaited_once()
        release, direction = strategy.push_frame.await_args.args
        self.assertEqual((release.turn_start_frame_id, release.turn_stop_frame_id), (start.id, stop.id))
        self.assertEqual(direction, FrameDirection.UPSTREAM)

    async def test_server_vad_failure_finalizes_without_inference_and_releases(self) -> None:
        """Apply an item-scoped failure before acknowledging its native turn."""
        strategy = RealtimeServerVADTurnStopStrategy()
        strategy.trigger_user_turn_stopped = AsyncMock()
        strategy.trigger_user_turn_finalized = AsyncMock()
        strategy.push_frame = AsyncMock()
        start = RealtimeVADUserStartedSpeakingFrame(audio_start_sample=0, sample_rate=16_000)
        stop = RealtimeVADUserStoppedSpeakingFrame(audio_stop_sample=0, sample_rate=16_000)
        for frame in (start, stop):
            await strategy.process_frame(frame)

        await strategy.process_frame(
            RealtimeASRTurnEndedFrame(
                start.id,
                stop.id,
                "failed",
                code="asr_provider_error",
                message="provider failed",
            )
        )

        strategy.trigger_user_turn_stopped.assert_not_awaited()
        strategy.trigger_user_turn_finalized.assert_awaited_once()
        self.assertIsInstance(strategy.push_frame.await_args.args[0], RealtimeASRTurnReleaseFrame)

    async def test_semantic_vad_requires_matching_completed_terminal(self) -> None:
        """Gate a complete semantic verdict on its segment's native terminal."""
        strategy = RealtimeSemanticTurnStopStrategy(turn_analyzer=MagicMock())
        strategy.trigger_user_turn_stopped = AsyncMock()
        strategy.trigger_user_turn_finalized = AsyncMock()
        strategy.push_frame = AsyncMock()
        start = RealtimeVADUserStartedSpeakingFrame(audio_start_sample=0, sample_rate=16_000)
        stop = RealtimeVADUserStoppedSpeakingFrame(audio_stop_sample=0, sample_rate=16_000)
        strategy._segment_start_frame_id = start.id
        strategy._segment_stop_frame_id = stop.id
        strategy._turn_complete = True
        strategy._vad_stopped = True

        await strategy.process_frame(_owned_transcript("ready", start.id))
        await strategy.process_frame(RealtimeASRTurnEndedFrame(start.id, stop.id + 1, "completed"))
        strategy.trigger_user_turn_stopped.assert_not_awaited()
        strategy.push_frame.assert_not_awaited()

        await strategy.process_frame(RealtimeASRTurnEndedFrame(start.id, stop.id, "completed"))
        strategy.trigger_user_turn_stopped.assert_awaited_once()
        self.assertIsInstance(strategy.push_frame.await_args.args[0], RealtimeASRTurnReleaseFrame)

    async def test_semantic_vad_empty_incomplete_segment_releases_without_finalizing(self) -> None:
        """Release a clean empty segment, then complete the same logical turn."""
        strategy = RealtimeSemanticTurnStopStrategy(turn_analyzer=MagicMock())
        strategy.trigger_user_turn_stopped = AsyncMock()
        strategy.trigger_user_turn_finalized = AsyncMock()
        strategy.push_frame = AsyncMock()
        start = RealtimeVADUserStartedSpeakingFrame(audio_start_sample=0, sample_rate=16_000)
        stop = RealtimeVADUserStoppedSpeakingFrame(audio_stop_sample=1_600, sample_rate=16_000)
        strategy._segment_start_frame_id = start.id
        strategy._segment_stop_frame_id = stop.id
        strategy._turn_complete = False

        await strategy.process_frame(RealtimeASRTurnEndedFrame(start.id, stop.id, "empty"))

        strategy.trigger_user_turn_stopped.assert_not_awaited()
        strategy.trigger_user_turn_finalized.assert_not_awaited()
        release, direction = strategy.push_frame.await_args.args
        self.assertEqual((release.turn_start_frame_id, release.turn_stop_frame_id), (start.id, stop.id))
        self.assertEqual(direction, FrameDirection.UPSTREAM)

        next_start = RealtimeVADUserStartedSpeakingFrame(audio_start_sample=1_600, sample_rate=16_000)
        next_stop = RealtimeVADUserStoppedSpeakingFrame(audio_stop_sample=3_200, sample_rate=16_000)
        await strategy.process_frame(next_start)
        strategy._segment_stop_frame_id = next_stop.id
        strategy._turn_complete = True
        await strategy.process_frame(_owned_transcript("JFK", next_start.id))
        await strategy.process_frame(RealtimeASRTurnEndedFrame(next_start.id, next_stop.id, "completed"))

        strategy.trigger_user_turn_stopped.assert_awaited_once()
        strategy.trigger_user_turn_finalized.assert_not_awaited()
        releases = [call.args[0] for call in strategy.push_frame.await_args_list]
        self.assertEqual(
            [(release.turn_start_frame_id, release.turn_stop_frame_id) for release in releases],
            [(start.id, stop.id), (next_start.id, next_stop.id)],
        )

    async def test_semantic_vad_trailing_empty_segment_uses_accumulated_text(self) -> None:
        """Run inference when Smart Turn completes after an empty trailing segment."""
        strategy = RealtimeSemanticTurnStopStrategy(turn_analyzer=MagicMock())
        strategy.trigger_user_turn_stopped = AsyncMock()
        strategy.trigger_user_turn_finalized = AsyncMock()
        strategy.push_frame = AsyncMock()
        start = RealtimeVADUserStartedSpeakingFrame(audio_start_sample=1_600, sample_rate=16_000)
        stop = RealtimeVADUserStoppedSpeakingFrame(audio_stop_sample=3_200, sample_rate=16_000)
        strategy._segment_start_frame_id = start.id
        strategy._segment_stop_frame_id = stop.id
        strategy._turn_complete = True
        strategy._text = "earlier recognized text"

        await strategy.process_frame(RealtimeASRTurnEndedFrame(start.id, stop.id, "empty"))

        strategy.trigger_user_turn_stopped.assert_awaited_once()
        strategy.trigger_user_turn_finalized.assert_not_awaited()

    async def test_semantic_vad_terminal_finalizes_true_failure_or_late_empty_completion(self) -> None:
        """Finalize true errors immediately and empty turns once Smart Turn completes."""
        for status in ("failed", "empty"):
            with self.subTest(status=status):
                strategy = RealtimeSemanticTurnStopStrategy(turn_analyzer=MagicMock())
                strategy.trigger_user_turn_stopped = AsyncMock()
                strategy.trigger_user_turn_finalized = AsyncMock()
                strategy.push_frame = AsyncMock()
                start = RealtimeVADUserStartedSpeakingFrame(audio_start_sample=0, sample_rate=16_000)
                stop = RealtimeVADUserStoppedSpeakingFrame(audio_stop_sample=1_600, sample_rate=16_000)
                strategy._segment_start_frame_id = start.id
                strategy._segment_stop_frame_id = stop.id
                strategy._turn_complete = False
                terminal = (
                    RealtimeASRTurnEndedFrame(
                        start.id,
                        stop.id,
                        "failed",
                        code="asr_provider_error",
                        message="provider failed",
                    )
                    if status == "failed"
                    else RealtimeASRTurnEndedFrame(start.id, stop.id, "empty")
                )

                await strategy.process_frame(terminal)
                if status == "empty":
                    strategy.trigger_user_turn_finalized.assert_not_awaited()
                    strategy._turn_complete = True
                    await strategy._maybe_trigger_user_turn_stopped()

                strategy.trigger_user_turn_stopped.assert_not_awaited()
                strategy.trigger_user_turn_finalized.assert_awaited_once()
                self.assertIsInstance(strategy.push_frame.await_args.args[0], RealtimeASRTurnReleaseFrame)

    async def test_manual_commit_requires_its_matching_terminal(self) -> None:
        """Release a manual commit only for its immutable start/stop owner."""
        gate = MagicMock()
        strategy = RealtimeManualTurnStopStrategy(response_gate=gate)
        strategy.trigger_user_turn_stopped = AsyncMock()
        strategy.trigger_user_turn_finalized = AsyncMock()
        strategy.push_frame = AsyncMock()
        start = RealtimeManualUserStartedSpeakingFrame(audio_sample_cursor=0, sample_rate=16_000)
        stop = RealtimeManualUserStoppedSpeakingFrame(audio_sample_cursor=0, sample_rate=16_000)

        for frame in (start, stop, _owned_transcript("ready", start.id)):
            await strategy.process_frame(frame)
        await strategy.process_frame(RealtimeASRTurnEndedFrame(start.id, stop.id + 1, "completed"))
        strategy.trigger_user_turn_stopped.assert_not_awaited()

        await strategy.process_frame(RealtimeASRTurnEndedFrame(start.id, stop.id, "completed"))
        strategy.trigger_user_turn_stopped.assert_awaited_once()
        gate.fail_commit.assert_not_called()
        self.assertIsInstance(strategy.push_frame.await_args.args[0], RealtimeASRTurnReleaseFrame)

    async def test_manual_owner_survives_pipecat_reset_until_callback(self) -> None:
        """Keep the exact stop token through Pipecat's reset-before-callback order."""
        strategy = RealtimeManualTurnStopStrategy(response_gate=MagicMock())
        self.assertIsNone(strategy.turn_frame_id)
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=[ExternalUserTurnStartStrategy()],
                stop=[strategy],
            )
        )
        observed: list[int | None] = []

        @controller.event_handler("on_user_turn_stopped")
        async def on_stopped(_controller, stopped_by, _params) -> None:
            observed.append(stopped_by.turn_frame_id)

        task_manager = TaskManager()
        await controller.setup(task_manager)
        try:
            start = RealtimeManualUserStartedSpeakingFrame(audio_sample_cursor=0, sample_rate=16_000)
            stop = RealtimeManualUserStoppedSpeakingFrame(audio_sample_cursor=1, sample_rate=16_000)
            for frame in (
                start,
                stop,
                _owned_transcript("ready", start.id),
                RealtimeASRTurnEndedFrame(start.id, stop.id, "completed"),
            ):
                await controller.process_frame(frame)

            self.assertEqual(observed, [stop.id])
            await controller.process_frame(
                RealtimeManualUserStartedSpeakingFrame(audio_sample_cursor=1, sample_rate=16_000)
            )
            self.assertIsNone(strategy.turn_frame_id)
        finally:
            await controller.cleanup()

    async def test_manual_failure_fails_commit_without_inference_and_releases(self) -> None:
        """Unblock a failed manual commit without starting a model response."""
        gate = MagicMock()
        strategy = RealtimeManualTurnStopStrategy(response_gate=gate)
        strategy.trigger_user_turn_stopped = AsyncMock()
        strategy.trigger_user_turn_finalized = AsyncMock()
        strategy.push_frame = AsyncMock()
        start = RealtimeManualUserStartedSpeakingFrame(audio_sample_cursor=0, sample_rate=16_000)
        stop = RealtimeManualUserStoppedSpeakingFrame(audio_sample_cursor=0, sample_rate=16_000)
        for frame in (start, stop):
            await strategy.process_frame(frame)

        await strategy.process_frame(
            RealtimeASRTurnEndedFrame(
                start.id,
                stop.id,
                "failed",
                code="asr_provider_error",
                message="provider failed",
            )
        )

        gate.fail_commit.assert_called_once_with()
        strategy.trigger_user_turn_stopped.assert_not_awaited()
        strategy.trigger_user_turn_finalized.assert_awaited_once()
        self.assertIsInstance(strategy.push_frame.await_args.args[0], RealtimeASRTurnReleaseFrame)
