# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import unittest
from unittest.mock import AsyncMock, Mock

from pipecat.frames.frames import TranscriptionFrame, VADUserStartedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import TurnAnalyzerUserTurnStopStrategy

from examples.shared.force_eou_smart_turn_strategy import ForceEouSmartTurnStopStrategy
from examples.shared.stt_finalize_frame import STTFinalizeFrame


class ForceEouSmartTurnStopStrategyTests(unittest.IsolatedAsyncioTestCase):
    def _strategy(self) -> ForceEouSmartTurnStopStrategy:
        strategy = ForceEouSmartTurnStopStrategy(turn_analyzer=Mock())
        strategy.push_frame = AsyncMock()
        strategy.trigger_user_turn_stopped = AsyncMock()
        return strategy

    async def test_complete_without_finalized_transcript_pushes_finalize_once(self) -> None:
        strategy = self._strategy()
        strategy._turn_complete = True

        await strategy._maybe_trigger_user_turn_stopped()
        await strategy._maybe_trigger_user_turn_stopped()

        strategy.push_frame.assert_awaited_once()
        frame, direction = strategy.push_frame.await_args.args
        self.assertIsInstance(frame, STTFinalizeFrame)
        self.assertEqual(direction, FrameDirection.UPSTREAM)
        strategy.trigger_user_turn_stopped.assert_not_awaited()

    async def test_vad_start_resets_force_eou_so_next_complete_flushes_again(self) -> None:
        strategy = self._strategy()
        strategy._turn_complete = True
        await strategy._maybe_trigger_user_turn_stopped()
        strategy.push_frame.reset_mock()

        await strategy._handle_vad_user_started_speaking(VADUserStartedSpeakingFrame())
        self.assertFalse(strategy._force_eou_requested)

        strategy._turn_complete = True
        await strategy._maybe_trigger_user_turn_stopped()

        strategy.push_frame.assert_awaited_once()
        self.assertIsInstance(strategy.push_frame.await_args.args[0], STTFinalizeFrame)

    async def test_already_finalized_transcript_skips_force_eou_and_stops_turn(self) -> None:
        strategy = self._strategy()
        strategy._turn_complete = True
        strategy._text = "hello"
        strategy._transcript_finalized = True

        await strategy._maybe_trigger_user_turn_stopped()

        strategy.push_frame.assert_not_awaited()
        strategy.trigger_user_turn_stopped.assert_awaited_once()

    async def test_finalized_transcript_after_force_eou_stops_the_turn(self) -> None:
        strategy = self._strategy()
        strategy._turn_complete = True
        strategy._vad_stopped = True
        await strategy._maybe_trigger_user_turn_stopped()
        strategy.trigger_user_turn_stopped.reset_mock()

        await strategy._handle_transcription(TranscriptionFrame(text="hello", user_id="", timestamp="", finalized=True))

        strategy.trigger_user_turn_stopped.assert_awaited_once()

    def test_is_a_child_of_pipecat_smart_turn_stop_strategy(self) -> None:
        self.assertTrue(issubclass(ForceEouSmartTurnStopStrategy, TurnAnalyzerUserTurnStopStrategy))
