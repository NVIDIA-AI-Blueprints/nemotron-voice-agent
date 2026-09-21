# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import unittest
from unittest.mock import AsyncMock, Mock, patch

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy

from examples.omni_assistant.pipeline import _build_user_turn_strategies
from examples.omni_assistant.user_mute_processor import UserMuteProcessor


class OmniTurnStrategyTests(unittest.TestCase):
    def test_audio_turns_do_not_wait_for_an_upstream_transcript(self) -> None:
        analyzer = Mock()

        with patch(
            "examples.shared.pipeline_utils.build_smart_turn_analyzer",
            return_value=analyzer,
        ):
            strategies = _build_user_turn_strategies()

        self.assertEqual(len(strategies.start), 1)
        self.assertIsInstance(strategies.start[0], VADUserTurnStartStrategy)
        self.assertEqual(len(strategies.stop), 1)
        stop_strategy = strategies.stop[0]
        self.assertIsInstance(stop_strategy, TurnAnalyzerUserTurnStopStrategy)
        self.assertIs(stop_strategy._turn_analyzer, analyzer)
        self.assertFalse(stop_strategy.wait_for_transcript)


class UserMuteProcessorSetupTests(unittest.IsolatedAsyncioTestCase):
    async def test_setup_forwards_frame_processor_setup_to_mute_strategies(self) -> None:
        setup = Mock()
        strategy = Mock()
        strategy.setup = AsyncMock()
        processor = UserMuteProcessor(strategies=[strategy])

        with patch.object(FrameProcessor, "setup", new=AsyncMock()) as parent_setup:
            await processor.setup(setup)

        parent_setup.assert_awaited_once_with(setup)
        strategy.setup.assert_awaited_once_with(setup)
