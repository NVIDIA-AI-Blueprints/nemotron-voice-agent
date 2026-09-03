# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import unittest
from unittest.mock import AsyncMock, Mock, call

from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.frames.frames import InputAudioRawFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame

from examples.omni_assistant.audio_only_smart_turn_strategy import AudioOnlySmartTurnStopStrategy


class AudioOnlySmartTurnStopStrategyTests(unittest.IsolatedAsyncioTestCase):
    async def test_external_turn_stop_preserves_live_vad_state_until_vad_stop(self) -> None:
        analyzer = Mock()
        analyzer.append_audio.return_value = EndOfTurnState.INCOMPLETE
        analyzer.analyze_end_of_turn = AsyncMock(return_value=(EndOfTurnState.INCOMPLETE, None))
        strategy = AudioOnlySmartTurnStopStrategy(turn_analyzer=analyzer)

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.handle_user_turn_stopped()
        await strategy.process_frame(InputAudioRawFrame(b"during-boundary", 16000, 1))

        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await strategy.process_frame(InputAudioRawFrame(b"after-vad-stop", 16000, 1))

        analyzer.clear.assert_called_once_with()
        self.assertEqual(
            analyzer.append_audio.call_args_list,
            [
                call(b"during-boundary", True),
                call(b"after-vad-stop", False),
            ],
        )
