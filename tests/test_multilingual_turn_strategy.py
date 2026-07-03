# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import os
import unittest
from unittest.mock import patch

from pipecat.turns.user_start.transcription_user_turn_start_strategy import TranscriptionUserTurnStartStrategy
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy

from examples.multilingual.pipeline import _build_multilingual_user_aggregator_params


class _FakeTurnAnalyzer:
    pass


class _FakeVADAnalyzer:
    pass


def _assert_vad_only_start(testcase: unittest.TestCase, strategies) -> None:
    testcase.assertIsNotNone(strategies)
    assert strategies is not None
    testcase.assertEqual(len(strategies.start), 1)
    testcase.assertIsInstance(strategies.start[0], VADUserTurnStartStrategy)
    testcase.assertFalse(any(isinstance(strategy, TranscriptionUserTurnStartStrategy) for strategy in strategies.start))


class MultilingualTurnStrategyTests(unittest.TestCase):
    def test_default_multilingual_turn_start_is_vad_only(self) -> None:
        with (
            patch.dict(os.environ, {"USE_SILERO_VAD_TURN_DETECTION": "false"}),
            patch("examples.multilingual.pipeline.SileroVADAnalyzer", return_value=_FakeVADAnalyzer()),
            patch(
                "pipecat.audio.turn.smart_turn.local_smart_turn_v3.LocalSmartTurnAnalyzerV3",
                return_value=_FakeTurnAnalyzer(),
            ),
        ):
            params = _build_multilingual_user_aggregator_params()

        self.assertIsInstance(params.vad_analyzer, _FakeVADAnalyzer)
        _assert_vad_only_start(self, params.user_turn_strategies)

    def test_silero_timeout_multilingual_turn_start_is_vad_only(self) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "USE_SILERO_VAD_TURN_DETECTION": "true",
                    "SILERO_VAD_STOP_SECS": "0.5",
                },
            ),
            patch("examples.multilingual.pipeline.SileroVADAnalyzer", return_value=_FakeVADAnalyzer()),
        ):
            params = _build_multilingual_user_aggregator_params()

        self.assertIsInstance(params.vad_analyzer, _FakeVADAnalyzer)
        _assert_vad_only_start(self, params.user_turn_strategies)
