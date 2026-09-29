# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Tests for shared pipeline construction helpers."""

# ruff: noqa: D101, D102

import asyncio
import os
import unittest
from unittest.mock import patch

from pipecat.frames.frames import BotStartedSpeakingFrame, InterimTranscriptionFrame
from pipecat.turns.types import ProcessFrameResult

from examples.shared.pipeline_utils import (
    ObservedMinWordsUserTurnStartStrategy,
    build_user_aggregator_params,
)


class _FakeVADAnalyzer:
    def __init__(self, *, params):
        self.params = params


class UserAggregatorParamsTests(unittest.TestCase):
    def _build(
        self,
        *,
        use_silero: bool,
        vad_stop_secs: float | None = None,
        interruption_min_words: int | None = None,
    ):
        env = {
            "USE_SILERO_VAD_TURN_DETECTION": str(use_silero).lower(),
            "SILERO_VAD_STOP_SECS": "0.9",
        }
        with (
            patch.dict(os.environ, env),
            patch(
                "examples.shared.pipeline_utils.SileroVADAnalyzer",
                side_effect=_FakeVADAnalyzer,
            ),
            patch(
                "examples.shared.pipeline_utils.build_smart_turn_stop_strategies",
                return_value=[],
            ),
        ):
            return build_user_aggregator_params(
                welcome_enabled=False,
                vad_stop_secs=vad_stop_secs,
                interruption_min_words=interruption_min_words,
            )

    def test_existing_callers_keep_default_vad_finalization_delay(self) -> None:
        params = self._build(use_silero=False)

        self.assertEqual(params.vad_analyzer.params.stop_secs, 0.2)

    def test_pipeline_can_request_longer_vad_finalization_delay(self) -> None:
        params = self._build(use_silero=False, vad_stop_secs=0.5)

        self.assertEqual(params.vad_analyzer.params.stop_secs, 0.5)

    def test_explicit_pipeline_delay_also_applies_to_silero_timeout_mode(self) -> None:
        params = self._build(use_silero=True, vad_stop_secs=0.5)

        self.assertEqual(params.vad_analyzer.params.stop_secs, 0.5)

    def test_silero_timeout_mode_keeps_its_environment_default_without_override(self) -> None:
        params = self._build(use_silero=True)

        self.assertEqual(params.vad_analyzer.params.stop_secs, 0.9)

    def test_negative_explicit_delay_is_clamped(self) -> None:
        params = self._build(use_silero=False, vad_stop_secs=-1.0)

        self.assertEqual(params.vad_analyzer.params.stop_secs, 0.0)

    def test_generic_interruption_gate_uses_only_bot_aware_min_words_strategy(self) -> None:
        params = self._build(use_silero=False, interruption_min_words=2)

        strategies = params.user_turn_strategies.start
        self.assertEqual(len(strategies), 1)
        self.assertIsInstance(strategies[0], ObservedMinWordsUserTurnStartStrategy)
        self.assertEqual(strategies[0]._min_words, 2)

    def test_bot_speech_requires_two_words_and_records_only_the_real_trigger(self) -> None:
        triggers: list[tuple[str, int]] = []

        async def record(text: str, word_count: int) -> None:
            triggers.append((text, word_count))

        async def scenario() -> tuple[ProcessFrameResult, ProcessFrameResult]:
            strategy = ObservedMinWordsUserTurnStartStrategy(
                min_words=2,
                on_interruption_trigger=record,
            )
            await strategy.process_frame(BotStartedSpeakingFrame())
            one_word = await strategy.process_frame(InterimTranscriptionFrame("uh", "user", "now"))
            two_words = await strategy.process_frame(InterimTranscriptionFrame("wait please", "user", "now"))
            return one_word, two_words

        one_word, two_words = asyncio.run(scenario())
        self.assertIs(one_word, ProcessFrameResult.CONTINUE)
        self.assertIs(two_words, ProcessFrameResult.STOP)
        self.assertEqual(triggers, [("wait please", 2)])
