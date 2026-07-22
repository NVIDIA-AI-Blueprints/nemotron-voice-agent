# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import asyncio
import os
import unittest
from argparse import Namespace
from unittest.mock import AsyncMock, Mock, call, patch

from pipecat.runner.types import EvalRunnerArguments, RunnerArguments
from pipecat.turns.user_start.transcription_user_turn_start_strategy import TranscriptionUserTurnStartStrategy
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy

from examples.multilingual.pipeline import (
    _build_multilingual_user_aggregator_params,
    _is_eval_transport,
    _prepare_session_language_codes,
)


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
    def test_eval_transport_detects_runner_cli_transport(self) -> None:
        args = RunnerArguments()
        args.cli_args = Namespace(transport="eval")

        self.assertTrue(_is_eval_transport(args))

    def test_eval_transport_detects_eval_runner_arguments(self) -> None:
        args = EvalRunnerArguments(host="localhost", port=7900)

        self.assertTrue(_is_eval_transport(args))

    def test_eval_transport_rejects_non_eval_runner_transport(self) -> None:
        args = RunnerArguments()
        args.cli_args = Namespace(transport="websocket")

        self.assertFalse(_is_eval_transport(args))

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

    def test_eval_transport_skips_language_prewarm(self) -> None:
        to_thread = AsyncMock()
        get_lang_codes = Mock(return_value="en-US")

        with (
            patch("examples.multilingual.pipeline.asyncio.to_thread", to_thread),
            patch("examples.multilingual.pipeline.get_lang_codes", get_lang_codes),
        ):
            lang_codes = self._run_prepare_session_language_codes(EvalRunnerArguments(host="localhost", port=7900))

        self.assertEqual(lang_codes, "")
        to_thread.assert_not_awaited()
        get_lang_codes.assert_not_called()

    def test_non_eval_transport_prewarms_services_and_loads_language_codes(self) -> None:
        to_thread = AsyncMock()
        get_lang_codes = Mock(return_value="en-US,de-DE")
        prewarm_tts = Mock(name="prewarm_tts")
        prewarm_asr = Mock(name="prewarm_asr")

        with (
            patch("examples.multilingual.pipeline.config_store.get", return_value=None),
            patch("examples.multilingual.pipeline.asyncio.to_thread", to_thread),
            patch("examples.multilingual.pipeline.prewarm_tts", prewarm_tts),
            patch("examples.multilingual.pipeline.prewarm_asr", prewarm_asr),
            patch("examples.multilingual.pipeline.get_lang_codes", get_lang_codes),
        ):
            lang_codes = self._run_prepare_session_language_codes(RunnerArguments())

        self.assertEqual(lang_codes, "en-US,de-DE")
        to_thread.assert_has_awaits(
            [
                call(prewarm_asr, "asr.example:443", "parakeet", "fn-123"),
                call(prewarm_tts, "tts.example:443", "Magpie-Multilingual.EN-US.Aria"),
            ]
        )
        get_lang_codes.assert_called_once_with(
            asr_server="asr.example:443",
            asr_model="parakeet",
            asr_function_id="fn-123",
            tts_server="tts.example:443",
            tts_voice_id="Magpie-Multilingual.EN-US.Aria",
        )

    def test_non_eval_transport_skips_tts_prewarm_when_tts_is_configured(self) -> None:
        to_thread = AsyncMock()
        get_lang_codes = Mock(return_value="en-US,de-DE")
        prewarm_tts = Mock(name="prewarm_tts")
        prewarm_asr = Mock(name="prewarm_asr")

        with (
            patch("examples.multilingual.pipeline.config_store.get", return_value=object()),
            patch("examples.multilingual.pipeline.asyncio.to_thread", to_thread),
            patch("examples.multilingual.pipeline.prewarm_tts", prewarm_tts),
            patch("examples.multilingual.pipeline.prewarm_asr", prewarm_asr),
            patch("examples.multilingual.pipeline.get_lang_codes", get_lang_codes),
        ):
            lang_codes = self._run_prepare_session_language_codes(RunnerArguments())

        self.assertEqual(lang_codes, "en-US,de-DE")
        to_thread.assert_awaited_once_with(prewarm_asr, "asr.example:443", "parakeet", "fn-123")
        prewarm_tts.assert_not_called()

    def _run_prepare_session_language_codes(self, runner_args: RunnerArguments) -> str:
        return asyncio.run(
            _prepare_session_language_codes(
                runner_args,
                tts_server="tts.example:443",
                tts_voice="Magpie-Multilingual.EN-US.Aria",
                asr_server="asr.example:443",
                asr_model="parakeet",
                asr_function_id="fn-123",
            )
        )
