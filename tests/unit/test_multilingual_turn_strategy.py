# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102

import asyncio
import os
import unittest
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

from pipecat.runner.types import RunnerArguments, WebSocketRunnerArguments
from pipecat.turns.user_start.transcription_user_turn_start_strategy import TranscriptionUserTurnStartStrategy
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import TurnAnalyzerUserTurnStopStrategy

from examples.multilingual import pipeline as multilingual_pipeline
from examples.multilingual.pipeline import (
    _build_multilingual_user_aggregator_params,
    _is_eval_transport,
    _prepare_session_language_codes,
    _resolve_llm_supported_languages,
)
from examples.shared.pipeline_utils import SMART_TURN_FALLBACK_SECS


class _FakeTurnAnalyzer:
    def __init__(self, *args, **kwargs):
        self.params = kwargs.get("params")


class _FakeVADAnalyzer:
    pass


def _assert_vad_only_start(testcase: unittest.TestCase, strategies) -> None:
    testcase.assertIsNotNone(strategies)
    assert strategies is not None
    testcase.assertEqual(len(strategies.start), 1)
    testcase.assertIsInstance(strategies.start[0], VADUserTurnStartStrategy)
    testcase.assertFalse(any(isinstance(strategy, TranscriptionUserTurnStartStrategy) for strategy in strategies.start))


class MultilingualTurnStrategyTests(unittest.TestCase):
    def test_unknown_builtin_llm_id_is_rejected(self) -> None:
        with (
            patch("examples.multilingual.pipeline.load_service_entry_by_id", return_value={}),
            self.assertRaisesRegex(ValueError, "Unknown built-in LLM selection: cloud-nim:missing"),
        ):
            _resolve_llm_supported_languages(
                {"llm_id": "cloud-nim:missing", "model_id": "untrusted-model"},
                {"supported_languages": ["en"]},
            )

    def test_llm_capabilities_preserve_default_and_custom_behavior(self) -> None:
        default_llm = {"supported_languages": ["en", "de"]}

        self.assertEqual(_resolve_llm_supported_languages({}, default_llm), ["en", "de"])
        self.assertIsNone(_resolve_llm_supported_languages({"llm_id": "custom-user-llm"}, default_llm))
        self.assertIsNone(_resolve_llm_supported_languages({"model_id": "user/model"}, default_llm))

        with patch(
            "examples.multilingual.pipeline.load_service_entry_by_id",
            return_value={"supported_languages": ["fr"]},
        ) as load_entry:
            self.assertEqual(
                _resolve_llm_supported_languages({"llm_id": "cloud-nim:nemotron"}, default_llm),
                ["fr"],
            )
        load_entry.assert_called_once_with("llm", "cloud-nim:nemotron")

    def test_eval_transport_detects_runner_cli_transport(self) -> None:
        args = RunnerArguments()
        args.cli_args = Namespace(transport="eval")

        self.assertTrue(_is_eval_transport(args))

    def test_eval_transport_rejects_non_eval_runner_transport(self) -> None:
        args = RunnerArguments()
        args.cli_args = Namespace(transport="websocket")

        self.assertFalse(_is_eval_transport(args))

    def test_eval_transport_detects_websocket_session_transport(self) -> None:
        args = WebSocketRunnerArguments(websocket=Mock())
        args.transport_type = "eval"

        self.assertTrue(_is_eval_transport(args))

    def test_default_multilingual_turn_start_is_vad_only(self) -> None:
        with (
            patch.dict(os.environ, {"USE_SILERO_VAD_TURN_DETECTION": "false"}),
            patch("examples.multilingual.pipeline.SileroVADAnalyzer", return_value=_FakeVADAnalyzer()),
            patch(
                "examples.shared.pipeline_utils.LocalSmartTurnAnalyzerV3",
                side_effect=_FakeTurnAnalyzer,
            ),
        ):
            params = _build_multilingual_user_aggregator_params(welcome_enabled=True)

        self.assertIsInstance(params.vad_analyzer, _FakeVADAnalyzer)
        _assert_vad_only_start(self, params.user_turn_strategies)
        assert params.user_turn_strategies is not None
        self.assertEqual(len(params.user_turn_strategies.stop), 1)
        self.assertIsInstance(params.user_turn_strategies.stop[0], TurnAnalyzerUserTurnStopStrategy)
        analyzer = params.user_turn_strategies.stop[0]._turn_analyzer
        self.assertIsInstance(analyzer, _FakeTurnAnalyzer)
        self.assertEqual(analyzer.params.stop_secs, SMART_TURN_FALLBACK_SECS)
        # With an introduction the bot speaks first, so the user is muted until then.
        self.assertEqual(len(params.user_mute_strategies), 1)

    def test_disabled_welcome_message_drops_user_mute_strategy(self) -> None:
        with (
            patch.dict(os.environ, {"USE_SILERO_VAD_TURN_DETECTION": "false"}),
            patch("examples.multilingual.pipeline.SileroVADAnalyzer", return_value=_FakeVADAnalyzer()),
            patch(
                "examples.shared.pipeline_utils.LocalSmartTurnAnalyzerV3",
                side_effect=_FakeTurnAnalyzer,
            ),
        ):
            params = _build_multilingual_user_aggregator_params(welcome_enabled=False)

        # No welcome message means no first bot turn, so muting would deadlock.
        self.assertEqual(params.user_mute_strategies, [])

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
            params = _build_multilingual_user_aggregator_params(welcome_enabled=True)

        self.assertIsInstance(params.vad_analyzer, _FakeVADAnalyzer)
        _assert_vad_only_start(self, params.user_turn_strategies)

    def test_vad_only_forces_timeout_without_smart_turn(self) -> None:
        for stop_secs in ("", "0.8"):
            with (
                self.subTest(stop_secs=stop_secs),
                patch.dict(os.environ, {"USE_SILERO_VAD_TURN_DETECTION": "false", "SILERO_VAD_STOP_SECS": stop_secs}),
                patch("examples.multilingual.pipeline.build_smart_turn_stop_strategies") as smart,
            ):
                params = _build_multilingual_user_aggregator_params(False, vad_only=True)
            smart.assert_not_called()
            self.assertEqual(params.vad_analyzer.params.stop_secs, float(stop_secs or "0.5"))
            _assert_vad_only_start(self, params.user_turn_strategies)
            self.assertIsInstance(params.user_turn_strategies.stop[0], SpeechTimeoutUserTurnStopStrategy)
            self.assertEqual(params.user_turn_strategies.stop[0]._user_speech_timeout, 0.0)
            self.assertEqual(params.user_mute_strategies, [])

    def test_transcribe_prewarm_skips_grpc_and_forwards_static_languages(self) -> None:
        with (
            patch("examples.multilingual.pipeline.prewarm_asr") as asr,
            patch("examples.multilingual.pipeline.prewarm_tts") as tts,
            patch("examples.multilingual.pipeline.get_lang_codes", return_value="de-DE") as codes,
        ):
            result = asyncio.run(
                _prepare_session_language_codes(
                    RunnerArguments(),
                    tts_server="tts:50051",
                    tts_voice="Voice",
                    tts_function_id="",
                    tts_model="",
                    asr_server="http://transcribe/v1",
                    asr_model="transcribe",
                    asr_function_id="",
                    asr_supported_languages=["de", "hi"],
                    llm_supported_languages=["de"],
                )
            )
        self.assertEqual(result, "de-DE")
        asr.assert_not_called()
        tts.assert_called_once()
        self.assertEqual(codes.call_args.kwargs["asr_supported_languages"], ["de", "hi"])
        self.assertEqual(codes.call_args.kwargs["llm_supported_languages"], ["de"])

    def test_eval_transport_skips_language_prewarm(self) -> None:
        to_thread = AsyncMock()
        get_lang_codes = Mock(return_value="en-US")

        with (
            patch("examples.multilingual.pipeline.asyncio.to_thread", to_thread),
            patch("examples.multilingual.pipeline.get_lang_codes", get_lang_codes),
        ):
            args = RunnerArguments()
            args.cli_args = Namespace(transport="eval")
            lang_codes = self._run_prepare_session_language_codes(args)

        self.assertEqual(lang_codes, "")
        to_thread.assert_not_awaited()
        get_lang_codes.assert_not_called()

    def test_non_eval_transport_prewarms_services_and_loads_language_codes(self) -> None:
        to_thread = AsyncMock()
        get_lang_codes = Mock(return_value="en-US,de-DE")
        prewarm_tts = Mock(name="prewarm_tts")
        prewarm_asr = Mock(name="prewarm_asr")

        with (
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
                call(
                    prewarm_tts,
                    "tts.example:443",
                    "Magpie-Multilingual.EN-US.Aria",
                    "tts-fn-456",
                    "magpie-tts-multilingual",
                ),
            ]
        )
        get_lang_codes.assert_called_once_with(
            asr_server="asr.example:443",
            asr_model="parakeet",
            asr_function_id="fn-123",
            tts_server="tts.example:443",
            tts_voice_id="Magpie-Multilingual.EN-US.Aria",
            tts_function_id="tts-fn-456",
            tts_model="magpie-tts-multilingual",
        )

    def test_non_eval_transport_passes_llm_language_capabilities(self) -> None:
        get_lang_codes = Mock(return_value="en-US,de-DE")

        with (
            patch("examples.multilingual.pipeline.asyncio.to_thread", AsyncMock()),
            patch("examples.multilingual.pipeline.get_lang_codes", get_lang_codes),
        ):
            self._run_prepare_session_language_codes(
                RunnerArguments(),
                llm_supported_languages=["en", "de"],
            )

        self.assertEqual(get_lang_codes.call_args.kwargs["llm_supported_languages"], ["en", "de"])

    def _run_prepare_session_language_codes(
        self,
        runner_args: RunnerArguments,
        llm_supported_languages=None,
    ) -> str:
        return asyncio.run(
            _prepare_session_language_codes(
                runner_args,
                tts_server="tts.example:443",
                tts_voice="Magpie-Multilingual.EN-US.Aria",
                tts_function_id="tts-fn-456",
                tts_model="magpie-tts-multilingual",
                asr_server="asr.example:443",
                asr_model="parakeet",
                asr_function_id="fn-123",
                llm_supported_languages=llm_supported_languages,
            )
        )


class MultilingualASRSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_custom_or_unknown_asr_without_connection_fields_uses_default_server(self):
        default_asr = {"server": "default-asr.test:50051", "model": "default-asr-model"}
        for asr_id in ("custom-no-connection-fields", "self-hosted:unknown-asr"):
            with (
                self.subTest(asr_id=asr_id),
                patch.object(multilingual_pipeline, "create_transport", new_callable=AsyncMock),
                patch.object(multilingual_pipeline, "resolve_prompt", return_value=("test", "test prompt")),
                patch.object(
                    multilingual_pipeline,
                    "load_service_entry",
                    side_effect=lambda slot, _: default_asr if slot == "asr" else {},
                ),
                patch.object(multilingual_pipeline, "load_service_entry_by_id", return_value={}),
                patch.object(multilingual_pipeline, "nvidia_api_key", return_value=""),
                patch.object(multilingual_pipeline, "NvidiaSTTService") as streaming,
                patch.object(multilingual_pipeline, "NvidiaTranscribeSTTService") as transcribe,
                patch.object(
                    multilingual_pipeline,
                    "_prepare_session_language_codes",
                    new_callable=AsyncMock,
                    side_effect=RuntimeError("STT constructed"),
                ),
                self.assertRaisesRegex(RuntimeError, "STT constructed"),
            ):
                await multilingual_pipeline.bot(
                    SimpleNamespace(
                        body={
                            "pipeline_mode": "multilingual-assistant",
                            "asr_id": asr_id,
                        }
                    )
                )
            streaming.assert_called_once()
            self.assertEqual(streaming.call_args.kwargs["server"], default_asr["server"])
            self.assertEqual(streaming.call_args.kwargs["model_function_map"]["model_name"], default_asr["model"])
            transcribe.assert_not_called()

    async def test_selected_catalog_type_builds_stt_with_session_language_and_no_prompt(self):
        transcribe = {
            "asr_type": "transcribe",
            "language_code": "en-US",
            "supported_languages": ["de", "es", "hi"],
            "transcription_prompt": "English-only catalog prompt must be ignored.",
        }
        for language in ("de-DE", "es-419", "hi-IN", ""):
            session_language = language or "de-DE"
            for selected, expected in ((transcribe, True), ({}, False)):
                body = {
                    "pipeline_mode": "multilingual-assistant",
                    "asr_id": "self-hosted:asr",
                    "asr_server": "http://speech.invalid/v1",
                    "asr_model": "test-asr",
                    "asr_language_code": language,
                }
                with (
                    self.subTest(language=language, transcribe=expected),
                    patch.object(multilingual_pipeline, "create_transport", new_callable=AsyncMock),
                    patch.object(multilingual_pipeline, "resolve_prompt", return_value=("test", "test prompt")),
                    patch.object(multilingual_pipeline, "load_service_entry", return_value={}),
                    patch.object(multilingual_pipeline, "load_service_entry_by_id", return_value=selected),
                    patch.object(multilingual_pipeline, "nvidia_api_key", return_value=""),
                    patch.object(multilingual_pipeline, "NvidiaSTTService") as streaming,
                    patch.object(multilingual_pipeline, "NvidiaTranscribeSTTService") as http,
                    patch.object(
                        multilingual_pipeline,
                        "_prepare_session_language_codes",
                        new_callable=AsyncMock,
                        return_value=language,
                    ) as prepare,
                    patch.object(multilingual_pipeline, "NvidiaLLMService"),
                    patch.object(multilingual_pipeline, "NvidiaTTSService"),
                    patch.object(multilingual_pipeline, "resolve_voice_for_language", return_value="Voice"),
                    patch.object(multilingual_pipeline, "load_ipa_dictionary", return_value={}),
                    patch.object(multilingual_pipeline, "load_prompt_catalog", return_value={}),
                    patch.object(
                        multilingual_pipeline,
                        "_build_multilingual_user_aggregator_params",
                        side_effect=RuntimeError("params built"),
                    ) as params,
                    self.assertRaisesRegex(RuntimeError, "params built"),
                ):
                    await multilingual_pipeline.bot(SimpleNamespace(body=body))
                self.assertEqual(http.call_count, int(expected))
                self.assertEqual(streaming.call_count, int(not expected))
                params.assert_called_once_with(True, vad_only=expected)
                self.assertEqual(
                    prepare.call_args.kwargs["asr_supported_languages"], ["de", "es", "hi"] if expected else None
                )
                if expected:
                    http.Settings.assert_called_once_with(model="test-asr", language=session_language)
                    http.assert_called_once_with(base_url=body["asr_server"], settings=http.Settings.return_value)
