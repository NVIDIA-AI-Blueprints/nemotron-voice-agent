# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

import asyncio
import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from pipecat.frames.frames import ErrorFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings, NvidiaTTSSynthesisMode

from examples.generic.offline.factory import GenericOfflineRuntime, build_runtime
from examples.generic.offline.models import AudioSource, SessionResult
from examples.generic.offline.native_response import NativeResponseSession
from examples.shared.evaluation_trace import EvaluationTrace
from examples.shared.nvidia_tts_explicit_eof import EXPLICIT_EOF_COMPLETION_POLICY
from utils import load_prompt_catalog


class FactoryTests(unittest.TestCase):
    def build(
        self,
        config,
        local=False,
        *,
        policy="silero",
        tools=False,
        llm_api_key_env=None,
        prompt_key="test",
        prompt_entry=None,
        asr_settings=None,
        llm_settings=None,
        diarization_env="false",
        catalog_synthesis_mode=NvidiaTTSSynthesisMode.STITCHED,
    ):
        catalogs = {
            "asr": {"server": "asr:50051", "function_id": "asr-function"},
            "llm": {"model_id": "model", "base_url": "http://llm:8000/v1"},
            "tts": {"server": "tts:50051" if local else "grpc.nvcf.nvidia.com:443", "function_id": ""},
        }
        if llm_api_key_env is not None:
            catalogs["llm"]["api_key_env"] = llm_api_key_env
        catalogs["asr"].update(asr_settings or {})
        catalogs["llm"].update(llm_settings or {})
        if catalog_synthesis_mode is not None:
            catalogs["tts"]["synthesis_mode"] = catalog_synthesis_mode
        recognizer = Mock()
        detector = SimpleNamespace(configuration={"policy": policy, "vad": {"stop_secs": 0.5}})
        with ExitStack() as stack:
            prefix = "examples.generic.offline.factory."
            stack.enter_context(patch.dict(os.environ, {"ENABLE_SPEAKER_DIARIZATION": diarization_env}))
            stack.enter_context(patch(prefix + "load_service_entry", side_effect=lambda key, _: catalogs[key]))
            stack.enter_context(patch("utils.load_service_entry", side_effect=lambda key, _: catalogs[key]))
            stack.enter_context(patch("utils.load_service_entry_by_id", side_effect=lambda key, _: catalogs[key]))
            stack.enter_context(patch(prefix + "resolve_prompt", return_value=(prompt_key, "test prompt")))
            stack.enter_context(patch(prefix + "load_prompt_catalog", return_value={prompt_key: prompt_entry or {}}))
            tool_catalog = {"lookup": {"function": {"name": "lookup"}}} if tools else {}
            stack.enter_context(patch(prefix + "load_tools_catalog", return_value=tool_catalog))
            stack.enter_context(patch(prefix + "resolve_tools_available", return_value=["lookup"] if tools else []))
            stack.enter_context(patch("examples.generic.tool_handlers.TOOL_HANDLERS", {"lookup": Mock()}))
            stack.enter_context(patch(prefix + "load_ipa_dictionary", return_value=None))
            stack.enter_context(patch(prefix + "nvidia_api_key", return_value="test-key"))
            stack.enter_context(patch("utils.nvidia_api_key", return_value="test-key"))
            detector_factory = stack.enter_context(patch(prefix + "create_generic_detector", return_value=detector))
            asr_factory = stack.enter_context(patch(prefix + "create_riva_recognizer", return_value=recognizer))
            llm_factory = stack.enter_context(patch(prefix + "CapturingNativeNvidiaLLMService"))
            tts_factory = stack.enter_context(patch(prefix + "CapturingNativeNvidiaTTSService"))
            self.provider_factories = (detector_factory, asr_factory, llm_factory, tts_factory)
            runtime = build_runtime(config)
            capture = AsyncMock()
            runtime._synthesis._llm_factory(capture)
            runtime._synthesis._tts_factory(capture)
            runtime.test_llm_kwargs = llm_factory.call_args.kwargs
        return runtime, asr_factory.call_args.args[0], tts_factory.call_args.kwargs

    def test_cloud_default_retains_generic_magpie_function_id(self):
        runtime, asr, tts = self.build({})
        self.assertEqual(runtime.configuration["tts_function_id"], "877104f7-e885-42b9-8de8-f6e4c6303969")
        self.assertNotIn("model_function_map", tts)
        self.assertEqual(asr["asr_function_id"], "asr-function")
        self.assertFalse(runtime.configuration["automatic_introduction"])
        self.assertNotIn("api_key", runtime.configuration)
        self.assertEqual(runtime.configuration["detector"]["vad"]["stop_secs"], 0.5)

    def test_hub_llm_key_does_not_replace_speech_credentials(self):
        with patch.dict(os.environ, {"TEST_NVA_LLM_KEY": "hub-only-key"}):
            runtime, asr, tts = self.build({}, llm_api_key_env="TEST_NVA_LLM_KEY")
        self.assertEqual(runtime.test_llm_kwargs["api_key"], "hub-only-key")
        self.assertEqual(asr["api_key"], "test-key")
        self.assertEqual(tts["api_key"], "test-key")
        self.assertNotIn("hub-only-key", str(runtime.configuration))
        self.assertNotIn("test-key", str(runtime.configuration))

    def test_missing_explicit_llm_credential_fails_before_provider_construction(self):
        with (
            patch.dict(os.environ, {"TEST_NVA_LLM_KEY": ""}),
            self.assertRaisesRegex(RuntimeError, "TEST_NVA_LLM_KEY is not set"),
        ):
            self.build({}, llm_api_key_env="TEST_NVA_LLM_KEY")

    def test_custom_endpoint_without_a_selection_cannot_receive_catalog_credentials(self):
        with (
            patch.dict(os.environ, {"TEST_NVA_LLM_KEY": "hub-only-key"}),
            self.assertRaisesRegex(ValueError, "selected catalog endpoint"),
        ):
            self.build({"base_url": "https://client-selected.example/v1"}, llm_api_key_env="TEST_NVA_LLM_KEY")

    def test_explicit_function_id_wins(self):
        _, _, tts = self.build({"tts_function_id": "custom"})
        self.assertEqual(tts["model_function_map"]["function_id"], "custom")

    def test_local_tts_keeps_no_cloud_routing_id(self):
        _, _, tts = self.build({}, local=True)
        self.assertNotIn("model_function_map", tts)
        self.assertFalse(tts["use_ssl"])

    def test_omitted_synthesis_mode_matches_native_per_sentence_and_fails_before_providers(self):
        native = NvidiaTTSService(settings=NvidiaTTSSettings())
        self.assertEqual(native._settings.synthesis_mode, NvidiaTTSSynthesisMode.PER_SENTENCE)
        for omitted in (None, ""):
            with self.subTest(catalog_mode=omitted):
                with self.assertRaisesRegex(ValueError, "stitched Magpie"):
                    self.build({}, catalog_synthesis_mode=omitted)
                for factory in self.provider_factories:
                    factory.assert_not_called()

    def test_synthesis_mode_request_catalog_precedence_retains_explicit_stitched(self):
        cases = (
            ({}, NvidiaTTSSynthesisMode.STITCHED),
            ({"tts_synthesis_mode": ""}, NvidiaTTSSynthesisMode.STITCHED),
            ({"tts_synthesis_mode": "stitched"}, NvidiaTTSSynthesisMode.PER_SENTENCE),
            ({"tts_synthesis_mode": "stitched"}, None),
        )
        for config, catalog_mode in cases:
            with self.subTest(config=config, catalog_mode=catalog_mode):
                runtime, _, tts = self.build(config, catalog_synthesis_mode=catalog_mode)
                self.assertEqual(runtime.configuration["tts_synthesis_mode"], NvidiaTTSSynthesisMode.STITCHED)
                self.assertEqual(tts["settings"].synthesis_mode, NvidiaTTSSynthesisMode.STITCHED)
                self.assertIsNone(tts["stop_frame_timeout_s"])

    def test_effective_per_sentence_from_request_or_catalog_is_rejected_before_providers(self):
        cases = (
            ({"tts_synthesis_mode": "per_sentence"}, NvidiaTTSSynthesisMode.STITCHED),
            ({}, NvidiaTTSSynthesisMode.PER_SENTENCE),
            ({"tts_synthesis_mode": ""}, NvidiaTTSSynthesisMode.PER_SENTENCE),
        )
        for config, catalog_mode in cases:
            with self.subTest(config=config, catalog_mode=catalog_mode):
                with self.assertRaisesRegex(ValueError, "stitched Magpie"):
                    self.build(config, catalog_synthesis_mode=catalog_mode)
                for factory in self.provider_factories:
                    factory.assert_not_called()

    def test_effective_llm_flags_and_language_are_recorded(self):
        runtime, _, _ = self.build({"extra_params": '{"temperature":0.1}', "asr_language_code": "fr-FR"})
        self.assertEqual(runtime.configuration["llm_extra_params"], {"temperature": 0.1})
        self.assertEqual(runtime.configuration["asr_language_code"], "fr-FR")
        self.assertTrue(runtime.configuration["pronunciation_dictionary_sha256"])

    def test_extra_parameters_override_scalar_settings_like_the_live_request(self):
        runtime, asr, tts = self.build(
            {
                "extra_params": {"temperature": 0.8, "max_tokens": 8},
                "temperature": "0",
                "max_tokens": "128",
                "asr_model": "selected-asr",
                "asr_automatic_punctuation": "false",
                "tts_language_code": "FR-fr",
                "tool_choice": "none",
            }
        )
        self.assertEqual(runtime.configuration["llm_extra_params"], {"temperature": 0.8, "max_tokens": 8})
        self.assertEqual(asr["asr_model"], "selected-asr")
        self.assertFalse(asr["asr_automatic_punctuation"])
        self.assertEqual(tts["settings"].language, "fr-FR")
        self.assertEqual(runtime.configuration["tool_choice"], "none")
        self.assertFalse(runtime.configuration["asr_force_eou"])
        self.assertEqual(asr["asr_stop_history"], 320)
        self.assertEqual(runtime.configuration["asr_stop_history_ms"], asr["asr_stop_history"])
        self.assertEqual(runtime._synthesis._sample_rate, 22050)
        self.assertEqual(runtime.configuration["output_sample_rate"], 16000)

    def test_scalar_settings_are_used_when_not_overridden_in_extra(self):
        runtime, _, _ = self.build({"temperature": "0", "max_tokens": "128"})
        self.assertEqual(runtime.configuration["llm_extra_params"], {"temperature": 0.0, "max_tokens": 128})

    def test_temperature_uses_live_body_prompt_catalog_precedence_and_extra_override(self):
        cases = (
            ({"temperature": "0.2"}, {"temperature": 0.1}, {"temperature": 0.8}, 0.2, 0.2),
            ({"temperature": ""}, {"temperature": 0}, {"temperature": 0.8}, 0.0, 0.0),
            ({}, {}, {"temperature": 0.8}, 0.8, 0.8),
            ({"temperature": "0.2", "extra_params": {"temperature": 0.9}}, {"temperature": 0.1}, {}, 0.2, 0.9),
        )
        for config, prompt, catalog, scalar, effective in cases:
            with self.subTest(config=config, prompt=prompt, catalog=catalog):
                runtime, _, _ = self.build(config, prompt_entry=prompt, llm_settings=catalog)
                self.assertEqual(runtime.test_llm_kwargs["settings"].temperature, scalar)
                self.assertEqual(runtime.configuration["llm_extra_params"]["temperature"], effective)

    def test_invalid_body_temperature_does_not_fall_back_to_prompt_or_catalog(self):
        runtime, _, _ = self.build(
            {"temperature": "invalid"}, prompt_entry={"temperature": 0.1}, llm_settings={"temperature": 0.8}
        )
        self.assertNotIn("temperature", runtime.configuration["llm_extra_params"])

    def test_registered_multi_speaker_prompt_fails_before_any_provider_initialization(self):
        pipeline = Path(__file__).resolve().parents[2] / "src/examples/generic/pipeline.py"
        prompt = load_prompt_catalog(pipeline)["multi_speaker_assistant"]
        with self.assertRaisesRegex(ValueError, "multi-speaker prompts"):
            self.build({}, prompt_key="multi_speaker_assistant", prompt_entry=prompt)
        for factory in self.provider_factories:
            factory.assert_not_called()

    def test_supported_diarization_request_or_environment_fails_before_providers(self):
        cases = (
            ({"asr_speaker_diarization": True}, {"speaker_diarization_supported": True}, "false"),
            ({}, {"speaker_diarization_supported": True}, "true"),
            (
                {"asr_id": "selected", "asr_speaker_diarization": "true", "asr_speaker_diarization_supported": True},
                {},
                "false",
            ),
        )
        for config, asr, environment in cases:
            with self.subTest(config=config, environment=environment):
                with self.assertRaisesRegex(ValueError, "speaker diarization"):
                    self.build(config, asr_settings=asr, diarization_env=environment)
                for factory in self.provider_factories:
                    factory.assert_not_called()

    def test_native_diarization_resolution_preserves_valid_single_speaker_configuration(self):
        cases = (
            ({"asr_speaker_diarization": False}, {"speaker_diarization_supported": True}, "true"),
            ({"asr_speaker_diarization": True}, {"speaker_diarization_supported": False}, "false"),
            (
                {"asr_id": "selected", "asr_speaker_diarization": True, "asr_speaker_diarization_supported": False},
                {"speaker_diarization_supported": True},
                "false",
            ),
        )
        for config, asr, environment in cases:
            with self.subTest(config=config, environment=environment):
                runtime, _, _ = self.build(config, asr_settings=asr, diarization_env=environment)
                self.assertEqual(
                    runtime.configuration["turn_policy"], "actual_Pipecat_user_aggregator_with_sample_clock_waits"
                )

    def test_deployment_history_reaches_the_recognizer_and_effective_configuration(self):
        with patch.dict(os.environ, {"ASR_STOP_HISTORY_MS": "400"}):
            runtime, asr, _ = self.build({})
        self.assertEqual(asr["asr_stop_history"], 400)
        self.assertEqual(runtime.configuration["asr_stop_history_ms"], 400)

    def test_unsupported_turn_policy_names_the_actual_configuration_flag(self):
        with self.assertRaisesRegex(ValueError, "USE_SILERO_VAD_TURN_DETECTION=true"):
            self.build({}, policy="smart_turn")

    def test_production_uses_native_service_settings_and_filter_without_a_custom_openai_client(self):
        runtime, _, tts = self.build(
            {"temperature": "0", "max_tokens": "128", "extra_params": {"temperature": 0.8, "max_tokens": 8}},
            local=True,
        )
        self.assertIsNone(runtime._client)
        self.assertEqual(type(runtime._engine).__name__, "IncrementalSessionEngine")
        llm = runtime.test_llm_kwargs
        self.assertEqual(llm["base_url"], "http://llm:8000/v1")
        self.assertEqual(llm["settings"].model, "model")
        self.assertEqual(llm["settings"].temperature, 0)
        self.assertEqual(llm["settings"].max_tokens, 128)
        self.assertEqual(llm["settings"].extra, {"temperature": 0.8, "max_tokens": 8})
        self.assertEqual(tts["settings"].synthesis_mode, "stitched")
        self.assertEqual(type(tts["text_filters"][0]).__name__, "NemotronSpeechTextFilter")
        self.assertNotIn("enable_direct_mode", llm)
        self.assertNotIn("enable_direct_mode", tts)
        self.assertIn("stop_frame_timeout_s", tts)
        self.assertIsNone(tts["stop_frame_timeout_s"])
        self.assertEqual(runtime.configuration["tts_context_completion_policy"], EXPLICIT_EOF_COMPLETION_POLICY)
        self.assertEqual(
            runtime.configuration["provider_execution"], "overlapping_continuous_ASR_and_persistent_native_LLM_TTS"
        )
        self.assertEqual(runtime._provider_observations, runtime._synthesis.get_provider_observations)

    def test_summary_eligibility_is_explicitly_guarded_instead_of_customized_before_next_request(self):
        runtime, _, _ = self.build({})
        self.assertEqual(runtime._engine.history_recent_turns, 10)
        self.assertEqual(runtime._engine.preserve_prompt_messages, len(runtime._engine.initial_messages))
        self.assertEqual(
            runtime.configuration["history_summarization"], "unsupported_guarded_at_native_assistant_turn_stop"
        )

    def test_schema_bearing_prompts_are_rejected_even_with_tool_choice_none(self):
        with self.assertRaisesRegex(ValueError, "prompt without tools"):
            self.build({"tool_choice": "none"}, tools=True)

    def test_unsupported_streaming_input_and_synthesis_modes_fail_explicitly(self):
        cases = (
            ({"base_url": "ws://llm:8000/v1/streaming-session"}, "HTTP chat-completions"),
            ({"tts_synthesis_mode": "per_sentence"}, "stitched Magpie"),
            ({"tts_model": "magpie-tts-zeroshot"}, "stitched Magpie"),
            ({"tts_zero_shot_audio_prompt_file": "/trusted/prompt.wav"}, "stitched Magpie"),
        )
        for config, message in cases:
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, message):
                self.build(config)


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_cleanup_failure_still_releases_other_resources_and_can_be_retried(self):
        for failing_step in ("recognizer_cancel", "synthesis_cancel", "synthesis_close", "client_close"):
            with self.subTest(failing_step=failing_step):
                operations = {
                    "recognizer_cancel": Mock(),
                    "synthesis_cancel": Mock(),
                    "synthesis_close": AsyncMock(),
                    "client_close": AsyncMock(),
                    "trace_close": Mock(),
                }
                failure = RuntimeError("synthetic cleanup failure")
                operations[failing_step].side_effect = [failure, None]
                runtime = GenericOfflineRuntime(
                    None,
                    SimpleNamespace(close=operations["client_close"]),
                    SimpleNamespace(cancel=operations["recognizer_cancel"]),
                    SimpleNamespace(cancel=operations["synthesis_cancel"], aclose=operations["synthesis_close"]),
                    {},
                    evaluation_trace=SimpleNamespace(close=operations["trace_close"]),
                )
                with self.assertRaises(RuntimeError) as caught:
                    await runtime.aclose()
                self.assertIs(caught.exception, failure)
                self.assertFalse(runtime._closed)
                for operation in operations.values():
                    self.assertEqual(operation.call_count, 1)
                await runtime.aclose()
                await runtime.aclose()
                self.assertTrue(runtime._closed)
                for name, operation in operations.items():
                    self.assertEqual(operation.call_count, 2 if name == failing_step else 1)

    async def test_execution_error_survives_multiple_cleanup_failures_and_every_release_is_attempted(self):
        original = ValueError("synthetic execution failure")
        recognizer = SimpleNamespace(cancel=Mock(side_effect=RuntimeError("synthetic cancel failure")))
        synthesis = SimpleNamespace(aclose=AsyncMock(side_effect=OSError("synthetic session failure")))
        client = SimpleNamespace(close=AsyncMock())
        trace = SimpleNamespace(close=Mock(), offline_source=Mock())
        runtime = GenericOfflineRuntime(
            SimpleNamespace(run=AsyncMock(side_effect=original)),
            client,
            recognizer,
            synthesis,
            {},
            evaluation_trace=trace,
        )
        with self.assertRaises(ValueError) as caught:
            await runtime.run(AudioSource(bytes(2)))
        self.assertIs(caught.exception, original)
        self.assertIn("Runtime cleanup also failed: RuntimeError", original.__notes__)
        recognizer.cancel.assert_called_once()
        synthesis.aclose.assert_awaited_once()
        client.close.assert_awaited_once()
        trace.close.assert_called_once()

    async def test_cancelled_cleanup_attempt_still_releases_later_resources_and_can_retry(self):
        synthesis = SimpleNamespace(aclose=AsyncMock(side_effect=[asyncio.CancelledError(), None]))
        client = SimpleNamespace(close=AsyncMock())
        recognizer = SimpleNamespace(cancel=Mock())
        runtime = GenericOfflineRuntime(None, client, recognizer, synthesis, {})
        with self.assertRaises(asyncio.CancelledError):
            await runtime.aclose()
        client.close.assert_awaited_once()
        await runtime.aclose()
        self.assertEqual(synthesis.aclose.await_count, 2)
        recognizer.cancel.assert_called_once()
        client.close.assert_awaited_once()

    async def test_concurrent_close_calls_release_owned_resources_once(self):
        synthesis = SimpleNamespace(aclose=AsyncMock())
        client = SimpleNamespace(close=AsyncMock())
        recognizer = SimpleNamespace(cancel=Mock())
        runtime = GenericOfflineRuntime(None, client, recognizer, synthesis, {})
        await asyncio.gather(runtime.aclose(), runtime.aclose())
        recognizer.cancel.assert_called_once()
        synthesis.aclose.assert_awaited_once()
        client.close.assert_awaited_once()

    async def test_trace_closes_after_provider_cleanup_with_exact_source_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            trace = EvaluationTrace(Path(folder))
            source = AudioSource(bytes(1024))
            result = SessionResult((), 0, (), (), {})

            async def cleanup():
                self.assertFalse(trace.path.exists())
                trace.record("provider_cleanup_returning")

            runtime = GenericOfflineRuntime(
                SimpleNamespace(run=AsyncMock(return_value=result)),
                None,
                Mock(),
                SimpleNamespace(aclose=AsyncMock(side_effect=cleanup)),
                {},
                evaluation_trace=trace,
            )
            self.assertIs(await runtime.run(source), result)
            events = json.loads(trace.path.read_text())["events"]
            source_event = next(e for e in events if e["event"] == "offline_source")
            self.assertEqual(source_event["source_samples"], 512)
            self.assertEqual(source_event["source_sample_rate"], 16000)
            self.assertEqual(events[-2]["event"], "provider_cleanup_returning")
            self.assertEqual(events[-1]["event"], "trace_closed")

    async def test_failed_trace_writer_preserves_original_execution_error(self):
        with tempfile.TemporaryDirectory() as folder:
            trace = EvaluationTrace(Path(folder))
            original = RuntimeError("native execution failure")
            synthesis = SimpleNamespace(aclose=AsyncMock())
            runtime = GenericOfflineRuntime(
                SimpleNamespace(run=AsyncMock(side_effect=original)),
                None,
                Mock(),
                synthesis,
                {},
                evaluation_trace=trace,
            )
            with (
                patch.object(trace, "close", side_effect=OSError("writer unavailable")),
                self.assertRaises(RuntimeError) as caught,
            ):
                await runtime.run(AudioSource(bytes(2)))
            self.assertIs(caught.exception, original)
            synthesis.aclose.assert_awaited_once()
            self.assertEqual(trace._observation_errors, 1)

    async def test_observations_are_collected_after_cleanup_into_a_new_result(self):
        events = []
        result = SessionResult((), 0, (), (), {"existing": "value"})
        native_session = NativeResponseSession(None, None, sample_rate=16000)
        native_session._tts = object()

        async def close_session():
            events.append("cleanup")
            frame = ErrorFrame(error="secret-provider-body", processor=native_session._tts)
            frame.metadata["private"] = "secret-provider-context"
            await native_session._from_tts(frame, FrameDirection.UPSTREAM)

        def observations():
            self.assertEqual(events, ["cleanup"])
            events.append("collect")
            return native_session.get_provider_observations()

        engine = SimpleNamespace(run=AsyncMock(return_value=result))
        recognizer = Mock()
        session = SimpleNamespace(aclose=AsyncMock(side_effect=close_session))
        runtime = GenericOfflineRuntime(engine, None, recognizer, session, {}, provider_observations=observations)
        collected = await runtime.run(AudioSource(bytes(2)))
        self.assertIsNot(collected, result)
        self.assertEqual(result.metrics, {"existing": "value"})
        self.assertEqual(collected.metrics["existing"], "value")
        self.assertEqual(collected.metrics["native_provider_observations"]["total"], 1)
        self.assertEqual(
            collected.metrics["native_provider_observations"]["observations"],
            [{"provider_source": "tts", "observed_callback_boundary": "tts", "observation_index": 1}],
        )
        self.assertEqual(events, ["cleanup", "collect"])
        self.assertNotIn("secret", str(collected.metrics))
        await runtime.aclose()
        session.aclose.assert_awaited_once()
        self.assertEqual(events, ["cleanup", "collect"])

    async def test_optional_observation_collection_failure_preserves_the_result(self):
        result = SessionResult((), 0, (), (), {"existing": "value"})
        collector = Mock(side_effect=RuntimeError("secret-provider-body"))
        runtime = GenericOfflineRuntime(
            SimpleNamespace(run=AsyncMock(return_value=result)),
            None,
            Mock(),
            SimpleNamespace(aclose=AsyncMock()),
            {},
            provider_observations=collector,
        )
        self.assertIs(await runtime.run(AudioSource(bytes(2))), result)
        self.assertEqual(result.metrics, {"existing": "value"})
        collector.assert_called_once()

    async def test_execution_and_cleanup_errors_do_not_invoke_optional_collector(self):
        for failure_during in ("engine", "cleanup"):
            with self.subTest(failure_during=failure_during):
                failure = RuntimeError("primary failure")
                result = SessionResult((), 0, (), ())
                engine = SimpleNamespace(
                    run=AsyncMock(return_value=result, side_effect=failure if failure_during == "engine" else None)
                )
                session = SimpleNamespace(
                    aclose=AsyncMock(side_effect=failure if failure_during == "cleanup" else None)
                )
                collector = Mock(side_effect=RuntimeError("must not replace primary failure"))
                runtime = GenericOfflineRuntime(engine, None, Mock(), session, {}, provider_observations=collector)
                with self.assertRaises(RuntimeError) as caught:
                    await runtime.run(AudioSource(bytes(2)))
                self.assertIs(caught.exception, failure)
                collector.assert_not_called()

    async def test_native_runtime_closes_its_session_without_creating_or_requiring_a_separate_client(self):
        engine = SimpleNamespace(run=AsyncMock(return_value="result"))
        recognizer = Mock()
        session = SimpleNamespace(aclose=AsyncMock())
        runtime = GenericOfflineRuntime(engine, None, recognizer, session, {})
        self.assertEqual(await runtime.run(AudioSource(bytes(2))), "result")
        await runtime.aclose()
        recognizer.cancel.assert_called_once()
        session.aclose.assert_awaited_once()

    async def test_execution_failure_closes_owned_provider_resources_once(self):
        engine = SimpleNamespace(run=AsyncMock(side_effect=RuntimeError("failure")))
        client = SimpleNamespace(close=AsyncMock())
        recognizer, synthesis = Mock(), Mock()
        synthesis.aclose = AsyncMock()
        runtime = GenericOfflineRuntime(engine, client, recognizer, synthesis, {})
        with self.assertRaises(RuntimeError):
            await runtime.run(AudioSource(bytes(2)))
        await runtime.aclose()
        recognizer.cancel.assert_called_once()
        synthesis.cancel.assert_called_once()
        synthesis.aclose.assert_awaited_once()
        client.close.assert_awaited_once()
