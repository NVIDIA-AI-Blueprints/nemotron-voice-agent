# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from examples.generic.offline.factory import GenericOfflineRuntime, build_runtime
from examples.generic.offline.models import AudioSource


class FactoryTests(unittest.TestCase):
    def build(self, config, local=False, *, policy="silero", tools=False, llm_api_key_env=None):
        catalogs = {
            "asr": {"server": "asr:50051", "function_id": "asr-function"},
            "llm": {"model_id": "model", "base_url": "http://llm:8000/v1"},
            "tts": {"server": "tts:50051" if local else "grpc.nvcf.nvidia.com:443", "function_id": ""},
        }
        if llm_api_key_env is not None:
            catalogs["llm"]["api_key_env"] = llm_api_key_env
        recognizer = Mock()
        detector = SimpleNamespace(configuration={"policy": policy, "vad": {"stop_secs": 0.5}})
        with ExitStack() as stack:
            prefix = "examples.generic.offline.factory."
            stack.enter_context(patch(prefix + "load_service_entry", side_effect=lambda key, _: catalogs[key]))
            stack.enter_context(patch("utils.load_service_entry", side_effect=lambda key, _: catalogs[key]))
            stack.enter_context(patch("utils.load_service_entry_by_id", side_effect=lambda key, _: catalogs[key]))
            stack.enter_context(patch(prefix + "resolve_prompt", return_value=("test", "test prompt")))
            tool_catalog = {"lookup": {"function": {"name": "lookup"}}} if tools else {}
            stack.enter_context(patch(prefix + "load_tools_catalog", return_value=tool_catalog))
            stack.enter_context(patch(prefix + "resolve_tools_available", return_value=["lookup"] if tools else []))
            stack.enter_context(patch("examples.generic.tool_handlers.TOOL_HANDLERS", {"lookup": Mock()}))
            stack.enter_context(patch(prefix + "load_ipa_dictionary", return_value=None))
            stack.enter_context(patch(prefix + "nvidia_api_key", return_value="test-key"))
            stack.enter_context(patch("utils.nvidia_api_key", return_value="test-key"))
            stack.enter_context(patch(prefix + "create_generic_detector", return_value=detector))
            asr_factory = stack.enter_context(patch(prefix + "create_riva_recognizer", return_value=recognizer))
            llm_factory = stack.enter_context(patch(prefix + "CapturingNativeNvidiaLLMService"))
            tts_factory = stack.enter_context(patch(prefix + "CapturingNativeNvidiaTTSService"))
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
        self.assertEqual(
            runtime.configuration["provider_execution"], "overlapping_continuous_ASR_and_persistent_native_LLM_TTS"
        )

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
