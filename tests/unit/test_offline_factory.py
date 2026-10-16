# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from examples.generic.offline.factory import GenericOfflineRuntime, build_runtime
from examples.generic.offline.models import AudioSource


class FactoryTests(unittest.TestCase):
    def build(self, config, local=False):
        catalogs = {
            "asr": {"server": "asr:50051", "function_id": "asr-function"},
            "llm": {"model_id": "model", "base_url": "http://llm:8000/v1"},
            "tts": {"server": "tts:50051" if local else "grpc.nvcf.nvidia.com:443", "function_id": ""},
        }
        recognizer = Mock()
        synthesis = Mock()
        client = Mock()
        client.close = AsyncMock()
        detector = SimpleNamespace(configuration={"policy": "silero", "vad": {"stop_secs": 0.5}})
        with ExitStack() as stack:
            prefix = "examples.generic.offline.factory."
            stack.enter_context(patch(prefix + "load_service_entry", side_effect=lambda key, _: catalogs[key]))
            stack.enter_context(patch(prefix + "resolve_prompt", return_value=("test", "test prompt")))
            stack.enter_context(patch(prefix + "load_tools_catalog", return_value={}))
            stack.enter_context(patch(prefix + "resolve_tools_available", return_value=[]))
            stack.enter_context(patch(prefix + "load_ipa_dictionary", return_value=None))
            stack.enter_context(patch(prefix + "create_generic_detector", return_value=detector))
            asr_factory = stack.enter_context(patch(prefix + "create_riva_recognizer", return_value=recognizer))
            tts_factory = stack.enter_context(patch(prefix + "create_riva_synthesis", return_value=synthesis))
            stack.enter_context(patch(prefix + "AsyncOpenAI", return_value=client))
            runtime = build_runtime(config)
        return runtime, asr_factory.call_args.args[0], tts_factory.call_args.args[0]

    def test_cloud_default_retains_generic_magpie_function_id(self):
        runtime, asr, tts = self.build({})
        self.assertEqual(tts["tts_function_id"], "877104f7-e885-42b9-8de8-f6e4c6303969")
        self.assertEqual(asr["asr_function_id"], "asr-function")
        self.assertFalse(runtime.configuration["automatic_introduction"])
        self.assertNotIn("api_key", runtime.configuration)
        self.assertEqual(runtime.configuration["detector"]["vad"]["stop_secs"], 0.5)

    def test_explicit_function_id_wins(self):
        _, _, tts = self.build({"tts_function_id": "custom"})
        self.assertEqual(tts["tts_function_id"], "custom")

    def test_local_tts_keeps_no_cloud_routing_id(self):
        _, _, tts = self.build({}, local=True)
        self.assertEqual(tts["tts_function_id"], "")

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
        self.assertEqual(tts["tts_language_code"], "fr-FR")
        self.assertEqual(runtime.configuration["tool_choice"], "none")
        self.assertFalse(runtime.configuration["asr_force_eou"])
        self.assertEqual(asr["asr_stop_history"], 320)
        self.assertEqual(runtime.configuration["asr_stop_history_ms"], asr["asr_stop_history"])
        self.assertEqual(tts["tts_sample_rate"], 22050)
        self.assertEqual(runtime.configuration["output_sample_rate"], 16000)

    def test_scalar_settings_are_used_when_not_overridden_in_extra(self):
        runtime, _, _ = self.build({"temperature": "0", "max_tokens": "128"})
        self.assertEqual(runtime.configuration["llm_extra_params"], {"temperature": 0.0, "max_tokens": 128})

    def test_summary_explicit_token_limit_overrides_request_settings(self):
        async def invoke_summary(extra, limit):
            runtime, _, _ = self.build({"extra_params": extra})
            inference = runtime._client.chat.completions.create
            runtime._client.chat.completions.create = AsyncMock(
                return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))])
            )

            async def summarize(*, context, llm, **kwargs):
                self.assertEqual(await llm.run_inference(context, max_tokens=limit), "summary")

            with patch("examples.generic.offline.factory.apply_pinned_prompt_summary", summarize):
                await runtime._engine._responder._text_stream._history_preparer([{"role": "user", "content": "old"}])
            arguments = runtime._client.chat.completions.create.call_args.kwargs
            runtime._client.chat.completions.create = inference
            return arguments

        import asyncio

        for extra, limit, token_key, expected in (
            ({"max_tokens": 8}, 64, "max_tokens", 64),
            ({"max_completion_tokens": 8}, 64, "max_completion_tokens", 64),
            ({"max_tokens": 8}, None, "max_tokens", 8),
        ):
            with self.subTest(extra=extra, limit=limit):
                arguments = asyncio.run(invoke_summary(extra, limit))
                self.assertEqual(arguments[token_key], expected)

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
    async def test_execution_failure_closes_owned_provider_resources_once(self):
        engine = SimpleNamespace(run=AsyncMock(side_effect=RuntimeError("failure")))
        client = SimpleNamespace(close=AsyncMock())
        recognizer, synthesis = Mock(), Mock()
        runtime = GenericOfflineRuntime(engine, client, recognizer, synthesis, {})
        with self.assertRaises(RuntimeError):
            await runtime.run(AudioSource(bytes(2)))
        await runtime.aclose()
        recognizer.cancel.assert_called_once()
        synthesis.cancel.assert_called_once()
        client.close.assert_awaited_once()
