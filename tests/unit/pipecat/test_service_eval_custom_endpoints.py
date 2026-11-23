# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise custom service-eval endpoints without contacting inference services."""

from __future__ import annotations

import importlib
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import yaml
from openai import AsyncOpenAI
from pipecat.services.nvidia.llm import NvidiaLLMService
from pipecat.services.nvidia.stt import NvidiaSTTService

SERVICE_DIR = Path(__file__).resolve().parents[2] / "evals/pipecat/service"
with patch.object(sys, "path", [str(SERVICE_DIR), *sys.path]):
    endpoints = importlib.import_module("custom_endpoints")
    services = importlib.import_module("nvidia_inference_services")
    eval_bot = importlib.import_module("inference_eval_bot")
    runner = importlib.import_module("run_cloud_tts_evals")

LLM_ENV = {
    "CUSTOM_LLM_BASE_URL": "https://llm.example.org/v1",
    "CUSTOM_LLM_MODEL_ID": "organization/model:latest",
    "CUSTOM_LLM_API_KEY": "custom-llm-secret",
}
ASR_ENV = {
    "CUSTOM_ASR_SERVER": "asr.example.org:50051",
    "CUSTOM_ASR_MODEL_ID": "custom-streaming-asr",
    "CUSTOM_ASR_USE_SSL": "false",
    "CUSTOM_ASR_API_KEY": "custom-asr-secret",
}


class EndpointConfigTests(unittest.TestCase):
    """Validate configuration boundaries and credential isolation."""

    def test_unset_endpoints_preserve_defaults(self):
        """Leave native service selection unchanged when both pairs are empty."""
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(endpoints.custom_llm())
            self.assertIsNone(endpoints.custom_asr())

    def test_incomplete_pairs_and_invalid_values_fail(self):
        """Reject malformed endpoints, model identifiers, TLS, and request options."""
        cases = [
            (LLM_ENV, "CUSTOM_LLM_BASE_URL", "", endpoints.custom_llm),
            (LLM_ENV, "CUSTOM_LLM_MODEL_ID", "", endpoints.custom_llm),
            (LLM_ENV, "CUSTOM_LLM_BASE_URL", "https://user:secret@llm.example.org/v1", endpoints.custom_llm),
            (LLM_ENV, "CUSTOM_LLM_BASE_URL", "https://llm.example.org:65536/v1", endpoints.custom_llm),
            (LLM_ENV, "CUSTOM_LLM_BASE_URL", "https://llm.example.org/v1?key=secret", endpoints.custom_llm),
            (LLM_ENV, "CUSTOM_LLM_BASE_URL", "https://llm.example.org/v1\n", endpoints.custom_llm),
            (LLM_ENV, "CUSTOM_LLM_MODEL_ID", "invalid model", endpoints.custom_llm),
            (LLM_ENV, "CUSTOM_LLM_MAX_TOKENS", "0", endpoints.custom_llm),
            (LLM_ENV, "CUSTOM_LLM_MAX_TOKENS", "1.5", endpoints.custom_llm),
            (LLM_ENV, "CUSTOM_LLM_EXTRA_PARAMS", "[]", endpoints.custom_llm),
            (LLM_ENV, "CUSTOM_LLM_EXTRA_PARAMS", '{"temperature":NaN}', endpoints.custom_llm),
            (LLM_ENV, "CUSTOM_LLM_EXTRA_PARAMS", '{"extra_body":[]}', endpoints.custom_llm),
            (LLM_ENV, "CUSTOM_LLM_EXTRA_PARAMS", '{"model":"different-model"}', endpoints.custom_llm),
            (LLM_ENV, "CUSTOM_LLM_EXTRA_PARAMS", '{"extra_body":{"model":"different-model"}}', endpoints.custom_llm),
            (
                LLM_ENV,
                "CUSTOM_LLM_EXTRA_PARAMS",
                '{"extra_headers":{"Authorization":"other-key"}}',
                endpoints.custom_llm,
            ),
            (ASR_ENV, "CUSTOM_ASR_SERVER", "", endpoints.custom_asr),
            (ASR_ENV, "CUSTOM_ASR_MODEL_ID", "", endpoints.custom_asr),
            (ASR_ENV, "CUSTOM_ASR_SERVER", "https://asr.example.org/v1", endpoints.custom_asr),
            (ASR_ENV, "CUSTOM_ASR_SERVER", "asr.example.org:65536", endpoints.custom_asr),
            (ASR_ENV, "CUSTOM_ASR_USE_SSL", "yes", endpoints.custom_asr),
            (ASR_ENV, "CUSTOM_ASR_FUNCTION_ID", "bad-function-id", endpoints.custom_asr),
            (ASR_ENV, "CUSTOM_ASR_FUNCTION_ID", "01234567-89ab-cdef-0123-456789abcdef", endpoints.custom_asr),
        ]
        for env, key, value, read in cases:
            with (
                self.subTest(key=key, value=value),
                patch.dict(os.environ, {**env, key: value}, clear=True),
                self.assertRaises(ValueError),
            ):
                read()

    def test_missing_custom_credentials_do_not_use_hosted_keys(self):
        """Fail closed despite valid native speech and Inference API credentials."""
        for env, name, read in [
            (LLM_ENV, "CUSTOM_LLM_API_KEY", endpoints.custom_llm),
            (ASR_ENV, "CUSTOM_ASR_API_KEY", endpoints.custom_asr),
        ]:
            configured = {
                **env,
                "NVIDIA_API_KEY": "native-speech-secret",
                "NVIDIA_INFERENCE_API_KEY": "native-hub-secret",
            }
            for key in ["", "bad\nkey"]:
                with (
                    self.subTest(name=name, key=key),
                    patch.dict(os.environ, {**configured, name: key}, clear=True),
                    self.assertRaisesRegex(ValueError, name),
                ):
                    read()

    def test_nvcf_function_requires_tls_and_survives_validation(self):
        """Support NVCF routing as well as directly hosted plaintext Riva."""
        function = "01234567-89ab-cdef-0123-456789abcdef"
        with patch.dict(
            os.environ, {**ASR_ENV, "CUSTOM_ASR_USE_SSL": "true", "CUSTOM_ASR_FUNCTION_ID": function}, clear=True
        ):
            settings = endpoints.custom_asr()
        self.assertTrue(settings["use_ssl"])
        self.assertEqual(settings["function_id"], function)


class ServiceRoutingTests(unittest.TestCase):
    """Inspect native constructor arguments while preventing network calls."""

    def test_llm_override_replaces_inherited_thinking_options_without_mutating_settings(self):
        """Send the requested endpoint, key, token limit, and only explicit model options."""
        original = NvidiaLLMService.Settings(model="native-model", extra={"extra_body": {"enable_thinking": True}})
        env = {**LLM_ENV, "CUSTOM_LLM_MAX_TOKENS": "8192"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(NvidiaLLMService, "__init__", return_value=None) as init,
        ):
            services.InferenceNvidiaLLMService(
                api_key="native-speech-secret", base_url="https://native.example.org", settings=original
            )
        kwargs = init.call_args.kwargs
        self.assertEqual(kwargs["base_url"], LLM_ENV["CUSTOM_LLM_BASE_URL"])
        self.assertEqual(kwargs["api_key"], LLM_ENV["CUSTOM_LLM_API_KEY"])
        self.assertEqual(kwargs["settings"].model, LLM_ENV["CUSTOM_LLM_MODEL_ID"])
        self.assertEqual(kwargs["settings"].max_tokens, 8192)
        self.assertEqual(kwargs["settings"].extra, {})
        self.assertEqual(original.model, "native-model")
        self.assertTrue(original.extra["extra_body"]["enable_thinking"])

    def test_explicit_reasoning_options_survive_and_native_tts_keeps_hosted_key(self):
        """Separate custom LLM authentication from the unchanged hosted TTS adapter."""
        extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        env = {**LLM_ENV, "CUSTOM_LLM_EXTRA_PARAMS": json.dumps(extra), "NVIDIA_INFERENCE_API_KEY": "native-hub-secret"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(NvidiaLLMService, "__init__", return_value=None) as init,
        ):
            services.InferenceNvidiaLLMService()
            self.assertEqual(services._api_key(), "native-hub-secret")
        self.assertEqual(init.call_args.kwargs["settings"].extra, extra)

    def test_no_llm_override_preserves_native_service_constructor(self):
        """Keep the CI-only hosted key and the supplied settings by default."""
        original = NvidiaLLMService.Settings(model="native-model")
        with (
            patch.dict(os.environ, {"NVIDIA_INFERENCE_API_KEY": "native-hub-secret"}, clear=True),
            patch.object(NvidiaLLMService, "__init__", return_value=None) as init,
        ):
            services.InferenceNvidiaLLMService(api_key="native-speech-secret", settings=original)
        self.assertEqual(init.call_args.kwargs["api_key"], "native-hub-secret")
        self.assertIs(init.call_args.kwargs["settings"], original)

    def test_asr_override_clears_native_function_id_and_keeps_language(self):
        """Replace NVCF routing with direct Riva while preserving benchmark audio settings."""
        settings = NvidiaSTTService.Settings(language="en-US")
        with (
            patch.dict(os.environ, ASR_ENV, clear=True),
            patch.object(NvidiaSTTService, "__init__", return_value=None) as init,
        ):
            services.InferenceNvidiaSTTService(
                api_key="native-speech-secret",
                server="grpc.nvcf.nvidia.com:443",
                use_ssl=True,
                model_function_map={"function_id": "native-function"},
                settings=settings,
                stop_history=320,
            )
        kwargs = init.call_args.kwargs
        self.assertEqual(kwargs["server"], ASR_ENV["CUSTOM_ASR_SERVER"])
        self.assertEqual(kwargs["api_key"], ASR_ENV["CUSTOM_ASR_API_KEY"])
        self.assertFalse(kwargs["use_ssl"])
        self.assertEqual(
            kwargs["model_function_map"], {"model_name": ASR_ENV["CUSTOM_ASR_MODEL_ID"], "function_id": ""}
        )
        self.assertIs(kwargs["settings"], settings)
        self.assertEqual(kwargs["stop_history"], 320)


class EvalBotOverrideTests(unittest.IsolatedAsyncioTestCase):
    """Verify the disposable bot records the effective services without secret values."""

    async def test_native_streaming_request_reaches_custom_endpoint_with_only_its_key(self):
        """Exercise the real Pipecat/OpenAI request builder through an in-memory HTTP transport."""
        received = []

        def handler(request):
            received.append(request)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    'data: {"id":"test","object":"chat.completion.chunk","created":1,'
                    '"model":"organization/model:latest","choices":[{"index":0,'
                    '"delta":{"content":"hello"},"finish_reason":"stop"}]}\n\n'
                    "data: [DONE]\n\n"
                ),
            )

        def create_client(_self, **kwargs):
            return AsyncOpenAI(
                api_key=kwargs["api_key"],
                base_url=kwargs["base_url"],
                max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            )

        with (
            patch.dict(os.environ, LLM_ENV, clear=True),
            patch.object(NvidiaLLMService, "create_client", create_client),
        ):
            service = services.InferenceNvidiaLLMService(
                api_key="native-speech-secret",
                settings=NvidiaLLMService.Settings(
                    model="native-model", extra={"extra_body": {"enable_thinking": True}}
                ),
            )
            try:
                params = service.build_chat_completion_params({"messages": [{"role": "user", "content": "Hello"}]})
                stream = await service._client.chat.completions.create(**params)
                chunks = [chunk async for chunk in stream]
            finally:
                await service._client.close()
        self.assertEqual(chunks[0].choices[0].delta.content, "hello")
        self.assertEqual(len(received), 1)
        request = received[0]
        self.assertEqual(str(request.url), LLM_ENV["CUSTOM_LLM_BASE_URL"] + "/chat/completions")
        self.assertEqual(request.headers["Authorization"], "Bearer " + LLM_ENV["CUSTOM_LLM_API_KEY"])
        payload = json.loads(request.content)
        self.assertEqual(payload["model"], LLM_ENV["CUSTOM_LLM_MODEL_ID"])
        self.assertEqual(payload["max_tokens"], 4096)
        self.assertTrue(payload["stream"])
        self.assertNotIn("enable_thinking", payload)
        self.assertNotIn("chat_template_kwargs", payload)

    async def test_cascaded_runner_body_records_both_overrides_without_secrets(self):
        """Preserve the caller's body while passing effective endpoint metadata to the bot."""
        for mode in ["generic-assistant", "multilingual-assistant"]:
            original = {"pipeline_mode": mode, "session_id": "test-session", "tts_id": "unchanged-tts"}
            args = SimpleNamespace(body=original)
            with (
                patch.dict(os.environ, {**LLM_ENV, **ASR_ENV}, clear=True),
                patch.object(eval_bot, "_install_ci_services"),
                patch.object(eval_bot, "registered_eval_bot", new_callable=AsyncMock) as run,
            ):
                await eval_bot.bot(args)
            run.assert_awaited_once_with(args)
            self.assertNotIn("model_id", original)
            self.assertEqual(args.body["model_id"], LLM_ENV["CUSTOM_LLM_MODEL_ID"])
            self.assertEqual(args.body["asr_model"], ASR_ENV["CUSTOM_ASR_MODEL_ID"])
            self.assertEqual(args.body["tts_id"], "unchanged-tts")
            self.assertNotIn("secret", json.dumps(args.body))

    async def test_omni_body_keeps_multimodal_defaults(self):
        """Keep the documented Omni path separate from custom cascaded endpoints."""
        body = {"pipeline_mode": "omni-assistant", "model_id": "native-omni-model"}
        args = SimpleNamespace(body=body)
        with (
            patch.dict(os.environ, {**LLM_ENV, **ASR_ENV}, clear=True),
            patch.object(eval_bot, "_install_ci_services"),
            patch.object(eval_bot, "registered_eval_bot", new_callable=AsyncMock),
        ):
            await eval_bot.bot(args)
        self.assertEqual(args.body, body)

    def test_invalid_configuration_fails_before_launching_a_scenario(self):
        """Validate the local CLI as well as the workflow before any eval service starts."""
        args = SimpleNamespace(manifest=runner.DEFAULT_MANIFEST, scenario=None)
        with (
            patch.dict(os.environ, {"CUSTOM_LLM_BASE_URL": LLM_ENV["CUSTOM_LLM_BASE_URL"]}, clear=True),
            patch.object(runner, "_parse_args", return_value=args),
            patch.object(runner, "_run_scenario") as run,
        ):
            self.assertEqual(runner.main(), 2)
        run.assert_not_called()

    def test_custom_endpoints_cannot_silently_run_an_unmodified_manifest(self):
        """Reject native entrypoints and Omni-only selections instead of mislabeling results."""
        native = SERVICE_DIR / "generic_manifest.yaml"
        with self.assertRaisesRegex(ValueError, "CI inference_eval_bot"):
            runner._validate_custom_scenarios(native, ["generic_basic_response"])
        hosted = SERVICE_DIR / "cloud_tts_manifest.yaml"
        with self.assertRaisesRegex(ValueError, "Generic or Multilingual"):
            runner._validate_custom_scenarios(hosted, ["omni_basic_response"])
        runner._validate_custom_scenarios(hosted, runner._scenario_names(hosted))

    def test_workflow_exposes_endpoint_inputs_and_separate_secret_roles(self):
        """Keep custom defaults manual-only and validate them before starting Omni NIM."""
        workflow = yaml.safe_load((SERVICE_DIR.parents[3] / ".github/workflows/pipecat-service-evals.yml").read_text())
        job = workflow["jobs"]["pipecat-service-evals"]
        env = job["env"]
        self.assertIn("workflow_dispatch", env["CUSTOM_LLM_BASE_URL"])
        self.assertIn("workflow_dispatch", env["CUSTOM_ASR_SERVER"])
        self.assertIn("secrets[", env["CUSTOM_LLM_API_KEY"])
        self.assertIn("CUSTOM_ASR_API_KEY_ENV", env["CUSTOM_ASR_API_KEY"])
        steps = [step["name"] for step in job["steps"]]
        self.assertLess(steps.index("Validate custom endpoint configuration"), steps.index("Start local Omni NIM"))
