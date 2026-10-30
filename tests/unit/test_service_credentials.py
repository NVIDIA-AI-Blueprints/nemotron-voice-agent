# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103

import io
import os
import unittest
from unittest.mock import Mock, patch

from loguru import logger

import utils


class ServiceCredentialTests(unittest.TestCase):
    def test_explicit_catalog_key_is_independent_of_speech_key_and_is_trimmed(self):
        with (
            patch.dict(os.environ, {"NVIDIA_API_KEY": "speech-key", "TEST_NVA_LLM_KEY": "  hub-key  "}),
            patch("utils.load_service_entry_by_id", return_value={"api_key_env": "TEST_NVA_LLM_KEY"}) as lookup,
        ):
            self.assertEqual(utils.nvidia_llm_api_key("cloud-nim:lightning"), "hub-key")
            self.assertEqual(utils.nvidia_api_key(), "speech-key")
        lookup.assert_called_once_with("llm", "cloud-nim:lightning")

    def test_default_catalog_uses_its_declared_key(self):
        with (
            patch.dict(os.environ, {"NVIDIA_API_KEY": "speech-key", "TEST_NVA_LLM_KEY": "hub-key"}),
            patch("utils.load_service_entry", return_value={"api_key_env": "TEST_NVA_LLM_KEY"}) as lookup,
        ):
            self.assertEqual(utils.nvidia_llm_api_key(), "hub-key")
        lookup.assert_called_once_with("llm", "")

    def test_entries_without_override_preserve_shared_and_local_defaults(self):
        for raw, expected in ((" shared-key ", "shared-key"), ("", "not-needed")):
            with (
                self.subTest(raw=raw),
                patch.dict(os.environ, {"NVIDIA_API_KEY": raw, "TEST_NVA_LLM_KEY": "unused-key"}),
                patch("utils.load_service_entry_by_id", return_value={}),
            ):
                self.assertEqual(utils.nvidia_llm_api_key("custom-llm"), expected)

    def test_explicit_credential_is_bound_to_catalog_endpoints(self):
        entry = {
            "api_key_env": "TEST_NVA_LLM_KEY",
            "base_url": "https://trusted.example/v1",
            "streaming_url": "wss://trusted.example/v1/streaming-session",
        }
        with (
            patch.dict(os.environ, {"TEST_NVA_LLM_KEY": "hub-key"}),
            patch("utils.load_service_entry", return_value=entry),
        ):
            self.assertEqual(utils.nvidia_llm_api_key(base_url="https://trusted.example/v1/"), "hub-key")
            self.assertEqual(utils.nvidia_llm_api_key(base_url="wss://trusted.example/v1/streaming-session"), "hub-key")
            with self.assertRaisesRegex(ValueError, "selected catalog endpoint"):
                utils.nvidia_llm_api_key(base_url="https://client-selected.example/v1")

    def test_missing_explicit_key_never_falls_back_or_logs_other_credentials(self):
        output = io.StringIO()
        sink = logger.add(output)
        try:
            with (
                patch.dict(os.environ, {"NVIDIA_API_KEY": "private-speech-value", "TEST_NVA_LLM_KEY": "  "}),
                patch("utils.load_service_entry", return_value={"api_key_env": "TEST_NVA_LLM_KEY"}),
                self.assertRaisesRegex(RuntimeError, "TEST_NVA_LLM_KEY is not set") as error,
            ):
                utils.nvidia_llm_api_key()
        finally:
            logger.remove(sink)
        self.assertNotIn("private-speech-value", str(error.exception) + output.getvalue())

    def test_invalid_catalog_environment_names_do_not_echo_the_value(self):
        for invalid in (None, "", "bad-name", "possible-private-value", ["KEY"], "KEY\nOTHER"):
            with (
                self.subTest(invalid=invalid),
                patch("utils.load_service_entry", return_value={"api_key_env": invalid}),
                self.assertRaisesRegex(ValueError, "^LLM catalog api_key_env must name an environment variable$"),
            ):
                utils.nvidia_llm_api_key()

    def test_session_cannot_name_a_credential_environment(self):
        with patch("utils.hydrate_config_from_catalog"):
            filtered = utils.filter_session_config(
                {"api_key_env": "PRIVATE_PROCESS_ENV", "llm_api_key_env": "PRIVATE_PROCESS_ENV"}
            )
        self.assertNotIn("api_key_env", filtered)
        self.assertNotIn("llm_api_key_env", filtered)


class GenericCredentialIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_custom_endpoint_without_selection_fails_before_any_provider(self):
        from examples.generic import pipeline

        catalog = {"api_key_env": "TEST_NVA_LLM_KEY", "base_url": "https://trusted.example/v1"}
        with (
            patch.dict(os.environ, {"TEST_NVA_LLM_KEY": "hub-key"}),
            patch("utils.load_service_entry", return_value=catalog),
            patch.object(pipeline, "create_transport"),
            patch.object(pipeline, "create_evaluation_trace", return_value=None),
            patch.object(pipeline, "resolve_prompt", return_value=("custom", "prompt")),
            patch.object(pipeline, "load_service_entry", return_value=catalog),
            patch.object(pipeline, "NvidiaSTTService") as stt,
            patch.object(pipeline, "NvidiaLLMService") as llm,
            patch.object(pipeline.examples_registry, "welcome_message_enabled", return_value=False),
            self.assertRaisesRegex(ValueError, "selected catalog endpoint"),
        ):
            await pipeline.bot(Mock(body={"base_url": "https://client-selected.example/v1"}))
        stt.assert_not_called()
        llm.assert_not_called()

    async def test_live_generic_llm_uses_catalog_key_while_asr_keeps_speech_key(self):
        from examples.generic import pipeline

        class StopAfterLLM(RuntimeError):
            pass

        catalogs = {"llm": {"model_id": "model", "base_url": "http://llm/v1"}, "asr": {}, "tts": {}}
        with (
            patch.dict(os.environ, {"NVIDIA_API_KEY": "speech-key", "TEST_NVA_LLM_KEY": "hub-key"}),
            patch(
                "utils.load_service_entry_by_id",
                return_value={"api_key_env": "TEST_NVA_LLM_KEY", "base_url": "http://llm/v1"},
            ),
            patch.object(pipeline, "create_transport"),
            patch.object(pipeline, "create_evaluation_trace", return_value=None),
            patch.object(pipeline, "resolve_prompt", return_value=("custom", "prompt")),
            patch.object(pipeline, "load_service_entry", side_effect=lambda key, _: catalogs[key]),
            patch.object(pipeline, "NvidiaSTTService") as stt,
            patch.object(pipeline, "NvidiaLLMService", side_effect=StopAfterLLM) as llm,
            patch.object(pipeline.examples_registry, "welcome_message_enabled", return_value=False),
            self.assertRaises(StopAfterLLM),
        ):
            await pipeline.bot(Mock(body={"llm_id": "cloud-nim:lightning"}))
        self.assertEqual(stt.call_args.kwargs["api_key"], "speech-key")
        self.assertEqual(llm.call_args.kwargs["api_key"], "hub-key")

    async def test_missing_catalog_key_fails_before_speech_or_llm_clients_exist(self):
        from examples.generic import pipeline

        with (
            patch.dict(os.environ, {"TEST_NVA_LLM_KEY": ""}),
            patch(
                "utils.load_service_entry_by_id",
                return_value={
                    "api_key_env": "TEST_NVA_LLM_KEY",
                    "base_url": "https://integrate.api.nvidia.com/v1",
                },
            ),
            patch.object(pipeline, "create_transport", new=Mock()),
            patch.object(pipeline, "create_evaluation_trace", return_value=None),
            patch.object(pipeline, "resolve_prompt", return_value=("custom", "prompt")),
            patch.object(pipeline, "load_service_entry", return_value={}),
            patch.object(pipeline, "NvidiaSTTService") as stt,
            patch.object(pipeline, "NvidiaLLMService") as llm,
            patch.object(pipeline.examples_registry, "welcome_message_enabled", return_value=False),
            self.assertRaisesRegex(RuntimeError, "TEST_NVA_LLM_KEY is not set"),
        ):
            await pipeline.bot(Mock(body={"llm_id": "cloud-nim:lightning"}))
        stt.assert_not_called()
        llm.assert_not_called()
