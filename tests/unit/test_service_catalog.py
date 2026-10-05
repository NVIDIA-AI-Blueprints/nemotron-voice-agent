# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import examples_registry
import utils
from utils import (
    build_services_api_response,
    clear_service_context,
    filter_session_config,
    hydrate_config_from_catalog,
    load_service_entry,
    load_service_entry_by_id,
    set_service_context,
)

_CATALOG = dedent(
    """\
    server:
      llm:
        nemotron:
          name: Nemotron
          model_id: catalog-model
          base_url: http://nvidia-llm:8000/v1
          health_path: /v1/health/ready
          system_prompt: catalog system
          extra_params: '{"extra_body":{"repetition_penalty":1.05}}'
          settings: reasoning
          nvcf: {}
        local-only:
          name: Local Only
          model_id: local-model
          base_url: http://local-llm:8000/v1
      asr:
        parakeet:
          name: Parakeet
          server: parakeet-asr:50052
          model: nim-asr-model
          language_code: auto
          nvcf:
            function_id: catalog-asr-function
            model: cloud-asr-model
      tts:
        magpie:
          name: Magpie
          server: magpie-tts:50051
          model: magpie-tts-multilingual
          voice_id: Magpie-Multilingual.EN-US.Aria
          synthesis_mode: stitched
          language_code: en-US
          zero_shot_audio_prompt_file: /data/prompts/clone.wav
          nvcf:
            function_id: catalog-tts-function
        chatterbox:
          name: Chatterbox
          server: chatterbox-tts:50051
          model: chatterbox-tts-multilingual
          voice_id: Chatterbox-Multilingual.en-US.Male
          synthesis_mode: per_sentence
          nvcf:
            function_id: chatterbox-function
    single-gpu:
      llm:
        nemotron:
          name: Nemotron vLLM
          model_id: catalog-model
          base_url: http://nvidia-llm-vllm:8000/v1
          streaming_url: ws://nvidia-llm-vllm:8000/v1/streaming-session
      asr:
        parakeet:
          name: NeMo Speech ASR
          server: nemo-speech:50051
          model: gguf-asr-model
    """
)

_SETTINGS = dedent(
    """\
    reasoning:
      enable_thinking:
        label: Reasoning
        type: bool
        default: false
        path: extra_body.chat_template_kwargs.enable_thinking
      reasoning_budget:
        type: int
        default: 1024
        min: 128
        max: 4096
        path: extra_body.reasoning_budget
        requires: enable_thinking
    """
)


class _CatalogTestCase(unittest.TestCase):
    recipe = "server"
    api_key = "nvapi-test"

    def setUp(self) -> None:
        clear_service_context()
        self._tmpdir = tempfile.TemporaryDirectory()
        services_path = Path(self._tmpdir.name) / "services.yaml"
        services_path.write_text(_CATALOG, encoding="utf-8")
        (Path(self._tmpdir.name) / "settings.yaml").write_text(_SETTINGS, encoding="utf-8")
        self._env = patch.dict(
            os.environ,
            {
                "SERVICES_PATH": str(services_path),
                "SERVICE_RECIPE": self.recipe,
                "NVIDIA_API_KEY": self.api_key,
                "APP_RUNTIME": "container",
            },
        )
        self._env.start()

    def tearDown(self) -> None:
        self._env.stop()
        self._tmpdir.cleanup()
        clear_service_context()


class ServiceCatalogHydrationTests(_CatalogTestCase):
    def test_hydrates_selected_builtin_details_from_catalog(self) -> None:
        config = {
            "llm_id": "self-hosted:nemotron",
            "model_id": "client-model",
            "base_url": "https://client.example/v1",
            "system_prompt": "client system",
            "extra_params": "{}",
            "asr_id": "self-hosted:parakeet",
            "asr_server": "client-asr:443",
            "asr_model": "client-asr-model",
            "asr_function_id": "client-asr-function",
            "asr_language_code": "client-asr-language",
            "tts_id": "self-hosted:magpie",
            "tts_server": "client-tts:443",
            "tts_function_id": "client-tts-function",
            "tts_model": "client-tts-model",
            "tts_voice_id": "client-voice",
            "tts_synthesis_mode": "per_sentence",
        }

        hydrate_config_from_catalog(config)

        self.assertEqual(config["model_id"], "catalog-model")
        self.assertEqual(config["base_url"], "http://nvidia-llm:8000/v1")
        self.assertEqual(config["system_prompt"], "catalog system")
        self.assertEqual(
            json.loads(config["extra_params"]),
            {"extra_body": {"repetition_penalty": 1.05, "chat_template_kwargs": {"enable_thinking": False}}},
        )
        self.assertEqual(config["asr_server"], "parakeet-asr:50052")
        self.assertEqual(config["asr_model"], "nim-asr-model")
        self.assertNotIn("asr_function_id", config)
        self.assertEqual(config["asr_language_code"], "client-asr-language")
        self.assertEqual(config["tts_server"], "magpie-tts:50051")
        self.assertNotIn("tts_function_id", config)
        self.assertEqual(config["tts_model"], "magpie-tts-multilingual")
        self.assertEqual(config["tts_voice_id"], "client-voice")
        self.assertEqual(config["tts_synthesis_mode"], "stitched")
        self.assertEqual(config["tts_language_code"], "en-US")
        self.assertEqual(config["tts_zero_shot_audio_prompt_file"], "/data/prompts/clone.wav")

    def test_hydrates_nvcf_variant_with_overrides(self) -> None:
        config = {"asr_id": "cloud-nim:parakeet", "tts_id": "cloud-nim:magpie", "llm_id": "cloud-nim:nemotron"}

        hydrate_config_from_catalog(config)

        self.assertEqual(config["base_url"], utils.NVCF_LLM_BASE_URL)
        self.assertEqual(config["asr_server"], utils.NVCF_GRPC_SERVER)
        self.assertEqual(config["asr_function_id"], "catalog-asr-function")
        self.assertEqual(config["asr_model"], "cloud-asr-model")
        self.assertEqual(config["tts_server"], utils.NVCF_GRPC_SERVER)
        self.assertEqual(config["tts_function_id"], "catalog-tts-function")
        self.assertEqual(config["tts_voice_id"], "Magpie-Multilingual.EN-US.Aria")

    def test_chatterbox_hydrates_per_sentence_even_with_sticky_stitched(self) -> None:
        config = {"tts_id": "self-hosted:chatterbox", "tts_synthesis_mode": "stitched"}

        hydrate_config_from_catalog(config)

        self.assertEqual(config["tts_model"], "chatterbox-tts-multilingual")
        self.assertEqual(config["tts_synthesis_mode"], "per_sentence")

    def test_tts_language_code_client_override_wins_over_catalog(self) -> None:
        config = {"tts_id": "self-hosted:magpie", "tts_language_code": "es-US"}

        hydrate_config_from_catalog(config)

        self.assertEqual(config["tts_language_code"], "es-US")

    def test_custom_tts_language_code_keeps_only_usable_strings(self) -> None:
        for value in ({"evil": 1}, ["en-US"], True, "   "):
            with self.subTest(value=value):
                filtered = filter_session_config({"tts_id": "custom-tts", "tts_language_code": value})
                self.assertNotIn("tts_language_code", filtered)

        filtered = filter_session_config({"tts_id": "custom-tts", "tts_language_code": " en-US "})
        self.assertEqual(filtered["tts_language_code"], "en-US")

    def test_zero_shot_prompt_file_is_catalog_only_not_client_body(self) -> None:
        filtered = filter_session_config(
            {"tts_id": "self-hosted:magpie", "tts_zero_shot_audio_prompt_file": "/evil/client/path.wav"}
        )

        self.assertEqual(filtered["tts_zero_shot_audio_prompt_file"], "/data/prompts/clone.wav")

    def test_zero_shot_prompt_file_dropped_without_catalog_tts(self) -> None:
        for tts_id in ("", "custom-tts"):
            with self.subTest(tts_id=tts_id):
                body = {"tts_zero_shot_audio_prompt_file": "/evil/client/path.wav"}
                if tts_id:
                    body["tts_id"] = tts_id
                self.assertNotIn("tts_zero_shot_audio_prompt_file", filter_session_config(body))

    def test_hydrates_raw_catalog_key_from_self_hosted_first(self) -> None:
        config = {"llm_id": "nemotron"}

        hydrate_config_from_catalog(config)

        self.assertEqual(config["base_url"], "http://nvidia-llm:8000/v1")

    def test_nvidia_cloud_available_requires_a_real_api_key(self) -> None:
        cases = {"": False, "   ": False, "not-needed": False, "NOT-NEEDED": False, "nvapi-test": True}
        for raw, expected in cases.items():
            with self.subTest(raw=raw), patch.dict(os.environ, {"NVIDIA_API_KEY": raw}):
                self.assertEqual(utils.nvidia_cloud_available(), expected)
        with patch.dict(os.environ):
            os.environ.pop("NVIDIA_API_KEY", None)
            self.assertFalse(utils.nvidia_cloud_available())


class ServiceSettingsTests(_CatalogTestCase):
    def _extra(self, **body: object) -> dict:
        return json.loads(filter_session_config({"llm_id": "self-hosted:nemotron", **body})["extra_params"])

    def test_catalog_entry_carries_schema_and_baked_defaults(self) -> None:
        entry = load_service_entry("llm", "nemotron")

        self.assertEqual(entry["settings"]["enable_thinking"]["type"], "bool")
        self.assertEqual(
            json.loads(entry["extra_params"])["extra_body"]["chat_template_kwargs"], {"enable_thinking": False}
        )
        self.assertNotIn("reasoning_budget", json.loads(entry["extra_params"])["extra_body"])

    def test_entry_extra_params_override_profile_defaults(self) -> None:
        services_path = Path(os.environ["SERVICES_PATH"])
        services_path.write_text(
            _CATALOG.replace(
                """'{"extra_body":{"repetition_penalty":1.05}}'""",
                """'{"extra_body":{"chat_template_kwargs":{"enable_thinking":true},"reasoning_budget":2048}}'""",
                1,
            ),
            encoding="utf-8",
        )

        entry = load_service_entry("llm", "nemotron")

        self.assertIs(entry["settings"]["enable_thinking"]["default"], True)
        self.assertEqual(entry["settings"]["reasoning_budget"]["default"], 2048)
        self.assertEqual(json.loads(entry["extra_params"])["extra_body"]["reasoning_budget"], 2048)
        self.assertNotIn("reasoning_budget", self._extra(llm_settings={"enable_thinking": False})["extra_body"])

    def test_client_settings_write_values_at_their_paths(self) -> None:
        extra = self._extra(llm_settings={"enable_thinking": True, "reasoning_budget": 2048})

        self.assertEqual(extra["extra_body"]["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(extra["extra_body"]["reasoning_budget"], 2048)
        self.assertEqual(extra["extra_body"]["repetition_penalty"], 1.05)

    def test_client_settings_accept_json_strings_and_clamp_ranges(self) -> None:
        extra = self._extra(llm_settings='{"enable_thinking": "true", "reasoning_budget": 99999}')

        self.assertEqual(extra["extra_body"]["reasoning_budget"], 4096)

    def test_requires_drops_dependent_setting(self) -> None:
        extra = self._extra(llm_settings={"enable_thinking": False, "reasoning_budget": 2048})

        self.assertNotIn("reasoning_budget", extra["extra_body"])

    def test_invalid_and_unknown_settings_fall_back_to_defaults(self) -> None:
        extra = self._extra(llm_settings={"enable_thinking": "yes", "top_p": 0.1})

        self.assertEqual(
            extra["extra_body"], {"repetition_penalty": 1.05, "chat_template_kwargs": {"enable_thinking": False}}
        )

    def test_settings_never_reach_the_pipeline(self) -> None:
        for llm_id in ("self-hosted:nemotron", "self-hosted:local-only", "custom-llm"):
            with self.subTest(llm_id=llm_id):
                filtered = filter_session_config({"llm_id": llm_id, "llm_settings": {"enable_thinking": True}})
                self.assertNotIn("llm_settings", filtered)

    def test_sampling_settings_are_sent_only_when_set(self) -> None:
        services_path = Path(os.environ["SERVICES_PATH"])
        (services_path.with_name("settings.yaml")).write_text(
            _SETTINGS + "  temperature:\n    type: float\n    min: 0.0\n    max: 1.0\n",
            encoding="utf-8",
        )

        self.assertNotIn("temperature", self._extra())
        self.assertEqual(self._extra(llm_settings={"temperature": 1.7})["temperature"], 1.0)
        self.assertNotIn("temperature", self._extra(llm_settings='{"temperature": NaN}'))

    def test_example_settings_override_defaults_for_a_slot_that_reads_llm(self) -> None:
        set_service_context(
            {
                "services": {"llm": ["nemotron"], "thinker-llm": ["nemotron"]},
                "categories": {"thinker-llm": "llm"},
                "settings": {"thinker-llm": {"nemotron": {"enable_thinking": True, "reasoning_budget": 512}}},
            }
        )

        thinker = load_service_entry("thinker-llm", "")
        talker = load_service_entry("llm", "")

        self.assertEqual(thinker["settings"]["reasoning_budget"]["default"], 512)
        self.assertEqual(json.loads(thinker["extra_params"])["extra_body"]["reasoning_budget"], 512)
        self.assertNotIn("reasoning_budget", json.loads(talker["extra_params"])["extra_body"])
        hydrated = filter_session_config({"thinker_llm_id": "self-hosted:nemotron"})
        self.assertEqual(json.loads(hydrated["thinker_extra_params"])["extra_body"]["reasoning_budget"], 512)

    def test_example_settings_apply_only_to_their_service(self) -> None:
        set_service_context(
            {
                "services": {"llm": ["nemotron", "local-only"]},
                "settings": {"llm": {"nemotron": {"enable_thinking": True}}},
            }
        )

        self.assertIs(load_service_entry("llm", "nemotron")["settings"]["enable_thinking"]["default"], True)
        self.assertNotIn("settings", load_service_entry("llm", "local-only"))

    def test_unknown_example_setting_raises(self) -> None:
        set_service_context({"services": {"llm": ["nemotron"]}, "settings": {"llm": {"nemotron": {"top_k": 1}}}})

        with self.assertRaisesRegex(RuntimeError, "top_k"):
            load_service_entry("llm", "")

    def test_shared_anchor_keys_are_not_profiles(self) -> None:
        services_path = Path(os.environ["SERVICES_PATH"])
        (services_path.with_name("settings.yaml")).write_text("x-shared: &shared {}\n" + _SETTINGS, encoding="utf-8")

        self.assertIn("enable_thinking", load_service_entry("llm", "nemotron")["settings"])

    def test_unknown_settings_profile_raises(self) -> None:
        services_path = Path(os.environ["SERVICES_PATH"])
        services_path.write_text(_CATALOG.replace("settings: reasoning", "settings: missing"), encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "unknown settings 'missing'"):
            load_service_entry("llm", "nemotron")

    def test_missing_catalog_files_raise(self) -> None:
        services_path = Path(os.environ["SERVICES_PATH"])
        services_path.with_name("settings.yaml").unlink()

        with self.assertRaisesRegex(RuntimeError, "settings.yaml"):
            load_service_entry("llm", "nemotron")

        services_path.write_text("server: [", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Failed to load YAML"):
            load_service_entry("llm", "nemotron")


class SpeechSettingsTests(_CatalogTestCase):
    def setUp(self) -> None:
        super().setUp()
        services_path = Path(os.environ["SERVICES_PATH"])
        services_path.write_text(
            _CATALOG.replace("synthesis_mode: stitched", "settings: tts").replace(
                "model: nim-asr-model\n", "model: nim-asr-model\n      settings: asr\n"
            ),
            encoding="utf-8",
        )
        services_path.with_name("settings.yaml").write_text(
            _SETTINGS
            + dedent(
                """\
                tts:
                  synthesis_mode:
                    type: enum
                    options: [stitched, per_sentence]
                    default: stitched
                asr:
                  automatic_punctuation:
                    type: bool
                    default: true
                """
            ),
            encoding="utf-8",
        )

    def test_settings_write_the_session_field_of_their_entry_field(self) -> None:
        cases = {
            "": ("stitched", "True"),
            '{"synthesis_mode": "per_sentence"}': ("per_sentence", "True"),
            '{"synthesis_mode": "fast", "automatic_punctuation": false}': ("stitched", "False"),
        }
        for raw, (mode, punctuation) in cases.items():
            with self.subTest(raw=raw):
                config = filter_session_config(
                    {
                        "tts_id": "self-hosted:magpie",
                        "tts_settings": raw,
                        "asr_id": "self-hosted:parakeet",
                        "asr_settings": raw,
                    }
                )
                self.assertEqual(config["tts_synthesis_mode"], mode)
                self.assertEqual(config["asr_automatic_punctuation"], punctuation)
                self.assertNotIn("tts_settings", config)

    def test_speech_setting_without_a_session_field_raises(self) -> None:
        settings_path = Path(os.environ["SERVICES_PATH"]).with_name("settings.yaml")
        settings_path.write_text(
            settings_path.read_text(encoding="utf-8") + "  speed:\n    type: float\n", encoding="utf-8"
        )

        with self.assertRaisesRegex(RuntimeError, "unknown to asr"):
            load_service_entry("asr", "parakeet")


class StreamingInputTests(_CatalogTestCase):
    recipe = "single-gpu"
    api_key = ""

    def _base_url(self, capabilities: list[str], **body: object) -> str:
        set_service_context({"services": {"llm": ["nemotron"]}, "capabilities": capabilities})
        config = filter_session_config({"llm_id": "self-hosted:nemotron", **body})
        self.assertNotIn("llm_streaming", config)
        return config["base_url"]

    def test_streaming_uses_the_streaming_url_when_the_example_supports_it(self) -> None:
        base_url = self._base_url(["streaming_input"], llm_streaming="true")

        self.assertEqual(base_url, "ws://nvidia-llm-vllm:8000/v1/streaming-session")

    def test_streaming_is_ignored_without_the_capability(self) -> None:
        self.assertEqual(self._base_url([], llm_streaming="true"), "http://nvidia-llm-vllm:8000/v1")

    def test_streaming_off_keeps_the_http_url(self) -> None:
        self.assertEqual(self._base_url(["streaming_input"], llm_streaming="false"), "http://nvidia-llm-vllm:8000/v1")
        self.assertEqual(self._base_url(["streaming_input"]), "http://nvidia-llm-vllm:8000/v1")

    def test_host_runtime_rewrites_the_streaming_url(self) -> None:
        with patch.dict(os.environ, {"APP_RUNTIME": ""}):
            base_url = self._base_url(["streaming_input"], llm_streaming="true")

        self.assertEqual(base_url, "ws://localhost:18000/v1/streaming-session")

    def test_stored_session_keeps_streaming_and_settings_until_the_bot_starts(self) -> None:
        import server

        set_service_context({"services": {"llm": ["nemotron"]}, "capabilities": ["streaming_input"]})
        config = filter_session_config(
            {"llm_id": "self-hosted:nemotron", "llm_streaming": "true", "llm_settings": {"enable_thinking": True}}
        )
        resolved = server._resolve_config(server._store_session_config(config))

        self.assertEqual(resolved["base_url"], "ws://nvidia-llm-vllm:8000/v1/streaming-session")
        self.assertEqual(resolved, config)


class ServiceRecipeTests(_CatalogTestCase):
    def _ids(self, category: str) -> list[str]:
        return [entry["id"] for entry in build_services_api_response()[category]]

    def test_server_recipe_lists_self_hosted_then_nvcf_entries(self) -> None:
        self.assertEqual(self._ids("llm"), ["self-hosted:nemotron", "self-hosted:local-only", "cloud-nim:nemotron"])
        selected = [entry["id"] for entry in build_services_api_response()["llm"] if entry["selected"]]
        self.assertEqual(selected, ["self-hosted:nemotron"])

    def test_server_recipe_without_key_hides_nvcf_entries(self) -> None:
        with patch.dict(os.environ, {"NVIDIA_API_KEY": ""}):
            self.assertEqual(self._ids("llm"), ["self-hosted:nemotron", "self-hosted:local-only"])

    def test_cloud_recipe_uses_only_nvcf_entries(self) -> None:
        with patch.dict(os.environ, {"SERVICE_RECIPE": "cloud"}):
            self.assertEqual(self._ids("llm"), ["cloud-nim:nemotron"])
            self.assertEqual(self._ids("tts"), ["cloud-nim:magpie", "cloud-nim:chatterbox"])

    def test_cloud_recipe_without_key_is_empty(self) -> None:
        with patch.dict(os.environ, {"SERVICE_RECIPE": "cloud", "NVIDIA_API_KEY": ""}):
            self.assertEqual(build_services_api_response(), {})

    def test_single_gpu_recipe_uses_its_own_section(self) -> None:
        with patch.dict(os.environ, {"SERVICE_RECIPE": "single-gpu", "NVIDIA_API_KEY": ""}):
            self.assertEqual(self._ids("llm"), ["self-hosted:nemotron"])
            self.assertEqual(load_service_entry("asr", "")["model"], "gguf-asr-model")

    def test_unknown_recipe_raises(self) -> None:
        with patch.dict(os.environ, {"SERVICE_RECIPE": "gpu"}), self.assertRaisesRegex(RuntimeError, "SERVICE_RECIPE"):
            build_services_api_response()

    def test_example_services_filter_and_order_catalog(self) -> None:
        set_service_context({"services": {"llm": ["local-only", "nemotron"], "tts": ["chatterbox"]}})

        services = build_services_api_response()

        self.assertEqual(list(services), ["llm", "tts"])
        self.assertEqual(
            [entry["id"] for entry in services["llm"]],
            ["self-hosted:local-only", "self-hosted:nemotron", "cloud-nim:nemotron"],
        )
        self.assertEqual(load_service_entry("llm", "")["model_id"], "local-model")
        self.assertEqual(load_service_entry_by_id("tts", "self-hosted:magpie"), {})

    def test_host_runtime_rewrites_self_hosted_endpoints_only(self) -> None:
        with patch.dict(os.environ, {"APP_RUNTIME": ""}):
            self.assertEqual(
                load_service_entry_by_id("llm", "self-hosted:nemotron")["base_url"], "http://localhost:18000/v1"
            )
            self.assertEqual(load_service_entry_by_id("llm", "cloud-nim:nemotron")["base_url"], utils.NVCF_LLM_BASE_URL)

    def test_llm_health_url_follows_the_catalog_health_path_on_the_runtime_host(self) -> None:
        import server

        self.assertEqual(server._llm_health_url("http://nvidia-llm:8000/v1"), "http://nvidia-llm:8000/v1/health/ready")
        with patch.dict(os.environ, {"APP_RUNTIME": ""}):
            self.assertEqual(
                server._llm_health_url("http://localhost:18000/v1"), "http://localhost:18000/v1/health/ready"
            )
        self.assertEqual(server._llm_health_url(utils.NVCF_LLM_BASE_URL), "")
        self.assertEqual(server._llm_health_url("http://my-llm:9000/v1"), "")


class AutoRecipeTests(_CatalogTestCase):
    recipe = ""
    api_key = ""

    def _catalog_with(self, reachable: set[str]) -> dict:
        with patch("utils.is_endpoint_reachable", side_effect=lambda endpoint: endpoint in reachable):
            return build_services_api_response()

    def test_picks_section_with_most_reachable_entries(self) -> None:
        services = self._catalog_with({"http://nvidia-llm-vllm:8000/v1", "nemo-speech:50051"})

        self.assertEqual(services["llm"][0]["base_url"], "http://nvidia-llm-vllm:8000/v1")
        self.assertEqual(services["asr"][0]["model"], "gguf-asr-model")

    def test_lists_only_reachable_entries_in_a_slot(self) -> None:
        services = self._catalog_with({"http://nvidia-llm:8000/v1", "magpie-tts:50051"})

        self.assertEqual([entry["id"] for entry in services["llm"]], ["self-hosted:nemotron"])
        self.assertEqual([entry["id"] for entry in services["tts"]], ["self-hosted:magpie"])

    def test_keeps_slot_entries_when_none_is_reachable(self) -> None:
        services = self._catalog_with({"http://nvidia-llm:8000/v1"})

        self.assertEqual([entry["id"] for entry in services["asr"]], ["self-hosted:parakeet"])

    def test_nothing_reachable_uses_nvidia_cloud_when_available(self) -> None:
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-test"}):
            services = self._catalog_with(set())

        self.assertTrue(all(entry["source"] == "cloud-nim" for entries in services.values() for entry in entries))

    def test_probes_run_in_parallel(self) -> None:
        barrier = threading.Barrier(2, timeout=2)

        def probe(endpoint: str) -> bool:
            if endpoint in {"http://nvidia-llm:8000/v1", "http://nvidia-llm-vllm:8000/v1"}:
                barrier.wait()
            return False

        with patch("utils.is_endpoint_reachable", side_effect=probe):
            build_services_api_response()


class RepositoryCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_service_context()

    def tearDown(self) -> None:
        clear_service_context()

    def test_every_example_has_a_default_for_every_recipe(self) -> None:
        cases = (
            ("cloud", "nvapi-test", "cloud-nim:"),
            ("server", "", "self-hosted:"),
            ("single-gpu", "", "self-hosted:"),
        )
        for recipe, api_key, prefix in cases:
            with patch.dict(os.environ, {"SERVICE_RECIPE": recipe, "NVIDIA_API_KEY": api_key}):
                for key in examples_registry.EXAMPLES:
                    with self.subTest(recipe=recipe, example=key):
                        defaults = examples_registry.metadata(examples_registry._lookup_by_key(key))["defaults"]
                        for slot in examples_registry.EXAMPLES[key]["services"]:
                            self.assertTrue(defaults[slot][0]["id"].startswith(prefix))

    def test_cloud_recipe_without_api_key_raises_clear_error(self) -> None:
        example = examples_registry._lookup_by_key("generic-assistant")

        with (
            patch.dict(os.environ, {"SERVICE_RECIPE": "cloud", "NVIDIA_API_KEY": ""}),
            self.assertRaisesRegex(RuntimeError, "NVIDIA_API_KEY"),
        ):
            examples_registry.metadata(example)

    def test_registry_services_reference_catalog_keys(self) -> None:
        catalog = utils.load_yaml_file(Path("services.yaml"))
        for key, example in examples_registry.EXAMPLES.items():
            for slot, service_keys in example["services"].items():
                for service_key in service_keys:
                    with self.subTest(example=key, slot=slot, service=service_key):
                        category = example["categories"].get(slot, slot)
                        self.assertTrue(
                            any(
                                service_key in (catalog[recipe].get(category) or {})
                                for recipe in ("server", "single-gpu")
                            )
                        )

    def test_registry_rejects_categories_outside_the_slot_list(self) -> None:
        registry = {"examples": {"x": {"label": "X", "bot": "m:b", "services": {"llm": ["a"], "thinker-llm": ["a"]}}}}
        for bad in ({"planner-llm": "llm"}, {"thinker-llm": "vlm"}):
            with self.subTest(bad=bad):
                registry["examples"]["x"]["categories"] = bad
                with self.assertRaisesRegex(RuntimeError, "categories"):
                    examples_registry._load_examples(registry)

    def test_slot_without_category_reads_the_same_named_category(self) -> None:
        set_service_context({"services": {"llm": ["nemotron"], "thinker-llm": ["nemotron"]}})

        self.assertEqual(load_service_entry("thinker-llm", ""), {})

    def test_slots_follow_services_order(self) -> None:
        example = examples_registry._lookup_by_key("frontend-backend-agent")

        with patch.dict(os.environ, {"SERVICE_RECIPE": "single-gpu"}):
            self.assertEqual(examples_registry.metadata(example)["slots"], ["llm", "thinker-llm", "asr", "tts"])

    def test_multilingual_defaults_prefer_self_hosted_nemotron_asr(self) -> None:
        example = examples_registry._lookup_by_key("multilingual-assistant")
        for recipe, model in (
            ("server", "cache-aware-parakeet-rnnt-multi-asr-streaming-sortformer"),
            ("single-gpu", "nemotron-3.5-asr-streaming-0.6b"),
        ):
            with self.subTest(recipe=recipe), patch.dict(os.environ, {"SERVICE_RECIPE": recipe, "NVIDIA_API_KEY": ""}):
                default_asr = examples_registry.metadata(example)["defaults"]["asr"][0]
                self.assertEqual(default_asr["id"], "self-hosted:nemotron-asr-streaming-multilingual")
                self.assertEqual(default_asr["model"], model)

    def test_multilingual_falls_back_to_cloud_parakeet_on_cloud_recipe(self) -> None:
        example = examples_registry._lookup_by_key("multilingual-assistant")

        with patch.dict(os.environ, {"SERVICE_RECIPE": "cloud", "NVIDIA_API_KEY": "nvapi-test"}):
            default_asr = examples_registry.metadata(example)["defaults"]["asr"][0]

        self.assertEqual(default_asr["id"], "cloud-nim:parakeet-rnnt")

    def test_cloud_nemotron_asr_uses_current_english_model_name(self) -> None:
        set_service_context(examples_registry.EXAMPLES["generic-assistant"])
        with patch.dict(os.environ, {"SERVICE_RECIPE": "cloud", "NVIDIA_API_KEY": "nvapi-test"}):
            entry = load_service_entry_by_id("asr", "cloud-nim:nemotron-asr-streaming-english")

        self.assertEqual(entry["model"], "nemotron-asr-streaming")
        self.assertEqual(entry["server"], utils.NVCF_GRPC_SERVER)

    def test_lightning_entries_declare_supported_languages(self) -> None:
        catalog = utils.load_yaml_file(Path("services.yaml"))
        lightning_languages = ["en", "de", "es", "fr", "it", "ja"]
        for recipe in ("server", "single-gpu"):
            with self.subTest(recipe=recipe):
                self.assertEqual(
                    catalog[recipe]["llm"]["nemotron-lightning"]["supported_languages"], lightning_languages
                )

    def test_lightning_budget_key_follows_the_serving_engine(self) -> None:
        example = examples_registry.EXAMPLES["frontend-backend-agent"]
        set_service_context(example)
        for recipe, budget_key in (("server", "reasoning_budget"), ("single-gpu", "thinking_token_budget")):
            with self.subTest(recipe=recipe), patch.dict(os.environ, {"SERVICE_RECIPE": recipe, "NVIDIA_API_KEY": ""}):
                thinker = json.loads(load_service_entry("thinker-llm", "")["extra_params"])
                self.assertEqual(thinker["extra_body"]["chat_template_kwargs"], {"enable_thinking": True})
                self.assertEqual(thinker["extra_body"][budget_key], 1024)
                talker = json.loads(load_service_entry("llm", "")["extra_params"])
                self.assertEqual(talker["extra_body"]["chat_template_kwargs"], {"enable_thinking": False})

    def test_multilingual_agent_prompt_keys_are_registry_declared(self) -> None:
        unlocked = examples_registry.Selection(
            raw="all",
            locked=False,
            example_keys=tuple(examples_registry.EXAMPLES),
            default_key=next(iter(examples_registry.EXAMPLES)),
        )
        with patch.object(examples_registry, "_SELECTION", unlocked):
            keys = examples_registry.agent_prompt_keys("multilingual-assistant")
        self.assertEqual(keys, frozenset({"fixed_session_language_addon"}))

    def test_multilingual_default_session_language_is_registry_declared(self) -> None:
        example = examples_registry._lookup_by_key("multilingual-assistant")

        with patch.dict(os.environ, {"SERVICE_RECIPE": "server"}):
            metadata = examples_registry.metadata(example)

        self.assertEqual(metadata["default_session_language"], "de-DE")


class HostRuntimeRewriteTests(unittest.TestCase):
    def _rewrite(self, field: str, value: str) -> str:
        with patch.dict(os.environ, {"APP_RUNTIME": ""}):
            return utils._rewrite_local_runtime_endpoints({"x": {"y": {field: value}}})["x"]["y"][field]

    def test_host_runtime_rewrites_compose_endpoints(self) -> None:
        cases = {
            ("base_url", "http://nvidia-llm-omni:8000/v1"): "http://localhost:18002/v1",
            ("base_url", "http://nvidia-llm-vllm-omni:8002/v1"): "http://localhost:8002/v1",
            ("base_url", "ws://nvidia-llm-vllm:8000/v1/streaming-session"): "ws://localhost:18000/v1/streaming-session",
            ("base_url", "http://nemotron-3-super:8000/v1"): "http://localhost:18001/v1",
            ("server", "magpie-multilingual-tts-service:50051"): "localhost:50151",
            ("server", "parakeet-ctc-asr:50052"): "localhost:50352",
            ("server", "nemo-speech:50051"): "localhost:50051",
            ("server", "grpc.nvcf.nvidia.com:443"): "grpc.nvcf.nvidia.com:443",
        }
        for (field, value), expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(self._rewrite(field, value), expected)

    def test_speech_sidecars_publish_unique_ports_matching_the_rewrite_table(self) -> None:
        compose_services: dict[str, dict] = {}
        for path in (utils.PROJECT_ROOT / "docker").glob("docker-compose.*.yaml"):
            compose_services.update(utils.load_yaml_file(path).get("services") or {})

        host_ports: list[int] = []
        for services in utils.LOCAL_SPEECH_PORTS.values():
            for name, ports in services.items():
                published = {tuple(map(int, str(port).split(":"))) for port in compose_services[name]["ports"]}
                with self.subTest(service=name):
                    self.assertEqual(published, {(ports.host_grpc, ports.grpc), (ports.host_health, ports.health)})
                host_ports += [ports.host_grpc, ports.host_health]
        self.assertEqual(len(host_ports), len(set(host_ports)))


if __name__ == "__main__":
    unittest.main()
