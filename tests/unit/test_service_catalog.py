# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import os
import tempfile
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
)


class ServiceCatalogHydrationTests(unittest.TestCase):
    def setUp(self) -> None:
        # Env-based catalog patches must win; clear any leftover request context.
        clear_service_context()

    def tearDown(self) -> None:
        clear_service_context()

    def test_hydrates_selected_builtin_details_from_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cloud_path = Path(tmpdir) / "services.cloud.yaml"
            cloud_path.write_text(
                """
llm:
  nemotron:
    name: Nemotron
    model_id: catalog-model
    base_url: https://catalog.example/v1
    system_prompt: catalog system
    extra_params: '{"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}'
asr:
  parakeet:
    name: Parakeet
    server: catalog-asr:443
    model: catalog-asr-model
    function_id: catalog-asr-function
    language_code: auto
tts:
  magpie:
    name: Magpie
    server: catalog-tts:443
    function_id: catalog-tts-function
    model: magpie-tts-multilingual
    voice_id: Magpie-Multilingual.EN-US.Aria
    synthesis_mode: stitched
    language_code: en-US
    zero_shot_audio_prompt_file: /data/prompts/clone.wav
""",
                encoding="utf-8",
            )

            config = {
                "llm_id": "cloud-nim:nemotron",
                "model_id": "client-model",
                "base_url": "https://client.example/v1",
                "system_prompt": "client system",
                "extra_params": "{}",
                "asr_id": "cloud-nim:parakeet",
                "asr_server": "client-asr:443",
                "asr_model": "client-asr-model",
                "asr_function_id": "client-asr-function",
                "asr_language_code": "client-asr-language",
                "tts_id": "cloud-nim:magpie",
                "tts_server": "client-tts:443",
                "tts_function_id": "client-tts-function",
                "tts_model": "client-tts-model",
                "tts_voice_id": "client-voice",
                "tts_synthesis_mode": "per_sentence",
            }

            with patch.dict(
                os.environ,
                {
                    "SERVICES_CLOUD_PATH": str(cloud_path),
                    "SERVICES_LOCAL_PATH": str(Path(tmpdir) / "missing-services.local.yaml"),
                },
            ):
                hydrate_config_from_catalog(config)

            self.assertEqual(config["model_id"], "catalog-model")
            self.assertEqual(config["base_url"], "https://catalog.example/v1")
            self.assertEqual(config["system_prompt"], "catalog system")
            # Explicit client reasoning settings must survive catalog hydration.
            self.assertEqual(config["extra_params"], "{}")
            self.assertEqual(config["asr_server"], "catalog-asr:443")
            self.assertEqual(config["asr_model"], "catalog-asr-model")
            self.assertEqual(config["asr_function_id"], "catalog-asr-function")
            self.assertEqual(config["asr_language_code"], "client-asr-language")
            self.assertEqual(config["tts_server"], "catalog-tts:443")
            self.assertEqual(config["tts_function_id"], "catalog-tts-function")
            self.assertEqual(config["tts_model"], "magpie-tts-multilingual")
            self.assertEqual(config["tts_voice_id"], "client-voice")
            self.assertEqual(config["tts_synthesis_mode"], "stitched")
            self.assertEqual(config["tts_language_code"], "en-US")
            self.assertEqual(config["tts_zero_shot_audio_prompt_file"], "/data/prompts/clone.wav")

    def test_talker_and_thinker_temperatures_hydrate_independently(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cloud_path = Path(tmpdir) / "services.cloud.yaml"
            cloud_path.write_text(
                dedent(
                    """\
                    llm:
                      talker:
                        name: Talker
                        model_id: talker-model
                        base_url: https://catalog.example/v1
                        temperature: 0.2
                    thinker-llm:
                      thinker:
                        name: Thinker
                        model_id: thinker-model
                        base_url: https://catalog.example/v1
                        temperature: 0.0
                    """
                ),
                encoding="utf-8",
            )
            config = {
                "llm_id": "cloud-nim:talker",
                "thinker_llm_id": "cloud-nim:thinker",
                "temperature": "0.7",
                "thinker_temperature": "0.9",
            }

            with patch.dict(
                os.environ,
                {
                    "SERVICES_CLOUD_PATH": str(cloud_path),
                    "SERVICES_LOCAL_PATH": str(Path(tmpdir) / "missing-services.local.yaml"),
                },
            ):
                hydrate_config_from_catalog(config)

            self.assertEqual(config["model_id"], "talker-model")
            self.assertEqual(config["thinker_model_id"], "thinker-model")
            self.assertEqual(config["temperature"], "0.7")
            self.assertEqual(config["thinker_temperature"], "0.0")

    def test_chatterbox_hydrates_per_sentence_even_with_sticky_stitched(self) -> None:
        """UI TTS switches must not keep Magpie's stitched mode on Chatterbox."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cloud_path = Path(tmpdir) / "services.cloud.yaml"
            cloud_path.write_text(
                dedent(
                    """\
                    tts:
                      magpie:
                        name: Magpie
                        server: catalog-tts:443
                        model: magpie-tts-multilingual
                        voice_id: Magpie-Multilingual.EN-US.Aria
                        synthesis_mode: stitched
                      chatterbox:
                        name: Chatterbox
                        server: catalog-tts:443
                        model: chatterbox-tts-multilingual
                        voice_id: Chatterbox-Multilingual.en-US.Male
                        synthesis_mode: per_sentence
                    """
                ),
                encoding="utf-8",
            )
            config = {
                "tts_id": "cloud-nim:chatterbox",
                # Leftover from a prior Magpie selection / registry-default body.
                "tts_synthesis_mode": "stitched",
            }
            with patch.dict(
                os.environ,
                {
                    "SERVICES_CLOUD_PATH": str(cloud_path),
                    "SERVICES_LOCAL_PATH": str(Path(tmpdir) / "missing-services.local.yaml"),
                },
            ):
                hydrate_config_from_catalog(config)
            self.assertEqual(config["tts_model"], "chatterbox-tts-multilingual")
            self.assertEqual(config["tts_synthesis_mode"], "per_sentence")

    def test_tts_language_code_client_override_wins_over_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cloud_path = Path(tmpdir) / "services.cloud.yaml"
            cloud_path.write_text(
                dedent(
                    """\
                    tts:
                      magpie:
                        name: Magpie
                        server: catalog-tts:443
                        model: magpie-tts-zeroshot
                        voice_id: Magpie-ZeroShot-Multilingual.Female
                        language_code: en-US
                    """
                ),
                encoding="utf-8",
            )
            config = {
                "tts_id": "cloud-nim:magpie",
                "tts_language_code": "es-US",
            }
            with patch.dict(
                os.environ,
                {
                    "SERVICES_CLOUD_PATH": str(cloud_path),
                    "SERVICES_LOCAL_PATH": str(Path(tmpdir) / "missing-services.local.yaml"),
                },
            ):
                hydrate_config_from_catalog(config)
            self.assertEqual(config["tts_language_code"], "es-US")

    def test_custom_tts_language_code_keeps_only_usable_strings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cloud_path = Path(tmpdir) / "services.cloud.yaml"
            cloud_path.write_text("tts: {}\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "SERVICES_CLOUD_PATH": str(cloud_path),
                    "SERVICES_LOCAL_PATH": str(Path(tmpdir) / "missing-services.local.yaml"),
                },
            ):
                # Custom selections skip hydration, so the raw client value survives.
                for value in ({"evil": 1}, ["en-US"], True, "   "):
                    with self.subTest(value=value):
                        filtered = filter_session_config({"tts_id": "custom-tts", "tts_language_code": value})
                        self.assertNotIn("tts_language_code", filtered)

                filtered = filter_session_config({"tts_id": "custom-tts", "tts_language_code": " en-US "})
                self.assertEqual(filtered["tts_language_code"], "en-US")

    def test_zero_shot_prompt_file_is_catalog_only_not_client_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cloud_path = Path(tmpdir) / "services.cloud.yaml"
            cloud_path.write_text(
                dedent(
                    """\
                    tts:
                      magpie:
                        name: Magpie
                        server: catalog-tts:443
                        function_id: catalog-tts-function
                        model: magpie-tts-zeroshot
                        voice_id: Magpie-ZeroShot-Multilingual.Female
                        zero_shot_audio_prompt_file: /data/prompts/clone.wav
                    """
                ),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "SERVICES_CLOUD_PATH": str(cloud_path),
                    "SERVICES_LOCAL_PATH": str(Path(tmpdir) / "missing-services.local.yaml"),
                },
            ):
                filtered = filter_session_config(
                    {
                        "tts_id": "cloud-nim:magpie",
                        "tts_zero_shot_audio_prompt_file": "/evil/client/path.wav",
                    }
                )

            self.assertEqual(filtered["tts_zero_shot_audio_prompt_file"], "/data/prompts/clone.wav")

    def test_zero_shot_prompt_file_dropped_without_catalog_tts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cloud_path = Path(tmpdir) / "services.cloud.yaml"
            cloud_path.write_text("tts: {}\n", encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "SERVICES_CLOUD_PATH": str(cloud_path),
                    "SERVICES_LOCAL_PATH": str(Path(tmpdir) / "missing-services.local.yaml"),
                },
            ):
                for tts_id in ("", "custom-tts"):
                    with self.subTest(tts_id=tts_id):
                        body = {"tts_zero_shot_audio_prompt_file": "/evil/client/path.wav"}
                        if tts_id:
                            body["tts_id"] = tts_id
                        filtered = filter_session_config(body)
                        self.assertNotIn("tts_zero_shot_audio_prompt_file", filtered)

    def test_hydrates_raw_catalog_key_for_direct_clients(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cloud_path = Path(tmpdir) / "services.cloud.yaml"
            cloud_path.write_text(
                """
llm:
  nemotron:
    name: Nemotron
    model_id: catalog-model
    base_url: https://catalog.example/v1
    system_prompt: ""
    extra_params: '{"extra_body":{"top_k":1}}'
""",
                encoding="utf-8",
            )
            config = {"llm_id": "nemotron"}

            with patch.dict(
                os.environ,
                {
                    "SERVICES_CLOUD_PATH": str(cloud_path),
                    "SERVICES_LOCAL_PATH": str(Path(tmpdir) / "missing-services.local.yaml"),
                },
            ):
                hydrate_config_from_catalog(config)

            self.assertEqual(config["model_id"], "catalog-model")
            self.assertEqual(config["base_url"], "https://catalog.example/v1")
            self.assertEqual(config["extra_params"], '{"extra_body":{"top_k":1}}')

    def test_registry_defaults_fall_back_to_cloud_when_local_endpoint_is_unreachable(self) -> None:
        example = examples_registry._lookup_by_key("generic-assistant")

        with patch("examples_registry.is_endpoint_reachable", return_value=False):
            defaults = examples_registry.metadata(example)["defaults"]

        self.assertEqual(defaults["asr"][0]["id"], "cloud-nim:nemotron-asr-streaming-english")

    def test_registry_defaults_use_cloud_multilingual_when_local_only_default_is_unreachable(self) -> None:
        example = examples_registry._lookup_by_key("multilingual-assistant")

        with patch("examples_registry.is_endpoint_reachable", return_value=False):
            defaults = examples_registry.metadata(example)["defaults"]

        self.assertEqual(defaults["asr"][0]["id"], "cloud-nim:parakeet-rnnt")

    def test_cloud_nemotron_asr_uses_current_english_model_name(self) -> None:
        generic_catalog = utils.load_yaml_file(Path("src/examples/generic/services.cloud.yaml"))
        frontend_backend_catalog = utils.load_yaml_file(Path("src/examples/frontend_backend_agent/services.cloud.yaml"))
        multilingual_catalog = utils.load_yaml_file(Path("src/examples/multilingual/services.cloud.yaml"))

        self.assertEqual(generic_catalog["asr"]["nemotron-asr-streaming-english"]["model"], "nemotron-asr-streaming")
        self.assertEqual(
            frontend_backend_catalog["asr"]["nemotron-asr-streaming-english"]["model"],
            "nemotron-asr-streaming",
        )
        self.assertNotIn("nemotron-asr-streaming-multilingual", multilingual_catalog["asr"])

    def test_multilingual_llms_declare_supported_languages(self) -> None:
        cloud = utils.load_yaml_file(Path("src/examples/multilingual/services.cloud.yaml"))["llm"]
        local = utils.load_yaml_file(Path("src/examples/multilingual/services.local.yaml"))

        lightning_languages = ["en", "de", "es", "fr", "it", "ja"]
        super_languages = [*lightning_languages, "zh"]
        self.assertEqual(cloud["nemotron-lightning"]["supported_languages"], lightning_languages)
        self.assertEqual(cloud["nemotron-lightning-reasoning"]["supported_languages"], lightning_languages)
        self.assertEqual(cloud["nemotron-super"]["supported_languages"], super_languages)
        self.assertEqual(cloud["nemotron-super-reasoning"]["supported_languages"], super_languages)
        self.assertEqual(
            local["server"]["llm"]["nemotron-lightning"]["supported_languages"],
            lightning_languages,
        )
        self.assertEqual(
            local["singlegpu"]["llm"]["nemotron-lightning"]["supported_languages"],
            lightning_languages,
        )

        token = utils._service_context.set((Path("src/examples/multilingual"), ("llm", "asr", "tts")))
        try:
            selected_lightning = load_service_entry_by_id("llm", "cloud-nim:nemotron-lightning")
        finally:
            utils._service_context.reset(token)
        self.assertEqual(selected_lightning["supported_languages"], lightning_languages)

    def test_multilingual_agent_prompt_keys_are_registry_declared(self) -> None:
        unlocked = examples_registry.Selection(
            raw="all",
            locked=False,
            example_keys=tuple(examples_registry.EXAMPLES),
            default_key=next(iter(examples_registry.EXAMPLES)),
        )
        with patch.object(examples_registry, "_SELECTION", unlocked):
            keys = examples_registry.agent_prompt_keys("multilingual-assistant")
        self.assertEqual(
            keys,
            frozenset({"fixed_session_language_addon"}),
        )

    def test_multilingual_default_session_language_is_registry_declared(self) -> None:
        example = examples_registry._lookup_by_key("multilingual-assistant")

        metadata = examples_registry.metadata(example)

        self.assertEqual(metadata["default_session_language"], "de-DE")

    def test_registry_defaults_promote_reachable_local_multilingual_asr(self) -> None:
        example = examples_registry._lookup_by_key("multilingual-assistant")

        with patch("examples_registry.is_endpoint_reachable", return_value=True):
            defaults = examples_registry.metadata(example)["defaults"]

        self.assertEqual(defaults["asr"][0]["id"], "self-hosted:nemotron-asr-streaming-multilingual")
        self.assertEqual(defaults["asr"][0]["model"], "cache-aware-parakeet-rnnt-multi-asr-streaming-sortformer")

    def test_registry_default_uses_reachable_nemo_speech_asr(self) -> None:
        example = examples_registry._lookup_by_key("generic-assistant")

        def reachable(endpoint: str) -> bool:
            return endpoint in {"nemo-speech:50051", "localhost:50051"}

        with patch("examples_registry.is_endpoint_reachable", side_effect=reachable):
            defaults = examples_registry.metadata(example)["defaults"]

        self.assertEqual(defaults["asr"][0]["id"], "self-hosted:nemotron-asr-streaming-english")
        self.assertEqual(defaults["asr"][0]["model"], "nemotron-speech-streaming-en-0.6b")
        self.assertIn(defaults["asr"][0]["server"], {"nemo-speech:50051", "localhost:50051"})

    def test_runtime_default_uses_reachable_nemo_speech_asr(self) -> None:
        token = utils._service_context.set((Path("src/examples/generic"), ("llm", "asr", "tts")))
        try:

            def reachable(endpoint: str) -> bool:
                return endpoint in {"nemo-speech:50051", "localhost:50051"}

            with patch("utils.is_endpoint_reachable", side_effect=reachable):
                default_asr = load_service_entry("asr", "")
                services = build_services_api_response()["asr"]
        finally:
            utils._service_context.reset(token)

        self.assertIn(default_asr["server"], {"nemo-speech:50051", "localhost:50051"})
        self.assertEqual(default_asr["model"], "nemotron-speech-streaming-en-0.6b")
        selected = [entry for entry in services if entry.get("selected")]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["id"], "self-hosted:nemotron-asr-streaming-english")

    def test_server_tts_has_no_duplicate_self_hosted_entry(self) -> None:
        token = utils._service_context.set((Path("src/examples/generic"), ("llm", "asr", "tts")))
        try:

            def reachable(endpoint: str) -> bool:
                return endpoint in {"tts-service:50051", "localhost:50151"}

            with patch("utils.is_endpoint_reachable", side_effect=reachable):
                tts = build_services_api_response()["tts"]
        finally:
            utils._service_context.reset(token)

        self_hosted = [entry for entry in tts if entry["source"] == "self-hosted"]
        self_hosted_ids = {entry["id"] for entry in self_hosted}
        self.assertEqual(
            self_hosted_ids,
            {
                "self-hosted:magpie-multilingual-tts",
                "self-hosted:chatterbox-multilingual-tts",
                "self-hosted:magpie-zeroshot-tts",
            },
        )
        for entry in self_hosted:
            self.assertEqual(entry["server"], "localhost:50151")

    def test_reachable_local_services_win_over_other_recipe_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cloud_path = Path(tmpdir) / "services.cloud.yaml"
            cloud_path.write_text(
                dedent(
                    """\
                    asr:
                      cloud-asr:
                        name: Cloud ASR
                        server: cloud-asr:443
                    """
                ),
                encoding="utf-8",
            )
            local_path = Path(tmpdir) / "services.local.yaml"
            local_path.write_text(
                dedent(
                    """\
                    server:
                      asr:
                        server-asr:
                          name: Server ASR
                          server: nvidia-llm:8000
                    singlegpu:
                      asr:
                        single-gpu-asr:
                          name: Single GPU ASR
                          server: nemo-speech:50051
                    """
                ),
                encoding="utf-8",
            )

            def reachable(endpoint: str) -> bool:
                return endpoint in {"nemo-speech:50051", "localhost:50051"}

            with (
                patch.dict(
                    os.environ,
                    {
                        "SERVICES_CLOUD_PATH": str(cloud_path),
                        "SERVICES_LOCAL_PATH": str(local_path),
                    },
                ),
                patch("utils.is_endpoint_reachable", side_effect=reachable),
            ):
                services = build_services_api_response()["asr"]

        service_ids = {entry["id"] for entry in services}
        self.assertIn("self-hosted:single-gpu-asr", service_ids)
        self.assertNotIn("self-hosted:server-asr", service_ids)

    def test_host_runtime_rewrites_omni_nim_endpoint(self) -> None:
        entry = {"base_url": "http://nvidia-llm-omni:8000/v1"}

        with patch.dict(os.environ, {"APP_RUNTIME": ""}):
            rewritten = utils._rewrite_local_runtime_endpoints({"llm": {"omni": entry}})

        self.assertEqual(rewritten["llm"]["omni"]["base_url"], "http://localhost:18002/v1")

    def test_registry_host_runtime_rewrites_omni_endpoints(self) -> None:
        with patch.dict(os.environ, {"APP_RUNTIME": ""}):
            nim = examples_registry._rewrite_entry_for_host_runtime({"base_url": "http://nvidia-llm-omni:8000/v1"})
            vllm = examples_registry._rewrite_entry_for_host_runtime(
                {"base_url": "http://nvidia-llm-vllm-omni:8002/v1"}
            )

        self.assertEqual(nim["base_url"], "http://localhost:18002/v1")
        self.assertEqual(vllm["base_url"], "http://localhost:8002/v1")


class SessionToolsConfigTests(unittest.TestCase):
    """The additive per-session `tools` selection (default behavior preserved)."""

    def test_tools_list_survives_slot_filtering(self) -> None:
        # Active slots restrict keys, but `tools` is slot-agnostic (like prompt_key).
        token = utils._service_context.set((Path("src/examples/generic"), ("llm", "asr", "tts")))
        try:
            out = filter_session_config(
                {
                    "pipeline_mode": "generic-assistant",
                    "tools_available": "get_weather,calculate_bmi",
                    "bogus": "x",
                }
            )
        finally:
            utils._service_context.reset(token)
        self.assertEqual(out.get("tools_available"), "get_weather,calculate_bmi")
        self.assertNotIn("bogus", out)

    def test_absent_tools_preserves_prompt_default(self) -> None:
        out = filter_session_config({"pipeline_mode": "generic-assistant"})
        self.assertNotIn("tools_available", out)


if __name__ == "__main__":
    unittest.main()
