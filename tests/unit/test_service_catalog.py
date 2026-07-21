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
from utils import build_services_api_response, hydrate_config_from_catalog, load_service_entry


class ServiceCatalogHydrationTests(unittest.TestCase):
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
            self.assertEqual(
                config["extra_params"],
                '{"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}',
            )
            self.assertEqual(config["asr_server"], "catalog-asr:443")
            self.assertEqual(config["asr_model"], "catalog-asr-model")
            self.assertEqual(config["asr_function_id"], "catalog-asr-function")
            self.assertEqual(config["asr_language_code"], "client-asr-language")
            self.assertEqual(config["tts_server"], "catalog-tts:443")
            self.assertEqual(config["tts_function_id"], "catalog-tts-function")
            self.assertEqual(config["tts_model"], "magpie-tts-multilingual")
            self.assertEqual(config["tts_voice_id"], "client-voice")
            self.assertEqual(config["tts_synthesis_mode"], "stitched")

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

    def test_jetson_default_uses_reachable_nemotron_speech_asr(self) -> None:
        example = examples_registry._lookup_by_key("generic-assistant")

        def reachable(endpoint: str) -> bool:
            return endpoint in {"nemotron-speech:50051", "localhost:50051"}

        with patch("examples_registry.is_endpoint_reachable", side_effect=reachable):
            defaults = examples_registry.metadata(example)["defaults"]

        self.assertEqual(defaults["asr"][0]["id"], "self-hosted:parakeet-ctc")
        self.assertIn(defaults["asr"][0]["server"], {"nemotron-speech:50051", "localhost:50051"})

    def test_runtime_default_uses_reachable_nemotron_speech_asr(self) -> None:
        token = utils._service_context.set((Path("src/examples/generic"), ("llm", "asr", "tts")))
        try:

            def reachable(endpoint: str) -> bool:
                return endpoint in {"nemotron-speech:50051", "localhost:50051"}

            with patch("utils.is_endpoint_reachable", side_effect=reachable):
                default_asr = load_service_entry("asr", "")
                services = build_services_api_response()["asr"]
        finally:
            utils._service_context.reset(token)

        self.assertIn(default_asr["server"], {"nemotron-speech:50051", "localhost:50051"})
        self.assertEqual(default_asr["model"], "parakeet-ctc-1.1b-asr")
        selected = [entry for entry in services if entry.get("selected")]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["id"], "self-hosted:parakeet-ctc")

    def test_workstation_tts_has_no_duplicate_self_hosted_entry(self) -> None:
        token = utils._service_context.set((Path("src/examples/generic"), ("llm", "asr", "tts")))
        try:

            def reachable(endpoint: str) -> bool:
                return endpoint in {"tts-service:50051", "localhost:50151"}

            with patch("utils.is_endpoint_reachable", side_effect=reachable):
                tts = build_services_api_response()["tts"]
        finally:
            utils._service_context.reset(token)

        self_hosted = [entry for entry in tts if entry["source"] == "self-hosted"]
        self.assertEqual(len(self_hosted), 1)
        self.assertEqual(self_hosted[0]["id"], "self-hosted:magpie-multilingual-tts")
        self.assertEqual(self_hosted[0]["server"], "localhost:50151")

    def test_runtime_platform_filters_local_services(self) -> None:
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
                    workstation:
                      asr:
                        workstation-asr:
                          name: Workstation ASR
                          server: workstation-asr:50052
                    jetson:
                      asr:
                        jetson-asr:
                          name: Jetson ASR
                          server: jetson-asr:50051
                    """
                ),
                encoding="utf-8",
            )

            with (
                patch.dict(
                    os.environ,
                    {
                        "SERVICES_CLOUD_PATH": str(cloud_path),
                        "SERVICES_LOCAL_PATH": str(local_path),
                        "PLATFORM": "jetsonthor",
                    },
                ),
                patch("utils.is_endpoint_reachable", return_value=True),
            ):
                services = build_services_api_response()["asr"]

        service_ids = {entry["id"] for entry in services}
        self.assertIn("self-hosted:jetson-asr", service_ids)
        self.assertNotIn("self-hosted:workstation-asr", service_ids)


if __name__ == "__main__":
    unittest.main()
