# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

from __future__ import annotations

import asyncio
import copy
import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import yaml

import examples_registry
import server

_GENERIC_MODEL = "nvidia/nemotron-realtime"
_CLIENT_TOOLS_MODEL = "nvidia/nemotron-realtime-client-tools"
_OMNI_MODEL = "nvidia/nemotron-realtime-omni"


class RealtimeModelProfileTests(unittest.TestCase):
    def _registry(self) -> dict:
        return copy.deepcopy(examples_registry._REGISTRY_DATA)

    def _load(
        self,
        registry: dict,
        examples: dict[str, examples_registry.ExampleEntry] | None = None,
    ) -> dict[str, examples_registry.RealtimeModelProfile]:
        with patch.object(
            examples_registry,
            "is_endpoint_reachable",
            side_effect=AssertionError("static profile validation must not probe endpoints"),
        ):
            return examples_registry._load_realtime_model_profiles(
                registry,
                examples if examples is not None else examples_registry.EXAMPLES,
            )

    def test_checked_in_profiles_have_one_default_per_pipeline(self) -> None:
        profiles = self._load(self._registry())
        defaults: dict[str, list[str]] = {}
        for model, profile in profiles.items():
            if profile["default"]:
                defaults.setdefault(profile["pipeline_mode"], []).append(model)

        self.assertEqual(set(defaults), set(examples_registry.EXAMPLES))
        self.assertTrue(all(len(models) == 1 for models in defaults.values()))

    def test_generic_full_and_client_tool_profiles_are_distinct_complete_routes(self) -> None:
        full = examples_registry.resolve_realtime_model_profile(_GENERIC_MODEL)
        client_tools = examples_registry.resolve_realtime_model_profile(_CLIENT_TOOLS_MODEL)

        self.assertEqual(full["pipeline_mode"], "generic-assistant")
        self.assertEqual(client_tools["pipeline_mode"], "generic-assistant")
        self.assertTrue(full["default"])
        self.assertFalse(client_tools["default"])
        self.assertEqual(full["selectors"]["prompt_key"], "generic_assistant")
        self.assertEqual(client_tools["selectors"]["prompt_key"], "generic_assistant_without_tools")

    def test_every_checked_in_service_selector_uses_explicit_registry_default_indirection(self) -> None:
        for profile in examples_registry.realtime_model_profiles().values():
            for selector_name, selector_value in profile["selectors"].items():
                if selector_name == "prompt_key":
                    self.assertNotEqual(selector_value, examples_registry.REALTIME_REGISTRY_DEFAULT_SELECTOR)
                else:
                    self.assertEqual(selector_value, examples_registry.REALTIME_REGISTRY_DEFAULT_SELECTOR)

        multilingual = examples_registry.realtime_model_profiles()["nvidia/nemotron-realtime-multilingual"]
        self.assertEqual(multilingual["platform_overrides"], {"cloud": {"asr_id": "parakeet-rnnt"}})

    def test_materialization_is_platform_qualified_and_never_probes_reachability(self) -> None:
        profile = examples_registry.resolve_realtime_model_profile(_GENERIC_MODEL)
        with patch(
            "examples_registry.is_endpoint_reachable",
            side_effect=AssertionError("Realtime materialization must not probe endpoints"),
        ):
            for platform, expected in (
                ("cloud", "cloud-nim:nemotron-lightning"),
                ("server", "self-hosted:server:nemotron-lightning"),
                ("singlegpu", "self-hosted:singlegpu:nemotron-lightning"),
            ):
                with self.subTest(platform=platform):
                    selectors = examples_registry.materialize_realtime_profile_selectors(profile, platform)
                    self.assertEqual(selectors["llm_id"], expected)

    def test_sanitizing_materialized_route_never_loads_reachability_selected_defaults(self) -> None:
        with patch.dict(os.environ, {"REALTIME_SERVICE_PLATFORM": "singlegpu"}):
            route = server._resolve_realtime_model_route(None, {}, "generic-assistant")
            expected_llm = server.load_service_entry_by_id("llm", route.runtime_config["llm_id"])
            expected_tts = server.load_service_entry_by_id("tts", route.runtime_config["tts_id"])
            with (
                patch.object(
                    server,
                    "load_service_entry",
                    side_effect=AssertionError("Realtime setup must not load unqualified defaults"),
                ),
                patch.object(
                    examples_registry,
                    "is_endpoint_reachable",
                    side_effect=AssertionError("Realtime setup must not probe service endpoints"),
                ),
            ):
                sanitized = server._sanitize_realtime_config(
                    route.runtime_config,
                    fallback_example_key="generic-assistant",
                )

        self.assertEqual(sanitized["llm_id"], "self-hosted:singlegpu:nemotron-lightning")
        self.assertEqual(
            sanitized["asr_id"],
            "self-hosted:singlegpu:nemotron-asr-streaming-english",
        )
        self.assertEqual(sanitized["tts_id"], "self-hosted:singlegpu:magpie-multilingual-tts")
        self.assertEqual(sanitized["prompt_key"], "generic_assistant")
        self.assertEqual(sanitized["base_url"], expected_llm["base_url"])
        self.assertEqual(sanitized["tts_server"], expected_tts["server"])

    def test_realtime_readiness_uses_the_materialized_route_without_default_discovery(self) -> None:
        config = {
            "base_url": "http://nvidia-llm-vllm:8000/v1",
            "model_id": "nvidia/nemotron-3.5-lightning-30b-a3b",
            "asr_server": "nemo-speech:50051",
            "tts_server": "nemo-speech:50051",
            "tts_voice_id": "John",
            "tts_function_id": "",
            "tts_model": "magpie-tts-multilingual",
            "output_modalities": ["audio"],
        }
        example = examples_registry._lookup_by_key("generic-assistant")
        http_readiness = AsyncMock()
        port_readiness = AsyncMock()

        with (
            patch.object(
                server,
                "_get_default_llm_selection",
                side_effect=AssertionError("explicit Realtime LLM route must not load defaults"),
            ),
            patch.object(
                server,
                "_get_default_tts_selection",
                side_effect=AssertionError("explicit Realtime TTS route must not load defaults"),
            ),
            patch.object(server, "_run_http_readiness_check", http_readiness),
            patch.object(server, "_run_blocking", port_readiness),
        ):
            asyncio.run(
                server._ensure_services_ready_for_connection(
                    config,
                    example,
                    is_realtime=True,
                )
            )

        http_readiness.assert_awaited_once()
        self.assertEqual(http_readiness.await_args.args[:2], ("LLM", config["base_url"]))
        self.assertEqual(port_readiness.await_count, 2)
        self.assertEqual(
            [call.args[1:3] for call in port_readiness.await_args_list],
            [("ASR", config["asr_server"]), ("TTS", config["tts_server"])],
        )

    def test_multilingual_profile_selects_parakeet_only_in_cloud(self) -> None:
        profile = examples_registry.resolve_realtime_model_profile("nvidia/nemotron-realtime-multilingual")
        for platform, expected in (
            ("cloud", "cloud-nim:parakeet-rnnt"),
            ("server", "self-hosted:server:nemotron-asr-streaming-multilingual"),
            ("singlegpu", "self-hosted:singlegpu:nemotron-asr-streaming-multilingual"),
        ):
            with self.subTest(platform=platform):
                selectors = examples_registry.materialize_realtime_profile_selectors(profile, platform)
                self.assertEqual(selectors["asr_id"], expected)

    def test_trusted_service_overrides_canonicalize_only_on_explicit_platform(self) -> None:
        cases = (
            ("nemotron-lightning", "server", "self-hosted:server:nemotron-lightning"),
            ("self-hosted:nemotron-lightning", "server", "self-hosted:server:nemotron-lightning"),
            (
                "self-hosted:singlegpu:nemotron-lightning",
                "singlegpu",
                "self-hosted:singlegpu:nemotron-lightning",
            ),
            ("cloud-nim:nemotron-lightning", "cloud", "cloud-nim:nemotron-lightning"),
        )
        for selector, platform, expected in cases:
            with self.subTest(selector=selector, platform=platform):
                self.assertEqual(
                    examples_registry.canonicalize_realtime_service_selector(
                        "generic-assistant",
                        "llm_id",
                        selector,
                        platform,
                    ),
                    expected,
                )

        for selector, platform in (
            ("cloud-nim:nemotron-lightning", "server"),
            ("self-hosted:nemotron-lightning", "cloud"),
            ("self-hosted:server:nemotron-lightning", "singlegpu"),
            ("missing-llm", "cloud"),
        ):
            with self.subTest(selector=selector, platform=platform), self.assertRaises(RuntimeError):
                examples_registry.canonicalize_realtime_service_selector(
                    "generic-assistant",
                    "llm_id",
                    selector,
                    platform,
                )

    def test_realtime_service_platform_is_explicit_and_fail_closed(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(examples_registry.realtime_service_platform(), "cloud")
        with patch.dict(os.environ, {"REALTIME_SERVICE_PLATFORM": "server"}):
            self.assertEqual(examples_registry.realtime_service_platform(), "server")
        self.assertEqual(examples_registry.realtime_service_platform(" SINGLEGPU "), "singlegpu")
        for value in ("auto", "local", "reachable"):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "must be one of"):
                examples_registry.realtime_service_platform(value)

    def test_compose_pins_the_catalog_platform_for_every_recipe(self) -> None:
        compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
        expected = {
            "generic-assistant": "cloud",
            "generic-assistant-server": "server",
            "generic-assistant-single-gpu": "singlegpu",
            "generic-assistant-server-perf": "server",
            "multilingual-assistant": "cloud",
            "multilingual-assistant-server": "server",
            "multilingual-assistant-single-gpu": "singlegpu",
            "omni-assistant": "cloud",
            "omni-assistant-server": "server",
            "omni-assistant-single-gpu": "singlegpu",
            "omni-assistant-subagents": "cloud",
            "omni-assistant-subagents-server": "server",
            "omni-assistant-subagents-single-gpu": "singlegpu",
            "frontend-backend-agent": "cloud",
            "frontend-backend-agent-server": "server",
            "frontend-backend-agent-single-gpu": "singlegpu",
        }
        for service_name, platform in expected.items():
            with self.subTest(service=service_name):
                self.assertEqual(
                    compose["services"][service_name]["environment"]["REALTIME_SERVICE_PLATFORM"],
                    platform,
                )

    def test_exact_resolution_and_pipeline_defaults(self) -> None:
        self.assertEqual(
            examples_registry.default_realtime_model_id("generic-assistant"),
            _GENERIC_MODEL,
        )
        self.assertEqual(
            examples_registry.resolve_realtime_model_profile(
                _CLIENT_TOOLS_MODEL,
                pipeline_mode="generic-assistant",
            )["model"],
            _CLIENT_TOOLS_MODEL,
        )

        for model in (
            "NVIDIA/nemotron-realtime",
            f" {_GENERIC_MODEL}",
            f"{_GENERIC_MODEL} ",
            "nvidia/not-registered",
        ):
            with (
                self.subTest(model=model),
                self.assertRaises(examples_registry.RealtimeModelProfileNotAvailable) as raised,
            ):
                examples_registry.resolve_realtime_model_profile(model)
            self.assertEqual(raised.exception.code, "model_not_available")
            self.assertEqual(raised.exception.param, "model")
            self.assertEqual(raised.exception.model, model)

        with self.assertRaises(examples_registry.RealtimeModelProfileNotAvailable):
            examples_registry.resolve_realtime_model_profile(
                _OMNI_MODEL,
                pipeline_mode="generic-assistant",
            )

    def test_selection_lock_hides_models_without_falling_back(self) -> None:
        locked = examples_registry.Selection(
            raw="generic-assistant",
            locked=True,
            example_keys=("generic-assistant",),
            default_key="generic-assistant",
        )
        with patch.object(examples_registry, "_SELECTION", locked):
            self.assertEqual(
                [profile["model"] for profile in examples_registry.visible_realtime_model_profiles()],
                [_GENERIC_MODEL, _CLIENT_TOOLS_MODEL],
            )
            self.assertEqual(examples_registry.default_realtime_model_id(), _GENERIC_MODEL)
            with self.assertRaises(examples_registry.RealtimeModelProfileNotAvailable):
                examples_registry.resolve_realtime_model_profile(_OMNI_MODEL)
            with self.assertRaises(examples_registry.RealtimeModelProfileNotAvailable):
                examples_registry.resolve_realtime_model_profile(
                    pipeline_mode="omni-assistant",
                )

    def test_resolution_and_listing_return_detached_profiles(self) -> None:
        resolved = examples_registry.resolve_realtime_model_profile(_GENERIC_MODEL)
        resolved["label"] = "mutated"
        resolved["selectors"]["prompt_key"] = "mutated"
        listed = examples_registry.visible_realtime_model_profiles()
        listed[0]["selectors"]["prompt_key"] = "also-mutated"

        pristine = examples_registry.resolve_realtime_model_profile(_GENERIC_MODEL)
        self.assertEqual(pristine["label"], "Nemotron Realtime")
        self.assertEqual(pristine["selectors"]["prompt_key"], "generic_assistant")

    def test_parser_rejects_missing_or_duplicate_pipeline_defaults(self) -> None:
        no_default = self._registry()
        no_default["realtime_models"][_GENERIC_MODEL]["default"] = False
        duplicate = self._registry()
        duplicate["realtime_models"][_CLIENT_TOOLS_MODEL]["default"] = True

        for registry in (no_default, duplicate):
            with (
                self.subTest(registry=registry),
                self.assertRaisesRegex(
                    RuntimeError,
                    "exactly one default model",
                ),
            ):
                self._load(registry)

    def test_parser_rejects_unknown_pipeline_and_incomplete_or_incompatible_selectors(self) -> None:
        unknown_pipeline = self._registry()
        unknown_pipeline["realtime_models"][_GENERIC_MODEL]["pipeline_mode"] = "missing-pipeline"

        incomplete = self._registry()
        del incomplete["realtime_models"][_GENERIC_MODEL]["selectors"]["tts_id"]

        incompatible = self._registry()
        incompatible["realtime_models"][_OMNI_MODEL]["selectors"]["asr_id"] = "nemotron-asr-streaming-english"

        cases = (
            (unknown_pipeline, "unknown pipeline"),
            (incomplete, "requires selector 'tts_id'"),
            (incompatible, "is incompatible"),
        )
        for registry, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                self._load(registry)

    def test_parser_rejects_non_public_prompt_and_unknown_static_service(self) -> None:
        non_public_prompt = self._registry()
        multilingual = non_public_prompt["realtime_models"]["nvidia/nemotron-realtime-multilingual"]
        multilingual["selectors"]["prompt_key"] = "fixed_session_language_addon"

        unknown_service = self._registry()
        unknown_service["realtime_models"][_GENERIC_MODEL]["selectors"]["llm_id"] = "missing-llm"

        for registry, message in (
            (non_public_prompt, "is not public"),
            (unknown_service, "does not exist"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                self._load(registry)

    def test_registry_default_requires_a_configured_static_service_default(self) -> None:
        missing_default_examples = copy.deepcopy(examples_registry.EXAMPLES)
        missing_default_examples["generic-assistant"]["defaults"]["tts"] = []

        unknown_default_examples = copy.deepcopy(examples_registry.EXAMPLES)
        unknown_default_examples["generic-assistant"]["defaults"]["llm"] = ["missing-llm"]

        for examples, message in (
            (missing_default_examples, "requires a configured default"),
            (unknown_default_examples, "does not exist"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                self._load(self._registry(), examples)

    def test_platform_validation_never_falls_back_to_an_unrelated_service(self) -> None:
        missing_cloud_override = self._registry()
        del missing_cloud_override["realtime_models"]["nvidia/nemotron-realtime-multilingual"]["platform_overrides"]

        unrelated_cloud_service = self._registry()
        unrelated_cloud_service["realtime_models"][_GENERIC_MODEL]["platform_overrides"] = {
            "cloud": {"asr_id": "missing-asr"}
        }

        wrong_local_platform = self._registry()
        wrong_local_platform["realtime_models"][_GENERIC_MODEL]["platform_overrides"] = {
            "server": {"asr_id": "self-hosted:singlegpu:nemotron-asr-streaming-english"}
        }

        for registry, message in (
            (missing_cloud_override, "does not exist"),
            (unrelated_cloud_service, "does not exist"),
            (wrong_local_platform, "not valid for platform 'server'"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                self._load(registry)

    def test_parser_rejects_invalid_platform_overrides(self) -> None:
        unknown_platform = self._registry()
        unknown_platform["realtime_models"][_GENERIC_MODEL]["platform_overrides"] = {
            "auto": {"asr_id": "registry-default"}
        }
        prompt_override = self._registry()
        prompt_override["realtime_models"][_GENERIC_MODEL]["platform_overrides"] = {
            "cloud": {"prompt_key": "generic_assistant"}
        }

        for registry, message in (
            (unknown_platform, "unknown platform override"),
            (prompt_override, "unknown selector 'prompt_key'"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                self._load(registry)

    def test_registry_default_is_reserved_for_service_selectors(self) -> None:
        registry = self._registry()
        registry["realtime_models"][_GENERIC_MODEL]["selectors"]["prompt_key"] = (
            examples_registry.REALTIME_REGISTRY_DEFAULT_SELECTOR
        )

        with self.assertRaisesRegex(RuntimeError, "must name an explicit public prompt"):
            self._load(registry)

    def test_parser_rejects_unstable_model_ids_and_unknown_fields(self) -> None:
        invalid_id = self._registry()
        invalid_id["realtime_models"][" invalid model "] = invalid_id["realtime_models"].pop(_GENERIC_MODEL)

        unknown_field = self._registry()
        unknown_field["realtime_models"][_GENERIC_MODEL]["base_url"] = "https://untrusted.example/v1"

        for registry, message in (
            (invalid_id, "model id"),
            (unknown_field, "unknown field 'base_url'"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                self._load(registry)


if __name__ == "__main__":
    unittest.main()
