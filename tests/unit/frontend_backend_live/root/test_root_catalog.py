# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103
"""The root's catalog: the profiles' default models, keys and unlisted slots."""

import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlparse

import yaml

import examples_registry
from examples.frontend_backend_live.models.endpoints import ExampleCatalog, resolve_endpoint
from examples.frontend_backend_live.prompts.catalog import role_instructions
from utils import NVCF_LLM_BASE_URL

REPO = Path(__file__).resolve().parents[4]
SELF_HOSTED_RECIPES = ("server", "single-gpu")


def compose_services() -> dict[str, dict]:
    """Every service of the root Compose file and the files it includes, by name."""
    compose = yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))
    services = dict(compose["services"])
    for include in compose.get("include", []):
        services.update(yaml.safe_load((REPO / include["path"]).read_text(encoding="utf-8")).get("services") or {})
    return services


def environment(service: dict) -> dict[str, str]:
    """A service's environment as a mapping, with each ``${NAME:-default}`` read as its default."""
    raw = service.get("environment") or {}
    pairs = raw.items() if isinstance(raw, dict) else (item.split("=", 1) for item in raw if "=" in item)
    resolved = {}
    for name, value in pairs:
        match = re.fullmatch(r"\$\{\w+:-(.*)\}", str(value))
        resolved[name] = match.group(1) if match else str(value)
    return resolved


def started_hosts(services: dict[str, dict], profile: str) -> set[str]:
    """The names a profile's containers answer to on the Compose network: service names and network aliases."""
    hosts = set()
    for name, service in services.items():
        if profile in (service.get("profiles") or []):
            hosts.add(name)
            networks = service.get("networks")
            if isinstance(networks, dict):
                for network in networks.values():
                    hosts.update((network or {}).get("aliases") or [])
    return hosts


def entry_host(entry: dict) -> str:
    """The host a catalog entry points at: its ``base_url``'s, or its gRPC ``server``'s."""
    return urlparse(entry["base_url"]).hostname if entry.get("base_url") else entry["server"].rsplit(":", 1)[0]


class ComposeWiringTests(unittest.TestCase):
    def test_each_local_profile_starts_the_host_of_every_default_entry(self):
        services = compose_services()
        catalog = yaml.safe_load((REPO / "services.yaml").read_text(encoding="utf-8"))
        registry = yaml.safe_load((REPO / "examples_registry.yaml").read_text(encoding="utf-8"))["examples"]
        checked = 0
        for app in services.values():
            env = environment(app)
            recipe, example = env.get("SERVICE_RECIPE"), registry.get(env.get("EXAMPLE_SELECTION", ""))
            if recipe not in SELF_HOSTED_RECIPES or example is None:
                continue
            for profile in app.get("profiles") or []:
                hosts = started_hosts(services, profile)
                for slot, keys in example["services"].items():
                    section = catalog[recipe].get((example.get("categories") or {}).get(slot, slot), {})
                    default = next(key for key in keys if key in section)  # the registry's first listed entry
                    with self.subTest(profile=profile, slot=slot, entry=default):
                        self.assertIn(entry_host(section[default]), hosts)
                    checked += 1
        self.assertGreater(checked, 20)  # every example's local profiles were read

    def test_the_live_server_profile_defaults_to_the_model_it_starts(self):
        services = examples_registry.EXAMPLES["frontend-backend-live"]["services"]
        self.assertEqual((services["llm"][0], services["thinker-llm"][0]), ("nemotron-lightning", "nemotron-lightning"))


class EndpointTests(unittest.TestCase):
    ENTRY = {"model_id": "org/model", "base_url": "http://host:8000/v1"}

    def test_a_named_key_variable_is_used_alone(self):
        entry = {**self.ENTRY, "api_key_env": "ACME_API_KEY"}
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-secret", "ACME_API_KEY": "acme-key"}):
            self.assertEqual(resolve_endpoint(entry, {}, slot="llm").api_key, "acme-key")
        for empty in ("", "   "):
            with (
                self.subTest(value=empty),
                patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-secret", "ACME_API_KEY": empty}),
                self.assertRaisesRegex(ValueError, "ACME_API_KEY is empty: the 'llm' slot's entry"),
            ):
                resolve_endpoint(entry, {}, slot="llm")

    def test_an_entry_with_no_key_variable_uses_the_nvidia_key(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-secret"}):
            self.assertEqual(resolve_endpoint(self.ENTRY, {}, slot="llm").api_key, "nvapi-secret")

    def test_a_slot_with_no_model_names_itself(self):
        for entry in ({}, {"base_url": "http://host:8000/v1"}):
            with self.subTest(entry=entry), self.assertRaisesRegex(ValueError, "'thinker-llm' slot has no model"):
                resolve_endpoint(entry, {}, "thinker_", slot="thinker-llm")
        # A custom endpoint the session body names needs no catalog entry.
        body = {"thinker_model_id": "org/custom", "thinker_base_url": "http://c/v1"}
        custom = resolve_endpoint({}, body, "thinker_", slot="thinker-llm")
        self.assertEqual((custom.model, custom.base_url), ("org/custom", "http://c/v1"))

    def test_the_default_base_url_is_the_nvidia_cloud_url_from_utils(self):
        self.assertEqual(resolve_endpoint({"model_id": "org/model"}, {}, slot="llm").base_url, NVCF_LLM_BASE_URL)


class CatalogTests(unittest.TestCase):
    """The catalog the root reads once (``ServiceCatalog``): entries by slot, the default first, and the prompts."""

    CATALOG = ExampleCatalog(
        {
            "llm": [{"model_id": "org/a", "base_url": "http://talker/v1", "extra_params": '{"temperature": 0}'}],
            "thinker-llm": [
                {"model_id": "org/b", "base_url": "http://thinker/v1"},
                {"model_id": "org/a", "base_url": "http://thinker/v1", "extra_params": '{"temperature": 0.7}'},
            ],
            "asr": [{"server": "asr:50052", "model": "parakeet"}],
        }
    )

    def test_a_slot_gives_its_default_entry_and_the_sessions_overrides(self):
        self.assertEqual(self.CATALOG.model("thinker-llm").model, "org/b")
        self.assertEqual(self.CATALOG.model("llm", {"model_id": "org/custom"}).model, "org/custom")
        self.assertEqual(self.CATALOG.service("asr")["server"], "asr:50052")
        self.assertEqual(self.CATALOG.models("thinker-llm"), ["org/b", "org/a"])
        with self.assertRaisesRegex(ValueError, "'tts' slot has no model"):
            self.CATALOG.model("tts")

    def test_a_named_model_keeps_the_settings_of_the_slot_that_asks(self):
        self.assertEqual(self.CATALOG.named("thinker-llm", "org/a").extra_params, {"temperature": 0.7})
        self.assertEqual(self.CATALOG.named("llm", "org/a").extra_params, {"temperature": 0})
        self.assertEqual(self.CATALOG.named("llm", "org/b").base_url, "http://thinker/v1")  # another slot's entry
        with self.assertRaisesRegex(LookupError, "'org/z' is not in the catalog"):
            self.CATALOG.named("thinker-llm", "org/z")

    def test_a_prompt_is_the_roles_static_instructions(self):
        self.assertEqual(self.CATALOG.prompt("talker"), role_instructions("talker"))


if __name__ == "__main__":
    unittest.main()
