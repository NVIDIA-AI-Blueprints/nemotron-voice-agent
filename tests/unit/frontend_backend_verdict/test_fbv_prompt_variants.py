# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Prompt variants: the Tau catalog is pinned, the generic one is free to tune (plan §7.3).

Every ``tau3_eval*`` profile uses ``core/config/prompts/tau.yaml``, which is v1's
``voice/config/prompts.voice.yaml`` byte for byte. **Tau tuning rule:** changing ``tau.yaml``
needs a new :data:`TAU_CATALOG_SHA256` and an attached Tau A/B run. The ``default`` profile is
the default Tau arm with ``core/config/prompts/generic.yaml``, which any change may tune; the
root ``prompts.yaml`` (registry and UI) mirrors it.

The Tau profiles' merged configs and rendered prompts are compared with v1's in
``test_fbv_golden`` (N4).
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import unittest
from typing import Any
from unittest import mock

import yaml
from _fbv_core_fakes import PACKAGE_ROOT
from _fbv_golden import pinned_environment, to_plain

from examples.frontend_backend_verdict.core.config.profiles import (
    DEFAULT_PROFILE,
    GENERIC_PROFILE,
    SERVED_PROFILES,
    SHIPPED_PROFILES,
    profile_name,
    profile_path,
)
from examples.frontend_backend_verdict.core.config.voice import load_voice_config
from examples.frontend_backend_verdict.core.errors import VoiceConfigError

#: SHA-256 of v1's ``voice/config/prompts.voice.yaml`` at ``fbv-v1-baseline``. See the Tau tuning rule.
TAU_CATALOG_SHA256 = "a29ee978525777ad9811935c0f3efa1784d948889676ee13642745d4023a4a0b"
PROMPTS_DIR = PACKAGE_ROOT / "core" / "config" / "prompts"
_PLACEHOLDER = re.compile(r"(?<!\{)\{([a-z_]+)\}(?!\})")


def _catalog(path) -> dict[str, Any]:  # noqa: ANN001
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _placeholders(catalog: dict[str, Any]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for key, item in catalog.items():
        text = item["content"] if isinstance(item, dict) else str(item)
        out[key] = set(_PLACEHOLDER.findall(text))
    return out


class CatalogTests(unittest.TestCase):
    def test_tau_catalog_is_pinned(self) -> None:
        digest = hashlib.sha256((PROMPTS_DIR / "tau.yaml").read_bytes()).hexdigest()
        self.assertEqual(digest, TAU_CATALOG_SHA256, "tau.yaml changed: update the pin and attach a Tau A/B run")

    def test_generic_catalog_keeps_the_keys_and_placeholders(self) -> None:
        tau, generic = _catalog(PROMPTS_DIR / "tau.yaml"), _catalog(PROMPTS_DIR / "generic.yaml")
        self.assertEqual(set(generic), set(tau))
        self.assertEqual(_placeholders(generic), _placeholders(tau))

    def test_root_catalog_mirrors_generic(self) -> None:
        self.assertEqual((PACKAGE_ROOT / "prompts.yaml").read_bytes(), (PROMPTS_DIR / "generic.yaml").read_bytes())


class ProfileCatalogTests(unittest.TestCase):
    def test_each_profile_resolves_to_its_catalog(self) -> None:
        with pinned_environment():
            for name in SERVED_PROFILES:
                expected = "generic.yaml" if name == GENERIC_PROFILE else "tau.yaml"
                with self.subTest(profile=name):
                    path = load_voice_config(profile_path(name)).agent.prompts_path
                    self.assertEqual(path.resolve(), (PROMPTS_DIR / expected).resolve())

    def test_default_profile_is_the_default_tau_arm_with_generic_prompts(self) -> None:
        with pinned_environment():
            generic = load_voice_config(profile_path(GENERIC_PROFILE))
            tau = load_voice_config(profile_path(DEFAULT_PROFILE))

        def without_prompts_and_sources(config: Any) -> Any:
            prompts = dataclasses.replace(config.agent.prompts, path="")
            agent = dataclasses.replace(config.agent, prompts=prompts)
            plain = to_plain(dataclasses.replace(config, agent=agent))
            plain.pop("source_files")  # default.yaml adds one file to the extends chain
            return plain

        self.assertEqual(without_prompts_and_sources(generic), without_prompts_and_sources(tau))

    def test_served_profiles(self) -> None:
        self.assertEqual(SERVED_PROFILES, (*SHIPPED_PROFILES, GENERIC_PROFILE))
        self.assertEqual(profile_name({"FBV_PROFILE": "default"}), GENERIC_PROFILE)
        self.assertEqual(profile_name({}), DEFAULT_PROFILE)  # an unset variable keeps the Tau arm
        self.assertTrue(all(name.startswith("tau3_eval") for name in SHIPPED_PROFILES))

    def test_a_bad_prompts_override_fails_at_load(self) -> None:
        with (
            pinned_environment(),
            mock.patch.dict(os.environ, {"FBA_VOICE_PROMPTS": "/nonexistent/prompts.yaml"}),
            self.assertRaises(VoiceConfigError) as caught,
        ):
            load_voice_config(profile_path(DEFAULT_PROFILE))
        self.assertIn("agent.overrides.prompts.path", str(caught.exception))
        self.assertIn("/nonexistent/prompts.yaml", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
