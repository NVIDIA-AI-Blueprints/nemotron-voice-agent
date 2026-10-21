# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Prompt profiles: the customer-service catalog is pinned, the generic one is free to tune (plan §7.3).

``FBA_PROMPT_PROFILE`` names a prompt profile, ``core/config/prompts/<profile>.yaml``. Unset, the
``eval`` profile (OpenAI Realtime sessions, Tau) uses ``customer_service.yaml``: v1's
``voice/config/prompts.voice.yaml`` plus later customer-service rules and domain notes. **Tau tuning rule:** changing
``customer_service.yaml`` needs a new :data:`TAU_CATALOG_SHA256` and an attached Tau A/B run. The
``default`` profile (browser UI sessions) uses ``generic.yaml``, which any change may tune; the root
``prompts.yaml`` (registry and UI) mirrors it. ``general_qa.yaml`` answers any request directly, with
no tools.

The ``eval`` profile's merged config and rendered prompts are compared with v1's in
``test_fba_golden`` (N4).
"""

from __future__ import annotations

import hashlib
import os
import re
import unittest
from typing import Any
from unittest import mock

import yaml
from _fba_core_fakes import PACKAGE_ROOT
from _fba_golden import pinned_environment, to_plain

from examples.frontend_backend_agent.core.config.profiles import (
    DEFAULT_PROFILE,
    REALTIME_PROFILE,
    SERVED_PROFILES,
    ProfileError,
    profile_name,
    profile_path,
)
from examples.frontend_backend_agent.core.config.voice import load_voice_config
from examples.frontend_backend_agent.core.errors import VoiceConfigError

#: SHA-256 of ``customer_service.yaml``. See the Tau tuning rule. Re-pinned on 2026-10-21 for the
#: customer-service rules and domain notes; the previous pin, ``a29ee978…``, was v1's
#: ``voice/config/prompts.voice.yaml`` at ``fbv-v1-baseline``. The Tau A/B run for this pin is pending.
TAU_CATALOG_SHA256 = "ad0481c985601b55f0eca37a6dc52f28fcd5c38d9892d93eb26529d8e94fd38c"
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
        digest = hashlib.sha256((PROMPTS_DIR / "customer_service.yaml").read_bytes()).hexdigest()
        self.assertEqual(
            digest, TAU_CATALOG_SHA256, "customer_service.yaml changed: update the pin and attach a Tau A/B run"
        )

    def test_generic_catalog_keeps_the_keys_and_placeholders(self) -> None:
        tau = _catalog(PROMPTS_DIR / "customer_service.yaml")
        generic = _catalog(PROMPTS_DIR / "generic.yaml")
        self.assertEqual(set(generic), set(tau))
        self.assertEqual(_placeholders(generic), _placeholders(tau))

    def test_every_profile_has_the_customer_service_keys(self) -> None:
        keys = set(_catalog(PROMPTS_DIR / "customer_service.yaml"))
        for path in sorted(PROMPTS_DIR.glob("*.yaml")):
            with self.subTest(profile=path.stem):
                self.assertEqual(set(_catalog(path)), keys)

    def test_root_catalog_mirrors_generic(self) -> None:
        self.assertEqual((PACKAGE_ROOT / "prompts.yaml").read_bytes(), (PROMPTS_DIR / "generic.yaml").read_bytes())


class ProfileCatalogTests(unittest.TestCase):
    def test_each_profile_resolves_to_its_catalog(self) -> None:
        expected = {DEFAULT_PROFILE: "generic.yaml", REALTIME_PROFILE: "customer_service.yaml"}
        with pinned_environment():
            for name in SERVED_PROFILES:
                with self.subTest(profile=name):
                    path = load_voice_config(profile_path(name)).agent.prompts_path
                    self.assertEqual(path.resolve(), (PROMPTS_DIR / expected[name]).resolve())

    def test_default_profile_keeps_the_eval_agent(self) -> None:
        """The browser profile changes turn taking, greeting, tools and logs, not the agent itself."""
        with pinned_environment():
            browser = to_plain(load_voice_config(profile_path(DEFAULT_PROFILE)))
            realtime = to_plain(load_voice_config(profile_path(REALTIME_PROFILE)))
        for section in ("barge_in", "filler", "audio", "asr", "tts", "instructions"):
            with self.subTest(section=section):
                expected = dict(realtime[section])
                actual = dict(browser[section])
                if section == "filler":
                    expected.pop("log_path")
                    actual.pop("log_path")
                self.assertEqual(actual, expected)
        self.assertEqual(browser["normalization"]["transcript"], realtime["normalization"]["transcript"])
        self.assertEqual(
            browser["agent"]["backend"]["conversation_history"], realtime["agent"]["backend"]["conversation_history"]
        )

    def test_served_profiles(self) -> None:
        self.assertEqual(SERVED_PROFILES, ("default", "eval"))
        self.assertEqual(profile_name({}), DEFAULT_PROFILE)  # unset: the browser profile
        self.assertEqual(profile_name({"FBA_PROFILE": "eval"}), REALTIME_PROFILE)
        self.assertEqual(profile_name({"FBA_PROFILE": "default"}), DEFAULT_PROFILE)

    def test_old_profile_names_are_refused(self) -> None:
        for name in ("tau3_eval_frontend_verdict_speak_history", "tau3_eval", "generic", "ui"):
            with self.subTest(name=name), self.assertRaises(ProfileError) as caught:
                profile_name({"FBA_PROFILE": name})
            self.assertIn("'default'", str(caught.exception))
            self.assertIn("'eval'", str(caught.exception))

    def test_the_prompt_profile_variable_selects_the_catalog_in_both_profiles(self) -> None:
        for path in sorted(PROMPTS_DIR.glob("*.yaml")):
            for name in SERVED_PROFILES:
                with (
                    self.subTest(prompt_profile=path.stem, profile=name),
                    pinned_environment(),
                    mock.patch.dict(os.environ, {"FBA_PROMPT_PROFILE": path.stem}),
                ):
                    config = load_voice_config(profile_path(name))
                    self.assertEqual(config.agent.prompts_path.resolve(), path.resolve())

    def test_an_unknown_prompt_profile_fails_at_load(self) -> None:
        with (
            pinned_environment(),
            mock.patch.dict(os.environ, {"FBA_PROMPT_PROFILE": "no_such_profile"}),
            self.assertRaises(VoiceConfigError) as caught,
        ):
            load_voice_config(profile_path(REALTIME_PROFILE))
        self.assertIn("agent.overrides.prompts.path", str(caught.exception))
        self.assertIn("no_such_profile.yaml", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
