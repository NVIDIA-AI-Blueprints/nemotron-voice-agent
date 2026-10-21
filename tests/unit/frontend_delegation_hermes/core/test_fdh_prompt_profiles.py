# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Prompt profiles (``prompt-profiles-plan.md``): ``config/prompts/<profile>/``, chosen by ``FDH_PROMPT_PROFILE``.

* unset, the browser demo uses ``generic`` and evaluation runs use ``customer_service``, on both sides;
* every profile in ``config/prompts/`` loads and renders its frontend, voice and backend catalogs;
* ``generic`` and ``customer_service`` start as the same prompts.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from _fdh_voice_fakes import PACKAGE, PROFILES

from examples.frontend_delegation_hermes.core.config import load_delegation_config
from examples.frontend_delegation_hermes.core.frontend.prompts import load_prompts
from examples.frontend_delegation_hermes.core.prompt_features import render_backend_system
from examples.frontend_delegation_hermes.core.sidecar import gateway_config as gc
from examples.frontend_delegation_hermes.core.sidecar.templates import BackendTemplates

PROMPTS = PACKAGE / "config" / "prompts"
PROFILE_NAMES = sorted(path.name for path in PROMPTS.iterdir() if path.is_dir())
#: (delegation profile, gateway config, default prompt profile)
MODES = (
    ("pipecat_demo.yaml", "gateway.yaml", "generic"),
    ("browser_demo.yaml", "gateway.yaml", "generic"),
    ("realtime_eval.yaml", "gateway.eval.yaml", "customer_service"),
    ("pipecat_realtime_eval.yaml", "gateway.eval.yaml", "customer_service"),
)


def _env(profile: str | None) -> mock._patch:
    env = {key: value for key, value in os.environ.items() if key != "FDH_PROMPT_PROFILE"}
    if profile is not None:
        env["FDH_PROMPT_PROFILE"] = profile
    return mock.patch.dict(os.environ, env, clear=True)


def _paths(delegation: str, gateway: str, profile: str | None) -> tuple[Path, Path, Path]:
    with _env(profile):
        config = load_delegation_config(PROFILES / delegation)
        backend = gc.load_gateway_config(gc.CONFIG_DIR / gateway)
    return config.delegation.prompts_path, config.voice.agent.prompts_path, Path(backend.hermes.prompts_path)


class PromptProfileTests(unittest.TestCase):
    def test_the_three_profiles_ship(self) -> None:
        self.assertEqual(PROFILE_NAMES, ["customer_service", "general_qa", "generic"])
        for name in PROFILE_NAMES:
            with self.subTest(profile=name):
                self.assertEqual(
                    sorted(path.name for path in (PROMPTS / name).iterdir()),
                    ["backend.yaml", "frontend.yaml", "voice.yaml"],
                )

    def test_unset_each_mode_uses_its_default_profile(self) -> None:
        for delegation, gateway, profile in MODES:
            with self.subTest(delegation=delegation):
                expected = tuple(PROMPTS / profile / name for name in ("frontend.yaml", "voice.yaml", "backend.yaml"))
                self.assertEqual(_paths(delegation, gateway, None), expected)

    def test_the_variable_selects_every_profile_in_every_mode(self) -> None:
        for name in PROFILE_NAMES:
            for delegation, gateway, _ in MODES:
                with self.subTest(profile=name, delegation=delegation):
                    expected = tuple(PROMPTS / name / file for file in ("frontend.yaml", "voice.yaml", "backend.yaml"))
                    self.assertEqual(_paths(delegation, gateway, name), expected)

    def test_every_profile_renders(self) -> None:
        for name in PROFILE_NAMES:
            with self.subTest(profile=name):
                frontend = load_prompts(PROMPTS / name / "frontend.yaml")
                self.assertTrue(frontend.render("frontend_system", capabilities=[], normalization_note=""))
                with _env(name):
                    gateway = gc.load_gateway_config(gc.CONFIG_DIR / "gateway.eval.yaml")
                backend = BackendTemplates(gateway.hermes.prompts_path, prompt_features=gateway.prompt_features)
                self.assertIn("Be brief.", render_backend_system(backend, "Be brief."))
                self.assertTrue(backend.render("backend_soul"))

    def test_generic_starts_as_customer_service(self) -> None:
        for file in ("frontend.yaml", "backend.yaml", "voice.yaml"):
            with self.subTest(file=file):
                self.assertEqual(
                    (PROMPTS / "generic" / file).read_bytes(), (PROMPTS / "customer_service" / file).read_bytes()
                )

    def test_general_qa_delegates_and_answers_without_tools(self) -> None:
        frontend = load_prompts(PROMPTS / "general_qa" / "frontend.yaml")
        system = frontend.render("frontend_system", capabilities=[], normalization_note="")
        self.assertIn("delegate=true", system)
        self.assertIn('filler_text="One moment."', system)
        backend = BackendTemplates(PROMPTS / "general_qa" / "backend.yaml")
        text = render_backend_system(backend, "")
        self.assertIn("You have no tools", text)
        self.assertIn('"The answer is X."', text)
        self.assertNotIn("<policy>", text)


if __name__ == "__main__":
    unittest.main()
