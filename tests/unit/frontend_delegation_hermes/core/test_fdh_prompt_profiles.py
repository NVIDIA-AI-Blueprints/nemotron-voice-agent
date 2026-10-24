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
from examples.frontend_delegation_hermes.core.sidecar.session_runtime import backend_templates
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


def _env(profile: str | None, backend: str | None = None) -> mock._patch:
    env = {key: value for key, value in os.environ.items() if key != "FDH_PROMPT_PROFILE"}
    if profile is not None:
        env["FDH_PROMPT_PROFILE"] = profile
    env.pop("FDH_BACKEND", None)
    if backend is not None:
        env["FDH_BACKEND"] = backend
    return mock.patch.dict(os.environ, env, clear=True)


def _paths(delegation: str, gateway: str, profile: str | None, kind: str | None = None) -> tuple[Path, Path, Path]:
    with _env(profile, kind):
        config = load_delegation_config(PROFILES / delegation)
        backend = gc.load_gateway_config(gc.CONFIG_DIR / gateway)
    return config.delegation.prompts_path, config.voice.agent.prompts_path, Path(backend.backend.prompts_path)


def _files(profile: str, kind: str = "hermes") -> tuple[Path, Path, Path]:
    return (
        PROMPTS / profile / "frontend.yaml",
        PROMPTS / profile / "voice.yaml",
        PROMPTS / profile / "backend" / f"{kind}.yaml",
    )


class PromptProfileTests(unittest.TestCase):
    def test_the_three_profiles_ship(self) -> None:
        self.assertEqual(PROFILE_NAMES, ["customer_service", "general_qa", "generic"])
        for name in PROFILE_NAMES:
            with self.subTest(profile=name):
                self.assertEqual(
                    sorted(path.name for path in (PROMPTS / name).iterdir()),
                    ["backend", "frontend.yaml", "voice.yaml"],
                )
                # One backend prompt file per registered backend that renders prompts (the fake one reuses hermes.yaml).
                self.assertEqual(
                    sorted(p.name for p in (PROMPTS / name / "backend").iterdir()), ["hermes.yaml", "llm.yaml"]
                )

    def test_unset_each_mode_uses_its_default_profile(self) -> None:
        for delegation, gateway, profile in MODES:
            with self.subTest(delegation=delegation):
                self.assertEqual(_paths(delegation, gateway, None), _files(profile))
                self.assertEqual(_paths(delegation, gateway, None, "llm"), _files(profile, "llm"))

    def test_the_variable_selects_every_profile_in_every_mode(self) -> None:
        for name in PROFILE_NAMES:
            for delegation, gateway, _ in MODES:
                with self.subTest(profile=name, delegation=delegation):
                    self.assertEqual(_paths(delegation, gateway, name), _files(name))
                    self.assertEqual(_paths(delegation, gateway, name, "llm"), _files(name, "llm"))

    def test_every_profile_renders(self) -> None:
        for name in PROFILE_NAMES:
            with self.subTest(profile=name):
                frontend = load_prompts(PROMPTS / name / "frontend.yaml")
                self.assertTrue(frontend.render("frontend_system", capabilities=[], normalization_note=""))
                for kind in ("hermes", "llm"):
                    with _env(name, kind):
                        gateway = gc.load_gateway_config(gc.CONFIG_DIR / "gateway.eval.yaml")
                    backend = backend_templates(gateway)
                    self.assertIn("Be brief.", render_backend_system(backend, "Be brief."))
                    self.assertTrue(backend.render("backend_soul"))

    def test_generic_starts_as_customer_service(self) -> None:
        for file in ("frontend.yaml", "backend/hermes.yaml", "backend/llm.yaml", "voice.yaml"):
            with self.subTest(file=file):
                self.assertEqual(
                    (PROMPTS / "generic" / file).read_bytes(), (PROMPTS / "customer_service" / file).read_bytes()
                )

    def test_general_qa_delegates_and_answers_without_tools(self) -> None:
        frontend = load_prompts(PROMPTS / "general_qa" / "frontend.yaml")
        system = frontend.render("frontend_system", capabilities=[], normalization_note="")
        self.assertIn("delegate=true", system)
        self.assertIn('filler_text="One moment."', system)
        backend = BackendTemplates(PROMPTS / "general_qa" / "backend" / "hermes.yaml")
        text = render_backend_system(backend, "")
        self.assertIn("You have no tools", text)
        self.assertIn('"The answer is X."', text)
        self.assertNotIn("<policy>", text)


if __name__ == "__main__":
    unittest.main()
