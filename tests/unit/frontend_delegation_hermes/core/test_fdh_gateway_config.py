# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D100, D101, D102, D103, D107

"""Gateway config: shipped files load; interpolation, extends, strict keys and cross-checks."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from examples.frontend_delegation_hermes.core.sidecar import gateway_config as gc
from examples.frontend_delegation_hermes.core.sidecar.templates import REQUIRED_KEYS, BackendTemplates


class GatewayConfigTest(unittest.TestCase):
    def test_shipped_configs_load(self) -> None:
        config = gc.load_gateway_config()
        self.assertEqual(config.workers.mode, "process_per_session")
        self.assertEqual(config.hermes.agent_construct, "eager")
        self.assertGreater(config.hermes.run_hard_deadline_s, config.hermes.run_budget_seconds)
        self.assertTrue(Path(config.hermes.prompts_path).is_file())
        fake = gc.load_gateway_config(gc.FAKE_GATEWAY_CONFIG)
        self.assertEqual((fake.workers.agent_kind, fake.workers.python), ("fake", "self"))
        self.assertEqual(gc.default_fake_gateway_config().workers.mode, "in_process_fake")

    def test_environment_interpolation(self) -> None:
        with mock.patch.dict(os.environ, {"FDH_MAX_SESSIONS": "3", "FDH_GATEWAY_PORT": "9999"}):
            config = gc.load_gateway_config()
        self.assertEqual((config.gateway.max_sessions, config.gateway.port), (3, 9999))
        with self.assertRaisesRegex(gc.GatewayConfigError, "NOT_SET_ANYWHERE"):
            gc.interpolate_env("${NOT_SET_ANYWHERE}", {})

    def test_strict_keys_and_cross_checks(self) -> None:
        cases = [
            ({"workers": {"bogus": 1}}, "unknown config key 'workers.bogus'"),
            ({"hermes": {"run_hard_deadline_s": 10, "run_budget_seconds": 20}}, "run_hard_deadline_s"),
            ({"workers": {"stop_timeout_s": 1}, "hermes": {"unwind_timeout_s": 5}}, "stop_timeout_s"),
            ({"workers": {"mode": "threads"}}, "workers.mode"),
            ({"workers": {"mode": "in_process_fake", "agent_kind": "hermes"}}, "in_process_fake"),
            ({"gateway": {"max_sessions": 0}}, "max_sessions"),
            ({"hermes": {"agent_construct": "sometimes"}}, "agent_construct"),
        ]
        for overrides, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(gc.GatewayConfigError, message):
                gc.load_gateway_config(None, overrides=overrides)
        # Free-form request overrides are not key-checked.
        config = gc.load_gateway_config(None, overrides={"hermes": {"request_overrides": {"anything": {"x": 1}}}})
        self.assertEqual(config.hermes.request_overrides["anything"], {"x": 1})

    def test_extends_chain_and_relative_prompt_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base.yaml"
            base.write_text("gateway: {port: 1111}\nhermes: {prompts: {path: p.yaml}}\n")
            (Path(tmp) / "p.yaml").write_text("x: y\n")
            child = Path(tmp) / "child.yaml"
            child.write_text("extends: base.yaml\ngateway: {max_sessions: 2}\n")
            config = gc.load_gateway_config(child)
        self.assertEqual((config.gateway.port, config.gateway.max_sessions), (1111, 2))
        self.assertEqual(config.hermes.prompts_path, str((Path(tmp) / "p.yaml").resolve()))
        self.assertEqual(len(config.files), 2)

    def test_worker_settings_carry_what_the_worker_needs(self) -> None:
        settings = gc.load_gateway_config().hermes.worker_settings()
        for key in ("model", "base_url", "api_key_env", "agent_construct", "request_overrides", "unwind_timeout_s"):
            self.assertIn(key, settings)
        self.assertNotIn("NVIDIA_API_KEY=", str(settings))

    def test_backend_templates_are_complete(self) -> None:
        templates = BackendTemplates(gc.CONFIG_DIR / "prompts.backend.yaml")
        for key in REQUIRED_KEYS:
            self.assertIsInstance(key, str)
        self.assertIn("POLICY", templates.render("backend_system", instructions="POLICY"))
        self.assertNotIn("<policy>", templates.render("backend_system", instructions=""))
        self.assertEqual(templates.render("backend_delegation_message", block="", text="hi"), "hi")


if __name__ == "__main__":
    unittest.main()
