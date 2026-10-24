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
        self.assertEqual((config.backend.kind, config.workers.mode), ("hermes", "process_per_session"))
        self.assertEqual(config.backend.agent_construct, "eager")
        self.assertGreater(config.backend.run_hard_deadline_s, config.backend.run_budget_seconds)
        self.assertTrue(Path(config.backend.prompts_path).is_file())
        self.assertEqual(Path(config.backend.prompts_path).parts[-2:], ("backend", "hermes.yaml"))
        self.assertEqual(config.backend_settings.context_length, 131072)
        fake = gc.load_gateway_config(gc.FAKE_GATEWAY_CONFIG)
        self.assertEqual((fake.backend.kind, fake.workers.python), ("fake", "self"))
        self.assertIsNone(fake.backend_settings)
        self.assertEqual(gc.default_fake_gateway_config().workers.mode, "in_process")

    def test_backend_flag_selects_the_backend_and_its_prompt_file(self) -> None:
        with mock.patch.dict(os.environ, {"FDH_BACKEND": "llm"}):
            for name in ("gateway.yaml", "gateway.eval.yaml"):
                with self.subTest(config=name):
                    config = gc.load_gateway_config(gc.CONFIG_DIR / name)
                    self.assertEqual(config.backend.kind, "llm")
                    # The LLM backend runs only in-process: process_per_session follows the backend.
                    self.assertEqual(config.workers.mode, "in_process")
                    self.assertEqual(Path(config.backend.prompts_path).name, "llm.yaml")
                    self.assertEqual(config.backend_settings.reasoning_effort, "medium")
                    settings = config.worker_settings()
                    self.assertEqual(settings["max_retries"], 3)
                    self.assertNotIn("context_length", settings)
        with self.assertRaisesRegex(gc.GatewayConfigError, "backend.kind must be one of"):
            gc.load_gateway_config(None, overrides={"backend": {"kind": "openclaw"}})

    def test_renamed_keys_name_their_replacement(self) -> None:
        cases = [
            ({"workers": {"agent_kind": "hermes"}}, "replaced by backend.kind"),
            ({"workers": {"mode": "in_process_fake"}}, "renamed to in_process"),
            ({"hermes": {"model": "m"}}, "hermes.model moved to backend.model"),
            ({"hermes": {"prompts": {"path": "x.yaml"}}}, "hermes.prompts moved to backend.prompts"),
        ]
        for overrides, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(gc.GatewayConfigError, message):
                gc.load_gateway_config(None, overrides=overrides)

    def test_environment_interpolation(self) -> None:
        with mock.patch.dict(os.environ, {"FDH_MAX_SESSIONS": "3", "FDH_GATEWAY_PORT": "9999"}):
            config = gc.load_gateway_config()
        self.assertEqual((config.gateway.max_sessions, config.gateway.port), (3, 9999))
        with self.assertRaisesRegex(gc.GatewayConfigError, "NOT_SET_ANYWHERE"):
            gc.interpolate_env("${NOT_SET_ANYWHERE}", {})

    def test_strict_keys_and_cross_checks(self) -> None:
        cases = [
            ({"workers": {"bogus": 1}}, "unknown config key 'workers.bogus'"),
            ({"backend": {"run_hard_deadline_s": 10, "run_budget_seconds": 20}}, "run_hard_deadline_s"),
            ({"workers": {"stop_timeout_s": 1}, "backend": {"unwind_timeout_s": 5}}, "stop_timeout_s"),
            ({"workers": {"mode": "threads"}}, "workers.mode"),
            ({"workers": {"mode": "in_process"}}, "in_process is not supported by backend.kind: hermes"),
            ({"gateway": {"max_sessions": 0}}, "max_sessions"),
            ({"backend": {"agent_construct": "sometimes"}}, "agent_construct"),
            ({"llm": {"max_retries": 0}, "backend": {"kind": "llm"}}, "max_retries"),
        ]
        for overrides, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(gc.GatewayConfigError, message):
                gc.load_gateway_config(None, overrides=overrides)
        # Free-form request overrides are not key-checked.
        config = gc.load_gateway_config(None, overrides={"backend": {"request_overrides": {"anything": {"x": 1}}}})
        self.assertEqual(config.backend.request_overrides["anything"], {"x": 1})

    def test_extends_chain_and_relative_prompt_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base.yaml"
            base.write_text("gateway: {port: 1111}\nbackend: {prompts: {path: p.yaml}}\n")
            (Path(tmp) / "p.yaml").write_text("x: y\n")
            child = Path(tmp) / "child.yaml"
            child.write_text("extends: base.yaml\ngateway: {max_sessions: 2}\n")
            config = gc.load_gateway_config(child)
        self.assertEqual((config.gateway.port, config.gateway.max_sessions), (1111, 2))
        self.assertEqual(config.backend.prompts_path, str((Path(tmp) / "p.yaml").resolve()))
        self.assertEqual(len(config.files), 2)

    def test_worker_settings_carry_what_the_worker_needs(self) -> None:
        settings = gc.load_gateway_config().worker_settings()
        for key in ("model", "base_url", "api_key_env", "agent_construct", "request_overrides", "unwind_timeout_s"):
            self.assertIn(key, settings)
        for key in ("provider", "context_length", "load_soul_identity", "disable_streaming"):  # hermes:
            self.assertIn(key, settings)
        self.assertNotIn("NVIDIA_API_KEY=", str(settings))

    def test_backend_templates_are_complete(self) -> None:
        templates = BackendTemplates(gc.CONFIG_DIR / "prompts" / "customer_service" / "backend" / "hermes.yaml")
        for key in REQUIRED_KEYS:
            self.assertIsInstance(key, str)
        self.assertIn("POLICY", templates.render("backend_system", instructions="POLICY"))
        self.assertNotIn("<policy>", templates.render("backend_system", instructions=""))
        self.assertEqual(templates.render("backend_delegation_message", block="", text="hi"), "hi")


if __name__ == "__main__":
    unittest.main()
