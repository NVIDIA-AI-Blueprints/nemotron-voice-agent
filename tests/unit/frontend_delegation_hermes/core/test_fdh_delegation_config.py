# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Voice-side profiles: every one loads; 800 ms VAD, normalization and delay invariants; strict keys."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _fdh_variants import variant  # noqa: E402
from _fdh_voice_fakes import PROFILES

from examples.frontend_delegation_hermes.core.config import DelegationConfigError, load_delegation_config
from examples.frontend_delegation_hermes.core.sidecar import gateway_config as gc

SHIPPED = sorted(PROFILES.glob("*.yaml"))
#: Backend variants of tau3-geval-failure-fixes-plan.md: on by default, off in gateway.eval_control.yaml.
GEVAL_BACKEND_FIXES = (
    "format_correction",
    "write_consent_v2",
    "policy_check",
    "request_scope",
    "transfer_check",
    "capability_check",
)
#: fix-plan-f1-f2-f4-f8.md backend variants (F4, F8 name line), on by default since 2026-10-09.
FIX_PLAN_BACKEND_FIXES = ("situational_facts", "identity_from_identifier")


class ShippedProfileTests(unittest.TestCase):
    def test_every_profile_loads_with_the_invariants(self) -> None:
        self.assertEqual(
            [p.name for p in SHIPPED],
            [
                "browser_demo.yaml",
                "pipecat_demo.yaml",
                "pipecat_realtime_eval.yaml",
                "realtime_eval.yaml",
            ],
        )
        for path in SHIPPED:
            with self.subTest(profile=path.name):
                config = load_delegation_config(path)
                td = config.voice.turn_detection
                self.assertEqual((td.silence_duration_ms, td.honor_client_values), (800, False))
                self.assertTrue(config.voice.normalization.transcript.enabled)
                self.assertEqual(config.voice.server.ws_ping_interval_s, 0)
                expected_delay = 5.0 if path.name in ("browser_demo.yaml", "pipecat_demo.yaml") else 0.0
                self.assertEqual(config.backend.delay_seconds, expected_delay)

    def test_tau3_and_browser_specifics(self) -> None:
        tau3 = load_delegation_config(variant("profiles/tau3_eval.yaml"))
        self.assertEqual(
            (tau3.voice.server.port, tau3.tools.executor, tau3.voice.tools.source), (8775, "wire", "client")
        )
        self.assertTrue(tau3.voice.normalization.tool_arguments.enabled)
        self.assertFalse(tau3.voice.protocol.greeting_enabled)
        browser = load_delegation_config(PROFILES / "browser_demo.yaml")
        self.assertEqual((browser.voice.server.port, browser.tools.executor), (8776, "local"))
        self.assertFalse(browser.voice.normalization.tool_arguments.enabled)
        self.assertTrue(browser.voice.protocol.greeting_enabled)
        self.assertEqual([spec.name for spec in browser.voice.tools.config_tool_specs], ["get_order", "cancel_order"])

    def test_defaults_turn_on_every_fix_except_m2(self) -> None:
        expected = {"proactive_status", "clean_answers", "spelling_hold", "filler_dedupe", "spelled_runs"}
        # result_hints (I3/I4) needs client tools: on for tau3, not applicable to the browser demo.
        tau3_eval, browser = variant("profiles/tau3_eval.yaml"), PROFILES / "browser_demo.yaml"
        for path, extra in ((tau3_eval, {"result_hints"}), (browser, set())):
            with self.subTest(profile=path.name):
                features = load_delegation_config(path).features
                self.assertEqual({k for k, v in features.items() if v is True}, expected | extra)
                self.assertFalse(any(features["prompt_features"].values()))  # replay_intent (M2) stays off
        tau3 = load_delegation_config(variant("profiles/tau3_eval.yaml"))
        self.assertEqual(tau3.delegation.spelling_hold.complete_patterns, ("^[A-Za-z0-9]{6}$",))
        self.assertEqual(tau3.voice.normalization.tool_arguments.invalid_message_key, "tool_argument_invalid_readback")
        self.assertEqual(
            tau3.voice.normalization.tool_arguments.escalate_invalid_message_key, "tool_argument_invalid_spell_all"
        )
        hints = tau3.voice.normalization.tool_arguments.result_hints
        self.assertEqual(hints.tools, ("find_user_id_by_name_zip", "find_user_id_by_email", "get_user_details"))
        self.assertEqual(
            (hints.message_key, hints.escalate_message_key),
            ("lookup_not_found_hint", "lookup_not_found_hint_words"),
        )

    def test_gateway_defaults_turn_on_every_backend_variant_and_the_baseline_none(self) -> None:
        config_dir = PROFILES.parent
        defaults = gc.load_gateway_config(config_dir / "gateway.yaml").prompt_features
        self.assertEqual({k for k, v in defaults.items() if not v}, set())  # domain_notes too: the telecom note
        control = gc.load_gateway_config(variant("gateway.eval_control.yaml")).prompt_features
        self.assertEqual(
            {k for k, v in defaults.items() if v != control[k]},
            {*GEVAL_BACKEND_FIXES, *FIX_PLAN_BACKEND_FIXES, "domain_notes"},
        )
        self.assertFalse(any(gc.load_gateway_config(variant("gateway.baseline.yaml")).prompt_features.values()))

    def test_delay_is_env_configurable_in_the_browser_profile(self) -> None:
        with mock.patch.dict(os.environ, {"FDH_BACKEND_DELAY_S": "2.5"}):
            self.assertEqual(load_delegation_config(PROFILES / "browser_demo.yaml").backend.delay_seconds, 2.5)

    def test_secrets_are_redacted_and_hash_is_stable(self) -> None:
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "sk-secret"}):
            config = load_delegation_config(variant("profiles/tau3_eval.yaml"))
            again = load_delegation_config(variant("profiles/tau3_eval.yaml"))
        self.assertEqual(config.effective["frontend"]["llm"]["api_key"], "***")
        self.assertNotIn("sk-secret", str(config.effective))
        self.assertEqual(config.config_hash, again.config_hash)


class ValidationTests(unittest.TestCase):
    def load(self, text: str):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.yaml"
            path.write_text(f"extends: {variant('profiles/tau3_eval.yaml')}\n{text}")
            return load_delegation_config(path)

    def test_unknown_key_enum_and_cross_checks(self) -> None:
        with self.assertRaisesRegex(DelegationConfigError, "unknown key 'backend.nope'"):
            self.load("backend: {nope: 1}\n")
        with self.assertRaisesRegex(DelegationConfigError, "backend.steer_mode"):
            self.load("backend: {steer_mode: sometimes}\n")
        with self.assertRaisesRegex(DelegationConfigError, "executor: local requires"):
            self.load("tools: {executor: local}\n")
        with self.assertRaisesRegex(DelegationConfigError, "output.on_stale"):
            self.load("output: {on_stale: {status: maybe}}\n")


if __name__ == "__main__":
    unittest.main()
