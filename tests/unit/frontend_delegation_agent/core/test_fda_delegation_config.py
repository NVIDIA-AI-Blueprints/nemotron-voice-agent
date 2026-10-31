# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Voice-side profiles: every one loads; 800 ms VAD, normalization and delay invariants; strict keys."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _fda_variants import variant  # noqa: E402
from _fda_voice_fakes import PROFILES

from examples.frontend_delegation_agent.core.config import (
    DelegationConfigError,
    load_delegation_config,
    with_proactive_max_override,
)
from examples.frontend_delegation_agent.core.sidecar import gateway_config as gc

REPO_ROOT = Path(__file__).resolve().parents[4]
SHIPPED = sorted(PROFILES.glob("*.yaml"))
#: Backend variants of the geval failure-fixes plan: on by default, off in gateway.eval_control.yaml.
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

    def test_bench_and_browser_specifics(self) -> None:
        bench = load_delegation_config(variant("profiles/bench_eval.yaml"))
        self.assertEqual(
            (bench.voice.server.port, bench.tools.executor, bench.voice.tools.source), (8775, "wire", "client")
        )
        self.assertTrue(bench.voice.normalization.tool_arguments.enabled)
        self.assertFalse(bench.voice.protocol.greeting_enabled)
        browser = load_delegation_config(PROFILES / "browser_demo.yaml")
        self.assertEqual((browser.voice.server.port, browser.tools.executor), (8776, "local"))
        self.assertFalse(browser.voice.normalization.tool_arguments.enabled)
        self.assertTrue(browser.voice.protocol.greeting_enabled)
        self.assertEqual([spec.name for spec in browser.voice.tools.config_tool_specs], ["get_order", "cancel_order"])

    def test_defaults_turn_on_every_fix_except_m2(self) -> None:
        expected = {"proactive_status", "clean_answers", "spelling_hold", "filler_dedupe", "spelled_runs"}
        # result_hints (I3/I4) needs client tools: on for the benchmark profile, not applicable to the browser demo.
        bench_eval, browser = variant("profiles/bench_eval.yaml"), PROFILES / "browser_demo.yaml"
        for path, extra in ((bench_eval, {"result_hints"}), (browser, set())):
            with self.subTest(profile=path.name):
                features = load_delegation_config(path).features
                self.assertEqual({k for k, v in features.items() if v is True}, expected | extra)
                self.assertFalse(any(features["prompt_features"].values()))  # replay_intent (M2) stays off
        bench = load_delegation_config(variant("profiles/bench_eval.yaml"))
        self.assertEqual(bench.delegation.spelling_hold.complete_patterns, ("^[A-Za-z0-9]{6}$",))
        self.assertEqual(bench.voice.normalization.tool_arguments.invalid_message_key, "tool_argument_invalid_readback")
        self.assertEqual(
            bench.voice.normalization.tool_arguments.escalate_invalid_message_key, "tool_argument_invalid_spell_all"
        )
        hints = bench.voice.normalization.tool_arguments.result_hints
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
        with mock.patch.dict(os.environ, {"FDA_BACKEND_DELAY_S": "2.5"}):
            self.assertEqual(load_delegation_config(PROFILES / "browser_demo.yaml").backend.delay_seconds, 2.5)

    def test_proactive_status_defaults_to_15_s_and_one_line_per_run(self) -> None:
        clean = {k: v for k, v in os.environ.items() if k not in ("FDA_PROACTIVE_AFTER_S", "FDA_PROACTIVE_MAX_PER_RUN")}
        for name in ("realtime_eval.yaml", "pipecat_realtime_eval.yaml"):
            with self.subTest(profile=name), mock.patch.dict(os.environ, clean, clear=True):
                proactive = load_delegation_config(PROFILES / name).output.proactive_status
                self.assertEqual((proactive.enabled, proactive.after_s, proactive.max_per_run), (True, 15.0, 1))

    def test_proactive_status_of_a_big_bench_audio_run(self) -> None:
        # docker/docker-compose.fda-bba.yaml: a status line after 1.0 s of silence, at most 3 per backend run.
        # max_per_run is applied after loading (Realtime sessions only), so the config hash stays as it was.
        clean = {k: v for k, v in os.environ.items() if k not in ("FDA_PROACTIVE_AFTER_S", "FDA_PROACTIVE_MAX_PER_RUN")}
        with mock.patch.dict(os.environ, clean, clear=True):
            base = load_delegation_config(PROFILES / "pipecat_realtime_eval.yaml")
        with mock.patch.dict(os.environ, {**clean, "FDA_PROACTIVE_AFTER_S": "1.0"}, clear=True):
            loaded = load_delegation_config(PROFILES / "pipecat_realtime_eval.yaml")
        config = with_proactive_max_override(loaded, {"FDA_PROACTIVE_MAX_PER_RUN": "3"})
        proactive = config.output.proactive_status
        self.assertEqual((proactive.enabled, proactive.after_s, proactive.max_per_run), (True, 1.0, 3))
        self.assertEqual(config.config_hash, loaded.config_hash)
        for unset in ({}, {"FDA_PROACTIVE_MAX_PER_RUN": ""}, {"FDA_PROACTIVE_MAX_PER_RUN": "  "}):
            with self.subTest(environ=unset):
                self.assertIs(with_proactive_max_override(base, unset), base)
        for value in ("0", "11", "-1", "many", "1.5"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(DelegationConfigError, "FDA_PROACTIVE_MAX_PER_RUN must be a whole number"),
            ):
                with_proactive_max_override(base, {"FDA_PROACTIVE_MAX_PER_RUN": value})

    def test_secrets_are_redacted_and_hash_is_stable(self) -> None:
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "sk-secret"}):
            config = load_delegation_config(variant("profiles/bench_eval.yaml"))
            again = load_delegation_config(variant("profiles/bench_eval.yaml"))
        self.assertEqual(config.effective["frontend"]["llm"]["api_key"], "***")
        self.assertNotIn("sk-secret", str(config.effective))
        self.assertEqual(config.config_hash, again.config_hash)


class ValidationTests(unittest.TestCase):
    def load(self, text: str):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.yaml"
            path.write_text(f"extends: {variant('profiles/bench_eval.yaml')}\n{text}")
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

    def test_a_former_fdh_variable_fails_at_load_not_at_import(self) -> None:
        clean = {k: v for k, v in os.environ.items() if not k.startswith("FDH_")}
        with (
            mock.patch.dict(os.environ, {**clean, "FDH_PROMPT_PROFILE": "generic"}, clear=True),
            self.assertRaisesRegex(DelegationConfigError, r"renamed to FDA_\* \(found: FDH_PROMPT_PROFILE\)"),
        ):
            self.load("")
        # Every example's container reads the same .env: importing FDA (and the shared server) must not fail.
        code = "import examples.frontend_delegation_agent.core.config, examples.frontend_delegation_agent.pipeline"
        env = {**clean, "FDH_BACKEND": "llm", "PYTHONPATH": str(REPO_ROOT / "src")}
        subprocess.run([sys.executable, "-c", code], env=env, cwd=REPO_ROOT, check=True)  # noqa: S603


if __name__ == "__main__":
    unittest.main()
