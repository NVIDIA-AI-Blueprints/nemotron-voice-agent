# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Normalization configuration and the offline replay CLI (the shared normalization layer)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from typing import Any
from unittest import mock

from _voice_fakes import PACKAGE_DIR

from examples.frontend_delegation_hermes.cli.normalization_replay import replay
from examples.frontend_delegation_hermes.core.errors import VoiceConfigError
from examples.frontend_delegation_hermes.core.voice_config import load_voice_config

VOICE_PROFILES = PACKAGE_DIR / "config" / "voice"
SPOKEN = "my user ID is Jordan underscore Lee underscore eight two six four"
WRITTEN = "my user ID is jordan_lee_8264"
NOT_FOUND = "Error: User jordan_lee_8264 not found"


def _env() -> Any:
    return mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"})


with _env():
    PROFILE = load_voice_config(VOICE_PROFILES / "tau3.yaml")


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = _env()
        patcher.start()
        self.addCleanup(patcher.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def load(self, body: str) -> Any:
        path = Path(self._tmp.name) / "profile.yaml"
        path.write_text(f"extends: {PACKAGE_DIR / 'config' / 'voice_agent.yaml'}\n{body}", encoding="utf-8")
        return load_voice_config(path)

    def test_unknown_ruleset(self) -> None:
        with self.assertRaisesRegex(VoiceConfigError, "unknown ruleset"):
            self.load("normalization: {transcript: {ruleset: xx}}\n")

    def test_tool_arguments_need_client_tools(self) -> None:
        body = (
            "tools: {source: config, config_tools: 'examples.frontend_delegation_hermes.core.text.demo_tools:TOOLS'}\n"
            "normalization: {tool_arguments: {enabled: true, rules: [{tool: t, argument: a}]}}\n"
        )
        with self.assertRaisesRegex(VoiceConfigError, "requires tools.source: client"):
            self.load(body)

    def test_retry_guard_needs_a_pattern(self) -> None:
        body = "normalization: {tool_arguments: {enabled: true, retry_guard: {enabled: true, scope: all}}}\n"
        with self.assertRaisesRegex(VoiceConfigError, "permanent_failure_pattern is required"):
            self.load(body)

    def test_bad_rules(self) -> None:
        for rule, message in (
            ("{tool: t}", "argument is required"),
            ("{tool: t, argument: a, colour: red}", "unknown keys"),
            ("{tool: t, argument: a, pattern: '('}", "invalid regular expression"),
            ("{tool: t, argument: a, case: title}", "case must be one of"),
        ):
            with self.subTest(rule=rule), self.assertRaisesRegex(VoiceConfigError, message):
                self.load(f"normalization: {{tool_arguments: {{enabled: true, rules: [{rule}]}}}}\n")

    def test_missing_prompt_key(self) -> None:
        with self.assertRaisesRegex(VoiceConfigError, "not in the catalog"):
            self.load("normalization: {transcript: {enabled: true, frontend_note_key: nope}}\n")

    def test_non_english_asr_warns(self) -> None:
        body = (
            "asr: {source: inline, inline: {server: 'localhost:50051', language_code: de-DE}}\n"
            "normalization: {transcript: {enabled: true}}\n"
        )
        config = self.load(body)
        self.assertTrue(any("ruleset 'en'" in warning for warning in config.warnings))

    def test_normalization_on_in_the_tau3_voice_profiles_and_off_in_the_base(self) -> None:
        for name in ("tau3.yaml", "tau3_fixes.yaml", "realtime_eval.yaml"):
            with self.subTest(profile=name):
                config = load_voice_config(VOICE_PROFILES / name)
                self.assertTrue(config.normalization.transcript.enabled)
                self.assertEqual(config.normalization.transcript.case, "lower")
                self.assertTrue(config.normalization.tool_arguments.enabled)
        base = load_voice_config(PACKAGE_DIR / "config" / "voice_agent.yaml")
        self.assertFalse(base.normalization.transcript.enabled)
        self.assertFalse(base.normalization.tool_arguments.enabled)


class ReplayCliTests(unittest.TestCase):
    def test_replay_rewrites_screens_and_guards_conservatively(self) -> None:
        def calls(call_id: str, user_id: str) -> dict[str, Any]:
            return {
                "kind": "backend_tool_calls",
                "calls": [{"id": call_id, "name": "get_user_details", "arguments": {"user_id": user_id}}],
            }

        records = [
            {"kind": "asr_final", "transcript": SPOKEN},
            calls("c1", "Jordan_Lee_8264"),
            {"kind": "tool_output_in", "call_id": "c1", "output": "Error: User Jordan_Lee_8264 not found"},
            calls("c2", "jordan_lee_8264"),  # the recorded failure was for another string: not blocked
            {"kind": "tool_output_in", "call_id": "c2", "output": NOT_FOUND},
            calls("c3", "JORDAN_LEE_8264"),  # same canonical arguments as c2, which failed: blocked
            calls("c4", "YA"),
        ]
        out = StringIO()
        counts = replay(PROFILE, {"s1": records}, out)
        rows = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(rows[0]["normalized"], WRITTEN)
        self.assertEqual([row["decision"] for row in rows[1:]], ["sent", "sent", "already_failed", "invalid"])
        self.assertEqual(rows[1]["canonical"], {"user_id": "jordan_lee_8264"})
        self.assertEqual((counts["asr_rewritten"], counts["calls_rewritten"]), (1, 3))
