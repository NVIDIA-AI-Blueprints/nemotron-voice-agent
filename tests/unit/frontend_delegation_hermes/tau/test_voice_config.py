# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Voice config loading semantics and the shipped voice profiles (layering is in test_fdh_import_isolation)."""

from __future__ import annotations

import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from _voice_fakes import BASE_CONFIG, PACKAGE_DIR, REPO_ROOT

from examples.frontend_delegation_hermes.core.errors import VoiceConfigError
from examples.frontend_delegation_hermes.core.voice_config import IN_PATH_KEYS, deep_merge, load_voice_config

VOICE_PROFILES = PACKAGE_DIR / "config" / "voice"
SHIPPED = [BASE_CONFIG, *sorted(VOICE_PROFILES.glob("*.yaml"))]


def _load_from(cwd: Path, path: Path):
    previous = Path.cwd()
    os.chdir(cwd)
    try:
        return load_voice_config(path)
    finally:
        os.chdir(previous)


class ShippedConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_every_shipped_file_loads_identically_from_two_working_directories(self) -> None:
        self.assertGreaterEqual(len(SHIPPED), 7)
        with tempfile.TemporaryDirectory() as elsewhere:
            for path in SHIPPED:
                with self.subTest(path=path.name):
                    at_root = _load_from(REPO_ROOT, path)
                    at_other = _load_from(Path(elsewhere), path)
                    self.assertEqual(at_root.resolved_in_paths, at_other.resolved_in_paths)
                    for key, value in at_root.resolved_in_paths.items():
                        if value:
                            self.assertTrue(Path(value).is_absolute(), key)
                            self.assertTrue(Path(value).exists(), key)

    def test_base_defaults_are_tau3_safe(self) -> None:
        config = load_voice_config(BASE_CONFIG)
        self.assertEqual(config.filler.mode, "log_only")
        self.assertEqual(config.tools.source, "client")
        self.assertFalse(config.protocol.greeting_enabled)
        self.assertEqual(config.asr.endpoint.server, "nemo-speech:50051")
        self.assertFalse(config.asr.endpoint.use_ssl)
        self.assertEqual(config.agent.prompts_path, PACKAGE_DIR / "config" / "prompts.voice.yaml")

    def test_voice_profiles_change_only_their_keys(self) -> None:
        base = load_voice_config(VOICE_PROFILES / "base.yaml")
        self.assertEqual(base.turn_detection.silence_duration_ms, 800)
        self.assertFalse(base.turn_detection.honor_client_values)
        browser = load_voice_config(VOICE_PROFILES / "browser.yaml")
        self.assertEqual(browser.tools.source, "config")  # a browser cannot execute tools
        self.assertTrue(browser.protocol.greeting_enabled)
        self.assertTrue(browser.audio.pace_output)
        self.assertEqual([t.name for t in browser.tools.config_tool_specs], ["get_order", "cancel_order"])
        for name in ("tau3.yaml", "realtime_eval.yaml"):
            with self.subTest(profile=name):
                config = load_voice_config(VOICE_PROFILES / name)
                self.assertEqual(config.tools.source, "client")
                self.assertFalse(config.protocol.greeting_enabled)
                self.assertEqual(config.turn_detection.silence_duration_ms, 800)

    def test_nvcf_without_api_key_fails_at_load_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cloud.yaml"
            path.write_text(
                f"extends: {BASE_CONFIG}\n"
                "asr: {catalog: {services: cloud, platform: ''}}\n"
                "tts: {catalog: {services: cloud, platform: ''}}\n",
                encoding="utf-8",
            )
            with (
                mock.patch.dict(os.environ, {"NVIDIA_API_KEY": ""}),
                self.assertRaisesRegex(VoiceConfigError, "NVIDIA_API_KEY"),
            ):
                load_voice_config(path)

    def test_server_override_replaces_only_the_server(self) -> None:
        with mock.patch.dict(os.environ, {"FBA_ASR_SERVER": "localhost:50051"}):
            config = load_voice_config(BASE_CONFIG)
        self.assertEqual(config.asr.endpoint.server, "localhost:50051")
        self.assertEqual(config.asr.endpoint.model, "nemotron-speech-streaming-en-0.6b")
        self.assertEqual(config.tts.endpoint.server, "nemo-speech:50051")


class MergeRuleTests(unittest.TestCase):
    def _profile(self, directory: Path, name: str, body: str) -> Path:
        path = directory / name
        path.write_text(textwrap.dedent(body), encoding="utf-8")
        return path

    def test_merge_table(self) -> None:
        base = {"a": {"b": 1, "c": [1, 2]}, "d": "x"}
        merged = deep_merge(base, {"a": {"c": [3]}, "e": 5}, source="t", defaults={"a": {"b": 0, "c": []}})
        self.assertEqual(merged, {"a": {"b": 1, "c": [3]}, "d": "x", "e": 5})  # lists replace, keys added
        reset = deep_merge(base, {"a": {"b": None}}, source="t", defaults={"a": {"b": 0}})
        self.assertEqual(reset["a"]["b"], 0)  # null resets to the schema default
        with self.assertRaisesRegex(VoiceConfigError, "changes type"):
            deep_merge(base, {"a": "scalar"}, source="t")
        with self.assertRaisesRegex(VoiceConfigError, "changes type"):
            deep_merge(base, {"d": {"x": 1}}, source="t")

    def test_profile_chain_relative_paths_and_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nested").mkdir()
            self._profile(root, "mid.yaml", f"extends: {BASE_CONFIG}\nfiller: {{mode: speak}}\n")
            leaf = self._profile(
                root / "nested", "leaf.yaml", "extends: ../mid.yaml\ninstructions: {apply_to: [frontend]}\n"
            )
            config = load_voice_config(leaf)
            self.assertEqual(config.filler.mode, "speak")
            self.assertEqual(config.instructions.apply_to, ("frontend",))  # lists replace, never concatenate
            self.assertEqual(len(config.source_files), 3)

            self._profile(root, "a.yaml", "extends: b.yaml\n")
            self._profile(root, "b.yaml", "extends: a.yaml\n")
            with self.assertRaisesRegex(VoiceConfigError, "cycle"):
                load_voice_config(root / "a.yaml")

            unknown = self._profile(root, "unknown.yaml", f"extends: {BASE_CONFIG}\nfiller: {{mdoe: speak}}\n")
            with self.assertRaisesRegex(VoiceConfigError, r"filler\.mdoe"):
                load_voice_config(unknown)

            type_change = self._profile(root, "type.yaml", f"extends: {BASE_CONFIG}\nfiller: speak\n")
            with self.assertRaisesRegex(VoiceConfigError, "type.yaml.*filler"):
                load_voice_config(type_change)

            merge = self._profile(root, "merge.yaml", f"extends: {BASE_CONFIG}\ntools: {{source: merge}}\n")
            with self.assertRaisesRegex(VoiceConfigError, "18.1"):
                load_voice_config(merge)

            missing = self._profile(
                root, "missing.yaml", f"extends: {BASE_CONFIG}\ninstructions: {{fallback_file: nope.md}}\n"
            )
            with self.assertRaisesRegex(VoiceConfigError, "instructions.fallback_file.*missing.yaml"):
                load_voice_config(missing)

    def test_in_paths_resolve_against_the_declaring_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "policy.md").write_text("Only help with widgets.", encoding="utf-8")
            profile = self._profile(
                root, "p.yaml", f"extends: {BASE_CONFIG}\ninstructions: {{fallback_file: policy.md}}\n"
            )
            config = _load_from(Path(tempfile.gettempdir()), profile)
            self.assertEqual(config.instructions.fallback_file, str((root / "policy.md").resolve()))
            self.assertEqual(config.instructions.fallback_text, "Only help with widgets.")
        self.assertIn("agent.overrides.prompts.path", IN_PATH_KEYS)


class KeepaliveTests(unittest.TestCase):
    def test_every_fdh_voice_profile_disables_the_websocket_keepalive(self) -> None:
        for path in sorted(VOICE_PROFILES.glob("*.yaml")):
            with self.subTest(profile=path.name):
                self.assertEqual(load_voice_config(path).server.ws_ping_interval_s, 0.0)

    def test_base_config_keeps_the_uvicorn_default(self) -> None:
        config = load_voice_config(BASE_CONFIG)
        self.assertEqual((config.server.ws_ping_interval_s, config.server.ws_ping_timeout_s), (20.0, 20.0))


if __name__ == "__main__":
    unittest.main()
