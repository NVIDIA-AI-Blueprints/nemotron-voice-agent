# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Rules 17, 18, 20: config loading semantics, layering, declared imports, text prototype untouched."""

from __future__ import annotations

import ast
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from _fbv_voice_fakes import BASE_CONFIG, PACKAGE_DIR, REPO_ROOT

from examples.frontend_backend_verdict.voice.config import IN_PATH_KEYS, deep_merge, load_voice_config
from examples.frontend_backend_verdict.voice.errors import VoiceConfigError

SHIPPED = [BASE_CONFIG, *sorted((PACKAGE_DIR / "config" / "profiles").glob("*.yaml"))]


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
        self.assertEqual(len(SHIPPED), 15)  # R4 added tau3_eval_frontend_verdict_speak_history_write_gate.yaml
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
        self.assertEqual(config.agent.mode, "frontend_backend")
        self.assertEqual(config.filler.mode, "log_only")
        self.assertEqual(config.tools.source, "client")
        self.assertEqual(config.agent.backend.tools.execution, "external")
        self.assertFalse(config.protocol.greeting_enabled)
        self.assertEqual(config.asr.endpoint.server, "nemo-speech:50051")
        self.assertFalse(config.asr.endpoint.use_ssl)
        self.assertEqual(config.agent.prompts_path, PACKAGE_DIR / "config" / "prompts.voice.yaml")
        self.assertEqual(config.agent.logging.event_sink, "none")

    def test_profiles_change_only_their_keys(self) -> None:
        profiles = PACKAGE_DIR / "config" / "profiles"
        self.assertEqual(load_voice_config(profiles / "backend_only.yaml").agent.mode, "backend_only")
        cloud = load_voice_config(profiles / "cloud_speech.yaml")
        self.assertTrue(cloud.asr.endpoint.use_ssl)
        self.assertTrue(cloud.tts.endpoint.function_id)
        self.assertIn(["function-id", cloud.asr.endpoint.function_id], cloud.asr.endpoint.metadata())
        live = load_voice_config(profiles / "live_demo.yaml")
        self.assertEqual(live.filler.mode, "speak")
        self.assertEqual(live.tools.source, "config")
        self.assertEqual(live.agent.backend.tools.execution, "internal")
        self.assertEqual([t.name for t in live.tools.config_tool_specs], ["get_order", "cancel_order"])
        browser = load_voice_config(profiles / "browser_demo.yaml")
        self.assertEqual(browser.filler.mode, "log_only")  # silent; the page shows it in gray
        self.assertEqual(browser.tools.source, "config")  # a browser cannot execute tools
        self.assertTrue(browser.protocol.greeting_enabled)
        self.assertTrue(browser.audio.pace_output)
        slow = load_voice_config(profiles / "browser_demo_slow_backend.yaml")
        self.assertEqual(slow.filler.mode, "speak")  # the frontend's filler is heard
        self.assertEqual(slow.agent.backend.simulated_delay_s, 5.0)
        self.assertEqual(browser.agent.backend.simulated_delay_s, 0.0)

    def test_nvcf_without_api_key_fails_at_load_time(self) -> None:
        with (
            mock.patch.dict(os.environ, {"NVIDIA_API_KEY": ""}),
            self.assertRaisesRegex(VoiceConfigError, "NVIDIA_API_KEY"),
        ):
            load_voice_config(PACKAGE_DIR / "config" / "profiles" / "cloud_speech.yaml")

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


FORBIDDEN_IN_PURE_LAYERS = ("pipecat", "riva", "fastapi")
PURE = ["wire", "agent", "audio", "engine/segmenter.py", "engine/turn_manager.py", "engine/playback.py"]
ALLOWED_TOP_LEVEL = {
    "prototypes",
    "realtime",
    "utils",
    "examples",
    # distributions declared in [project.optional-dependencies] prototypes-voice(-mic)
    "pipecat",
    "riva",
    "fastapi",
    "uvicorn",
    "websockets",
    "numpy",
    "soxr",
    "yaml",
    "loguru",
    "openai",
    "rich",
    "sounddevice",
    "dotenv",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


class LayeringTests(unittest.TestCase):
    def test_pure_layers_import_no_pipecat_riva_or_fastapi(self) -> None:
        for entry in PURE:
            target = PACKAGE_DIR / entry
            files = [target] if target.is_file() else sorted(target.rglob("*.py"))
            for path in files:
                for name in _imports(path):
                    with self.subTest(file=str(path.relative_to(PACKAGE_DIR)), module=name):
                        self.assertNotIn(name.split(".")[0], FORBIDDEN_IN_PURE_LAYERS)

    def test_every_import_is_stdlib_repo_or_declared(self) -> None:
        stdlib = set(sys.stdlib_module_names)
        for path in PACKAGE_DIR.rglob("*.py"):
            if "tau2_gates" in path.parts:
                continue  # run with the tau2 checkout's interpreter; they import tau2 by design
            for name in _imports(path):
                top = name.split(".")[0]
                with self.subTest(file=str(path.relative_to(PACKAGE_DIR)), module=name):
                    self.assertTrue(top in stdlib or top in ALLOWED_TOP_LEVEL, f"undeclared import {name}")

    @unittest.skip("prototype-repository packaging: this repository declares the dependencies itself")
    def test_declared_extra_lists_the_direct_dependencies(self) -> None:
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        for dist in ("prototypes-voice = [", "nvidia-riva-client", "websockets", "numpy", "fastapi", "uvicorn", "soxr"):
            self.assertIn(dist, pyproject)
        self.assertIn('prototypes-voice-mic = ["nemotron-voice-agent[prototypes-voice]", "sounddevice', pyproject)

    def test_text_prototype_is_imported_never_modified(self) -> None:
        text_pkg = REPO_ROOT / "src" / "examples" / "frontend_backend_verdict" / "text"
        for path in text_pkg.rglob("*.py"):
            for name in _imports(path):
                self.assertFalse(name.startswith("examples.frontend_backend_verdict.voice"), f"{path} imports voice")
        used = set()
        for path in PACKAGE_DIR.rglob("*.py"):
            used.update(name for name in _imports(path) if name.startswith("examples.frontend_backend_verdict.text"))
        self.assertIn("examples.frontend_backend_verdict.text.agent", used)
        self.assertIn("examples.frontend_backend_verdict.text.session", used)


class KeepaliveTests(unittest.TestCase):
    def test_tau3_profiles_disable_the_websocket_keepalive(self) -> None:
        for name in ("tau3_eval", "backend_only"):
            config = load_voice_config(PACKAGE_DIR / "config" / "profiles" / f"{name}.yaml")
            self.assertEqual(config.server.ws_ping_interval_s, 0.0, name)

    def test_base_config_keeps_the_uvicorn_default(self) -> None:
        config = load_voice_config(BASE_CONFIG)
        self.assertEqual((config.server.ws_ping_interval_s, config.server.ws_ping_timeout_s), (20.0, 20.0))
