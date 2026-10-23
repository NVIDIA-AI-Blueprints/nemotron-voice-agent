# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Rules 17, 18, 20: config loading semantics, layering, declared imports, and the core/ import rule."""

from __future__ import annotations

import ast
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from _fba_core_fakes import BACKEND_ONLY_PROFILE, BASE_CONFIG, CONFIG_DIR, PACKAGE_ROOT, REPO_ROOT

from examples.frontend_backend_agent.core.config.voice import IN_PATH_KEYS, deep_merge, load_voice_config
from examples.frontend_backend_agent.core.errors import VoiceConfigError

SHIPPED = [BASE_CONFIG, *sorted((CONFIG_DIR / "profiles").glob("*.yaml"))]


def _cloud_speech_profile(directory: Path) -> Path:
    """The former ``cloud_speech.yaml`` profile (deleted in R0), written for the NVCF catalog checks."""
    path = directory / "cloud_speech.yaml"
    path.write_text(
        f"extends: {BASE_CONFIG}\n"
        "asr:\n  catalog: {services: cloud, platform: ''}\n"
        "tts:\n  catalog: {services: cloud, platform: ''}\n",
        encoding="utf-8",
    )
    return path


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
        self.assertEqual(len(SHIPPED), 3)  # base + the "default" (browser) and "eval" profiles
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
        self.assertEqual(config.agent.prompts_path, CONFIG_DIR / "prompts" / "customer_service.yaml")
        self.assertEqual(config.agent.logging.event_sink, "none")

    def test_profiles_change_only_their_keys(self) -> None:
        self.assertEqual(load_voice_config(BACKEND_ONLY_PROFILE).agent.mode, "backend_only")
        with tempfile.TemporaryDirectory() as tmp:
            cloud = load_voice_config(_cloud_speech_profile(Path(tmp)))
        self.assertTrue(cloud.asr.endpoint.use_ssl)
        self.assertTrue(cloud.tts.endpoint.function_id)
        self.assertIn(["function-id", cloud.asr.endpoint.function_id], cloud.asr.endpoint.metadata())

    def test_nvcf_without_api_key_fails_at_load_time(self) -> None:
        with (
            mock.patch.dict(os.environ, {"NVIDIA_API_KEY": ""}),
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(VoiceConfigError, "NVIDIA_API_KEY"),
        ):
            load_voice_config(_cloud_speech_profile(Path(tmp)))

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
#: core/ holds all FBA logic and stays Pipecat-free (plan shared-stack-plan.md §8).
PURE = ["core"]
#: What core/ must never import (plan §8, import rule).
FORBIDDEN_IN_CORE = ("pipecat", "fastapi", "starlette", "realtime", "examples.frontend_backend_agent.adapters")
ALLOWED_TOP_LEVEL = {
    "prototypes",
    "realtime",
    "utils",
    "examples",
    # repository modules in src/ like utils, read by the browser UI adapter (FBA-removal plan 5.2, 5.5):
    # the shared welcome-message switch and the shared tracing setup
    "examples_registry",
    "tracing",
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
    # the agent's prompt templates (core/agent/prompts.py; was text/prompts.py, outside the old voice/ scan)
    "jinja2",
    # the browser profile's tool handlers (tool_handlers.py, copied from frontend_backend_agent/generic)
    "httpx",
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
            target = PACKAGE_ROOT / entry
            files = [target] if target.is_file() else sorted(target.rglob("*.py"))
            for path in files:
                for name in _imports(path):
                    with self.subTest(file=str(path.relative_to(PACKAGE_ROOT)), module=name):
                        self.assertNotIn(name.split(".")[0], FORBIDDEN_IN_PURE_LAYERS)

    def test_every_import_is_stdlib_repo_or_declared(self) -> None:
        stdlib = set(sys.stdlib_module_names)
        for path in PACKAGE_ROOT.rglob("*.py"):
            for name in _imports(path):
                top = name.split(".")[0]
                with self.subTest(file=str(path.relative_to(PACKAGE_ROOT)), module=name):
                    self.assertTrue(top in stdlib or top in ALLOWED_TOP_LEVEL, f"undeclared import {name}")

    @unittest.skip("prototype-repository packaging: this repository declares the dependencies itself")
    def test_declared_extra_lists_the_direct_dependencies(self) -> None:
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        for dist in ("prototypes-voice = [", "nvidia-riva-client", "websockets", "numpy", "fastapi", "uvicorn", "soxr"):
            self.assertIn(dist, pyproject)
        self.assertIn('prototypes-voice-mic = ["nemotron-voice-agent[prototypes-voice]", "sounddevice', pyproject)

    def test_core_imports_no_adapter_pipecat_or_realtime(self) -> None:
        """The core/ import rule (plan §8): core/ imports no voice stack and no adapter."""
        for path in sorted((PACKAGE_ROOT / "core").rglob("*.py")):
            for name in _imports(path):
                with self.subTest(file=str(path.relative_to(PACKAGE_ROOT)), module=name):
                    self.assertFalse(
                        any(name == root or name.startswith(f"{root}.") for root in FORBIDDEN_IN_CORE),
                        f"{path} imports {name}",
                    )
        used = set()
        for path in PACKAGE_ROOT.rglob("*.py"):
            used.update(name for name in _imports(path) if name.startswith("examples.frontend_backend_agent.core"))
        self.assertIn("examples.frontend_backend_agent.core.agent.agent", used)
        self.assertIn("examples.frontend_backend_agent.core.agent.session", used)


class KeepaliveTests(unittest.TestCase):
    def test_profiles_disable_the_websocket_keepalive(self) -> None:
        profiles = CONFIG_DIR / "profiles"
        for path in (profiles / "eval.yaml", profiles / "default.yaml", BACKEND_ONLY_PROFILE):
            config = load_voice_config(path)
            self.assertEqual(config.server.ws_ping_interval_s, 0.0, path.name)

    def test_base_config_keeps_the_uvicorn_default(self) -> None:
        config = load_voice_config(BASE_CONFIG)
        self.assertEqual((config.server.ws_ping_interval_s, config.server.ws_ping_timeout_s), (20.0, 20.0))
