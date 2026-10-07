# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The prototype sync script refuses as a whole, before any deletion, when local files would be lost.

Each case runs the real ``scripts/frontend_backend_verdict_sync.py`` against a throwaway
prototype git repository and example tree, then compares the hash of every file in the
example and test trees before and after.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "frontend_backend_verdict_sync.py"

PROTOTYPE_FILES = {
    "src/prototypes/text_frontend_backend_agent/__init__.py": "",
    "src/prototypes/text_frontend_backend_agent/agent.py": (
        "from prototypes.text_frontend_backend_agent import tools\n"
    ),
    "src/prototypes/voice_frontend_backend_agent/__init__.py": "",
    "src/prototypes/voice_frontend_backend_agent/engine.py": (
        "from prototypes.voice_frontend_backend_agent import agent\n"
    ),
    "src/prototypes/voice_frontend_backend_agent/config/voice_agent.yaml": (
        'config: "../../text_frontend_backend_agent/config/agent.yaml"   # [in-path]\n'
        'asr: {example_dir: "../../../examples/frontend_backend_agent"   # [in-path]}\n'
        'tts: {example_dir: "../../../examples/frontend_backend_agent"   # [in-path]}\n'
    ),
    "src/prototypes/voice_frontend_backend_agent/config/prompts.voice.yaml": "greeting: hello\n",
    "tests/unit/prototypes/_fakes.py": "FAKE = 1\n",
    "tests/unit/prototypes/test_agent.py": "from _fakes import FAKE\n",
    "tests/unit/prototypes/voice/test_voice_engine.py": ("import prototypes.voice_frontend_backend_agent.engine\n"),
    "tests/unit/prototypes/voice/fixtures/case.json": '{"package": "prototypes.voice_frontend_backend_agent"}\n',
    "misc/prototypes/voice/verdict_cases.jsonl": "{}\n",
}


def _load_script():
    spec = importlib.util.spec_from_file_location("fbv_sync_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree_hashes(*roots: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                hashes[path.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


@unittest.skipIf(shutil.which("git") is None, "git is required")
class SyncPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.proto = root / "prototype"
        for rel, text in PROTOTYPE_FILES.items():
            path = self.proto / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        # A throwaway repository: never sign, prompt, or read the user's git configuration.
        env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0"}
        git = ["git", "-C", str(self.proto), "-c", "user.name=t", "-c", "user.email=t@example.com"]
        git += ["-c", "commit.gpgsign=false"]
        subprocess.run(["git", "init", "-q", str(self.proto)], check=True, env=env, timeout=30)
        subprocess.run([*git, "add", "-A"], check=True, env=env, timeout=30)
        subprocess.run([*git, "commit", "-q", "-m", "prototype"], check=True, env=env, timeout=30)

        self.sync = _load_script()
        self.repo = root / "repo"
        self.example = self.repo / "example"
        self.tests = self.repo / "tests"
        self.backups = self.repo / "backups"
        self.example.mkdir(parents=True)
        self.sync.REPO_ROOT = self.repo
        self.sync.EXAMPLE_DIR = self.example
        self.sync.TEST_DIR = self.tests
        self.sync.PROVENANCE_FILE = self.example / "provenance.json"
        self.sync.BACKUP_ROOT = self.backups
        self._run()  # a clean first sync

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, *extra: str) -> int:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                self.sync.main(["--prototype-root", str(self.proto), *extra])
            except SystemExit as exc:
                return int(exc.code or 0)
        return 0

    def _assert_refused_unchanged(self) -> None:
        before = _tree_hashes(self.example, self.tests)
        self.assertNotEqual(self._run(), 0)
        self.assertEqual(_tree_hashes(self.example, self.tests), before)
        self.assertFalse(self.backups.exists())

    def test_clean_tree_syncs_again(self) -> None:
        self.assertEqual(self.sync.preflight(self.proto), [])
        before = _tree_hashes(self.example, self.tests)
        self.assertEqual(self._run(), 0)
        self.assertEqual(_tree_hashes(self.example, self.tests), before)

    def test_diverged_entry_refuses(self) -> None:
        provenance = json.loads(self.sync.PROVENANCE_FILE.read_text(encoding="utf-8"))
        provenance["files"]["voice/engine.py"]["diverged"] = {"reason": "edited in the example", "change": "R1"}
        self.sync.PROVENANCE_FILE.write_text(json.dumps(provenance), encoding="utf-8")
        with (self.example / "voice" / "engine.py").open("a", encoding="utf-8") as handle:
            handle.write("RESUME = True\n")
        self._assert_refused_unchanged()

    def test_unrecorded_edit_refuses(self) -> None:
        with (self.example / "text" / "agent.py").open("a", encoding="utf-8") as handle:
            handle.write("LOCAL = True\n")
        self._assert_refused_unchanged()

    def test_extra_file_refuses(self) -> None:
        (self.example / "voice" / "write_gate.py").write_text("GATE = True\n", encoding="utf-8")
        self._assert_refused_unchanged()

    def test_edited_ported_test_refuses(self) -> None:
        with (self.tests / "test_fbv_agent.py").open("a", encoding="utf-8") as handle:
            handle.write("# local\n")
        self._assert_refused_unchanged()

    def test_extra_fixture_refuses(self) -> None:
        (self.tests / "fixtures" / "local_case.json").write_text("{}\n", encoding="utf-8")
        self._assert_refused_unchanged()

    def test_force_backs_up_every_listed_file_then_syncs(self) -> None:
        engine = self.example / "voice" / "engine.py"
        extra = self.example / "voice" / "write_gate.py"
        engine.write_text(engine.read_text(encoding="utf-8") + "LOCAL = True\n", encoding="utf-8")
        extra.write_text("GATE = True\n", encoding="utf-8")
        listed = {path for path, _ in self.sync.preflight(self.proto)}
        self.assertEqual(listed, {engine, extra})

        self.assertEqual(self._run("--force", "--backup-dir", str(self.backups)), 0)

        self.assertIn("LOCAL = True", (self.backups / "example" / "voice" / "engine.py").read_text(encoding="utf-8"))
        self.assertEqual(
            (self.backups / "example" / "voice" / "write_gate.py").read_text(encoding="utf-8"), "GATE = True\n"
        )
        self.assertNotIn("LOCAL = True", engine.read_text(encoding="utf-8"))
        self.assertFalse(extra.exists())
        self.assertEqual(self.sync.preflight(self.proto), [])


if __name__ == "__main__":
    unittest.main()
