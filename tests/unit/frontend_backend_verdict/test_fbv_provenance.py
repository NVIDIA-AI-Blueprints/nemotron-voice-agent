# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Every copied prototype file is byte-identical to the prototype after the recorded rewrites.

``scripts/frontend_backend_verdict_sync.py`` copies the prototype and records each original
file's SHA-256 in ``provenance.json``. Reversing the package rewrite and the listed value
edits must restore those bytes, so a hand edit to copied code fails here unless its entry
records it as ``diverged`` with a reason and a change id. Re-run the sync script to take a
newer prototype commit (it refuses while any file is diverged).
"""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
#: A change id from the analysis report, e.g. ``R1``, or several joined with ``+`` (``R1+R4``).
CHANGE_ID = re.compile(r"^[A-Z][A-Za-z0-9.]*(\+[A-Z][A-Za-z0-9.]*)*$")
EXAMPLE_DIR = REPO_ROOT / "src" / "examples" / "frontend_backend_verdict"
PROVENANCE = json.loads((EXAMPLE_DIR / "provenance.json").read_text(encoding="utf-8"))
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".jsonl", ".md", ".html", ".txt"}


def _restore(text: str, entry: dict) -> str:
    for edit in reversed(entry.get("value_edits", [])):
        text = text.replace(edit["new"], edit["old"])
    for old, new in reversed(PROVENANCE["package_rewrites"]):
        text = text.replace(new, old)
    return text


class ProvenanceTests(unittest.TestCase):
    def test_copied_package_files_match_the_prototype(self) -> None:
        checked = diverged_count = 0
        for key, entry in PROVENANCE["files"].items():
            if entry.get("kind") == "test":
                continue
            path = EXAMPLE_DIR / key
            with self.subTest(file=key):
                self.assertTrue(path.is_file(), f"{key} is missing")
                diverged = entry.get("diverged")
                if diverged is not None:
                    # Edited directly in the example: no byte check, but the edit is explained.
                    self.assertTrue(str(diverged.get("reason", "")).strip(), f"{key} diverged without a reason")
                    self.assertRegex(str(diverged.get("change", "")), CHANGE_ID, f"{key} diverged without a change id")
                    diverged_count += 1
                    continue
                data = path.read_bytes()
                if path.suffix in TEXT_SUFFIXES:
                    data = _restore(data.decode("utf-8"), entry).encode("utf-8")
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"], f"{key} differs from the prototype")
                checked += 1
        self.assertGreater(checked + diverged_count, 90)
        self.assertGreater(checked, diverged_count)

    def test_every_copied_file_is_recorded(self) -> None:
        recorded = {key for key, entry in PROVENANCE["files"].items() if entry.get("kind") != "test"}
        for package in ("text", "voice"):
            for path in (EXAMPLE_DIR / package).rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    rel = path.relative_to(EXAMPLE_DIR).as_posix()
                    self.assertIn(rel, recorded, f"{rel} is not a recorded prototype copy")

    def test_registry_prompt_catalog_is_the_runtime_catalog(self) -> None:
        self.assertEqual(
            (EXAMPLE_DIR / "prompts.yaml").read_bytes(),
            (EXAMPLE_DIR / "voice" / "config" / "prompts.voice.yaml").read_bytes(),
        )

    def test_only_listed_value_edits_exist(self) -> None:
        edited = {key for key, entry in PROVENANCE["files"].items() if entry.get("value_edits")}
        self.assertEqual(edited, {"voice/config/voice_agent.yaml"})
