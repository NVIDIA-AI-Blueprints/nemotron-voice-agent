# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""The startup Pipecat version check stays in step with pyproject.toml and uv.lock."""

# ruff: noqa: D101, D102

from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path
from unittest import mock

import dependency_check

_ROOT = Path(__file__).resolve().parents[2]


class DependencyCheckTests(unittest.TestCase):
    def test_required_version_matches_the_pins(self) -> None:
        pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        pins = [dep for dep in pyproject["project"]["dependencies"] if dep.startswith("pipecat-ai")]
        self.assertEqual([pin.split("==")[-1] for pin in pins], [dependency_check.REQUIRED_PIPECAT_VERSION])
        lock = (_ROOT / "uv.lock").read_text(encoding="utf-8")
        locked = re.search(r'^name = "pipecat-ai"\nversion = "([^"]+)"', lock, flags=re.M)
        self.assertIsNotNone(locked)
        self.assertEqual(locked.group(1), dependency_check.REQUIRED_PIPECAT_VERSION)

    def test_this_environment_matches(self) -> None:
        self.assertIsNone(dependency_check.pipecat_mismatch())

    def test_an_older_image_is_refused_with_a_rebuild_hint(self) -> None:
        message = dependency_check.pipecat_mismatch("1.7.0")
        self.assertIn("1.7.0", message or "")
        self.assertIn("--build", message or "")
        with (
            mock.patch.object(dependency_check, "pipecat_mismatch", return_value="old image"),
            self.assertRaises(SystemExit) as caught,
        ):
            dependency_check.require_matching_pipecat()
        self.assertEqual(str(caught.exception), "old image")

    def test_both_entry_points_run_the_check_first(self) -> None:
        for path in ("src/server.py", "src/realtime_server.py"):
            with self.subTest(path=path):
                source = (_ROOT / path).read_text(encoding="utf-8")
                main = source[source.index("def main(") :]
                body = main[main.index('"""', main.index('"""') + 3) + 3 :].lstrip()
                self.assertTrue(body.startswith("require_matching_pipecat()"), body[:80])


if __name__ == "__main__":
    unittest.main()
