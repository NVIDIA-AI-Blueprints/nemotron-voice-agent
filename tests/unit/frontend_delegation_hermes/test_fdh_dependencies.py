# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""C13: the image lock's package list and the ``fdh`` extra name the same packages; the lock pins Pipecat 1.11."""

from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK = REPO_ROOT / "docker" / "fdh" / "requirements.txt"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _name(requirement: str) -> str:
    return re.split(r"[\[<>=!~ ;@]", requirement.strip(), maxsplit=1)[0].lower().replace("_", "-")


def _header_packages() -> set[str]:
    """The package specifiers quoted in the header's ``printf`` recipe (the Hermes checkout excluded)."""
    header = "\n".join(line for line in LOCK.read_text(encoding="utf-8").splitlines() if line.startswith("#"))
    specifiers = re.findall(r"'([A-Za-z][A-Za-z0-9_.\-]*(?:\[[^\]']*\])?[<>=!~][^'\s]*)'", header)
    return {_name(item) for item in specifiers}


class DependencyTests(unittest.TestCase):
    def test_lock_header_lists_the_fdh_extra(self) -> None:
        project = tomllib.loads(PYPROJECT.read_text())["project"]
        extra = {_name(item) for item in project["optional-dependencies"]["fdh"]}
        self.assertEqual(_header_packages() - {"pipecat-ai"}, extra)

    def test_lock_pins_the_repository_pipecat(self) -> None:
        dependencies = tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]
        repo_pipecat = next(item for item in dependencies if _name(item) == "pipecat-ai").split("==")[1]
        locked = re.search(r"^pipecat-ai==(\S+)", LOCK.read_text(encoding="utf-8"), re.MULTILINE)
        assert locked is not None
        self.assertEqual(locked.group(1), repo_pipecat)
        self.assertIn(f"'pipecat-ai[silero,nvidia,websocket]=={repo_pipecat}'", LOCK.read_text(encoding="utf-8"))

    def test_dev_group_installs_the_fdh_extra(self) -> None:
        self.assertIn("nemotron-voice-agent[fdh]", tomllib.loads(PYPROJECT.read_text())["dependency-groups"]["dev"])


if __name__ == "__main__":
    unittest.main()
