# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "voiceclaw"
FORBIDDEN_PREFIXES = ("examples", "livekit", "nemoclaw", "pipecat", "realtime")


def test_domain_application_and_ports_have_no_framework_imports() -> None:
    checked_roots = (PACKAGE_ROOT / "domain", PACKAGE_ROOT / "application", PACKAGE_ROOT / "ports")

    violations: list[str] = []
    for root in checked_roots:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
                for name in names:
                    if name.startswith(FORBIDDEN_PREFIXES):
                        violations.append(f"{path.relative_to(PACKAGE_ROOT)} imports {name}")

    assert violations == []
