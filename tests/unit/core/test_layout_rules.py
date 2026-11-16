# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The import rules of the package layout, read from each module's import statements."""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[3] / "src" / "nemotron_voice_agent"
FRAMEWORKS = ("pipecat", "livekit")


def imports(folder: Path) -> list[tuple[str, str]]:
    """Return ``(module file, imported module)`` for every import under ``folder``, with relative imports resolved."""
    found = []
    for path in sorted(folder.rglob("*.py")):
        file = str(path.relative_to(PACKAGE))
        package = [PACKAGE.name, *path.relative_to(PACKAGE).parent.parts]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                found += [(file, alias.name) for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                found.append((file, node.module))
            elif isinstance(node, ast.ImportFrom):
                base = ".".join(package[: len(package) - node.level + 1])
                if node.module:
                    found.append((file, f"{base}.{node.module}"))
                else:
                    found += [(file, f"{base}.{alias.name}") for alias in node.names]
    return found


def within(name: str, package: str) -> bool:
    """Return whether ``name`` is ``package`` or one of its modules."""
    return name == package or name.startswith(package + ".")


@pytest.mark.parametrize("neutral", ("core", "sidecars"))
def test_a_framework_neutral_package_imports_no_framework(neutral):
    """Core and sidecar code import neither Pipecat nor LiveKit."""
    framework_packages = [*FRAMEWORKS, *(f"nemotron_voice_agent.{name}" for name in FRAMEWORKS)]
    crossing = [
        (file, name) for file, name in imports(PACKAGE / neutral) if any(within(name, p) for p in framework_packages)
    ]
    assert crossing == []


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_a_framework_imports_no_other_framework(framework):
    """Pipecat code never imports LiveKit, and LiveKit code never imports Pipecat."""
    others = [name for name in FRAMEWORKS if name != framework]
    packages = [*others, *(f"nemotron_voice_agent.{name}" for name in others)]
    crossing = [(file, name) for file, name in imports(PACKAGE / framework) if any(within(name, p) for p in packages)]
    assert crossing == []


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_a_runtime_imports_no_example(framework):
    """A framework runtime never imports one of its examples."""
    examples = f"nemotron_voice_agent.{framework}.examples"
    crossing = [(file, name) for file, name in imports(PACKAGE / framework / "runtime") if within(name, examples)]
    assert crossing == []
