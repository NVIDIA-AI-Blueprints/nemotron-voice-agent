# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The examples registry, `examples_registry.yaml`, which has one section per framework."""

from pathlib import Path

from nemotron_voice_agent.core.catalog import load_yaml_file
from nemotron_voice_agent.core.paths import registry_path


def framework_section(framework: str, path: Path | None = None) -> dict:
    """Return the registry section of one framework, from `REGISTRY_PATH` or the repository root."""
    path = path or registry_path()
    section = load_yaml_file(path, required=True).get(framework)
    if not isinstance(section, dict):
        raise RuntimeError(f"{path} requires a '{framework}' mapping")
    return section
