# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The repository root and the shared configuration files, which are `/app` and its files in the containers."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _override(variable: str, default: Path) -> Path:
    value = os.getenv(variable, "").strip()
    return Path(value) if value else default


def services_path() -> Path:
    """Return `services.yaml` from `SERVICES_PATH`, or from the repository root."""
    return _override("SERVICES_PATH", REPO_ROOT / "services.yaml")


def registry_path() -> Path:
    """Return `examples_registry.yaml` from the repository root."""
    return REPO_ROOT / "examples_registry.yaml"
