# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Static assets for the optional dependency-free VoiceClaw UI."""

from __future__ import annotations

from importlib.resources import files
from importlib.resources.abc import Traversable


def asset_root() -> Traversable:
    """Return the package resource directory containing the UI application."""
    return files(__package__)


def read_asset(name: str) -> str:
    """Read one top-level UTF-8 UI asset by its exact file name."""
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError("UI asset name must be one top-level file name")
    return asset_root().joinpath(name).read_text(encoding="utf-8")


__all__ = ["asset_root", "read_asset"]
