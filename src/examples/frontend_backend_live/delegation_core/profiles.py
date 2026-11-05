# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Settings files with profiles.

A profile names the file it builds on with ``extends`` (a path relative to the profile's own folder) and sets only what
differs: a mapping is merged key by key into the one it extends, and any other value replaces it (``null`` included,
so a profile can turn a setting off).
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_profile(path: Path) -> Any:
    """The settings of the file at ``path``, merged over the files it ``extends``.

    Raises:
        ValueError: A file cannot be read or parsed, ``extends`` loops or names no file, or a file it extends is not a
            mapping.
    """
    return _profile(Path(path), ())


def _profile(path: Path, seen: tuple[Path, ...]) -> Any:
    """The settings of the file at ``path``, merged over the file it ``extends`` (``seen``: the files extending it)."""
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"{path} extends itself: {' -> '.join(map(str, (*seen, resolved)))}")
    try:
        data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(data, dict) or "extends" not in data:
        return data
    data = dict(data)
    base = data.pop("extends")
    if not isinstance(base, str) or not base.strip():
        raise ValueError(f"extends in {path} must name a file")
    extended = _profile(resolved.parent / base, (*seen, resolved))
    if not isinstance(extended, dict):
        raise ValueError(f"{resolved.parent / base}, which {path} extends, must be a mapping")
    return merged(extended, data)


def merged(base: Any, changes: Any) -> Any:
    """``changes`` over ``base``: mappings merge key by key, anything else replaces."""
    if not isinstance(base, Mapping) or not isinstance(changes, Mapping):
        return deepcopy(changes)
    result = {key: deepcopy(value) for key, value in base.items()}
    for key, value in changes.items():
        result[key] = merged(result[key], value) if key in result else deepcopy(value)
    return result
