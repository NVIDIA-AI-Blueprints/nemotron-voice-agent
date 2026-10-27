# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and validate ``config.yaml``. Unknown keys are errors, so a typo cannot silently do nothing.

A profile is a file that names the file it builds on (``extends: config.yaml``) and sets only what differs, as
``config.bench.yaml`` does: a mapping is merged key by key into the one it extends, and any other value replaces it
(``null`` included, so a profile can turn a setting off). The delegation core's own files (``core.yaml`` and its
profiles) follow the same rule.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .schema import ConfigError, ExampleConfig

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = EXAMPLE_DIR / "config.yaml"

__all__ = ["CONFIG_PATH", "EXAMPLE_DIR", "ConfigError", "config_path", "load_live_config"]


def config_path() -> Path:
    """The example's configuration file: ``LIVE_CONFIG_PATH``, or the bundled ``config.yaml``."""
    return Path(os.environ.get("LIVE_CONFIG_PATH") or CONFIG_PATH)


def load_live_config(path: Path | None = None) -> ExampleConfig:
    """Read and validate the example's configuration file (default :func:`config_path`), resolving ``extends``.

    Raises:
        ConfigError: A file cannot be read or parsed, ``extends`` loops or names no file, or the merged settings are
            malformed (``ExampleConfig.parse``).
    """
    return ExampleConfig.parse(_profile(Path(path or config_path()), ()))


def _profile(path: Path, seen: tuple[Path, ...]) -> Any:
    """The settings of the file at ``path``, merged over the file it ``extends`` (``seen``: the files extending it)."""
    resolved = path.resolve()
    if resolved in seen:
        raise ConfigError(f"{path} extends itself: {' -> '.join(map(str, (*seen, resolved)))}")
    try:
        data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(data, dict) or "extends" not in data:
        return data
    data = dict(data)
    base = data.pop("extends")
    if not isinstance(base, str) or not base.strip():
        raise ConfigError(f"extends in {path} must name a file")
    extended = _profile(resolved.parent / base, (*seen, resolved))
    if not isinstance(extended, dict):
        raise ConfigError(f"{resolved.parent / base}, which {path} extends, must be a mapping")
    return _merged(extended, data)


def _merged(base: Any, changes: Any) -> Any:
    """``changes`` over ``base``: mappings merge key by key, anything else replaces."""
    if not isinstance(base, Mapping) or not isinstance(changes, Mapping):
        return deepcopy(changes)
    merged = {key: deepcopy(value) for key, value in base.items()}
    for key, value in changes.items():
        merged[key] = _merged(merged[key], value) if key in merged else deepcopy(value)
    return merged
