# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and validate ``config.yaml``. Unknown keys are errors, so a typo cannot silently do nothing.

A profile is a file that names the file it builds on (``extends: config.yaml``) and sets only what differs: a mapping
is merged key by key into the one it extends, and any other value replaces it
(``null`` included, so a profile can turn a setting off). The delegation core's own files (``core.yaml`` and its
profiles) follow the same rule, read by the same loader (``examples.frontend_backend_live.delegation_core.profiles``),
which the composition roots hand in.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .schema import ConfigError, ExampleConfig

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = EXAMPLE_DIR / "config.yaml"

__all__ = ["CONFIG_PATH", "EXAMPLE_DIR", "ConfigError", "config_path", "load_live_config"]


def config_path() -> Path:
    """The example's configuration file: ``LIVE_CONFIG_PATH``, or the bundled ``config.yaml``."""
    return Path(os.environ.get("LIVE_CONFIG_PATH") or CONFIG_PATH)


def load_live_config(load_profile: Callable[[Path], Any], path: Path | None = None) -> ExampleConfig:
    """Read and validate the example's configuration file (default :func:`config_path`), resolving ``extends``.

    ``load_profile`` reads a file merged over the files it ``extends``: the delegation core's loader, which a
    composition root hands in.

    Raises:
        ConfigError: A file cannot be read or parsed, ``extends`` loops or names no file, or the merged settings are
            malformed (``ExampleConfig.parse``).
    """
    try:
        data = load_profile(Path(path or config_path()))
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return ExampleConfig.parse(data)
