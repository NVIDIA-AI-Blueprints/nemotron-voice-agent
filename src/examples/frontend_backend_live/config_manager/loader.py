# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and validate ``config.yaml``. Unknown keys are errors, so a typo cannot silently do nothing."""

from __future__ import annotations

import os
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from .schema import DelegationConfig, GuardConfig, LiveConfig, ReliabilityConfig, RoleConfig

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
PROMPT_VERSIONS = ("v1", "v2")
# Every other number is a timeout, a limit or a count of rounds, where 0 would disable or break the feature.
ZERO_ALLOWED = frozenset({"provider_retries"})


class ConfigError(ValueError):
    """Raised when ``config.yaml`` is malformed."""


def _section(data: dict, name: str, allowed: set[str]) -> dict[str, Any]:
    """Return ``data[name]`` as a mapping, rejecting unknown keys."""
    raw = data.get(name, {})
    if not isinstance(raw, dict):
        raise ConfigError(f"{name} must be a mapping")
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"{name} has unknown keys: {', '.join(sorted(unknown))}")
    return raw


def _typed(section: str, values: dict[str, Any], cls: type) -> dict[str, Any]:
    """Check each value against the type of the default it replaces."""
    defaults = cls()
    checked = {}
    for name, value in values.items():
        expected = type(getattr(defaults, name))
        if expected is tuple:
            if not (isinstance(value, list) and all(isinstance(v, str) for v in value)):
                raise ConfigError(f"{section}.{name} must be a list of strings")
            value = tuple(value)
        elif expected is bool:
            if not isinstance(value, bool):
                raise ConfigError(f"{section}.{name} must be true or false")
        elif expected in (int, float):
            minimum = "non-negative" if name in ZERO_ALLOWED else "positive"
            valid = not isinstance(value, bool) and isinstance(value, (int, float))
            if not valid or value < 0 or (value == 0 and name not in ZERO_ALLOWED):
                raise ConfigError(f"{section}.{name} must be a {minimum} number")
            value = expected(value)
        checked[name] = value
    return checked


def _role(data: dict, name: str, default_prompt: str) -> RoleConfig:
    """Parse the ``frontend`` or ``backend`` section."""
    raw = _section(data, name, {"provider", "slot", "prompt"})
    provider, slot, prompt = raw.get("provider"), raw.get("slot"), raw.get("prompt", default_prompt)
    if not isinstance(provider, str) or not provider or not isinstance(slot, str) or not slot:
        raise ConfigError(f"{name} needs a provider (registered implementation) and a slot (services slot)")
    if not isinstance(prompt, str) or not prompt:
        raise ConfigError(f"{name}.prompt must be the name of a prompts.yaml entry")
    return RoleConfig(provider, slot, prompt)


def parse_config(data: Any) -> LiveConfig:
    """Validate parsed YAML and return a :class:`LiveConfig`."""
    if not isinstance(data, dict):
        raise ConfigError("config.yaml must be a mapping")
    top = {
        "frontend",
        "backend",
        "prompt_version",
        "reliability",
        "guards",
        "delegation",
        "backend_context_tokens",
        "chat_history_recent_turns",
    }
    unknown = set(data) - top
    if unknown:
        raise ConfigError(f"config.yaml has unknown keys: {', '.join(sorted(unknown))}")
    version = data.get("prompt_version", "v1")
    if version not in PROMPT_VERSIONS:
        raise ConfigError(f"prompt_version must be one of {', '.join(PROMPT_VERSIONS)}")
    reliability = _section(data, "reliability", {f.name for f in fields(ReliabilityConfig)})
    guards = _section(data, "guards", {f.name for f in fields(GuardConfig)})
    delegation = _section(data, "delegation", {f.name for f in fields(DelegationConfig)})
    extra = {}
    for key in ("backend_context_tokens", "chat_history_recent_turns"):
        if key in data:
            value = data[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigError(f"{key} must be a positive integer")
            extra[key] = value
    return LiveConfig(
        frontend=_role(data, "frontend", "talker"),
        backend=_role(data, "backend", "thinker"),
        prompt_version=version,
        reliability=ReliabilityConfig(**_typed("reliability", reliability, ReliabilityConfig)),
        guards=GuardConfig(**_typed("guards", guards, GuardConfig)),
        delegation=DelegationConfig(**_typed("delegation", delegation, DelegationConfig)),
        **extra,
    )


def load_live_config(path: Path | None = None) -> LiveConfig:
    """Read and validate the example's configuration file (``LIVE_CONFIG_PATH`` overrides the bundled one)."""
    path = path or Path(os.environ.get("LIVE_CONFIG_PATH") or CONFIG_PATH)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot read {path}: {exc}") from exc
    return parse_config(data)
