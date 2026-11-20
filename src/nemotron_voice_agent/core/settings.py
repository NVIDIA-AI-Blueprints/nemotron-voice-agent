# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The editable service settings of `settings.yaml`: load the profiles, then validate and apply client values."""

import json
import math
from collections.abc import Mapping

from loguru import logger

from nemotron_voice_agent.core.catalog import load_yaml_file
from nemotron_voice_agent.core.paths import services_path

SETTING_TYPES = frozenset({"bool", "int", "float", "enum"})
NUMBER_TYPES = frozenset({"int", "float"})
MISSING = object()


def _coerce_setting(name: str, spec: dict, raw: object) -> object:
    """Validate one client value against its schema, falling back to the default."""
    default = spec.get("default")
    if raw is None:
        return default
    kind = spec.get("type")
    try:
        if kind == "bool":
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, str) and raw.strip().lower() in {"true", "false"}:
                return raw.strip().lower() == "true"
            raise ValueError
        if kind == "enum":
            if raw in (spec.get("options") or []):
                return raw
            raise ValueError
        if isinstance(raw, bool):
            raise ValueError
        value: int | float = int(raw) if kind == "int" else float(raw)
        if not math.isfinite(value):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        logger.warning(f"Ignoring invalid value for setting {name!r}: {raw!r}")
        return default
    if spec.get("min") is not None:
        value = max(value, spec["min"])
    if spec.get("max") is not None:
        value = min(value, spec["max"])
    return value


def _cap_below(name: str, spec: dict, value: int | float, limit: int | float | None) -> int | float | None:
    """Keep ``value`` at least ``reserve`` under ``limit``, or return ``None`` when that leaves less than ``min``."""
    if limit is None:
        return value
    cap = limit - spec.get("reserve", 0)
    if value <= cap:
        return value
    if cap < spec.get("min", 0):
        turned_off = f" and turning off {spec['requires']!r}" if spec.get("requires") else ""
        logger.warning(f"Dropping setting {name!r}{turned_off}: {spec['below']}={limit}")
        return None
    logger.warning(f"Capping setting {name!r} from {value} to {cap} to stay under {spec['below']}={limit}")
    return cap


def _set_path(target: dict, path: str, value: object) -> None:
    *parents, leaf = path.split(".")
    for part in parents:
        child = target.get(part)
        if not isinstance(child, dict):
            child = target[part] = {}
        target = child
    target[leaf] = value


def _pop_path(target: dict, path: str) -> None:
    *parents, leaf = path.split(".")
    for part in parents:
        target = target.get(part)
        if not isinstance(target, dict):
            return
    target.pop(leaf, None)


def get_path(source: dict, path: str) -> object:
    """Return the value at the dotted ``path`` in ``source``, or ``MISSING``."""
    for part in path.split("."):
        if not isinstance(source, dict) or part not in source:
            return MISSING
        source = source[part]
    return source


def apply_service_settings(extra_params: dict, schema: Mapping[str, dict], values: Mapping[str, object]) -> dict:
    """Return ``extra_params`` with each schema setting written at its ``path``.

    Values come from ``values`` when valid, else the schema default. A setting
    whose ``requires`` setting resolves falsy is removed instead. A ``below``
    setting that cannot stay under its limit turns its ``requires`` setting off.
    """
    resolved = {name: _coerce_setting(name, spec, values.get(name)) for name, spec in schema.items()}

    def active(name: str, spec: dict) -> bool:
        requirement = spec.get("requires")
        return resolved[name] is not None and (not requirement or bool(resolved.get(requirement)))

    for name, spec in schema.items():
        if not spec.get("below") or not active(name, spec):
            continue
        resolved[name] = _cap_below(name, spec, resolved[name], resolved.get(spec["below"]))
        if resolved[name] is None and spec.get("requires"):
            resolved[spec["requires"]] = False
    out = json.loads(json.dumps(extra_params))
    for name, spec in schema.items():
        if not active(name, spec):
            _pop_path(out, spec["path"])
        else:
            _set_path(out, spec["path"], resolved[name])
    return out


def load_settings_profiles() -> dict[str, dict]:
    """Load the ``settings.yaml`` profiles next to the catalog, with each setting's ``path`` defaulting to its name."""
    profiles: dict[str, dict] = {}
    data = load_yaml_file(services_path().with_name("settings.yaml"), required=True)
    for profile_name, profile in data.items():
        if str(profile_name).startswith("x-"):
            continue
        if not isinstance(profile, dict):
            raise RuntimeError(f"settings.yaml profile {profile_name!r} must be a mapping")
        for name, spec in profile.items():
            if not isinstance(spec, dict) or spec.get("type") not in SETTING_TYPES:
                raise RuntimeError(
                    f"settings.yaml {profile_name}.{name} needs a type ({', '.join(sorted(SETTING_TYPES))})"
                )
            below = spec.get("below")
            target = profile.get(below) if below is not None else None
            target_type = target.get("type") if isinstance(target, dict) else None
            if below is not None and (spec["type"] not in NUMBER_TYPES or target_type != spec["type"]):
                raise RuntimeError(
                    f"settings.yaml {profile_name}.{name} below must name a number setting of the same type"
                )
            if "reserve" in spec and below is None:
                raise RuntimeError(f"settings.yaml {profile_name}.{name} reserve needs below")
            reserve = spec.get("reserve", 0)
            if not isinstance(reserve, int) or isinstance(reserve, bool) or reserve < 0:
                raise RuntimeError(f"settings.yaml {profile_name}.{name} reserve must be a non-negative integer")
        profiles[profile_name] = {name: {**spec, "path": spec.get("path") or name} for name, spec in profile.items()}
    return profiles


def entry_settings_schema(profile: Mapping[str, dict], extra_params: dict) -> dict[str, dict]:
    """Use a value already set at a setting's ``path`` in ``extra_params`` as that entry's default."""
    schema: dict[str, dict] = {}
    for name, spec in profile.items():
        value = get_path(extra_params, spec["path"])
        schema[name] = spec if value is MISSING else {**spec, "default": value}
    return schema
