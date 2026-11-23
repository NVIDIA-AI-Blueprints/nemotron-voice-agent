# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The service catalog, `services.yaml`: its recipes, its NVIDIA Cloud endpoints, and its cloud entries."""

from pathlib import Path

import yaml
from loguru import logger

from nemotron_voice_agent.core.paths import services_path

NVCF_GRPC_SERVER = "grpc.nvcf.nvidia.com:443"
NVCF_LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
SELF_HOSTED_RECIPES: tuple[str, ...] = ("server", "single-gpu")
RECIPES: tuple[str, ...] = ("cloud", *SELF_HOSTED_RECIPES)


def load_yaml_file(filepath: Path, *, required: bool = False) -> dict:
    """Load and return the contents of a YAML file as a dict.

    Returns an empty dict if the file is absent, unreadable, malformed, or
    the parsed value is not a mapping. With ``required``, raises
    ``RuntimeError`` in those cases instead.
    """
    try:
        data = yaml.safe_load(filepath.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        if required:
            raise RuntimeError(f"Failed to load YAML from {filepath}: {exc}") from exc
        if filepath.is_file():
            logger.warning(f"Failed to load YAML from {filepath}: {exc}")
        return {}
    if isinstance(data, dict):
        return data
    if required:
        raise RuntimeError(f"YAML root must be a mapping: {filepath}")
    return {}


def load_catalog(path: Path | None = None) -> dict:
    """Load the service catalog, from `SERVICES_PATH` or the repository root unless ``path`` is given."""
    return load_yaml_file(path or services_path(), required=True)


def strip_nvcf(entry: dict) -> dict:
    """Return a self-hosted entry without its NVIDIA Cloud overrides."""
    return {key: value for key, value in entry.items() if key != "nvcf"}


def nvcf_entry(entry: dict) -> dict | None:
    """Return the NVIDIA Cloud variant of a ``server`` entry, or ``None`` when it has no ``nvcf`` key."""
    if "nvcf" not in entry:
        return None
    overrides = entry.get("nvcf") or {}
    if not isinstance(overrides, dict):
        raise RuntimeError(f"services.yaml nvcf must be a mapping (got {overrides!r})")
    out = strip_nvcf(entry)
    if "server" in out:
        out["server"] = NVCF_GRPC_SERVER
    if "base_url" in out:
        out["base_url"] = NVCF_LLM_BASE_URL
    out.update(overrides)
    return out
