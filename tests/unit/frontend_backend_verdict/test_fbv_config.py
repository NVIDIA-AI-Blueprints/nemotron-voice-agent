# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Config loading, env interpolation, mode reconciliation, and the pipecat ban."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from _fbv_fakes import make_config

from examples.frontend_backend_verdict.text.config import (
    build_config,
    interpolate_env,
    load_config,
    resolve_runtime_modes,
)
from examples.frontend_backend_verdict.text.errors import ConfigError

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "src" / "examples" / "frontend_backend_verdict" / "text"
FORBIDDEN_IMPORT_ROOTS = {"pipecat"}


def test_env_interpolation_uses_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FBA_TEST_VALUE", raising=False)
    assert interpolate_env({"a": "${FBA_TEST_VALUE:-fallback}"}) == {"a": "fallback"}


def test_env_interpolation_prefers_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FBA_TEST_VALUE", "live")
    assert interpolate_env(["${FBA_TEST_VALUE:-fallback}"]) == ["live"]


def test_missing_variable_without_default_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FBA_TEST_ABSENT", raising=False)
    assert interpolate_env("x${FBA_TEST_ABSENT}y") == "xy"


def test_mode_auto_derives_stateful_backend() -> None:
    assert resolve_runtime_modes("backend_only", "auto") == ("backend_only", True)
    assert resolve_runtime_modes("frontend_backend", "auto") == ("frontend_backend", False)


def test_stateful_backend_conflicts_with_paired_mode() -> None:
    with pytest.raises(ConfigError, match="stateless per delegation"):
        resolve_runtime_modes("frontend_backend", True)


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ConfigError, match="agent.mode"):
        resolve_runtime_modes("solo", "auto")


def test_shipped_default_config_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    config = load_config(PACKAGE_ROOT / "config" / "agent.yaml")
    assert config.frontend_enabled
    assert config.frontend.llm.api_key == "test-key"
    assert config.prompts_path.is_file()


def test_no_pipecat_import() -> None:
    offenders: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name.split(".")[0] in FORBIDDEN_IMPORT_ROOTS for name in names):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []


def test_simulated_delay_defaults_off_and_rejects_negatives() -> None:
    assert make_config().backend.simulated_delay_s == 0.0
    with pytest.raises(ConfigError, match="simulated_delay_s"):
        build_config({"agent": {"mode": "backend_only"}, "backend": {"llm": {"model": "m"}, "simulated_delay_s": -1}})
