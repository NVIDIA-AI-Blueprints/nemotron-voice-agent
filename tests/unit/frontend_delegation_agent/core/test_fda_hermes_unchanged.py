# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D100, D101, D102, D103, D107

"""Hermes runs exactly as before the backend registry (llm-backend-plan.md section 3).

``fixtures/hermes_snapshot.json`` was recorded from the code before the refactor: for the shipped
gateway configs, with and without ``FDA_PROMPT_PROFILE``, the settings each Hermes worker receives,
the ``SOUL.md`` and ``config.yaml`` bytes of its home, the rendered ephemeral prompt and the
fingerprints. Regenerate it only for an intended Hermes change:
``uv run python tests/unit/frontend_delegation_agent/core/test_fda_hermes_unchanged.py``.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from examples.frontend_delegation_agent.core.backends.hermes import home as homes
from examples.frontend_delegation_agent.core.prompt_features import render_backend_system, sha256_text
from examples.frontend_delegation_agent.core.sidecar import gateway_config as gc
from examples.frontend_delegation_agent.core.sidecar.session_runtime import controller_settings
from examples.frontend_delegation_agent.core.sidecar.templates import BackendTemplates

FIXTURE = Path(__file__).parent / "fixtures" / "hermes_snapshot.json"
REPO_ROOT = Path(__file__).resolve().parents[4]

CONFIGS = ("gateway.yaml", "gateway.eval.yaml")
PROFILES = (None, "general_qa")
INSTRUCTIONS = "You are a helpful airline agent. Follow the policy."
TOOL_SETS = {"telecom": ["get_customer_by_phone", "suspend_line"], "plain": ["get_order", "cancel_order"]}


def _env(profile: str | None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FDA_", "BACKEND_LLM_"))}
    if profile:
        env["FDA_PROMPT_PROFILE"] = profile
    return env


def _rel(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace(str(REPO_ROOT), "<ROOT>")
    if isinstance(value, dict):
        return {k: _rel(v) for k, v in value.items()}
    return value


def _domain(config: gc.GatewayConfig, names: list[str]) -> str:
    for domain, tools_any in config.domains:
        if set(names).intersection(tools_any):
            return domain
    return ""


def snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {}
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)  # the shipped configs resolve output paths against the working directory
    try:
        for name in CONFIGS:
            for profile in PROFILES:
                with mock.patch.dict(os.environ, _env(profile), clear=True):
                    config = gc.load_gateway_config(gc.CONFIG_DIR / name)
                templates = BackendTemplates(config.backend.prompts_path, prompt_features=config.prompt_features)
                soul = templates.render("backend_soul")
                with tempfile.TemporaryDirectory() as tmp:
                    path = homes.create_home(tmp, "w", soul=soul, context_length=config.backend_settings.context_length)
                    soul_md = (path / "SOUL.md").read_text(encoding="utf-8")
                    config_yaml = (path / "config.yaml").read_text(encoding="utf-8")
                systems = {}
                for label, tools in TOOL_SETS.items():
                    note = templates.domain_note(_domain(config, tools))
                    text = render_backend_system(templates, INSTRUCTIONS, note)
                    systems[label] = {"sha256": sha256_text(text), "chars": len(text)}
                out[f"{name}|{profile or '-'}"] = {
                    "worker_settings": _rel(controller_settings(config).worker_backend),
                    "prompts_path": str(Path(config.backend.prompts_path).relative_to(gc.CONFIG_DIR)),
                    "prompt_features": dict(config.prompt_features),
                    "domains": [[d, list(t)] for d, t in config.domains],
                    "soul_md": soul_md,
                    "config_yaml": config_yaml,
                    "backend_catalog_sha256": templates.catalog_sha256,
                    "backend_soul_sha256": sha256_text(soul),
                    "backend_system": systems,
                }
    finally:
        os.chdir(cwd)
    return out


class HermesUnchangedTest(unittest.TestCase):
    def test_hermes_matches_the_recorded_snapshot(self) -> None:
        expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
        actual = snapshot()
        self.assertEqual(sorted(actual), sorted(expected))
        for key, value in expected.items():
            for field, recorded in value.items():
                with self.subTest(case=key, field=field):
                    self.assertEqual(actual[key][field], recorded)

    def test_eval_config_defaults_to_customer_service(self) -> None:
        recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(
            recorded["gateway.eval.yaml|-"]["prompts_path"], "prompts/customer_service/backend/hermes.yaml"
        )


if __name__ == "__main__":
    FIXTURE.write_text(json.dumps(snapshot(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(f"wrote {FIXTURE}\n")
