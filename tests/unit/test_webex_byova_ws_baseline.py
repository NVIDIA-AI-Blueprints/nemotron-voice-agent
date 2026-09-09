# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Regression checks for the Webex WebSocket Generic baseline."""

# ruff: noqa: D103

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
GENERIC = ROOT / "src" / "examples" / "generic"
WEBEX_WS = ROOT / "src" / "examples" / "webex_byova_ws"


def test_service_catalogs_match_generic_exactly() -> None:
    for name in ("services.cloud.yaml", "services.local.yaml"):
        assert (WEBEX_WS / name).read_bytes() == (GENERIC / name).read_bytes()


def test_prompt_and_tool_catalogs_match_generic_exactly() -> None:
    for name in ("prompts.yaml", "tools.yaml"):
        assert (WEBEX_WS / name).read_bytes() == (GENERIC / name).read_bytes()


def test_registry_defaults_match_generic() -> None:
    registry = yaml.safe_load((ROOT / "examples_registry.yaml").read_text(encoding="utf-8"))["examples"]
    generic = registry["generic-assistant"]
    webex_ws = registry["webex-byova-ws-assistant"]

    assert webex_ws["slots"] == generic["slots"]
    assert webex_ws["defaults"] == generic["defaults"]
    assert webex_ws["activity_check"] == generic["activity_check"]
    assert webex_ws["bot"] == "examples.webex_byova_ws.pipeline:bot"


def test_pipeline_disables_browser_pacing_for_cisco_only() -> None:
    pipeline = (WEBEX_WS / "pipeline.py").read_text(encoding="utf-8")
    transport = (WEBEX_WS / "transport.py").read_text(encoding="utf-8")

    assert "OutputCompletionObserver" not in pipeline
    assert "audio_out_10ms_chunks" not in pipeline
    assert "AUDIO_OUT_10MS_CHUNKS" not in pipeline
    assert "_write_audio_sleep =" not in pipeline
    assert "create_webex_transport(runner_args)" in pipeline
    assert "class WebexUnpacedOutputTransport" in transport
    assert "await self._write_frame(normalized)" in transport
    assert "build_pipeline_params(" in pipeline
