# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Browser UI state at ``84affa7`` that the FBA-removal additions build on (plan 2.2.4).

* the UI ``PipelineWorker`` arguments;
* the ``/api/tools`` and ``/api/services`` payloads for FBV, and that ``/api/services`` drops
  unreachable self-hosted entries (the menu behavior 5.1 relies on).

A later change to these pins is allowed only for a deliberate browser-UI change (plan section 5),
with a written reason in the PR.

Changes since ``84affa7``:

* 5.1 (optional local LLM): while reachable, the LLM menu also lists the two
  ``nemotron-3.5-lightning-local`` entries after the Inference Hub entry, which stays selected.

To print a fresh capture:

    PYTHONPATH=src:tests/unit/frontend_backend_verdict python test_fbv_ui_pins.py --print
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from _fbv_eval_capture import jsonable
from _fbv_golden import pinned_environment
from _fbv_pipecat_fakes import EnergyVADAnalyzer, FakeWebSocket, StubRecognizer, StubSTT, ToneSynthesizer, ToneTTS

PINS_FILE = Path(__file__).resolve().parent / "pins" / "ui_pins.json"
FBV = "frontend-backend-verdict-agent"


def _speech(_session: Any) -> Any:
    from examples.frontend_backend_verdict.adapters.pipecat.pipeline import SpeechStack  # noqa: PLC0415

    return SpeechStack(vad=EnergyVADAnalyzer(), stt=StubSTT(StubRecognizer()), tts=ToneTTS(ToneSynthesizer()))


def capture_worker() -> dict[str, Any]:
    from dataclasses import replace  # noqa: PLC0415

    from examples.frontend_backend_verdict.adapters.pipecat import ui_pipeline, ui_session  # noqa: PLC0415
    from examples.frontend_backend_verdict.adapters.pipecat.runtime import PipecatRuntime  # noqa: PLC0415
    from examples.frontend_backend_verdict.adapters.pipecat.settings import load_pipecat_settings  # noqa: PLC0415
    from examples.frontend_backend_verdict.core.agent.factory import ServerOptions  # noqa: PLC0415
    from examples.frontend_backend_verdict.core.config.profiles import DEFAULT_PROFILE, profile_path  # noqa: PLC0415
    from examples.frontend_backend_verdict.core.config.voice import load_voice_config  # noqa: PLC0415
    from examples.shared.pipeline_utils import create_transport  # noqa: PLC0415

    settings = load_pipecat_settings("default.yaml")
    settings = replace(settings, turn=replace(settings.turn, smart_turn=False))
    runtime = PipecatRuntime(
        load_voice_config(profile_path(DEFAULT_PROFILE)),
        profile=DEFAULT_PROFILE,
        settings=settings,
        options=ServerOptions(stub_speech=True, stub_agent="scripted"),
        speech_factory=_speech,
    )
    calls: list[dict[str, Any]] = []
    original = ui_pipeline.PipelineWorker

    def record(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return original(*args, **kwargs)

    args = SimpleNamespace(websocket=FakeWebSocket(), body={"protocol": "rtvi"})
    with mock.patch.object(ui_pipeline, "PipelineWorker", side_effect=record):
        ui_session.UiSession(
            runtime=ui_session._session_runtime(runtime, runtime.config),
            transport=create_transport(args),
            speech=_speech,
        )
    (worker,) = calls
    params = worker["params"]
    return {
        "keys": sorted(worker),
        "enable_rtvi": worker["enable_rtvi"],
        "idle_timeout_secs": worker["idle_timeout_secs"],
        "observers": [type(o).__name__ for o in worker.get("observers") or []],
        "params": {
            name: getattr(params, name)
            for name in ("audio_in_sample_rate", "audio_out_sample_rate", "enable_metrics", "enable_usage_metrics")
        },
    }


def capture_api(reachable: bool) -> dict[str, Any]:
    import server  # noqa: PLC0415
    import utils  # noqa: PLC0415

    app = server.create_app()
    routes = {route.path: route.endpoint for route in app.routes if hasattr(route, "endpoint")}
    import asyncio  # noqa: PLC0415

    with mock.patch.object(utils, "is_endpoint_reachable", return_value=reachable):
        services = asyncio.run(routes["/api/services"](pipeline_mode=FBV))
        tools = asyncio.run(routes["/api/tools"](pipeline_mode=FBV))
    return {"services": jsonable(services), "tools": jsonable(tools)}


def capture_all() -> dict[str, Any]:
    with pinned_environment(), mock.patch.dict(os.environ, {"REALTIME_SERVICE_PLATFORM": "singlegpu"}):
        return {
            "worker": jsonable(capture_worker()),
            "api_reachable": capture_api(True),
            "api_unreachable": capture_api(False),
        }


class UiPinTests(unittest.TestCase):
    def test_matches_the_pins(self) -> None:
        pins = json.loads(PINS_FILE.read_text(encoding="utf-8"))
        captured = json.loads(json.dumps(capture_all(), sort_keys=True))
        for key in ("worker", "api_reachable", "api_unreachable"):
            with self.subTest(key=key):
                self.assertEqual(captured[key], pins[key])

    def test_unreachable_local_entries_leave_the_menu(self) -> None:
        pins = json.loads(PINS_FILE.read_text(encoding="utf-8"))
        for slot, entries in pins["api_unreachable"]["services"].items():
            with self.subTest(slot=slot):
                self.assertFalse([e for e in entries if e.get("source") == "self-hosted"])


if __name__ == "__main__":
    if "--print" in sys.argv:  # PYTHONPATH=src:tests/unit/frontend_backend_verdict
        print(json.dumps(capture_all(), indent=1, sort_keys=True))
    else:
        unittest.main()
