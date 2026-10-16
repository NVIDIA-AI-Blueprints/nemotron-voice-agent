# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Eval lock: an FBA OpenAI Realtime (eval) session behaves as at ``84affa7``.

Plan ``misc/frontend-backend-verdict/fba-removal-plan.md``, sections 2.2.4 and 2.3. The recorded
values are in ``pins/eval_pins.json``:

* per platform (cloud, ``server``, ``singlegpu``): the session body the route hands FBA and the
  resulting eval ``VoiceConfig``, for a ``REALTIME_API_KEY`` bearer connection and for ``ek_`` keys
  minted from several templates (including unknown LLM ids, which are refused);
* the ``session.created``/``session.updated`` echo and the agent's tools for each Tau fixture;
* today's answers to ``input_audio_buffer.commit``, ``semantic_vad``, ``turn_detection: null`` and
  an ``mcp`` tool.

The same values must hold with the FBA-removal switches unset, with only ``ENABLE_TRACING=true``,
and with every switch on in an eval-mode container (``FBA_PROFILE=eval``).
"""

from __future__ import annotations

import json
import os
import unittest
from typing import Any
from unittest import mock

from _fba_eval_capture import PINS_FILE, SWITCHES_ON, capture_all
from _fba_golden import pinned_environment
from _fba_pipecat_fakes import EnergyVADAnalyzer, FakeWebSocket, StubRecognizer, StubSTT, ToneSynthesizer, ToneTTS

from examples.frontend_backend_agent.adapters.pipecat import pipeline as pipeline_module
from examples.frontend_backend_agent.adapters.pipecat.pipeline import SpeechStack
from examples.frontend_backend_agent.adapters.pipecat.runtime import PipecatRuntime
from examples.frontend_backend_agent.adapters.pipecat.session import PipecatSession
from examples.frontend_backend_agent.adapters.pipecat.settings import load_pipecat_settings
from examples.frontend_backend_agent.core.agent.factory import ServerOptions
from examples.frontend_backend_agent.core.config.profiles import REALTIME_PROFILE, profile_path
from examples.frontend_backend_agent.core.config.voice import load_voice_config

ENVIRONMENTS: dict[str, dict[str, str]] = {
    "switches unset": {},
    "ENABLE_TRACING only": {"ENABLE_TRACING": "true"},
    "every switch on, FBA_PROFILE=eval": {**SWITCHES_ON, "FBA_PROFILE": "eval"},
}
#: Every keyword argument of the Realtime ``PipelineWorker`` (no ``enable_tracing``, no ``observers``).
WORKER_KWARGS = {"enable_rtvi": False, "idle_timeout_secs": None, "cancel_on_idle_timeout": False}


def _pins() -> dict[str, Any]:
    return json.loads(PINS_FILE.read_text(encoding="utf-8"))


def _speech(_session: Any) -> SpeechStack:
    return SpeechStack(
        vad=EnergyVADAnalyzer(), stt=StubSTT(StubRecognizer(), ttfs_p99=3.0), tts=ToneTTS(ToneSynthesizer())
    )


class EvalCaptureTests(unittest.TestCase):
    def test_route_and_session_match_the_pins(self) -> None:
        pins = _pins()
        for name, env in ENVIRONMENTS.items():
            with self.subTest(environment=name):
                captured = json.loads(json.dumps(capture_all(env), sort_keys=True))
                self.assertEqual(captured["route"], pins["route"])
                self.assertEqual(captured["session"], pins["session"])

    def test_unknown_llm_ids_are_refused(self) -> None:
        for platform, route in _pins()["route"].items():
            for case in ("ek_d_unknown_llm_id", "ek_e_unknown_thinker_llm_id"):
                with self.subTest(platform=platform, case=case):
                    self.assertEqual(route[case]["refused"]["code"], "invalid_value")

    def test_ek_keys_log_in_and_their_templates_do_not_change_the_session(self) -> None:
        """5.7: an ``ek_`` key works on FBA's route; FBA reads only the LLM and speech endpoints."""
        for platform, route in _pins()["route"].items():
            for case in ("ek_a_no_template", "ek_b_standard_fields", "ek_c2_canonical_selectors"):
                with self.subTest(platform=platform, case=case):
                    self.assertNotIn("refused", route[case])
                    self.assertEqual(route[case]["voice_config"], route["bearer"]["voice_config"])


class RealtimeBuildTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_and_stop_strategies(self) -> None:
        for name, env in ENVIRONMENTS.items():
            with self.subTest(environment=name):
                calls: dict[str, list[dict[str, Any]]] = {}

                def recorder(target: str, calls: dict[str, list[dict[str, Any]]] = calls) -> Any:
                    original = getattr(pipeline_module, target)

                    def wrapper(*args: Any, **kwargs: Any) -> Any:
                        calls.setdefault(target, []).append(kwargs)
                        return original(*args, **kwargs)

                    return mock.patch.object(pipeline_module, target, side_effect=wrapper)

                with pinned_environment(), mock.patch.dict(os.environ, env):
                    config = load_voice_config(profile_path(REALTIME_PROFILE))
                    patches = [recorder(target) for target in ("PipelineWorker", "UserTurnProcessor")]
                    for patch in patches:
                        patch.start()
                    try:
                        runtime = PipecatRuntime(
                            config,
                            profile=REALTIME_PROFILE,
                            settings=load_pipecat_settings("eval.yaml"),
                            options=ServerOptions(stub_speech=True, stub_agent="scripted"),
                            speech_factory=_speech,
                        )
                        PipecatSession(runtime=runtime, websocket=FakeWebSocket(), speech=_speech)  # type: ignore[arg-type]
                    finally:
                        for patch in patches:
                            patch.stop()
                (worker,) = calls["PipelineWorker"]
                self.assertEqual(set(worker), {"params", *WORKER_KWARGS})
                self.assertEqual({key: worker[key] for key in WORKER_KWARGS}, WORKER_KWARGS)
                (turns,) = calls["UserTurnProcessor"]
                self.assertEqual(set(turns), {"user_turn_strategies"})
                strategies = turns["user_turn_strategies"]
                self.assertEqual([type(s).__name__ for s in strategies.stop], ["SpeechTimeoutUserTurnStopStrategy"])
                self.assertEqual(
                    [type(s).__name__ for s in strategies.start], ["GatedVADStart", "GatedTranscriptionStart"]
                )


if __name__ == "__main__":
    unittest.main()
