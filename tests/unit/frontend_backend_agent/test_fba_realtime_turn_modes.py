# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""5.8: opt-in Realtime turn modes (``FBA_REALTIME_TURN_MODES``), never in eval mode.

Plan ``misc/frontend-backend-verdict/fba-removal-design-5.8-5.9.md`` (approved):

* ``manual``: with ``turn_detection: null`` the VAD no longer ends turns; ``input_audio_buffer.commit``
  does (and is refused with ``input_audio_buffer_commit_empty`` when no turn is in progress);
* ``semantic_vad``: a ``semantic_vad`` request switches to the Smart Turn strategy with a 0.2 s VAD stop;
* unset, or ``FBA_PROFILE=eval``: the pipeline is built and behaves as before (``test_fba_eval_pins.py``).
"""

from __future__ import annotations

import os
import unittest
from typing import Any
from unittest import mock

from _fba_core_fakes import pcmu_silence, pcmu_speech, tau2_session_update
from _fba_golden import pinned_environment
from _fba_pipecat_fakes import PipecatHarness, PipecatTestCase
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy

from examples.frontend_backend_agent.adapters.pipecat import switches, turn_modes
from examples.frontend_backend_agent.core.config.profiles import REALTIME_PROFILE, profile_path
from examples.frontend_backend_agent.core.config.voice import load_voice_config
from examples.frontend_backend_agent.core.engine.scripted import ScriptedAgentPort


class _StubSmartTurn(BaseUserTurnStopStrategy):
    """Stands in for the Smart Turn strategy (its model is not needed here)."""


def _update(turn_detection: Any) -> dict[str, Any]:
    return {
        "type": "session.update",
        "session": {"type": "realtime", "audio": {"input": {"turn_detection": turn_detection}}},
    }


class SwitchTests(unittest.TestCase):
    def test_parsing(self) -> None:
        cases = {
            "": {"server_vad"},
            "manual": {"server_vad", "manual"},
            " Manual , semantic_vad ": {"server_vad", "manual", "semantic_vad"},
            "manual,bogus": {"server_vad", "manual"},
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(switches.realtime_turn_modes({"FBA_REALTIME_TURN_MODES": raw}), expected)

    def test_eval_mode_allows_server_vad_only(self) -> None:
        env = {"FBA_REALTIME_TURN_MODES": "manual,semantic_vad", "FBA_PROFILE": "eval"}
        self.assertEqual(switches.realtime_turn_modes(env), {"server_vad"})


class TurnModeSessionTests(PipecatTestCase):
    def _harness(self, env: dict[str, str]) -> PipecatHarness:
        with pinned_environment(), mock.patch.dict(os.environ, env):
            config = load_voice_config(profile_path(REALTIME_PROFILE))
            with mock.patch(
                "examples.shared.pipeline_utils.build_smart_turn_stop_strategies", return_value=[_StubSmartTurn()]
            ):
                return PipecatHarness(config=config, agent_factory=lambda _: ScriptedAgentPort([{"say": "Okay."}]))

    async def test_off_builds_no_wrapper_and_refuses_commit(self) -> None:
        for env in ({}, {"FBA_REALTIME_TURN_MODES": "manual,semantic_vad", "FBA_PROFILE": "eval"}):
            with self.subTest(env=env):
                harness = self._harness(env)
                self.assertIsNone(harness.session.pipeline.turn_modes)
                await harness.start(tau2_session_update())
                await harness.send({"type": "input_audio_buffer.commit"})
                self.assertEqual(harness.of_type("error")[-1]["error"]["code"], "unsupported_event")
                await harness.close()

    async def test_manual_commit_ends_the_turn(self) -> None:
        harness = self._harness({"FBA_REALTIME_TURN_MODES": "manual"})
        modes = harness.session.pipeline.turn_modes
        self.assertIsNotNone(modes)
        self.assertEqual(modes.allowed, {"server_vad", "manual"})
        await harness.start(tau2_session_update())
        await harness.send(_update(None))
        self.assertEqual(modes.mode, turn_modes.MANUAL)
        controller = harness.session.pipeline.user_turns._user_turn_controller
        self.assertEqual(controller._user_turn_stop_timeout, turn_modes.MANUAL_STOP_TIMEOUT_SECS)

        await harness.send({"type": "input_audio_buffer.commit"})
        self.assertEqual(harness.of_type("error")[-1]["error"]["code"], "input_audio_buffer_commit_empty")

        await harness.feed(pcmu_speech(600) + pcmu_silence(1500))
        self.assertEqual(harness.of_type("input_audio_buffer.committed"), [], "the VAD no longer ends turns")
        self.assertTrue(modes._user_turn)

        errors = len(harness.of_type("error"))
        await harness.send({"type": "input_audio_buffer.commit"})
        await harness.wait_for("input_audio_buffer.committed")
        self.assertEqual(len(harness.of_type("error")), errors)
        self.assertFalse(modes._user_turn)
        # turn_detection null sets create_response false: the client asks for the answer.
        await harness.send({"type": "response.create"})
        await harness.wait_for("response.done")
        self.assertEqual(len(harness.of_type("error")), errors)

        await harness.send(_update({"type": "server_vad"}))
        self.assertEqual(modes.mode, turn_modes.SERVER_VAD)
        self.assertNotEqual(controller._user_turn_stop_timeout, turn_modes.MANUAL_STOP_TIMEOUT_SECS)
        await harness.close()

    async def test_semantic_vad_uses_smart_turn_and_a_short_vad_stop(self) -> None:
        harness = self._harness({"FBA_REALTIME_TURN_MODES": "semantic_vad"})
        modes = harness.session.pipeline.turn_modes
        self.assertEqual(modes.allowed, {"server_vad", "semantic_vad"})
        self.assertIsInstance(modes._strategies["semantic_vad"][0], _StubSmartTurn)
        await harness.start(tau2_session_update())
        self.assertEqual(modes.mode, turn_modes.SERVER_VAD)
        self.assertEqual(harness.session.pipeline._vad_params.stop_secs, 0.8)
        updated = harness.of_type("session.updated")[-1]["session"]["audio"]["input"]["turn_detection"]
        self.assertEqual(updated["x_nvidia_turn_strategy"], "server_vad")

        await harness.send(_update({"type": "semantic_vad", "eagerness": "auto"}))
        self.assertEqual(modes.mode, turn_modes.SEMANTIC_VAD)
        self.assertEqual(harness.session.pipeline._vad_params.stop_secs, turn_modes.SMART_TURN_VAD_STOP_SECS)
        updated = harness.of_type("session.updated")[-1]["session"]["audio"]["input"]["turn_detection"]
        self.assertEqual((updated["type"], updated["x_nvidia_turn_strategy"]), ("semantic_vad", "smart_turn"))

        await harness.send(_update(None))  # manual is not allowed: today's behavior (VAD still ends turns)
        self.assertEqual(modes.mode, turn_modes.SERVER_VAD)
        await harness.send({"type": "input_audio_buffer.commit"})
        self.assertEqual(harness.of_type("error")[-1]["error"]["code"], "unsupported_event")
        await harness.close()


if __name__ == "__main__":
    unittest.main()
