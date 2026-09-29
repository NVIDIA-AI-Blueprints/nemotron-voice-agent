# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams, VADState
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy, VADUserTurnStartStrategy

from examples.shared.pipeline_utils import (
    build_silero_vad_analyzer,
    build_vad_params,
    build_vad_user_turn_start_strategies,
    realtime_vad_prefix_padding_secs,
)
from realtime.gateway import _controller_from_runtime
from realtime.protocol import RealtimeProtocolError
from realtime.transport import realtime_input_audio_processors
from realtime.vad import RealtimeSileroVADAnalyzer


def _gateway_controller(pipeline_mode: str, *, server_vad: bool = False):
    with patch.dict(
        os.environ,
        {"USE_SILERO_VAD_TURN_DETECTION": "true" if server_vad else "false"},
    ):
        return _controller_from_runtime(
            {
                "pipeline_mode": pipeline_mode,
                "model_id": "test-model",
                "tts_voice_id": "test-voice",
                "asr_model": "test-asr",
            },
            server_tools=[],
            delegate_tools=[],
        )


class RealtimeVADPipelineProjectionTests(unittest.TestCase):
    def test_omni_analyzer_selection_is_realtime_only(self) -> None:
        defaults = VADParams()
        ordinary = build_silero_vad_analyzer(defaults)
        with patch(
            "examples.shared.pipeline_utils.realtime_turn_detection_config",
            return_value={"type": "semantic_vad", "eagerness": "auto"},
        ):
            realtime = build_silero_vad_analyzer(defaults, transport=object())

        self.assertIs(type(ordinary), SileroVADAnalyzer)
        self.assertIsInstance(realtime, RealtimeSileroVADAnalyzer)

    def test_server_vad_values_project_to_pipecat_analyzer(self) -> None:
        defaults = VADParams(confidence=0.5, start_secs=0.2, stop_secs=0.5, min_volume=0.6)
        with patch(
            "examples.shared.pipeline_utils.realtime_turn_detection_config",
            return_value={
                "type": "server_vad",
                "threshold": 0.72,
                "silence_duration_ms": 875,
            },
        ):
            params = build_vad_params(defaults, transport=object())

        self.assertEqual(params.confidence, 0.72)
        self.assertEqual(params.stop_secs, 0.875)
        self.assertEqual(params.start_secs, defaults.start_secs)
        self.assertEqual(params.min_volume, defaults.min_volume)

    def test_effective_openai_server_vad_defaults_override_pipeline_defaults(self) -> None:
        defaults = VADParams(confidence=0.42, start_secs=0.15, stop_secs=0.9, min_volume=0.55)
        controller = _gateway_controller("generic-assistant", server_vad=True)
        with patch("realtime.transport.realtime_controller", return_value=controller):
            params = build_vad_params(defaults, transport=object())

        self.assertEqual(params.confidence, 0.5)
        self.assertEqual(params.stop_secs, 0.5)
        self.assertEqual(params.start_secs, defaults.start_secs)
        self.assertEqual(params.min_volume, defaults.min_volume)
        self.assertIsNot(params, defaults)

    def test_semantic_vad_does_not_project_server_analyzer_tuning(self) -> None:
        defaults = VADParams(confidence=0.42, stop_secs=0.9)
        with patch(
            "examples.shared.pipeline_utils.realtime_turn_detection_config",
            return_value={"type": "semantic_vad", "eagerness": "auto"},
        ):
            params = build_vad_params(defaults, transport=object())

        self.assertIs(params, defaults)

    def test_interrupt_response_false_disables_both_turn_start_interruptions(self) -> None:
        with patch(
            "examples.shared.pipeline_utils.realtime_turn_detection_config",
            return_value={"type": "server_vad", "interrupt_response": False},
        ):
            strategies = build_vad_user_turn_start_strategies(
                transport=object(),
                include_transcription=True,
            )

        self.assertEqual(len(strategies), 2)
        self.assertIsInstance(strategies[0], VADUserTurnStartStrategy)
        self.assertIsInstance(strategies[1], TranscriptionUserTurnStartStrategy)
        self.assertTrue(all(strategy._enable_interruptions is False for strategy in strategies))

    def test_server_vad_prefix_padding_projects_to_fused_pre_speech_buffer(self) -> None:
        controller = SimpleNamespace(
            turn_detection_type="server_vad",
            server_vad_prefix_padding_ms=425,
        )
        with patch("realtime.transport.realtime_controller", return_value=controller):
            value = realtime_vad_prefix_padding_secs(0.2, transport=object())
        self.assertEqual(value, 0.425)

    def test_semantic_vad_preserves_fused_pre_speech_buffer_default(self) -> None:
        controller = SimpleNamespace(
            turn_detection_type="semantic_vad",
            server_vad_prefix_padding_ms=0,
        )
        with patch("realtime.transport.realtime_controller", return_value=controller):
            value = realtime_vad_prefix_padding_secs(0.2, transport=object())
        self.assertEqual(value, 0.2)

    def test_cascaded_semantic_vad_retains_detector_pre_speech_audio(self) -> None:
        context = SimpleNamespace(
            controller=SimpleNamespace(
                manual_input_mode=False,
                turn_detection_type="semantic_vad",
                turn_detection_config={"type": "semantic_vad", "eagerness": "auto"},
            ),
            asr_input_sequencer=None,
        )
        contexts = MagicMock()
        contexts.get.return_value = context
        analyzer = MagicMock()
        analyzer.params = VADParams()
        with (
            patch("realtime.transport._CONTEXTS", contexts),
            patch("realtime.vad.RealtimeSileroVADAnalyzer", return_value=analyzer) as analyzer_cls,
        ):
            vad, _sequencer = realtime_input_audio_processors(object())

        self.assertEqual(vad._vad_prefix_padding_secs, VADParams().start_secs)
        configured = analyzer_cls.call_args.kwargs["params"]
        self.assertEqual(configured.start_secs, VADParams().start_secs)

    def test_realtime_silero_defers_periodic_reset_until_stably_quiet(self) -> None:
        for state, confidence, reset_expected in (
            (VADState.SPEAKING, 0.1, False),
            (VADState.QUIET, 0.8, False),
            (VADState.QUIET, 0.1, True),
        ):
            with self.subTest(state=state, confidence=confidence):
                analyzer = object.__new__(RealtimeSileroVADAnalyzer)
                analyzer._last_reset_time = 0.0
                analyzer._sample_rate = 16000
                analyzer._params = VADParams(confidence=0.7)
                analyzer._vad_state = state
                analyzer._model = MagicMock(return_value=[confidence])

                with patch("realtime.vad.time.time", return_value=10.0):
                    analyzer.voice_confidence(b"\x00" * 1024)

                self.assertEqual(analyzer._model.reset_states.called, reset_expected)
                self.assertEqual(analyzer._last_reset_time, 10.0 if reset_expected else 0.0)

        analyzer = object.__new__(RealtimeSileroVADAnalyzer)
        analyzer._vad_state = VADState.STOPPING
        analyzer._last_reset_time = 0.0
        analyzer._model = MagicMock()
        with (
            patch("pipecat.audio.vad.silero.SileroVADAnalyzer._run_analyzer", return_value=VADState.QUIET),
            patch("realtime.vad.time.time", return_value=11.0),
        ):
            self.assertEqual(analyzer._run_analyzer(b""), VADState.QUIET)
        analyzer._model.reset_states.assert_called_once_with()
        self.assertEqual(analyzer._last_reset_time, 11.0)


class RealtimeVADPipelineCapabilityTests(unittest.TestCase):
    def test_all_cascaded_families_accept_exact_server_vad_controls(self) -> None:
        for pipeline_mode in (
            "frontend-backend-agent",
            "generic-assistant",
            "multilingual-assistant",
        ):
            with self.subTest(pipeline_mode=pipeline_mode):
                controller = _gateway_controller(pipeline_mode, server_vad=True)
                controller.apply_session_update(
                    {
                        "audio": {
                            "input": {
                                "turn_detection": {
                                    "type": "server_vad",
                                    "threshold": 0.6,
                                    "prefix_padding_ms": 225,
                                    "silence_duration_ms": 700,
                                    "create_response": False,
                                    "interrupt_response": False,
                                }
                            }
                        }
                    }
                )
                self.assertFalse(controller.automatic_response_enabled)
                self.assertFalse(controller.interrupt_response_enabled)

    def test_all_omni_families_fail_closed_for_unsupported_false_response_flags(self) -> None:
        for pipeline_mode in ("omni-assistant", "omni-assistant-subagents"):
            for name in ("create_response", "interrupt_response"):
                with self.subTest(pipeline_mode=pipeline_mode, name=name):
                    controller = _gateway_controller(pipeline_mode)
                    with self.assertRaises(RealtimeProtocolError) as raised:
                        controller.apply_session_update(
                            {
                                "audio": {
                                    "input": {
                                        "turn_detection": {
                                            "type": "semantic_vad",
                                            name: False,
                                        }
                                    }
                                }
                            }
                        )
                    self.assertEqual(raised.exception.code, "unsupported_capability")
                    self.assertEqual(
                        raised.exception.param,
                        f"session.audio.input.turn_detection.{name}",
                    )

    def test_all_omni_families_accept_semantic_auto_and_true_response_flags(self) -> None:
        for pipeline_mode in ("omni-assistant", "omni-assistant-subagents"):
            with self.subTest(pipeline_mode=pipeline_mode):
                controller = _gateway_controller(pipeline_mode)
                updated = controller.apply_session_update(
                    {
                        "audio": {
                            "input": {
                                "turn_detection": {
                                    "type": "semantic_vad",
                                    "eagerness": "auto",
                                    "create_response": True,
                                    "interrupt_response": True,
                                }
                            }
                        }
                    }
                )
                self.assertEqual(
                    updated["session"]["audio"]["input"]["turn_detection"],
                    {
                        "type": "semantic_vad",
                        "eagerness": "auto",
                        "create_response": True,
                        "interrupt_response": True,
                    },
                )
