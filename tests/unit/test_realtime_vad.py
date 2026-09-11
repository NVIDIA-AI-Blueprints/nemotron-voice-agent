# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy, VADUserTurnStartStrategy

from examples.shared.pipeline_utils import (
    build_vad_params,
    build_vad_user_turn_start_strategies,
    realtime_vad_prefix_padding_secs,
)
from realtime.gateway import _controller_from_runtime
from realtime.protocol import RealtimeProtocolError


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
