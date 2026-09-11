# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103

from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from typing import Any

from realtime_helpers import FakeWebSocket

from realtime.controller import RealtimeSessionController
from realtime.events import (
    SERVER_CONVERSATION_CREATED,
    SERVER_ERROR,
    SERVER_SESSION_CREATED,
    SERVER_SESSION_UPDATED,
)
from realtime.gateway import _select_realtime_subprotocol, handle_realtime_websocket
from realtime.protocol import RealtimeProtocolError
from realtime.session import (
    AudioFormatCapability,
    CanonicalRealtimeSession,
    RealtimeSessionCapabilities,
)

MODEL = "nvidia/nemotron-realtime"
VOICE = "Magpie-Multilingual.EN-US.Aria"
SECOND_VOICE = "Magpie-Multilingual.EN-US.Jason"
SERVER_VAD_DEFAULTS = {
    "type": "server_vad",
    "threshold": 0.5,
    "prefix_padding_ms": 300,
    "silence_duration_ms": 500,
    "create_response": True,
    "interrupt_response": True,
    "idle_timeout_ms": None,
}


def _capabilities(**overrides: Any) -> RealtimeSessionCapabilities:
    values: dict[str, Any] = {"voices": frozenset({VOICE})}
    values.update(overrides)
    return RealtimeSessionCapabilities(**values)


def _session(
    *,
    capabilities: RealtimeSessionCapabilities | None = None,
    **overrides: Any,
) -> CanonicalRealtimeSession:
    values: dict[str, Any] = {
        "model": MODEL,
        "voice": VOICE,
        "capabilities": capabilities or _capabilities(),
    }
    values.update(overrides)
    return CanonicalRealtimeSession(**values)


def _sanitize_runtime(data: dict[str, Any], **_: Any) -> dict[str, Any]:
    runtime = dict(data)
    runtime.setdefault("pipeline_mode", "generic-assistant")
    runtime.setdefault("model_id", MODEL)
    runtime.setdefault("tts_voice_id", VOICE)
    runtime.setdefault("prompt_key", "generic_assistant")
    return runtime


class CanonicalRealtimeSessionTests(unittest.TestCase):
    def test_defaults_are_canonical_ga_and_public_view_is_detached(self) -> None:
        session = _session(input_transcription_model="nvidia-asr")

        view = session.public_view()
        self.assertEqual(view["object"], "realtime.session")
        self.assertEqual(view["type"], "realtime")
        self.assertEqual(view["model"], MODEL)
        self.assertEqual(view["output_modalities"], ["audio"])
        self.assertEqual(view["audio"]["input"]["format"], {"type": "audio/pcm", "rate": 24000})
        self.assertEqual(view["audio"]["output"]["format"], {"type": "audio/pcm", "rate": 24000})
        self.assertEqual(view["audio"]["input"]["turn_detection"], SERVER_VAD_DEFAULTS)
        self.assertEqual(view["audio"]["input"]["transcription"], {"model": "nvidia-asr"})

        view["instructions"] = "mutated outside the session"
        self.assertEqual(session.public_view()["instructions"], "")

    def test_supported_partial_update_deep_merges(self) -> None:
        session = _session()

        updated = session.apply_update(
            {
                "instructions": "Be brief.",
                "max_output_tokens": 256,
                "output_modalities": ["text"],
                "audio": {"output": {"voice": VOICE}},
            }
        )

        self.assertEqual(updated["instructions"], "Be brief.")
        self.assertEqual(updated["max_output_tokens"], 256)
        self.assertEqual(updated["output_modalities"], ["text"])
        self.assertEqual(updated["audio"]["input"]["turn_detection"], SERVER_VAD_DEFAULTS)
        self.assertEqual(updated["audio"]["output"]["voice"], VOICE)
        self.assertEqual(
            session.response_defaults(),
            {
                "instructions": "Be brief.",
                "output_modalities": ["text"],
                "max_output_tokens": 256,
                "parallel_tool_calls": True,
            },
        )

    def test_pre_ga_flat_and_beta_fields_are_rejected(self) -> None:
        cases = (
            ({"voice": VOICE}, "session.voice"),
            ({"temperature": 0.8}, "session.temperature"),
            ({"input_audio_format": {"type": "audio/pcm", "rate": 24000}}, "session.input_audio_format"),
            ({"modalities": ["audio"]}, "session.modalities"),
            ({"nvidia": {"pipeline_mode": "generic-assistant"}}, "session.nvidia"),
        )

        for patch, param in cases:
            with self.subTest(param=param), self.assertRaises(RealtimeProtocolError) as raised:
                _session().apply_update(patch)
            self.assertEqual(raised.exception.code, "unknown_parameter")
            self.assertEqual(raised.exception.param, param)

    def test_model_is_explicit_and_immutable(self) -> None:
        session = _session()
        self.assertEqual(session.apply_update({"model": MODEL})["model"], MODEL)

        with self.assertRaises(RealtimeProtocolError) as raised:
            session.apply_update({"model": "different-model"})
        self.assertEqual(raised.exception.code, "immutable_field")
        self.assertEqual(raised.exception.param, "session.model")

    def test_max_output_tokens_require_supported_integer_or_inf(self) -> None:
        session = _session()
        self.assertEqual(session.apply_update({"max_output_tokens": "inf"})["max_output_tokens"], "inf")

        for value, code in (
            (True, "invalid_type"),
            (2.9, "invalid_type"),
            (0, "invalid_value"),
            (4097, "invalid_value"),
        ):
            with self.subTest(value=value), self.assertRaises(RealtimeProtocolError) as raised:
                session.apply_update({"max_output_tokens": value})
            self.assertEqual(raised.exception.code, code)

    def test_response_inference_overrides_validate_without_mutating_defaults(self) -> None:
        session = _session(instructions="session prompt", max_output_tokens=256)

        self.assertEqual(session.validate_response_instructions(""), "")
        self.assertEqual(session.validate_response_max_output_tokens("inf"), "inf")
        self.assertEqual(session.validate_response_max_output_tokens(32), 32)
        self.assertEqual(session.public_view()["instructions"], "session prompt")
        self.assertEqual(session.public_view()["max_output_tokens"], 256)

        for value, code in (
            (None, "invalid_type"),
            (True, "invalid_type"),
            (0, "invalid_value"),
            (4097, "invalid_value"),
        ):
            with self.subTest(value=value), self.assertRaises(RealtimeProtocolError) as raised:
                session.validate_response_max_output_tokens(value)
            self.assertEqual(raised.exception.code, code)
            self.assertEqual(raised.exception.param, "response.max_output_tokens")

    def test_output_modalities_are_exactly_audio_or_text(self) -> None:
        session = _session()
        self.assertEqual(session.apply_update({"output_modalities": ["text"]})["output_modalities"], ["text"])
        self.assertEqual(session.apply_update({"output_modalities": ["audio"]})["output_modalities"], ["audio"])

        for value in ([], ["audio", "text"], ["text", "audio"]):
            with self.subTest(value=value), self.assertRaises(RealtimeProtocolError) as raised:
                session.apply_update({"output_modalities": value})
            self.assertEqual(raised.exception.code, "invalid_value")

    def test_session_update_is_atomic_when_a_later_field_fails(self) -> None:
        session = _session(instructions="Original")

        with self.assertRaises(RealtimeProtocolError) as raised:
            session.apply_update(
                {
                    "instructions": "Must not leak",
                    "audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}}},
                }
            )

        self.assertEqual(raised.exception.code, "unsupported_capability")
        self.assertEqual(session.public_view()["instructions"], "Original")
        self.assertEqual(session.public_view()["audio"]["input"]["format"]["rate"], 24000)

    def test_audio_formats_are_driven_by_explicit_capabilities(self) -> None:
        pcm16 = AudioFormatCapability("audio/pcm", 16000)
        with self.assertRaises(RealtimeProtocolError) as raised:
            _session().apply_update({"audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}}}})
        self.assertEqual(raised.exception.code, "unsupported_capability")

        capabilities = _capabilities(
            input_formats=frozenset({AudioFormatCapability("audio/pcm", 24000), pcm16}),
            output_formats=frozenset({AudioFormatCapability("audio/pcm", 24000), pcm16}),
        )
        session = _session(capabilities=capabilities)
        updated = session.apply_update(
            {
                "audio": {
                    "input": {"format": {"type": "audio/pcm", "rate": 16000}},
                    "output": {"format": {"type": "audio/pcm", "rate": 16000}},
                }
            }
        )
        self.assertEqual(updated["audio"]["input"]["format"]["rate"], 16000)
        self.assertEqual(updated["audio"]["output"]["format"]["rate"], 16000)

    def test_omitted_pcm_rates_normalize_to_24khz_for_session_and_response(self) -> None:
        pcm16 = AudioFormatCapability("audio/pcm", 16000)
        session = _session(
            capabilities=_capabilities(
                input_formats=frozenset({AudioFormatCapability("audio/pcm", 24000), pcm16}),
                output_formats=frozenset({AudioFormatCapability("audio/pcm", 24000), pcm16}),
            )
        )

        updated = session.apply_update(
            {
                "audio": {
                    "input": {"format": {"type": "audio/pcm"}},
                    "output": {"format": {"type": "audio/pcm"}},
                }
            }
        )

        self.assertEqual(updated["audio"]["input"]["format"], {"type": "audio/pcm", "rate": 24000})
        self.assertEqual(updated["audio"]["output"]["format"], {"type": "audio/pcm", "rate": 24000})
        response_audio = session.validate_response_audio_output({"output": {"format": {"type": "audio/pcm"}}})
        self.assertEqual(response_audio["format"], {"type": "audio/pcm", "rate": 24000})

    def test_explicit_non_integer_pcm_rates_remain_invalid(self) -> None:
        for side in ("input", "output"):
            for rate in (None, True, 24000.0):
                with self.subTest(side=side, rate=rate), self.assertRaises(RealtimeProtocolError) as raised:
                    _session().apply_update({"audio": {side: {"format": {"type": "audio/pcm", "rate": rate}}}})
                self.assertEqual(raised.exception.code, "invalid_type")
                self.assertEqual(raised.exception.param, f"session.audio.{side}.format.rate")

        for rate in (None, True, 24000.0):
            with self.subTest(scope="response", rate=rate), self.assertRaises(RealtimeProtocolError) as raised:
                _session().validate_response_audio_output({"output": {"format": {"type": "audio/pcm", "rate": rate}}})
            self.assertEqual(raised.exception.code, "invalid_type")
            self.assertEqual(raised.exception.param, "response.audio.output.format.rate")

    def test_turn_detection_rejects_manual_and_unavailable_type(self) -> None:
        session = _session()
        cases = (
            None,
            {"type": "semantic_vad"},
        )
        for value in cases:
            with self.subTest(value=value), self.assertRaises(RealtimeProtocolError):
                session.apply_update({"audio": {"input": {"turn_detection": value}}})

        manual = _session(capabilities=_capabilities(supports_manual_input=True))
        self.assertIsNone(
            manual.apply_update({"audio": {"input": {"turn_detection": None}}})["audio"]["input"]["turn_detection"]
        )

    def test_server_vad_controls_validate_and_echo_exactly(self) -> None:
        session = _session()
        config = {
            "type": "server_vad",
            "threshold": 0.625,
            "prefix_padding_ms": 275,
            "silence_duration_ms": 640,
            "create_response": False,
            "interrupt_response": False,
        }

        updated = session.apply_update({"audio": {"input": {"turn_detection": config}}})

        self.assertEqual(
            updated["audio"]["input"]["turn_detection"],
            {**config, "idle_timeout_ms": None},
        )
        config["threshold"] = 0.1
        self.assertEqual(session.public_view()["audio"]["input"]["turn_detection"]["threshold"], 0.625)

    def test_server_vad_controls_reject_invalid_types_values_and_cross_mode_fields(self) -> None:
        cases = (
            ("threshold", True, "invalid_type"),
            ("threshold", "0.5", "invalid_type"),
            ("threshold", -0.1, "invalid_value"),
            ("threshold", 1.1, "invalid_value"),
            ("threshold", float("nan"), "invalid_value"),
            ("threshold", float("inf"), "invalid_value"),
            ("prefix_padding_ms", True, "invalid_type"),
            ("prefix_padding_ms", 1.5, "invalid_type"),
            ("prefix_padding_ms", -1, "invalid_value"),
            ("silence_duration_ms", True, "invalid_type"),
            ("silence_duration_ms", 1.5, "invalid_type"),
            ("silence_duration_ms", -1, "invalid_value"),
            ("create_response", 1, "invalid_type"),
            ("interrupt_response", 0, "invalid_type"),
        )
        for name, value, code in cases:
            with self.subTest(name=name, value=value), self.assertRaises(RealtimeProtocolError) as raised:
                _session().apply_update({"audio": {"input": {"turn_detection": {"type": "server_vad", name: value}}}})
            self.assertEqual(raised.exception.code, code)
            self.assertEqual(raised.exception.param, f"session.audio.input.turn_detection.{name}")

        with self.assertRaises(RealtimeProtocolError) as cross_mode:
            _session().apply_update(
                {"audio": {"input": {"turn_detection": {"type": "server_vad", "eagerness": "auto"}}}}
            )
        self.assertEqual(cross_mode.exception.code, "unknown_parameter")
        self.assertEqual(cross_mode.exception.param, "session.audio.input.turn_detection.eagerness")

    def test_server_vad_idle_timeout_is_supported_nullable_and_positive(self) -> None:
        param = "session.audio.input.turn_detection.idle_timeout_ms"

        session = _session()
        for value in (None, 1, 5_000, 30_000, 300_000):
            with self.subTest(value=value):
                updated = session.apply_update(
                    {
                        "audio": {
                            "input": {
                                "turn_detection": {
                                    "type": "server_vad",
                                    "idle_timeout_ms": value,
                                }
                            }
                        }
                    }
                )
                self.assertEqual(updated["audio"]["input"]["turn_detection"]["idle_timeout_ms"], value)

        for value, code in (
            (True, "invalid_type"),
            (5_000.0, "invalid_type"),
            (0, "invalid_value"),
            (-1, "invalid_value"),
        ):
            with self.subTest(value=value), self.assertRaises(RealtimeProtocolError) as raised:
                _session().apply_update(
                    {
                        "audio": {
                            "input": {
                                "turn_detection": {
                                    "type": "server_vad",
                                    "idle_timeout_ms": value,
                                }
                            }
                        }
                    }
                )
            self.assertEqual(raised.exception.code, code)
            self.assertEqual(raised.exception.param, param)

        without_idle_timeout = _capabilities(
            turn_detection_options=frozenset(SERVER_VAD_DEFAULTS) - {"idle_timeout_ms"}
        )
        with self.assertRaises(RealtimeProtocolError) as unsupported:
            _session(capabilities=without_idle_timeout).apply_update(
                {
                    "audio": {
                        "input": {
                            "turn_detection": {
                                "type": "server_vad",
                                "idle_timeout_ms": 10_000,
                            }
                        }
                    }
                }
            )
        self.assertEqual(unsupported.exception.code, "unsupported_capability")
        self.assertEqual(unsupported.exception.param, param)

    def test_partial_server_vad_update_preserves_effective_defaults(self) -> None:
        updated = _session().apply_update(
            {
                "audio": {
                    "input": {
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.65,
                        }
                    }
                }
            }
        )

        self.assertEqual(
            updated["audio"]["input"]["turn_detection"],
            {**SERVER_VAD_DEFAULTS, "threshold": 0.65},
        )

    def test_semantic_vad_accepts_auto_and_response_controls_only(self) -> None:
        capabilities = _capabilities(
            turn_detection_types=frozenset({"semantic_vad"}),
            default_turn_detection_type="semantic_vad",
        )
        session = _session(capabilities=capabilities)
        config = {
            "type": "semantic_vad",
            "eagerness": "auto",
            "create_response": False,
            "interrupt_response": False,
        }
        updated = session.apply_update({"audio": {"input": {"turn_detection": config}}})
        self.assertEqual(updated["audio"]["input"]["turn_detection"], config)

        for eagerness, code in (
            ("low", "unsupported_capability"),
            ("medium", "unsupported_capability"),
            (1, "invalid_type"),
        ):
            with self.subTest(eagerness=eagerness), self.assertRaises(RealtimeProtocolError) as raised:
                session.apply_update(
                    {
                        "audio": {
                            "input": {
                                "turn_detection": {
                                    "type": "semantic_vad",
                                    "eagerness": eagerness,
                                }
                            }
                        }
                    }
                )
            self.assertEqual(raised.exception.code, code)
            self.assertEqual(raised.exception.param, "session.audio.input.turn_detection.eagerness")

        with self.assertRaises(RealtimeProtocolError) as cross_mode:
            session.apply_update(
                {
                    "audio": {
                        "input": {
                            "turn_detection": {
                                "type": "semantic_vad",
                                "threshold": 0.5,
                            }
                        }
                    }
                }
            )
        self.assertEqual(cross_mode.exception.code, "unknown_parameter")
        self.assertEqual(cross_mode.exception.param, "session.audio.input.turn_detection.threshold")

    def test_turn_detection_response_controls_are_capability_gated_by_value(self) -> None:
        capabilities = _capabilities(
            turn_detection_create_response_values=frozenset({True}),
            turn_detection_interrupt_response_values=frozenset({True}),
        )
        session = _session(capabilities=capabilities)
        accepted = session.apply_update(
            {
                "audio": {
                    "input": {
                        "turn_detection": {
                            "type": "server_vad",
                            "create_response": True,
                            "interrupt_response": True,
                        }
                    }
                }
            }
        )
        self.assertTrue(accepted["audio"]["input"]["turn_detection"]["create_response"])

        for name in ("create_response", "interrupt_response"):
            with self.subTest(name=name), self.assertRaises(RealtimeProtocolError) as raised:
                session.apply_update({"audio": {"input": {"turn_detection": {"type": "server_vad", name: False}}}})
            self.assertEqual(raised.exception.code, "unsupported_capability")
            self.assertEqual(raised.exception.param, f"session.audio.input.turn_detection.{name}")

    def test_transcription_selectors_require_exact_backend_capabilities(self) -> None:
        session = _session(input_transcription_model="nvidia-asr")
        same = session.apply_update({"audio": {"input": {"transcription": {"model": "nvidia-asr"}}}})
        self.assertEqual(same["audio"]["input"]["transcription"], {"model": "nvidia-asr"})

        with self.assertRaises(RealtimeProtocolError) as changed:
            session.apply_update({"audio": {"input": {"transcription": {"model": "whisper-1"}}}})
        self.assertEqual(changed.exception.code, "unsupported_capability")

        configurable = _session(
            capabilities=_capabilities(
                input_transcription_models=frozenset({"whisper-1"}),
                input_transcription_language_aliases=(("en", "en-US"), ("en-us", "en-US")),
            )
        )
        updated = configurable.apply_update(
            {"audio": {"input": {"transcription": {"model": "whisper-1", "language": "en"}}}}
        )
        self.assertEqual(updated["audio"]["input"]["transcription"]["model"], "whisper-1")
        self.assertEqual(updated["audio"]["input"]["transcription"]["language"], "en-US")

        with self.assertRaises(RealtimeProtocolError) as unsupported_language:
            configurable.apply_update({"audio": {"input": {"transcription": {"model": "whisper-1", "language": "es"}}}})
        self.assertEqual(unsupported_language.exception.code, "unsupported_capability")

    def test_current_transcription_options_are_recognized_and_fail_explicitly(self) -> None:
        session = _session(input_transcription_model="nvidia-asr")
        cases = (
            ("delay", "low"),
            ("keywords", ["NVIDIA", "Nemotron"]),
            ("languages", ["en", "hi"]),
        )

        for name, value in cases:
            param = f"session.audio.input.transcription.{name}"
            with self.subTest(name=name), self.assertRaises(RealtimeProtocolError) as raised:
                session.apply_update(
                    {
                        "audio": {
                            "input": {
                                "transcription": {
                                    "model": "nvidia-asr",
                                    name: value,
                                }
                            }
                        }
                    }
                )
            self.assertEqual(raised.exception.code, "unsupported_capability")
            self.assertEqual(raised.exception.param, param)

        invalid_cases = (
            ("delay", 1, "invalid_type"),
            ("delay", "fastest", "invalid_value"),
            ("keywords", "NVIDIA", "invalid_type"),
            ("keywords", [""], "invalid_value"),
            ("languages", [1], "invalid_value"),
        )
        for name, value, code in invalid_cases:
            with self.subTest(name=name, value=value), self.assertRaises(RealtimeProtocolError) as raised:
                session.apply_update(
                    {
                        "audio": {
                            "input": {
                                "transcription": {
                                    "model": "nvidia-asr",
                                    name: value,
                                }
                            }
                        }
                    }
                )
            self.assertEqual(raised.exception.code, code)

    def test_optional_ga_features_fail_with_explicit_capability_errors(self) -> None:
        cases = (
            {"prompt": {"id": "pmpt_123"}},
            {"reasoning": {"effort": "medium"}},
            {"include": ["item.input_audio_transcription.logprobs"]},
            {"tracing": {"workflow_name": "voice"}},
            {"truncation": "auto"},
        )
        for patch in cases:
            with self.subTest(patch=patch), self.assertRaises(RealtimeProtocolError) as raised:
                _session().apply_update(patch)
            self.assertEqual(raised.exception.code, "unsupported_capability")

    def test_truncation_defaults_to_auto_and_validates_retention_ratio(self) -> None:
        session = _session(capabilities=_capabilities(truncation=True))

        self.assertEqual(session.public_view()["truncation"], "auto")
        updated = session.apply_update(
            {
                "truncation": {
                    "type": "retention_ratio",
                    "retention_ratio": 0.8,
                    "token_limits": {"post_instructions": 2048},
                }
            }
        )
        self.assertEqual(
            updated["truncation"],
            {
                "type": "retention_ratio",
                "retention_ratio": 0.8,
                "token_limits": {"post_instructions": 2048},
            },
        )
        self.assertEqual(session.apply_update({"truncation": "disabled"})["truncation"], "disabled")
        self.assertEqual(
            session.apply_update(
                {"truncation": {"type": "retention_ratio", "retention_ratio": 0.8, "token_limits": {}}}
            )["truncation"]["token_limits"],
            {},
        )

        for value, code, param in (
            ("oldest", "invalid_value", "session.truncation"),
            (
                {"type": "retention_ratio", "retention_ratio": 1.1},
                "invalid_value",
                "session.truncation.retention_ratio",
            ),
            (
                {"type": "retention_ratio", "retention_ratio": True},
                "invalid_type",
                "session.truncation.retention_ratio",
            ),
            (
                {"type": "retention_ratio", "retention_ratio": 0.8, "token_limits": {"post_instructions": 0}},
                "invalid_value",
                "session.truncation.token_limits.post_instructions",
            ),
            (
                {"type": "retention_ratio", "retention_ratio": 0.8, "token_limits": {"post_instructions": None}},
                "invalid_type",
                "session.truncation.token_limits.post_instructions",
            ),
        ):
            with self.subTest(value=value), self.assertRaises(RealtimeProtocolError) as raised:
                session.apply_update({"truncation": value})
            self.assertEqual(raised.exception.code, code)
            self.assertEqual(raised.exception.param, param)

    def test_nullable_optional_ga_fields_accept_exact_no_ops(self) -> None:
        session = _session()

        updated = session.apply_update({"prompt": None, "tracing": None})

        self.assertIsNone(updated["prompt"])
        self.assertNotIn("tracing", updated)
        with self.assertRaises(RealtimeProtocolError) as prompt_error:
            session.apply_update({"prompt": {"id": "pmpt_123"}})
        self.assertEqual(prompt_error.exception.code, "unsupported_capability")
        with self.assertRaises(RealtimeProtocolError) as tracing_error:
            session.apply_update({"tracing": "auto"})
        self.assertEqual(tracing_error.exception.code, "unsupported_capability")

    def test_voice_locks_after_output_audio_starts(self) -> None:
        capabilities = _capabilities(voices=frozenset({VOICE, SECOND_VOICE}))
        session = _session(capabilities=capabilities)
        self.assertEqual(
            session.apply_update({"audio": {"output": {"voice": SECOND_VOICE}}})["audio"]["output"]["voice"],
            SECOND_VOICE,
        )

        session.begin_response("resp_1")
        session.mark_output_audio_started("resp_1")
        session.finish_response("resp_1")
        with self.assertRaises(RealtimeProtocolError) as raised:
            session.apply_update({"audio": {"output": {"voice": VOICE}}})
        self.assertEqual(raised.exception.code, "immutable_field")

    def test_output_speed_requires_backend_support_and_response_boundary(self) -> None:
        with self.assertRaises(RealtimeProtocolError) as unsupported:
            _session().apply_update({"audio": {"output": {"speed": 1.25}}})
        self.assertEqual(unsupported.exception.code, "unsupported_capability")

        session = _session(capabilities=_capabilities(supports_output_speed=True))
        self.assertEqual(
            session.apply_update({"audio": {"output": {"speed": 1.25}}})["audio"]["output"]["speed"],
            1.25,
        )
        session.begin_response("resp_1")
        with self.assertRaises(RealtimeProtocolError) as active:
            session.apply_update({"audio": {"output": {"speed": 1.0}}})
        self.assertEqual(active.exception.code, "immutable_field")


class RealtimeSessionControllerTests(unittest.TestCase):
    def test_created_events_and_public_nvidia_view_are_canonical(self) -> None:
        runtime = {
            "pipeline_mode": "generic-assistant",
            "model_id": MODEL,
            "tts_voice_id": VOICE,
            "llm_id": "cloud-nim:nemotron-lightning",
            "base_url": "https://internal.example/v1",
            "asr_server": "asr.internal:443",
            "tts_server": "tts.internal:443",
            "asr_function_id": "asr-private-function",
            "tts_function_id": "tts-private-function",
        }
        controller = RealtimeSessionController(
            model=MODEL,
            voice=VOICE,
            runtime_config=runtime,
            server_tools=["get_weather"],
            delegate_tools=["call_backend"],
            capabilities=_capabilities(),
        )

        created = controller.created_events()
        self.assertEqual([event["type"] for event in created], [SERVER_SESSION_CREATED, SERVER_CONVERSATION_CREATED])
        public = created[0]["session"]
        self.assertEqual(public["nvidia"]["pipeline_mode"], "generic-assistant")
        self.assertEqual(public["nvidia"]["server_tools"], ["get_weather"])
        self.assertEqual(public["nvidia"]["delegate_tools"], ["call_backend"])
        for key in (
            "base_url",
            "asr_server",
            "tts_server",
            "asr_function_id",
            "tts_function_id",
        ):
            self.assertNotIn(key, public["nvidia"])

    def test_controller_applies_standard_updates_without_mutating_runtime_routing(self) -> None:
        runtime = {"pipeline_mode": "generic-assistant", "model_id": MODEL, "tts_voice_id": VOICE}
        controller = RealtimeSessionController(
            model=MODEL,
            voice=VOICE,
            runtime_config=runtime,
            capabilities=_capabilities(),
        )

        event = controller.apply_session_update({"instructions": "New instructions", "output_modalities": ["text"]})

        self.assertEqual(event["type"], SERVER_SESSION_UPDATED)
        self.assertEqual(event["session"]["instructions"], "New instructions")
        self.assertEqual(event["session"]["output_modalities"], ["text"])
        self.assertEqual(controller.runtime_config, runtime)

    def test_turn_detection_accessors_expose_defaults_and_detached_initial_config(self) -> None:
        controller = RealtimeSessionController(
            model=MODEL,
            voice=VOICE,
            runtime_config={"pipeline_mode": "generic-assistant"},
            capabilities=_capabilities(),
        )
        self.assertEqual(controller.turn_detection_config, SERVER_VAD_DEFAULTS)
        self.assertTrue(controller.automatic_response_enabled)
        self.assertTrue(controller.interrupt_response_enabled)
        self.assertEqual(controller.server_vad_prefix_padding_ms, 300)

        controller.apply_session_update(
            {
                "audio": {
                    "input": {
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.7,
                            "prefix_padding_ms": 125,
                            "silence_duration_ms": 450,
                            "create_response": False,
                            "interrupt_response": False,
                        }
                    }
                }
            }
        )
        config = controller.turn_detection_config
        assert config is not None
        config["prefix_padding_ms"] = 999
        self.assertEqual(controller.server_vad_prefix_padding_ms, 125)
        self.assertFalse(controller.automatic_response_enabled)
        self.assertFalse(controller.interrupt_response_enabled)


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    def test_only_canonical_realtime_subprotocol_is_negotiated(self) -> None:
        beta_only = SimpleNamespace(
            headers={"sec-websocket-protocol": "openai-insecure-api-key.sk-secret,openai-beta.realtime-v1"}
        )
        canonical = SimpleNamespace(headers={"sec-websocket-protocol": "openai-insecure-api-key.sk-secret,realtime"})

        self.assertIsNone(_select_realtime_subprotocol(beta_only))  # type: ignore[arg-type]
        self.assertEqual(_select_realtime_subprotocol(canonical), "realtime")  # type: ignore[arg-type]

    async def test_session_lifetime_covers_pipeline_handoff_and_closes_cleanly(self) -> None:
        pipeline_started = False

        async def start_bot(ws: Any, config: dict[str, Any], controller: RealtimeSessionController) -> None:  # noqa: ARG001
            nonlocal pipeline_started
            pipeline_started = True
            await asyncio.Event().wait()

        ws = FakeWebSocket([json.dumps({"type": "response.create"})])

        await handle_realtime_websocket(
            ws,
            sanitize_session_config=_sanitize_runtime,
            start_bot=start_bot,
            session_max_duration_secs=0.01,
        )

        self.assertTrue(pipeline_started)
        self.assertTrue(ws.closed)
        self.assertEqual(ws.close_code, 1000)
        self.assertEqual(ws.close_reason, "realtime session expired")

    async def test_invalid_json_returns_structured_error_after_created_events(self) -> None:
        ws = FakeWebSocket(["{not-json"])

        await handle_realtime_websocket(ws, sanitize_session_config=_sanitize_runtime)  # type: ignore[arg-type]

        self.assertEqual(ws.sent[0]["type"], SERVER_SESSION_CREATED)
        self.assertEqual(ws.sent[1]["type"], SERVER_CONVERSATION_CREATED)
        self.assertEqual(ws.sent[2]["type"], SERVER_ERROR)
        self.assertEqual(ws.sent[2]["error"]["code"], "invalid_json")

    async def test_initial_event_envelope_rejects_unknown_root_fields_and_allows_retry(self) -> None:
        ws = FakeWebSocket(
            [
                json.dumps(
                    {
                        "type": "session.update",
                        "event_id": "bad_root",
                        "session": {"type": "realtime"},
                        "unexpected": True,
                    }
                ),
                json.dumps({"type": "session.update", "session": {"type": "realtime"}}),
            ]
        )

        await handle_realtime_websocket(ws, sanitize_session_config=_sanitize_runtime)

        self.assertEqual(ws.sent[2]["type"], SERVER_ERROR)
        self.assertEqual(ws.sent[2]["error"]["code"], "unknown_parameter")
        self.assertEqual(ws.sent[2]["error"]["param"], "unexpected")
        self.assertEqual(ws.sent[2]["error"]["event_id"], "bad_root")
        self.assertEqual(ws.sent[3]["type"], SERVER_SESSION_UPDATED)

    async def test_initial_event_id_length_is_bounded_and_retryable(self) -> None:
        ws = FakeWebSocket(
            [
                json.dumps(
                    {
                        "type": "session.update",
                        "event_id": "x" * 513,
                        "session": {"type": "realtime"},
                    }
                ),
                json.dumps({"type": "session.update", "session": {"type": "realtime"}}),
            ]
        )

        await handle_realtime_websocket(ws, sanitize_session_config=_sanitize_runtime)

        self.assertEqual(ws.sent[2]["type"], SERVER_ERROR)
        self.assertEqual(ws.sent[2]["error"]["code"], "invalid_value")
        self.assertEqual(ws.sent[2]["error"]["param"], "event_id")
        self.assertNotIn("event_id", ws.sent[2]["error"])
        self.assertEqual(ws.sent[3]["type"], SERVER_SESSION_UPDATED)

    async def test_pre_ga_session_field_is_rejected_without_handoff(self) -> None:
        started = False

        async def start_bot(ws: Any, config: dict[str, Any], controller: RealtimeSessionController) -> None:  # noqa: ARG001
            nonlocal started
            started = True

        ws = FakeWebSocket(
            [json.dumps({"type": "session.update", "event_id": "pre_ga_1", "session": {"voice": VOICE}})]
        )

        await handle_realtime_websocket(
            ws,  # type: ignore[arg-type]
            sanitize_session_config=_sanitize_runtime,
            start_bot=start_bot,
        )

        self.assertFalse(started)
        error = ws.sent[2]
        self.assertEqual(error["type"], SERVER_ERROR)
        self.assertEqual(error["error"]["code"], "unknown_parameter")
        self.assertEqual(error["error"]["param"], "session.voice")
        self.assertEqual(error["error"]["event_id"], "pre_ga_1")

    async def test_nvidia_routing_is_immutable_and_retryable(self) -> None:
        ws = FakeWebSocket(
            [
                json.dumps(
                    {
                        "type": "session.update",
                        "event_id": "bad_route",
                        "session": {"nvidia": {"model_id": "attacker-controlled-model"}},
                    }
                ),
                json.dumps(
                    {
                        "type": "session.update",
                        "event_id": "good_route",
                        "session": {"nvidia": {"model_id": MODEL}},
                    }
                ),
            ]
        )

        await handle_realtime_websocket(ws, sanitize_session_config=_sanitize_runtime)  # type: ignore[arg-type]

        self.assertEqual(ws.sent[2]["type"], SERVER_ERROR)
        self.assertEqual(ws.sent[2]["error"]["code"], "immutable_field")
        self.assertEqual(ws.sent[2]["error"]["event_id"], "bad_route")
        self.assertEqual(ws.sent[3]["type"], SERVER_SESSION_UPDATED)


if __name__ == "__main__":
    unittest.main()
