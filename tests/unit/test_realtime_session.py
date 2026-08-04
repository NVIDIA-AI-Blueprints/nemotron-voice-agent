# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from realtime_helpers import FakeWebSocket

from realtime.events import SERVER_ERROR, SERVER_SESSION_CREATED, SERVER_SESSION_UPDATED
from realtime.gateway import _select_realtime_subprotocol, handle_realtime_websocket
from realtime.session import (
    DEFAULT_PIPELINE_MODE,
    DEFAULT_PROMPT_KEY,
    RealtimeSession,
    map_session_update_to_flat_config,
    unsupported_live_session_fields,
)


class MapSessionUpdateTests(unittest.TestCase):
    def test_defaults_pipeline_and_prompt_without_tools(self) -> None:
        flat = map_session_update_to_flat_config({})
        self.assertEqual(flat["pipeline_mode"], DEFAULT_PIPELINE_MODE)
        self.assertEqual(flat["prompt_key"], DEFAULT_PROMPT_KEY)

    def test_instructions_map_to_prompt_content(self) -> None:
        flat = map_session_update_to_flat_config({"instructions": "Be brief."})
        self.assertEqual(flat["prompt_content"], "Be brief.")
        self.assertNotIn("system_prompt", flat)

    def test_voice_maps_to_tts_voice_id(self) -> None:
        flat = map_session_update_to_flat_config({"voice": "Magpie-Multilingual.EN-US.Aria"})
        self.assertEqual(flat["tts_voice_id"], "Magpie-Multilingual.EN-US.Aria")
        nested = map_session_update_to_flat_config({"audio": {"output": {"voice": "Magpie-Multilingual.EN-US.Aria"}}})
        self.assertEqual(nested["tts_voice_id"], "Magpie-Multilingual.EN-US.Aria")

    def test_model_is_ignored(self) -> None:
        flat = map_session_update_to_flat_config({"model": "gpt-realtime"})
        self.assertNotIn("model", flat)
        self.assertNotIn("model_id", flat)

    def test_nvidia_fields_pass_through_without_voice_duplicate(self) -> None:
        flat = map_session_update_to_flat_config(
            {
                "nvidia": {
                    "pipeline_mode": "generic-assistant",
                    "llm_id": "cloud-nim:nemotron-nano",
                    "asr_id": "cloud-nim:nemotron-asr",
                    "tts_id": "cloud-nim:magpie-multilingual-tts",
                    "prompt_key": "generic_assistant",
                    "tts_voice_id": "should-be-ignored",
                }
            }
        )
        self.assertEqual(flat["llm_id"], "cloud-nim:nemotron-nano")
        self.assertEqual(flat["prompt_key"], "generic_assistant")
        self.assertNotIn("tts_voice_id", flat)

    def test_tools_are_ignored(self) -> None:
        omitted = map_session_update_to_flat_config({})
        empty = map_session_update_to_flat_config({"tools": []})
        populated = map_session_update_to_flat_config(
            {
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Weather",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ]
            }
        )
        self.assertNotIn("client_tools", omitted)
        self.assertNotIn("client_tools", empty)
        self.assertNotIn("client_tools", populated)
        self.assertEqual(empty["prompt_key"], DEFAULT_PROMPT_KEY)
        self.assertEqual(populated["prompt_key"], DEFAULT_PROMPT_KEY)

    def test_rejects_whisper_transcription(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            map_session_update_to_flat_config({"audio": {"input": {"transcription": {"model": "whisper-1"}}}})
        message = str(ctx.exception)
        self.assertIn("separate OpenAI transcription model", message)
        self.assertIn("input_audio_transcription.*", message)

    def test_null_input_audio_transcription_ok(self) -> None:
        flat = map_session_update_to_flat_config({"input_audio_transcription": None})
        self.assertEqual(flat["pipeline_mode"], DEFAULT_PIPELINE_MODE)

    def test_temperature_maps_to_llm_temperature(self) -> None:
        flat = map_session_update_to_flat_config({"temperature": 0.8})
        self.assertEqual(flat["temperature"], 0.8)

    def test_rejects_invalid_temperature(self) -> None:
        with self.assertRaises(ValueError):
            map_session_update_to_flat_config({"temperature": "hot"})
        with self.assertRaises(ValueError):
            map_session_update_to_flat_config({"temperature": -1})

    def test_rejects_text_only_modalities(self) -> None:
        with self.assertRaises(ValueError):
            map_session_update_to_flat_config({"output_modalities": ["text"]})
        with self.assertRaises(ValueError):
            map_session_update_to_flat_config({"modalities": ["text"]})

    def test_audio_and_text_modalities_ok(self) -> None:
        flat = map_session_update_to_flat_config(
            {"output_modalities": ["audio", "text"], "modalities": ["text", "audio"]}
        )
        self.assertEqual(flat["pipeline_mode"], DEFAULT_PIPELINE_MODE)

    def test_tool_choice_string_and_realtime_object(self) -> None:
        self.assertEqual(map_session_update_to_flat_config({"tool_choice": "auto"})["tool_choice"], "auto")
        self.assertEqual(
            map_session_update_to_flat_config({"tool_choice": {"type": "function", "name": "get_weather"}})[
                "tool_choice"
            ],
            {"type": "function", "function": {"name": "get_weather"}},
        )
        self.assertEqual(
            map_session_update_to_flat_config(
                {"tool_choice": {"type": "function", "function": {"name": "get_weather"}}}
            )["tool_choice"],
            {"type": "function", "function": {"name": "get_weather"}},
        )

    def test_rejects_invalid_tool_choice(self) -> None:
        with self.assertRaises(ValueError):
            map_session_update_to_flat_config({"tool_choice": "sometimes"})
        with self.assertRaises(ValueError):
            map_session_update_to_flat_config({"tool_choice": {"type": "function"}})


class RealtimeSessionApplyTests(unittest.TestCase):
    def test_apply_update_reflects_sanitized_nvidia(self) -> None:
        session = RealtimeSession()
        public = session.apply_update(
            {
                "instructions": "Hello",
                "nvidia": {"pipeline_mode": "generic-assistant", "llm_id": "cloud-nim:x"},
            },
            sanitized_flat={
                "pipeline_mode": "generic-assistant",
                "llm_id": "cloud-nim:x",
                "model_id": "hydrated-model",
                "prompt_key": DEFAULT_PROMPT_KEY,
                "prompt_content": "Hello",
            },
        )
        self.assertEqual(public["instructions"], "Hello")
        self.assertEqual(public["nvidia"]["pipeline_mode"], "generic-assistant")
        self.assertEqual(public["nvidia"]["model_id"], "hydrated-model")
        self.assertEqual(public["nvidia"]["prompt_key"], DEFAULT_PROMPT_KEY)

    def test_apply_update_strips_example_specific_nvidia_keys(self) -> None:
        session = RealtimeSession()
        public = session.apply_update(
            {
                "nvidia": {
                    "pipeline_mode": "frontend-backend-agent",
                    "thinker_llm_id": "cloud-nim:x",
                }
            },
            sanitized_flat={"pipeline_mode": "frontend-backend-agent"},
        )
        self.assertEqual(public["nvidia"]["pipeline_mode"], "frontend-backend-agent")
        self.assertNotIn("thinker_llm_id", public["nvidia"])

    def test_public_nvidia_omits_service_endpoints(self) -> None:
        session = RealtimeSession()
        public = session.apply_update(
            {"nvidia": {"pipeline_mode": "generic-assistant"}},
            sanitized_flat={
                "pipeline_mode": "generic-assistant",
                "prompt_key": DEFAULT_PROMPT_KEY,
                "base_url": "https://internal.example/v1",
                "asr_server": "asr.internal:443",
                "tts_server": "tts.internal:443",
                "asr_function_id": "asr-fn",
                "tts_function_id": "tts-fn",
                "llm_id": "cloud-nim:nemotron-nano",
            },
        )
        nvidia = public["nvidia"]
        self.assertEqual(nvidia["llm_id"], "cloud-nim:nemotron-nano")
        for key in ("base_url", "asr_server", "tts_server", "asr_function_id", "tts_function_id"):
            self.assertNotIn(key, nvidia)

    def test_rejects_bool_and_truncated_max_output_tokens(self) -> None:
        with self.assertRaises(ValueError):
            map_session_update_to_flat_config({"max_output_tokens": True})
        with self.assertRaises(ValueError):
            map_session_update_to_flat_config({"max_output_tokens": 2.9})


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    def test_subprotocol_does_not_echo_api_key_token(self) -> None:
        from types import SimpleNamespace

        ws = SimpleNamespace(
            headers={"sec-websocket-protocol": "openai-insecure-api-key.sk-secret,openai-beta.realtime-v1"}
        )
        self.assertIsNone(_select_realtime_subprotocol(ws))  # type: ignore[arg-type]
        ws2 = SimpleNamespace(headers={"sec-websocket-protocol": "openai-insecure-api-key.sk-secret,realtime"})
        self.assertEqual(_select_realtime_subprotocol(ws2), "realtime")  # type: ignore[arg-type]

    async def test_session_created_then_updated(self) -> None:
        ws = FakeWebSocket(
            [
                json.dumps(
                    {
                        "type": "session.update",
                        "event_id": "client_1",
                        "session": {
                            "instructions": "Speak briefly.",
                            "nvidia": {
                                "pipeline_mode": "generic-assistant",
                                "llm_id": "cloud-nim:nemotron-nano",
                            },
                        },
                    }
                )
            ]
        )

        def sanitize(data: dict, fallback_example_key: str = "") -> dict:
            out = dict(data)
            out.setdefault("pipeline_mode", "generic-assistant")
            out["model_id"] = "from-sanitize"
            out["prompt_key"] = out.get("prompt_key") or DEFAULT_PROMPT_KEY
            return out

        with patch("realtime.gateway.resolve_realtime_tts_voice", return_value=None):
            await handle_realtime_websocket(ws, sanitize_session_config=sanitize)

        self.assertTrue(ws.accepted)
        self.assertGreaterEqual(len(ws.sent), 2)
        self.assertEqual(ws.sent[0]["type"], SERVER_SESSION_CREATED)
        self.assertIn("session", ws.sent[0])
        self.assertEqual(ws.sent[0]["session"]["nvidia"]["pipeline_mode"], "generic-assistant")

        self.assertEqual(ws.sent[1]["type"], SERVER_SESSION_UPDATED)
        self.assertEqual(ws.sent[1]["session"]["instructions"], "Speak briefly.")
        self.assertEqual(ws.sent[1]["session"]["nvidia"]["llm_id"], "cloud-nim:nemotron-nano")
        self.assertEqual(ws.sent[1]["session"]["nvidia"]["model_id"], "from-sanitize")

    async def test_unsupported_event_returns_error(self) -> None:
        ws = FakeWebSocket(
            [
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "event_id": "client_audio",
                        "audio": "AAAA",
                    }
                )
            ]
        )

        await handle_realtime_websocket(ws, sanitize_session_config=lambda data, **_: dict(data))

        self.assertEqual(ws.sent[0]["type"], SERVER_SESSION_CREATED)
        self.assertEqual(ws.sent[1]["type"], SERVER_ERROR)
        self.assertEqual(ws.sent[1]["error"]["code"], "unsupported_event")
        self.assertEqual(ws.sent[1]["error"]["event_id"], "client_audio")

    async def test_invalid_json_returns_error(self) -> None:
        ws = FakeWebSocket(["{not-json"])
        await handle_realtime_websocket(ws, sanitize_session_config=lambda data, **_: dict(data))
        self.assertEqual(ws.sent[1]["type"], SERVER_ERROR)
        self.assertEqual(ws.sent[1]["error"]["code"], "invalid_json")


class LiveSessionUpdateFieldTests(unittest.TestCase):
    def test_voice_and_turn_detection_are_live(self) -> None:
        self.assertEqual(
            unsupported_live_session_fields(
                {
                    "voice": "Magpie-Multilingual.EN-US.Aria",
                    "turn_detection": None,
                    "audio": {
                        "input": {"format": {"type": "audio/pcm", "rate": 24000}, "turn_detection": None},
                        "output": {"voice": "Magpie-Multilingual.EN-US.Aria"},
                    },
                },
                current={},
            ),
            [],
        )

    def test_echoed_agent_fields_allowed_when_unchanged(self) -> None:
        current = {
            "instructions": "Be brief.",
            "tools": [{"type": "function", "name": "get_weather"}],
            "temperature": 0.8,
            "nvidia": {"pipeline_mode": "generic-assistant"},
        }
        self.assertEqual(
            unsupported_live_session_fields(
                {
                    "instructions": "Be brief.",
                    "tools": [{"type": "function", "name": "get_weather"}],
                    "temperature": 0.8,
                    "turn_detection": {"type": "server_vad"},
                    "nvidia": {"pipeline_mode": "generic-assistant"},
                },
                current=current,
            ),
            [],
        )

    def test_new_non_live_field_rejected_when_absent(self) -> None:
        bad = unsupported_live_session_fields(
            {"temperature": 0.8},
            current={"instructions": "Be brief."},
        )
        self.assertIn("temperature", bad)

    def test_changed_instructions_tools_nvidia_rejected(self) -> None:
        current = {
            "instructions": "old",
            "tools": [],
            "nvidia": {"pipeline_mode": "generic-assistant"},
        }
        bad = unsupported_live_session_fields(
            {
                "instructions": "new",
                "tools": [{"type": "function", "name": "get_weather"}],
                "nvidia": {"pipeline_mode": "omni-assistant"},
            },
            current=current,
        )
        self.assertEqual(set(bad), {"instructions", "tools", "nvidia.pipeline_mode"})

    def test_non_null_transcription_not_live(self) -> None:
        bad = unsupported_live_session_fields(
            {"audio": {"input": {"transcription": {"model": "whisper-1"}}}},
            current={},
        )
        self.assertIn("audio.input.transcription", bad)

    def test_null_transcription_allowed(self) -> None:
        self.assertEqual(
            unsupported_live_session_fields(
                {"audio": {"input": {"transcription": None}}},
                current={},
            ),
            [],
        )


class SanitizeIntegrationTests(unittest.TestCase):
    """Exercise mapping through the real catalog sanitize path (no server import)."""

    def test_sanitize_with_generic_catalog(self) -> None:
        from pathlib import Path

        import examples_registry
        from utils import clear_service_context, filter_session_config, set_service_context

        flat = map_session_update_to_flat_config(
            {
                "instructions": "Be helpful.",
                "nvidia": {"pipeline_mode": "generic-assistant"},
            }
        )
        example = examples_registry.find("generic-assistant")
        if not flat.get("prompt_key") and not flat.get("prompt_content"):
            prompt_key = examples_registry.prompt_default_key(example["key"])
            if prompt_key:
                flat["prompt_key"] = prompt_key
        set_service_context(Path("src/examples/generic"), example.get("slots") or None)
        try:
            sanitized = filter_session_config(flat)
            self.assertEqual(sanitized.get("pipeline_mode"), "generic-assistant")
            self.assertEqual(sanitized.get("prompt_content"), "Be helpful.")
            self.assertNotEqual(sanitized.get("system_prompt"), "Be helpful.")
        finally:
            clear_service_context()


if __name__ == "__main__":
    unittest.main()
