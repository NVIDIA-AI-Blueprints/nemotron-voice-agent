# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Server-owned protocol selection and ordinary RTVI isolation."""

# ruff: noqa: D102

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import server
from examples.shared.pipeline_utils import runner_protocol
from realtime.controller import RealtimeSessionController


class RunnerProtocolOwnershipTests(unittest.TestCase):
    """Only a server-created Realtime controller may select that protocol."""

    def test_only_a_server_controller_selects_realtime(self) -> None:
        controller = RealtimeSessionController(
            model="test-model",
            voice="test-voice",
            runtime_config={},
        )
        cases = (
            ({"protocol": "realtime"}, "rtvi"),
            ({"protocol": "realtime", "realtime_controller": {"spoofed": True}}, "rtvi"),
            ({"protocol": "rtvi", "realtime_controller": controller}, "realtime"),
        )
        for body, expected in cases:
            with self.subTest(body=body):
                self.assertEqual(runner_protocol(SimpleNamespace(body=body)), expected)

    def test_rtvi_runner_body_removes_controller_and_stamps_protocol(self) -> None:
        body = server._build_rtvi_runner_body(
            {"pipeline_mode": "generic-assistant", "model_id": "trusted-model"},
            session_id="session-123",
            request_data={
                "protocol": "realtime",
                "realtime_controller": {"spoofed": True},
                "model_id": "untrusted-model",
                "application_data": "preserved",
            },
        )

        self.assertEqual(body["protocol"], "rtvi")
        self.assertNotIn("realtime_controller", body)
        self.assertEqual(body["session_id"], "session-123")
        self.assertEqual(body["model_id"], "trusted-model")
        self.assertEqual(body["application_data"], "preserved")
        self.assertEqual(runner_protocol(SimpleNamespace(body=body)), "rtvi")


class ServiceReadinessOwnershipTests(unittest.IsolatedAsyncioTestCase):
    """Realtime-only media settings cannot alter ordinary readiness checks."""

    async def _readiness_calls(self, *, is_realtime: bool) -> tuple[AsyncMock, AsyncMock, AsyncMock]:
        llm_ready = AsyncMock()
        asr_ready = AsyncMock()
        tts_ready = AsyncMock()
        with (
            patch.object(server, "_ensure_llm_ready_for_connection", llm_ready),
            patch.object(server, "_ensure_asr_ready_for_connection", asr_ready),
            patch.object(server, "_ensure_tts_ready_for_connection", tts_ready),
        ):
            await server._ensure_services_ready_for_connection(
                {"output_modalities": ["text"]},
                {},
                is_realtime=is_realtime,
            )
        return llm_ready, asr_ready, tts_ready

    async def test_text_output_skips_tts_readiness_only_for_realtime(self) -> None:
        for is_realtime in (False, True):
            with self.subTest(is_realtime=is_realtime):
                llm_ready, asr_ready, tts_ready = await self._readiness_calls(is_realtime=is_realtime)
                llm_ready.assert_awaited_once()
                asr_ready.assert_awaited_once()
                if is_realtime:
                    tts_ready.assert_not_awaited()
                else:
                    tts_ready.assert_awaited_once()


class RealtimeToolCapabilityTests(unittest.IsolatedAsyncioTestCase):
    """Advertise tools only for pipelines that can execute them."""

    async def _capabilities(self, pipeline_mode: str):
        config = {
            "pipeline_mode": pipeline_mode,
            "tts_server": "test-tts:50051",
            "tts_voice_id": "TestVoice",
            "tts_function_id": "",
            "tts_model": "test-tts-model",
        }
        with (
            patch.object(
                server,
                "_bind_example_context_by_key",
                return_value={"key": pipeline_mode, "slots": ["llm", "tts"]},
            ),
            patch.object(server, "_trusted_realtime_service_entry", side_effect=[{}, {}]),
            patch.object(server, "_validate_realtime_service_route"),
            patch.object(
                server,
                "peek_cached_tts_config",
                return_value={"voices": [{"id": "TestVoice"}]},
            ),
        ):
            return await server._resolve_realtime_session_capabilities(config)

    async def test_tool_capabilities_follow_pipeline_support(self) -> None:
        for pipeline_mode, expected in (
            ("omni-assistant-subagents", False),
            ("generic-assistant", True),
        ):
            with self.subTest(pipeline_mode=pipeline_mode):
                capabilities = await self._capabilities(pipeline_mode)
                self.assertEqual(capabilities.function_tools, expected)
                self.assertEqual(capabilities.mcp_tools, expected)
                self.assertEqual(capabilities.sequential_tool_calls, expected)


if __name__ == "__main__":
    unittest.main()
