# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103, D107

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from realtime_helpers import FakeWebSocket

from realtime.gateway import handle_realtime_websocket
from realtime.session import map_session_update_to_flat_config


class ClientToolsTests(unittest.TestCase):
    def test_session_tools_map_to_client_tools(self) -> None:
        flat = map_session_update_to_flat_config(
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
        self.assertEqual(flat["client_tools"][0]["name"], "get_weather")

    def test_empty_tools_clear_client_tools(self) -> None:
        flat = map_session_update_to_flat_config({"tools": []})
        self.assertEqual(flat["client_tools"], [])

    def test_invalid_and_duplicate_tools_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "function tool"):
            map_session_update_to_flat_config({"tools": [{"type": "mcp", "name": "bad"}]})
        tool = {
            "type": "function",
            "name": "same",
            "parameters": {"type": "object", "properties": {}},
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            map_session_update_to_flat_config({"tools": [tool, tool]})


class GatewayClientToolsTests(unittest.IsolatedAsyncioTestCase):
    async def test_client_tools_are_passed_to_pipeline(self) -> None:
        captured: dict = {}

        async def start_bot(ws, config, session):  # noqa: ARG001
            captured["config"] = config
            captured["session"] = session

        ws = FakeWebSocket(
            [
                json.dumps(
                    {
                        "type": "session.update",
                        "event_id": "e1",
                        "session": {
                            "tools": [
                                {
                                    "type": "function",
                                    "name": "totally_fake_tool",
                                    "description": "x",
                                    "parameters": {"type": "object", "properties": {}},
                                }
                            ],
                            "nvidia": {"pipeline_mode": "generic-assistant"},
                        },
                    }
                )
            ]
        )

        with patch("realtime.gateway.resolve_realtime_tts_voice", return_value=None):
            await handle_realtime_websocket(
                ws,
                sanitize_session_config=lambda data, **_: dict(data),
                start_bot=start_bot,
                resolve_server_tools=lambda _config: ["get_weather", "set_memory"],
            )

        self.assertIn("config", captured)
        self.assertEqual(captured["config"]["client_tools"][0]["name"], "totally_fake_tool")
        self.assertEqual(captured["session"]["tools"][0]["name"], "totally_fake_tool")
        self.assertEqual(
            captured["session"]["nvidia"]["server_tools"],
            ["get_weather", "set_memory"],
        )

    async def test_client_server_name_conflict_is_rejected(self) -> None:
        started = False

        async def start_bot(ws, config, session):  # noqa: ARG001
            nonlocal started
            started = True

        ws = FakeWebSocket(
            [
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "tools": [
                                {
                                    "type": "function",
                                    "name": "get_weather",
                                    "parameters": {"type": "object", "properties": {}},
                                }
                            ]
                        },
                    }
                )
            ]
        )
        with patch("realtime.gateway.resolve_realtime_tts_voice", return_value=None):
            await handle_realtime_websocket(
                ws,
                sanitize_session_config=lambda data, **_: dict(data),
                start_bot=start_bot,
                resolve_server_tools=lambda _config: ["get_weather"],
            )
        self.assertFalse(started)
        self.assertEqual(ws.sent[-1]["type"], "error")
        self.assertEqual(ws.sent[-1]["error"]["code"], "invalid_session")
