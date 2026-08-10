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


class ToolsIgnoredTests(unittest.TestCase):
    def test_session_tools_do_not_map_to_client_tools(self) -> None:
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
        self.assertNotIn("client_tools", flat)

    def test_empty_tools_also_ignored(self) -> None:
        flat = map_session_update_to_flat_config({"tools": []})
        self.assertNotIn("client_tools", flat)


class GatewayToolsIgnoredTests(unittest.IsolatedAsyncioTestCase):
    async def test_client_tools_not_passed_to_pipeline(self) -> None:
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
            )

        self.assertIn("config", captured)
        self.assertNotIn("client_tools", captured["config"])
        self.assertEqual(captured["session"].get("tools"), [])
