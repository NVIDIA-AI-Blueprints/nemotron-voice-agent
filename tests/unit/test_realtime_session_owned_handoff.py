# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103

"""A session-owned pipeline receives the routed socket before the gateway sends any event."""

from __future__ import annotations

import json
import unittest
from typing import Any

from realtime_helpers import FakeWebSocket

from realtime.gateway import _SESSION_OWNED_PIPELINES, handle_realtime_websocket

_MODEL = "test-realtime-model"
_VOICE = "Magpie-Multilingual.EN-US.Aria"


def _sanitizer(pipeline_mode: str):
    def _sanitize(data: dict[str, Any], **_: Any) -> dict[str, Any]:
        config = dict(data)
        config.setdefault("pipeline_mode", pipeline_mode)
        config.setdefault("model_id", _MODEL)
        config.setdefault("tts_voice_id", _VOICE)
        return config

    return _sanitize


async def _ready(_config: dict[str, Any]) -> None:
    return None


class SessionOwnedHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_verdict_pipeline_is_session_owned(self) -> None:
        self.assertIn("frontend-backend-verdict-agent", _SESSION_OWNED_PIPELINES)

    async def test_session_owned_pipeline_gets_the_socket_before_any_event(self) -> None:
        update = json.dumps({"type": "session.update", "session": {"type": "realtime"}})
        ws = FakeWebSocket([update])
        handed_off: list[tuple[dict[str, Any], int, list[str]]] = []

        async def start_bot(websocket, config, controller) -> None:  # noqa: ARG001
            # Nothing was sent and nothing was read: the pipeline owns the whole protocol.
            handed_off.append((config, len(websocket.sent), list(websocket._messages)))

        await handle_realtime_websocket(
            ws,
            sanitize_session_config=_sanitizer("frontend-backend-verdict-agent"),
            ensure_services_ready=_ready,
            start_bot=start_bot,
            default_example_key="frontend-backend-verdict-agent",
            default_pipeline_mode="frontend-backend-verdict-agent",
        )

        self.assertTrue(ws.accepted)
        self.assertEqual(len(handed_off), 1)
        config, sent_before_handoff, unread = handed_off[0]
        self.assertEqual(config["pipeline_mode"], "frontend-backend-verdict-agent")
        self.assertEqual(sent_before_handoff, 0)
        self.assertEqual(unread, [update])
        self.assertEqual(ws.sent, [])

    async def test_other_pipelines_keep_the_gateway_handshake(self) -> None:
        ws = FakeWebSocket([json.dumps({"type": "session.update", "session": {"type": "realtime"}})])
        handed_off: list[dict[str, Any]] = []

        async def start_bot(websocket, config, controller) -> None:  # noqa: ARG001
            handed_off.append(config)

        await handle_realtime_websocket(
            ws,
            sanitize_session_config=_sanitizer("generic-assistant"),
            ensure_services_ready=_ready,
            start_bot=start_bot,
            default_example_key="generic-assistant",
        )

        self.assertEqual(len(handed_off), 1)
        self.assertEqual(
            [event["type"] for event in ws.sent], ["session.created", "conversation.created", "session.updated"]
        )
