# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103

"""In-process Realtime WS e2e tests (no live NIM)."""

from __future__ import annotations

import base64
import json
import unittest

from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from realtime.gateway import handle_realtime_websocket
from realtime.transport import create_realtime_transport


def _build_test_app(
    *,
    bot_events: list[str] | None = None,
    ensure_ready=None,
) -> FastAPI:
    """Minimal FastAPI app exposing /v1/realtime with mocked readiness/bot."""
    app = FastAPI()
    events = bot_events if bot_events is not None else []

    @app.websocket("/v1/realtime")
    async def realtime_endpoint(websocket: WebSocket):
        async def _default_ready(config: dict) -> None:
            return None

        async def start_bot(ws: WebSocket, config: dict, session_view: dict) -> None:
            _ = create_realtime_transport(ws, session_view=session_view)
            events.append("bot_started")
            events.append(str(config.get("pipeline_mode", "")))
            await ws.close()

        await handle_realtime_websocket(
            websocket,
            sanitize_session_config=lambda data, **_: {
                **data,
                "pipeline_mode": data.get("pipeline_mode") or "generic-assistant",
            },
            ensure_services_ready=ensure_ready or _default_ready,
            start_bot=start_bot,
            fallback_example_key="generic-assistant",
        )

    return app


class RealtimeE2ETests(unittest.TestCase):
    """End-to-end-ish WS flows against an in-process FastAPI app."""

    def test_connect_and_handoff(self) -> None:
        bot_events: list[str] = []
        app = _build_test_app(bot_events=bot_events)
        client = TestClient(app)
        with client.websocket_connect("/v1/realtime") as ws:
            created = ws.receive_json()
            self.assertEqual(created["type"], "session.created")
            ws.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "instructions": "Be brief.",
                        "audio": {
                            "input": {"format": {"type": "audio/pcm", "rate": 16000}},
                            "output": {"format": {"type": "audio/pcm", "rate": 16000}},
                        },
                    },
                }
            )
            updated = ws.receive_json()
            self.assertEqual(updated["type"], "session.updated")
        self.assertIn("bot_started", bot_events)
        self.assertIn("generic-assistant", bot_events)

    def test_connect_with_optional_client_headers(self) -> None:
        """Stock clients may send Authorization; the gateway does not require it."""
        app = _build_test_app()
        client = TestClient(app)
        with client.websocket_connect(
            "/v1/realtime",
            headers={"Authorization": "Bearer optional-client-header"},
        ) as ws:
            created = ws.receive_json()
            self.assertEqual(created["type"], "session.created")

    def test_readiness_failure_keeps_socket_open_for_retry(self) -> None:
        bot_events: list[str] = []
        calls = {"n": 0}

        async def ensure_ready(config: dict) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("ASR not reachable")

        app = _build_test_app(bot_events=bot_events, ensure_ready=ensure_ready)
        client = TestClient(app)
        with client.websocket_connect("/v1/realtime") as ws:
            created = ws.receive_json()
            self.assertEqual(created["type"], "session.created")
            update = {
                "type": "session.update",
                "session": {"type": "realtime"},
            }
            ws.send_json(update)
            err = ws.receive_json()
            self.assertEqual(err["type"], "error")
            self.assertEqual(err["error"]["code"], "services_not_ready")

            ws.send_json(update)
            updated = ws.receive_json()
            self.assertEqual(updated["type"], "session.updated")

        self.assertEqual(calls["n"], 2)
        self.assertIn("bot_started", bot_events)

    def test_append_pcm_payload_shape(self) -> None:
        """Scaling-perf style: build a silence PCM chunk clients would append."""
        pcm = b"\x00\x00" * 320  # 20ms mono s16le @ 16kHz
        b64 = base64.b64encode(pcm).decode("ascii")
        event = {"type": "input_audio_buffer.append", "audio": b64}
        self.assertTrue(json.dumps(event))
        self.assertEqual(base64.b64decode(b64), pcm)


if __name__ == "__main__":
    unittest.main()
