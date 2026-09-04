# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100,D101,D102

import json
import sys
import unittest
from pathlib import Path

SCALING_PERF_DIR = Path(__file__).resolve().parents[2] / "benchmarking_tools" / "scaling-perf"
sys.path.insert(0, str(SCALING_PERF_DIR))

from openai_realtime_ws import OpenAIRealtimeSocket, RealtimeProtocolError, RealtimeTurnError  # noqa: E402


class FakeWebSocket:
    def __init__(self, *events: object):
        """Create a fake socket that yields serialized Realtime events."""
        self.events = iter(json.dumps(event) for event in events)
        self.sent: list[dict] = []

    async def recv(self) -> str:
        return next(self.events)

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


class RealtimeSocketTests(unittest.IsolatedAsyncioTestCase):
    async def test_recv_event_rejects_non_object_json(self) -> None:
        socket = OpenAIRealtimeSocket("wss://example.test")
        socket.ws = FakeWebSocket(["not", "an", "object"])

        with self.assertRaisesRegex(RealtimeProtocolError, "expected a JSON object"):
            await socket.recv_event()

    async def test_configure_waits_for_session_updated(self) -> None:
        socket = OpenAIRealtimeSocket("wss://example.test")
        websocket = FakeWebSocket(
            {"type": "session.created", "session": {}},
            {"type": "session.updated", "session": {"audio": {"output": {"format": {"rate": 16_000}}}}},
        )
        socket.ws = websocket

        event = await socket.configure_input(24_000)

        self.assertEqual(event["type"], "session.updated")
        self.assertEqual(socket.output_sample_rate, 16_000)
        self.assertEqual(websocket.sent[0]["type"], "session.update")

    async def test_item_failure_details_are_retained_for_turn_metrics(self) -> None:
        socket = OpenAIRealtimeSocket("wss://example.test")
        socket.ws = FakeWebSocket(
            {
                "type": "conversation.item.input_audio_transcription.failed",
                "error": {"message": "transcription unavailable"},
            }
        )

        await socket.recv_event()

        self.assertEqual(socket.events[0]["error"], "transcription unavailable")

    async def test_noncompleted_response_is_terminal_turn_failure(self) -> None:
        socket = OpenAIRealtimeSocket("wss://example.test")
        socket.ws = FakeWebSocket(
            {
                "type": "response.done",
                "response": {
                    "status": "failed",
                    "status_details": {"error": {"message": "generation failed"}},
                },
            }
        )

        with self.assertRaises(RealtimeTurnError) as raised:
            await socket.recv_audio()

        self.assertTrue(raised.exception.terminal)
        self.assertIn("generation failed", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
