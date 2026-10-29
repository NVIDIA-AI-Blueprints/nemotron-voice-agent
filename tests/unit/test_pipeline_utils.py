# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102

import asyncio
import json
import unittest
from types import SimpleNamespace

from pipecat.evals.fastapi_transport import EvalFastAPIWebsocketTransport
from pipecat.frames.frames import InterruptionFrame, OutputTransportMessageFrame
from pipecat.runner.types import WebSocketRunnerArguments

from examples.shared.pipeline_utils import (
    WebClientFrameSerializer,
    create_transport,
    latency_metric_group,
    turn_taking_latency_secs,
)


def _breakdown(**overrides):
    fields = {
        "user_turn_secs": 0.2,
        "ttfb": [SimpleNamespace(processor="NvidiaLLMService#1", duration_secs=0.0375)],
        "text_aggregation": None,
        "contributions": [
            SimpleNamespace(key="endpointing_wait", start_time=10.0, duration_secs=0.2),
            SimpleNamespace(key="transcription", start_time=10.2, duration_secs=0.3),
            SimpleNamespace(key="turn_detection", start_time=10.5, duration_secs=0.25),
            SimpleNamespace(key="llm_inference", start_time=10.75, duration_secs=0.4),
        ],
    }
    return SimpleNamespace(**(fields | overrides))


class LatencyMetricGroupTest(unittest.TestCase):
    def test_group_names_the_turn_and_reports_milliseconds(self):
        group = latency_metric_group(2, 0.5, _breakdown())

        self.assertEqual((group["type"], group["group_id"], group["group_label"]), ("metric-group", "turn-2", "Turn 2"))
        self.assertEqual(
            [(m["key"], m["value"]) for m in group["metrics"]],
            [
                ("total_latency_ms", 500.0),
                ("user_turn_ms", 200.0),
                ("turn_taking_latency_ms", 550.0),
                ("ttfb_0_NvidiaLLMService_1", 37.5),
            ],
        )

    def test_group_skips_missing_user_turn(self):
        group = latency_metric_group(1, 0.25, _breakdown(user_turn_secs=None))
        self.assertNotIn("user_turn_ms", [m["key"] for m in group["metrics"]])

    def test_turn_taking_latency_is_vad_stop_to_llm_request(self):
        self.assertAlmostEqual(turn_taking_latency_secs(_breakdown()), 0.55)

    def test_turn_taking_latency_requires_vad_and_llm_boundaries(self):
        self.assertIsNone(turn_taking_latency_secs(_breakdown(contributions=[])))


class WebClientFrameSerializerTest(unittest.TestCase):
    def test_skips_interruptions_and_keeps_messages(self):
        serializer = WebClientFrameSerializer()
        self.assertIsNone(asyncio.run(serializer.serialize(InterruptionFrame())))
        self.assertTrue(asyncio.run(serializer.serialize(OutputTransportMessageFrame(message={"label": "rtvi-ai"}))))


class EvalTransportFactoryTest(unittest.IsolatedAsyncioTestCase):
    async def test_accepted_eval_websocket_uses_native_eval_transport(self):
        class _WebSocket:
            def iter_text(self):
                async def messages():
                    yield json.dumps(
                        {
                            "label": "rtvi-ai",
                            "type": "client-message",
                            "data": {"t": "eval-connect", "d": {"skip_tts": True}},
                        }
                    )

                return messages()

        args = WebSocketRunnerArguments(websocket=_WebSocket())
        transport = await create_transport(args)

        self.assertIsInstance(transport, EvalFastAPIWebsocketTransport)
        self.assertEqual(args.transport_type, "eval")


if __name__ == "__main__":
    unittest.main()
