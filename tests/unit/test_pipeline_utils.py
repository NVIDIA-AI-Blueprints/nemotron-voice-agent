# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102

import asyncio
import unittest
from types import SimpleNamespace

from pipecat.frames.frames import InterruptionFrame, OutputTransportMessageFrame

from examples.shared.pipeline_utils import WebClientFrameSerializer, latency_metric_group


def _breakdown(**overrides):
    fields = {
        "user_turn_secs": 0.2,
        "ttfb": [SimpleNamespace(processor="NvidiaLLMService#1", duration_secs=0.0375)],
        "text_aggregation": None,
    }
    return SimpleNamespace(**(fields | overrides))


class LatencyMetricGroupTest(unittest.TestCase):
    def test_group_names_the_turn_and_reports_milliseconds(self):
        group = latency_metric_group(2, 0.5, _breakdown())

        self.assertEqual((group["type"], group["group_id"], group["group_label"]), ("metric-group", "turn-2", "Turn 2"))
        self.assertEqual(
            [(m["key"], m["value"]) for m in group["metrics"]],
            [("total_latency_ms", 500.0), ("user_turn_ms", 200.0), ("ttfb_0_NvidiaLLMService_1", 37.5)],
        )

    def test_group_skips_missing_user_turn(self):
        group = latency_metric_group(1, 0.25, _breakdown(user_turn_secs=None))
        self.assertNotIn("user_turn_ms", [m["key"] for m in group["metrics"]])


class WebClientFrameSerializerTest(unittest.TestCase):
    def test_skips_interruptions_and_keeps_messages(self):
        serializer = WebClientFrameSerializer()
        self.assertIsNone(asyncio.run(serializer.serialize(InterruptionFrame())))
        self.assertTrue(asyncio.run(serializer.serialize(OutputTransportMessageFrame(message={"label": "rtvi-ai"}))))


if __name__ == "__main__":
    unittest.main()
