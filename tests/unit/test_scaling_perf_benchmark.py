# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

SCALING_PERF_DIR = Path(__file__).resolve().parents[2] / "benchmarking_tools" / "scaling-perf"
sys.path.insert(0, str(SCALING_PERF_DIR))

from benchmark_aggregation import aggregate_run_dir, aggregate_suite_dir  # noqa: E402
from rtvi_client import RTVIClient  # noqa: E402


def _client_result(stream_id: str, ttfat: float, sample_count: int) -> dict:
    return {
        "stream_id": stream_id,
        "average_latency": 1.0,
        "individual_latencies": [1.0],
        "valid_latencies": [1.0],
        "num_turns": 1,
        "num_valid_turns": 1,
        "failed_turns": 0,
        "reverse_barge_ins_count": 0,
        "glitch_detected": False,
        "reverse_barge_in_threshold": 0.4,
        "turn_response_timeout": 10.0,
        "metrics_start_time": 1.0,
        "test_duration": 30.0,
        "server_metrics": {
            "samples": {"llm_ttfat": [ttfat] * sample_count},
            "average": {"llm_ttfat": ttfat},
            "sample_counts": {"llm_ttfat": sample_count},
        },
        "rtvi_messages": [],
        "timestamp": "2026-08-26T00:00:00",
        "error": None,
    }


class BenchmarkMetricTests(unittest.IsolatedAsyncioTestCase):
    async def test_process_server_message_captures_only_llm_ttfat(self) -> None:
        logger = types.SimpleNamespace(log=AsyncMock())
        client = RTVIClient(
            stream_id="client_1",
            host="localhost",
            port=7860,
            logger=logger,
            connect_timeout=1,
        )
        client.set_collecting_metrics(True)

        await client.process_server_message(
            {
                "type": "metrics",
                "data": {
                    "ttfat": [
                        {
                            "processor": "NvidiaLLMService#1",
                            "ttfat": 0.441,
                            "ttfb": 0.290,
                            "thinking_time": 0.151,
                        },
                        {"processor": "NvidiaTTSService#1", "ttfat": 9.0},
                        {"processor": "NvidiaLLMService#1", "ttfat": "invalid"},
                    ]
                },
            }
        )

        self.assertEqual(client.server_metric_samples["llm_ttfat"], [0.441])
        logger.log.assert_awaited_once()

    def test_run_and_suite_aggregation_export_llm_ttfat(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            clients = (
                _client_result("client_1", ttfat=0.2, sample_count=2),
                _client_result("client_2", ttfat=0.5, sample_count=1),
            )
            for client in clients:
                client_dir = run_dir / client["stream_id"]
                client_dir.mkdir()
                (client_dir / f"result_{client['stream_id']}.json").write_text(
                    json.dumps(client),
                    encoding="utf-8",
                )

            summary_path = aggregate_run_dir(run_dir, num_clients=2)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            server_metrics = summary["results"]["server_metrics"]

            self.assertAlmostEqual(server_metrics["average"]["llm_ttfat"], 0.3)
            self.assertEqual(server_metrics["sample_counts"]["llm_ttfat"], 3)

            tsv_path, _, json_path = aggregate_suite_dir(run_dir)
            tsv_lines = tsv_path.read_text(encoding="utf-8").splitlines()
            rows = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertIn("LLM TTFAT", tsv_lines[0].split("\t"))
        self.assertEqual([row["llm_ttfat"] for row in rows], [0.2, 0.5, 0.3])


if __name__ == "__main__":
    unittest.main()
