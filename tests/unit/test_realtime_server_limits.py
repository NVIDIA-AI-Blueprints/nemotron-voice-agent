# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Tests for the WebSocket budget shared by both Realtime server entry points."""

from __future__ import annotations

import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import realtime_server
import server
from realtime.audio import max_base64_audio_chars
from realtime.protocol import MAX_AUDIO_APPEND_BYTES, MAX_REALTIME_EVENT_BYTES


class RealtimeServerWebSocketLimitTests(unittest.TestCase):
    """Keep transport and protocol message budgets aligned."""

    def test_wire_budget_can_carry_the_maximum_audio_append(self) -> None:
        """Account for Base64 expansion and the surrounding JSON event."""
        envelope_overhead = len(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "event_id": "x" * 512,
                    "audio": "",
                },
                separators=(",", ":"),
            ).encode("utf-8")
        )
        self.assertGreaterEqual(
            MAX_REALTIME_EVENT_BYTES,
            max_base64_audio_chars(MAX_AUDIO_APPEND_BYTES) + envelope_overhead,
        )

    def test_combined_server_sets_the_budget_for_single_and_multi_worker_runs(self) -> None:
        """Configure the combined server identically for either worker mode."""
        args = SimpleNamespace(host="127.0.0.1", port=7860)
        app = Mock()
        with patch.object(server.uvicorn, "run") as run:
            server._run_single_worker(args, app, {})
            self.assertEqual(run.call_args.kwargs["ws_max_size"], MAX_REALTIME_EVENT_BYTES)

            run.reset_mock()
            server._run_multi_worker(args, 3, {})
            self.assertEqual(run.call_args.kwargs["ws_max_size"], MAX_REALTIME_EVENT_BYTES)

    def test_standalone_server_sets_the_budget_for_single_and_multi_worker_runs(self) -> None:
        """Configure the API-only server identically for either worker mode."""
        app = Mock()
        for workers in (1, 3):
            with (
                self.subTest(workers=workers),
                patch.object(sys, "argv", ["realtime_server.py", "--workers", str(workers)]),
                patch.object(realtime_server, "_configure_logging"),
                patch.object(realtime_server, "_ssl_config", return_value=("http", {})),
                patch.object(realtime_server, "create_realtime_app", return_value=app),
                patch.object(realtime_server.uvicorn, "run") as run,
                patch.dict("os.environ", {}, clear=False),
            ):
                realtime_server.main()

            self.assertEqual(run.call_args.kwargs["ws_max_size"], MAX_REALTIME_EVENT_BYTES)


if __name__ == "__main__":
    unittest.main()
