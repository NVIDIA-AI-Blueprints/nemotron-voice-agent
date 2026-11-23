# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scaling-perf diagnostics across protocol recovery and result aggregation."""

# ruff: noqa: D101,D102

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

SCALING_PERF_DIR = Path(__file__).resolve().parents[3] / "benchmarking_tools" / "scaling-perf"
sys.path.insert(0, str(SCALING_PERF_DIR))

from benchmark_aggregation import aggregate_run_dir  # noqa: E402
from benchmark_diagnostics import (  # noqa: E402
    DIAGNOSTICS_TYPE,
    MAX_ERROR_MESSAGE_CHARS,
    NO_VALID_TURNS,
    SessionDiagnostics,
)
from realtime_client import RealtimeClient  # noqa: E402
from realtime_transport import RealtimeTransport  # noqa: E402
from rtvi_client import RTVIClient  # noqa: E402


def _batch(*events: dict) -> dict:
    return {
        "type": DIAGNOSTICS_TYPE,
        "schema_version": 1,
        "session_id": "session-a",
        "events": [{"seq": idx, **event} for idx, event in enumerate(events, start=1)],
    }


class RecoveryDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    def _client(self) -> RealtimeClient:
        client = RealtimeClient(stream_id="client", ws_url="wss://example.test", logger=AsyncMock(), connect_timeout=1)
        client.diagnostics = SessionDiagnostics()
        client.realtime = RealtimeTransport(client.uri, on_event=client._capture_event)
        return client

    async def test_recovery_invalidates_equal_sized_audio_epochs_after_discard(self) -> None:
        client = self._client()
        collector = client.diagnostics
        collector.ingest(
            _batch(
                {"kind": "audio_epoch_start", "epoch": "a", "audio_sequence": 0, "output_audio_ms": 0},
                {"kind": "audio_epoch_end", "epoch": "a", "audio_sequence": 1, "output_audio_ms": 400},
                {
                    "kind": "audio_epoch_start",
                    "epoch": "b",
                    "audio_sequence": 1,
                    "output_audio_ms": 400,
                    "tool_call_ids": ["tool-1"],
                },
                {"kind": "audio_epoch_end", "epoch": "b", "audio_sequence": 2, "output_audio_ms": 800},
                {
                    "kind": "audio_epoch_start",
                    "epoch": "c",
                    "audio_sequence": 2,
                    "output_audio_ms": 800,
                    "tool_call_ids": ["tool-2"],
                },
                {"kind": "audio_epoch_end", "epoch": "c", "audio_sequence": 3, "output_audio_ms": 1200},
            )
        )

        def record_audio():
            collector.audio(
                nbytes=19_200, sample_rate=24_000, channels=1, monotonic_ns=2, unix_s=2, deficit_ms=80, detect=True
            )

        record_audio()
        record_audio()
        client.realtime.ws = AsyncMock()
        client.realtime.ws.recv.side_effect = [
            json.dumps({"type": "response.output_audio.delta", "delta": "AAAA" * 6400}),
            json.dumps({"type": "response.done", "response": {"status": "cancelled"}}),
        ]
        await client.recover_turn(timeout=1)
        record_audio()

        result = collector.finish()
        self.assertEqual(result["coverage"]["audio_discontinuity_after_sequence"], 2)
        self.assertEqual(result["glitch_events"][1]["classification"], "tool_wait")
        self.assertEqual(result["glitch_events"][2]["classification"], "unknown")
        self.assertEqual(result["glitch_events"][2]["evidence"], ["realtime_audio_discontinuity"])

    async def test_no_active_response_during_recovery_is_not_reported_as_error(self) -> None:
        client = self._client()
        client.realtime.ws = AsyncMock()
        event = {"type": "error", "error": {"code": "response_cancel_not_active", "message": "No active response"}}
        client.realtime.ws.recv.return_value = json.dumps(event)

        await client.recover_turn(timeout=1)
        self.assertEqual(client.diagnostics.errors, [])
        self.assertIsNone(client.diagnostics.audio_discontinuity_after_sequence)

        await client.realtime.recv_event(timeout=1)
        self.assertEqual(client.diagnostics.errors[0]["code"], "response_cancel_not_active")

    async def test_realtime_diagnostic_events_do_not_extend_turn_history(self) -> None:
        client = self._client()
        client.realtime.ws = AsyncMock()
        client.realtime.ws.recv.return_value = json.dumps(_batch({"kind": "session_start"}))

        await client.realtime.recv_event(timeout=1)

        self.assertEqual(client.diagnostics.events, [{"seq": 1, "kind": "session_start"}])
        self.assertEqual(client.realtime.events, [])

    async def test_rtvi_diagnostic_batches_are_kept_only_in_diagnostics(self) -> None:
        client = RTVIClient(stream_id="client", host="example.test", port=443, logger=AsyncMock(), connect_timeout=1)
        client.diagnostics = SessionDiagnostics()

        await client.process_server_message({"type": "server-message", "data": _batch({"kind": "session_start"})})

        self.assertEqual(client.diagnostics.events, [{"seq": 1, "kind": "session_start"}])
        self.assertEqual(client.rtvi_messages, [])
        client.logger.log.assert_not_awaited()

        await client.process_server_message({"type": "bot-started-speaking"})
        self.assertEqual(client.rtvi_messages[0]["type"], "bot-started-speaking")
        client.logger.log.assert_awaited_once()


class SessionReportTests(unittest.TestCase):
    def test_old_no_valid_turns_sentinel_counts_as_no_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            errors = [NO_VALID_TURNS, "Hard deadline reached (60s)", "WebSocket connection closed"]
            for idx, reason in enumerate(errors, start=1):
                client_dir = run_dir / f"client_{idx}"
                client_dir.mkdir()
                (client_dir / "result_client.json").write_text(
                    json.dumps({"stream_id": f"client_{idx}", "num_valid_turns": 0, "error": reason}),
                    encoding="utf-8",
                )
            result = json.loads(aggregate_run_dir(run_dir, len(errors)).read_text(encoding="utf-8"))["results"]
            self.assertEqual(result["no_response_clients"], 2)
            self.assertEqual(result["failed_clients"], 1)
            self.assertEqual(result["successful_clients"], 0)
            quality = json.loads((run_dir / "session_quality.json").read_text(encoding="utf-8"))
            self.assertEqual([client["status"] for client in quality], ["no_response", "no_response", "failed"])
            self.assertEqual(quality[0]["reported_errors"], 0)

    def test_error_normalization_redacts_and_preserves_initial_phase(self) -> None:
        collector = SessionDiagnostics()
        collector.start = 10
        collector.duration = 10
        message = "Failed with Authorization: Bearer dummy-test-secret"
        collector.add_error("  " + message, source="protocol", received_unix_s=9.9)
        collector.add_error(message, source="pipecat_frame", received_unix_s=10.1, component="tts", fatal=False)
        self.assertEqual(len(collector.errors), 1)
        self.assertEqual(collector.errors[0]["phase"], "warmup")
        self.assertEqual(collector.errors[0]["component"], "tts")
        self.assertNotIn("dummy-test-secret", collector.errors[0]["message"])
        collector.add_error(message, source="tool", received_unix_s=10.2, component="tool")
        self.assertEqual(len(collector.errors), 2)

    def test_stored_diagnostic_errors_are_bounded_and_never_merge_on_clipped_prefix(self) -> None:
        collector = SessionDiagnostics()
        message = ("x" * (MAX_ERROR_MESSAGE_CHARS - 32)) + " Authorization: Bearer dummy-test-secret " + ("x" * 64)
        collector.add_error(message + " first", source="protocol")
        collector.ingest(_batch({"kind": "error", "message": message + " second"}))

        self.assertEqual(len(collector.errors), 2)
        stored = collector.events[0]["message"]
        self.assertNotIn("dummy-test-secret", stored)
        self.assertEqual(len(stored), MAX_ERROR_MESSAGE_CHARS)
