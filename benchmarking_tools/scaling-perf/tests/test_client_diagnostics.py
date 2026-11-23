# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise default diagnostics using native client events and audio."""

# ruff: noqa: D101, D102

import asyncio
import csv
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_core import AudioFrame, EndOfBotUtterance, PerfClient  # noqa: E402
from benchmark_diagnostics import (  # noqa: E402
    NO_VALID_TURNS,
    SessionDiagnostics,
    quality_status,
    write_diagnostic_reports,
)


class NativeTrace:
    """Build one benchmark turn with known receipt timestamps in milliseconds."""

    def __init__(self):
        """Set up a diagnostics session and start its first input turn."""
        self.diagnostics = SessionDiagnostics()
        self.diagnostics.start = 100.0
        self.diagnostics.duration = 10.0
        self.diagnostics.begin_turn("input.wav", monotonic_ns=0, unix_s=100.0)

    def event(self, ms, kind, **fields):
        """Deliver a native message at a specific receipt time."""
        self.diagnostics.ingest(
            {"type": kind, **fields}, received_monotonic_ns=ms * 1_000_000, received_unix_s=100 + ms / 1000
        )

    def audio(self, ms, *, deficit=None, response=None):
        """Deliver 100 ms of PCM with a known playback deficit."""
        self.diagnostics.audio(
            nbytes=3200,
            sample_rate=16000,
            channels=1,
            monotonic_ns=ms * 1_000_000,
            unix_s=100 + ms / 1000,
            deficit_ms=deficit,
            detect=True,
            response_id=response,
        )

    def first_reply_and_tool(self):
        """Deliver an initial spoken reply and a completed RTVI tool call."""
        self.event(110, "bot-llm-started")
        self.event(130, "bot-started-speaking")
        self.audio(140)
        self.event(150, "bot-llm-stopped")
        self.event(160, "llm-function-call-in-progress", data={"tool_call_id": "call-1", "args": "fixture-secret"})
        self.event(230, "bot-stopped-speaking")
        self.event(240, "llm-function-call-stopped", data={"tool_call_id": "call-1", "result": "fixture-secret"})

    def finish(self):
        """Close the input turn and return the reported diagnostics."""
        self.diagnostics.end_turn(input_end_monotonic_s=0.1)
        return self.diagnostics.finish()


class NativeDiagnosticsTests(unittest.TestCase):
    def test_rtvi_rounds_first_audio_and_tool_gap_are_independent(self):
        trace = NativeTrace()
        trace.first_reply_and_tool()
        trace.event(260, "bot-llm-started")
        trace.event(300, "bot-started-speaking")
        trace.audio(330, deficit=90)
        trace.event(440, "bot-stopped-speaking")
        trace.event(460, "bot-started-speaking")
        trace.audio(480, deficit=50)
        trace.event(500, "bot-llm-stopped")
        diagnostics = trace.finish()
        turn = diagnostics["turns"][0]
        followup = turn["tool_followups"][0]
        self.assertEqual(turn["llm_round_trips"], 2)
        self.assertEqual(turn["first_audio_latency_ms"], 40)
        self.assertEqual(followup["tool_observed_ms"], 80)
        self.assertEqual(followup["tool_completion_to_llm_start_ms"], 20)
        self.assertEqual(followup["llm_start_to_followup_audio_ms"], 70)
        self.assertEqual(followup["playback_gap_ms"], 90)
        self.assertEqual([gap["classification"] for gap in diagnostics["glitch_events"]], ["tool_wait", "unknown"])
        self.assertNotIn("fixture-secret", json.dumps(diagnostics))

    def test_later_speech_start_cannot_replace_first_followup_pcm(self):
        trace = NativeTrace()
        trace.first_reply_and_tool()
        trace.event(260, "bot-llm-started")
        trace.audio(330, deficit=90)
        trace.event(440, "bot-stopped-speaking")
        trace.event(460, "bot-started-speaking")
        trace.audio(480, deficit=50)
        diagnostics = trace.finish()
        followup = diagnostics["turns"][0]["tool_followups"][0]
        self.assertEqual(followup["llm_start_to_followup_audio_ms"], 70)
        self.assertIsNone(followup["playback_gap_ms"])
        self.assertEqual({gap["classification"] for gap in diagnostics["glitch_events"]}, {"unknown"})

    def test_silent_round_does_not_claim_later_rounds_audio(self):
        trace = NativeTrace()
        trace.event(110, "bot-llm-started")
        trace.event(120, "bot-llm-stopped")
        trace.event(140, "bot-llm-started")
        trace.audio(180)
        diagnostics = trace.finish()
        turns = diagnostics["turns"][0]["round_trips"]
        self.assertIsNone(turns[0]["first_pcm_after_start_monotonic_ns"])
        self.assertEqual(turns[1]["first_pcm_after_start_monotonic_ns"], 180_000_000)

    def test_missing_followup_audio_remains_in_reports(self):
        trace = NativeTrace()
        trace.first_reply_and_tool()
        trace.event(260, "bot-llm-started")
        diagnostics = trace.finish()
        followup = diagnostics["turns"][0]["tool_followups"][0]
        self.assertIsNone(followup["followup_audio_received_monotonic_ns"])
        self.assertIn("no_followup_audio_observed", followup["evidence"])
        with TemporaryDirectory() as directory:
            path = Path(directory)
            write_diagnostic_reports(path, [{"stream_id": "client-1", "diagnostics": diagnostics}])
            turns = json.loads((path / "turn_timing.json").read_text())
            self.assertEqual(turns[0]["llm_round_trips"], 2)
            with (path / "tool_followups.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["tool_call_id"], "call-1")
            self.assertEqual(rows[0]["followup_audio_received_monotonic_ns"], "")

    def test_realtime_response_is_not_an_llm_round(self):
        trace = NativeTrace()
        trace.event(120, "response.created", response={"id": "first"})
        trace.audio(140, response="first")
        trace.event(160, "nvidia.tool.started", tool_call_id="call-1")
        trace.event(240, "nvidia.tool.completed", tool_call_id="call-1")
        trace.event(260, "response.created", response={"id": "followup"})
        trace.audio(300, response="first")
        trace.audio(330, response="followup", deficit=30)
        diagnostics = trace.finish()
        turn = diagnostics["turns"][0]
        self.assertIsNone(turn["llm_round_trips"])
        self.assertEqual(turn["tool_followups"][0]["tool_completion_to_followup_audio_ms"], 90)
        self.assertEqual(diagnostics["glitch_events"][0]["classification"], "unknown")

    def test_discarded_audio_prevents_gap_attribution(self):
        trace = NativeTrace()
        trace.first_reply_and_tool()
        trace.diagnostics.mark_audio_discontinuity()
        trace.event(260, "bot-llm-started")
        trace.event(300, "bot-started-speaking")
        trace.audio(330, deficit=90)
        diagnostics = trace.finish()
        self.assertEqual(diagnostics["coverage"]["audio_discontinuity_after_sequence"], 1)
        self.assertEqual(diagnostics["glitch_events"][0]["classification"], "unknown")

    def test_no_valid_turn_sentinel_and_runtime_error_have_different_status(self):
        self.assertEqual(quality_status({"num_valid_turns": 0, "error": NO_VALID_TURNS}), "no_response")
        self.assertEqual(quality_status({"num_valid_turns": 0, "error": "WebSocket connection closed"}), "failed")


class PerfClientTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_audio_is_measured_at_receipt_once_per_turn(self):
        protocol = SimpleNamespace(
            begin_turn_observation=lambda: None,
            record_turn_observation=AsyncMock(),
            recv_audio=AsyncMock(
                side_effect=[
                    AudioFrame(bytes(3200), 16000, 1, 1_250_000_000, 100.25),
                    AudioFrame(bytes(3200), 16000, 1, 2_800_000_000, 101.80),
                    EndOfBotUtterance,
                ]
            ),
        )
        client = PerfClient(
            stream_id="client-1",
            protocol_client=protocol,
            audio_files=[],
            start_delay=0,
            metrics_start_time=100,
            session_end_time=None,
            test_duration=10,
            reverse_barge_in_threshold=0,
            audio_output_path=None,
            logger=SimpleNamespace(log=AsyncMock()),
        )
        client.collecting_metrics = True
        client.silence_event = asyncio.Event()

        async def send(_):
            client.input_audio_file_end_monotonic = 1

        client._send_audio_file = send
        await client._process_conversation_turn(Path("input.wav"), None)
        result = client.diagnostics.finish()
        self.assertEqual(client.valid_latency_values, [0.25])
        self.assertEqual(result["turns"][0]["first_audio_latency_ms"], 250)
        self.assertEqual(result["turns"][0]["audio_packets_observed"], 2)
        self.assertIn("event_loop_samples", result["turns"][0]["client_timing"])
