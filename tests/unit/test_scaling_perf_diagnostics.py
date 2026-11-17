# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise diagnostic evidence, conservative classification, and session quality."""

# ruff: noqa: D101, D102

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarking_tools" / "scaling-perf"))

from benchmark_aggregation import aggregate_run_dir  # noqa: E402
from benchmark_diagnostics import (  # noqa: E402
    DIAGNOSTICS_TYPE,
    SessionDiagnostics,
    quality_status,
    session_errors,
    write_diagnostic_reports,
)
from pipecat.frames.frames import (  # noqa: E402
    AggregatedTextFrame,
    ErrorFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMFullResponseStartFrame,
    OutputAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.observers.base_observer import FramePushed  # noqa: E402
from pipecat.processors.frame_processor import FrameDirection  # noqa: E402

from examples.shared.perf_diagnostics import PerfDiagnosticsObserver  # noqa: E402


def batch(events, **extra):
    """Wrap synthetic server evidence in the actual on-wire envelope."""
    return {
        "type": DIAGNOSTICS_TYPE,
        "schema_version": 1,
        "session_id": "test",
        "events": [{"seq": i + 1, **e} for i, e in enumerate(events)],
        **extra,
    }


def collected_gap(*, boundary=0, aligned=True, metadata=True, end=True):
    """Receive audio first and semantic metadata afterward, like a delayed batch."""
    collector = SessionDiagnostics()
    collector.start, collector.duration = 100, 10
    for i in range(2):
        collector.audio(
            nbytes=800,
            sample_rate=1000,
            channels=1,
            monotonic_ns=i * 500000000,
            unix_s=101 + i,
            deficit_ms=80 if i else None,
            detect=True,
        )
    events = [
        {"kind": "audio_epoch_start", "epoch": "a", "audio_sequence": 0, "output_audio_ms": 0},
        {"kind": "sentence_boundary", "epoch": "a", "source_audio_ms": boundary, "sentence_aligned": aligned},
    ]
    if end:
        events.append({"kind": "audio_epoch_end", "epoch": "a", "audio_sequence": 2, "output_audio_ms": 800})
    if metadata:
        collector.ingest(batch(events))
    return collector


class GapClassificationTests(unittest.TestCase):
    def test_late_metadata_classifies_within_sentence(self):
        gap = collected_gap().finish()["glitch_events"][0]
        self.assertEqual(gap["classification"], "within_sentence")
        self.assertEqual(gap["epoch_audio_offset_ms"], 400)
        self.assertEqual(gap["deficit_ms"], 80)

    def test_sentence_boundary_uses_audio_offset(self):
        self.assertEqual(
            collected_gap(boundary=400).finish()["glitch_events"][0]["classification"], "between_sentences"
        )

    def test_token_streaming_unaligned_text_is_unknown(self):
        self.assertEqual(collected_gap(aligned=False).finish()["glitch_events"][0]["classification"], "unknown")

    def test_missing_metadata_or_end_is_unknown(self):
        for collector in (collected_gap(metadata=False), collected_gap(end=False)):
            self.assertEqual(collector.finish()["glitch_events"][0]["classification"], "unknown")

    def test_dropped_or_mismatched_evidence_is_unknown(self):
        for damage in ("dropped", "sequence", "duration"):
            collector = collected_gap()
            if damage == "dropped":
                collector.server_dropped = 1
            elif damage == "sequence":
                collector.sequence_gaps = 1
            else:
                collector.events[-1]["output_audio_ms"] = 799 - 10
            self.assertEqual(collector.finish()["glitch_events"][0]["classification"], "unknown")

    def test_tool_followup_and_interruption_are_context_not_cause(self):
        collector = collected_gap()
        collector.events.insert(1, {"kind": "interruption", "audio_sequence": 1})
        self.assertEqual(collector.finish()["glitch_events"][0]["classification"], "interruption")
        collector.events[1] = {
            "kind": "audio_epoch_start",
            "epoch": "b",
            "audio_sequence": 1,
            "output_audio_ms": 400,
            "tool_call_ids": ["tool-1"],
        }
        collector.events[-1] = {"kind": "audio_epoch_end", "epoch": "b", "audio_sequence": 2, "output_audio_ms": 800}
        gap = collector.finish()["glitch_events"][0]
        self.assertEqual(gap["classification"], "tool_wait")
        self.assertIn("not_proof_of_tool_execution_duration", gap["evidence"])

    def test_initial_audio_is_not_a_between_sentence_gap(self):
        collector = collected_gap()
        collector.gaps = [{**collector.audio_packets[0], "deficit_ms": 80}]
        self.assertEqual(collector.finish()["glitch_events"][0]["classification"], "unknown")

    def test_phase_capture_and_bounded_storage(self):
        collector = SessionDiagnostics(max_events=1)
        collector.start, collector.duration = 100, 10
        self.assertEqual([collector.phase(t) for t in (99, 105, 111)], ["warmup", "measurement", "reply_drain"])
        collector.ingest(batch([{"kind": "llm_start"}, {"kind": "llm_end"}]))
        self.assertEqual(len(collector.events), 1)
        self.assertEqual(collector.finish()["coverage"]["local_dropped_events"], 1)


class ErrorReportingTests(unittest.TestCase):
    def test_protocol_and_frame_duplicate_preserves_nonfatal_details(self):
        collector = SessionDiagnostics()
        collector.ingest({"type": "error", "data": {"message": "TTS queue full"}})
        collector.ingest(batch([{"kind": "error", "message": "TTS queue full", "component": "tts", "fatal": False}]))
        self.assertEqual(len(collector.errors), 1)
        self.assertEqual(collector.errors[0]["component"], "tts")
        self.assertFalse(collector.errors[0]["fatal"])
        self.assertEqual(collector.errors[0]["also_reported_by"], ["protocol"])
        collector.ingest({"type": "error", "data": {"message": "another error"}})
        collector.ingest({"type": "error", "data": {"message": "another error"}})
        self.assertEqual(len(collector.errors), 3)

    def test_errors_survive_successful_turns_and_audio_flags_are_independent(self):
        client = {"num_valid_turns": 3, "average_latency": 1, "error": None, "glitch_detected": True}
        self.assertEqual(quality_status(client), "successful")
        client["session_errors"] = [{"message": "Recovered ASR failure", "fatal": False}]
        self.assertEqual(quality_status(client), "degraded")
        client["session_errors"] = []
        client["failed_turns"] = 1
        self.assertEqual(quality_status(client), "degraded")
        client["error"] = "connection closed"
        self.assertEqual(quality_status(client), "failed")

    def test_cancelled_response_is_not_a_provider_error(self):
        collector = SessionDiagnostics()
        collector.ingest({"type": "response.done", "response": {"status": "cancelled"}})
        self.assertEqual(collector.errors, [])
        collector.ingest({"type": "response.done", "response": {"status": "failed"}})
        self.assertEqual(len(collector.errors), 1)

    def test_legacy_errors_have_unknown_timing(self):
        errors = session_errors(
            {
                "rtvi_messages": [
                    {
                        "timestamp": "2026-01-01T00:00:00",
                        "data": {"type": "error", "data": {"message": "TTS queue full"}},
                    }
                ]
            }
        )
        self.assertEqual(errors[0]["phase"], "unknown")
        self.assertIsNone(errors[0]["received_unix_s"])
        self.assertEqual(errors[0]["recorded_timestamp"], "2026-01-01T00:00:00")

    def test_reports_include_failed_clients_and_flag_missing_legacy_evidence(self):
        collector = collected_gap()
        clients = [
            {
                "stream_id": "client_1",
                "num_valid_turns": 2,
                "average_latency": 1,
                "diagnostics": collector.finish(),
                "session_errors": [],
                "glitch_detected": True,
            },
            {"stream_id": "client_2", "error": "closed", "glitch_detected": True},
        ]
        with TemporaryDirectory() as tmp:
            summary = write_diagnostic_reports(Path(tmp), clients)
            self.assertEqual(summary["measurement"]["within_sentence"]["events"], 1)
            self.assertEqual(summary["measurement"]["within_sentence"]["duration_percent"], 100)
            self.assertEqual(summary["flagged_clients_without_event_evidence"], 1)
            self.assertEqual(json.loads((Path(tmp) / "session_errors.json").read_text())[0]["client"], "client_2")
            self.assertTrue((Path(tmp) / "session_quality.csv").exists())

    def test_aggregate_no_longer_counts_recovered_error_as_successful(self):
        client = {
            "stream_id": "client_1",
            "average_latency": 1,
            "valid_latencies": [1],
            "num_valid_turns": 1,
            "session_errors": [{"message": "TTS queue full", "component": "tts"}],
        }
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "client_1").mkdir()
            (root / "client_1" / "result_1.json").write_text(json.dumps(client))
            summary = json.loads(aggregate_run_dir(root, 1).read_text())["results"]
            self.assertEqual(summary["successful_clients"], 0)
            self.assertEqual(summary["degraded_clients"], 1)
            self.assertEqual(summary["completed_clients"], 1)
            self.assertEqual(summary["error_detection"]["reported_error_messages"], 1)


class ObserverTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.output = SimpleNamespace(name="Output")
        self.tts = SimpleNamespace(name="NvidiaTTSService", _is_streaming_tokens=False)
        self.emit = AsyncMock()
        self.observer = PerfDiagnosticsObserver(self.output, self.emit)

    async def push(self, frame, *, output=False, first=True):
        await self.observer.on_push_frame(
            FramePushed(
                source=self.output if output else self.tts,
                destination=SimpleNamespace() if output else self.output,
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=1234,
                first_push=first,
            )
        )

    async def test_offsets_follow_queued_pcm_and_no_content_is_exported(self):
        text = AggregatedTextFrame("private spoken content", "sentence")
        text.context_id, text.will_be_spoken = "ctx", True
        await self.push(text)  # The native service announces before TTSStartedFrame.
        start = TTSStartedFrame(context_id="ctx")
        await self.push(start)
        await self.push(text, output=True, first=False)
        await self.push(start, output=True, first=False)
        audio = OutputAudioRawFrame(audio=b"\x00" * 800, sample_rate=1000, num_channels=1)
        await self.push(audio)
        await self.push(audio, output=True, first=False)
        await self.push(TTSStoppedFrame(context_id="ctx"), output=True)
        await self.observer.flush()
        payload = self.emit.call_args.args[0]
        encoded = json.dumps(payload)
        self.assertNotIn("private spoken content", encoded)
        events = payload["events"]
        boundary = next(e for e in events if e["kind"] == "sentence_boundary")
        self.assertTrue(boundary["sentence_aligned"])
        self.assertEqual(boundary["source_audio_ms"], 0)
        self.assertEqual(events[-1]["output_audio_ms"], 400)
        self.assertEqual(events[-1]["audio_sequence"], 1)

    async def test_tools_export_identifiers_without_arguments_or_results(self):
        await self.push(FunctionCallInProgressFrame("weather", "t1", {"secret": "private arguments"}))
        await self.push(FunctionCallResultFrame("weather", "t1", {}, "private result"))
        await self.push(LLMFullResponseStartFrame())
        await self.push(TTSStartedFrame(context_id="followup"))
        await self.observer.flush()
        encoded = json.dumps(self.emit.call_args.args[0])
        self.assertNotIn("private arguments", encoded)
        self.assertNotIn("private result", encoded)
        self.assertEqual(self.observer.epochs["followup"]["tool_call_ids"], ["t1"])

    async def test_error_frame_is_recorded_once_with_component(self):
        frame = ErrorFrame("queue full", processor=self.tts, fatal=False)
        await self.push(frame)
        await self.push(frame, output=True, first=False)
        self.assertEqual(len(self.observer.pending), 1)
        self.assertEqual(self.observer.pending[0]["component"], "tts")

    async def test_token_text_has_no_claimed_sentence_alignment(self):
        self.tts._is_streaming_tokens = True
        text = AggregatedTextFrame("private", "sentence")
        text.context_id, text.will_be_spoken = "ctx", True
        await self.push(text)
        await self.push(text, output=True, first=False)
        self.assertFalse(self.observer.pending[-1]["sentence_aligned"])

    async def test_bounded_queue_does_not_send_inside_audio_callback(self):
        self.observer.max_pending = 1
        await self.push(ErrorFrame("first"))
        await self.push(ErrorFrame("second"))
        self.emit.assert_not_awaited()
        await self.observer.flush()
        self.assertEqual(self.emit.call_args.args[0]["dropped_events"], 1)

    async def test_native_observer_payload_and_client_audio_classify_after_delivery(self):
        text = AggregatedTextFrame("hello", "sentence")
        text.context_id, text.will_be_spoken = "ctx", True
        start = TTSStartedFrame(context_id="ctx")
        await self.push(text)
        await self.push(start)
        await self.push(text, output=True, first=False)
        await self.push(start, output=True, first=False)
        collector = collected_gap(metadata=False)
        for _ in range(2):
            audio = OutputAudioRawFrame(audio=b"\x00" * 800, sample_rate=1000, num_channels=1)
            await self.push(audio)
            await self.push(audio, output=True, first=False)
        await self.push(TTSStoppedFrame(context_id="ctx"), output=True)
        await self.observer.flush()
        collector.ingest(self.emit.call_args.args[0])
        self.assertEqual(collector.finish()["glitch_events"][0]["classification"], "within_sentence")

    async def test_structured_errors_redact_credentials(self):
        await self.push(ErrorFrame("Authorization: Bearer dummy-test-secret"))
        await self.observer.flush()
        encoded = json.dumps(self.emit.call_args.args[0])
        self.assertNotIn("dummy-test-secret", encoded)
        self.assertIn("[REDACTED]", encoded)


class StartupDiagnosticTests(unittest.TestCase):
    def test_readiness_failure_has_structured_reason_when_opted_in(self):
        import os

        from fastapi.testclient import TestClient
        from starlette.testclient import WebSocketDenialResponse

        import server

        with (
            patch.dict(os.environ, {"SCALING_PERF_DIAGNOSTICS": "true"}),
            patch.object(
                server,
                "_ensure_services_ready_for_connection",
                AsyncMock(side_effect=RuntimeError("LLM service unavailable")),
            ),
            TestClient(server.create_app()) as client,
            self.assertRaises(WebSocketDenialResponse) as rejected,
            client.websocket_connect("/api/ws"),
        ):
            pass
        self.assertEqual(rejected.exception.status_code, 503)
        self.assertEqual(rejected.exception.json()["error"]["component"], "llm")
        self.assertEqual(rejected.exception.json()["error"]["code"], "service_not_ready")

    def test_default_readiness_rejection_keeps_existing_close_behavior(self):
        import os

        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        import server

        with (
            patch.dict(os.environ, {"SCALING_PERF_DIAGNOSTICS": "false"}),
            patch.object(
                server,
                "_ensure_services_ready_for_connection",
                AsyncMock(side_effect=RuntimeError("LLM service unavailable")),
            ),
            TestClient(server.create_app()) as client,
            self.assertRaises(WebSocketDisconnect) as rejected,
            client.websocket_connect("/api/ws"),
        ):
            pass
        self.assertEqual(rejected.exception.code, 1011)


class ProtocolDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    async def test_realtime_metadata_survives_per_turn_event_reset(self):
        from realtime_transport import RealtimeTransport

        collector = SessionDiagnostics()
        socket = RealtimeTransport("wss://example.test", on_event=collector.ingest)
        socket.ws = SimpleNamespace(recv=AsyncMock(return_value=json.dumps(batch([{"kind": "llm_start"}]))))
        await socket.recv_event()
        socket.events.clear()
        self.assertEqual(collector.events[0]["kind"], "llm_start")

    async def test_denied_admission_reason_reaches_client_error_collection(self):
        from rtvi_client import RTVIClient
        from rtvi_transport import RTVIStartupError

        collector = SessionDiagnostics()
        client = RTVIClient(
            stream_id="test", host="test", port=443, connect_timeout=1, logger=SimpleNamespace(log=AsyncMock())
        )
        client.diagnostics = collector
        client.transport.__aenter__ = AsyncMock(
            side_effect=RTVIStartupError(
                {"message": "LLM unavailable", "component": "llm", "code": "service_not_ready"}
            )
        )
        with self.assertRaises(RTVIStartupError):
            await client.__aenter__()
        self.assertEqual(collector.errors[0]["source"], "handshake")
        self.assertEqual(collector.errors[0]["component"], "llm")

    async def test_transport_decodes_structured_http_denial(self):
        from rtvi_transport import RTVIStartupError, RTVITransport
        from websockets.datastructures import Headers
        from websockets.exceptions import InvalidStatus
        from websockets.http11 import Response

        body = json.dumps({"error": {"message": "LLM unavailable", "component": "llm"}}).encode()
        transport = RTVITransport(stream_id="test", host="test", port=443, connect_timeout=1)
        with (
            patch(
                "rtvi_transport.websockets.connect",
                AsyncMock(side_effect=InvalidStatus(Response(503, "Service Unavailable", Headers(), body))),
            ),
            self.assertRaises(RTVIStartupError) as rejected,
        ):
            await transport.__aenter__()
        self.assertEqual(rejected.exception.detail["component"], "llm")
