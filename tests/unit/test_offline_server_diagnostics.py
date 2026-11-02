# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D102

"""Scalar observation must be private, concurrent, bounded, and cancellation-safe."""

import asyncio
import hashlib
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from test_offline_websocket import FakeSocket, completion_message, start_message

from examples.generic.offline.detection import SampleTimedSileroMixin
from examples.generic.offline.diagnostics import (
    ServerDiagnostics,
    bind_diagnostics,
    create_diagnostics,
    current_diagnostics,
    duration,
    record,
    reset_diagnostics,
)
from examples.generic.offline.models import SessionResult
from examples.generic.offline.protocol import SessionLimits, encode_packet
from examples.generic.offline.websocket import _LiveEventWriter, serve_offline_session


class ScalarDiagnosticsTests(unittest.TestCase):
    """Do not turn an observer into a provider or a private transcript recorder."""

    def test_bounded_concurrent_records_and_window_aggregates(self):
        sink = ServerDiagnostics(max_events=32)

        def worker():
            for _ in range(100):
                sink.record("callback", source="asr", duration_ms=0.5)
                sink.observe_duration("vad_await", 0.5)

        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda _: worker(), range(6)))
        value = sink.snapshot()
        self.assertEqual(value["total_events"], 600)
        self.assertEqual(len(value["events"]), 32)
        self.assertTrue(value["truncated"])
        self.assertEqual(value["duration_summaries"]["vad_await"], {"count": 600, "sum_ms": 300, "max_ms": 0.5})
        wall = [event["wall_elapsed_ms"] for event in value["events"]]
        self.assertEqual(wall, sorted(wall))

    def test_private_payload_and_nonfinite_numbers_are_never_retained(self):
        sink = ServerDiagnostics()
        sink.record(
            "native_metric",
            source="llm",
            boundary="llm",
            response_id="response-1",
            value_ms=12.5,
            text="private text",
            authorization="private-key",
            request={"messages": "private body"},
            invalid=float("inf"),
        )
        value = sink.snapshot()
        event = value["events"][0]
        self.assertEqual(event["boundary"], "llm")
        self.assertEqual(event["value_ms"], 12.5)
        self.assertEqual(value["rejected_fields"], 4)
        self.assertNotIn("private", json.dumps(event))
        event["value_ms"] = 99
        self.assertEqual(sink.snapshot()["events"][0]["value_ms"], 12.5)

    def test_observer_failures_do_not_change_provider_results(self):
        class Broken:
            def record(self, *args, **kwargs):
                raise RuntimeError("observer failed")

            def observe_duration(self, *args, **kwargs):
                raise RuntimeError("observer failed")

        record(Broken(), "callback", duration_ms=1)
        duration(Broken(), "vad_await", 1)

    def test_only_trusted_environment_enables_persistence(self):
        with patch.dict(os.environ, {"NVA_OFFLINE_SERVER_METRICS_DIR": ""}):
            self.assertEqual(create_diagnostics(), (None, None))
        self.assertIsNone(current_diagnostics())
        sink = ServerDiagnostics()
        token = bind_diagnostics(sink)
        try:
            self.assertIs(current_diagnostics(), sink)
        finally:
            reset_diagnostics(token)
        self.assertIsNone(current_diagnostics())


class DiagnosticSessionTests(unittest.IsolatedAsyncioTestCase):
    """Observe request boundaries without replacing ASGI or cancellation policy."""

    async def test_real_sealed_text_session_persists_source_and_wire_session_identity(self):
        async def run(source, *, event_sink):
            return SessionResult((), source.sample_count, (), (), {})

        socket = FakeSocket()
        pcm = b"\0\0" * 512
        socket.feed(start_message(pcm, completion="text"))
        socket.feed(encode_packet(b"NVAI", 0, 0, pcm))
        socket.feed(completion_message(pcm, 1))
        runtime = SimpleNamespace(configuration={"model": "fixture"}, run=run)
        with tempfile.TemporaryDirectory() as root:
            metrics_root = Path(root) / "metrics"
            with patch.dict(os.environ, {"NVA_OFFLINE_SERVER_METRICS_DIR": str(metrics_root)}):
                await serve_offline_session(socket, lambda start: runtime, Path(root) / "audio")
            self.assertFalse(any(item.get("type") == "offline.session.error" for item in socket.sent))
            self.assertTrue(any(item.get("type") == "offline.events.complete" for item in socket.sent))
            config = next(item for item in socket.sent if item.get("type") == "offline.runtime.configuration")
            evidence = json.loads((metrics_root / (config["server_metrics_session_id"] + ".json")).read_text())
            sealed = next(event for event in evidence["events"] if event["name"] == "source_sealed")
            self.assertEqual(sealed["source_sha256"], hashlib.sha256(pcm).hexdigest())
            self.assertEqual(sealed["input_samples"], 512)
            self.assertEqual(evidence["session_id"], config["server_metrics_session_id"])
        self.assertIsNone(current_diagnostics())

    async def drain_writer(self, writer, payloads):
        task = asyncio.create_task(writer.run())
        try:
            for payload in payloads:
                await writer.queue.put((payload, 0))
            await asyncio.wait_for(writer.queue.join(), 5)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_disabled_writer_does_not_inspect_or_track_text_for_metrics(self):
        class UninspectedText(str):
            def strip(self, *args):
                raise AssertionError("Disabled metrics must not inspect text")

        socket = SimpleNamespace(send_json=AsyncMock())
        writer = _LiveEventWriter(socket, SessionLimits())
        payload = {
            "event": {
                "type": "provider_observation",
                "source": "llm",
                "frame": "LLMTextFrame",
                "response_id": "response-1",
                "text": UninspectedText("synthetic"),
            }
        }
        await self.drain_writer(writer, [payload])
        socket.send_json.assert_awaited_once_with(payload)
        self.assertEqual(writer._first_text_sent, set())

    async def test_writer_first_text_tracking_is_bounded_after_metrics_capacity(self):
        sink = ServerDiagnostics(max_events=1)
        token = bind_diagnostics(sink)
        try:
            writer = _LiveEventWriter(SimpleNamespace(send_json=AsyncMock()), SessionLimits())
        finally:
            reset_diagnostics(token)
        payloads = [
            {
                "event": {
                    "type": "provider_observation",
                    "source": "llm",
                    "frame": "LLMTextFrame",
                    "response_id": f"response-{index}",
                    "text": "synthetic",
                }
            }
            for index in range(4100)
        ]
        await self.drain_writer(writer, payloads)
        self.assertEqual(len(writer._first_text_sent), 4096)
        self.assertEqual(len(sink.snapshot()["events"]), 1)
        self.assertTrue(sink.snapshot()["truncated"])

    async def test_full_native_vad_executor_timing_preserves_native_states(self):
        class ObservedVAD(SampleTimedSileroMixin, SileroVADAnalyzer):
            pass

        # Only the ONNX model boundary is synthetic; both analyzers retain the
        # actual pinned native executor, volume tracking and state algorithm.
        sink = ServerDiagnostics()
        model = Mock(return_value=(0.9,))
        token = bind_diagnostics(sink)
        try:
            with patch("pipecat.audio.vad.silero.SileroOnnxModel", return_value=model):
                observed = ObservedVAD(sample_rate=16000, params=VADParams(min_volume=0))
                plain = SileroVADAnalyzer(sample_rate=16000, params=VADParams(min_volume=0))
        finally:
            reset_diagnostics(token)
        observed.set_sample_rate(16000)
        plain.set_sample_rate(16000)
        pcm = b"\x10\x20" * 512
        try:
            for index in range(3):
                observed.set_source_window_end((index + 1) * 512)
                self.assertEqual(await observed.analyze_audio(pcm), await plain.analyze_audio(pcm))
        finally:
            await observed.cleanup()
            await plain.cleanup()
        values = sink.snapshot()["duration_summaries"]
        self.assertEqual(values["vad_model"]["count"], 3)
        self.assertEqual(values["vad_executor"]["count"], 3)
        self.assertGreaterEqual(values["vad_executor"]["sum_ms"], values["vad_model"]["sum_ms"])

    async def test_cancelled_session_preserves_scalar_evidence_and_propagates_cancellation(self):
        async def cancelled(*args, **kwargs):
            sink = current_diagnostics()
            self.assertIsNotNone(sink)
            record(sink, "source_sealed", source_sha256="a" * 64, input_samples=24000)
            raise asyncio.CancelledError

        with tempfile.TemporaryDirectory() as root:
            with (
                patch.dict(os.environ, {"NVA_OFFLINE_SERVER_METRICS_DIR": root}),
                patch("examples.generic.offline.websocket._serve_offline_session", cancelled),
                self.assertRaises(asyncio.CancelledError),
            ):
                await serve_offline_session(None, None, root)
            paths = list(Path(root).glob("*.json"))
            self.assertEqual(len(paths), 1)
            value = json.loads(paths[0].read_text())
            self.assertEqual(
                [event["name"] for event in value["events"]],
                ["session_started", "source_sealed", "session_error", "session_returned"],
            )
            self.assertEqual(paths[0].stat().st_mode & 0o777, 0o600)
            self.assertFalse(value["full_provider_eof_or_cleanup_attested"])
        self.assertIsNone(current_diagnostics())

    async def test_request_context_does_not_leak_between_concurrent_sessions(self):
        ready = asyncio.Event()
        identifiers = set()

        async def normal(*args, **kwargs):
            sink = current_diagnostics()
            identifiers.add(sink.session_id)
            if len(identifiers) == 2:
                ready.set()
            await ready.wait()
            await asyncio.to_thread(record, sink, "worker_observed", source="asr")

        with tempfile.TemporaryDirectory() as root:
            with (
                patch.dict(os.environ, {"NVA_OFFLINE_SERVER_METRICS_DIR": root}),
                patch("examples.generic.offline.websocket._serve_offline_session", normal),
            ):
                await asyncio.gather(serve_offline_session(None, None, root), serve_offline_session(None, None, root))
            files = list(Path(root).glob("*.json"))
            self.assertEqual(len(files), 2)
            self.assertEqual({json.loads(path.read_text())["session_id"] for path in files}, identifiers)
        self.assertIsNone(current_diagnostics())
