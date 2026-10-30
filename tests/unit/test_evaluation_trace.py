# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify evaluation observations preserve the live services and frame flow."""

import asyncio
import hashlib
import itertools
import json
import os
import stat
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import riva.client
from pipecat.frames.frames import (
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    OutputAudioRawFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSTextFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.nvidia.stt import NvidiaSTTService
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings
from pipecat.transports.base_output import BaseOutputTransport
from riva.client.proto import riva_asr_pb2 as rasr

from examples.shared.evaluation_trace import (
    EvaluationNvidiaSTTService,
    EvaluationNvidiaTTSService,
    EvaluationTrace,
    create_evaluation_trace,
)


class EvaluationTraceTests(unittest.IsolatedAsyncioTestCase):
    """Exercise source clocks, metric scopes, and thread-safe bounded persistence."""

    def setUp(self):
        """Allocate a private directory and deterministic monotonic clock."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.clock = itertools.count(0, 1_000_000).__next__
        self.trace = EvaluationTrace(Path(self.directory.name), clock=self.clock)

    async def push(self, frame, *, source=None, direction=FrameDirection.DOWNSTREAM):
        """Simulate the legitimate observer API without forwarding any frame."""
        await self.trace.on_push_frame(
            FramePushed(
                source=source or MagicMock(name="source"),
                destination=MagicMock(),
                frame=frame,
                direction=direction,
                timestamp=123,
            )
        )

    def saved(self):
        """Read the finalized document through its public output artifact."""
        self.trace.close()
        return json.loads(self.trace.path.read_text())

    async def test_inline_transcription_flags_are_exact_immutable_and_do_not_retain_private_metadata(self):
        """Keep actual true/false flags without altering frames or retaining raw ASR results."""
        owner = SimpleNamespace(name="native-user")
        private_result = object()
        frames = [
            TranscriptionFrame("first", "private-user", "private-timestamp", result=private_result, finalized=False),
            TranscriptionFrame("second", "private-user", "private-timestamp", result=private_result, finalized=True),
            InterimTranscriptionFrame("interim", "private-user", "private-timestamp", result=private_result),
        ]
        for index, frame in enumerate(frames):
            frame.includes_inter_frame_spaces = index != 0
            before = dict(vars(frame))
            self.trace.processor_boundary("source_delivery", owner, frame, role="asr", logical_sample=512)
            self.trace.processor_boundary("before_process", owner, frame, role="user", logical_sample=512)
            self.trace.processor_boundary("after_process", owner, frame, role="user", logical_sample=512)
            self.assertEqual(vars(frame), before)
            self.assertIs(frame.result, private_result)
        # Later mutations cannot rewrite what an earlier inline hook saw.
        frames[0].finalized = True
        frames[1].finalized = False
        for frame in frames:
            frame.includes_inter_frame_spaces = False
        document = self.saved()
        events = [event for event in document["events"] if event["event"] == "processor_boundary"]
        for index, original in enumerate(frames):
            snapshots = [event for event in events if event["frame_id"] == original.id]
            self.assertEqual(len(snapshots), 3)
            self.assertEqual([event["logical_sample"] for event in snapshots], [512, 512, 512])
            self.assertTrue(all(event["includes_inter_frame_spaces"] is (index != 0) for event in snapshots))
            if index < 2:
                self.assertTrue(all(event["finalized"] is (index == 1) for event in snapshots))
            else:
                self.assertTrue(all("finalized" not in event for event in snapshots))
        self.assertNotIn("private-user", json.dumps(document))
        self.assertNotIn("private-timestamp", json.dumps(document))
        self.assertTrue(all("result" not in event and "user_id" not in event for event in events))
        self.assertEqual(document["observation_errors"], 0)

    async def test_inline_stt_metadata_retains_nondefault_finite_policy_value_without_service_name(self):
        """Copy configured TTFS metadata at publication and processing, without changing it."""
        owner = SimpleNamespace(name="native-stt")
        frame = STTMetadataFrame("private.endpoint.invalid:443", 2.75)
        before = dict(vars(frame))
        self.trace.processor_boundary("before_push", owner, frame, role="asr")
        self.trace.processor_boundary("before_process", owner, frame, role="user", logical_sample=0)
        self.trace.processor_boundary("after_process", owner, frame, role="user", logical_sample=0)
        self.assertEqual(vars(frame), before)
        frame.ttfs_p99_latency = 9.0
        document = self.saved()
        events = [event for event in document["events"] if event["event"] == "processor_boundary"]
        self.assertEqual(len(events), 3)
        for event in events:
            self.assertEqual(event["frame_type"], "STTMetadataFrame")
            self.assertEqual(event["stt_ttfs_p99_latency_s"], 2.75)
            self.assertEqual(event["stt_ttfs_p99_latency_ms"], 2750.0)
            self.assertIs(event["stt_ttfs_p99_latency_finite"], True)
            self.assertNotIn("service_name", event)
        self.assertNotIn("private.endpoint.invalid", json.dumps(document))
        self.assertEqual(document["observation_errors"], 0)

    async def test_inline_nonfinite_stt_values_and_nonboolean_flags_remain_unobserved_null(self):
        """Do not invent policy defaults or truth values for unsupported metadata."""
        owner = SimpleNamespace(name="native-stt")
        for value in (float("nan"), float("inf"), float("-inf"), True):
            with self.subTest(value_type=type(value).__name__):
                frame = STTMetadataFrame("NvidiaSTTService", value)
                self.trace.processor_boundary("before_push", owner, frame, role="asr")
        frame = TranscriptionFrame("text", "user", "", finalized=None)
        frame.includes_inter_frame_spaces = "unknown"
        self.trace.processor_boundary("before_process", owner, frame, role="user")
        document = self.saved()
        events = [event for event in document["events"] if event["event"] == "processor_boundary"]
        for event in events[:-1]:
            self.assertIsNone(event["stt_ttfs_p99_latency_s"])
            self.assertIsNone(event["stt_ttfs_p99_latency_ms"])
            self.assertIs(event["stt_ttfs_p99_latency_finite"], False)
        self.assertIsNone(events[-1]["finalized"])
        self.assertIsNone(events[-1]["includes_inter_frame_spaces"])
        # Null diagnostic values keep JSON strict even when an original native
        # field is NaN. The observer never edits that original frame.
        json.dumps(document, allow_nan=False)
        self.assertEqual(document["observation_errors"], 0)

    async def test_counts_input_once_and_retains_backdating_parameters(self):
        """Arrival cursors count PCM, while VAD silence remains explicit metadata."""
        frame = InputAudioRawFrame(audio=b"\0" * 1_024, sample_rate=16000, num_channels=1)
        await self.push(frame)
        await self.push(frame)
        await self.push(VADUserStoppedSpeakingFrame(stop_secs=0.2))
        events = self.saved()["events"]
        inputs = [event for event in events if event["event"] == "input_audio_received"]
        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0]["input_start_sample"], 0)
        self.assertEqual(inputs[0]["input_end_sample"], 512)
        stop = next(event for event in events if event["event"] == "VADUserStoppedSpeakingFrame")
        self.assertEqual(stop["stop_secs"], 0.2)
        self.assertEqual(stop["input_end_sample"], 512)
        self.assertEqual(stop["elapsed_ms"], stop["elapsed_ns"] / 1e6)

    async def test_retains_first_nonempty_text_and_metrics_without_unit_confusion(self):
        """Role-only/empty text is excluded; original seconds and converted ms coexist."""
        await self.push(LLMFullResponseStartFrame())
        await self.push(LLMTextFrame(text=""))
        await self.push(LLMTextFrame(text=" "))
        await self.push(LLMTextFrame(text="Hello"))
        await self.push(LLMFullResponseEndFrame())
        metric = MetricsFrame(data=[TTFBMetricsData(processor="NvidiaLLMService", value=0.0125)])
        await self.push(metric)
        await self.push(metric)
        document = self.saved()
        text = [event for event in document["events"] if event["event"] == "llm_text"]
        self.assertEqual([event["text"] for event in text], ["Hello"])
        self.assertEqual(text[0]["llm_request_index"], 1)
        metrics = [event for event in document["events"] if event["event"] == "pipecat_metrics"]
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["metrics"][0]["value"], 0.0125)
        self.assertEqual(metrics[0]["metrics"][0]["value_ms"], 12.5)
        self.assertIn("inferred user speech end", document["timing_scopes"]["asr_ttfb"])

    async def test_post_write_audio_is_separate_from_first_frame_observation(self):
        """Observe the transport's second push once without losing its sent edge."""
        frame = OutputAudioRawFrame(audio=b"\0" * 320, sample_rate=16000, num_channels=1)
        await self.push(frame)
        source = MagicMock(spec=BaseOutputTransport)
        await self.push(frame, source=source)
        await self.push(frame, source=source)
        events = [event for event in self.saved()["events"] if event["event"] == "output_audio_sent"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["audio_bytes"], 320)

    async def test_ingested_supplied_and_observer_cursors_remain_independent(self):
        """Buffered input is not evidence that Riva consumed or processed that prefix."""
        first = InputAudioRawFrame(audio=b"\x01\x02" * 512, sample_rate=16000, num_channels=1)
        second = InputAudioRawFrame(audio=b"\x03\x04" * 512, sample_rate=16000, num_channels=1)
        await self.push(first)
        await self.push(second)
        self.trace.asr_input(first)
        self.trace.asr_input(first)
        self.trace.asr_audio(first.audio[:512])
        document = self.saved()
        ingested = [event for event in document["events"] if event["event"] == "asr_input_ingested"]
        supplied = next(event for event in document["events"] if event["event"] == "asr_request_supplied")
        self.assertEqual(len(ingested), 1)
        self.assertEqual(first.metadata["nva_eval_source"], {"seq": 0, "start": 0, "end": 512, "rate": 16000})
        self.assertEqual(ingested[0]["input_end_sample"], 1024)
        self.assertEqual(supplied["asr_supplied_end_sample_16k"], 256)
        self.assertEqual(supplied["audio_sha256"], hashlib.sha256(first.audio[:512]).hexdigest())
        self.assertEqual(document["asr_supplied_audio"]["samples_16k"], 256)
        self.assertEqual(document["asr_supplied_audio"]["sha256"], supplied["audio_sha256"])

    async def test_context_is_captured_once_at_llm_edge_without_losing_original_clock(self):
        """Earlier edges must not hide the request snapshot; later mutation cannot rewrite it."""
        context = LLMContext(messages=[{"role": "user", "content": "Original question"}])
        frame = LLMContextFrame(context=context)
        await self.push(frame)
        llm = MagicMock(spec=LLMService)
        for timestamp in (10_000_000, 20_000_000):
            await self.trace.on_push_frame(
                FramePushed(
                    source=MagicMock(),
                    destination=llm,
                    frame=frame,
                    direction=FrameDirection.DOWNSTREAM,
                    timestamp=timestamp,
                )
            )
        context.add_message({"role": "assistant", "content": "Later reply"})
        document = self.saved()
        requests = [event for event in document["events"] if event["event"] == "llm_context_request_edge"]
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["messages"], [{"role": "user", "content": "Original question"}])
        self.assertEqual(requests[0]["pipeline_timestamp_ns"], 10_000_000)
        self.assertEqual(requests[0]["context_version"], 1)
        self.assertIsNone(requests[0]["llm_request_index"])
        self.assertIn("mutable context may have changed", document["timing_scopes"]["llm_context_request_edge"])

    async def test_tts_text_first_observation_and_native_output_delivery_are_separate(self):
        """Queued sentence text must not be mistaken for LLM tokens or output delivery."""
        await self.push(LLMTextFrame(text="Hello"))
        frame = TTSTextFrame(text="Hello there.", aggregated_by="sentence", context_id="context-one")
        await self.push(frame)
        output = MagicMock(spec=BaseOutputTransport)
        await self.push(frame, source=output)
        await self.push(frame, source=output)
        events = self.saved()["events"]
        self.assertEqual([event["text"] for event in events if event["event"] == "llm_text"], ["Hello"])
        text = [event for event in events if event["event"] == "tts_text_frame"]
        delivered = [event for event in events if event["event"] == "tts_text_output_delivered"]
        self.assertEqual(len(text), 1)
        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0]["frame_id"], text[0]["frame_id"])
        self.assertGreater(delivered[0]["sequence"], text[0]["sequence"])

    def test_processed_audio_metadata_is_optional_and_does_not_invent_source_ownership(self):
        """Missing or malformed provider counters are omitted, preserving transcript order."""
        results = [
            SimpleNamespace(is_final=True, alternatives=[SimpleNamespace(transcript="known")], audio_processed=1.25),
            SimpleNamespace(is_final=False, alternatives=[SimpleNamespace(transcript="unknown")]),
        ]
        for value in (None, "unknown", float("nan"), -1, True):
            results.append(SimpleNamespace(is_final=False, alternatives=[], audio_processed=value))
        self.trace.asr_response(SimpleNamespace(results=results))
        observed = next(event for event in self.saved()["events"] if event["event"] == "asr_provider_result")["results"]
        self.assertEqual(observed[0]["audio_processed_seconds"], 1.25)
        self.assertEqual(observed[0]["processed_sample_16k"], 20000)
        self.assertEqual(observed[1]["transcript"], "unknown")
        self.assertTrue(all("processed_sample_16k" not in item for item in observed[1:]))
        self.assertTrue(all("dependency_chunk_end" not in item for item in observed))

    def test_source_metadata_preserves_existing_values_and_cumulative_rate_conversion(self):
        """Observation cannot overwrite metadata or accumulate per-chunk rounding drift."""
        frames = [InputAudioRawFrame(audio=b"\0\0" * 7, sample_rate=44100, num_channels=1) for _ in range(10)]
        frames[0].metadata["nva_eval_source"] = {"owner": "existing"}
        for frame in frames:
            self.trace.asr_input(frame)
            self.trace.asr_audio(frame.audio, sample_rate=44100)
        document = self.saved()
        ingested = [event for event in document["events"] if event["event"] == "asr_input_ingested"]
        self.assertEqual(frames[0].metadata["nva_eval_source"], {"owner": "existing"})
        self.assertFalse(ingested[0]["source_tag_written"])
        self.assertEqual(ingested[-1]["source_tag"]["end"], round(70 * 16000 / 44100))
        self.assertEqual(document["asr_supplied_audio"]["samples_16k"], round(70 * 16000 / 44100))

    async def test_interruption_is_observed_upstream_once(self):
        """Retain barge-in even when its first edge travels upstream."""
        frame = InterruptionFrame()
        await self.push(frame, direction=FrameDirection.UPSTREAM)
        await self.push(frame)
        events = [event for event in self.saved()["events"] if event["event"] == "InterruptionFrame"]
        self.assertEqual(len(events), 1)

    async def test_optional_or_invalid_vad_timestamps_remain_missing(self):
        """Optional metadata must not stop observation of later valid frames."""
        for timestamp in (None, "unknown", float("nan"), True):
            await self.push(VADUserStoppedSpeakingFrame(timestamp=timestamp))
        await self.push(LLMFullResponseStartFrame())
        document = self.saved()
        stops = [event for event in document["events"] if event["event"] == "VADUserStoppedSpeakingFrame"]
        self.assertEqual(len(stops), 4)
        self.assertTrue(all(event["frame_timestamp_unix_ms"] is None for event in stops))
        self.assertTrue(any(event["event"] == "LLMFullResponseStartFrame" for event in document["events"]))
        self.assertEqual(document["observation_errors"], 0)

    async def test_invalid_metric_does_not_discard_valid_measurements_or_observer(self):
        """Skip malformed custom metrics while retaining missing and valid numeric values."""
        await self.push(
            MetricsFrame(
                data=[
                    object(),
                    TTFBMetricsData.model_construct(processor="custom", value=None),
                    TTFBMetricsData(processor="NvidiaLLMService", value=0.03),
                ]
            )
        )
        await self.push(LLMFullResponseStartFrame())
        document = self.saved()
        metric = next(event for event in document["events"] if event["event"] == "pipecat_metrics")
        self.assertEqual([value["value_ms"] for value in metric["metrics"]], [None, 30])
        self.assertEqual(document["observation_errors"], 1)
        self.assertTrue(any(event["event"] == "LLMFullResponseStartFrame" for event in document["events"]))

    async def test_observer_callback_failure_cannot_escape_or_stop_later_observations(self):
        """An unexpected optional trace failure is contained inside the callback."""
        with patch.object(self.trace, "_observe_frame", side_effect=RuntimeError("trace failure")):
            await self.push(LLMTextFrame(text="hello"))
        await self.push(LLMFullResponseStartFrame())
        document = self.saved()
        self.assertEqual(document["observation_errors"], 1)
        self.assertTrue(any(event["event"] == "LLMFullResponseStartFrame" for event in document["events"]))

    async def test_original_pipeline_timestamps_survive_observer_queue_delay(self):
        """Retain the original push clock needed for frame-boundary LLM durations."""
        for frame, timestamp in (
            (LLMFullResponseStartFrame(), 100_000_000),
            (LLMTextFrame(text="hello"), 130_000_000),
            (LLMFullResponseEndFrame(), 200_000_000),
        ):
            await self.trace.on_push_frame(
                FramePushed(
                    source=MagicMock(),
                    destination=MagicMock(),
                    frame=frame,
                    direction=FrameDirection.DOWNSTREAM,
                    timestamp=timestamp,
                )
            )
        document = self.saved()
        frames = [event for event in document["events"] if "pipeline_timestamp_ns" in event]
        start, first_text, end = [event["pipeline_timestamp_ns"] for event in frames]
        self.assertEqual((first_text - start) / 1e6, 30)
        self.assertEqual((end - start) / 1e6, 100)
        self.assertNotEqual(frames[1]["elapsed_ns"] - frames[0]["elapsed_ns"], first_text - start)
        self.assertIn("Use pipeline_timestamp_ns differences", document["timing_scopes"]["llm_frames"])

    async def test_late_pcm_preserves_its_context_request_association(self):
        """A newer LLM request cannot relabel the previous response's provider events."""
        await self.push(LLMFullResponseStartFrame())
        self.trace.tts_text("context-one", "hello", "stitched")
        await self.push(LLMFullResponseStartFrame())
        frame = TTSAudioRawFrame(audio=b"\0\0", sample_rate=16000, num_channels=1)
        self.trace.tts_pcm("context-one", frame)
        self.trace.tts_pcm("context-one", frame)
        self.trace.record("tts_worker_finished", context_id="context-one")
        events = self.saved()["events"]
        pcm = [event for event in events if event["event"] == "tts_pcm_ready"]
        self.assertEqual(len(pcm), 1)
        self.assertEqual(pcm[0]["llm_request_index"], 1)
        worker = next(event for event in events if event["event"] == "tts_worker_finished")
        self.assertEqual(worker["llm_request_index"], 1)

    def test_default_is_disabled_and_client_cannot_choose_trace_path(self):
        """Require the server environment rather than an untrusted session field."""
        with patch.dict("os.environ", {"NVA_EVAL_TRACE_DIR": ""}):
            self.assertIsNone(create_evaluation_trace({"NVA_EVAL_TRACE_DIR": "/tmp/client-choice"}))
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_only_safe_configuration_fields_are_persisted(self):
        """Correlate Realtime session IDs without logging endpoint keys or credentials."""
        with patch.dict("os.environ", {"NVA_EVAL_TRACE_DIR": self.directory.name}):
            trace = create_evaluation_trace(
                {
                    "protocol": "realtime",
                    "realtime_session_view": {"id": "sess-known"},
                    "api_key": "do-not-store",
                    "base_url": "private-endpoint",
                }
            )
        trace.close()
        document = trace.path.read_text()
        self.assertIn("sess-known", document)
        self.assertNotIn("do-not-store", document)
        self.assertNotIn("private-endpoint", document)

    def test_close_is_idempotent_and_bounds_threaded_events(self):
        """Bound memory and keep a valid trace even when provider threads emit events."""
        trace = EvaluationTrace(Path(self.directory.name), clock=self.clock, max_events=10)
        threads = [threading.Thread(target=lambda: [trace.record("provider") for _ in range(10)]) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        trace.close()
        first = trace.path.read_bytes()
        trace.record("too-late")
        trace.close()
        self.assertEqual(trace.path.read_bytes(), first)
        document = json.loads(first)
        self.assertEqual(len(document["events"]), 10)
        self.assertGreater(document["dropped_events"], 0)
        self.assertEqual([event["sequence"] for event in document["events"]], list(range(10)))

    async def test_context_bookkeeping_stops_at_event_cap_without_relabeling_captured_requests(self):
        """Keep captured joins and bounded metadata when later synthesis keeps running."""
        trace = EvaluationTrace(Path(self.directory.name), clock=self.clock, max_events=5)
        trace._llm_request_index = 1
        trace.tts_text("retained", "hello", "stitched")
        trace.tts_pcm("retained", TTSAudioRawFrame(audio=b"\0\0", sample_rate=16000, num_channels=1))
        trace.record("tts_worker_finished", context_id="retained")
        trace.record("fill_cap")
        self.assertEqual(len(trace._events), 5)
        contexts = dict(trace._contexts)
        threads = [
            threading.Thread(
                target=lambda prefix=prefix: [trace.tts_text(f"{prefix}-{n}", "later", "stitched") for n in range(50)]
            )
            for prefix in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(trace._contexts, contexts)
        self.assertEqual(trace._dropped, 200)
        trace.close()
        document = json.loads(trace.path.read_text())
        self.assertEqual(len(document["events"]), 5)
        captured = [event for event in document["events"] if event.get("context_id") == "retained"]
        self.assertEqual(len(captured), 3)
        self.assertTrue(all(event["llm_request_index"] == 1 for event in captured))
        trace.tts_text("after-close", "later", "stitched")
        self.assertEqual(trace._contexts, contexts)

    def test_unwritable_optional_trace_does_not_raise(self):
        """A tracing failure must not turn a successful conversation into a failure."""
        with patch("examples.shared.evaluation_trace.os.open", side_effect=OSError("disk full")):
            self.trace.close()

    def test_request_trace_file_is_private_even_with_permissive_umask(self):
        """Opt-in conversational request snapshots are not world readable."""
        previous = os.umask(0)
        try:
            self.trace.close()
        finally:
            os.umask(previous)
        self.assertEqual(stat.S_IMODE(self.trace.path.stat().st_mode), 0o600)


class EvaluationServiceTests(unittest.IsolatedAsyncioTestCase):
    """Verify provider wrappers delegate the same data, results, and cancellation."""

    def setUp(self):
        """Allocate a session trace without connecting to any provider."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.trace = EvaluationTrace(Path(self.directory.name))

    def events(self):
        """Read recorded events after provider work has stopped."""
        self.trace.close()
        return json.loads(self.trace.path.read_text())["events"]

    def asr_client(self, service, *, response=()):
        """Initialize the real Riva serializer with a non-networking capture stub."""
        requests_seen = []

        def streaming(requests, **kwargs):
            requests_seen.extend(requests)
            return iter(response)

        client = object.__new__(riva.client.ASRService)
        client.auth = SimpleNamespace(get_auth_metadata=lambda: [])
        client.stub = SimpleNamespace(StreamingRecognize=streaming)

        def initialize(instance):
            instance._asr_service = client

        with patch.object(NvidiaSTTService, "_initialize_client", initialize):
            service._initialize_client()
        return requests_seen

    async def test_asr_requests_remain_byte_identical_to_standard_stt_without_force_eou(self):
        """Keep the default 320 endpoint and serialize only original source PCM."""
        service = EvaluationNvidiaSTTService(evaluation_trace=self.trace, use_ssl=False)
        service._sample_rate = 16000
        config = service._create_recognition_config()
        self.assertEqual(config.config.endpointing_config.stop_history, 320)
        chunks = [b"speech", b"\0" * 1024]
        requests = self.asr_client(service)
        list(service._asr_service.streaming_response_generator(iter(chunks), config))
        expected = list(riva.client.asr.streaming_request_generator(iter(chunks), config))
        self.assertEqual(
            [request.SerializeToString() for request in requests],
            [request.SerializeToString() for request in expected],
        )
        self.assertEqual([request.audio_content for request in requests[1:]], chunks)
        self.assertTrue(all(not request.runtime_config for request in requests))
        supplied = [event for event in self.events() if event["event"] == "asr_request_supplied"]
        self.assertEqual([event["audio_bytes"] for event in supplied], [6, 1024])
        self.assertTrue(all(not event["force_eou"] for event in supplied))
        configuration = json.loads(self.trace.path.read_text())["configuration"]
        self.assertEqual(configuration["asr_stop_history_ms"], 320)
        self.assertFalse(configuration["asr_force_eou"])

    async def test_standard_stream_drop_retains_original_iterator_for_reconnect(self):
        """The observation adapter must not break Pipecat's iterator identity guard."""
        service = EvaluationNvidiaSTTService(evaluation_trace=self.trace, use_ssl=False)
        service._sample_rate = 16000
        service._config = service._create_recognition_config()
        self.asr_client(service)

        class Iterator:
            closed = False

            def __iter__(self):
                return iter([b"original"])

        iterator = Iterator()
        service._audio_iterator = iterator
        reconnect_completed = asyncio.Event()

        async def acknowledge_reconnect():
            reconnect_completed.set()

        reconnect = AsyncMock(side_effect=acknowledge_reconnect)
        with (
            patch.object(service, "get_event_loop", return_value=asyncio.get_running_loop()),
            patch.object(service, "_request_reconnect", reconnect),
        ):
            await asyncio.to_thread(service._response_handler, iterator)
            await asyncio.wait_for(reconnect_completed.wait(), timeout=2)
        reconnect.assert_awaited_once_with()

    async def test_asr_response_callback_and_worker_delegate_without_reinterpretation(self):
        """Observe original result objects and preserve the parent's worker return."""
        service = EvaluationNvidiaSTTService(evaluation_trace=self.trace, use_ssl=False)
        response = rasr.StreamingRecognizeResponse(
            results=[
                rasr.StreamingRecognitionResult(
                    is_final=True, alternatives=[rasr.SpeechRecognitionAlternative(transcript="hello")]
                )
            ]
        )
        parent = AsyncMock()
        with patch.object(NvidiaSTTService, "_handle_response", parent):
            await service._handle_response(response)
        parent.assert_awaited_once_with(response)
        iterator = SimpleNamespace(closed=False)
        with patch.object(NvidiaSTTService, "_response_handler", return_value="parent-result") as worker:
            self.assertEqual(service._response_handler(iterator), "parent-result")
        worker.assert_called_once_with(iterator)
        events = self.events()
        self.assertEqual(
            next(event for event in events if event["event"] == "asr_provider_result")["results"][0]["transcript"],
            "hello",
        )
        self.assertFalse(next(event for event in events if event["event"] == "asr_worker_finished")["iterator_closed"])

    async def test_source_ingestion_observation_precedes_unchanged_parent_handling(self):
        """The native parent receives the original frame and its return value survives."""
        service = EvaluationNvidiaSTTService(evaluation_trace=self.trace, use_ssl=False)
        frame = InputAudioRawFrame(audio=b"\x01\x02" * 512, sample_rate=16000, num_channels=1)
        frame.metadata["unrelated"] = "preserved"

        async def parent(frame_arg, direction_arg):
            self.assertIs(frame_arg, frame)
            self.assertEqual(direction_arg, FrameDirection.DOWNSTREAM)
            self.assertEqual(frame.metadata["nva_eval_source"]["end"], 512)
            return "native-result"

        with patch.object(NvidiaSTTService, "process_audio_frame", side_effect=parent) as delegate:
            self.assertEqual(await service.process_audio_frame(frame, FrameDirection.DOWNSTREAM), "native-result")
        delegate.assert_awaited_once_with(frame, FrameDirection.DOWNSTREAM)
        self.assertEqual(frame.audio, b"\x01\x02" * 512)
        self.assertEqual(frame.metadata["unrelated"], "preserved")

    async def test_failed_ingestion_observation_cannot_mask_native_cancellation(self):
        """Trace errors remain optional even at the newly observed source edge."""
        service = EvaluationNvidiaSTTService(evaluation_trace=self.trace, use_ssl=False)
        frame = InputAudioRawFrame(audio=b"\0\0", sample_rate=16000, num_channels=1)
        with (
            patch.object(self.trace, "asr_input", side_effect=RuntimeError("trace failure")),
            patch.object(NvidiaSTTService, "process_audio_frame", AsyncMock(side_effect=asyncio.CancelledError())),
            self.assertRaises(asyncio.CancelledError),
        ):
            await service.process_audio_frame(frame, FrameDirection.DOWNSTREAM)
        self.assertEqual(json.loads(self.trace.path.read_text()) if self.trace.path.exists() else None, None)
        self.trace.close()
        self.assertEqual(json.loads(self.trace.path.read_text())["observation_errors"], 1)

    async def test_tts_handoff_and_generated_pcm_are_distinct_and_pass_through(self):
        """Stitched run_tts return cannot be mistaken for completion of synthesis."""
        service = EvaluationNvidiaTTSService(
            evaluation_trace=self.trace, use_ssl=False, settings=NvidiaTTSSettings(synthesis_mode="stitched")
        )
        frame = TTSAudioRawFrame(audio=b"\0\0", sample_rate=22050, num_channels=1)

        async def parent_run(text, context_id):
            yield frame
            yield None

        with patch.object(NvidiaTTSService, "run_tts", side_effect=parent_run):
            received = [item async for item in service.run_tts("hello", "context")]
        self.assertEqual(received, [frame, None])
        parent_append = AsyncMock()
        with patch.object(NvidiaTTSService, "append_to_audio_context", parent_append):
            await service.append_to_audio_context("context", frame)
        parent_append.assert_awaited_once_with("context", frame)
        events = self.events()
        self.assertEqual(len([event for event in events if event["event"] == "tts_pcm_ready"]), 1)
        self.assertTrue(any(event["event"] == "tts_run_tts_returned" for event in events))
        self.assertFalse(any(event["event"] == "tts_worker_finished" for event in events))

    async def test_tts_worker_cancellation_propagates_and_is_labeled(self):
        """Preserve cancellation rather than recording a successful stream EOF."""
        service = EvaluationNvidiaTTSService(evaluation_trace=self.trace, use_ssl=False)
        state = SimpleNamespace(context_id="context", stop_event=threading.Event())
        state.stop_event.set()
        parent = AsyncMock(side_effect=asyncio.CancelledError())
        with (
            patch.object(NvidiaTTSService, "_synth_task_handler", parent),
            self.assertRaises(asyncio.CancelledError),
        ):
            await service._synth_task_handler(state)
        parent.assert_awaited_once_with(state)
        finished = next(event for event in self.events() if event["event"] == "tts_worker_finished")
        self.assertTrue(finished["stop_requested"])
        self.assertEqual(finished["outcome"], "cancelled")

    async def test_tts_worker_return_preserves_parent_result(self):
        """Normal worker return is recorded without implying provider success."""
        service = EvaluationNvidiaTTSService(evaluation_trace=self.trace, use_ssl=False)
        state = SimpleNamespace(context_id="context", stop_event=threading.Event())
        parent = AsyncMock(return_value="parent-result")
        with patch.object(NvidiaTTSService, "_synth_task_handler", parent):
            self.assertEqual(await service._synth_task_handler(state), "parent-result")
        parent.assert_awaited_once_with(state)
        finished = next(event for event in self.events() if event["event"] == "tts_worker_finished")
        self.assertFalse(finished["stop_requested"])
        self.assertEqual(finished["outcome"], "returned")

    async def test_asr_trace_callback_failures_do_not_change_requests_or_provider_errors(self):
        """Contain only trace failures; preserve errors raised by the parent provider."""
        trace = MagicMock(spec=EvaluationTrace)
        trace.asr_audio.side_effect = RuntimeError("observation failed")
        trace.asr_response.side_effect = RuntimeError("observation failed")
        trace.record.side_effect = RuntimeError("observation failed")
        service = EvaluationNvidiaSTTService(evaluation_trace=trace, use_ssl=False)
        requests = self.asr_client(service)
        list(service._asr_service.streaming_response_generator([b"audio"], rasr.StreamingRecognitionConfig()))
        self.assertEqual(requests[1].audio_content, b"audio")
        response = SimpleNamespace(results=None)
        provider_error = RuntimeError("provider failed")
        with (
            patch.object(NvidiaSTTService, "_handle_response", AsyncMock(side_effect=provider_error)),
            self.assertRaises(RuntimeError) as failure,
        ):
            await service._handle_response(response)
        self.assertIs(failure.exception, provider_error)

    async def test_service_observation_failure_increments_trace_quality_counter(self):
        """A skipped service observation must remain visible in saved evidence."""
        service = EvaluationNvidiaTTSService(evaluation_trace=self.trace, use_ssl=False)
        frame = TTSAudioRawFrame(audio=b"\0\0", sample_rate=22050, num_channels=1)
        with (
            patch.object(self.trace, "tts_pcm", side_effect=RuntimeError("trace failed")),
            patch.object(NvidiaTTSService, "append_to_audio_context", AsyncMock()),
        ):
            await service.append_to_audio_context("context", frame)
        self.trace.close()
        self.assertEqual(json.loads(self.trace.path.read_text())["observation_errors"], 1)

    async def test_tts_trace_callback_failures_preserve_audio_and_provider_error(self):
        """No observation failure can replace audio or mask a real synthesis error."""
        trace = MagicMock(spec=EvaluationTrace)
        trace.tts_text.side_effect = RuntimeError("observation failed")
        trace.tts_pcm.side_effect = RuntimeError("observation failed")
        trace.record.side_effect = RuntimeError("observation failed")
        service = EvaluationNvidiaTTSService(evaluation_trace=trace, use_ssl=False)
        frame = TTSAudioRawFrame(audio=b"\0\0", sample_rate=22050, num_channels=1)
        provider_error = RuntimeError("provider failed")

        async def parent_run(text, context_id):
            yield frame
            raise provider_error

        with patch.object(NvidiaTTSService, "run_tts", side_effect=parent_run):
            stream = service.run_tts("hello", "context")
            self.assertIs(await anext(stream), frame)
            with self.assertRaises(RuntimeError) as failure:
                await anext(stream)
        self.assertIs(failure.exception, provider_error)
        parent_append = AsyncMock()
        with patch.object(NvidiaTTSService, "append_to_audio_context", parent_append):
            await service.append_to_audio_context("context", frame)
        parent_append.assert_awaited_once_with("context", frame)
