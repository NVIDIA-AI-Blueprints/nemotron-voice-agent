# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify evaluation observations preserve the live services and frame flow."""

import asyncio
import itertools
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.frames.frames import (
    InputAudioRawFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    OutputAudioRawFrame,
    TTSAudioRawFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings
from pipecat.transports.base_output import BaseOutputTransport
from riva.client.proto import riva_asr_pb2 as rasr

from examples.shared.evaluation_trace import (
    EvaluationNvidiaForceEouSTTService,
    EvaluationNvidiaTTSService,
    EvaluationTrace,
    create_evaluation_trace,
)
from examples.shared.nvidia_force_eou_stt import NvidiaForceEouSTTService


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

    def test_unwritable_optional_trace_does_not_raise(self):
        """A tracing failure must not turn a successful conversation into a failure."""
        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            self.trace.close()


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

    async def test_asr_requests_and_force_eou_remain_byte_identical(self):
        """Do not alter the parent request stream while measuring consumption."""
        service = EvaluationNvidiaForceEouSTTService(evaluation_trace=self.trace, use_ssl=False)
        service._sample_rate = 16000
        silence = service._force_eou_silence()
        service._force_eou_silences.append(silence)
        config = rasr.StreamingRecognitionConfig()
        requests = list(service._iter_streaming_requests([b"speech", silence], config))
        self.assertEqual(requests[1].audio_content, b"speech")
        self.assertEqual(requests[2].audio_content, silence)
        self.assertEqual(dict(requests[1].runtime_config), {})
        self.assertEqual(dict(requests[2].runtime_config), {"force_eou": "true"})
        supplied = [event for event in self.events() if event["event"] == "asr_request_supplied"]
        self.assertEqual([event["force_eou"] for event in supplied], [False, False, True])

    async def test_asr_response_callback_and_worker_delegate_without_reinterpretation(self):
        """Observe original result objects and preserve the parent's worker return."""
        service = EvaluationNvidiaForceEouSTTService(evaluation_trace=self.trace, use_ssl=False)
        response = rasr.StreamingRecognizeResponse(
            results=[
                rasr.StreamingRecognitionResult(
                    is_final=True, alternatives=[rasr.SpeechRecognitionAlternative(transcript="hello")]
                )
            ]
        )
        parent = AsyncMock()
        with patch.object(NvidiaForceEouSTTService, "_handle_response", parent):
            await service._handle_response(response)
        parent.assert_awaited_once_with(response)
        iterator = SimpleNamespace(closed=False)
        with patch.object(NvidiaForceEouSTTService, "_response_handler", return_value="parent-result") as worker:
            self.assertEqual(service._response_handler(iterator), "parent-result")
        worker.assert_called_once_with(iterator)
        events = self.events()
        self.assertEqual(
            next(event for event in events if event["event"] == "asr_provider_result")["results"][0]["transcript"],
            "hello",
        )
        self.assertFalse(next(event for event in events if event["event"] == "asr_worker_finished")["iterator_closed"])

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
        trace.asr_request.side_effect = RuntimeError("observation failed")
        trace.asr_response.side_effect = RuntimeError("observation failed")
        trace.record.side_effect = RuntimeError("observation failed")
        service = EvaluationNvidiaForceEouSTTService(evaluation_trace=trace, use_ssl=False)
        requests = list(service._iter_streaming_requests([b"audio"], rasr.StreamingRecognitionConfig()))
        self.assertEqual(requests[1].audio_content, b"audio")
        response = SimpleNamespace(results=None)
        provider_error = RuntimeError("provider failed")
        with (
            patch.object(NvidiaForceEouSTTService, "_handle_response", AsyncMock(side_effect=provider_error)),
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
