# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102

import asyncio
import threading
import unittest
from types import SimpleNamespace

from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.metrics.metrics import ProcessingMetricsData, TTFBMetricsData
from pipecat.services.nvidia.tts import NvidiaTTSSettings

from examples.generic.offline.tts_trace import (
    CapturedTTSFrame,
    CapturingNvidiaTTSService,
    LiveTTSResponder,
    _provider_timing_metrics,
)
from examples.shared.nemotron_speech_text_filter import NemotronSpeechTextFilter


class _Call:
    def __init__(self, requests, stub):
        self.requests = requests
        self.stub = stub
        self.cancelled = threading.Event()

    def __iter__(self):
        for request in self.requests:
            if self.cancelled.is_set():
                return
            self.stub.texts.append(request.text)
            self.stub.called.set()
            if self.stub.error:
                raise RuntimeError("test synthesis failure")
            yield SimpleNamespace(audio=b"\x01\x00" * 320)

    def cancel(self):
        self.cancelled.set()


class _Stub:
    def __init__(self, *, error=False):
        self.error = error
        self.calls = []
        self.texts = []
        self.called = threading.Event()
        self.closed = False

    def SynthesizeOnline(self, requests, *, metadata):
        call = _Call(requests, self)
        self.calls.append(call)
        return call


class _Service(CapturingNvidiaTTSService):
    def __init__(self, capture, stub):
        super().__init__(
            capture=capture,
            server="unused:1234",
            use_ssl=False,
            settings=NvidiaTTSSettings(synthesis_mode="stitched"),
            text_filters=[NemotronSpeechTextFilter()],
            enable_direct_mode=True,
        )
        self.stub = stub

    def _initialize_client(self):
        self._service = SimpleNamespace(
            stub=self.stub,
            auth=SimpleNamespace(
                get_auth_metadata=lambda: [],
                channel=SimpleNamespace(close=lambda: setattr(self.stub, "closed", True)),
            ),
        )

    def _create_synthesis_config(self):
        return object()


class _Text:
    def __init__(self, chunks):
        self.chunks = chunks

    async def stream(self, messages):
        for text in self.chunks:
            yield text


class LiveTTSResponderTests(unittest.IsolatedAsyncioTestCase):
    async def make_responder(self, chunks, *, stub=None):
        stub = stub or _Stub()
        service = []

        def factory(capture):
            instance = _Service(capture, stub)
            service.append(instance)
            return instance

        responder = LiveTTSResponder(_Text(chunks), factory, sample_rate=16000)
        self.addAsyncCleanup(responder.aclose)
        return responder, stub, service

    async def test_actual_service_orders_context_text_audio_and_end(self):
        responder, stub, services = await self.make_responder(["Hello ", "*world*. ", "Next sentence."])
        trace = await responder.respond([{"role": "user", "content": "test"}], "turn-1")
        frames = [item.frame for item in trace.frames]
        self.assertIsInstance(
            next(frame for frame in frames if not isinstance(frame, MetricsFrame)), LLMFullResponseStartFrame
        )
        self.assertIsInstance(frames[-1], LLMFullResponseEndFrame)
        self.assertEqual(sum(isinstance(frame, TTSStartedFrame) for frame in frames), 1)
        self.assertEqual(sum(isinstance(frame, TTSStoppedFrame) for frame in frames), 1)
        self.assertTrue(any(isinstance(frame, TTSAudioRawFrame) for frame in frames))
        self.assertTrue(any(isinstance(frame, TTSTextFrame) for frame in frames))
        self.assertEqual(len(stub.calls), 1)
        self.assertNotIn("*", "".join(stub.texts))
        self.assertEqual(
            "Hello world.Next sentence.",
            "".join(frame.text for frame in frames if isinstance(frame, TTSTextFrame)),
        )
        self.assertEqual(sum(len(chunk.pcm) for chunk in trace.audio), 640 * len(stub.texts))
        self.assertTrue(trace.metrics["tts_rpc_eof_drained"])
        self.assertIsNone(services[0]._stream_state)
        self.assertEqual(
            sorted(item.ready_offset_ns for item in trace.frames), [item.ready_offset_ns for item in trace.frames]
        )

    async def test_two_responses_reuse_processor_but_open_separate_stitched_calls(self):
        responder, stub, services = await self.make_responder(["Hello."])
        first = await responder.respond([], "turn-1")
        second = await responder.respond([], "turn-2")
        self.assertEqual(len(services), 1)
        self.assertEqual(len(stub.calls), 2)
        self.assertNotEqual(
            next(item.frame.context_id for item in first.frames if isinstance(item.frame, TTSStartedFrame)),
            next(item.frame.context_id for item in second.frames if isinstance(item.frame, TTSStartedFrame)),
        )

    async def test_native_service_metrics_are_captured_without_estimating_from_llm_text(self):
        responder, _, _ = await self.make_responder(["Hello. Next sentence."])
        trace = await responder.respond([], "turn-1")
        native = [
            (item, datum)
            for item in trace.frames
            if isinstance(item.frame, MetricsFrame)
            for datum in item.frame.data
            if isinstance(datum, (TTFBMetricsData, ProcessingMetricsData))
        ]
        ttfb = [datum for _, datum in native if isinstance(datum, TTFBMetricsData)]
        self.assertEqual(len(ttfb), 1)
        self.assertEqual(trace.metrics["tts_first_audio_latency_ns"], round(ttfb[0].value * 1_000_000_000))
        self.assertEqual(
            trace.metrics["tts_provider_metrics"],
            [
                {
                    "kind": "ttfb" if isinstance(datum, TTFBMetricsData) else "processing",
                    "value_ns": round(datum.value * 1_000_000_000),
                    "ready_offset_ns": item.ready_offset_ns,
                }
                for item, datum in native
            ],
        )
        self.assertTrue(any(isinstance(datum, ProcessingMetricsData) for _, datum in native))
        self.assertIn("not complete stitched synthesis", trace.metrics["tts_provider_timing_note"])

    async def test_empty_response_does_not_invent_tts_first_byte_latency(self):
        responder, _, _ = await self.make_responder([])
        trace = await responder.respond([], "turn-1")
        self.assertNotIn("tts_first_audio_latency_ns", trace.metrics)
        self.assertEqual(trace.metrics["tts_provider_metrics"], [])

    def test_closed_timer_without_audio_does_not_become_first_byte_latency(self):
        frames = [CapturedTTSFrame(MetricsFrame(data=[TTFBMetricsData(processor="test", value=0.2)]), 250)]
        metrics = _provider_timing_metrics(frames)
        self.assertNotIn("tts_first_audio_latency_ns", metrics)
        self.assertEqual(
            metrics["tts_provider_metrics"], [{"kind": "ttfb", "value_ns": 200_000_000, "ready_offset_ns": 250}]
        )

    async def test_empty_response_has_no_synthesis_call_and_drains_end(self):
        responder, stub, _ = await self.make_responder([])
        trace = await responder.respond([], "turn-1")
        self.assertEqual(trace.audio, ())
        self.assertEqual(stub.calls, [])
        self.assertIsInstance(trace.frames[-1].frame, LLMFullResponseEndFrame)

    async def test_native_audio_frames_remain_native_while_compatibility_pcm_is_canonical(self):
        stub = _Stub()
        responder = LiveTTSResponder(_Text(["Hello."]), lambda capture: _Service(capture, stub), sample_rate=22050)
        self.addAsyncCleanup(responder.aclose)
        trace = await responder.respond([], "turn-1")
        native = [item.frame for item in trace.frames if isinstance(item.frame, TTSAudioRawFrame)]
        self.assertEqual([frame.sample_rate for frame in native], [22050])
        self.assertEqual(native[0].audio, b"\x01\x00" * 320)
        # Continuous resampling flushes the fractional final chunk exactly once.
        self.assertAlmostEqual(sum(len(chunk.pcm) // 2 for chunk in trace.audio), 320 * 16000 / 22050, delta=1)

    async def test_serialized_end_without_rpc_eof_is_rejected(self):
        class PrematureEnd(_Service):
            async def process_frame(self, frame, direction):
                if isinstance(frame, LLMFullResponseEndFrame):
                    self._stream_state = object()
                    await self.push_frame(frame, direction)
                    self._stream_state = None
                else:
                    await super().process_frame(frame, direction)

        responder = LiveTTSResponder(_Text([]), lambda capture: PrematureEnd(capture, _Stub()), sample_rate=16000)
        self.addAsyncCleanup(responder.aclose)
        with self.assertRaisesRegex(RuntimeError, "before its synthesis RPC drained"):
            await responder.respond([], "turn-1")

    async def test_rpc_failure_propagates_instead_of_producing_successful_silent_trace(self):
        responder, _, _ = await self.make_responder(["Hello."], stub=_Stub(error=True))
        with self.assertRaisesRegex(RuntimeError, "test synthesis failure"):
            await responder.respond([], "turn-1")

    async def test_rpc_failure_aborts_llm_while_waiting_for_another_token(self):
        responder, _, _ = await self.make_responder([], stub=_Stub(error=True))
        closed = asyncio.Event()

        class HeldText:
            async def stream(self, messages):
                try:
                    yield "Hello. More "
                    await asyncio.Event().wait()
                finally:
                    closed.set()

        responder._text_stream = HeldText()
        with self.assertRaisesRegex(RuntimeError, "test synthesis failure"):
            async with asyncio.timeout(3):
                await responder.respond([], "turn-1")
        self.assertTrue(closed.is_set())

    async def test_cancellation_aborts_rpc_and_next_response_recreates_service(self):
        responder, stub, services = await self.make_responder([])

        class HeldText:
            async def stream(self, messages):
                yield "Hello. More "
                await asyncio.Event().wait()

        responder._text_stream = HeldText()
        task = asyncio.create_task(responder.respond([], "turn-1"))
        self.assertTrue(await asyncio.to_thread(stub.called.wait, 3))
        responder.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(stub.calls[0].cancelled.is_set())
        self.assertTrue(stub.closed)
        responder._text_stream = _Text(["Again."])
        trace = await responder.respond([], "turn-2")
        self.assertEqual(len(services), 2)
        self.assertTrue(trace.metrics["tts_rpc_eof_drained"])

    async def test_compute_deadline_stops_generation_before_returning_trace(self):
        responder, _, _ = await self.make_responder([])

        class HeldText:
            async def stream(self, messages):
                yield "Hello. More "
                await asyncio.Event().wait()

        responder._text_stream = HeldText()
        trace = await responder.respond([], "turn-1", compute_budget_ns=100_000_000)
        self.assertTrue(trace.metrics["compute_interrupted_at_audio_deadline"])
        self.assertFalse(trace.metrics["tts_rpc_eof_drained"])
        self.assertEqual(trace.compute_ns, 100_000_000)
        self.assertTrue(all(item.ready_offset_ns <= trace.compute_ns for item in trace.frames))
        self.assertIsNone(responder._service)
