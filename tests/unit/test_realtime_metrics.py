# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Tests for Pipecat metrics exposed through Realtime response fields."""

import json
import unittest
from unittest.mock import MagicMock

from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMTextFrame,
    MetricsFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFBMetricsData,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame

from realtime.metrics import RealtimeMetricsAccumulator
from realtime.observer import RealtimeLifecycleObserver
from realtime.serializer import RealtimeFrameSerializer


def metrics_frame() -> MetricsFrame:
    """Build representative cascaded-service metrics for one response."""
    return MetricsFrame(
        data=[
            TTFBMetricsData(processor="NvidiaSTTService#1", value=0.8),
            TTFBMetricsData(processor="NvidiaLLMService#1", value=0.2),
            TTFBMetricsData(processor="NvidiaTTSService#1", value=0.1),
            ProcessingMetricsData(processor="NvidiaLLMService#1", value=0.5),
            LLMUsageMetricsData(
                processor="NvidiaLLMService#1",
                value=LLMTokenUsage(
                    prompt_tokens=20,
                    completion_tokens=10,
                    total_tokens=30,
                    cache_read_input_tokens=4,
                    reasoning_tokens=2,
                ),
            ),
        ]
    )


class RealtimeMetricsAccumulatorTests(unittest.TestCase):
    """Validate usage conversion and compact NVIDIA latency metadata."""

    def test_maps_pipecat_metrics_to_standard_usage_and_metadata(self) -> None:
        """Map service metrics without confusing STT and TTS processor names."""
        accumulator = RealtimeMetricsAccumulator()
        accumulator.consume(metrics_frame())
        accumulator.consume(RTVIServerMessageFrame(data={"type": "user-bot-latency", "latency": 1.25, "first": False}))
        accumulator.consume(RTVIServerMessageFrame(data={"type": "latency-breakdown", "vad_smart_turn": 0.7}))

        telemetry = accumulator.finish()

        self.assertEqual(
            telemetry.usage,
            {
                "total_tokens": 30,
                "input_tokens": 20,
                "output_tokens": 10,
                "input_token_details": {
                    "cached_tokens": 4,
                    "cached_tokens_details": {"text_tokens": 4},
                },
            },
        )
        self.assertIsNotNone(telemetry.metadata)
        encoded = telemetry.metadata["nvidia_metrics"]
        self.assertLessEqual(len(encoded), 512)
        self.assertEqual(
            json.loads(encoded),
            {
                "asr_ttfb": [0.8],
                "llm_processing_time": [0.5],
                "llm_tokens_per_sec": [20.0],
                "llm_ttft": [0.2],
                "server_e2e": [1.25],
                "tts_ttfb": [0.1],
                "vad_smart_turn": [0.7],
            },
        )

    def test_preserves_samples_and_pairs_usage_in_either_order(self) -> None:
        """Match usage and processing frames without collapsing RTVI samples."""
        accumulator = RealtimeMetricsAccumulator()
        first_usage = LLMUsageMetricsData(
            processor="NvidiaLLMService#1",
            value=LLMTokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        )
        second_usage = LLMUsageMetricsData(
            processor="NvidiaLLMService#1",
            value=LLMTokenUsage(prompt_tokens=20, completion_tokens=5, total_tokens=25),
        )
        accumulator.consume(
            MetricsFrame(
                data=[
                    first_usage,
                    ProcessingMetricsData(processor="NvidiaLLMService#1", value=2.0),
                    ProcessingMetricsData(processor="NvidiaLLMService#1", value=1.0),
                    second_usage,
                ]
            )
        )

        telemetry = accumulator.finish()

        self.assertEqual(telemetry.usage["total_tokens"], 45)
        metrics = json.loads(telemetry.metadata["nvidia_metrics"])
        self.assertEqual(metrics["llm_processing_time"], [2.0, 1.0])
        self.assertEqual(metrics["llm_tokens_per_sec"], [5.0, 5.0])

    def test_finish_clears_response_scoped_metrics(self) -> None:
        """Do not leak one response's usage into the next response."""
        accumulator = RealtimeMetricsAccumulator()
        accumulator.consume(metrics_frame())
        self.assertIsNotNone(accumulator.finish().usage)
        self.assertIsNone(accumulator.finish().usage)

    def test_total_tokens_is_authoritative_for_standard_input_usage(self) -> None:
        """Reconcile input usage when a provider reports cache-exclusive prompts."""
        accumulator = RealtimeMetricsAccumulator()
        accumulator.consume(
            MetricsFrame(
                data=[
                    LLMUsageMetricsData(
                        processor="NvidiaLLMService#1",
                        value=LLMTokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=20),
                    )
                ]
            )
        )

        self.assertEqual(accumulator.finish().usage["input_tokens"], 15)


class RealtimeMetricsLifecycleTests(unittest.IsolatedAsyncioTestCase):
    """Validate completed and cancelled response wire payloads."""

    async def _push_frame(self, observer: RealtimeLifecycleObserver, frame, *, timestamp: int = 0) -> None:
        await observer.on_push_frame(
            FramePushed(
                source=MagicMock(),
                destination=MagicMock(),
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=timestamp,
            )
        )

    async def _push_metrics(self, observer: RealtimeLifecycleObserver) -> None:
        await self._push_frame(observer, metrics_frame())

    async def test_cancelled_response_includes_partial_usage(self) -> None:
        """Preserve metrics gathered before a response cancellation."""
        emitted: list[dict] = []

        async def emit(event: dict) -> None:
            emitted.append(event)

        serializer = RealtimeFrameSerializer()
        serializer.set_emit(emit)
        observer = RealtimeLifecycleObserver(emit=emit, conversation=serializer.conversation)
        serializer.set_on_response_cancel(observer.on_response_cancelled)
        serializer.set_response_telemetry_provider(observer.take_response_telemetry)
        serializer.conversation.begin_response()
        await self._push_metrics(observer)

        await serializer.deserialize(json.dumps({"type": "response.cancel"}))

        done = next(event for event in emitted if event["type"] == "response.done")
        self.assertEqual(done["response"]["status"], "cancelled")
        self.assertEqual(done["response"]["usage"]["output_tokens"], 10)
        self.assertIn("nvidia_metrics", done["response"]["metadata"])

    async def test_text_response_keeps_all_metrics_since_previous_done(self) -> None:
        """Keep stream metrics without resetting at a text response boundary."""
        emitted: list[dict] = []

        async def emit(event: dict) -> None:
            emitted.append(event)

        serializer = RealtimeFrameSerializer()
        serializer.set_emit(emit)
        observer = RealtimeLifecycleObserver(emit=emit, conversation=serializer.conversation)
        serializer.set_on_response_cancel(observer.on_response_cancelled)
        serializer.set_response_telemetry_provider(observer.take_response_telemetry)
        serializer.conversation.open_client_text()
        await self._push_metrics(observer)

        await serializer.deserialize(json.dumps({"type": "response.create"}))
        await self._push_metrics(observer)
        await serializer.deserialize(json.dumps({"type": "response.cancel"}))

        done = next(event for event in emitted if event["type"] == "response.done")
        self.assertEqual(done["response"]["usage"]["total_tokens"], 60)

    async def test_barge_in_keeps_late_and_next_turn_metrics(self) -> None:
        """Average all samples irrespective of cancellation or turn attribution."""
        emitted: list[dict] = []

        async def emit(event: dict) -> None:
            emitted.append(event)

        serializer = RealtimeFrameSerializer()
        serializer.set_emit(emit)
        observer = RealtimeLifecycleObserver(emit=emit, conversation=serializer.conversation)
        serializer.set_on_response_cancel(observer.on_response_cancelled)
        serializer.set_response_telemetry_provider(observer.take_response_telemetry)
        serializer.conversation.begin_response()
        await self._push_metrics(observer)
        await self._push_frame(observer, UserStartedSpeakingFrame(), timestamp=1)
        await serializer.deserialize(json.dumps({"type": "response.cancel"}))

        await self._push_frame(
            observer,
            MetricsFrame(
                data=[
                    TTFBMetricsData(processor="NvidiaLLMService#old", value=9.0),
                    ProcessingMetricsData(processor="NvidiaLLMService#old", value=9.0),
                    LLMUsageMetricsData(
                        processor="NvidiaLLMService#old",
                        value=LLMTokenUsage(prompt_tokens=90, completion_tokens=90, total_tokens=180),
                    ),
                ]
            ),
        )
        await self._push_frame(
            observer,
            MetricsFrame(data=[TTFBMetricsData(processor="NvidiaSTTService#new", value=0.4)]),
        )
        await self._push_frame(observer, LLMFullResponseEndFrame())
        await self._push_frame(observer, UserStoppedSpeakingFrame(), timestamp=2)
        serializer.conversation.begin_response()
        await self._push_frame(
            observer,
            MetricsFrame(
                data=[
                    TTFBMetricsData(processor="NvidiaLLMService#new", value=0.2),
                    ProcessingMetricsData(processor="NvidiaLLMService#new", value=0.5),
                    LLMUsageMetricsData(
                        processor="NvidiaLLMService#new",
                        value=LLMTokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
                    ),
                ]
            ),
        )
        await serializer.deserialize(json.dumps({"type": "response.cancel"}))

        done_events = [event for event in emitted if event["type"] == "response.done"]
        second = done_events[1]["response"]
        self.assertEqual(second["usage"]["total_tokens"], 210)
        metrics = json.loads(second["metadata"]["nvidia_metrics"])
        self.assertEqual(metrics["asr_ttfb"], [0.4])
        self.assertEqual(metrics["llm_ttft"], [9.0, 0.2])
        self.assertEqual(metrics["llm_processing_time"], [9.0, 0.5])

    async def test_cancel_without_stale_llm_end_does_not_block_next_text_response(self) -> None:
        """Keep baseline text lifecycle when a cancelled generation emits no LLM end."""
        emitted: list[dict] = []

        async def emit(event: dict) -> None:
            emitted.append(event)

        serializer = RealtimeFrameSerializer()
        serializer.set_emit(emit)
        observer = RealtimeLifecycleObserver(emit=emit, conversation=serializer.conversation)
        serializer.set_on_response_cancel(observer.on_response_cancelled)
        serializer.set_response_telemetry_provider(observer.take_response_telemetry)

        serializer.conversation.begin_response()
        await self._push_metrics(observer)
        await serializer.deserialize(json.dumps({"type": "response.cancel"}))

        serializer.conversation.open_client_text()
        await serializer.deserialize(json.dumps({"type": "response.create"}))
        await self._push_metrics(observer)
        await self._push_frame(observer, LLMTextFrame(text="next response"))
        end = LLMFullResponseEndFrame()
        end.skip_tts = True
        await self._push_frame(observer, end)

        done_events = [event for event in emitted if event["type"] == "response.done"]
        second = done_events[1]["response"]
        self.assertEqual(second["status"], "completed")
        self.assertEqual(second["output"][0]["content"][0]["transcript"], "next response")
        self.assertEqual(second["usage"]["total_tokens"], 30)

    async def test_audio_turn_keeps_all_stream_metrics(self) -> None:
        """Do not reset metric samples when an audio turn begins."""
        emitted: list[dict] = []

        async def emit(event: dict) -> None:
            emitted.append(event)

        serializer = RealtimeFrameSerializer()
        serializer.set_emit(emit)
        observer = RealtimeLifecycleObserver(emit=emit, conversation=serializer.conversation)
        serializer.set_on_response_cancel(observer.on_response_cancelled)
        serializer.set_response_telemetry_provider(observer.take_response_telemetry)
        await self._push_metrics(observer)
        await observer.on_push_frame(
            FramePushed(
                source=MagicMock(),
                destination=MagicMock(),
                frame=UserStartedSpeakingFrame(),
                direction=FrameDirection.DOWNSTREAM,
                timestamp=1,
            )
        )
        serializer.conversation.begin_response()
        await self._push_metrics(observer)

        await serializer.deserialize(json.dumps({"type": "response.cancel"}))

        done = next(event for event in emitted if event["type"] == "response.done")
        self.assertEqual(done["response"]["usage"]["total_tokens"], 60)

    async def test_metrics_do_not_change_baseline_lifecycle_events(self) -> None:
        """Adding metric frames changes only response usage and metadata."""

        async def run(*, include_metrics: bool) -> list[dict]:
            emitted: list[dict] = []

            async def emit(event: dict) -> None:
                emitted.append(event)

            serializer = RealtimeFrameSerializer()
            serializer.set_emit(emit)
            observer = RealtimeLifecycleObserver(emit=emit, conversation=serializer.conversation)
            serializer.conversation.begin_response()
            await self._push_frame(observer, LLMTextFrame(text="unchanged"))
            if include_metrics:
                await self._push_metrics(observer)
            end = LLMFullResponseEndFrame()
            end.skip_tts = True
            await self._push_frame(observer, end)
            return emitted

        def lifecycle_view(value):
            if isinstance(value, list):
                return [lifecycle_view(item) for item in value]
            if not isinstance(value, dict):
                return value
            return {
                key: "<id>" if key == "id" or key.endswith("_id") else lifecycle_view(item)
                for key, item in value.items()
                if key not in {"usage", "metadata"}
            }

        self.assertEqual(
            lifecycle_view(await run(include_metrics=False)),
            lifecycle_view(await run(include_metrics=True)),
        )

    async def test_malformed_metric_value_does_not_affect_response_completion(self) -> None:
        """Ignore malformed telemetry while preserving the response lifecycle."""
        emitted: list[dict] = []

        async def emit(event: dict) -> None:
            emitted.append(event)

        observer = RealtimeLifecycleObserver(emit=emit, conversation=RealtimeFrameSerializer().conversation)
        observer._conversation.begin_response()
        malformed = TTFBMetricsData(processor="NvidiaSTTService#1", value=0.1)
        malformed.value = "invalid"
        await self._push_frame(observer, MetricsFrame(data=[malformed]))
        await self._push_frame(observer, LLMTextFrame(text="still completes"))
        end = LLMFullResponseEndFrame()
        end.skip_tts = True
        await self._push_frame(observer, end)

        done = next(event for event in emitted if event["type"] == "response.done")
        self.assertEqual(done["response"]["status"], "completed")
        self.assertEqual(done["response"]["output"][0]["content"][0]["transcript"], "still completes")
        self.assertNotIn("metadata", done["response"])

    async def test_telemetry_snapshot_failure_does_not_block_response_completion(self) -> None:
        """Fail open when telemetry serialization unexpectedly raises."""
        emitted: list[dict] = []

        async def emit(event: dict) -> None:
            emitted.append(event)

        observer = RealtimeLifecycleObserver(emit=emit, conversation=RealtimeFrameSerializer().conversation)
        observer._metrics = MagicMock()
        observer._metrics.finish.side_effect = RuntimeError("broken metrics")
        observer._conversation.begin_response()
        await self._push_frame(observer, LLMTextFrame(text="unaffected"))
        end = LLMFullResponseEndFrame()
        end.skip_tts = True
        await self._push_frame(observer, end)

        done = next(event for event in emitted if event["type"] == "response.done")
        self.assertEqual(done["response"]["status"], "completed")
        self.assertNotIn("usage", done["response"])
        self.assertNotIn("metadata", done["response"])
