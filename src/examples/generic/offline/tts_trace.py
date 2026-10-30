# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture the live NVIDIA TTS frame stream without a paced playback sink.

The service, rather than this adapter, owns sentence aggregation, text filtering,
stitched synthesis, audio contexts and the relative order of audio and text.
Measured elapsed time describes provider readiness, not a conversational timer.
"""

import asyncio
import copy
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    CancelFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
    StartFrame,
    TextFrame,
    TTSAudioRawFrame,
)
from pipecat.metrics.metrics import ProcessingMetricsData, TTFBMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings
from pipecat.utils.asyncio.task_manager import TaskManager

from examples.shared.nemotron_speech_text_filter import NemotronSpeechTextFilter
from utils import is_nvcf

from .models import RATE, GeneratedAudio, ResponseTrace

Capture = Callable[[Frame, FrameDirection], Awaitable[None]]


@dataclass(frozen=True)
class CapturedTTSFrame:
    """An actual service output with its measured response-relative readiness."""

    frame: Frame
    ready_offset_ns: int
    direction: FrameDirection = FrameDirection.DOWNSTREAM


@dataclass(frozen=True)
class TTSResponseTrace(ResponseTrace):
    """Provider trace whose frames retain the live downstream ordering.

    Frames are runtime objects and deliberately live outside JSON metrics. The
    playback adapter must consume them in order, including native-rate audio,
    instead of rebuilding text commits from the completed response string.
    """

    frames: tuple[CapturedTTSFrame, ...] = ()


class CapturingNvidiaTTSService(NvidiaTTSService):
    """Observe the normal push path after NVIDIA/base-service side effects."""

    def __init__(self, *, capture: Capture, **kwargs):
        """Keep synthesis behavior unchanged and inject a private trace sink."""
        super().__init__(**kwargs)
        self._capture = capture

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Retain upstream error reporting and actual serialized output order."""
        await super().push_frame(frame, direction)
        await self._capture(frame, direction)


class LiveTTSResponder:
    """Drive the same TTS processor used online from streamed LLM text.

    One processor is reused across responses in a session. Its setup runs before
    the per-response measurement, as it does online. Cancellation tears down the
    active RPC and processor; a subsequent response gets a fresh processor.
    """

    def __init__(self, text_stream, service_factory, *, sample_rate: int, clock=time.monotonic_ns):
        """Inject the LLM stream and a ``factory(capture)`` NVIDIA service."""
        if sample_rate <= 0:
            raise ValueError("TTS sample rate must be positive")
        self._text_stream = text_stream
        self._factory = service_factory
        self._sample_rate = sample_rate
        self._clock = clock
        self._service = None
        self._manager = None
        self._active_task = None
        self._capture_response = None
        self._setup_ns = 0

    async def _capture(self, frame, direction):
        if self._capture_response is not None:
            await self._capture_response(frame, direction)

    async def _ensure_service(self):
        if self._service is not None:
            return
        began = self._clock()
        self._manager = TaskManager()
        clock = SystemClock()
        clock.start()
        service = self._factory(self._capture)
        self._service = service
        try:
            await service.setup(
                FrameProcessorSetup(
                    clock=clock,
                    task_manager=self._manager,
                    pipeline_worker=None,
                    audio_in_sample_rate=RATE,
                    audio_out_sample_rate=self._sample_rate,
                    enable_metrics=True,
                )
            )
            await service.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        except BaseException:
            await self._discard_service()
            raise
        self._setup_ns = self._clock() - began

    async def _discard_service(self):
        service, self._service = self._service, None
        try:
            if service is not None:
                try:
                    await service.cancel(CancelFrame())
                finally:
                    await service.cleanup()
        finally:
            if self._manager is not None:
                for task in tuple(self._manager.current_tasks()):
                    await self._manager.cancel_task(task)
                self._manager = None

    def cancel(self):
        """Cancel current generation; its finally block aborts synthesis."""
        if self._active_task is not None:
            self._active_task.cancel()

    async def aclose(self):
        """Release the session-owned processor and its gRPC channel."""
        task = self._active_task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._discard_service()

    async def respond(self, messages, turn_id, *, compute_budget_ns=None) -> TTSResponseTrace:
        """Capture one response through explicit RPC EOF and serialized end."""
        if self._active_task is not None:
            raise RuntimeError("A session cannot synthesize overlapping responses")
        if compute_budget_ns is not None and compute_budget_ns <= 0:
            return TTSResponseTrace("", (), 0, metrics={"compute_interrupted_at_audio_deadline": True})
        self._active_task = asyncio.current_task()
        frames = []
        texts = []
        contexts = []
        done = asyncio.get_running_loop().create_future()
        started = None
        llm_first = None
        llm_end = None
        interrupted = False
        original_end = LLMFullResponseEndFrame()

        async def capture(frame, direction):
            offset = self._clock() - started
            frames.append(CapturedTTSFrame(frame, offset, direction))
            if isinstance(frame, ErrorFrame) and not done.done():
                done.set_exception(RuntimeError(f"Live NVIDIA TTS failed: {frame.error}"))
            if frame is original_end and direction == FrameDirection.DOWNSTREAM and not done.done():
                if self._service._stream_state is not None:
                    done.set_exception(RuntimeError("TTS context ended before its synthesis RPC drained"))
                else:
                    done.set_result(None)

        async def generate():
            nonlocal llm_first, llm_end
            await self._service.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
            async for event in self._text_stream.stream(copy.deepcopy(messages)):
                offset = self._clock() - started
                if isinstance(event, dict):
                    contexts.append(
                        {
                            "ready_offset_ns": offset,
                            **(event if "replace_messages" in event else {"message": event}),
                        }
                    )
                    continue
                if llm_first is None:
                    llm_first = offset
                texts.append(event)
                await self._service.process_frame(TextFrame(event), FrameDirection.DOWNSTREAM)
            llm_end = self._clock() - started
            await self._service.process_frame(original_end, FrameDirection.DOWNSTREAM)

        try:
            await self._ensure_service()
            started = self._clock()
            self._capture_response = capture
            try:
                async with asyncio.timeout(None if compute_budget_ns is None else compute_budget_ns / 1_000_000_000):
                    generation = asyncio.create_task(generate())
                    try:
                        # A synthesis failure must abort a still-streaming LLM,
                        # rather than wait indefinitely for its next token.
                        await asyncio.gather(generation, done)
                    finally:
                        if not generation.done():
                            generation.cancel()
                        await asyncio.gather(generation, return_exceptions=True)
            except TimeoutError:
                if compute_budget_ns is None:
                    raise
                interrupted = True
                await self._discard_service()
            elapsed = self._clock() - started
            compute_ns = min(elapsed, compute_budget_ns) if interrupted else elapsed
            frames = [item for item in frames if item.ready_offset_ns <= compute_ns]
            contexts = [item for item in contexts if item["ready_offset_ns"] <= compute_ns]
            audio = _canonical_audio(frames, compute_ns)
            first_audio = next(
                (item.ready_offset_ns for item in frames if isinstance(item.frame, TTSAudioRawFrame)), None
            )
            metrics = {
                "turn_id": turn_id,
                "llm_first_text_ns": llm_first,
                "llm_complete_ns": llm_end,
                "first_audio_ready_ns": first_audio,
                "response_observed_wall_ns": elapsed,
                "compute_interrupted_at_audio_deadline": interrupted,
                "tts_setup_ns": self._setup_ns,
                "tts_rpc_eof_drained": not interrupted,
                "tts_frame_policy": "actual_NvidiaTTSService_serialized_frames",
                "context_events": contexts,
                "timing_basis": "overlapping_provider_wall_trace",
                **_provider_timing_metrics(frames),
            }
            provider_metrics = getattr(self._text_stream, "last_run_metrics", None)
            if provider_metrics is not None:
                metrics["llm_provider_metrics"] = copy.deepcopy(provider_metrics)
            return TTSResponseTrace("".join(texts), audio, compute_ns, metrics=metrics, frames=tuple(frames))
        except BaseException:
            await self._discard_service()
            raise
        finally:
            self._capture_response = None
            self._active_task = None
            if done.done() and not done.cancelled():
                done.exception()


def _provider_timing_metrics(frames):
    """Keep native provider measurements distinct from response readiness.

    NVIDIA starts its stock TTFB timer when the synthesis context opens and
    stops it when the first synthesis audio arrives. Its processing timers
    instead surround individual ``run_tts`` calls, which hand off text and
    return before stitched synthesis completes. Neither is an LLM-token-to-
    audio estimate or the duration of the complete synthesis RPC.
    """
    records = []
    has_audio = any(isinstance(item.frame, TTSAudioRawFrame) for item in frames)
    for item in frames:
        if not isinstance(item.frame, MetricsFrame):
            continue
        for datum in item.frame.data:
            if not isinstance(datum, (TTFBMetricsData, ProcessingMetricsData)):
                continue
            if not math.isfinite(datum.value) or datum.value < 0:
                continue
            records.append(
                {
                    "kind": "ttfb" if isinstance(datum, TTFBMetricsData) else "processing",
                    "value_ns": round(datum.value * 1_000_000_000),
                    "ready_offset_ns": item.ready_offset_ns,
                }
            )
    metrics = {
        "tts_provider_metrics": records,
        "tts_provider_timing_note": (
            "Native Pipecat wall-clock metrics: TTFB runs from synthesis-context start "
            "to first provider audio; processing values cover individual run_tts text "
            "handoffs, not complete stitched synthesis. Response readiness is measured separately."
        ),
    }
    # stop_all_metrics() can also close a timer during cancellation. A native
    # timing record without provider audio is not evidence of a first byte.
    ttfb = next((record["value_ns"] for record in records if record["kind"] == "ttfb"), None)
    if has_audio and ttfb is not None:
        metrics["tts_first_audio_latency_ns"] = ttfb
    return metrics


def _canonical_audio(frames, compute_ns):
    """Provide compatibility PCM; faithful playback consumes native frames."""
    import numpy as np
    import soxr

    result = []
    stream = None
    source_rate = None
    for item in frames:
        frame = item.frame
        if not isinstance(frame, TTSAudioRawFrame) or item.direction != FrameDirection.DOWNSTREAM:
            continue
        if frame.num_channels != 1 or len(frame.audio) % 2:
            raise ValueError("NVIDIA TTS must emit mono PCM16")
        if source_rate is None:
            source_rate = frame.sample_rate
            if source_rate != RATE:
                stream = soxr.ResampleStream(source_rate, RATE, 1, dtype="int16", quality="VHQ")
        elif source_rate != frame.sample_rate:
            raise ValueError("TTS changed sample rate inside one response")
        pcm = frame.audio
        if stream is not None:
            pcm = stream.resample_chunk(np.frombuffer(pcm, dtype=np.int16)).tobytes()
        if pcm:
            result.append(GeneratedAudio(pcm, item.ready_offset_ns))
    if stream is not None:
        tail = stream.resample_chunk(np.empty(0, dtype=np.int16), last=True).tobytes()
        if tail:
            result.append(GeneratedAudio(tail, compute_ns))
    return tuple(result)


def create_live_tts_responder(text_stream, settings):
    """Use the generic online NVIDIA TTS settings and speech text filter."""

    def factory(capture):
        model_map = {}
        if settings.get("tts_function_id") or settings.get("tts_model"):
            model_map["model_function_map"] = {
                "function_id": settings.get("tts_function_id", ""),
                "model_name": settings.get("tts_model", ""),
            }
        return CapturingNvidiaTTSService(
            capture=capture,
            api_key=settings.get("api_key"),
            server=settings["tts_server"],
            use_ssl=is_nvcf(settings["tts_server"]),
            settings=NvidiaTTSSettings(
                voice=settings["tts_voice_id"],
                language=settings.get("tts_language_code") or "en-US",
                synthesis_mode="stitched",
            ),
            text_filters=[NemotronSpeechTextFilter()],
            custom_dictionary=settings.get("custom_dictionary"),
            enable_direct_mode=True,
            **model_map,
        )

    return LiveTTSResponder(text_stream, factory, sample_rate=settings["tts_sample_rate"])
