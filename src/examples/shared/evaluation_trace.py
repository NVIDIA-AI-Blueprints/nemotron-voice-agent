# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in passive timing traces for paired online and offline evaluation.

Set ``NVA_EVAL_TRACE_DIR`` to retain one JSON file per Generic Assistant session.
The trace observes the normal pipeline; it never creates frames, changes pacing,
or makes provider requests. PCM is counted rather than retained. Writing happens
after the worker has stopped, so disk I/O is outside the conversational path.

The optional service subclasses delegate every operation to the pinned Pipecat
1.12 implementation. Their small private-method compatibility surface observes
ASR request yields/results and stitched TTS worker completion. Worker completion
is distinct from output playback completion and may follow cancellation.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import hashlib
import json
import math
import os
import threading
import time
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ClientConnectedFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    OutputAudioRawFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.services.llm_service import LLMService
from pipecat.services.nvidia.stt import NvidiaSTTService
from pipecat.services.nvidia.tts import NvidiaTTSService
from pipecat.transports.base_output import BaseOutputTransport

_LIFECYCLE_FRAMES = (
    ClientConnectedFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
)

TIMING_SCOPES = {
    "clock": (
        "elapsed_ns uses a session-local monotonic callback clock; elapsed_ms = elapsed_ns / 1e6. "
        "Observer callbacks may wait in Pipecat's proxy queue"
    ),
    "input_end_sample": "Received PCM cursor at observation; not a backdated raw speech boundary",
    "asr_input_ingested": (
        "STT process_audio_frame entered before the parent handles mute/reconnect/resampling; "
        "its own cursor is normalized to 16000 Hz and is not proof of provider consumption"
    ),
    "asr_request_supplied": (
        "Parent Riva audio iterator consumed byte-identical PCM; independent cumulative cursor normalized "
        "to 16000 Hz, not socket delivery or an ASR completeness acknowledgement"
    ),
    "asr_provider_result": "Existing nonempty-results callback entered on the event loop; not raw gRPC first byte",
    "audio_processed_seconds": (
        "Provider-reported processed-audio counter when finite and nonnegative; processed_sample_16k is its "
        "rounded 16000-Hz equivalent, not word ownership, event time, or a completeness watermark"
    ),
    "llm_context_request_edge": (
        "LLMContextFrame pushed to an LLMService, copied when the delayed observer callback runs. "
        "pipeline_timestamp_ns is the original edge push, not an HTTP dispatch acknowledgement; "
        "the mutable context may have changed before this snapshot"
    ),
    "llm_frames": (
        "Use pipeline_timestamp_ns differences for LLM start/first nonempty text/end: this is the original "
        "frame push clock. elapsed_ns is delayed observer callback time. Neither clock epoch is interchangeable"
    ),
    "tts_text_submitted": "Filtered text entered run_tts; not text consumed by the gRPC request iterator",
    "tts_text_frame": "First observed TTSTextFrame, distinct from raw LLM token text and from played PCM",
    "tts_text_output_delivered": (
        "TTSTextFrame pushed by the native output transport after its FIFO handling; "
        "not proof of client playback or assistant-context consumption"
    ),
    "tts_pcm_ready": "Generated PCM entered the TTS audio-context queue, before paced output transport",
    "tts_worker_finished": (
        "Stitched synthesis worker task ended; includes LLM input stalls. A cancelled task is not RPC EOF. "
        "The provider implementation catches RPC errors, so normal return alone does not establish RPC success"
    ),
    "tts_run_tts_returned": "Per-sentence synthesis end, or a text handoff in stitched mode; inspect synthesis_mode",
    "output_audio_sent": "Normal output transport pushed a PCM frame after write_audio_frame returned",
    "pipecat_metrics": "Original Pipecat metric values remain seconds; corresponding value_ms is supplied",
    "asr_ttfb": "Pipecat: inferred user speech end to finalized transcript; not stream-start first transcript",
    "llm_ttft": "Pipecat: first choices-bearing completion chunk, which may contain no text",
    "server_e2e": "Pipecat: inferred user speech end to BotStartedSpeakingFrame",
    "vad_smart_turn": "Pipecat: inferred user speech end to semantic turn release, including ASR finalization",
    "missing": "An absent event or duration is unobserved, never zero",
}


def _finite_number(value) -> float | None:
    """Return only finite numeric metadata, preserving missing values as null."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _observe(trace, method: str, *args, **kwargs) -> None:
    """Contain tracing exceptions without ever wrapping a provider operation."""
    try:
        getattr(trace, method)(*args, **kwargs)
    except Exception as error:
        try:
            trace._note_failure(error)
        except Exception:
            logger.debug(f"Optional evaluation observation skipped ({type(error).__name__})")


class EvaluationTrace(BaseObserver):
    """Buffer a session's timing observations without touching its frame flow."""

    def __init__(self, directory: Path, *, configuration=None, clock=time.perf_counter_ns, max_events=100_000):
        """Allocate a unique trace with a bounded event buffer and injected clock."""
        super().__init__(observe_every_push=True)
        self.session_id = str(uuid.uuid4())
        self.path = directory / f"{self.session_id}.json"
        self._configuration = configuration or {}
        self._clock = clock
        self._started = clock()
        self._lock = threading.RLock()
        self._events: list[dict[str, Any]] = []
        self._seen: set[tuple[str, int]] = set()
        self._contexts: dict[str, int] = {}
        self._input_end_sample = 0
        self._asr_ingest_samples_16k = Fraction(0)
        self._asr_ingest_sequence = 0
        self._asr_supplied_samples_16k = Fraction(0)
        self._asr_supplied_audio_bytes = 0
        self._asr_supplied_audio_hash = hashlib.sha256()
        self._context_version = 0
        self._llm_request_index = 0
        self._max_events = max_events
        self._dropped = 0
        self._observation_errors = 0
        self._closed = False
        self.record("trace_started")

    def record(self, event: str, **fields) -> None:
        """Append one observation, including calls from provider worker threads."""
        try:
            self._record(event, **fields)
        except Exception as error:
            self._note_failure(error)

    def _note_failure(self, error: Exception) -> None:
        with self._lock:
            self._observation_errors += 1
        logger.debug(f"Optional evaluation observation skipped ({type(error).__name__})")

    def _record(self, event: str, **fields) -> None:
        with self._lock:
            if self._closed:
                return
            if len(self._events) >= self._max_events:
                self._dropped += 1
                return
            if fields.get("context_id") in self._contexts:
                fields.setdefault("llm_request_index", self._contexts[fields["context_id"]] or None)
            elapsed = self._clock() - self._started
            self._events.append(
                {
                    "sequence": len(self._events),
                    "event": event,
                    "elapsed_ns": elapsed,
                    "elapsed_ms": elapsed / 1_000_000,
                    "input_end_sample": self._input_end_sample,
                    "llm_request_index": self._llm_request_index or None,
                    **fields,
                }
            )

    def _once(self, scope: str, frame) -> bool:
        key = (scope, frame.id)
        with self._lock:
            if key in self._seen:
                return False
            if len(self._seen) < self._max_events:
                self._seen.add(key)
            return True

    def tts_pcm(self, context_id: str | None, frame: TTSAudioRawFrame) -> None:
        """Count generated PCM once even when both service hooks see the frame."""
        if not frame.audio or not self._once("tts_pcm", frame):
            return
        self.record(
            "tts_pcm_ready",
            context_id=context_id,
            llm_request_index=self._contexts.get(context_id),
            audio_bytes=len(frame.audio),
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
        )

    def tts_text(self, context_id: str, text: str, synthesis_mode: str) -> None:
        """Associate a synthesis context with the currently observed LLM request."""
        with self._lock:
            self._contexts.setdefault(context_id, self._llm_request_index)
        self.record(
            "tts_text_submitted",
            context_id=context_id,
            llm_request_index=self._contexts[context_id] or None,
            synthesis_mode=synthesis_mode,
            text=text,
        )

    def configure_asr(self, *, stop_history: int) -> None:
        """Retain endpointing values from the initialized standard service."""
        with self._lock:
            self._configuration.update(
                asr_service="NvidiaSTTService", asr_stop_history_ms=stop_history, asr_force_eou=False
            )

    def asr_input(self, frame: InputAudioRawFrame) -> None:
        """Tag source-frame ingestion without claiming the parent accepted its PCM."""
        if not self._once("asr_input", frame):
            return
        count = Fraction(len(frame.audio) * 16000, 2 * max(frame.num_channels, 1) * frame.sample_rate)
        with self._lock:
            start = round(self._asr_ingest_samples_16k)
            self._asr_ingest_samples_16k += count
            tag = {
                "seq": self._asr_ingest_sequence,
                "start": start,
                "end": round(self._asr_ingest_samples_16k),
                "rate": 16000,
            }
            self._asr_ingest_sequence += 1
            # Preserve any pre-existing metadata. Only this opt-in observer uses
            # the private tag; the native processors receive the same frame.
            tagged = "nva_eval_source" not in frame.metadata
            if tagged:
                frame.metadata["nva_eval_source"] = copy.deepcopy(tag)
            self.record(
                "asr_input_ingested",
                frame_id=frame.id,
                source_tag=tag,
                source_tag_written=tagged,
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
                audio_bytes=len(frame.audio),
                audio_sha256=hashlib.sha256(frame.audio).hexdigest(),
            )

    def asr_audio(self, audio: bytes, *, sample_rate: int = 16000, num_channels: int = 1) -> None:
        """Count original PCM when the standard Riva request iterator consumes it."""
        count = Fraction(len(audio) * 16000, 2 * max(num_channels, 1) * sample_rate)
        with self._lock:
            start = round(self._asr_supplied_samples_16k)
            self._asr_supplied_samples_16k += count
            self._asr_supplied_audio_bytes += len(audio)
            self._asr_supplied_audio_hash.update(audio)
            self.record(
                "asr_request_supplied",
                audio_bytes=len(audio),
                audio_sha256=hashlib.sha256(audio).hexdigest(),
                asr_supplied_start_sample_16k=start,
                asr_supplied_end_sample_16k=round(self._asr_supplied_samples_16k),
                asr_supplied_total_audio_bytes=self._asr_supplied_audio_bytes,
                sample_rate=sample_rate,
                num_channels=num_channels,
                force_eou=False,
                is_configuration=False,
            )

    def asr_response(self, response) -> None:
        """Read result metadata without putting extraction into provider control flow."""
        results = []
        for result in response.results:
            item = {
                "is_final": result.is_final,
                "transcript": result.alternatives[0].transcript if result.alternatives else "",
            }
            seconds = _finite_number(getattr(result, "audio_processed", None))
            if seconds is not None and seconds >= 0:
                item.update(audio_processed_seconds=seconds, processed_sample_16k=round(seconds * 16000))
            results.append(item)
        self.record(
            "asr_provider_result",
            results=results,
        )

    async def on_push_frame(self, data: FramePushed) -> None:
        """Observe frame boundaries, deduplicating regular and post-write edges."""
        try:
            await self._observe_frame(data)
        except Exception as error:
            self._note_failure(error)

    async def _observe_frame(self, data: FramePushed) -> None:
        frame = data.frame
        if data.direction != FrameDirection.DOWNSTREAM and not isinstance(frame, InterruptionFrame):
            return
        output_edge = isinstance(frame, OutputAudioRawFrame) and isinstance(data.source, BaseOutputTransport)
        if output_edge and self._once("output", frame):
            self.record(
                "output_audio_sent",
                audio_bytes=len(frame.audio),
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
                pipeline_timestamp_ns=data.timestamp,
            )
        if (
            isinstance(frame, LLMContextFrame)
            and isinstance(data.destination, LLMService)
            and self._once("llm_context_request", frame)
        ):
            messages = copy.deepcopy(frame.context.get_messages())
            with self._lock:
                self._context_version += 1
                self.record(
                    "llm_context_request_edge",
                    context_version=self._context_version,
                    llm_request_index=None,
                    messages=messages,
                    speculation=frame.speculation,
                    frame_id=frame.id,
                    source=data.source.name,
                    destination=data.destination.name,
                    pipeline_timestamp_ns=data.timestamp,
                )
        if (
            isinstance(frame, TTSTextFrame)
            and isinstance(data.source, BaseOutputTransport)
            and self._once("tts_text_output", frame)
        ):
            self.record(
                "tts_text_output_delivered",
                text=frame.text,
                context_id=frame.context_id,
                frame_id=frame.id,
                source=data.source.name,
                pipeline_timestamp_ns=data.timestamp,
            )
        if not self._once("frame", frame):
            return
        shared = {"frame_id": frame.id, "source": data.source.name, "pipeline_timestamp_ns": data.timestamp}
        if isinstance(frame, InputAudioRawFrame):
            samples = len(frame.audio) // (2 * max(frame.num_channels, 1))
            with self._lock:
                start = self._input_end_sample
                self._input_end_sample += samples
            self.record(
                "input_audio_received",
                input_start_sample=start,
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
                audio_bytes=len(frame.audio),
                **shared,
            )
        elif isinstance(frame, TTSTextFrame):
            self.record("tts_text_frame", text=frame.text, context_id=frame.context_id, **shared)
        elif isinstance(frame, LLMTextFrame):
            if frame.text and frame.text.strip():
                self.record("llm_text", text=frame.text, **shared)
        elif isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            self.record(
                "asr_final_text" if isinstance(frame, TranscriptionFrame) else "asr_interim_text",
                text=frame.text,
                finalized=getattr(frame, "finalized", None),
                **shared,
            )
        elif isinstance(frame, MetricsFrame):
            values = []
            for item in frame.data:
                try:
                    value = item.model_dump() if hasattr(item, "model_dump") else dataclasses.asdict(item)
                except Exception as error:
                    self._note_failure(error)
                    continue
                value["metric_type"] = type(item).__name__
                if type(item).__name__ in {"TTFBMetricsData", "ProcessingMetricsData"}:
                    number = _finite_number(value.get("value"))
                    value["value_ms"] = number * 1000 if number is not None else None
                values.append(value)
            self.record("pipecat_metrics", metrics=values, **shared)
        elif isinstance(frame, RTVIServerMessageFrame):
            if isinstance(frame.data, dict) and frame.data.get("type") in {
                "user-bot-latency",
                "latency-breakdown",
                "user-turn-finalized",
            }:
                self.record("server_message", data=frame.data, **shared)
        elif isinstance(frame, _LIFECYCLE_FRAMES):
            if isinstance(frame, LLMFullResponseStartFrame):
                self._llm_request_index += 1
            timestamp = _finite_number(getattr(frame, "timestamp", None))
            vad_frame = isinstance(frame, (VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame))
            self.record(
                type(frame).__name__,
                start_secs=getattr(frame, "start_secs", None),
                stop_secs=getattr(frame, "stop_secs", None),
                frame_timestamp_unix_ms=timestamp * 1000 if vad_frame and timestamp is not None else None,
                context_id=getattr(frame, "context_id", None),
                **shared,
            )

    def close(self) -> None:
        """Write the final trace once; tracing failures never fail a voice session."""
        try:
            with self._lock:
                if self._closed:
                    return
                self.record("trace_closed")
                self._closed = True
                document = {
                    "schema": "nva.evaluation.trace.v1",
                    "session_id": self.session_id,
                    "configuration": self._configuration,
                    "asr_supplied_audio": {
                        "audio_bytes": self._asr_supplied_audio_bytes,
                        "samples_16k": round(self._asr_supplied_samples_16k),
                        "sha256": self._asr_supplied_audio_hash.hexdigest(),
                    },
                    "timing_scopes": TIMING_SCOPES,
                    "dropped_events": self._dropped,
                    "observation_errors": self._observation_errors,
                    "events": self._events,
                }
            self.path.write_text(json.dumps(document, indent=2, default=str) + "\n")
            logger.info(f"Evaluation timing trace saved: {self.path}")
        except Exception:
            logger.exception("Unable to save optional evaluation timing trace")


def create_evaluation_trace(body: dict) -> EvaluationTrace | None:
    """Enable tracing only through a server-owned directory environment setting."""
    raw = os.getenv("NVA_EVAL_TRACE_DIR", "").strip()
    if not raw:
        return None
    try:
        directory = Path(raw)
        directory.mkdir(parents=True, exist_ok=True)
        view = body.get("realtime_session_view") or {}
        return EvaluationTrace(
            directory,
            configuration={
                "protocol": body.get("protocol", "rtvi"),
                "pipeline_mode": body.get("pipeline_mode", "generic-assistant"),
                "realtime_session_id": view.get("id") if isinstance(view, dict) else None,
                "prompt_key": body.get("prompt_key"),
            },
        )
    except Exception:
        logger.exception("Unable to initialize optional evaluation timing trace")
        return None


class _ObservedASRClient:
    """Delegate Riva requests while observing consumption of their original PCM."""

    def __init__(self, client, trace: EvaluationTrace):
        """Wrap one initialized client without replacing Pipecat's stream iterator."""
        self._client = client
        self._trace = trace

    def __getattr__(self, name):
        """Preserve the initialized client's remaining attributes unchanged."""
        return getattr(self._client, name)

    def streaming_response_generator(self, audio_chunks, streaming_config):
        """Let Riva create every request from byte-identical source chunks."""

        def audio():
            for chunk in audio_chunks:
                _observe(
                    self._trace,
                    "asr_audio",
                    chunk,
                    sample_rate=streaming_config.config.sample_rate_hertz or 16000,
                    num_channels=streaming_config.config.audio_channel_count or 1,
                )
                yield chunk

        return self._client.streaming_response_generator(audio_chunks=audio(), streaming_config=streaming_config)


class EvaluationNvidiaSTTService(NvidiaSTTService):
    """Observe the normal continuous ASR service without changing its requests."""

    def __init__(self, *args, evaluation_trace: EvaluationTrace, **kwargs):
        """Retain the session observer and initialize the unmodified ASR service."""
        self._evaluation_trace = evaluation_trace
        super().__init__(*args, **kwargs)
        _observe(self._evaluation_trace, "configure_asr", stop_history=self._stop_history)

    def _initialize_client(self):
        """Wrap the real client once while retaining the parent's iterator identity."""
        super()._initialize_client()
        self._asr_service = _ObservedASRClient(self._asr_service, self._evaluation_trace)

    def _response_handler(self, iterator):
        """Observe the existing ASR worker lifetime, including reconnect attempts."""
        _observe(self._evaluation_trace, "record", "asr_worker_started")
        try:
            return super()._response_handler(iterator)
        finally:
            _observe(self._evaluation_trace, "record", "asr_worker_finished", iterator_closed=iterator.closed)

    async def _handle_response(self, response):
        """Timestamp the existing event-loop callback without changing its data."""
        _observe(self._evaluation_trace, "asr_response", response)
        return await super()._handle_response(response)

    async def process_audio_frame(self, frame, direction):
        """Observe actual source ingestion before delegating its unchanged handling."""
        _observe(self._evaluation_trace, "asr_input", frame)
        return await super().process_audio_frame(frame, direction)


class EvaluationNvidiaTTSService(NvidiaTTSService):
    """Observe text handoff, generated PCM, and the stitched worker lifecycle."""

    def __init__(self, *args, evaluation_trace: EvaluationTrace, **kwargs):
        """Retain the session observer and initialize the unmodified TTS service."""
        self._evaluation_trace = evaluation_trace
        super().__init__(*args, **kwargs)

    async def run_tts(self, text: str, context_id: str):
        """Delegate synthesis unchanged, separating handoff from worker EOF."""
        mode = str(self._settings.synthesis_mode)
        _observe(self._evaluation_trace, "tts_text", context_id, text, mode)
        outcome = "returned"
        try:
            async for frame in super().run_tts(text, context_id):
                if isinstance(frame, TTSAudioRawFrame):
                    _observe(self._evaluation_trace, "tts_pcm", context_id, frame)
                yield frame
        except BaseException as error:
            outcome = "cancelled" if isinstance(error, asyncio.CancelledError) else "raised"
            raise
        finally:
            _observe(
                self._evaluation_trace,
                "record",
                "tts_run_tts_returned",
                context_id=context_id,
                synthesis_mode=mode,
                outcome=outcome,
            )

    async def append_to_audio_context(self, context_id, frame):
        """Timestamp generated stitched PCM before the normal audio queue append."""
        if isinstance(frame, TTSAudioRawFrame):
            _observe(self._evaluation_trace, "tts_pcm", context_id, frame)
        return await super().append_to_audio_context(context_id, frame)

    async def _synth_task_handler(self, state):
        """Observe pinned Pipecat's worker return, separate from paced playback."""
        _observe(self._evaluation_trace, "record", "tts_worker_started", context_id=state.context_id)
        outcome = "returned"
        try:
            return await super()._synth_task_handler(state)
        except BaseException as error:
            outcome = "cancelled" if isinstance(error, asyncio.CancelledError) else "raised"
            raise
        finally:
            _observe(
                self._evaluation_trace,
                "record",
                "tts_worker_finished",
                context_id=state.context_id,
                stop_requested=state.stop_event.is_set(),
                outcome=outcome,
            )
