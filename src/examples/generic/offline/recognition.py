# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drain continuous ASR into ordered processing-prefix events, without word times.

The closed RPC is the completeness barrier for this finite source-only pass.
``audio_processed`` is the provider's cumulative audio-processing counter, not a
word boundary or a wall-clock arrival timestamp. It is mapped to an actually
supplied PCM chunk. A serial processing clock charges each observed interval
once, after both that chunk and preceding work are available. This bulk service
profile does not reconstruct a counterfactual realtime ASR execution.
"""

import asyncio
import bisect
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .models import RATE, AudioSource, ns_to_samples, samples_to_ns


@dataclass(frozen=True)
class ASRChunkEvent:
    """One nonempty interim or final, ordered by the continuous response stream."""

    sequence: int
    processed_sample: int
    dependency_chunk_end: int
    available_sample: int
    text: str
    final: bool
    received_ns: int = 0
    processing_ns: int = 0


@dataclass(frozen=True)
class ASREventTrace:
    """Complete finite-stream events, with explicit processing and timing scope."""

    events: tuple[ASRChunkEvent, ...]
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ASRProcessingClock:
    """Schedule serial processing checkpoints without cumulative backlog replay.

    ``max(source_prefix, prior_completion) + observed_interval`` allows earlier
    processing to overlap later source progression. All arithmetic stays in
    nanoseconds until the cumulative availability is rounded to a sample.
    """

    available_ns: int = 0

    def advance(self, chunk_end: int, processing_ns: int) -> int:
        """Complete one ordered checkpoint after source readiness and prior work."""
        if chunk_end < 0 or processing_ns < 0:
            raise ValueError("ASR processing positions and intervals must be nonnegative")
        self.available_ns = max(samples_to_ns(chunk_end), self.available_ns) + processing_ns
        return ns_to_samples(self.available_ns)


class ChunkProgressASR:
    """Capture every transcription event and await the response iterator's EOF."""

    def __init__(self, stream, *, cancel=None, clock=time.monotonic_ns):
        """Own one continuous recognition call and its cancellation hook."""
        self._stream = stream
        self._cancel = cancel
        self._clock = clock

    def cancel(self):
        """Cancel RPC work when its owning session is cancelled."""
        if self._cancel:
            self._cancel()

    async def recognize(self, source: AudioSource) -> ASREventTrace:
        """Source-only lookahead is complete before any conversation is replayed."""
        try:
            return await asyncio.to_thread(self._recognize, source)
        except asyncio.CancelledError:
            self.cancel()
            raise

    def _recognize(self, source: AudioSource) -> ASREventTrace:
        started = self._clock()
        positions: list[int] = []
        supplied: list[int] = []
        lock = threading.Lock()
        events = []
        previous_progress = 0
        processing_clock = ASRProcessingClock()
        previous_arrival = started
        steps = []
        responses = 0
        exhausted = False
        exhausted_at = None
        first_response = None
        first_transcript = None
        first_final = None

        def audio():
            nonlocal exhausted, exhausted_at
            for start in range(0, source.sample_count, 512):
                end = min(start + 512, source.sample_count)
                with lock:
                    positions.append(end)
                    supplied.append(self._clock())
                yield source.pcm[start * 2 : end * 2]
            exhausted = True
            exhausted_at = self._clock() - started

        for response in self._stream(audio()):
            arrival = self._clock()
            if first_response is None:
                first_response = arrival - started
            responses += 1
            for result in response.results:
                alternatives = result.alternatives
                text = alternatives[0].transcript if alternatives else ""
                # Riva represents processed seconds as float32. Accept only
                # single-sample quantization drift, never a reset or word time.
                seconds = float(result.audio_processed)
                if not math.isfinite(seconds) or seconds < 0:
                    raise ValueError("ASR processing progress must be finite and nonnegative")
                progress = round(seconds * RATE)
                if progress < previous_progress - 1 or progress > source.sample_count + 1:
                    raise ValueError("ASR processing counter is not a cumulative source prefix")
                progress = min(source.sample_count, max(previous_progress, progress))
                previous_progress = progress
                if text and progress == 0:
                    raise ValueError("ASR text lacks a verified processed-audio prefix")
                with lock:
                    if progress:
                        index = bisect.bisect_left(positions, progress)
                        if index == len(positions):
                            raise ValueError("ASR processing progress exceeds supplied source chunks")
                        chunk_end, supplied_at = positions[index], supplied[index]
                    else:
                        chunk_end, supplied_at = 0, started
                # Bulk arrival intervals include provider/transport/queue work,
                # not isolated GPU inference. Exclude time before the required
                # source chunk was supplied; charge each interval exactly once.
                processing_ns = max(0, arrival - max(previous_arrival, supplied_at))
                previous_arrival = arrival
                available = processing_clock.advance(chunk_end, processing_ns)
                steps.append(
                    {
                        "processed_sample": progress,
                        "dependency_chunk_end": chunk_end,
                        "processing_ns": processing_ns,
                        "available_sample": available,
                        "arrival_ns": arrival - started,
                    }
                )
                if not text:
                    continue
                if first_transcript is None:
                    first_transcript = arrival - started
                if result.is_final and first_final is None:
                    first_final = arrival - started
                events.append(
                    ASRChunkEvent(
                        len(events),
                        progress,
                        chunk_end,
                        available,
                        text,
                        bool(result.is_final),
                        arrival - started,
                        processing_ns,
                    )
                )
        if not exhausted or (source.sample_count and positions[-1] != source.sample_count):
            raise ValueError("ASR RPC ended before consuming the entire finite source")
        ended = self._clock() - started
        return ASREventTrace(
            tuple(events),
            {
                "asr_mode": "continuous_stream",
                "asr_streams": 1,
                "asr_word_timestamps": False,
                "asr_source_samples": source.sample_count,
                "asr_final_processed_sample": previous_progress,
                "asr_response_count": responses,
                "asr_event_count": len(events),
                "asr_rpc_eof_drained": True,
                "asr_observed_wall_ns": ended,
                "asr_stream_end_ns": ended,
                "asr_first_response_ns": first_response,
                "asr_first_transcript_ns": first_transcript,
                "asr_first_final_transcript_ns": first_final,
                "asr_input_exhausted_ns": exhausted_at,
                "asr_first_audio_supplied_ns": supplied[0] - started if supplied else None,
                "asr_last_audio_supplied_ns": supplied[-1] - started if supplied else None,
                "asr_events": [
                    {
                        "sequence": event.sequence,
                        "final": event.final,
                        "processed_sample": event.processed_sample,
                        "dependency_chunk_end": event.dependency_chunk_end,
                        "available_sample": event.available_sample,
                        "arrival_ns": event.received_ns,
                        "processing_ns": event.processing_ns,
                    }
                    for event in events
                ],
                "asr_progress_policy": "provider_cumulative_processed_audio_to_supplied_chunk",
                "asr_completeness_barrier": "finite_response_stream_eof",
                "asr_processing_steps": steps,
                "asr_availability_policy": "serial_chunk_progress_with_incremental_observed_processing",
                "asr_timing_note": (
                    "Serial source/processing schedule using observed bulk response intervals, including transport "
                    "and queueing; not recovered realtime latency or isolated inference."
                ),
            },
        )
