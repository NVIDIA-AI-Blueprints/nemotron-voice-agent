# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe continuous ASR as incremental processing-prefix events, without word times.

Callbacks may be consumed while the continuous RPC is active. The closed RPC
remains the terminal completeness barrier for the finite immutable source.
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

from .actors import ActorEvent, ActorStream, EventActor
from .models import RATE, AudioSource, ns_to_samples, samples_to_ns


def _progress_sample_tolerance(seconds: float) -> int:
    """Bound float32 counter rounding, retaining the one-sample short-stream floor."""
    # A normal float32 has 24 significant bits. Above 2048 seconds half an
    # ULP exceeds one 16-kHz sample; the protobuf can then round valid EOF
    # progress two samples past the source. Subnormals use a fixed spacing.
    exponent = math.frexp(seconds)[1]
    half_ulp_seconds = math.ldexp(1.0, max(-149, exponent - 24)) / 2
    return max(1, math.ceil(half_ulp_seconds * RATE))


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
        """Retain the finite tape API using the same incremental producer."""
        actor = self.start(source, asyncio.Event())
        try:
            return await actor.result()
        except asyncio.CancelledError:
            await actor.aclose()
            raise

    def start(self, source: AudioSource, changed: asyncio.Event) -> ActorStream:
        """Start one continuous RPC without waiting for its finite EOF.

        Before source supply is complete, another callback can require a source
        chunk supplied later, so only prior logical completion is a safe bound.
        After supply EOF, elapsed pending work advances the chosen serial ASR
        processing model. This is not a fabricated server acknowledgement.
        """
        state = {"available_ns": 0, "previous_wall": None, "input_eof_wall": None}

        def lower_bound(now):
            eof = state["input_eof_wall"]
            if eof is None:
                return state["available_ns"]
            previous = state["previous_wall"]
            return state["available_ns"] + max(0, now - max(eof, previous if previous is not None else eof))

        actor = EventActor(changed, clock=self._clock, lower_bound=lower_bound, cancel=self.cancel)

        async def produce():
            try:
                trace = await asyncio.to_thread(self._recognize, source, actor=actor, state=state)
            except BaseException as exc:
                actor.finish(error=exc)
            else:
                actor.finish(trace)

        actor.bind_worker(asyncio.create_task(produce()))
        return actor

    def _recognize(self, source: AudioSource, *, actor=None, state=None) -> ASREventTrace:
        started = self._clock()
        positions: list[int] = []
        supplied: list[int] = []
        lock = threading.Lock()
        events = []
        previous_progress = 0
        previous_progress_tolerance = 1
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
            for chunk in source.chunks():
                with lock:
                    positions.append(chunk.end_sample)
                    supplied.append(self._clock())
                yield chunk.pcm

            def supplied_eof(now):
                nonlocal exhausted, exhausted_at
                exhausted = True
                exhausted_at = now - started
                if state is not None:
                    state["input_eof_wall"] = now
                return ()

            if actor is None:
                supplied_eof(self._clock())
            else:
                actor.publish_batch(supplied_eof)

        def observe(response, arrival):
            nonlocal first_response, first_transcript, first_final, responses, previous_progress, previous_arrival
            nonlocal previous_progress_tolerance
            if first_response is None:
                first_response = arrival - started
            responses += 1
            batch = []
            for result in response.results:
                alternatives = result.alternatives
                text = alternatives[0].transcript if alternatives else ""
                # Riva represents processed seconds as float32. Bound its
                # magnitude-dependent quantization, never a reset or word time.
                seconds = float(result.audio_processed)
                if not math.isfinite(seconds) or seconds < 0:
                    raise ValueError("ASR processing progress must be finite and nonnegative")
                tolerance = _progress_sample_tolerance(seconds)
                if seconds > (source.sample_count + tolerance) / RATE:
                    raise ValueError("ASR processing counter is not a cumulative source prefix")
                progress = round(seconds * RATE)
                if (
                    progress < previous_progress - max(tolerance, previous_progress_tolerance)
                    or progress > source.sample_count + tolerance
                ):
                    raise ValueError("ASR processing counter is not a cumulative source prefix")
                progress = min(source.sample_count, max(previous_progress, progress))
                previous_progress = progress
                previous_progress_tolerance = tolerance
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
                        "dependency_chunk_sequence": index if progress else None,
                        "processing_ns": processing_ns,
                        "available_ns": processing_clock.available_ns,
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
                event = ASRChunkEvent(
                    len(events),
                    progress,
                    chunk_end,
                    available,
                    text,
                    bool(result.is_final),
                    arrival - started,
                    processing_ns,
                )
                events.append(event)
                batch.append(ActorEvent(processing_clock.available_ns, "asr", event))
            return batch

        for response in self._stream(audio()):
            if actor is None:
                observe(response, self._clock())
            else:

                def update_progress():
                    # The callback batch, including blank processing checkpoints,
                    # must be visible before its revised future bound is visible.
                    state["available_ns"] = processing_clock.available_ns
                    state["previous_wall"] = previous_arrival

                actor.publish_batch(lambda now, response=response: observe(response, now), update=update_progress)
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
                "asr_source_chunk_count": len(positions),
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
                        "dependency_chunk_sequence": bisect.bisect_left(positions, event.dependency_chunk_end),
                        "available_sample": event.available_sample,
                        "arrival_ns": event.received_ns,
                        "processing_ns": event.processing_ns,
                    }
                    for event in events
                ],
                "asr_progress_policy": "provider_cumulative_processed_audio_to_supplied_chunk",
                "asr_completeness_barrier": "finite_response_stream_eof",
                "asr_delivery_mode": "incremental_callback_actor" if actor is not None else "finite_tape",
                "asr_pending_bound_policy": "prior_completion_until_supply_eof_then_observed_pending_interval",
                "asr_processing_steps": steps,
                "asr_availability_policy": "serial_chunk_progress_with_incremental_observed_processing",
                "asr_timing_note": (
                    "Serial source/processing schedule using observed bulk response intervals, including transport "
                    "and queueing; not recovered realtime latency or isolated inference."
                ),
            },
        )
