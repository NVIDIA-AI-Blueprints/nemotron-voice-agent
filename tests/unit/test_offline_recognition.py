# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the continuous ASR processing-prefix contract without word offsets."""

import asyncio
import struct
import threading
import unittest
from types import SimpleNamespace

from examples.generic.offline.models import RATE, AudioSource
from examples.generic.offline.recognition import ASRProcessingClock, ChunkProgressASR


class ManualClock:
    """Deterministic observed compute time, independent of source duration."""

    def __init__(self):
        """Begin before the first source chunk is supplied."""
        self.now = 0

    def __call__(self):
        """Return the current observed compute position."""
        return self.now

    def advance(self, nanoseconds):
        """Advance only when the fake backend does work."""
        self.now += nanoseconds


class AlternativeWithoutWordAccess:
    """Fail if the implementation tries to reconstruct word ownership."""

    def __init__(self, text):
        """Expose only the transcript used by the actual live wrapper."""
        self.transcript = text

    @property
    def words(self):
        """ASR word timestamps are deliberately outside this contract."""
        raise AssertionError("ASR word timestamps must never be accessed")


def result(processed_samples, text="", *, final=True, seconds=None):
    """Represent a Riva result with cumulative processing progress."""
    return SimpleNamespace(
        audio_processed=processed_samples / RATE if seconds is None else seconds,
        alternatives=[AlternativeWithoutWordAccess(text)],
        is_final=final,
    )


def response(*results):
    """Allow several ordered recognizer results in one backend response."""
    return SimpleNamespace(results=list(results))


class RecognitionContractTests(unittest.IsolatedAsyncioTestCase):
    """Check chunk provenance, complete EOF, and rejection of invalid progress."""

    async def recognize(self, source_samples, responses, *, clock=None, retain_audio=True):
        """Drain every source chunk before releasing scripted responses."""
        captured = []

        def stream(audio):
            for chunk in audio:
                if retain_audio:
                    captured.append(chunk)
            yield from responses

        adapter = ChunkProgressASR(stream, clock=clock or ManualClock())
        trace = await adapter.recognize(AudioSource(bytes(source_samples * 2)))
        return trace, captured

    async def test_one_continuous_stream_drains_all_chunks_and_preserves_interim_final_order(self):
        """Only final flags and stream order distinguish transcript events."""
        trace, chunks = await self.recognize(
            1700,
            [
                response(result(512, "Fir", final=False)),
                response(result(1024, "First", final=False), result(1536, "First statement.", final=True)),
                response(result(1700)),
            ],
        )
        self.assertEqual([len(chunk) // 2 for chunk in chunks], [512, 512, 512, 164])
        self.assertEqual([event.sequence for event in trace.events], [0, 1, 2])
        self.assertEqual([event.text for event in trace.events], ["Fir", "First", "First statement."])
        self.assertEqual([event.final for event in trace.events], [False, False, True])
        self.assertEqual([event.dependency_chunk_end for event in trace.events], [512, 1024, 1536])
        self.assertTrue(trace.metrics["asr_rpc_eof_drained"])
        self.assertFalse(trace.metrics["asr_word_timestamps"])
        self.assertEqual(trace.metrics["asr_streams"], 1)
        self.assertEqual(trace.metrics["asr_final_processed_sample"], 1700)

    async def test_progress_inside_a_chunk_is_attached_to_that_supplied_chunk_end(self):
        """A processed prefix is not snapped to an acoustic boundary."""
        trace, _ = await self.recognize(1700, [response(result(700, "Leading word"))])
        self.assertEqual(trace.events[0].processed_sample, 700)
        self.assertEqual(trace.events[0].dependency_chunk_end, 1024)
        self.assertEqual(trace.events[0].available_sample, 1024)

    async def test_transcript_text_matches_live_nvidia_frame_payload_without_stripping(self):
        """Offline delivery must not silently normalize the online event trace."""
        trace, _ = await self.recognize(
            1024,
            [response(result(512, "  ", final=False)), response(result(1024, " Leading final. ", final=True))],
        )
        self.assertEqual([event.text for event in trace.events], ["  ", " Leading final. "])

    async def test_incremental_processing_is_charged_once_after_its_source_prefix(self):
        """Earlier work overlaps later media progression instead of becoming delay."""
        clock = ManualClock()

        def stream(audio):
            list(audio)
            clock.advance(16_000_000)
            yield response(result(512, "First", final=False))
            clock.advance(4_000_000)
            yield response(result(1024, "First final", final=True))

        trace = await ChunkProgressASR(stream, clock=clock).recognize(AudioSource(bytes(2048)))
        self.assertEqual([event.available_sample for event in trace.events], [512 + 256, 1024 + 64])
        self.assertEqual([event.processing_ns for event in trace.events], [16_000_000, 4_000_000])
        self.assertEqual(trace.metrics["asr_observed_wall_ns"], 20_000_000)
        self.assertIn("not recovered realtime latency", trace.metrics["asr_timing_note"])

    async def test_fast_processing_does_not_accumulate_all_prior_prefix_work_as_lag(self):
        """Ten 2 ms steps remain 2 ms behind their 32 ms media checkpoints."""
        clock = ManualClock()

        def stream(audio):
            list(audio)
            for index in range(1, 11):
                clock.advance(2_000_000)
                yield response(result(index * 512, str(index)))

        trace = await ChunkProgressASR(stream, clock=clock).recognize(AudioSource(bytes(5120 * 2)))
        self.assertEqual([event.available_sample - event.dependency_chunk_end for event in trace.events], [32] * 10)
        self.assertEqual(sum(step["processing_ns"] for step in trace.metrics["asr_processing_steps"]), 20_000_000)

    async def test_slow_processing_retains_a_real_serial_backlog(self):
        """Work longer than its source interval queues behind earlier work."""
        clock = ManualClock()

        def stream(audio):
            list(audio)
            for index in range(1, 4):
                clock.advance(40_000_000)
                yield response(result(index * 512, str(index)))

        trace = await ChunkProgressASR(stream, clock=clock).recognize(AudioSource(bytes(1536 * 2)))
        self.assertEqual([event.available_sample for event in trace.events], [1152, 1792, 2432])

    async def test_empty_progress_and_coalesced_results_preserve_startup_work_once(self):
        """Blank checkpoints retain startup cost without inventing transcripts."""
        clock = ManualClock()

        def stream(audio):
            list(audio)
            clock.advance(100_000_000)
            yield response(result(512, ""), result(512, "First", final=False), result(512, "First final"))

        trace = await ChunkProgressASR(stream, clock=clock).recognize(AudioSource(bytes(1024)))
        self.assertEqual([event.available_sample for event in trace.events], [2112, 2112])
        self.assertEqual([step["processing_ns"] for step in trace.metrics["asr_processing_steps"]], [100_000_000, 0, 0])
        self.assertEqual([event.text for event in trace.events], ["First", "First final"])

    async def test_source_supply_wait_is_not_charged_as_provider_work(self):
        """A delayed required chunk already advances media readiness."""
        clock = ManualClock()

        def stream(audio):
            chunks = iter(audio)
            next(chunks)
            clock.advance(10_000_000)
            yield response(result(512, "First"))
            clock.advance(1_000_000_000)
            list(chunks)
            yield response(result(1024, "Second"))

        trace = await ChunkProgressASR(stream, clock=clock).recognize(AudioSource(bytes(2048)))
        self.assertEqual([event.processing_ns for event in trace.events], [10_000_000, 0])
        self.assertEqual([event.available_sample for event in trace.events], [672, 1024])

    def test_subsample_intervals_are_accumulated_before_rounding(self):
        """Many tiny processing steps cannot accumulate one-sample rounding gaps."""
        clock = ASRProcessingClock()
        self.assertEqual([clock.advance(512, 1) for _ in range(20)], [513] * 20)
        self.assertEqual(clock.available_ns, 32_000_020)

    async def test_arrival_order_remains_stable_when_later_chunk_has_less_residual_delay(self):
        """A newer transcript cannot overtake an earlier response from the stream."""
        clock = ManualClock()

        def stream(audio):
            iterator = iter(audio)
            next(iterator)
            clock.advance(100_000_000)
            yield response(result(512, "First", final=False))
            list(iterator)
            yield response(result(1024, "First final", final=True))

        trace = await ChunkProgressASR(stream, clock=clock).recognize(AudioSource(bytes(2048)))
        self.assertEqual([event.available_sample for event in trace.events], [2112, 2112])
        self.assertEqual([event.final for event in trace.events], [False, True])

    async def test_trailing_silence_and_empty_progress_results_do_not_fabricate_text(self):
        """RPC EOF proves completeness even when the last spoken final is earlier."""
        trace, chunks = await self.recognize(4096, [response(result(1024, "Question")), response(result(4096, ""))])
        self.assertEqual(len(chunks), 8)
        self.assertEqual([event.text for event in trace.events], ["Question"])
        self.assertEqual(trace.metrics["asr_final_processed_sample"], 4096)
        self.assertTrue(trace.metrics["asr_rpc_eof_drained"])

    async def test_silence_with_no_results_still_consumes_complete_source(self):
        """No progress result is required to manufacture an utterance."""
        trace, chunks = await self.recognize(1024, [])
        self.assertEqual(sum(len(chunk) for chunk in chunks), 2048)
        self.assertEqual(trace.events, ())
        self.assertTrue(trace.metrics["asr_rpc_eof_drained"])

    async def test_empty_alternative_results_can_advance_progress_without_text(self):
        """The recognizer is permitted to send processing-only results."""
        empty = result(1024)
        empty.alternatives = []
        trace, _ = await self.recognize(1024, [response(empty)])
        self.assertEqual(trace.events, ())
        self.assertEqual(trace.metrics["asr_final_processed_sample"], 1024)

    async def test_missing_progress_for_text_is_rejected(self):
        """The latest uploaded pointer must never substitute for backend progress."""
        with self.assertRaisesRegex(ValueError, "lacks a verified"):
            await self.recognize(1024, [response(result(0, "Text"))])

    async def test_reset_processing_counter_is_rejected_even_without_text(self):
        """A per-utterance counter cannot masquerade as cumulative source progress."""
        with self.assertRaisesRegex(ValueError, "cumulative source prefix"):
            await self.recognize(2048, [response(result(1024, "First")), response(result(512, ""))])

    async def test_single_sample_quantization_regression_is_clamped_without_reordering(self):
        """Float32 rounding may move a repeated prefix by one sample."""
        trace, _ = await self.recognize(
            2048, [response(result(1024, "First", final=False)), response(result(1023, "First final"))]
        )
        self.assertEqual([event.processed_sample for event in trace.events], [1024, 1024])

    async def test_float32_progress_and_source_eof_rounding_are_accepted(self):
        """Match the actual protobuf field precision on a long recording."""
        source_samples = 4_800_123
        seconds = struct.unpack("f", struct.pack("f", source_samples / RATE))[0]
        trace, _ = await self.recognize(source_samples, [response(result(0, "Final", seconds=seconds))])
        self.assertLessEqual(abs(trace.events[0].processed_sample - source_samples), 1)
        self.assertEqual(trace.events[0].dependency_chunk_end, source_samples)
        bounded, _ = await self.recognize(1024, [response(result(1025, "EOF rounding"))])
        self.assertEqual(bounded.events[0].processed_sample, 1024)

    async def test_float32_partial_eof_across_counter_magnitudes_through_one_hour(self):
        """The advertised source extent includes counters whose half ULP exceeds one sample."""
        for source_samples in (1024 * RATE + 1, 2048 * RATE - 2, 2048 * RATE + 2, 3599 * RATE + 2, 3600 * RATE):
            with self.subTest(source_samples=source_samples):
                seconds = struct.unpack("f", struct.pack("f", source_samples / RATE))[0]
                trace, _ = await self.recognize(
                    source_samples,
                    [response(result(0, "Final", seconds=seconds))],
                    retain_audio=False,
                )
                self.assertEqual(trace.events[0].processed_sample, min(source_samples, round(seconds * RATE)))
                self.assertLessEqual(abs(trace.events[0].processed_sample - source_samples), 2)
                self.assertEqual(trace.metrics["asr_source_samples"], source_samples)
                self.assertTrue(trace.metrics["asr_rpc_eof_drained"])

    async def test_long_float32_counter_still_rejects_real_overrun_or_reset(self):
        """A magnitude-aware tolerance must not hide changes larger than quantization."""
        source_samples = 2048 * RATE + 2

        def seconds(samples):
            return struct.unpack("f", struct.pack("f", samples / RATE))[0]

        cases = (
            [response(result(0, "Beyond EOF", seconds=seconds(source_samples + 8)))],
            [
                response(result(0, "First", seconds=seconds(source_samples))),
                response(result(0, "Reset", seconds=seconds(source_samples - 8))),
            ],
        )
        for scripted in cases:
            with self.subTest(results=len(scripted)), self.assertRaisesRegex(ValueError, "cumulative source prefix"):
                await self.recognize(source_samples, scripted, retain_audio=False)

    async def test_zero_progress_and_empty_source_eof_preserve_strict_text_requirements(self):
        """Zero is a valid blank checkpoint, never verified progress for recognized text."""
        trace, chunks = await self.recognize(0, [response(result(0))])
        self.assertEqual(chunks, [])
        self.assertEqual(trace.events, ())
        self.assertEqual(trace.metrics["asr_final_processed_sample"], 0)
        self.assertTrue(trace.metrics["asr_rpc_eof_drained"])
        with self.assertRaisesRegex(ValueError, "lacks a verified"):
            await self.recognize(0, [response(result(0, "Unverified"))])
        with self.assertRaisesRegex(ValueError, "cumulative source prefix"):
            await self.recognize(0, [response(result(2))])

    async def test_large_finite_progress_fails_validation_without_numeric_overflow(self):
        """Malformed finite metadata must produce the same bounded contract failure."""
        with self.assertRaisesRegex(ValueError, "cumulative source prefix"):
            await self.recognize(1024, [response(result(0, "Invalid", seconds=1e308))])

    async def test_progress_beyond_source_is_rejected(self):
        """Counter overrun cannot extend the user's immutable recording."""
        with self.assertRaisesRegex(ValueError, "cumulative source prefix"):
            await self.recognize(1024, [response(result(1026, "Past EOF"))])

    async def test_nonfinite_and_negative_counters_are_rejected(self):
        """Invalid backend metadata is an explicit failure, not an accuracy sample."""
        for seconds in [float("nan"), float("inf"), -0.1]:
            with self.subTest(seconds=seconds), self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                await self.recognize(1024, [response(result(0, "Text", seconds=seconds))])

    async def test_progress_cannot_reference_a_chunk_not_yet_supplied(self):
        """Cumulative progress is checked against the actual submission ledger."""

        def stream(audio):
            next(iter(audio))
            yield response(result(1024, "Impossible"))

        with self.assertRaisesRegex(ValueError, "exceeds supplied"):
            await ChunkProgressASR(stream, clock=ManualClock()).recognize(AudioSource(bytes(4096)))

    async def test_rpc_ending_before_request_eof_is_rejected(self):
        """A final transcript is not evidence that a continuous stream drained."""

        def stream(audio):
            next(iter(audio))
            yield response(result(512, "Early final"))

        with self.assertRaisesRegex(ValueError, "before consuming"):
            await ChunkProgressASR(stream, clock=ManualClock()).recognize(AudioSource(bytes(4096)))

    async def test_session_cancellation_cancels_the_underlying_continuous_rpc(self):
        """A cancelled HTTP session must not abandon a blocking ASR worker."""
        entered = threading.Event()
        released = threading.Event()
        finished = threading.Event()
        cancellations = []

        def stream(audio):
            list(audio)
            entered.set()
            try:
                released.wait()
                yield response(result(1024))
            finally:
                finished.set()

        def cancel():
            cancellations.append(True)
            released.set()

        adapter = ChunkProgressASR(stream, cancel=cancel, clock=ManualClock())
        task = asyncio.create_task(adapter.recognize(AudioSource(bytes(2048))))
        self.assertTrue(await asyncio.to_thread(entered.wait, 1))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(await asyncio.to_thread(finished.wait, 1))
        self.assertEqual(cancellations, [True])


if __name__ == "__main__":
    unittest.main()
