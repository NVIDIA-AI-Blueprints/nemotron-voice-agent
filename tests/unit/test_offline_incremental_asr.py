# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Incremental continuous ASR exposes real callbacks before request/RPC EOF."""

import asyncio
import math
import threading
import unittest

from test_offline_recognition import ManualClock, response, result

from examples.generic.offline.models import AudioSource
from examples.generic.offline.recognition import ChunkProgressASR


class IncrementalASRTests(unittest.IsolatedAsyncioTestCase):
    """Distinguish callback readiness, supply EOF, and actual response-stream EOF."""

    async def test_first_callback_is_consumable_while_source_and_rpc_are_still_open(self):
        """The incremental path must not secretly drain a finite tape first."""
        paused = threading.Event()
        release = threading.Event()
        clock = ManualClock()

        def stream(audio):
            iterator = iter(audio)
            next(iterator)
            yield response(result(512, "Leading", final=False))
            paused.set()
            release.wait()
            list(iterator)
            yield response(result(1024, "Leading final", final=True))

        actor = ChunkProgressASR(stream, clock=clock, cancel=release.set).start(
            AudioSource(bytes(2048)), asyncio.Event()
        )
        try:
            self.assertTrue(await asyncio.to_thread(paused.wait, 1))
            first = actor.drain()
            self.assertEqual([event.payload.text for event in first.events], ["Leading"])
            self.assertEqual(first.events[0].ready_ns, 32_000_000)
            self.assertFalse(first.done)
            clock.advance(1_000_000_000)
            self.assertEqual(actor.drain().lower_bound_ns, 32_000_000)
            release.set()
            trace = await asyncio.wait_for(actor.result(), 1)
            rest = actor.drain()
            self.assertEqual([event.payload.text for event in rest.events], ["Leading final"])
            self.assertTrue(rest.done)
            self.assertEqual(rest.lower_bound_ns, math.inf)
            self.assertEqual([event.text for event in trace.events], ["Leading", "Leading final"])
        finally:
            release.set()
            await actor.aclose()

    async def test_silence_advances_a_pending_bound_without_fabricating_text_or_eof(self):
        """Supplying all silence does not mean the recognizer has completed."""
        supplied = threading.Event()
        release = threading.Event()
        clock = ManualClock()

        def stream(audio):
            list(audio)
            supplied.set()
            release.wait()
            return
            yield  # Make the deliberately response-free RPC an iterator.

        actor = ChunkProgressASR(stream, clock=clock, cancel=release.set).start(
            AudioSource(bytes(32000)), asyncio.Event()
        )
        try:
            self.assertTrue(await asyncio.to_thread(supplied.wait, 1))
            clock.advance(50_000_000)
            snapshot = actor.drain()
            self.assertEqual(snapshot.events, ())
            self.assertEqual(snapshot.lower_bound_ns, 50_000_000)
            self.assertFalse(snapshot.done)
            release.set()
            trace = await asyncio.wait_for(actor.result(), 1)
            self.assertEqual(trace.events, ())
            self.assertTrue(trace.metrics["asr_rpc_eof_drained"])
            self.assertEqual(actor.drain().lower_bound_ns, math.inf)
        finally:
            release.set()
            await actor.aclose()

    async def test_pending_bound_charges_only_work_since_previous_callback_after_supply_eof(self):
        """Elapsed pending work is a bound in the declared serial compute model."""
        paused = threading.Event()
        release = threading.Event()
        clock = ManualClock()

        def stream(audio):
            list(audio)
            clock.advance(16_000_000)
            yield response(result(512, "First", final=False))
            paused.set()
            release.wait()
            clock.advance(4_000_000)
            yield response(result(1024, "First final", final=True))

        actor = ChunkProgressASR(stream, clock=clock, cancel=release.set).start(
            AudioSource(bytes(2048)), asyncio.Event()
        )
        try:
            self.assertTrue(await asyncio.to_thread(paused.wait, 1))
            snapshot = actor.drain()
            self.assertEqual(snapshot.events[0].ready_ns, 48_000_000)
            clock.advance(3_000_000)
            self.assertEqual(actor.drain().lower_bound_ns, 51_000_000)
            release.set()
            await asyncio.wait_for(actor.result(), 1)
            final = actor.drain().events[0]
            self.assertEqual(final.ready_ns, 71_000_000)
            self.assertGreaterEqual(final.ready_ns, 51_000_000)
        finally:
            release.set()
            await actor.aclose()

    async def test_blank_and_tied_interim_final_results_publish_as_one_complete_callback(self):
        """Blank processing retains startup work and emits no synthetic transcript."""
        paused = threading.Event()
        release = threading.Event()
        clock = ManualClock()

        def stream(audio):
            list(audio)
            clock.advance(100_000_000)
            yield response(result(512), result(512, "First", final=False), result(512, "First final"))
            paused.set()
            release.wait()

        actor = ChunkProgressASR(stream, clock=clock, cancel=release.set).start(
            AudioSource(bytes(1024)), asyncio.Event()
        )
        try:
            self.assertTrue(await asyncio.to_thread(paused.wait, 1))
            snapshot = actor.drain()
            self.assertEqual([event.payload.text for event in snapshot.events], ["First", "First final"])
            self.assertEqual([event.ready_ns for event in snapshot.events], [132_000_000, 132_000_000])
            self.assertEqual(snapshot.lower_bound_ns, 132_000_000)
            release.set()
            trace = await asyncio.wait_for(actor.result(), 1)
            self.assertEqual(
                [step["processing_ns"] for step in trace.metrics["asr_processing_steps"]], [100_000_000, 0, 0]
            )
        finally:
            release.set()
            await actor.aclose()

    async def test_invalid_prefix_propagates_without_exposing_a_partial_invalid_callback(self):
        """All results in a callback must validate before its text is delivered."""

        def stream(audio):
            next(iter(audio))
            yield response(result(512, "valid"), result(1024, "unsupplied"))

        actor = ChunkProgressASR(stream, clock=ManualClock()).start(AudioSource(bytes(2048)), asyncio.Event())
        with self.assertRaisesRegex(ValueError, "exceeds supplied"):
            await actor.result()
        with self.assertRaisesRegex(ValueError, "exceeds supplied"):
            actor.drain()
        await actor.aclose()

    async def test_cancelled_rpc_finishes_thread_before_close_returns(self):
        """The session owns its single continuous RPC until actual cancellation."""
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        cancellations = []

        def stream(audio):
            list(audio)
            entered.set()
            try:
                release.wait()
                raise RuntimeError("RPC cancelled")
                yield
            finally:
                finished.set()

        def cancel():
            cancellations.append(True)
            release.set()

        actor = ChunkProgressASR(stream, clock=ManualClock(), cancel=cancel).start(
            AudioSource(bytes(2048)), asyncio.Event()
        )
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            await actor.aclose()
            self.assertTrue(finished.is_set())
            self.assertEqual(cancellations, [True])
            with self.assertRaisesRegex(RuntimeError, "RPC cancelled"):
                await actor.result()
        finally:
            release.set()
            await actor.aclose()

    async def test_incremental_and_direct_tape_preserve_identical_observed_service_profile(self):
        """Incremental delivery changes scheduling opportunities, not captured math."""

        def adapter():
            clock = ManualClock()

            def stream(audio):
                list(audio)
                clock.advance(100_000_000)
                yield response(result(512), result(512, "First", final=False))
                clock.advance(10_000_000)
                yield response(result(1024, "First final"))

            return ChunkProgressASR(stream, clock=clock)

        source = AudioSource(bytes(2048))
        tape = adapter()._recognize(source)
        actor = adapter().start(source, asyncio.Event())
        incremental = await actor.result()
        self.assertEqual(tape.events, incremental.events)
        self.assertEqual(tape.metrics["asr_processing_steps"], incremental.metrics["asr_processing_steps"])
        self.assertEqual([event.payload for event in actor.drain().events], list(tape.events))
        await actor.aclose()


if __name__ == "__main__":
    unittest.main()
