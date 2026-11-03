# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own early native preparation without moving service state off its loop."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from examples.generic.offline.factory import GenericOfflineRuntime
from examples.generic.offline.models import AudioSource, SessionResult, SourceAbortedError, StreamingAudioSource


class PreparedServices:
    """An owned async setup boundary with observable cancellation and cleanup."""

    def __init__(self):
        """Hold readiness until the test permits setup completion."""
        self.calls = 0
        self.loops = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False
        self.closed = False

    async def start(self):
        """Represent real loop-owned service and queue setup without providers."""
        self.calls += 1
        self.loops.append(asyncio.get_running_loop())
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    async def aclose(self):
        """Release only this synthetic service owner."""
        self.closed = True


class RuntimePreparationTests(unittest.IsolatedAsyncioTestCase):
    """One preparation task is shared by upload-time warmup and execution."""

    def runtime(self, services, run=None):
        """Construct owned synthetic ports, preserving the actual runtime wrapper."""
        engine = SimpleNamespace(run=run or AsyncMock(return_value=SessionResult((), 0, (), ())))
        recognizer = SimpleNamespace(cancel=Mock())
        return GenericOfflineRuntime(engine, None, recognizer, services, {})

    async def test_parallel_prepare_and_run_use_exactly_one_owning_loop_setup(self):
        """Starting execution during warmup must await the same native readiness."""
        services = PreparedServices()
        run = AsyncMock(return_value=SessionResult((), 0, (), ()))
        runtime = self.runtime(services, run)
        warmup = asyncio.create_task(runtime.prepare())
        await services.entered.wait()
        execution = asyncio.create_task(runtime.run(AudioSource(b"")))
        await asyncio.sleep(0)
        run.assert_not_awaited()
        services.release.set()
        await asyncio.wait_for(asyncio.gather(warmup, execution), 1)
        self.assertEqual(services.calls, 1)
        self.assertEqual(services.loops, [asyncio.get_running_loop()])
        run.assert_awaited_once()
        self.assertTrue(services.closed)

    async def test_cancelled_waiter_does_not_cancel_other_preparation_waiter(self):
        """The owner, rather than an incidental waiter, controls setup lifetime."""
        services = PreparedServices()
        runtime = self.runtime(services)
        first = asyncio.create_task(runtime.prepare())
        second = asyncio.create_task(runtime.prepare())
        await services.entered.wait()
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertFalse(services.cancelled)
        services.release.set()
        await asyncio.wait_for(second, 1)
        self.assertEqual(services.calls, 1)
        await runtime.aclose()

    async def test_close_cancels_pending_setup_and_releases_all_owned_ports(self):
        """Abandoned upload preparation cannot keep native queues or tasks alive."""
        services = PreparedServices()
        runtime = self.runtime(services)
        preparation = asyncio.create_task(runtime.prepare())
        await services.entered.wait()
        await asyncio.wait_for(runtime.aclose(), 1)
        with self.assertRaises(asyncio.CancelledError):
            await preparation
        self.assertTrue(services.cancelled)
        self.assertTrue(services.closed)
        runtime._recognizer.cancel.assert_called_once()
        await runtime.aclose()
        with self.assertRaisesRegex(RuntimeError, "closing or closed"):
            await runtime.prepare()

    async def test_setup_error_survives_cleanup_error_without_starting_engine(self):
        """Preparation failures retain their primary error and attempt every release."""
        primary = ValueError("synthetic setup failure")
        services = SimpleNamespace(
            start=AsyncMock(side_effect=primary), aclose=AsyncMock(side_effect=OSError("cleanup"))
        )
        run = AsyncMock()
        runtime = self.runtime(services, run)
        warmup = asyncio.create_task(runtime.prepare())
        await asyncio.sleep(0)
        with self.assertRaises(ValueError) as caught:
            await runtime.run(AudioSource(b""))
        await asyncio.gather(warmup, return_exceptions=True)
        self.assertIs(caught.exception, primary)
        self.assertTrue(any("cleanup also failed" in note for note in primary.__notes__))
        run.assert_not_awaited()
        services.aclose.assert_awaited_once()
        runtime._recognizer.cancel.assert_called_once()

    async def test_cancelled_streaming_run_aborts_input_and_pending_preparation(self):
        """A blocked source reader is woken before runtime cleanup joins work."""
        services = PreparedServices()
        runtime = self.runtime(services)
        source = StreamingAudioSource(512)
        source_reader = asyncio.create_task(asyncio.to_thread(lambda: tuple(source.chunks())))
        warmup = asyncio.create_task(runtime.prepare())
        await services.entered.wait()
        execution = asyncio.create_task(runtime.run(source))
        await asyncio.sleep(0)
        execution.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(execution, 1)
        with self.assertRaises(SourceAbortedError):
            await asyncio.wait_for(source_reader, 1)
        self.assertTrue(services.cancelled)
        self.assertTrue(services.closed)
        await asyncio.gather(warmup, return_exceptions=True)

    async def test_default_sealed_run_enters_engine_before_native_setup(self):
        """Unrequested warmup preserves the original engine-relative time origin."""
        order = []

        async def start():
            order.append("native_setup")

        services = SimpleNamespace(start=start, aclose=AsyncMock())

        async def run(source):
            order.append("engine_entered")
            await services.start()
            return SessionResult((), source.sample_count, (), ())

        runtime = self.runtime(services, run)
        await runtime.run(AudioSource(b""))
        self.assertEqual(order, ["engine_entered", "native_setup"])
        self.assertIsNone(runtime._prepare_task)

    async def test_first_preparation_cannot_race_engine_owned_setup_after_run_enters(self):
        """Late warmup must not launch a second concurrent native initialization."""
        services = PreparedServices()

        async def run(source):
            await services.start()
            return SessionResult((), source.sample_count, (), ())

        runtime = self.runtime(services, run)
        execution = asyncio.create_task(runtime.run(AudioSource(b"")))
        await services.entered.wait()
        with self.assertRaisesRegex(RuntimeError, "before execution"):
            await runtime.prepare()
        self.assertEqual(services.calls, 1)
        services.release.set()
        await asyncio.wait_for(execution, 1)

    async def test_trace_receives_immutable_source_only_after_validated_seal(self):
        """A fast completion cannot lose or prematurely read the source receipt."""
        services = PreparedServices()
        services.release.set()
        source = StreamingAudioSource(512)
        trace = SimpleNamespace(offline_source=Mock(), close=Mock())

        async def run(value):
            self.assertIs(value, source)
            trace.offline_source.assert_not_called()
            source.append(bytes(1024))
            source.seal()
            return SessionResult((), 512, (), ())

        runtime = self.runtime(services, run)
        runtime._evaluation_trace = trace
        await asyncio.wait_for(runtime.run(source), 1)
        trace.offline_source.assert_called_once()
        self.assertIsInstance(trace.offline_source.call_args.args[0], AudioSource)
        self.assertEqual(trace.offline_source.call_args.args[0].pcm, bytes(1024))
