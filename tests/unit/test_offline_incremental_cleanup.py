# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider/cleanup faults must preserve the primary error and release all work."""

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from test_offline_incremental_engine import ControlledASR, ControlledDetector, ControlledResponses

from examples.generic.offline.actors import EventActor
from examples.generic.offline.incremental import IncrementalSessionEngine, _Session
from examples.generic.offline.models import AudioSource
from examples.generic.offline.native_response import NativeResponseSession


def provider(*, cancel_error=None, setup_error=None):
    """A resource whose independent cleanup operations can fail separately."""
    return SimpleNamespace(
        setup=AsyncMock(side_effect=setup_error),
        cancel=AsyncMock(side_effect=cancel_error),
        cleanup=AsyncMock(),
        queue_frame=AsyncMock(),
        _client=SimpleNamespace(close=AsyncMock()),
    )


class NativeCleanupFaultTests(unittest.IsolatedAsyncioTestCase):
    """An interruption or one service failure cannot strand the other resources."""

    async def test_failed_interruption_still_closes_both_services_client_and_every_manager_task(self):
        """Cleanup attempts all independent resources and raises its first failure."""
        first = RuntimeError("interruption failed")
        llm = provider(cancel_error=RuntimeError("LLM cancel failed"))
        tts = provider()
        llm.cleanup.side_effect = RuntimeError("LLM cleanup failed")
        session = NativeResponseSession(lambda _: llm, lambda _: tts, sample_rate=16000)
        session._llm, session._tts = llm, tts
        session._active = SimpleNamespace(closed=False, identity="first", logical_start_ns=0)
        session.interrupt = AsyncMock(side_effect=first)
        session._manager = SimpleNamespace(
            current_tasks=lambda: ("one", "two"),
            cancel_task=AsyncMock(side_effect=[RuntimeError("first task failed"), None]),
        )
        with self.assertRaises(RuntimeError) as raised:
            await session.aclose()
        self.assertIs(raised.exception, first)
        for service in (llm, tts):
            service.cancel.assert_awaited_once()
            service.cleanup.assert_awaited_once()
        llm._client.close.assert_awaited_once()
        self.assertEqual(session._manager.cancel_task.await_count, 2)
        await session.aclose()
        self.assertEqual(tts.cleanup.await_count, 1)

    async def test_start_preserves_original_setup_error_when_its_cleanup_also_fails(self):
        """A secondary cancellation error cannot replace the failed setup cause."""
        first = RuntimeError("original setup failed")
        llm = provider(setup_error=first, cancel_error=RuntimeError("cleanup failed"))
        tts = provider()
        session = NativeResponseSession(lambda _: llm, lambda _: tts, sample_rate=16000)
        with self.assertRaises(RuntimeError) as raised:
            await session.start()
        self.assertIs(raised.exception, first)
        tts.cleanup.assert_awaited_once()
        llm._client.close.assert_awaited_once()
        self.assertIn("setup cleanup also failed", first.__notes__[0])

    async def test_failed_dispatch_does_not_leave_a_generation_future_pending(self):
        """A request may fail before its result worker is created."""
        dispatch_error = RuntimeError("queue dispatch failed")
        llm, tts = provider(), provider()
        llm.queue_frame.side_effect = dispatch_error
        session = NativeResponseSession(lambda _: llm, lambda _: tts, sample_rate=16000)
        # Already initialized services isolate failure in begin(), after the
        # generation actor exists but before its result worker can be bound.
        session._llm, session._tts = llm, tts
        with self.assertRaisesRegex(RuntimeError, "queue dispatch failed"):
            await session.begin([{"role": "user", "content": "test"}], 0, "first", asyncio.Event())
        actor = session._generations["first"].actor
        with self.assertRaises(RuntimeError) as raised:
            await session.aclose()
        self.assertIs(raised.exception, dispatch_error)
        with self.assertRaisesRegex(RuntimeError, "closed before dispatch"):
            await asyncio.wait_for(actor.result(), 1)
        tts.cleanup.assert_awaited_once()
        llm._client.close.assert_awaited_once()


class EngineCleanupFaultTests(unittest.IsolatedAsyncioTestCase):
    """A failed response cleanup must not strand ASR, VAD or aggregation tasks."""

    async def execute(self, source_failure=None):
        """Run an actual scheduler while observing every independent teardown."""
        detector = ControlledDetector(block=source_failure is not None)
        recognizer = ControlledASR(error=source_failure)
        responder = ControlledResponses()
        cleanup_error = RuntimeError("response cleanup failed")
        responder.aclose = AsyncMock(side_effect=cleanup_error)
        engine = IncrementalSessionEngine(
            detector,
            recognizer,
            responder,
            params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0)]
                ),
                empty_user_turn=None,
            ),
        )
        session = _Session(engine, AudioSource(bytes(2048)))
        with (
            patch.object(session.assistant, "close", AsyncMock(wraps=session.assistant.close)) as assistant_close,
            patch.object(session.user, "close", AsyncMock(wraps=session.user.close)) as user_close,
            patch.object(session.output, "close", Mock(wraps=session.output.close)) as output_close,
        ):
            with self.assertRaises(RuntimeError) as raised:
                await asyncio.wait_for(session.run(), 2)
            self.assertIs(raised.exception, source_failure or cleanup_error)
            assistant_close.assert_awaited()
            self.assertTrue(session.assistant._closed)
            user_close.assert_awaited_once()
            output_close.assert_called_once()
            for actor in session.actors.values():
                self.assertTrue(actor._worker.done())
        return detector, raised.exception

    async def test_original_source_failure_survives_response_cleanup_failure(self):
        """Source-producer errors remain the reason the benchmark request failed."""
        failure = RuntimeError("original ASR failure")
        detector, raised = await self.execute(failure)
        self.assertTrue(detector.cancelled)
        self.assertIs(raised, failure)
        self.assertIn("cleanup also failed", failure.__notes__[0])

    async def test_successful_body_reports_first_cleanup_failure_after_releasing_everything(self):
        """A response is not reported successful if owned resources failed to close."""
        await self.execute()

    async def test_actor_waits_for_worker_even_if_its_cancel_hook_reports_an_error(self):
        """An exception after sending cancellation cannot abandon the worker."""
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        failure = RuntimeError("cancel hook failed after release")

        def cancel():
            release.set()
            raise failure

        actor = EventActor(asyncio.Event(), lower_bound=lambda _: 0, cancel=cancel)

        def provider_work():
            entered.set()
            release.wait()
            finished.set()

        async def produce():
            await asyncio.to_thread(provider_work)
            actor.finish()

        actor.bind_worker(asyncio.create_task(produce()))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            with self.assertRaises(RuntimeError) as raised:
                await actor.aclose()
            self.assertIs(raised.exception, failure)
            self.assertTrue(finished.is_set())
            self.assertTrue(actor._worker.done())
        finally:
            release.set()
            await actor.aclose()

    async def test_cancelling_close_itself_still_waits_for_actual_provider_cleanup(self):
        """A second cancellation must not detach the thread while it releases RPC state."""
        entered, release = threading.Event(), threading.Event()
        cleanup_started, cleanup_release, finished = threading.Event(), threading.Event(), threading.Event()
        actor = EventActor(asyncio.Event(), lower_bound=lambda _: 0, cancel=release.set)

        def provider_work():
            entered.set()
            release.wait()
            cleanup_started.set()
            cleanup_release.wait()
            finished.set()

        async def produce():
            await asyncio.to_thread(provider_work)
            actor.finish()

        actor.bind_worker(asyncio.create_task(produce()))
        closing = None
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            closing = asyncio.create_task(actor.aclose())
            self.assertTrue(await asyncio.to_thread(cleanup_started.wait, 1))
            closing.cancel()
            await asyncio.sleep(0)
            self.assertFalse(closing.done())
            cleanup_release.set()
            with self.assertRaises(asyncio.CancelledError):
                await closing
            self.assertTrue(finished.is_set())
        finally:
            release.set()
            cleanup_release.set()
            if closing is not None:
                await asyncio.gather(closing, return_exceptions=True)
            await actor.aclose()


if __name__ == "__main__":
    unittest.main()
