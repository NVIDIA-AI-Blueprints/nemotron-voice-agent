# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concurrency contracts for provider batches and logical readiness bounds."""

import asyncio
import math
import threading
import unittest

from test_offline_recognition import ManualClock

from examples.generic.offline.actors import ActorEvent, EventActor


class EventActorTests(unittest.IsolatedAsyncioTestCase):
    """A scheduler cannot see a new bound before its complete callback batch."""

    async def test_snapshot_waits_for_complete_tied_callback_before_observing_new_bound(self):
        """A concurrent drain sees both tied frames and the revised bound together."""
        entered = threading.Event()
        release = threading.Event()
        state = {"bound": 0}
        actor = EventActor(asyncio.Event(), lower_bound=lambda _: state["bound"], clock=ManualClock())

        def callback(_):
            entered.set()
            release.wait()
            return [ActorEvent(100, "frame", "start"), ActorEvent(100, "frame", "audio")]

        producer = asyncio.create_task(
            asyncio.to_thread(actor.publish_batch, callback, update=lambda: state.update(bound=100))
        )
        drain = None
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            drain = asyncio.create_task(asyncio.to_thread(actor.drain))
            await asyncio.sleep(0)
            self.assertFalse(drain.done())
            release.set()
            await producer
            snapshot = await drain
            self.assertEqual([event.payload for event in snapshot.events], ["start", "audio"])
            self.assertEqual(snapshot.lower_bound_ns, 100)
            self.assertFalse(snapshot.done)
            actor.finish("complete")
            self.assertEqual(await actor.result(), "complete")
            finished = actor.drain()
            self.assertTrue(finished.done)
            self.assertEqual(finished.lower_bound_ns, math.inf)
        finally:
            release.set()
            await producer
            if drain is not None:
                await drain

    async def test_a_provider_cannot_publish_behind_a_bound_already_seen_by_the_scheduler(self):
        """Late callback timestamping must fail rather than change prior decisions."""
        clock = ManualClock()
        clock.now = 100
        actor = EventActor(asyncio.Event(), lower_bound=lambda now: now, clock=clock)
        self.assertEqual(actor.drain().lower_bound_ns, 100)
        with self.assertRaisesRegex(ValueError, "precedes"):
            actor.publish_batch([ActorEvent(99, "frame", "late")])
        actor.publish_batch(lambda now: [ActorEvent(now, "frame", "tied")])
        self.assertEqual([event.payload for event in actor.drain().events], ["tied"])
        actor.finish()

    async def test_original_provider_failure_propagates_from_drain_and_result(self):
        """An error is not converted to successful EOF or an empty answer."""
        actor = EventActor(asyncio.Event(), lower_bound=lambda _: 0)
        failure = ValueError("bad backend progress")
        actor.finish(error=failure)
        with self.assertRaisesRegex(ValueError, "bad backend progress"):
            actor.drain()
        with self.assertRaisesRegex(ValueError, "bad backend progress"):
            await actor.result()

    async def test_close_cancels_once_and_waits_for_actual_worker_cleanup(self):
        """Cancelling an async wrapper alone must not abandon its provider thread."""
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        cancellations = []

        def cancel():
            cancellations.append(True)
            release.set()

        actor = EventActor(asyncio.Event(), lower_bound=lambda _: 0, cancel=cancel)

        def provider():
            entered.set()
            try:
                release.wait()
            finally:
                finished.set()

        async def produce():
            await asyncio.to_thread(provider)
            actor.finish()

        actor.bind_worker(asyncio.create_task(produce()))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            await actor.aclose()
            self.assertTrue(finished.is_set())
            await actor.aclose()
            self.assertEqual(cancellations, [True])
            self.assertTrue(actor.drain().done)
        finally:
            release.set()
            await actor.aclose()


if __name__ == "__main__":
    unittest.main()
