# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delivering events to the connections of one session, without letting a slow reader stall the others."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from copy import deepcopy

from live.events.permissions import server_event_allowed
from live.protocol import ProtocolError, SessionConfig, uid

MAX_SIDEBAND_OBSERVERS = 8
SUBSCRIBER_QUEUE_SIZE = 128
SEND_TIMEOUT_SECONDS = 3
DRAIN_TIMEOUT_SECONDS = 1

Send = Callable[[dict], Awaitable[None]]


class Subscription:
    """An independently bounded writer, so a slow observer cannot stall the primary connection."""

    def __init__(self, send: Send, on_failure: Callable[[], Awaitable[None]], source: str):
        """Start the writer task for one connection."""
        self.send, self.on_failure, self.source = send, on_failure, source
        self.queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)
        self.task = asyncio.create_task(self._run(), name=f"{source}-event-writer")
        self._failure: asyncio.Task | None = None

    def _fail(self) -> None:
        """Run the failure handler in its own task: it closes the session, which cancels this writer."""
        if self._failure is None:
            self._failure = asyncio.create_task(self.on_failure(), name=f"{self.source}-subscriber-failure")

    def put(self, message: dict) -> None:
        """Queue a message; a full queue means the peer is too slow and its connection is failed."""
        try:
            self.queue.put_nowait(message)
        except asyncio.QueueFull:
            self.task.cancel()
            self._fail()

    async def _run(self) -> None:
        try:
            while True:
                message = await self.queue.get()
                try:
                    async with asyncio.timeout(SEND_TIMEOUT_SECONDS):
                        await self.send(message)
                finally:
                    self.queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception:
            self._fail()

    async def close(self) -> None:
        """Stop the writer."""
        self.task.cancel()
        if self.task is not asyncio.current_task():
            await asyncio.gather(self.task, return_exceptions=True)


class Fanout:
    """The connections of one session (one ``primary``, several ``sideband``) and who may see which event."""

    def __init__(
        self, config: Callable[[], SessionConfig], transport: str, on_primary_lost: Callable[[], Awaitable[None]]
    ):
        """Create an empty fan-out.

        Args:
            config: Returns the session's current configuration (permissions can be read from it).
            transport: ``webrtc`` or ``websocket``.
            on_primary_lost: Awaited when the primary connection fails; the session closes.
        """
        self.config, self.transport, self.on_primary_lost = config, transport, on_primary_lost
        self.subscribers: dict[str, Subscription] = {}

    def subscribe(self, send: Send, source: str = "sideband") -> str:
        """Attach a connection (``primary`` or ``sideband``) and return its key."""
        if (
            source == "sideband"
            and sum(s.source == "sideband" for s in self.subscribers.values()) >= MAX_SIDEBAND_OBSERVERS
        ):
            raise ProtocolError("Too many sideband observers", "rate_limit_exceeded")
        key = uid("subscriber")

        async def failure() -> None:
            if source == "primary":
                await self.on_primary_lost()
            else:
                await self.unsubscribe(key)

        self.subscribers[key] = Subscription(send, failure, source)
        return key

    async def unsubscribe(self, key: str) -> None:
        """Detach a connection."""
        sub = self.subscribers.pop(key, None)
        if sub:
            await sub.close()

    async def publish(self, message: dict, audience: str = "all") -> None:
        """Send ``message`` to every permitted connection of ``audience`` (``all``, ``primary`` or ``sideband``)."""
        for sub in tuple(self.subscribers.values()):
            if audience != "all" and sub.source != audience:
                continue
            if server_event_allowed(self.config(), self.transport, message, sub.source):
                sub.put(deepcopy(message))

    async def drain(self) -> None:
        """Wait (briefly) until every connection has been sent what was queued for it."""

        async def one(sub: Subscription) -> None:
            try:
                async with asyncio.timeout(DRAIN_TIMEOUT_SECONDS):
                    await sub.queue.join()
            except TimeoutError:
                pass

        await asyncio.gather(*(one(sub) for sub in tuple(self.subscribers.values())))

    async def close(self) -> None:
        """Detach every connection."""
        for key in tuple(self.subscribers):
            await self.unsubscribe(key)
