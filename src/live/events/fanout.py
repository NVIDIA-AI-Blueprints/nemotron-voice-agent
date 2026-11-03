# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delivering events to the connections of one session, without letting a slow reader stall the others.

Each connection has its own bounded queue and writer. An event is copied once when it is published, and that one copy
goes to every connection that may see it. The relay of a delegation's events waits for room in the primary
connection's queue (:meth:`Fanout.wait_for_room`), so a burst slows the relay instead of overflowing the queue. A
primary connection that falls a whole queue behind, or whose send fails or stalls, closes its session; a sideband that
does is dropped and closed with the reason.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from copy import deepcopy

from loguru import logger

from live.events.permissions import server_event_allowed
from live.protocol import ProtocolError, SessionConfig, uid

MAX_SIDEBAND_OBSERVERS = 8
# How many events each connection may fall behind by, by its role: the primary carries the session's audio and every
# delegation event; a sideband only observes.
QUEUE_SIZES = {"primary": 1024, "sideband": 128}
SEND_TIMEOUT_SECONDS = 3
DRAIN_TIMEOUT_SECONDS = 1

Send = Callable[[dict], Awaitable[None]]
Close = Callable[[str], Awaitable[None]]  # closes a dropped connection, with the reason


class Subscription:
    """An independently bounded writer, so a slow observer cannot stall the primary connection."""

    def __init__(self, send: Send, on_failure: Callable[[str], Awaitable[None]], source: str):
        """Start the writer task for one connection; ``on_failure(reason)`` runs once if the connection fails."""
        self.send, self.on_failure, self.source = send, on_failure, source
        self.queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=QUEUE_SIZES[source])
        self.half = self.queue.maxsize // 2
        # Set while the queue is at most half full, or once the writer has stopped: nothing then waits for room.
        self.room = asyncio.Event()
        self.room.set()
        self.stopped = False
        self.task = asyncio.create_task(self._run(), name=f"{source}-event-writer")
        self._failure: asyncio.Task | None = None

    def _fail(self, reason: str) -> None:
        """Run the failure handler in its own task: it closes the session or the connection, which stops this writer."""
        self._stop()
        if self._failure is None:
            self._failure = asyncio.create_task(self.on_failure(reason), name=f"{self.source}-subscriber-failure")

    def _stop(self) -> None:
        """Queue nothing more, and let whoever waits for room go on."""
        self.stopped = True
        self.room.set()

    def put(self, message: dict) -> None:
        """Queue a message; a full queue means the peer is too slow and its connection is failed."""
        if self.stopped:
            return
        try:
            self.queue.put_nowait(message)
        except asyncio.QueueFull:
            self.task.cancel()
            self._fail(f"The {self.source} connection fell {self.queue.maxsize} events behind the session")
            return
        if self.queue.qsize() > self.half:
            self.room.clear()

    async def _run(self) -> None:
        try:
            while True:
                message = await self.queue.get()
                if self.queue.qsize() <= self.half:
                    self.room.set()
                try:
                    async with asyncio.timeout(SEND_TIMEOUT_SECONDS):
                        await self.send(message)
                finally:
                    self.queue.task_done()
        except TimeoutError:
            self._fail(f"An event could not be sent to the {self.source} connection within {SEND_TIMEOUT_SECONDS} s")
        except Exception as exc:
            logger.warning(f"An event could not be sent to a {self.source} connection: {exc!r}")
            self._fail(f"An event could not be sent to the {self.source} connection")
        finally:
            self._stop()

    async def close(self) -> None:
        """Stop the writer."""
        self._stop()
        self.task.cancel()
        if self.task is not asyncio.current_task():
            # wait, not gather: if this waiter is cancelled too, gather raises the writer's cancel in place of the
            # waiter's own, and an anyio cancel scope (Starlette's) lets a cancel that is not its own escape
            await asyncio.wait({self.task})


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

    def subscribe(self, send: Send, source: str, close: Close | None = None) -> str:
        """Attach a connection in its role (``primary`` or ``sideband``) and return its key.

        ``close(reason)`` closes a sideband that falls behind or fails, once it is detached; the primary's failure
        closes the session instead.
        """
        if (
            source == "sideband"
            and sum(s.source == "sideband" for s in self.subscribers.values()) >= MAX_SIDEBAND_OBSERVERS
        ):
            raise ProtocolError(f"Too many sideband observers: at most {MAX_SIDEBAND_OBSERVERS}", "rate_limit_exceeded")
        key = uid("subscriber")

        async def failure(reason: str) -> None:
            logger.warning(reason)
            if source == "primary":
                await self.on_primary_lost()
                return
            await self.unsubscribe(key)
            if close is not None:
                await close(f"{reason}; attach again to observe")

        self.subscribers[key] = Subscription(send, failure, source)
        return key

    def role(self, key: str) -> str | None:
        """The role of the connection ``key`` (``primary`` or ``sideband``), or ``None`` once it is detached."""
        sub = self.subscribers.get(key)
        return sub.source if sub is not None else None

    async def unsubscribe(self, key: str) -> None:
        """Detach a connection."""
        sub = self.subscribers.pop(key, None)
        if sub:
            await sub.close()

    async def publish(self, message: dict, audience: str = "all") -> None:
        """Send ``message`` to every permitted connection of ``audience`` (``all``, ``primary`` or ``sideband``).

        The message is copied once, and that copy goes to every connection: a connection only sends it.
        """
        copy = None
        for sub in tuple(self.subscribers.values()):
            if audience != "all" and sub.source != audience:
                continue
            if server_event_allowed(self.config(), self.transport, message, sub.source):
                if copy is None:
                    copy = deepcopy(message)
                sub.put(copy)

    async def send_to(self, key: str, message: dict) -> None:
        """Send ``message`` to the connection ``key`` only, if it may see it; nothing once it is detached."""
        sub = self.subscribers.get(key)
        if sub is not None and server_event_allowed(self.config(), self.transport, message, sub.source):
            sub.put(deepcopy(message))

    async def wait_for_room(self) -> None:
        """Wait while the primary connection's queue is more than half full; at once without a primary."""
        for sub in tuple(self.subscribers.values()):
            if sub.source == "primary":
                await sub.room.wait()

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
