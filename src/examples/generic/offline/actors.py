# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe provider events and conservative bounds for incremental replay.

Bounds describe the selected observed-compute scheduling model. They are not
backend acknowledgements. A callback's events and its revised future bound are
published under one lock, so a scheduler cannot advance past an unpublished
callback. Integer nanoseconds retain tied events without sample rounding drift.
"""

import asyncio
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActorEvent:
    """One provider observation ready at an absolute logical position."""

    ready_ns: int
    kind: str
    payload: Any
    generation: str | None = None


@dataclass(frozen=True)
class ActorSnapshot:
    """Atomically drained events and a lower bound for subsequent observations."""

    events: tuple[ActorEvent, ...]
    lower_bound_ns: int | float
    done: bool


class EventActor:
    """Bridge one asynchronous provider producer to a logical-time scheduler."""

    def __init__(
        self,
        changed: asyncio.Event,
        *,
        lower_bound: Callable[[int], int | float],
        clock: Callable[[], int] = time.monotonic_ns,
        cancel: Callable[[], None] | None = None,
    ):
        """Bind notifications to the creating loop and own provider cleanup."""
        self.changed = changed
        self._loop = asyncio.get_running_loop()
        self._clock = clock
        self._lower_bound = lower_bound
        self._cancel = cancel
        self._lock = threading.RLock()
        self._events: deque[ActorEvent] = deque()
        self._done = False
        self._closing = False
        self._error: BaseException | None = None
        self._completion: Future = Future()
        self._worker: asyncio.Task | None = None
        self._observed_bound: int | float = 0

    def bind_worker(self, worker: asyncio.Task) -> None:
        """Retain the producer task so closure awaits its actual cleanup."""
        if self._worker is not None:
            raise RuntimeError("Actor already owns a producer task")
        self._worker = worker

    def _notify(self) -> None:
        if not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self.changed.set)

    def publish_batch(
        self,
        events: Iterable[ActorEvent] | Callable[[int], Iterable[ActorEvent]],
        *,
        update: Callable[[], None] | None = None,
    ) -> None:
        """Publish a complete callback before changing its future readiness bound.

        A callable builds events using a wall-clock observation sampled inside
        the publication lock. Producers whose bounds advance with elapsed wall
        time must use this form, rather than timestamping before acquiring the
        lock. Empty batches may advance progress without inventing text events.
        """
        with self._lock:
            if self._done:
                raise RuntimeError("Cannot publish after actor completion")
            batch = tuple(events(self._clock()) if callable(events) else events)
            if any(event.ready_ns < 0 or event.ready_ns < self._observed_bound for event in batch):
                raise ValueError("Provider event precedes its published readiness lower bound")
            self._events.extend(batch)
            if update is not None:
                update()
        self._notify()

    def finish(self, result: Any = None, *, error: BaseException | None = None) -> None:
        """Mark EOF only after the producer has enqueued every callback event."""
        with self._lock:
            if self._done:
                return
            self._done = True
            self._error = error
            if error is None:
                self._completion.set_result(result)
            else:
                self._completion.set_exception(error)
        self._notify()

    def drain(self) -> ActorSnapshot:
        """Return queued events and a bound from the same locked observation."""
        with self._lock:
            if self._error is not None:
                raise self._error
            bound = math.inf if self._done else self._lower_bound(self._clock())
            if bound < self._observed_bound or math.isnan(bound):
                raise ValueError("Actor readiness lower bound must be monotone")
            self._observed_bound = bound
            events = tuple(self._events)
            self._events.clear()
            return ActorSnapshot(events, bound, self._done)

    async def result(self) -> Any:
        """Await the provider's complete trace or its original exception."""
        return await asyncio.shield(asyncio.wrap_future(self._completion))

    async def aclose(self) -> None:
        """Cancel the RPC once and await the producer instead of abandoning it."""
        with self._lock:
            closing = self._closing
            self._closing = True
        cancel_error = None
        if not closing and self._cancel is not None and not self._done:
            try:
                self._cancel()
            except BaseException as error:
                cancel_error = error
        if self._worker is not None:
            # Errors remain available from drain()/result(). RPC cancellation
            # is expected during session cleanup, not an additional failure.
            close_cancelled = None
            while not self._worker.done():
                try:
                    await asyncio.shield(self._worker)
                except asyncio.CancelledError as error:
                    if not self._worker.cancelled():
                        close_cancelled = error
                except Exception:
                    break
            if close_cancelled is not None and cancel_error is None:
                cancel_error = close_cancelled
        if cancel_error is not None:
            raise cancel_error


ActorStream = EventActor
