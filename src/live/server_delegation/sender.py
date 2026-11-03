# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The server delegation's ordered sender: one session's requests reach its core in order, off the command lock.

Requests come from the frontend adapter's ``delegate()`` and from the client's ``response.create`` (a continuation for
each delegation its outputs answer, or a typed run). The sender settles each :class:`Request`'s ``outcome``: the
delegation id once the core accepts it, or the refusal, which comes before any event. A ``response.create`` queues its
requests as one batch, and the first refusal cancels the requests after it. The sender never waits on a caller: a
``delegate()`` whose caller gave up before its request was sent is skipped, and one that gave up while it was sent
still has its stream read, so no accepted request is left unread.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class Request:
    """One request a session sends its delegation core: a delegation, a continuation or a typed run."""

    input: list[dict]
    delegation_id: str
    offset_ms: int  # where the request was made on the session's input clock
    starts: bool  # it starts a delegation (a delegation or a typed run), so its first response announces it
    typed_run: bool = False  # the client's typed messages, or nothing, run as a delegation of its own
    client_event_id: str | None = None  # the response.create that made it
    # Settled by the sender: the delegation id once the core accepted the request, or the refusal; cancelled when its
    # caller gave up before it was sent, or when an earlier request of its batch was refused.
    outcome: asyncio.Future = field(default_factory=lambda: asyncio.get_running_loop().create_future())
    # Set once the frontend has the delegation id: no event of the request reaches the frontend before it.
    returned: asyncio.Event = field(default_factory=asyncio.Event)


class RequestSender:
    """Sends one session's requests to its delegation core, a batch at a time, in the order they were queued."""

    def __init__(self, send: Callable[[Request], Awaitable[bool]]):
        """``send(request)`` sends one request, settles its outcome, and says whether the core accepted it."""
        self.send = send
        self.queue: asyncio.Queue[list[Request]] = asyncio.Queue()
        self.task: asyncio.Task | None = None
        self.batch: list[Request] = []  # the batch being sent
        self.closed = False

    def start(self) -> None:
        """Start sending."""
        self.task = asyncio.create_task(self._run(), name="request-sender")

    def put(self, *batch: Request) -> None:
        """Queue requests to be sent in this order; from the close on, each is refused instead.

        Raises:
            RuntimeError: The session is closing.
        """
        if self.closed:
            raise RuntimeError("Session is closing")
        self.queue.put_nowait(list(batch))

    async def _run(self) -> None:
        while True:
            self.batch = batch = await self.queue.get()
            for position, request in enumerate(batch):
                if request.outcome.done():  # its caller gave up before it was sent: nothing is sent
                    continue
                try:
                    accepted = await self.send(request)
                except Exception as exc:  # the sender itself must not stop: every later request still goes out
                    logger.exception(f"The request of delegation {request.delegation_id} could not be sent: {exc}")
                    accepted = False
                    if not request.outcome.done():
                        request.outcome.set_exception(exc)
                if not accepted:
                    for later in batch[position + 1 :]:
                        later.outcome.cancel()
                    break

    async def close(self) -> None:
        """Send nothing more: every request not yet accepted is refused as the session closes. Safe twice."""
        self.closed = True
        if self.task is not None:
            self.task.cancel()
            await asyncio.wait({self.task})
        unsent = [self.batch]
        while not self.queue.empty():
            unsent.append(self.queue.get_nowait())
        for batch in unsent:
            for request in batch:
                if not request.outcome.done():
                    request.outcome.set_exception(RuntimeError("Session is closing"))
