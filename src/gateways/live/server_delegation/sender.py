# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The server delegation's ordered sender: one session's requests reach its core in order, off the command lock."""

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
    offset_ms: int
    starts: bool
    typed_run: bool = False
    client_event_id: str | None = None
    # What the frontend stated about a delegation: its own metadata pairs, under the gateway's keys.
    metadata: dict[str, str] = field(default_factory=dict)
    # The delegation id once the core accepted the request, or the refusal; cancelled when its caller gave up before it
    # was sent, or when an earlier request of its batch was refused.
    outcome: asyncio.Future = field(default_factory=lambda: asyncio.get_running_loop().create_future())
    # Set once the frontend has the delegation id: no event of the request reaches the frontend before it.
    returned: asyncio.Event = field(default_factory=asyncio.Event)


class RequestSender:
    """Sends one session's requests to its delegation core, a batch at a time, in the order they were queued."""

    def __init__(self, send: Callable[[Request], Awaitable[bool]]):
        """Send each request through ``send``, which settles its outcome and returns whether the core accepted it."""
        self.send = send
        self.queue: asyncio.Queue[list[Request]] = asyncio.Queue()
        self.task: asyncio.Task | None = None
        self.batch: list[Request] = []
        self.closed = False

    def start(self) -> None:
        """Start sending."""
        self.task = asyncio.create_task(self._run(), name="request-sender")

    def put(self, *batch: Request) -> None:
        """Queue requests to be sent in this order; raise ``RuntimeError`` once the sender is closed."""
        if self.closed:
            raise RuntimeError("Session is closing")
        self.queue.put_nowait(list(batch))

    async def _run(self) -> None:
        while True:
            self.batch = batch = await self.queue.get()
            for position, request in enumerate(batch):
                if request.outcome.done():
                    continue
                try:
                    accepted = await self.send(request)
                except Exception as exc:
                    # The sender itself must not stop: every later request still goes out.
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
