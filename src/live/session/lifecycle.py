# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A session's life: starting it, reporting usage and expiry while it runs, and closing it in order.

A start opens the delegation part (the server delegation opens the session's conversation), then starts the frontend
adapter, each with its own view of the session. A close has five steps, which the parts' docstrings cite by number:
(1) the session is ``closing``; (2) the delegation part stops handing (no new request, nothing more to the frontend);
(3) the session's other tasks stop; (4) the frontend adapter closes, then, always, the delegation part (the server
delegation deletes the conversation), each within its own limit; (5) the last steps (``session.closed``, the drain, the
closed mark, ``on_closed``, the fan-out's close) run whatever happened before them.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING

from loguru import logger

from live.ports import ContextMeter
from live.protocol import ProtocolError, approx_tokens

if TYPE_CHECKING:
    from live.session.session import LiveProtocolSession

USAGE_INTERVAL_SECONDS = 5
CONTEXT_WINDOW_TOKENS = 128_000
# The frontend adapter's close limit. The delegation part gives its own (``DelegationPart.close_seconds``): the server
# delegation's covers its core's close (running work's grace, then the stop wait) and the relay of the ends it writes.
FRONTEND_CLOSE_SECONDS = 10


async def start(session: LiveProtocolSession, client_event_id: str | None) -> None:
    """Open the delegation part, start the frontend adapter, then announce ``session.started``.

    A conversation that the delegation core cannot create fails the start of a ``responses`` session, as an adapter
    that cannot start does. The settings the session started with are then checked in the background, as an update
    that changes nothing: ``session.started`` does not wait for that check.
    """
    if session.status != "created":
        raise ProtocolError("Session already started", "session_already_started")
    session.status = "starting"
    session.started_at = time.monotonic()
    try:
        await session.delegation.open(session.delegation_view)
        if session.status == "starting":  # a session closed while its delegation part opened starts no frontend
            await session.frontend.start(session.adapter_view, session.delegation_port)
    except Exception as exc:
        logger.exception(f"session={session.id} start failed: {exc}")
        await session.report_error("The voice pipeline could not start. Check server model access and configuration.")
        await session.close("connection_lost", asked_by="the gateway: the start failed")
        raise
    if session.status != "starting":
        # The session was closed while its parts were starting: close() could not stop what had not started yet.
        await close_parts(session)
        return
    session.status = "active"
    session.ready.set()
    session.started_at = time.monotonic()
    session.expires_at = int(time.time() + session.max_session_seconds)
    await session.emit("session.started", session=session.snapshot(), client_event_id=client_event_id)
    session.config = await session.delegation.update(session.config, {}, client_event_id)
    if session.config.client is not None and session.config.client.data_channel.restricted:
        await session.emit(
            "info",
            code="data_channel_permissions",
            message="The frontend data channel is configured with restricted event permissions.",
        )
    session.spawn(maintain(session), "maintenance")


def context_ratio(session: LiveProtocolSession) -> float:
    """Return how full the model's context is, from the frontend adapter's own count when it keeps one.

    A frontend adapter may be a :class:`~live.ports.ContextMeter`: the tokens its frontend currently holds, out of its
    context window. Without a count, the size of the session's startup configuration is the only figure the protocol
    layer has; without a window, :data:`CONTEXT_WINDOW_TOKENS`.
    """
    meter = session.frontend if isinstance(session.frontend, ContextMeter) else None
    tokens = meter.context_tokens() if meter is not None else None
    if tokens is None:
        tokens = approx_tokens(json.dumps(session.config.model_dump()))
    window = (meter.context_window_tokens() if meter is not None else None) or CONTEXT_WINDOW_TOKENS
    return min(tokens / window, 1)


async def maintain(session: LiveProtocolSession) -> None:
    """Report usage every few seconds and close the session when it reaches its lifetime."""
    while True:
        await asyncio.sleep(USAGE_INTERVAL_SECONDS)
        elapsed = session.clock_ms() / 1000
        await session.emit(
            "session.usage.updated",
            usage={"seconds": round(elapsed, 3)},
            context_window={"usage_ratio": context_ratio(session)},
        )
        if elapsed >= session.max_session_seconds:
            session.spawn(session.close("expired", asked_by="the gateway: the session's lifetime ended"), "expiry")
            return


async def close(session: LiveProtocolSession, reason: str, client_event_id: str | None, asked_by: str) -> None:
    """Close both parts, emit ``session.closed``, drain it to each peer, then clean up.

    The close runs as its own task, and every caller waits for that one task. A caller that is cancelled while it
    waits (a client that leaves mid-close, a server shutting down) cannot cut the order short and leave the
    session ``closing``. The first caller's ``reason`` and ``asked_by`` are the close's.
    """
    if session.close_task is None:
        session.status = "closing"
        session.ready.set()
        session.close_task = asyncio.create_task(
            _close_in_order(session, reason, client_event_id, asked_by, caller=asyncio.current_task()),
            name=f"{session.id}:close",
        )
    if session.close_task is not asyncio.current_task():
        await asyncio.shield(session.close_task)


async def close_parts(session: LiveProtocolSession) -> None:
    """Close the frontend adapter, then, always, the delegation part, each within its own time limit.

    The server delegation's close deletes the session's conversation, and its readers relay the ends the core writes;
    the client delegation's forgets the ids it announced. A part whose close fails or hangs is logged, and the
    delegation part still closes, so a hung frontend adapter never leaves the core's work running.
    """
    try:
        await _close_part(session, session.frontend, "frontend adapter", FRONTEND_CLOSE_SECONDS)
    finally:
        await _close_part(session, session.delegation, "delegation part", session.delegation.close_seconds)


async def _close_part(session: LiveProtocolSession, part, name: str, limit: float) -> None:
    try:
        async with asyncio.timeout(limit):
            await part.close()
    except TimeoutError:
        logger.error(f"session={session.id} {name} did not close within {limit} s")
    except Exception:
        logger.exception(f"session={session.id} {name} cleanup failed")


async def _close_in_order(
    session: LiveProtocolSession,
    reason: str,
    client_event_id: str | None,
    asked_by: str,
    caller: asyncio.Task | None,
) -> None:
    logger.info(
        f"session={session.id} closes: reason={reason}, lifetime={time.time() - session.created_at:.1f} s, "
        f"asked by {asked_by}"
    )
    try:
        # The delegation part sends no new request and hands nothing more to the frontend; the client still gets what
        # the core writes until the part's close.
        try:
            session.delegation.stop_handing()
        except Exception:
            logger.exception(f"session={session.id} delegation part could not stop handing events")
        # The task that asked to close keeps running; the session's other tasks stop.
        pending = [task for task in session.tasks if task is not caller]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.wait(pending)
        await close_parts(session)
    finally:
        await _finish(session, reason, client_event_id)


async def _finish(session: LiveProtocolSession, reason: str, client_event_id: str | None) -> None:
    """The last steps of a close, each in its own guard, so none can leave the session stuck in ``closing``."""

    async def closed_event() -> None:
        await session.emit(
            "session.closed",
            reason=reason,
            session=session.snapshot(),
            usage={"seconds": round(session.clock_ms() / 1000, 3)},
            client_event_id=client_event_id,
        )

    async def mark_closed() -> None:
        session.status = "closed"
        session.closed.set()

    async def on_closed() -> None:
        if session.on_closed:
            await session.on_closed(session)

    for name, step in (
        ("session.closed", closed_event),
        ("drain", session.fanout.drain),
        ("mark closed", mark_closed),
        ("on_closed", on_closed),
        ("fan-out close", session.fanout.close),
    ):
        try:
            await step()
        except Exception:
            logger.exception(f"session={session.id} close step {name!r} failed")
