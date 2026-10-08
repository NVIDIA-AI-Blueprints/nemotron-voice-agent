# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A session's life: starting it, reporting usage and expiry while it runs, and closing it in order."""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING

from loguru import logger

from live.events import error_event
from live.protocol import ProtocolError, approx_tokens

if TYPE_CHECKING:
    from live.session.session import LiveProtocolSession

USAGE_INTERVAL_SECONDS = 5
CONTEXT_WINDOW_TOKENS = 128_000
ENGINE_CLOSE_TIMEOUT_SECONDS = 10


async def start(session: LiveProtocolSession, client_event_id: str | None) -> None:
    """Start the engine, then announce ``session.started``."""
    if session.status != "created":
        raise ProtocolError("Session already started", "session_already_started")
    session.status = "starting"
    session.started_at = time.monotonic()
    try:
        await session.engine.start(session)
    except Exception as exc:
        logger.exception(f"session={session.id} start failed: {exc}")
        await session.fail("The voice pipeline could not start. Check server model access and configuration.")
        await session.close("connection_lost")
        raise
    if session.status != "starting":
        # The session was closed while its engine was starting: close() could not stop what had not started yet.
        try:
            await session.engine.close()
        except Exception:
            logger.exception(f"session={session.id} engine cleanup after an early close failed")
        return
    session.status = "active"
    session.ready.set()
    session.started_at = time.monotonic()
    session.expires_at = int(time.time() + session.max_session_seconds)
    await session.emit("session.started", session=session.snapshot(), client_event_id=client_event_id)
    session.spawn(maintain(session), "maintenance")


def context_ratio(session: LiveProtocolSession) -> float:
    """Return how full the model's context is, from the engine's own count when it keeps one.

    An engine may provide ``context_tokens()`` (the tokens its frontend currently holds). Without it, the size of
    the session's startup configuration is the only figure the protocol layer has.
    """
    count = getattr(session.engine, "context_tokens", None)
    tokens = count() if callable(count) else None
    if tokens is None:
        tokens = approx_tokens(json.dumps(session.config.model_dump()))
    return min(tokens / CONTEXT_WINDOW_TOKENS, 1)


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
            session.spawn(session.close("expired"), "expiry-close")
            return


async def close(session: LiveProtocolSession, reason: str, client_event_id: str | None) -> None:
    """Stop the engine, emit ``session.closed``, drain it to each peer, then clean up."""
    async with session.close_lock:
        if session.status in {"closing", "closed"}:
            return
        session.status = "closing"
        session.ready.set()
        current = asyncio.current_task()
        pending = [task for task in session.tasks if task is not current]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        try:
            async with asyncio.timeout(ENGINE_CLOSE_TIMEOUT_SECONDS):
                await session.engine.close()
        except Exception:
            logger.exception(f"session={session.id} engine cleanup failed")
        for ack in session.timeline.discard():
            await session.publish(
                error_event(
                    ProtocolError("Session closed before append reached timeline", "session_closed"),
                    ack.get("client_event_id"),
                )
            )
        await session.emit(
            "session.closed",
            reason=reason,
            session=session.snapshot(),
            usage={"seconds": round(session.clock_ms() / 1000, 3)},
            client_event_id=client_event_id,
        )
        await session.fanout.drain()
        session.status = "closed"
        session.closed.set()
        if session.on_closed:
            await session.on_closed(session)
        await session.fanout.close()
