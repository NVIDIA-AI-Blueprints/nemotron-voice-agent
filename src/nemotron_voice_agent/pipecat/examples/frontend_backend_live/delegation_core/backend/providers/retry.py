# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded retries, only where no application action can repeat: a round that delivered and dispatched nothing."""

import asyncio
import random
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from loguru import logger
from openai import APIConnectionError, InternalServerError, RateLimitError


class StreamErrorEvent(Exception):
    """An error event in a round's stream, with no HTTP status, such as an overloaded server's."""


# Errors that can clear up on their own (APITimeoutError is an APIConnectionError); any other 4xx would fail again.
TRANSIENT: tuple[type[BaseException], ...] = (
    APIConnectionError,
    RateLimitError,
    InternalServerError,
    StreamErrorEvent,
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)
MAX_RETRY_AFTER_SECONDS = 3.0


def retry_delay(exc: BaseException, attempt: int) -> float:
    """Return the backoff before the next attempt, honoring a short ``Retry-After``."""
    response = getattr(exc, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    if value:
        try:
            seconds: float | None = float(value)
        except ValueError:
            try:
                seconds = (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
            except (ValueError, TypeError):
                seconds = None
        if seconds is not None:
            return min(MAX_RETRY_AFTER_SECONDS, max(0.0, seconds))
    return (0.4 if attempt == 1 else 1.2) * random.uniform(0.75, 1.25)


async def back_off(exc: BaseException, attempt: int, *, role: str) -> None:
    """Log that ``attempt`` failed with ``exc`` and will be tried again, then wait its backoff."""
    delay = retry_delay(exc, attempt)
    fields = {
        "role": role,
        "attempt": attempt,
        "error_type": type(exc).__name__,
        "status_code": getattr(exc, "status_code", None),
        "delay_ms": round(delay * 1000, 3),
    }
    logger.warning(f"provider.retry {fields}")
    await asyncio.sleep(delay)


async def with_retries(
    call: Callable[[], Awaitable[Any]],
    *,
    attempts: int,
    transient: tuple[type[BaseException], ...] = TRANSIENT,
    role: str,
) -> Any:
    """Run ``call`` up to ``attempts`` times, trying again only after a ``transient`` error, and return its result."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(1, attempts + 1):
        try:
            return await call()
        except transient as exc:
            if attempt == attempts:
                raise
            failure = exc
        await back_off(failure, attempt, role=role)
