# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded retries, only at boundaries where no application action can repeat: opening a provider's round."""

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
    """A round's stream opened with an error event and no HTTP status, such as an overloaded server's."""


# Errors that can clear up on their own: connection failures and client timeouts (APITimeoutError is an
# APIConnectionError), 429, 5xx, an error event that opens a stream, and httpx's network and protocol errors; any other
# 4xx would fail again.
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


async def with_retries(
    call: Callable[[], Awaitable[Any]],
    *,
    attempts: int,
    transient: tuple[type[BaseException], ...] = TRANSIENT,
    role: str,
) -> Any:
    """Run ``call`` up to ``attempts`` times, trying again only after a ``transient`` error.

    Args:
        call: Zero-argument coroutine function; one invocation is one attempt.
        attempts: Total attempts, at least one.
        transient: The errors worth another attempt; any other error is raised at once.
        role: Label for logs (``backend``).

    Returns:
        Whatever ``call`` returns.

    Raises:
        ValueError: ``attempts`` is not positive.
        Exception: The last error when it is not transient or attempts are exhausted.
    """
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(1, attempts + 1):
        try:
            return await call()
        except transient as exc:
            if attempt == attempts:
                raise
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
