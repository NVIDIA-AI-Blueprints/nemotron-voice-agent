# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded retries only at boundaries where no application action can repeat."""

import asyncio
import random
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

from loguru import logger
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

current_attempt: ContextVar[int] = ContextVar("provider_attempt", default=1)
RETRYABLE = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError, TimeoutError)
MAX_RETRY_AFTER_SECONDS = 3.0


def retry_delay(exc: Exception, attempt: int) -> float:
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
    role: str,
    observe: Callable[..., None] | None = None,
) -> Any:
    """Run ``call`` up to ``attempts`` times, retrying only transient provider errors.

    Args:
        call: Zero-argument coroutine function; one invocation is one attempt.
        attempts: Total attempts, at least one.
        role: Label for logs and the ``observe`` hook (``talker``, ``commentary``, ``thinker``).
        observe: Optional hook called with ``("provider.retry", **fields)`` before each retry.

    Returns:
        Whatever ``call`` returns.

    Raises:
        ValueError: ``attempts`` is not positive.
        Exception: The last error when it is not transient or attempts are exhausted.
    """
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(1, attempts + 1):
        token = current_attempt.set(attempt)
        try:
            return await call()
        except RETRYABLE as exc:
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
            if observe:
                observe("provider.retry", **fields)
        finally:
            current_attempt.reset(token)
        await asyncio.sleep(delay)
    raise ValueError("attempts must be positive")
