# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend model provider: one round of Responses events from a model API, read through one template.

A provider is a :class:`Backend` subclass with a ``name`` (in ``backend.llm.provider``), its round's events
(:meth:`Backend._events`), its ``transient_errors`` and the request settings it cannot honor
(:meth:`Backend.unsupported`). :meth:`Backend.stream` retries only the opening of a round, and only after a transient
error, so a refused request is never repeated. It limits the wait for the first event and the silence between events;
a provider yields ``None`` while its model works with nothing to pass on, which resets the silence limit. Reading stops
at the first terminal event. A provider keeps no conversation: every round is sent the whole history.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from openai import AsyncOpenAI

from delegation_core.backend.port import TERMINALS

from ...config import ModelEndpoint
from .retry import TRANSIENT, with_retries


@dataclass(frozen=True)
class ProviderLimits:
    """How a provider's round is opened and timed (the ``backend.llm`` keys of the same names)."""

    provider_retries: int = 2  # tries after the first, when opening a round fails with a transient error
    first_event_timeout_seconds: float = 45.0  # from the request to the round's first event
    idle_timeout_seconds: float = 30.0  # the longest silence between two events of a round


# The limits of a provider made without its adapter's settings (the backend.llm block's defaults).
DEFAULT_LIMITS = ProviderLimits()


class BrokenStreamError(RuntimeError):
    """A round's stream broke the Responses rules (no event, no terminal, a second ``response.created``).

    The adapter reads it like a failed attempt: nothing of the round was delivered or dispatched, so it may try the
    round once more.
    """


def make_client(endpoint: ModelEndpoint, limits: ProviderLimits) -> AsyncOpenAI:
    """The one OpenAI client maker of the providers: the endpoint's address and key, and no retries of its own.

    Its timeout is only a backstop above both of the template's limits, which time every round.
    """
    return AsyncOpenAI(
        api_key=endpoint.api_key or "not-needed",
        base_url=endpoint.base_url,
        timeout=limits.first_event_timeout_seconds + limits.idle_timeout_seconds,
        max_retries=0,
    )


class Backend:
    """A backend model provider: Responses-style events for one round of delegated work, with tools."""

    name: ClassVar[str] = ""  # its name in backend.llm.provider and in providers.BACKENDS
    # The errors worth another try when opening a round: they can clear up on their own.
    transient_errors: ClassVar[tuple[type[BaseException], ...]] = TRANSIENT

    def __init__(self, endpoint: ModelEndpoint, limits: ProviderLimits = DEFAULT_LIMITS):
        """Bind the model endpoint and the round's limits; nothing connects yet."""
        self.endpoint, self.limits = endpoint, limits

    @classmethod
    def unsupported(cls, settings: Mapping[str, Any]) -> list[tuple[str, str]]:
        """The request settings this provider cannot honor, as ``(param, message)``; none by default."""
        return []

    async def stream(self, payload: dict) -> AsyncIterator[dict]:
        """Yield the Responses events of one round, up to and including its first terminal (the template).

        Raises:
            TimeoutError: The first event, or a next one, did not come within its limit.
            BrokenStreamError: The round's stream ended before its first event.
            Exception: The provider's error that ended the round (after the transient ones were tried again).
        """
        events, event = await with_retries(
            lambda: self._open(payload),
            attempts=self.limits.provider_retries + 1,
            transient=self.transient_errors,
            role="backend",
        )
        try:
            while True:
                yield event
                if event.get("type") in TERMINALS:
                    return
                try:
                    event = await self._next(events)
                except StopAsyncIteration:
                    return  # no terminal: the adapter reads the round as broken
        finally:
            await events.aclose()

    async def _open(self, payload: dict) -> tuple[AsyncIterator[dict | None], dict]:
        """One try at opening a round: the provider's events and their first one, within the first-event limit."""
        events = self._events(payload)
        try:
            async with asyncio.timeout(self.limits.first_event_timeout_seconds):
                event = None
                while event is None:  # a sign of life is not the first event
                    event = await anext(events)
                return events, event
        except StopAsyncIteration:
            await events.aclose()
            raise BrokenStreamError("The backend's stream ended before its first event") from None
        except BaseException:
            await events.aclose()
            raise

    async def _next(self, events: AsyncIterator[dict | None]) -> dict:
        """The round's next event, each step within the silence limit; a sign of life (``None``) restarts the limit."""
        while True:
            async with asyncio.timeout(self.limits.idle_timeout_seconds):
                event = await anext(events)
            if event is not None:
                return event

    def _events(self, payload: dict) -> AsyncIterator[dict | None]:
        """The provider's own events for one round: ``response.created`` once the model answers, the items, a terminal.

        No retries and no limits here: :meth:`stream` adds them. ``payload`` is a Responses request: ``model``,
        ``instructions``, ``input`` (the whole history), ``tools``, ``tool_choice`` and the other settings the IM gives.
        A ``None`` says the model is working with nothing to pass on yet; it restarts the silence limit only.
        """
        raise NotImplementedError(f"{type(self).__name__} gives no events")

    async def close(self) -> None:
        """Release the provider's connections."""
