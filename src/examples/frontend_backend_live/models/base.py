# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The two model roles of the cascade, as base classes a provider implements.

* ``Frontend``: the talker. One fast call per turn that returns a ``TalkerDecision`` (``speak`` or
  ``delegate``), and the commentary call that phrases an update for the caller.
* ``Backend``: the thinker. A Responses-style event stream for delegated work, with tools.

Adding a provider is adding a subclass and registering it by name (``registry.py``), then naming it
in ``config.yaml``. Prompts, retries, the no-silence floor and the guards live outside these classes,
so every provider gets them unchanged.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import ClassVar

from examples.frontend_backend_live.common.retry import with_retries

from ..config_manager.schema import ModelEndpoint, ReliabilityConfig
from .decision import TalkerDecision

ROUTING_ACTIONS = ("speak", "delegate")


class Frontend(ABC):
    """The talker model. Subclasses implement ``complete_once``; retries and timeouts are shared."""

    name: ClassVar[str] = ""

    def __init__(self, endpoint: ModelEndpoint, reliability: ReliabilityConfig):
        """Bind the model endpoint and the retry policy."""
        self.endpoint, self.reliability = endpoint, reliability

    async def complete(
        self,
        instructions: str,
        history: list[dict],
        actions: tuple[str, ...] = ROUTING_ACTIONS,
        max_output_tokens: int = 400,
    ) -> TalkerDecision:
        """Return one complete decision, retried on transient failures (nothing is emitted until done)."""
        return await self.retried(
            lambda: self.complete_once(instructions, history, actions, max_output_tokens), actions
        )

    async def retried(self, call: Callable[[], Awaitable[TalkerDecision]], actions: tuple[str, ...]) -> TalkerDecision:
        """Run ``call`` under the per-attempt timeout and the retry policy."""

        async def attempt() -> TalkerDecision:
            async with asyncio.timeout(self.reliability.talker_attempt_timeout_seconds):
                return await call()

        return await with_retries(
            attempt,
            attempts=self.reliability.provider_retries + 1,
            role="talker" if "delegate" in actions else "commentary",
        )

    @abstractmethod
    async def complete_once(
        self,
        instructions: str,
        history: list[dict],
        actions: tuple[str, ...],
        max_output_tokens: int,
    ) -> TalkerDecision:
        """Make a single attempt. Raise on failure; the caller retries transient errors."""

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Release the provider's connections."""


class Backend(ABC):
    """The thinker model: Responses-style events for a delegated task."""

    name: ClassVar[str] = ""
    # True when the provider keeps the conversation (``previous_response_id``); a stateless chat
    # API is sent the delegate's complete history every round instead.
    stateful: ClassVar[bool] = True

    def __init__(self, endpoint: ModelEndpoint, reliability: ReliabilityConfig):
        """Bind the model endpoint and the retry policy."""
        self.endpoint, self.reliability = endpoint, reliability

    def validate_model(self, model: str) -> None:  # noqa: B027 - optional hook
        """Raise ``ValueError`` if this backend cannot serve ``model``; the default accepts any model."""

    @abstractmethod
    def stream(self, payload: dict) -> AsyncIterator[dict]:
        """Yield ``response.*`` events for one backend round (an async generator)."""

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Release the provider's connections."""
