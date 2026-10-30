# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The frontend model role of the cascade, as a base class a provider implements.

* ``Frontend``: the cascade's own model, which the talker, the writer and the normalizer call. Each call states what it
  is in a :class:`CallSpec` (its role, the actions its decision may take, its schema hint), and gets one
  ``TalkerDecision``.
* ``Backend``, the backend model provider, is in the delegation core (``delegation_core.backend.providers``).

Adding a provider is one subclass with its ``name``, one entry in ``FRONTENDS`` (``models/__init__.py``), and the name
in ``config.yaml``. Prompts, retries and the talker's floor live outside these classes, so every provider
gets them unchanged. Every call's output limit is the slot's ``max_tokens``, so the settings decide it, not the code.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import ClassVar

from examples.frontend_backend_live.models.retry import with_retries

from ..config_manager.schema import ModelEndpoint, ReliabilityConfig
from .decision import TalkerDecision

# A call's output limit when the slot's settings give none.
DEFAULT_MAX_OUTPUT_TOKENS = 512


@dataclass(frozen=True)
class CallSpec:
    """What one model call is, stated by its caller: nothing about a call is inferred from its actions."""

    role: str  # who calls: talker, writer or normalizer; it names the call in the logs
    actions: tuple[str, ...]  # the actions the decision may take: the JSON schema's enum
    hint: str  # the description of ``speech`` in the JSON schema
    attempts: int | None = None  # tries of the call; None: the reliability setting's (provider_retries + 1)


class Frontend(ABC):
    """The cascade's model. Subclasses implement ``complete_once``; retries and timeouts are shared."""

    name: ClassVar[str] = ""  # the provider's name in config.yaml and in FRONTENDS
    # Text the provider adds to every call's instructions itself, so a view of the request (/history) can show it.
    reply_format: ClassVar[str] = ""

    def __init__(self, endpoint: ModelEndpoint, reliability: ReliabilityConfig, client=None):
        """Bind the model endpoint and the retry policy; each provider builds its own client when ``client`` is None."""
        self.endpoint, self.reliability = endpoint, reliability
        # The slot's ``max_tokens`` (settings.yaml writes it into ``extra_params``) is every call's output limit.
        self.max_output_tokens = int(endpoint.extra_params.get("max_tokens") or DEFAULT_MAX_OUTPUT_TOKENS)

    async def complete(self, instructions: str, history: list[dict], spec: CallSpec) -> TalkerDecision:
        """Return one complete decision, retried on transient failures (nothing is emitted until done)."""
        return await self.retried(lambda: self.complete_once(instructions, history, spec), spec)

    async def retried(self, call: Callable[[], Awaitable[TalkerDecision]], spec: CallSpec) -> TalkerDecision:
        """Run ``call`` under the per-attempt timeout and the call's retry policy."""

        async def attempt() -> TalkerDecision:
            async with asyncio.timeout(self.reliability.talker_attempt_timeout_seconds):
                return await call()

        return await with_retries(
            attempt,
            attempts=spec.attempts or self.reliability.provider_retries + 1,
            role=spec.role,
        )

    @abstractmethod
    async def complete_once(self, instructions: str, history: list[dict], spec: CallSpec) -> TalkerDecision:
        """Make a single attempt. Raise on failure; the caller retries transient errors."""

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Release the provider's connections."""
