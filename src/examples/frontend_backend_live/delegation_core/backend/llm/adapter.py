# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One IM session's backend LLM: the steer rule for its conductor, one task at a time, and one provider per model."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from examples.frontend_backend_live.delegation_core.backend.conductors import SteerOpenWork
from examples.frontend_backend_live.delegation_core.backend.port import Conductor, TaskContext, ToolSource
from examples.frontend_backend_live.delegation_core.backend.providers import Backend

from .run import LlmRun, Retryable

if TYPE_CHECKING:
    from .source import LlmSource


class LlmAdapter:
    """The backend of one IM session (``port.BackendAdapter``): one backend, never stacked."""

    max_running: ClassVar[int] = 1

    def __init__(self, source: LlmSource, *, tools: Sequence[ToolSource], retryable: Retryable):
        """Serve one IM session from ``source``, with its server tool sources and the failed rounds it tries again."""
        self.source, self.tools, self.retryable = source, tuple(tools), retryable
        self.providers: dict[str | None, Backend] = {}

    def conductor(self) -> Conductor:
        """Return the session's conductor: the steer rule, which places each request with no model call."""
        return SteerOpenWork()

    def provider(self, model: str | None) -> Backend:
        """Return this session's provider of ``model``, made at its first round; ``ValueError`` if it is unlisted."""
        if model not in self.providers:
            self.providers[model] = self.source.make_provider(self.source.endpoint(model))
        return self.providers[model]

    def start(self, task: TaskContext) -> LlmRun:
        """Return one task's run, an ``LlmRun``; no model call happens before its ``events()``."""
        return LlmRun(
            task, providers=self.provider, tools=self.tools, settings=self.source.settings, retryable=self.retryable
        )

    def release_idle(self) -> None:
        """Release nothing: it keeps no idle session."""

    async def close(self) -> None:
        """Close every provider the session made."""
        providers, self.providers = list(self.providers.values()), {}
        await asyncio.gather(*(provider.close() for provider in providers), return_exceptions=True)
