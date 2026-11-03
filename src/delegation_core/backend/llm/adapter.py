# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One IM session's backend LLM: its rule conductor, one task at a time, and one provider per model."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from delegation_core.backend.port import Conductor, TaskContext, ToolSource
from delegation_core.backend.providers import Backend

from .run import LlmRun

if TYPE_CHECKING:
    from .source import LlmSource


class LlmAdapter:
    """The backend of one IM session (``port.BackendAdapter``): one backend, never stacked."""

    # One task at a time, and one task per Work: the IM's scheduler starts the next when it ends.
    max_running: ClassVar[int] = 1

    def __init__(self, source: LlmSource, *, tools: Sequence[ToolSource], attempts: int):
        """Serve one IM session from ``source``, with its server tool sources and tries per round."""
        self.source, self.tools, self.attempts = source, tuple(tools), attempts
        self.providers: dict[str | None, Backend] = {}

    def conductor(self) -> Conductor:
        """The session's conductor: the rule ``backend.llm.conductor`` names."""
        return self.source.settings.conductor()

    def provider(self, model: str | None) -> Backend:
        """The provider of ``model`` in this session, made on its first round (one per model).

        Raises:
            ValueError: The catalog does not list ``model`` and ``unlisted_models`` reports it.
        """
        if model not in self.providers:
            self.providers[model] = self.source.make_provider(self.source.endpoint(model))
        return self.providers[model]

    def start(self, task: TaskContext) -> LlmRun:
        """Run one task: an ``LlmRun``; no model call happens before its ``events()``."""
        return LlmRun(
            task, providers=self.provider, tools=self.tools, settings=self.source.settings, attempts=self.attempts
        )

    async def close(self) -> None:
        """Close every provider the session made."""
        providers, self.providers = list(self.providers.values()), {}
        await asyncio.gather(*(provider.close() for provider in providers), return_exceptions=True)
