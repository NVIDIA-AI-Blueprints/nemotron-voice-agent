# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One IM session's backend LLM agent: its model conductor, and LLM workers, up to ``max_running`` tasks at once."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from delegation_core.backend.llm import LlmAdapter
from delegation_core.backend.port import BackendRun, TaskContext

from .conductor import LlmConductor

if TYPE_CHECKING:
    from .source import AgentSource


class AgentAdapter:
    """The backend of one IM session (``port.BackendAdapter``): one backend, never stacked.

    Its workers are the backend LLM's: the session's ``LlmAdapter`` (the session's tool sources, one provider per model,
    two attempts per round) starts an ``LlmRun`` for each task. The IM's scheduler runs at most ``max_running`` tasks
    at once, the tasks of every Work counted together.
    """

    def __init__(self, source: AgentSource):
        """Serve one IM session from ``source``; the conductor is made when the IM first asks for it."""
        self.source = source
        self.max_running: int = source.settings.max_running
        self.workers: LlmAdapter = source.workers.new()
        self._conductor: LlmConductor | None = None

    def conductor(self) -> LlmConductor:
        """The session's conductor, with its own memory; the IM asks once per session, and gets the same one again."""
        if self._conductor is None:
            source = self.source
            settings, model = source.settings.conductor, source.conductor_endpoint.model
            self._conductor = LlmConductor(source.conductor_provider, source.vocabulary, settings, model)
        return self._conductor

    def start(self, task: TaskContext) -> BackendRun:
        """Run one task: an ``LlmRun`` of the session's workers; no model call happens before its ``events()``."""
        return self.workers.start(task)

    async def close(self) -> None:
        """Close the workers' providers and the conductor's."""
        closing = [self.workers.close(), *([self._conductor.close()] if self._conductor is not None else [])]
        await asyncio.gather(*closing, return_exceptions=True)
