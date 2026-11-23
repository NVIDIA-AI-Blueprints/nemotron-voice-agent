# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One IM session's backend LLM agent: its model conductor, and LLM workers, up to ``max_running`` tasks at once."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.conductors import SteerOpenWork
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.llm import LlmAdapter
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import (
    BackendRun,
    ControlContext,
    ControlRun,
    TaskContext,
)

from .conductor import LlmConductor

if TYPE_CHECKING:
    from .source import AgentSource


class AgentAdapter:
    """The backend of one IM session (``port.BackendAdapter``): a model conductor and the session's LLM workers."""

    def __init__(self, source: AgentSource):
        """Serve one IM session from ``source``; the conductor is made when the IM first asks for it."""
        self.source = source
        self.max_running: int = source.settings.max_running
        self.workers: LlmAdapter = source.workers.new()
        self._conductor: LlmConductor | None = None

    @staticmethod
    def rule(max_running: int) -> type[SteerOpenWork] | None:
        """Return the steer rule at ``max_running`` 1, where no model call places a request; else ``None``."""
        return SteerOpenWork if max_running == 1 else None

    def can_run_without_attachment(self) -> bool:
        """Whether every native tool effect is covered by the IM's attachment fence."""
        return True

    def conductor(self) -> LlmConductor | SteerOpenWork:
        """Return the session's conductor: the rule of :meth:`rule`, else one model conductor with its own memory."""
        rule = self.rule(self.max_running)
        if rule is not None:
            return rule()
        if self._conductor is None:
            source = self.source
            settings, model = source.settings.conductor, source.conductor_endpoint.model
            self._conductor = LlmConductor(
                source.conductor_provider, source.vocabulary, settings, model, self.max_running
            )
        return self._conductor

    def start(self, task: TaskContext) -> BackendRun:
        """Start one task as an ``LlmRun`` of the session's workers; no model call happens before its ``events()``."""
        return self.workers.start(task)

    def start_control(self, task: ControlContext, *, parent: BackendRun | None = None) -> ControlRun:
        """Interpret this task's control input with its worker, without a conductor or business sources."""
        return self.workers.start_control(task, parent=parent)

    def usage_identity(self, task: TaskContext) -> dict:
        """Identify the actual LLM worker rather than the conductor's model."""
        return {**self.workers.usage_identity(task), "backend": "llm-agent"}

    def release_idle(self) -> None:
        """Release nothing: it keeps no idle session."""

    async def close(self) -> None:
        """Close the workers' providers and the conductor's."""
        closing = [self.workers.close(), *([self._conductor.close()] if self._conductor is not None else [])]
        await asyncio.gather(*closing, return_exceptions=True)
