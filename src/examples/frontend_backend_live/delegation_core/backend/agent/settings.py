# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend LLM agent's settings: the ``backend.llm-agent`` block and every default.

:class:`AgentSettings` checks the block when the core is built, so an unknown key, a fraction where a whole number is
due, or an unknown name fails the build. The ``worker`` block takes the backend LLM's keys (``backend.llm``).
"""

from __future__ import annotations

from typing import Annotated

from pydantic import ConfigDict, Field

from examples.frontend_backend_live.delegation_core.backend.llm import LlmSettings
from examples.frontend_backend_live.delegation_core.backend.llm.settings import Provider
from examples.frontend_backend_live.delegation_core.backend.providers import ChatCompletionsBackend
from examples.frontend_backend_live.delegation_core.config import Seconds, Settings, Text, WholeNumber


class ConductorSettings(Settings):
    """How the agent's conductor model is called: one call per ``decide()``, each decision through a control tool."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="backend.llm-agent.conductor")

    provider: Provider = ChatCompletionsBackend  # chat-completions | openai-responses (providers.BACKENDS)
    model: Text | None = None  # a model of the catalog; None: the backend's default (the slot's)
    # A decision round that takes longer places nothing more; the IM's steer rule places its requests.
    timeout_seconds: Seconds = 20.0
    memory_tokens: WholeNumber = 8000  # the conductor's own memory of earlier decision rounds, kept or dropped whole
    max_output_tokens_per_request: WholeNumber = 256  # the round's output cap, for each new request it places
    # Sampling for a placement, a number of at least 0; None: the endpoint's own.
    temperature: Annotated[float, Field(strict=True, ge=0)] | None = 0.0
    reasoning_effort: Text | None = "none"  # the official reasoning.effort (none: thinking off); None: the model's own

    def describe(self) -> dict:
        """The settings as plain JSON, by name."""
        return {
            "provider": self.provider.name,
            "timeout_seconds": self.timeout_seconds,
            "memory_tokens": self.memory_tokens,
            "max_output_tokens_per_request": self.max_output_tokens_per_request,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
        }


class AgentSettings(Settings):
    """The backend LLM agent's settings, with every name already turned into what it names."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="backend.llm-agent")

    max_running: WholeNumber = 3  # tasks running at once in one IM session
    conductor: ConductorSettings = Field(default_factory=ConductorSettings)
    worker: LlmSettings = Field(default_factory=LlmSettings)  # the backend LLM's keys, for the workers
