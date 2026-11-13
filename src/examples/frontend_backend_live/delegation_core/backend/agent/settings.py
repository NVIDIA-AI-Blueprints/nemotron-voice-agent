# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend LLM agent's settings: the ``backend.llm-agent`` block and the default of every key."""

from __future__ import annotations

from typing import Annotated

from pydantic import ConfigDict, Field

from examples.frontend_backend_live.delegation_core.backend.llm import LlmSettings
from examples.frontend_backend_live.delegation_core.config import Seconds, Settings, Text, WholeNumber


class ConductorSettings(Settings):
    """The ``backend.llm-agent.conductor`` block: how the conductor model is called."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="backend.llm-agent.conductor")

    # A model of the catalog; null: the backend's default model (backend.slot's).
    model: Text | None = None
    # How long a decision round may run; the IM's fallback rule places what it leaves unplaced.
    timeout_seconds: Seconds = 20.0
    # The conductor's memory of earlier decision rounds, in tokens; the oldest rounds leave it whole.
    memory_tokens: WholeNumber = 8000
    # The sampling temperature of a decision round; null: the endpoint's own.
    temperature: Annotated[float, Field(strict=True, ge=0)] | None = 0.0
    # "none" turns thinking off (the endpoint's enable_thinking, where its settings have one); null: the model's own.
    reasoning_effort: Text | None = "none"

    def describe(self) -> dict:
        """Return the settings as plain JSON, by name."""
        return {
            "timeout_seconds": self.timeout_seconds,
            "memory_tokens": self.memory_tokens,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
        }


class AgentSettings(Settings):
    """The ``backend.llm-agent`` block: how many tasks run at once, and the conductor's and the workers' settings."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="backend.llm-agent")

    max_running: WholeNumber = 3
    conductor: ConductorSettings = Field(default_factory=ConductorSettings)
    # The workers' backend.llm keys; their round time limits apply to the conductor's rounds too, and their
    # provider_retries to a conductor round with a transient error before its first event.
    worker: LlmSettings = Field(default_factory=LlmSettings)
