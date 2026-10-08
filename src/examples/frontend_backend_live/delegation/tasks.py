# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a delegation is: the settings every backend round shares, and each task the caller's requests create."""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field

from live.protocol import SessionConfig

RESPONSES, CLIENT = "responses", "client"


@dataclass(frozen=True)
class BackendTask:
    """What every backend round shares: model, instructions and tools."""

    model: str
    instructions: str
    tools: list[dict]
    tool_choice: str | dict = "auto"
    max_output_tokens: int | None = None
    # ``reasoning``, ``text``, ``service_tier`` and ``parallel_tool_calls``, when the client set them. The Responses
    # backend applies them; a chat-completions backend has no equivalent and ignores them.
    options: dict = field(default_factory=dict)

    def payload(self) -> dict:
        """Return the Responses-style request skeleton for one round."""
        payload = {
            "model": self.model,
            "instructions": self.instructions,
            "tools": deepcopy(self.tools),
            "tool_choice": self.tool_choice,
        }
        if self.max_output_tokens:
            payload["max_output_tokens"] = self.max_output_tokens
        payload.update(deepcopy(self.options))
        return payload


@dataclass
class DelegatedTask:
    """One delegation: the caller's request as input items, and its progress."""

    id: str
    input: list[dict]
    round: int = 0
    emitted_function: bool = False
    queued_at: float = field(default_factory=time.monotonic)


def backend_task_for(config: SessionConfig, fallback_model: str, default_instructions: str) -> BackendTask | None:
    """The backend settings a session's ``delegation.responses`` block describes (``None`` in client mode).

    The block's ``instructions`` win; without them the backend runs on ``default_instructions``, the static
    prompt ``config.yaml`` names for it.
    """
    if config.mode != RESPONSES:
        return None
    responses = config.delegation.responses
    return BackendTask(
        model=responses.model or fallback_model,
        instructions=responses.instructions or default_instructions,
        tools=responses.tools,
        tool_choice=responses.tool_choice,
        max_output_tokens=responses.max_output_tokens,
        options=responses.model_dump(
            include={"parallel_tool_calls", "reasoning", "text", "service_tier"},
            exclude_defaults=True,
            exclude_none=True,
        ),
    )
