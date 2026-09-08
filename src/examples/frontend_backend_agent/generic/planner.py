# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""One-shot, JSON-only planner for the generic domain."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.nvidia.llm import NvidiaLLMService

from examples.frontend_backend_agent.src.planner import parse_plan_json
from examples.frontend_backend_agent.src.tools import ToolSpec, render_tool_block


class GenericPlanner(Protocol):
    """Planner boundary consumed by the generic backend."""

    async def plan(self, *, query: str, state: dict[str, Any]) -> dict[str, Any]:
        """Return one unexecuted JSON plan."""


class NvidiaGenericPlanner:
    """Ask the Thinker model for one bounded plan over allowlisted tools."""

    def __init__(
        self,
        *,
        llm: NvidiaLLMService,
        system_prompt: str,
        enabled_tools: Sequence[ToolSpec],
        max_tokens: int = 2048,
    ) -> None:
        """Bind the planner to one fixed prompt and allowlisted tool subset."""
        if not system_prompt.strip():
            raise ValueError("Generic Thinker requires a non-empty system prompt")
        enabled = tuple(enabled_tools)
        self._llm = llm
        self._system_prompt = f"{system_prompt.rstrip()}{render_tool_block(enabled)}"
        self._enabled_tools = tuple(spec.name for spec in enabled)
        self._max_tokens = max_tokens

    async def plan(self, *, query: str, state: dict[str, Any]) -> dict[str, Any]:
        """Return a parsed plan; the dispatcher remains the authority for validation."""
        now = datetime.now().astimezone()
        payload = {
            "untrusted_user_request": query,
            "enabled_tools": list(self._enabled_tools),
            "session_state": state,
            "runtime_context": {
                "local_datetime": now.isoformat(timespec="seconds"),
                "date": now.date().isoformat(),
                "timezone": str(now.tzinfo),
            },
        }
        context = LLMContext(
            [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        )
        raw = await self._llm.run_inference(context, max_tokens=self._max_tokens)
        if not raw:
            raise RuntimeError("Generic Thinker returned an empty plan")
        return parse_plan_json(raw)
