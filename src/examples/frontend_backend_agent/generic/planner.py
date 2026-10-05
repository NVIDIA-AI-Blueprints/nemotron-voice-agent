# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Bounded JSON planner for the generic domain."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.nvidia.llm import NvidiaLLMService

from examples.frontend_backend_agent.src.planner import parse_plan_json
from examples.frontend_backend_agent.src.runtime_context import (
    SessionClock,
    session_clock_from_instructions,
    session_runtime_fields,
)
from examples.frontend_backend_agent.src.stage_metrics import StageMetricsCoordinator, run_streamed_inference_result
from examples.frontend_backend_agent.src.tools import ToolSpec, render_tool_block, session_server_tools


@dataclass(frozen=True, slots=True)
class GenericPlannerSessionUpdate:
    """Prepared Thinker prompt and tool policy for one atomic session update."""

    session_instructions: str
    session_messages: tuple[dict[str, Any], ...]
    trusted_system_prompt: str
    enabled_tools: tuple[str, ...]
    client_tools: tuple[dict[str, Any], ...]
    #: The present as the session's instructions state it; None when they state none.
    session_clock: SessionClock | None = None


class PlanTruncatedError(RuntimeError):
    """The Thinker stopped at its token limit, so its plan is incomplete."""


class GenericPlanner(Protocol):
    """Planner boundary consumed by the generic backend."""

    async def plan(self, *, query: str, state: dict[str, Any], history: Any = None) -> dict[str, Any]:
        """Return one unexecuted JSON plan; ``history`` is the session's earlier delegations, if any."""


class NvidiaGenericPlanner:
    """Ask the Thinker model for one bounded plan over allowlisted tools."""

    def __init__(
        self,
        *,
        llm: NvidiaLLMService,
        system_prompt: str,
        enabled_tools: Sequence[ToolSpec],
        client_tools: Sequence[Mapping[str, Any]] = (),
        client_instructions: str = "",
        max_tokens: int = 2048,
        stage_metrics: StageMetricsCoordinator | None = None,
        model_name: str = "",
    ) -> None:
        """Bind trusted planner policy plus an exact client instruction layer."""
        if not system_prompt.strip():
            raise ValueError("Generic Thinker requires a non-empty trusted system prompt")
        self._llm = llm
        self._base_system_prompt = system_prompt.rstrip()
        self._server_specs = tuple(enabled_tools)
        self._max_tokens = max_tokens
        self._stage_metrics = stage_metrics
        self._model_name = model_name
        self._session_instruction_context = LLMContext([])
        #: Token usage of the most recent plan, for per-round logging.
        self.last_usage: dict[str, int] = {}
        initial = self.prepare_session_update(
            instructions=client_instructions,
            client_tools=client_tools,
        )
        self.commit_session_update(initial)

    @property
    def session_instruction_context(self) -> LLMContext:
        """Return the canonical context whose first message is session.instructions verbatim."""
        return self._session_instruction_context

    @staticmethod
    def render_session_instructions(instructions: str) -> list[dict[str, Any]]:
        """Render the exact client-owned Thinker instruction message."""
        return [{"role": "system", "content": instructions}]

    def prepare_session_update(
        self,
        *,
        instructions: str,
        client_tools: Sequence[Mapping[str, Any]],
    ) -> GenericPlannerSessionUpdate:
        """Build a detached Thinker prompt/tool snapshot without mutating live state."""
        if not isinstance(instructions, str):
            raise TypeError("Generic Thinker session instructions must be text")
        clients = tuple(copy.deepcopy(dict(tool)) for tool in client_tools)
        server_specs = session_server_tools(self._server_specs, clients)
        trusted_prompt = f"{self._base_system_prompt}{render_tool_block(server_specs, clients)}"
        enabled = tuple(spec.name for spec in server_specs) + tuple(str(tool["name"]) for tool in clients)
        return GenericPlannerSessionUpdate(
            session_instructions=instructions,
            session_messages=tuple(self.render_session_instructions(instructions)),
            trusted_system_prompt=trusted_prompt,
            enabled_tools=enabled,
            client_tools=clients,
            session_clock=session_clock_from_instructions(instructions),
        )

    @property
    def session_clock(self) -> SessionClock | None:
        """Return the present as the active session instructions state it."""
        return self._session_clock

    def snapshot_session_update(self) -> GenericPlannerSessionUpdate:
        """Capture the active Thinker prompt/tool snapshot for rollback."""
        return GenericPlannerSessionUpdate(
            session_instructions=self._session_instructions,
            session_messages=tuple(copy.deepcopy(self._session_instruction_context.get_messages())),
            trusted_system_prompt=self._system_prompt,
            enabled_tools=self._enabled_tools,
            client_tools=tuple(copy.deepcopy(self._client_tools)),
            session_clock=self._session_clock,
        )

    def commit_session_update(self, prepared: GenericPlannerSessionUpdate) -> None:
        """Install one previously prepared Thinker prompt/tool snapshot."""
        if not isinstance(prepared, GenericPlannerSessionUpdate):
            raise TypeError("Generic Thinker prompt update has an invalid receipt")
        messages = copy.deepcopy(list(prepared.session_messages))
        if messages != self.render_session_instructions(prepared.session_instructions):
            raise ValueError("Generic Thinker instructions are not verbatim")
        self._session_instruction_context.set_messages(messages)
        self._session_instructions = prepared.session_instructions
        self._system_prompt = prepared.trusted_system_prompt
        self._enabled_tools = prepared.enabled_tools
        self._client_tools = tuple(copy.deepcopy(prepared.client_tools))
        self._session_clock = prepared.session_clock

    def restore_session_update(self, snapshot: GenericPlannerSessionUpdate) -> None:
        """Restore an exact Thinker snapshot after a failed outer transaction."""
        self.commit_session_update(snapshot)

    async def plan(self, *, query: str, state: dict[str, Any], history: Any = None) -> dict[str, Any]:
        """Return a parsed plan; the dispatcher remains the authority for validation."""
        payload: dict[str, Any] = {"untrusted_user_request": query}
        if history is not None:
            payload["conversation_history"] = history
        payload.update(
            {
                "enabled_tools": list(self._enabled_tools),
                "session_state": state,
                "runtime_context": session_runtime_fields(self._session_clock),
            }
        )
        context = LLMContext(
            [
                *copy.deepcopy(self._session_instruction_context.get_messages()),
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        )
        call_id = str(state.get("active_call_id") or "unbound")
        attempt = int(state.get("planner_attempt") or 1)
        planning_round = int(state.get("planning_round") or 1)
        span = None
        if self._stage_metrics is not None:
            span = await self._stage_metrics.start_backend(
                call_id,
                model=self._model_name,
                attempt=attempt,
                planning_round=planning_round,
            )
        result = await run_streamed_inference_result(self._llm, context, span, max_tokens=self._max_tokens)
        self.last_usage = dict(result.usage_tokens)
        if result.truncated:
            raise PlanTruncatedError(f"Generic Thinker stopped at max_tokens={self._max_tokens}")
        if not result.text:
            raise RuntimeError("Generic Thinker returned an empty plan")
        return parse_plan_json(result.text)
