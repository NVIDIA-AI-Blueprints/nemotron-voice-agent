# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend agent: a resumable tool-calling turn.

``step()`` is a state machine rather than a blocking loop, because a suspended
turn has to survive across the public ``send()`` boundary when the caller (not
this process) executes the tools.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from examples.frontend_backend_verdict.text import events
from examples.frontend_backend_verdict.text.events import EventSink, InternalEvent
from examples.frontend_backend_verdict.text.history import History
from examples.frontend_backend_verdict.text.llm import ChatClient
from examples.frontend_backend_verdict.text.messages import Message, ToolCall, ToolResult, UsageTotals
from examples.frontend_backend_verdict.text.protocol import ensure_ids
from examples.frontend_backend_verdict.text.tools import ToolRegistry

ERROR_TEXT = "I could not complete that request right now. Please try again."
ITERATION_CAP_TEXT = "I could not finish that after several steps. Could you try rephrasing the request?"


@dataclass(frozen=True, slots=True)
class Query:
    """Backend input: a fresh request."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolResults:
    """Backend input: results for the outstanding tool calls."""

    results: tuple[ToolResult, ...]


BackendInput = Query | ToolResults


@dataclass(frozen=True, slots=True)
class NeedsTools:
    """Backend output: tool calls must be executed before the turn continues."""

    tool_calls: tuple[ToolCall, ...]


@dataclass(frozen=True, slots=True)
class Final:
    """Backend output: the user-facing answer for this turn."""

    text: str


BackendStep = NeedsTools | Final


class BackendAgent:
    """Runs the task: owns every tool, never speaks except through its final text."""

    def __init__(
        self,
        *,
        client: ChatClient,
        system_prompt: str,
        registry: ToolRegistry,
        sink: EventSink,
    ) -> None:
        """Bind the backend to its client, prompt, tools and event sink."""
        self._client = client
        self._system_prompt = system_prompt
        self._registry = registry
        self._sink = sink

    async def step(
        self, inp: BackendInput, history: History, *, session_id: str
    ) -> tuple[BackendStep, History, UsageTotals]:
        """Advance the turn by one LLM call."""
        history = self._apply_input(inp, history)
        totals = UsageTotals()
        try:
            response = await self._client.complete(
                messages=[Message.system(self._system_prompt), *history.messages],
                tools=self._registry.schemas() or None,
            )
        except Exception as exc:  # noqa: BLE001 - a transport failure must not escape a turn
            self._sink.emit(
                InternalEvent(events.BACKEND_ERROR, session_id, {"error": str(exc), "type": type(exc).__name__})
            )
            return Final(text=ERROR_TEXT), history.append(Message.assistant(ERROR_TEXT)), totals
        totals = totals.add_call("backend", usage=response.usage, latency_ms=response.latency_ms, cost=response.cost)
        if response.tool_calls:
            calls = ensure_ids(response.tool_calls)
            history = history.append(Message.assistant_tool_calls(calls))
            self._sink.emit(
                InternalEvent(
                    events.BACKEND_TOOL_CALLS,
                    session_id,
                    {"calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in calls]},
                )
            )
            return NeedsTools(tool_calls=calls), history, totals
        text = (response.content or "").strip() or ERROR_TEXT
        self._sink.emit(InternalEvent(events.BACKEND_FINAL, session_id, {"text": text}))
        return Final(text=text), history.append(Message.assistant(text)), totals

    def force_final(self, history: History, *, session_id: str, reason: str) -> tuple[Final, History]:
        """End a turn without another LLM call (used for the iteration cap)."""
        self._sink.emit(InternalEvent(events.ITERATION_CAP, session_id, {"reason": reason}))
        return Final(text=ITERATION_CAP_TEXT), history.append(Message.assistant(ITERATION_CAP_TEXT))

    @staticmethod
    def _apply_input(inp: BackendInput, history: History) -> History:
        if isinstance(inp, Query):
            return history.append(Message.user(inp.text))
        return history.extend(Message.tool(result.tool_call_id, result.content) for result in inp.results)


def outstanding_ids(calls: Sequence[ToolCall]) -> tuple[str, ...]:
    """Return the ids of a tool-call batch, in emission order."""
    return tuple(call.id for call in calls)
