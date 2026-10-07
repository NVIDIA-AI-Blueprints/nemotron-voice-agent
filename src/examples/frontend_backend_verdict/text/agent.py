# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composition root: routes a turn through the frontend and the backend.

One agent instance is immutable and reusable; all conversation state lives in
the :class:`~examples.frontend_backend_verdict.text.session.SessionState` the
caller passes in and gets back. Exactly one user-visible payload leaves each
step, and it is either the frontend's own answer or the backend's final text —
never the filler in between.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from examples.frontend_backend_verdict.text import events
from examples.frontend_backend_verdict.text.backend import (
    BackendAgent,
    BackendInput,
    Final,
    NeedsTools,
    Query,
    ToolResults,
    outstanding_ids,
)
from examples.frontend_backend_verdict.text.backend_context import (
    BackendRequest,
    backend_system_prompt,
    build_request,
    close_discarded_turn,
    request_template,
)
from examples.frontend_backend_verdict.text.config import Config, load_config
from examples.frontend_backend_verdict.text.errors import ToolProtocolError
from examples.frontend_backend_verdict.text.events import EventSink, InternalEvent, build_sink
from examples.frontend_backend_verdict.text.frontend import (
    ContractFallback,
    Decision,
    Delegate,
    DirectAnswer,
    FrontendAgent,
)
from examples.frontend_backend_verdict.text.history import History
from examples.frontend_backend_verdict.text.llm import ChatClient, OpenAIChatClient
from examples.frontend_backend_verdict.text.messages import AgentTurn, Message, ToolResult, UsageTotals
from examples.frontend_backend_verdict.text.prompts import load_catalog
from examples.frontend_backend_verdict.text.protocol import validate_tool_results
from examples.frontend_backend_verdict.text.session import PendingTurn, SessionState
from examples.frontend_backend_verdict.text.tools import InternalToolDriver, ToolRegistry, ToolSpec


@dataclass(frozen=True, slots=True)
class FrontendStep:
    """The frontend's decision for one user message, not yet carried out.

    ``session`` is the state the decision was made on; :meth:`FrontendBackendAgent.continue_turn`
    carries the decision out on exactly that state.
    """

    user_text: str
    session: SessionState
    decision: Decision
    totals: UsageTotals


class FrontendBackendAgent:
    """Frontend + backend composition with an explicit, caller-owned session."""

    def __init__(
        self,
        *,
        config: Config,
        backend: BackendAgent,
        registry: ToolRegistry,
        sink: EventSink,
        frontend: FrontendAgent | None = None,
        backend_request_template: str = "",
    ) -> None:
        """Wire the immutable parts of one agent."""
        self._config = config
        self._request_template = backend_request_template
        self._backend = backend
        self._frontend = frontend
        self._registry = registry
        self._sink = sink
        self._driver = InternalToolDriver(registry, parallel=config.backend.tools.parallel_execution)

    @property
    def config(self) -> Config:
        """The resolved configuration this agent was built from."""
        return self._config

    def new_session(self) -> SessionState:
        """Create a fresh session state."""
        return SessionState()

    async def send(self, text: str, session: SessionState) -> tuple[AgentTurn, SessionState]:
        """Process one user message and return the single outward payload."""
        if not self._config.frontend_enabled:
            session = self._resolve_pending_before_user_message(session)
            return await self._run_backend_only(text, session)
        return await self.continue_turn(await self.decide_turn(text, session))

    async def decide_turn(
        self,
        text: str,
        session: SessionState,
        *,
        in_progress: Mapping[str, Any] | None = None,
        confirmation: Mapping[str, Any] | None = None,
    ) -> FrontendStep:
        """Run only the frontend for one user message (paired mode).

        The session is not changed and no delegation, filler or backend event is
        emitted; :meth:`continue_turn` does both. ``in_progress`` is passed to
        :meth:`FrontendAgent.decide`; without it the request is exactly the one
        :meth:`send` makes. So is ``confirmation`` (a pending action summary).
        """
        assert self._frontend is not None  # noqa: S101 - guaranteed by build_agent for this mode
        session = self._resolve_pending_before_user_message(session)
        decision, totals = await self._frontend.decide(
            text,
            session.frontend_history,
            session_id=session.session_id,
            in_progress=in_progress,
            confirmation=confirmation,
        )
        return FrontendStep(user_text=text, session=session, decision=decision, totals=totals)

    async def continue_turn(self, step: FrontendStep) -> tuple[AgentTurn, SessionState]:
        """Carry out a frontend decision: finish a direct answer, or delegate and run the backend."""
        text, session, decision, totals = step.user_text, step.session, step.decision, step.totals
        if isinstance(decision, DirectAnswer):
            self._sink.emit(InternalEvent(events.DIRECT_ANSWER, session.session_id, {"text": decision.text}))
            return self._finish_direct(decision.text, text, session, totals)
        if isinstance(decision, ContractFallback):
            return self._finish_direct(decision.text, text, session, totals)
        self._emit_delegation(decision, session.session_id)
        request = build_request(
            self._config,
            template=self._request_template,
            query=decision.query,
            user_message=text,
            frontend_history=session.frontend_history,
            backend_history=session.backend_history,
        )
        self._emit_backend_context(request, session.session_id)
        return await self._drive(
            Query(text=request.text),
            request.history,
            session,
            user_message=text,
            delegation_query=decision.query,
            frontend_assistant=decision.assistant_message,
            totals=totals,
            iterations=0,
        )

    async def send_tool_results(
        self, results: Sequence[ToolResult], session: SessionState, *, user_message: Message | None = None
    ) -> tuple[AgentTurn, SessionState]:
        """Resume a suspended turn with caller-executed tool results.

        ``user_message`` is caller speech from while the calls were out. The
        backend gets it right after the tool messages, and the frontend history
        records it after the delegation's result when the turn closes.
        """
        pending = session.pending
        if pending is None:
            raise ToolProtocolError("no tool calls are outstanding")
        ordered = validate_tool_results(
            pending.outstanding, results, on_incomplete=self._config.backend.tools.on_incomplete_results
        )
        caller_inputs = pending.caller_inputs + ((user_message,) if user_message is not None else ())
        return await self._drive(
            ToolResults(results=ordered, user_message=user_message),
            pending.backend_history,
            session,
            user_message=pending.user_message,
            delegation_query=pending.delegation_query,
            frontend_assistant=pending.frontend_assistant,
            totals=UsageTotals(),
            iterations=pending.iterations,
            caller_inputs=caller_inputs,
        )

    def close_turn(
        self, results: Sequence[ToolResult], text: str, session: SessionState
    ) -> tuple[AgentTurn, SessionState]:
        """Close a suspended turn with ``results`` and the final ``text``, without another LLM call."""
        pending = session.pending
        if pending is None:
            raise ToolProtocolError("no tool calls are outstanding")
        ordered = validate_tool_results(
            pending.outstanding, results, on_incomplete=self._config.backend.tools.on_incomplete_results
        )
        history = pending.backend_history.extend(Message.tool(r.tool_call_id, r.content) for r in ordered).append(
            Message.assistant(text)
        )
        return self._finish_backend(
            text,
            session,
            pending.user_message,
            pending.frontend_assistant,
            history,
            UsageTotals(),
            pending.caller_inputs,
        )

    # -- turn routing -----------------------------------------------------

    async def _run_backend_only(self, text: str, session: SessionState) -> tuple[AgentTurn, SessionState]:
        return await self._drive(
            Query(text=text),
            session.backend_history,
            session,
            user_message=text,
            delegation_query="",
            frontend_assistant=None,
            totals=UsageTotals(),
            iterations=0,
        )

    async def _drive(
        self,
        first_input: BackendInput,
        history: History,
        session: SessionState,
        *,
        user_message: str,
        delegation_query: str,
        frontend_assistant: Message | None,
        totals: UsageTotals,
        iterations: int,
        caller_inputs: tuple[Message, ...] = (),
    ) -> tuple[AgentTurn, SessionState]:
        """Run the backend to a final answer, or suspend for the caller."""
        tools_config = self._config.backend.tools
        delay_s = self._config.backend.simulated_delay_s
        if delay_s > 0 and isinstance(first_input, Query):
            await asyncio.sleep(delay_s)  # testing aid: simulate a slow backend once per delegation
        current: BackendInput = first_input
        while True:
            step, history, step_totals = await self._backend.step(current, history, session_id=session.session_id)
            totals = totals.merge(step_totals)
            if isinstance(step, Final):
                return self._finish_backend(
                    step.text, session, user_message, frontend_assistant, history, totals, caller_inputs
                )
            iterations += 1
            if tools_config.execution == "external":
                pending = PendingTurn(
                    delegation_query=delegation_query,
                    user_message=user_message,
                    backend_history=history,
                    outstanding=outstanding_ids(step.tool_calls),
                    iterations=iterations,
                    frontend_assistant=frontend_assistant,
                    caller_inputs=caller_inputs,
                )
                session = replace(session, pending=pending, usage=session.usage.merge(totals))
                self._emit_usage(session, totals)
                return AgentTurn(tool_calls=step.tool_calls, usage=totals), session
            if iterations > tools_config.max_tool_iterations:
                final, history = self._backend.force_final(
                    history, session_id=session.session_id, reason=f"exceeded {tools_config.max_tool_iterations}"
                )
                return self._finish_backend(
                    final.text, session, user_message, frontend_assistant, history, totals, caller_inputs
                )
            current = ToolResults(results=await self._execute_batch(step, session))

    async def _execute_batch(self, step: NeedsTools, session: SessionState) -> tuple[ToolResult, ...]:
        results = await self._driver.run_batch(step.tool_calls)
        for call, result in zip(step.tool_calls, results, strict=True):
            self._sink.emit(
                InternalEvent(
                    events.TOOL_EXECUTED,
                    session.session_id,
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                        "result": result.content,
                        "is_error": result.is_error,
                    },
                )
            )
        return results

    # -- history closing --------------------------------------------------

    def _finish_direct(
        self, text: str, user_message: str, session: SessionState, totals: UsageTotals
    ) -> tuple[AgentTurn, SessionState]:
        """Close a non-delegated turn: user + assistant, exactly what was shown."""
        history = session.frontend_history.append(Message.user(user_message), Message.assistant(text)).prune(
            self._config.frontend.history.max_groups
        )
        session = replace(session, frontend_history=history, pending=None, usage=session.usage.merge(totals))
        self._emit_usage(session, totals)
        return AgentTurn(final_text=text, usage=totals), session

    def _finish_backend(
        self,
        text: str,
        session: SessionState,
        user_message: str,
        frontend_assistant: Message | None,
        backend_history: History,
        totals: UsageTotals,
        caller_inputs: tuple[Message, ...] = (),
    ) -> tuple[AgentTurn, SessionState]:
        """Close a backend-completed turn in whichever histories own it.

        Caller speech from while tools were out follows the delegation's result
        in the frontend history, as it followed the tool messages in the backend's.
        """
        if self._config.backend.stateful:
            session = replace(session, backend_history=backend_history.prune(self._config.backend.history.max_groups))
        if not self._config.frontend_enabled:
            session = replace(session, pending=None, usage=session.usage.merge(totals))
            self._emit_usage(session, totals)
            return AgentTurn(final_text=text, usage=totals), session

        assert frontend_assistant is not None  # noqa: S101 - set on every delegated path
        call_id = frontend_assistant.tool_calls[0].id
        history = session.frontend_history.append(
            Message.user(user_message),
            frontend_assistant,
            Message.tool(call_id, text),
            *caller_inputs,
            Message.assistant(text),
        ).prune(self._config.frontend.history.max_groups)
        session = replace(session, frontend_history=history, pending=None, usage=session.usage.merge(totals))
        self._emit_usage(session, totals)
        return AgentTurn(final_text=text, usage=totals), session

    # -- helpers ----------------------------------------------------------

    def _resolve_pending_before_user_message(self, session: SessionState) -> SessionState:
        if session.pending is None:
            return session
        policy = self._config.backend.tools.on_user_message_while_pending
        if policy == "error":
            raise ToolProtocolError(
                f"tool results are outstanding for {sorted(session.pending.outstanding)}; "
                "call send_tool_results() or set on_user_message_while_pending: discard_pending"
            )
        pending = session.pending
        close = self._config.frontend_enabled and self._config.backend.conversation_history.keeps_backend_turns
        self._sink.emit(
            InternalEvent(
                events.PENDING_DISCARDED,
                session.session_id,
                {"outstanding": list(pending.outstanding), "closed_in_backend_history": close},
            )
        )
        if not close:
            return replace(session, pending=None)
        # The caller may already have run these calls (writes included): keep them visible.
        closed = close_discarded_turn(pending.backend_history, pending.outstanding)
        return replace(session, pending=None, backend_history=closed.prune(self._config.backend.history.max_groups))

    def _emit_delegation(self, decision: Delegate, session_id: str) -> None:
        logging_config = self._config.logging
        if logging_config.log_delegation_query:
            self._sink.emit(InternalEvent(events.DELEGATION, session_id, {"query": decision.query}))
        if decision.filler_text and logging_config.log_filler_text:
            self._sink.emit(InternalEvent(events.FILLER, session_id, {"text": decision.filler_text}))

    def _emit_backend_context(self, request: BackendRequest, session_id: str) -> None:
        history_cfg = self._config.backend.conversation_history
        self._sink.emit(
            InternalEvent(
                events.BACKEND_CONTEXT,
                session_id,
                {
                    "enabled": history_cfg.enabled,
                    "include": history_cfg.effective_include,
                    "guidance": history_cfg.resolved_guidance_key,
                    "history_groups": len(request.history.groups()),
                    "history_messages": len(request.history),
                    "earlier_turns": request.earlier_turns,
                    "request_chars": len(request.text),
                },
            )
        )

    def _emit_usage(self, session: SessionState, totals: UsageTotals) -> None:
        self._sink.emit(
            InternalEvent(
                events.STEP_USAGE,
                session.session_id,
                {
                    "step": totals.summary(),
                    "calls": totals.calls,
                    "total_tokens": totals.usage.total_tokens,
                    "cost": totals.cost,
                    "session_calls": session.usage.calls,
                },
            )
        )


def build_agent(
    config_path: str | Path,
    *,
    tools: Sequence[ToolSpec] = (),
    event_sink: EventSink | None = None,
) -> FrontendBackendAgent:
    """Build an agent from ``agent.yaml``, wiring real OpenAI-compatible clients."""
    config = load_config(config_path)
    frontend_client = (
        OpenAIChatClient(config.frontend.llm, accounting=config.accounting) if config.frontend_enabled else None
    )
    backend_client = OpenAIChatClient(config.backend.llm, accounting=config.accounting)
    return assemble_agent(
        config,
        tools=tools,
        event_sink=event_sink,
        frontend_client=frontend_client,
        backend_client=backend_client,
    )


def assemble_agent(
    config: Config,
    *,
    tools: Sequence[ToolSpec] = (),
    event_sink: EventSink | None = None,
    frontend_client: ChatClient | None = None,
    backend_client: ChatClient | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> FrontendBackendAgent:
    """Assemble an agent from an already-loaded config and injected clients.

    This is the seam tests and the future tau2-bench adapter use: swap the chat
    clients, keep every rule in the scaffold identical. ``prompt_context`` holds
    the Jinja variables of the prompt templates (the voice agent passes its
    configuration); the frontend template also always gets ``in_progress``.
    """
    if backend_client is None:
        raise ValueError("assemble_agent requires a backend chat client")
    sink = event_sink or build_sink(config.logging.event_sink, config.logging.event_sink_path)
    catalog = load_catalog(config.prompts_path, config.prompts.inline)
    registry = ToolRegistry(tools, max_result_chars=config.backend.tools.max_result_chars)
    backend = BackendAgent(
        client=backend_client,
        system_prompt=backend_system_prompt(catalog, config, prompt_context),
        registry=registry,
        sink=sink,
    )
    frontend = None
    if config.frontend_enabled:
        if frontend_client is None:
            raise ValueError("frontend_backend mode requires a frontend chat client")
        frontend = FrontendAgent(
            client=frontend_client,
            template=catalog.get(config.frontend.prompt_key),
            config=config,
            sink=sink,
            catalog=catalog,
            base_context=prompt_context,
            prompt_key=config.frontend.prompt_key,
        )
    return FrontendBackendAgent(
        config=config,
        backend=backend,
        registry=registry,
        sink=sink,
        frontend=frontend,
        backend_request_template=request_template(catalog, config),
    )
