# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The frontend agent: answer directly, or delegate. Nothing else.

``decide()`` returns a closed union rather than a message, which is what makes
"filler text can never reach the caller" a property of the types: there is no
path from :class:`Delegate.filler_text` to an outward envelope.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from examples.frontend_backend_verdict.text import events
from examples.frontend_backend_verdict.text.config import Config
from examples.frontend_backend_verdict.text.delegation import (
    CALL_BACKEND,
    FRONTEND_TOOLS,
    FRONTEND_TOOLS_IN_PROGRESS,
    TASK_CONTINUE,
    TASK_NEW,
)
from examples.frontend_backend_verdict.text.errors import FrontendContractError
from examples.frontend_backend_verdict.text.events import EventSink, InternalEvent
from examples.frontend_backend_verdict.text.history import History
from examples.frontend_backend_verdict.text.llm import ChatClient
from examples.frontend_backend_verdict.text.messages import Message, ToolCall, UsageTotals
from examples.frontend_backend_verdict.text.prompts import PromptCatalog, compile_template, render_template
from examples.frontend_backend_verdict.text.tools import decode_arguments

_REPAIR_INSTRUCTION = (
    "Your previous turn broke the contract: {problem}. Reply again for the same user turn. "
    f"Either answer the user directly in plain text, or emit exactly one {CALL_BACKEND} tool call "
    "whose 'query' states the complete current request. Never do both, and never write a tool call as text."
)

# Text containing any of these is a tool call (or just its arguments) typed out as prose: the
# model imitating the prompt's examples, or a template that failed to parse. Never an answer.
_TYPED_TOOL_CALL_MARKERS = (CALL_BACKEND, "filler_text", "tool_calls", "<tool_call", "<function=")


@dataclass(frozen=True, slots=True)
class DirectAnswer:
    """The frontend answered the user itself."""

    text: str


@dataclass(frozen=True, slots=True)
class Delegate:
    """The frontend handed the task to the backend."""

    query: str
    filler_text: str
    assistant_message: Message
    #: ``continue`` or ``new`` when the call carried a valid in-progress ``task``; ``""`` otherwise.
    task: str = ""


@dataclass(frozen=True, slots=True)
class ContractFallback:
    """The frontend broke its contract and a safe answer was substituted."""

    text: str
    problem: str


Decision = DirectAnswer | Delegate | ContractFallback


class FrontendAgent:
    """Talks to the user; owns no tools except delegation."""

    def __init__(
        self,
        *,
        client: ChatClient,
        template: str,
        config: Config,
        sink: EventSink,
        catalog: PromptCatalog | None = None,
        base_context: Mapping[str, Any] | None = None,
        prompt_key: str = "",
    ) -> None:
        """Bind the frontend to its client, prompt template and event sink.

        The template is rendered with ``base_context`` plus ``in_progress``: ``None``
        on normal turns (rendered once, here) and a mapping on in-progress calls.
        """
        self._client = client
        self._config = config
        self._sink = sink
        self._prompt_key = prompt_key
        self._template = compile_template(template, catalog, key=prompt_key)
        self._base_context = dict(base_context or {})
        self._system_prompt = self._render(None)

    @property
    def system_prompt(self) -> str:
        """The system prompt of a normal turn."""
        return self._system_prompt

    def _render(self, in_progress: Mapping[str, Any] | None) -> str:
        context = {**self._base_context, "in_progress": in_progress}
        return render_template(self._template, self._config, context, key=self._prompt_key)

    async def decide(
        self, user_text: str, history: History, *, session_id: str, in_progress: Mapping[str, Any] | None = None
    ) -> tuple[Decision, UsageTotals]:
        """Choose between answering directly and delegating.

        On a contract violation the frontend is reprompted up to
        ``max_repair_attempts`` times and then **fails closed**: the model's raw
        text is never returned as the answer, because that is precisely when it
        is most likely to contain tool-call markup or a half-formed promise.

        ``in_progress`` (a caller's description of a request already being worked
        on) is passed to the prompt template, and adds the required ``task`` field
        to ``call_backend``. Without it the call is unchanged.
        """
        totals = UsageTotals()
        delegation = self._config.frontend.delegation
        attempt_messages = [*history.messages, Message.user(user_text)]
        system_prompt = self._system_prompt
        tools = FRONTEND_TOOLS
        if in_progress is not None:
            system_prompt = self._render(in_progress)
            tools = FRONTEND_TOOLS_IN_PROGRESS
        problem = ""
        for attempt in range(delegation.max_repair_attempts + 1):
            response = await self._client.complete(
                messages=[Message.system(system_prompt), *attempt_messages],
                tools=list(tools),
            )
            totals = totals.add_call(
                "frontend", usage=response.usage, latency_ms=response.latency_ms, cost=response.cost
            )
            decision, problem = self._interpret(
                response.content, response.tool_calls, session_id=session_id, in_progress=in_progress is not None
            )
            if decision is not None:
                return decision, totals
            self._sink.emit(
                InternalEvent(
                    events.FRONTEND_REPAIR,
                    session_id,
                    {"attempt": attempt + 1, "problem": problem, "rejected_text": response.content},
                )
            )
            attempt_messages = [
                *attempt_messages,
                Message.assistant(response.content or ""),
                Message.user(_REPAIR_INSTRUCTION.format(problem=problem)),
            ]
        self._sink.emit(InternalEvent(events.FRONTEND_CONTRACT_VIOLATION, session_id, {"problem": problem}))
        if delegation.on_contract_violation == "error":
            raise FrontendContractError(problem)
        return ContractFallback(text=delegation.fallback_text, problem=problem), totals

    def _interpret(
        self, content: str | None, tool_calls: tuple[ToolCall, ...], *, session_id: str, in_progress: bool = False
    ) -> tuple[Decision | None, str]:
        """Map one raw completion onto a decision, or report why it is invalid."""
        if tool_calls:
            unknown = [call.name for call in tool_calls if call.name != CALL_BACKEND]
            if unknown:
                return None, f"called unavailable tool(s) {unknown}; only {CALL_BACKEND} exists"
            if len(tool_calls) > 1:
                return None, "emitted several delegation calls; exactly one is allowed"
            call = tool_calls[0]
            arguments = decode_arguments(call)
            query = str(arguments.get("query") or "").strip()
            if not query:
                return None, "called call_backend without a non-empty 'query'"
            if content and content.strip():
                self._sink.emit(
                    InternalEvent(events.FRONTEND_CONTRACT_VIOLATION, session_id, {"discarded_text": content.strip()})
                )
            filler = str(arguments.get("filler_text") or "").strip()
            task = ""
            if in_progress:
                # A missing or unknown value is not a contract violation (no repair round): the caller
                # treats it as a new request.
                raw_task = str(arguments.get("task") or "").strip().lower()
                task = raw_task if raw_task in (TASK_CONTINUE, TASK_NEW) else ""
            return Delegate(
                query=query,
                filler_text=filler,
                assistant_message=Message.assistant_tool_calls((call,)),
                task=task,
            ), ""
        text = (content or "").strip()
        if not text:
            return None, "returned neither text nor a tool call"
        if any(marker in text for marker in _TYPED_TOOL_CALL_MARKERS):
            return None, f"wrote a {CALL_BACKEND} tool call as text instead of emitting a real tool call"
        return DirectAnswer(text=text), ""
