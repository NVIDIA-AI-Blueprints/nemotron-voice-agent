# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``LLMAgent``: the ``AgentLike`` of the LLM backend, a tool loop with the control a voice session needs.

It does what FDA needs from a backend agent and nothing else (llm-backend-plan.md section 6): one
blocking ``run_conversation`` per run on the worker thread, steer / redirect / hard interrupt from the
loop thread, cumulative usage. Its behaviour:

* **Redirect** (35-54 % of sessions): abort the request at once, retry with an assistant placeholder
  and the correction; a reply that finished just before the redirect is discarded; never partial text.
* **Steer**: a wrapped user message after the newest tool result (even one of an earlier turn, as
  the marker in ``harness_system`` says), else returned as ``pending_steer``.
* **Tool calls**: name repair, unknown-tool and bad-argument results, three rounds of only
  unknown names end the run, two loop warnings on repeated failures.
* **Replies**: reasoning-only replies become the answer; a reply that ends on an announced action is
  re-prompted with "Continue now" (twice at most); one retry for an empty reply; after
  ``max_iterations`` calls a summary request answered without tools.

Stored rows are already what goes on the wire (see ``wire.py``). Stdlib only, apart from the client.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from examples.frontend_delegation_agent.core.backends.llm import tools as tool_rules
from examples.frontend_delegation_agent.core.backends.llm.client import Aborted, ChatClient, ChatResult
from examples.frontend_delegation_agent.core.backends.llm.wire import build, strip_think

logger = logging.getLogger(__name__)

ToolCaller = Callable[[str, dict[str, Any], str], str]
Render = Callable[..., str]

#: A redirect that keeps arriving while its correction is retried becomes a steer after this many restarts.
MAX_REDIRECT_RESTARTS = 3
#: Re-prompts with ``harness_continue_now`` per run.
MAX_CONTINUE_NUDGES = 2
#: Rounds in a row with only unknown tool names before the run ends.
MAX_INVALID_NAME_ROUNDS = 3
#: Joins two redirects that arrive before the first is applied (``[Additional user correction]``).
REDIRECT_JOIN = "\n\n[Additional user correction]\n"

# A short reply whose tail announces the next action ("I'll now check ...").
_TRAILING_INTENT = re.compile(
    r"(?:\blet me now\b|\bi(?:'|’)?ll now\b|\bi will now\b|\bnow i(?:(?:'|’)ll| will)\b|\bnext[,:] i\b)"
    r"[^.!?\n]{0,100}[.:…]?\s*$",
    re.IGNORECASE,
)
# Reasoning whose last sentence is a first-person plan ("Let me look up the order.").
_PLAN_TAIL = re.compile(
    r"(?:^|[.!?:—–\n]\s*|…\s*)"
    r"(?:let(?:(?:'|’)s| me)\b|i(?:(?:'|’)ll| will| need to| should| am going to|(?:'|’)m going to)\b"
    r"|next[,:]? i\b|now i(?:(?:'|’)ll| will| need to)\b|first[,:]? i(?:(?:'|’)ll| will| need to)\b)"
    r"[^.!?\n]{0,160}(?:[.:…]+)?\s*$",
    re.IGNORECASE,
)


def announces_action(text: str, *, promoted: bool) -> bool:
    """Whether a reply without a tool call ends by announcing an action instead of answering."""
    text = text.strip()
    if not text:
        return False
    if len(text) <= 400 and _TRAILING_INTENT.search(text[-160:]):
        return True
    return promoted and bool(_PLAN_TAIL.search(text[-240:]))


class LLMAgent:
    """One session's agent; ``run_conversation`` runs on the worker thread, the controls on the loop."""

    def __init__(
        self,
        *,
        client: ChatClient,
        system: str,
        tools: list[dict[str, Any]],
        call_tool: ToolCaller,
        render: Render,
        max_iterations: int,
        on_activity: Callable[[str, str, str], None] | None = None,
        trace: Any = None,
    ) -> None:
        """``tools`` is the request's ``tools`` field; ``render(key, **vars)`` renders ``llm.yaml`` text."""
        self._client = client
        self._system = system
        self._tools = tools
        self._names = [t["function"]["name"] for t in tools]
        self._call_tool = call_tool
        self._render = render
        self._max_iterations = max_iterations
        self._on_activity = on_activity
        self._trace = trace
        self._lock = threading.Lock()
        self._interrupted = threading.Event()
        self._interrupt_message = ""
        self._model_active = threading.Event()
        self._steers: list[str] = []
        self._redirects: list[str] = []
        self._current_tool: str | None = None
        self._api_calls = 0
        self._last_activity = time.time()
        self._usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}

    # -- AgentLike: controls (loop thread) ----------------------------------------------

    def steer(self, text: str) -> bool:
        """Queue ``text``; it is delivered after the newest tool result (never interrupts)."""
        text = text.strip()
        if text:
            with self._lock:
                self._steers.append(text)
        return True

    def redirect(self, text: str) -> bool:
        """During a model request: abort it and retry with ``text``. During a tool: a steer. Else ``False``."""
        with self._lock:
            if self._current_tool is not None:
                self._steers.append(text.strip())
                return True
            if not self._model_active.is_set():
                return False
            self._redirects.append(text.strip())
        self._client.abort()
        return True

    def hard_interrupt(self, message: str) -> None:
        """End the run as soon as possible; a running tool returns first (the bridge checks the flag)."""
        self._interrupt_message = message
        self._interrupted.set()
        self._client.abort()

    def clear_interrupt(self) -> None:
        """Drop an interrupt left over from an idle period (pending steers stay)."""
        self._interrupted.clear()
        self._interrupt_message = ""

    def is_interrupted(self) -> bool:
        """Whether the running turn was interrupted (the tool bridge stops waiting)."""
        return self._interrupted.is_set()

    def model_request_active(self) -> bool:
        """Whether a model request is in flight."""
        return self._model_active.is_set()

    def get_activity_summary(self) -> dict[str, Any]:
        """For ``status``: the call count, the running tool and the last activity."""
        return {
            "api_call_count": self._api_calls,
            "current_tool": self._current_tool,
            "last_activity_ts": round(self._last_activity, 3),
            "model_request_active": self._model_active.is_set(),
        }

    def usage_snapshot(self) -> dict[str, int]:
        """Cumulative token counters."""
        return dict(self._usage)

    def close(self) -> None:
        """Release the HTTP client."""
        self._client.close()

    # -- AgentLike: the run (worker thread) ------------------------------------------------

    def run_conversation(
        self, user_message: str, conversation_history: list[dict[str, Any]], task_id: str
    ) -> dict[str, Any]:
        """One run: model calls and tool calls until an answer, an interrupt or a failure."""
        messages = [dict(row) for row in conversation_history]
        messages.append({"role": "user", "content": user_message})
        warnings = tool_rules.FailureWarnings(self._render)
        invalid_rounds = nudges = restarts = 0
        retried_empty = limit_sent = False
        calls = 0
        while True:
            if self._interrupted.is_set():
                return self._interrupted_result(messages)
            final_call = calls >= self._max_iterations
            if final_call and not limit_sent:
                limit_sent = True
                messages.append({"role": "user", "content": self._render("harness_iteration_limit")})
            self._deliver_steer(messages)
            self._apply_redirect(messages)
            try:
                result = self._model_call(messages, task_id, calls + 1, tools=None if final_call else self._tools)
            except Aborted:
                restarts = self._after_abort(restarts)
                continue
            except Exception as exc:  # noqa: BLE001 - the run fails; the controller apologises
                logger.exception("model call failed")
                return self._result(messages, None, failed=True, error=f"{type(exc).__name__}: {exc}")
            if result is None:  # a redirect crossed the finished reply: discard it and retry
                restarts = self._after_abort(restarts)
                continue
            restarts = 0
            calls += 1
            if result.tool_calls and not final_call:
                outcome = self._tool_round(messages, result, task_id, warnings, invalid_rounds)
                if isinstance(outcome, dict):
                    return outcome
                invalid_rounds = outcome
                continue
            text = strip_think(result.content).strip()
            promoted = False
            if not text and result.finish_reason == "stop" and result.reasoning.strip():
                text, promoted = strip_think(result.reasoning).strip(), True
            if (
                not final_call
                and self._tools
                and nudges < MAX_CONTINUE_NUDGES
                and announces_action(text, promoted=promoted)
            ):
                nudges += 1
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": self._render("harness_continue_now")})
                continue
            if not text and not retried_empty and not final_call:
                retried_empty = True
                continue
            messages.append({"role": "assistant", "content": text})
            return self._result(messages, text)

    # -- internals ----------------------------------------------------------------------

    def _model_call(
        self, messages: list[dict[str, Any]], task_id: str, number: int, *, tools: list[dict[str, Any]] | None
    ) -> ChatResult | None:
        wire = build(self._system, messages, self._render("harness_placeholder"))
        request_id = uuid.uuid4().hex[:12]
        if self._trace is not None:
            self._trace.write(
                "request",
                api_request_id=request_id,
                task_id=task_id,
                api_call_count=number,
                model=self._client.model,
                messages=wire,
                system_prompt=self._system,
                tool_count=len(tools or ()),
            )
        started = time.monotonic()
        with self._lock:
            self._model_active.set()
        self._last_activity = time.time()
        try:
            result = self._client.complete(wire, tools, should_abort=self._abort_pending)
        finally:
            with self._lock:
                self._model_active.clear()
                crossed = bool(self._redirects)
        if crossed:
            return None
        self._api_calls += 1
        self._last_activity = time.time()
        for key, value in result.usage.items():
            self._usage[key] = self._usage.get(key, 0) + int(value)
        if self._trace is not None:
            self._trace.write(
                "response",
                api_request_id=request_id,
                task_id=task_id,
                api_call_count=number,
                finish_reason=result.finish_reason,
                assistant_message={
                    "content": result.content,
                    "reasoning": result.reasoning,
                    "tool_calls": result.tool_calls,
                },
                usage=result.usage,
                api_duration=round(time.monotonic() - started, 3),
            )
        return result

    def _abort_pending(self) -> bool:
        """An interrupt or a redirect that arrived before the model request started."""
        with self._lock:
            return self._interrupted.is_set() or bool(self._redirects)

    def _after_abort(self, restarts: int) -> int:
        restarts += 1
        if restarts > MAX_REDIRECT_RESTARTS:
            with self._lock:
                self._steers.extend(self._redirects)
                self._redirects.clear()
            return 0
        return restarts

    def _apply_redirect(self, messages: list[dict[str, Any]]) -> None:
        with self._lock:
            pending, self._redirects = self._redirects, []
        if not pending:
            return
        if not messages or messages[-1].get("role") != "assistant":
            messages.append({"role": "assistant", "content": ""})  # sent as harness_placeholder
        messages.append(
            {"role": "user", "content": self._render("harness_redirect_user", text=REDIRECT_JOIN.join(pending))}
        )

    def _deliver_steer(self, messages: list[dict[str, Any]]) -> None:
        """Insert pending steers after the newest tool result anywhere in ``messages`` (the steer placement)."""
        newest = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "tool"), None)
        if newest is None:
            return
        with self._lock:
            pending, self._steers = self._steers, []
        if pending:
            row = {"role": "user", "content": self._render("harness_steer_wrap", text="\n".join(pending))}
            messages.insert(newest + 1, row)

    def _tool_round(
        self,
        messages: list[dict[str, Any]],
        result: ChatResult,
        task_id: str,
        warnings: tool_rules.FailureWarnings,
        invalid_rounds: int,
    ) -> int | dict[str, Any]:
        """Run one batch of tool calls; returns the new invalid-name count, or a final result."""
        calls = []
        for index, call in enumerate(result.tool_calls):
            name = (
                call["name"]
                if call["name"] in self._names
                else (tool_rules.repair_name(call["name"], self._names) or call["name"])
            )
            args, error = tool_rules.parse_arguments(call["arguments"])
            calls.append({"id": call["id"] or f"call_{index}", "name": name, "args": args, "error": error})
        unknown = [c for c in calls if c["name"] not in self._names]
        if unknown and len(unknown) == len(calls):
            invalid_rounds += 1
            if invalid_rounds >= MAX_INVALID_NAME_ROUNDS:
                text = self._render("harness_invalid_tool_exit", name=unknown[0]["name"])
                return self._result(messages, text, partial=True, error=text)
        else:
            invalid_rounds = 0
        messages.append(
            {
                "role": "assistant",
                "content": strip_think(result.content).strip(),
                "tool_calls": [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {
                            "name": c["name"],
                            "arguments": tool_rules.canonical_arguments(c["args"]) if c["args"] is not None else "{}",
                        },
                    }
                    for c in calls
                ],
            }
        )
        for call in calls:
            if self._interrupted.is_set():
                output = self._render("harness_interrupted")
            elif call["name"] not in self._names:
                output = self._render(
                    "harness_unknown_tool", name=call["name"], available=", ".join(sorted(self._names))
                )
            elif call["args"] is None:
                output = self._render("harness_invalid_args", error=call["error"])
            else:
                output = warnings.after(call["name"], call["args"], self._run_tool(call["name"], call["args"], task_id))
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
        return invalid_rounds

    def _run_tool(self, name: str, args: dict[str, Any], task_id: str) -> str:
        with self._lock:
            self._current_tool = name
        self._activity("tool_started", name, str(args))
        try:
            output = self._call_tool(name, args, task_id)
        finally:
            with self._lock:
                self._current_tool = None
            self._last_activity = time.time()
        self._activity("tool_completed", name, output)
        return output

    def _activity(self, event: str, name: str, preview: str) -> None:
        if self._on_activity is not None:
            self._on_activity(event, name, preview[:200])

    def _interrupted_result(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        result = self._result(messages, None)
        result.update(
            interrupted=True,
            completed=False,
            turn_exit_reason="interrupted_by_user",
            interrupt_message=self._interrupt_message,
        )
        return result

    def _result(
        self,
        messages: list[dict[str, Any]],
        final: str | None,
        *,
        failed: bool = False,
        partial: bool = False,
        error: str | None = None,
    ) -> dict[str, Any]:
        """The ``run_conversation`` result; a history ending in a tool result is closed with ``harness_interrupted``."""
        if messages and messages[-1].get("role") == "tool":
            messages.append(
                {"role": "assistant", "content": (final or "").strip() or self._render("harness_interrupted")}
            )
        with self._lock:
            pending, self._steers = self._steers, []
        result: dict[str, Any] = {
            "final_response": final,
            "messages": messages,
            "interrupted": False,
            "failed": failed,
            "partial": partial,
            "completed": final is not None and not failed and not partial,
            "turn_exit_reason": "failed" if failed else ("partial" if partial else "text_response"),
            "api_calls": self._api_calls,
        }
        if pending:
            result["pending_steer"] = "\n".join(pending)
        if error:
            result["error"] = error
        return result
