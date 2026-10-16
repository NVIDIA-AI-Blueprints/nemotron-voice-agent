# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Hermes surface a worker uses (``AgentLike``) and a scriptable stand-in (``FakeAgent``).

``FakeAgent`` imitates the Hermes behaviour the design relies on, verified in the
P0 spike: a blocking ``run_conversation`` on one thread, ``steer`` delivered after a
tool batch (else returned as ``pending_steer``), ``redirect`` accepted only during a
"model request", ``hard_interrupt`` ending the run with ``interrupted=True``, and
cumulative usage counters.

Directives in the last line of the user message (the user's own words) script it
(tests and ``--stub-backend``):

``[slow:N]``         a model request lasting N seconds (interruptible; redirect-able)
``[tool:NAME {json}]`` one tool call through the bridge (repeatable, in order)
``[schemas]``        answer with each registered tool's parameter keys
``[histlen]``        answer with the length of the conversation history received
``[hang]``           never return, ignore interrupts (the watchdog must kill the worker);
                     without ``allow_crash`` (in-process) it waits for an interrupt instead
``[crash]``          ``os._exit(3)`` (only inside a worker process)
``[raise]``          raise ``RuntimeError``
``[empty]``          return an empty final response
``[fail]``           return ``failed=True`` with the messages so far

Without a tool directive, the first registered tool whose name appears in the text
is called with ``{}`` so a plain "status of order 1234" still makes a tool call when a
``get_order``-like tool exists and the words ``order`` + digits are present.

Stdlib only.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

ToolCaller = Callable[[str, dict[str, Any], str], str]

_TOOL = re.compile(r"\[tool:([A-Za-z0-9_\-]+)\s*(\{.*?\})?\]")
_SLOW = re.compile(r"\[slow:([0-9.]+)\]")
_ORDER = re.compile(r"\border\s*#?\s*(\d{3,})", re.IGNORECASE)


class AgentLike(Protocol):
    """What the worker core needs from an agent (Hermes ``AIAgent`` or ``FakeAgent``)."""

    def run_conversation(
        self, user_message: str, conversation_history: list[dict[str, Any]], task_id: str
    ) -> dict[str, Any]:
        """Run one turn (blocking)."""
        ...

    def steer(self, text: str) -> bool:
        """Queue text for delivery after the current tool batch."""
        ...

    def redirect(self, text: str) -> bool:
        """Abort the in-flight model request and retry with ``text``; ``False`` outside one."""
        ...

    def get_activity_summary(self) -> dict[str, Any]:
        """What the agent is doing now."""
        ...

    def clear_interrupt(self) -> None:
        """Drop an interrupt left over from an idle period."""
        ...

    def hard_interrupt(self, message: str) -> None:
        """Stop the running turn as soon as possible."""
        ...

    def model_request_active(self) -> bool:
        """Whether a model request is in flight."""
        ...

    def usage_snapshot(self) -> dict[str, int]:
        """Cumulative token counters."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


class FakeAgent:
    """A deterministic, scriptable agent with Hermes-like control semantics."""

    def __init__(
        self,
        *,
        call_tool: ToolCaller,
        tools: Mapping[str, Mapping[str, Any]],
        on_activity: Callable[[str, str, str], None] | None = None,
        allow_crash: bool = True,
        default_tool: bool = False,
    ) -> None:
        """``call_tool(name, args, task_id)`` is the bridge; ``tools`` maps name → JSON schema parameters.

        ``default_tool``: a run with no directive calls the first tool once (stub gates need a function call).
        """
        self._call_tool = call_tool
        self._tools = dict(tools)
        self._on_activity = on_activity
        self._allow_crash = allow_crash
        self._default_tool = default_tool
        self._lock = threading.Lock()
        self._steers: list[str] = []
        self._redirect: str | None = None
        self._interrupted = threading.Event()
        self._model_active = threading.Event()
        self._current_tool: str | None = None
        self._api_calls = 0
        self._usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
        self.closed = False

    # -- AgentLike ---------------------------------------------------------------------

    def run_conversation(
        self, user_message: str, conversation_history: list[dict[str, Any]], task_id: str
    ) -> dict[str, Any]:
        """Execute the directives in ``user_message``."""
        messages = [*conversation_history, {"role": "user", "content": user_message}]
        # Directives count only in the last line (the user's own words); context lines above it
        # may quote earlier requests, which must not re-trigger them.
        text = user_message.strip().splitlines()[-1] if user_message.strip() else ""
        if "[crash]" in text and self._allow_crash:
            os._exit(3)
        if "[raise]" in text:
            raise RuntimeError("fake agent raised")
        if "[hang]" in text:
            while self._allow_crash or not self._interrupted.is_set():  # a real hang ignores interrupts
                time.sleep(0.05)
            return self._result(messages, None, interrupted=True)
        replies: list[str] = []
        slow = _SLOW.search(text)
        if slow is not None:
            outcome = self._model_request(float(slow.group(1)))
            if outcome == "interrupted":
                return self._result(messages, None, interrupted=True)
            if outcome is not None:  # redirected
                messages.append({"role": "user", "content": outcome})
                replies.append(f"redirected: {outcome}")
        for index, (name, args) in enumerate(self._tool_calls(text)):
            if self._interrupted.is_set():
                return self._result(messages, None, interrupted=True)
            call_id = f"fake-{task_id}-{index}"
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                    ],
                }
            )
            self._current_tool = name
            self._activity("tool_started", name, json.dumps(args))
            output = self._call_tool(name, args, task_id)
            self._activity("tool_completed", name, output[:120])
            self._current_tool = None
            self._api_calls += 1
            messages.append({"role": "tool", "tool_call_id": call_id, "content": output})
            replies.append(f"{name} -> {output}")
            steered = self._drain_steers()
            if steered:
                messages.append({"role": "user", "content": f"[OUT-OF-BAND USER MESSAGE]\n{steered}"})
                replies.append(f"steered: {steered}")
            if self._interrupted.is_set():
                return self._result(messages, None, interrupted=True)
        if "[schemas]" in text:
            replies.append(
                "; ".join(
                    f"{name}={sorted((params or {}).get('properties', {}))}"
                    for name, params in sorted(self._tools.items())
                )
            )
        if "[histlen]" in text:
            replies.append(f"history={len(conversation_history)}")
        if "[fail]" in text:
            return self._result(messages, None, failed=True)
        final = "" if "[empty]" in text else ("Fake answer: " + " | ".join(replies) if replies else "Fake answer.")
        self._api_calls += 1
        self._usage["input_tokens"] += 10 + len(user_message) // 4
        self._usage["output_tokens"] += 5 + len(final) // 4
        messages.append({"role": "assistant", "content": final})
        result = self._result(messages, final)
        pending = self._drain_steers()
        if pending:
            result["pending_steer"] = pending
        return result

    def steer(self, text: str) -> bool:
        """Queue ``text`` (Hermes returns ``True`` for any non-empty text)."""
        if not text.strip():
            return False
        with self._lock:
            self._steers.append(text)
        return True

    def redirect(self, text: str) -> bool:
        """Accepted only during a model request."""
        if not self._model_active.is_set():
            return False
        with self._lock:
            self._redirect = text
        return True

    def get_activity_summary(self) -> dict[str, Any]:
        """A Hermes-shaped summary."""
        return {
            "current_tool": self._current_tool,
            "api_call_count": self._api_calls,
            "budget_used": self._api_calls,
            "budget_max": 30,
            "seconds_since_activity": 0.0,
            "description": f"executing tool: {self._current_tool}" if self._current_tool else "thinking",
        }

    def clear_interrupt(self) -> None:
        """Reset the interrupt flag."""
        self._interrupted.clear()

    def hard_interrupt(self, message: str) -> None:
        """Set the interrupt flag (``[hang]`` ignores it)."""
        self._interrupted.set()

    def model_request_active(self) -> bool:
        """Whether a ``[slow:N]`` phase is running."""
        return self._model_active.is_set()

    def usage_snapshot(self) -> dict[str, int]:
        """Cumulative counters."""
        return dict(self._usage)

    def close(self) -> None:
        """Mark closed."""
        self.closed = True

    # -- internals -----------------------------------------------------------------------

    def _model_request(self, seconds: float) -> str | None:
        """Sleep ``seconds``; ``"interrupted"``, a redirect text, or ``None``."""
        self._model_active.set()
        try:
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if self._interrupted.is_set():
                    return "interrupted"
                with self._lock:
                    redirect, self._redirect = self._redirect, None
                if redirect is not None:
                    return redirect
                time.sleep(0.02)
            return None
        finally:
            self._model_active.clear()

    def _tool_calls(self, text: str) -> list[tuple[str, dict[str, Any]]]:
        calls: list[tuple[str, dict[str, Any]]] = []
        for match in _TOOL.finditer(text):
            try:
                args = json.loads(match.group(2)) if match.group(2) else {}
            except ValueError:
                args = {}
            calls.append((match.group(1), args if isinstance(args, dict) else {}))
        if calls or any(d in text for d in ("[schemas]", "[empty]", "[fail]", "[slow:", "[histlen]")):
            return calls
        order = _ORDER.search(text)
        if order is not None:
            for name in self._tools:
                if "order" in name:
                    return [(name, {"order_id": order.group(1)})]
        if self._default_tool:
            for name in self._tools:
                if name != "transfer_to_human_agents":  # tau2 ends the simulation on that one
                    return [(name, {})]
        return calls

    def _drain_steers(self) -> str:
        with self._lock:
            steers, self._steers = self._steers, []
        return "\n".join(steers)

    def _activity(self, event: str, name: str, preview: str) -> None:
        if self._on_activity is not None:
            self._on_activity(event, name, preview)

    def _result(
        self, messages: list[dict[str, Any]], final: str | None, *, interrupted: bool = False, failed: bool = False
    ) -> dict[str, Any]:
        return {
            "final_response": final,
            "messages": messages,
            "interrupted": interrupted,
            "failed": failed,
            "completed": not interrupted and not failed,
            "turn_exit_reason": "interrupted" if interrupted else ("failed" if failed else "text_response"),
        }
