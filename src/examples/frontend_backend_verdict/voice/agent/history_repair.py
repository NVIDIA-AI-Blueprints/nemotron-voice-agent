# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rewrite every stored copy of an interrupted answer to what the user heard.

The text prototype stores a spoken answer in more than one place (plan section
8.3). A paired, delegated turn closes its frontend group as::

    user, assistant(call_backend), tool(call_id, <answer>), assistant(<answer>)

so both the tool message and the assistant continuation hold the full answer.
Caller speech said while the turn's tool calls were out (user messages marked
``CONTINUES_TURN``) sits between the tool message and the continuation and is
part of the same group.
A direct answer is ``user, assistant(<answer>)``; ``backend_only`` keeps the
answer as the final assistant message of the last backend group. With the paired
backend history on (``backend.conversation_history``), a delegated answer is also
the backend's final message, and both histories are repaired or neither is. The
target is located structurally *and* verified by exact text equality; a mismatch
raises instead of rewriting a different message. ``SessionState`` is immutable, so
the repair returns a new state.
"""

from __future__ import annotations

from dataclasses import replace

from examples.frontend_backend_verdict.text.delegation import CALL_BACKEND
from examples.frontend_backend_verdict.text.history import History
from examples.frontend_backend_verdict.text.messages import CONTINUES_TURN, Message
from examples.frontend_backend_verdict.text.session import SessionState
from examples.frontend_backend_verdict.voice.errors import HistoryRepairError


def _continues_turn(message: Message) -> bool:
    return message.role == "user" and bool((message.meta or {}).get(CONTINUES_TURN))


def _last_group_bounds(history: History) -> tuple[int, int]:
    messages = history.messages
    start = 0
    for index, message in enumerate(messages):
        if message.role == "user" and not _continues_turn(message):
            start = index
    return start, len(messages)


def _replace_content(message: Message, full_text: str, replacement: str, where: str) -> Message:
    if message.content != full_text:
        raise HistoryRepairError(f"{where}: stored text does not match the spoken answer")
    return replace(message, content=replacement)


def _repair_frontend(history: History, full_text: str, replacement: str) -> tuple[History, bool]:
    """Return the repaired history and whether the last group was a delegated turn."""
    messages = list(history.messages)
    start, end = _last_group_bounds(history)
    group = messages[start:end]
    if len(group) == 2 and group[0].role == "user" and group[1].role == "assistant" and not group[1].tool_calls:
        messages[start + 1] = _replace_content(group[1], full_text, replacement, "direct answer")
        return History(tuple(messages)), False
    if (
        len(group) >= 4
        and group[0].role == "user"
        and group[1].role == "assistant"
        and len(group[1].tool_calls) == 1
        and group[1].tool_calls[0].name == CALL_BACKEND
        and group[2].role == "tool"
        and group[2].tool_call_id == group[1].tool_calls[0].id
        and all(_continues_turn(message) for message in group[3:-1])
        and group[-1].role == "assistant"
        and not group[-1].tool_calls
    ):
        last = start + len(group) - 1
        messages[start + 2] = _replace_content(group[2], full_text, replacement, "call_backend tool result")
        messages[last] = _replace_content(group[-1], full_text, replacement, "assistant continuation")
        return History(tuple(messages)), True
    raise HistoryRepairError(f"unexpected shape of the last frontend group: {[m.role for m in group]}")


def _repair_backend(history: History, full_text: str, replacement: str) -> History:
    messages = list(history.messages)
    if not messages:
        raise HistoryRepairError("backend history is empty")
    last = messages[-1]
    if last.role != "assistant" or last.tool_calls:
        raise HistoryRepairError(f"last backend message is {last.role} with tool calls={bool(last.tool_calls)}")
    messages[-1] = _replace_content(last, full_text, replacement, "backend final answer")
    return History(tuple(messages))


def repair_interrupted_answer(
    state: SessionState,
    *,
    full_text: str,
    replacement: str,
    frontend_enabled: bool,
    backend_history: bool = False,
) -> SessionState:
    """Return ``state`` with every copy of ``full_text`` in the last group replaced by ``replacement``.

    ``backend_history`` (paired mode with the backend history on) also repairs
    the backend's copy of a delegated answer. Both repairs are computed before
    either is applied, so a failure leaves both histories unchanged.
    """
    if not frontend_enabled:
        return replace(state, backend_history=_repair_backend(state.backend_history, full_text, replacement))
    frontend, delegated = _repair_frontend(state.frontend_history, full_text, replacement)
    if not (backend_history and delegated):
        return replace(state, frontend_history=frontend)
    backend = _repair_backend(state.backend_history, full_text, replacement)
    return replace(state, frontend_history=frontend, backend_history=backend)
