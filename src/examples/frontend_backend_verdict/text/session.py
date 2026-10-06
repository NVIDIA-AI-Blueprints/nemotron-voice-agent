# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit, caller-owned session state.

The agent instance holds only configuration, clients, prompts and tool specs.
Everything mutable lives here and is passed in and returned on every step, so
one agent can serve many sessions and a caller can snapshot, fork or checkpoint
a conversation.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from examples.frontend_backend_verdict.text.errors import StateReplayError
from examples.frontend_backend_verdict.text.history import History
from examples.frontend_backend_verdict.text.messages import Message, ToolCall, UsageTotals


@dataclass(frozen=True, slots=True)
class PendingTurn:
    """A backend turn suspended while the caller executes its tool calls."""

    delegation_query: str
    user_message: str
    backend_history: History
    outstanding: tuple[str, ...]
    iterations: int = 1
    frontend_assistant: Message | None = None

    def with_step(self, *, backend_history: History, outstanding: Sequence[str]) -> PendingTurn:
        """Return a copy advanced by one backend round."""
        return replace(
            self,
            backend_history=backend_history,
            outstanding=tuple(outstanding),
            iterations=self.iterations + 1,
        )


@dataclass(frozen=True, slots=True)
class SessionState:
    """All mutable state for one conversation."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    frontend_history: History = field(default_factory=History)
    backend_history: History = field(default_factory=History)
    pending: PendingTurn | None = None
    usage: UsageTotals = field(default_factory=UsageTotals)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this state (for checkpointing our own runs)."""
        return {
            "session_id": self.session_id,
            "frontend_history": [_message_to_dict(m) for m in self.frontend_history.messages],
            "backend_history": [_message_to_dict(m) for m in self.backend_history.messages],
            "pending": None
            if self.pending is None
            else {
                "delegation_query": self.pending.delegation_query,
                "user_message": self.pending.user_message,
                "backend_history": [_message_to_dict(m) for m in self.pending.backend_history.messages],
                "outstanding": list(self.pending.outstanding),
                "iterations": self.pending.iterations,
                "frontend_assistant": _message_to_dict(self.pending.frontend_assistant)
                if self.pending.frontend_assistant
                else None,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionState:
        """Rebuild state previously produced by :meth:`to_dict`."""
        pending_raw = data.get("pending")
        pending = None
        if pending_raw:
            pending = PendingTurn(
                delegation_query=str(pending_raw.get("delegation_query", "")),
                user_message=str(pending_raw.get("user_message", "")),
                backend_history=History(tuple(_message_from_dict(m) for m in pending_raw.get("backend_history", []))),
                outstanding=tuple(str(item) for item in pending_raw.get("outstanding", [])),
                iterations=int(pending_raw.get("iterations", 1)),
                frontend_assistant=_message_from_dict(pending_raw["frontend_assistant"])
                if pending_raw.get("frontend_assistant")
                else None,
            )
        return cls(
            session_id=str(data.get("session_id") or uuid.uuid4().hex[:12]),
            frontend_history=History(tuple(_message_from_dict(m) for m in data.get("frontend_history", []))),
            backend_history=History(tuple(_message_from_dict(m) for m in data.get("backend_history", []))),
            pending=pending,
        )

    @classmethod
    def from_message_history(cls, messages: Sequence[Message], *, backend_only: bool) -> SessionState:
        """Replay a *foreign* transcript (for example tau2's) into session state.

        This is deliberately separate from :meth:`from_dict`: a transcript
        records only what crossed the agent boundary, which is strictly less
        than a session — the delegation query, the filler and the frontend's own
        tool-call turn were never emitted. Fabricating them would invent
        history, so the rule is explicit and lossy:

        * paired mode — user and assistant *text* messages replay into the
          frontend history; assistant tool-call messages and tool results do not
          replay at all, because each delegation is stateless by design;
        * ``backend_only`` — everything replays into the backend history in
          transcript order, assistant tool-call messages *together with* their
          results, and the frontend history stays empty.

        A transcript whose final tool calls have no results is rejected rather
        than replayed as a half-open group.
        """
        pendingless = tuple(messages)
        _reject_unresolved_tail(pendingless)
        if backend_only:
            return cls(backend_history=History(pendingless))
        frontend = tuple(
            message
            for message in pendingless
            if message.role in ("user", "assistant") and not message.tool_calls and message.content is not None
        )
        return cls(frontend_history=History(frontend))


def _reject_unresolved_tail(messages: Sequence[Message]) -> None:
    called: list[str] = []
    answered: set[str] = set()
    for message in messages:
        called.extend(call.id for call in message.tool_calls)
        if message.role == "tool" and message.tool_call_id:
            answered.add(message.tool_call_id)
    if missing := [call_id for call_id in called if call_id not in answered]:
        raise StateReplayError(f"transcript ends with unresolved tool calls: {sorted(missing)}")


def _message_to_dict(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        payload["tool_calls"] = [
            {"id": call.id, "name": call.name, "arguments_json": call.arguments_json} for call in message.tool_calls
        ]
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    return payload


def _message_from_dict(data: dict[str, Any]) -> Message:
    return Message(
        role=data["role"],
        content=data.get("content"),
        tool_calls=tuple(
            ToolCall(id=call["id"], name=call["name"], arguments_json=call.get("arguments_json", "{}"))
            for call in data.get("tool_calls", [])
        ),
        tool_call_id=data.get("tool_call_id"),
    )
