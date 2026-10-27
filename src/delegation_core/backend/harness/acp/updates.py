# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ACP ``session/update`` notifications, read into the harness's session events (``harness/protocol.py``).

One :class:`SessionUpdates` per session reads them in wire order, through the session's agent class, with one parser
per update kind in one table. It reads the raw JSON, not the SDK's models, so an agent's ``_meta`` extensions and a
newer schema's extra fields never stop an update; kinds a run does not use give no event. A message chunk with no
``messageId`` is read as the agent class says, and a reasoning chunk never ends the id-less message being streamed,
since a model call may reason between parts of its text.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from ..protocol import MessageChunk, PlanUpdate, SessionEvent, ThoughtChunk, ToolCard, ToolCardEnd, UserText

if TYPE_CHECKING:
    from .agent import AcpAgent

# The statuses of a tool card that end it.
ENDED = {"completed": False, "failed": True}  # status: whether it failed
# How much of a tool card's output an event keeps (the bridge reports its own calls; this is for the agent's own).
OUTPUT_CHARS = 8000


def block_text(content: Any) -> str:
    """The text of a content block, or of a list of tool content items (``{"type": "content", "content": {...}}``)."""
    if isinstance(content, dict):
        inner = content.get("content")
        return block_text(inner) if isinstance(inner, dict | list) else str(content.get("text") or "")
    if isinstance(content, list):
        return "\n".join(text for item in content if (text := block_text(item)))
    return ""


def _message(updates: SessionUpdates, update: dict) -> SessionEvent:
    message_id = update.get("messageId") or updates.agent.unnamed_message(updates)
    return MessageChunk(message_id, block_text(update.get("content")))


def _thought(updates: SessionUpdates, update: dict) -> SessionEvent:
    return ThoughtChunk(update.get("messageId"))


def _card(updates: SessionUpdates, update: dict) -> SessionEvent:
    arguments = update.get("rawInput")
    name = updates.agent.tool_name(update.get("title"))
    return ToolCard(
        str(update.get("toolCallId") or ""),
        name,
        arguments if isinstance(arguments, dict) else None,
        updates.agent.bridged(name, updates.server),
    )


def _card_update(updates: SessionUpdates, update: dict) -> SessionEvent | None:
    status = update.get("status")
    if status not in ENDED:
        return None
    raw = update.get("rawOutput")
    output = raw if isinstance(raw, str) else block_text(update.get("content"))
    return ToolCardEnd(str(update.get("toolCallId") or ""), ENDED[status], output[:OUTPUT_CHARS])


def _plan(updates: SessionUpdates, update: dict) -> SessionEvent:
    entries = update.get("entries") or []
    lines = tuple(
        f"{entry.get('content', '')} ({entry.get('status', 'pending')})"
        for entry in entries
        if isinstance(entry, dict) and entry.get("content")
    )
    return PlanUpdate(lines)


def _user(updates: SessionUpdates, update: dict) -> SessionEvent:
    return UserText(block_text(update.get("content")))


PARSERS: Mapping[str, Callable[[SessionUpdates, dict], SessionEvent | None]] = {
    "agent_message_chunk": _message,
    "agent_thought_chunk": _thought,
    "tool_call": _card,
    "tool_call_update": _card_update,
    "plan": _plan,
    "user_message_chunk": _user,
}


class SessionUpdates:
    """One session's ``session/update`` notifications, read into events through its agent class."""

    def __init__(self, agent: AcpAgent, server: str):
        """Read a session of ``agent`` whose bridged tools are served under the MCP server name ``server``."""
        self.agent, self.server = agent, server
        self._unnamed: str | None = None  # the id the id-less message being streamed is read under

    def read(self, update: Any) -> SessionEvent | None:
        """The event one ``session/update`` carries, or ``None`` for a kind a run does not use."""
        if not isinstance(update, dict):
            return None
        parser = PARSERS.get(update.get("sessionUpdate"))
        if parser is None:
            return None
        if parser not in (_message, _thought):
            self._unnamed = None  # a tool call, a plan or a prompt ends the id-less message being streamed
        return parser(self, update)

    def continued(self) -> str:
        """The id of the id-less message being streamed: the one so far, or a fresh one after a tool, plan or prompt."""
        if self._unnamed is None:
            self._unnamed = str(uuid.uuid4())
        return self._unnamed
