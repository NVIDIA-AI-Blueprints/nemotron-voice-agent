# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Group-aware conversation history with tool-call-safe pruning.

A flat sliding window over messages can separate an assistant tool-call batch
from its tool results, which most providers reject outright. History is
therefore modelled as *logical groups*::

    Group = user message
          + assistant message (text, or a tool-call batch)
          [+ all tool results for that batch
           + the assistant continuation (text, or another batch -> repeat)]

Pruning drops whole groups, oldest first, and never splits one. System prompts
are not stored here at all (they are rendered at call time), so "pinned" is
structural rather than a special case.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from examples.frontend_backend_verdict.text.messages import CONTINUES_TURN, Message


@dataclass(frozen=True, slots=True)
class History:
    """An immutable message list with group-aware operations."""

    messages: tuple[Message, ...] = ()

    def __len__(self) -> int:
        """Number of messages."""
        return len(self.messages)

    def __bool__(self) -> bool:
        """Whether any message is present."""
        return bool(self.messages)

    def append(self, *messages: Message) -> History:
        """Return a copy with ``messages`` appended."""
        return History(self.messages + tuple(messages))

    def extend(self, messages: Iterable[Message]) -> History:
        """Return a copy with an iterable of messages appended."""
        return History(self.messages + tuple(messages))

    def groups(self) -> tuple[tuple[Message, ...], ...]:
        """Split the history into logical groups, one per user message.

        Any messages preceding the first user message form a leading group so
        that no message is ever lost by grouping. A user message marked
        ``CONTINUES_TURN`` (speech while the turn's tool calls were out) stays in
        the group of the turn it was said in.
        """
        groups: list[list[Message]] = []
        for message in self.messages:
            if (message.role == "user" and not (message.meta or {}).get(CONTINUES_TURN)) or not groups:
                groups.append([message])
            else:
                groups[-1].append(message)
        return tuple(tuple(group) for group in groups)

    def prune(self, max_groups: int) -> History:
        """Return a copy keeping at most ``max_groups`` trailing logical groups.

        The newest group is always kept, so an in-flight (incomplete) group is
        never a pruning candidate, and a single oversized group survives intact.
        """
        if max_groups < 1:
            return self
        groups = self.groups()
        if len(groups) <= max_groups:
            return self
        kept = groups[-max_groups:]
        return History(tuple(message for group in kept for message in group))


def group_is_complete(group: Iterable[Message]) -> bool:
    """Whether every tool call in ``group`` has a result and vice versa.

    A complete group also ends with an assistant message that carries text, so
    the model's last word in the group is always a continuation rather than a
    dangling tool-call batch.
    """
    messages = tuple(group)
    if not messages:
        return False
    called: set[str] = set()
    answered: set[str] = set()
    for message in messages:
        for call in message.tool_calls:
            called.add(call.id)
        if message.role == "tool" and message.tool_call_id:
            answered.add(message.tool_call_id)
    if called != answered:
        return False
    last = messages[-1]
    return last.role == "assistant" and last.content is not None and not last.tool_calls


def validate_tool_pairing(messages: Iterable[Message]) -> None:
    """Raise ``ValueError`` when tool calls and tool results do not correspond."""
    called: set[str] = set()
    answered: set[str] = set()
    for message in messages:
        for call in message.tool_calls:
            called.add(call.id)
        if message.role == "tool" and message.tool_call_id:
            answered.add(message.tool_call_id)
    if orphan_results := answered - called:
        raise ValueError(f"tool results without a matching call: {sorted(orphan_results)}")
    if orphan_calls := called - answered:
        raise ValueError(f"tool calls without a matching result: {sorted(orphan_calls)}")
