# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A worker's session kept for the next task of the call, when that task's history continues what it read."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .bridge import WorkerDoor
    from .protocol import HarnessSession, Mailbox, Runtime


def item_key(item: Mapping[str, Any]) -> tuple:
    """Return what identifies one item of the IM's conversation in every task's view of it.

    A call is its id and name, an output its call's id (the IM cuts long outputs again), and a message its role, id and
    first part's text: later parts, such as the records, are only in the reading task's view.
    """
    kind = item.get("type", "message")
    if kind == "function_call_output":
        return kind, item.get("call_id")
    if kind == "function_call":
        return kind, item.get("call_id"), item.get("name")
    content = item.get("content")
    parts = [content] if isinstance(content, str) else [p.get("text") for p in content or () if isinstance(p, Mapping)]
    return kind, item.get("role"), item.get("id"), parts[0] if parts else None


def keys(items: Sequence[Mapping[str, Any]]) -> list[tuple]:
    """Return the keys of ``items`` (:func:`item_key`), in order."""
    return [item_key(item) for item in items]


@dataclass
class KeptSession:
    """A session a task answered in, kept for a later task with what that task's run takes over."""

    runtime: Runtime
    session: HarnessSession
    door: WorkerDoor
    mailbox: Mailbox
    instructions: str
    shown: list[tuple]
    total: int | None
    usage: Mapping[str, Any] | None
    tool_rows: bool

    def retire(self, bridge: Any) -> None:
        """Retire the session and close its door on ``bridge``; a kept session holds no call."""
        self.session.retire()
        bridge.close(self.door, "")
