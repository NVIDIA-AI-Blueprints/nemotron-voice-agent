# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whether a worker's session serves more than one task (``backend.harness.session``).

``task`` (the default) opens a fresh session for each task. ``call`` keeps the session of a task that answered in the
adapter's one slot, and the next task takes it when that task's ``history()`` begins with exactly the items the session
was shown or wrote, with more after them: the session then reads only the items that follow, as its next prompt. Items
compare by :func:`item_key`, so a request matches with or without the work records its task read in its message, and
an output matches by its call, however the IM cut it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .bridge import WorkerDoor
    from .protocol import HarnessSession, Mailbox, Runtime, Runtimes


def item_key(item: Mapping[str, Any]) -> tuple:
    """What identifies one item of the IM's conversation in every task's view of it.

    A call is its id and its name, and an output its call's id, since the IM cuts a long output again when it joins the
    shared conversation. A message is its role, its id and its first part's text: the records block and the closing
    line are later parts of a request's message, which only the view of the task that read the request carries.
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
    """The keys of ``items`` (:func:`item_key`), in order."""
    return [item_key(item) for item in items]


@dataclass
class KeptSession:
    """A session a task answered in, kept for a later task with what that task's run takes over."""

    runtime: Runtime  # its process: it holds no place there while it waits
    session: HarnessSession
    door: WorkerDoor  # its bridge door, which keeps its name and token for the next task
    mailbox: Mailbox
    instructions: str  # the composed instructions it was opened with
    shown: list[tuple]  # the keys of every item it was shown or wrote, in the IM's order
    total: int | None  # its token total and usage as last reported: ACP counts them across all of its turns
    usage: Mapping[str, Any] | None
    tool_rows: bool  # its transcript holds a tool result

    def retire(self, bridge: Any) -> None:
        """Never use it again: the session retired and its door closed on ``bridge`` (it holds no call)."""
        self.session.retire()
        bridge.close(self.door, "")


class PerTask:
    """``task``: a fresh session for each task; no session is kept."""

    @classmethod
    def from_settings(cls, block: Mapping[str, Any]) -> PerTask:
        """It reads no key."""
        return cls()

    def take(self, host: Runtimes) -> KeptSession | None:
        """None: the task opens a fresh session."""
        return None

    def keep(self, host: Runtimes, kept: KeptSession) -> bool:
        """Never: the run retires its session."""
        return False


class PerCall:
    """``call``: the adapter keeps one answered session, and the next task takes it (see the module's docstring)."""

    @classmethod
    def from_settings(cls, block: Mapping[str, Any]) -> PerCall:
        """It reads no key."""
        return cls()

    def take(self, host: Runtimes) -> KeptSession | None:
        """The session ``host`` keeps, which leaves it: two tasks never share a session."""
        kept, host.kept = host.kept, None
        return kept

    def keep(self, host: Runtimes, kept: KeptSession) -> bool:
        """Keep ``kept`` while ``host`` keeps none: the first session kept stays, and the run retires a later one."""
        if host.kept is not None:
            return False
        host.kept = kept
        return True
