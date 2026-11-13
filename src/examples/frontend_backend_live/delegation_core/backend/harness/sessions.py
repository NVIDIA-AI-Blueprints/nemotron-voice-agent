# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A call's idle worker sessions, kept for later tasks: a task that takes one sends it only what it has not read."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from examples.frontend_backend_live.delegation_core.backend.port import TaskContext

    from .bridge import WorkerDoor
    from .protocol import HarnessSession, Mailbox, Runtime


def item_key(item: Mapping[str, Any]) -> tuple:
    """Return what identifies one item of the IM's conversation in every task's view of it: never its text.

    A call and its output are their kind and the call's id; any other item is its kind and its id. A request's messages
    carry ids from its delegation's id (``with_ids``), and every other message has its own.
    """
    kind = item.get("type", "message")
    return kind, item.get("call_id") if kind in ("function_call", "function_call_output") else item.get("id")


def keys(items: Sequence[Mapping[str, Any]]) -> list[tuple]:
    """Return the keys of ``items`` (:func:`item_key`), in order."""
    return [item_key(item) for item in items]


@dataclass(frozen=True)
class Unread:
    """What a session has not read of a task's history: ``items``, or ``None`` with ``why`` when it does not fit."""

    items: list[dict] | None
    # The unread items before its last answer, such as another task's steps: it reads them after its own.
    before: int = 0
    # The items it read that the history does not hold: the budget trimmed them, and the agent still holds them.
    absent: int = 0
    why: str | None = None


def unread(read: Sequence[tuple], history: Sequence[dict]) -> Unread:
    """Return the items of ``history`` that a session has not read, given the keys of all it read, its last answer last.

    It fits when its last answer is in the history and something is new: every item whose key it has not read, in the
    history's order.
    """
    seen, held = set(read), keys(history)
    if not read or read[-1] not in held:
        return Unread(None, why="its last answer not in the history")
    items = [item for item, key in zip(history, held, strict=True) if key not in seen]
    if not items:
        return Unread(None, why="nothing new")
    end = held.index(read[-1])
    return Unread(items, sum(key not in seen for key in held[:end]), len(seen - set(held)))


@dataclass(eq=False)
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
    # Every Work and task it served, oldest first, and its last task.
    served: list[str]
    last_task: str
    # When it went idle, by time.monotonic().
    idle_since: float

    def retire(self, bridge: Any) -> bool:
        """Retire the session and close its door on ``bridge``; return whether a close went out (it holds no call)."""
        closed = self.session.retire()
        bridge.close(self.door, "")
        return closed


class SessionPool:
    """A call's idle worker sessions, at most ``bound``: the sessions of tasks that answered, until a task takes one."""

    def __init__(self, bound: int):
        """Keep at most ``bound`` idle sessions."""
        self.bound = bound
        self.idle: list[KeptSession] = []
        # Closed once the caller left, until a task starts: no task of the call is left to take a session.
        self.closed = False

    def keep(self, kept: KeptSession) -> KeptSession | None:
        """Pool ``kept``; return the session idle longest when the pool passes its bound, for the caller to retire.

        While the pool is closed, it returns ``kept`` itself.
        """
        if self.closed:
            return kept
        self.idle.append(kept)
        if len(self.idle) <= self.bound:
            return None
        evicted = min(self.idle, key=lambda each: each.idle_since)
        self.idle.remove(evicted)
        return evicted

    def candidates(self, task: TaskContext) -> list[KeptSession]:
        """Return the idle sessions of ``task``'s thread in the order it tries them, the most recently idle first.

        They are the sessions that served the Work or task its Work continues; a new thread has none. A session holds
        its earlier tasks' work as its own turns, so only a task that continues that work takes it. Any other task
        starts clean, and the idle sessions stay for their own threads, such as one that asked a question.
        """
        by_age = sorted(self.idle, key=lambda each: each.idle_since)
        named = task.continues
        return [each for each in reversed(by_age) if named is not None and named in each.served]

    def take(self, kept: KeptSession) -> None:
        """Take ``kept`` out of the pool, so two tasks never share it."""
        self.idle.remove(kept)

    def drain(self) -> list[KeptSession]:
        """Take every idle session out of the pool, for the caller to retire."""
        idle, self.idle = self.idle, []
        return idle

    def close(self) -> list[KeptSession]:
        """Close the pool until :meth:`open`, and take every idle session out of it, for the caller to retire."""
        self.closed = True
        return self.drain()

    def open(self) -> None:
        """Pool the sessions of later tasks again."""
        self.closed = False
