# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The work ledger's records: what the interaction manager knows about one session's work."""

from __future__ import annotations

import time
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Literal

MAX_KEPT_CLOSED = 32

# Who runs a tool call: the app, the backend (its tool sources), or the IM (its own work tools).
Runner = Literal["app", "backend", "im"]


class TransitionError(ValueError):
    """Raised for a move that a record's table does not allow, so a wrong move fails at once."""


def moves(**table: tuple[str, ...]) -> dict[str, frozenset[str]]:
    """Return a table of allowed moves: each state, and the states it may move to (none for a final state)."""
    return {state: frozenset(targets) for state, targets in table.items()}


@dataclass(eq=False)
class Record:
    """A record with one state field, which only ``move`` changes, along the table ``MOVES``."""

    MOVES: ClassVar[dict[str, frozenset[str]]] = {}
    START: ClassVar[str] = ""

    id: str
    state: str = field(init=False)

    def __post_init__(self) -> None:
        """Start in ``START``."""
        self.state = self.START

    def move(self, to: str) -> None:
        """Move to the state ``to``; ``TransitionError`` when the table does not allow it."""
        if to not in self.MOVES[self.state]:
            raise TransitionError(f"{type(self).__name__} {self.id} may not move from {self.state} to {to}")
        self.state = to

    @property
    def ended(self) -> bool:
        """Whether the record is in a final state, one that its table lets nothing leave."""
        return not self.MOVES[self.state]


@dataclass(eq=False)
class Session(Record):
    """One IM session: ``active`` while a call is attached, ``detached`` while it waits for a later call, ``ended``."""

    MOVES = moves(active=("detached", "ended"), detached=("active", "ended"), ended=())
    START = "active"


@dataclass(eq=False)
class Delegation(Record):
    """One delegation: one request and the responses that answer it; its id is ``metadata.live_delegation_id``."""

    # It never fails while queued: an arrival no decision placed is placed by the IM's fallback rule.
    MOVES = moves(
        queued=("running", "answered", "handed_over", "cancelled"),
        running=("waiting_client", "answered", "handed_over", "failed", "cancelled"),
        waiting_client=("running", "handed_over", "failed", "cancelled"),
        answered=(),
        handed_over=(),
        failed=(),
        cancelled=(),
    )
    START = "queued"

    input: list[dict] = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    work: str = ""
    # The tasks of its Work that it joined, once placed.
    tasks: tuple[str, ...] = ()
    responses: list[str] = field(default_factory=list)


@dataclass(eq=False)
class Response(Record):
    """One Responses API response of a delegation, which only the IM's response book writes."""

    MOVES = moves(
        queued=("in_progress", "incomplete"),
        in_progress=("completed", "failed", "incomplete"),
        completed=(),
        failed=(),
        incomplete=(),
    )
    START = "queued"

    delegation: str = ""


@dataclass(eq=False)
class Stop(Record):
    """A task's stop, with its task's id: requested, then ``stopped``, or ``unconfirmed`` at the stop limit."""

    MOVES = moves(none=("stop_requested",), stop_requested=("stopped", "unconfirmed"), stopped=(), unconfirmed=())
    START = "none"


@dataclass(eq=False)
class Result(Record):
    """A task's result: ``final`` once answered, ``pushed`` once written, ``expired`` once forgotten."""

    MOVES = moves(none=("final",), final=("pushed",), pushed=("expired",), expired=())
    START = "none"


@dataclass(eq=False)
class Task(Record):
    """One task of a Work: one worker, one backend run; its id is its Work's and its number, such as ``W1.2``."""

    MOVES = moves(
        queued=("running", "cancelled"),
        running=("waiting_client", "answered", "failed", "cancelled"),
        waiting_client=("running", "failed", "cancelled"),
        answered=(),
        failed=(),
        cancelled=(),
    )
    START = "queued"

    work: str = ""
    brief: str | None = None
    acted: bool = False
    progress: str | None = None
    question: str | None = None
    # The text of its answer, once it answered.
    answer: str | None = None
    started_at: float | None = None
    ended_at: float | None = None
    stop: Stop = field(init=False)
    result: Result = field(init=False)

    def __post_init__(self) -> None:
        """Start queued, with no stop requested and no result yet."""
        super().__post_init__()
        self.stop, self.result = Stop(self.id), Result(self.id)


@dataclass(eq=False)
class Work(Record):
    """One unit of backend work: the requests it carries, run by its tasks; the newest to join carries it."""

    MOVES = moves(
        queued=("running", "cancelled"),
        running=("waiting_client", "answered", "failed", "cancelled"),
        waiting_client=("running", "failed", "cancelled"),
        answered=(),
        failed=(),
        cancelled=(),
    )
    START = "queued"

    carrier: str = ""
    requests: list[str] = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    tasks: list[Task] = field(default_factory=list)
    queued_at: float = field(default_factory=time.monotonic)
    # The Work or task (W1, W1.2) whose thread it continues, by the IM's placement; None for a new thread.
    continues: str | None = None
    # Set once a task's continue_after_call was written: the Work goes on after the call ends.
    after_call: bool = False


@dataclass(eq=False)
class Steering(Record):
    """Requests that joined a running or waiting task and wait for its next safe point; its ids are S1, S2 ..."""

    # Unapplied when its task ends, it goes follow_up: its requests start a follow-up Work, unless their call has ended.
    MOVES = moves(
        waiting=("taken", "follow_up"),
        taken=("applied", "not_applied", "follow_up"),
        not_applied=("taken", "follow_up"),
        applied=(),
        follow_up=(),
    )
    START = "waiting"

    task: str = ""


@dataclass(eq=False)
class Action(Record):
    """One tool call: ``started``, then ``done``, ``failed``, ``unclear`` or ``not_run``; its id is its ``call_id``."""

    MOVES = moves(started=("done", "failed", "unclear", "not_run"), done=(), failed=(), unclear=(), not_run=())
    START = "started"

    task: str = ""
    work: str = ""
    name: str = ""
    runner: Runner | None = None
    output: str | None = None


class WorkLedger:
    """One IM session's records, in memory: every open record, what open Works need, and the newest closed ones."""

    def __init__(self, session_id: str):
        """Open the ledger of the IM session ``session_id``."""
        self.session = Session(session_id)
        self.delegations: dict[str, Delegation] = {}
        self.responses: dict[str, Response] = {}
        self.works: dict[str, Work] = {}
        self.tasks: dict[str, Task] = {}
        self.steerings: dict[str, Steering] = {}
        self.actions: dict[str, Action] = {}
        self._works_made = 0
        self._steerings_made = 0

    def new_delegation(
        self, delegation_id: str, *, input_items: list[dict], settings: dict | None = None
    ) -> Delegation:
        """Record a new queued delegation with its request: its input items, and what it set."""
        delegation = Delegation(delegation_id, input=list(input_items), settings=dict(settings or {}))
        self.delegations[delegation_id] = delegation
        return delegation

    def new_response(self, response_id: str, delegation_id: str) -> Response:
        """Record a new queued response of the delegation ``delegation_id``."""
        response = self.responses[response_id] = Response(response_id, delegation=delegation_id)
        return response

    def new_work(self, requests: Sequence[str], settings: dict, briefs: Sequence[str] = ()) -> Work:
        """Record a new queued Work for ``requests``, oldest first, with one queued task per brief (one with none)."""
        self._works_made += 1
        work_id = f"W{self._works_made}"
        work = Work(work_id, carrier=requests[-1], requests=list(requests), settings=dict(settings))
        for number, brief in enumerate(briefs or (None,), start=1):
            task = self.tasks[f"{work_id}.{number}"] = Task(f"{work_id}.{number}", work=work_id, brief=brief)
            work.tasks.append(task)
        self.works[work_id] = work
        return work

    def new_steering(self, task_id: str) -> Steering:
        """Record new waiting steering of the task ``task_id``."""
        self._steerings_made += 1
        steering_id = f"S{self._steerings_made}"
        steering = Steering(steering_id, task=task_id)
        self.steerings[steering_id] = steering
        return steering

    def new_action(self, call_id: str, task_id: str, name: str, runner: Runner) -> Action:
        """Record a started tool call that a round of the task ``task_id`` made, and who runs it."""
        action = Action(call_id, task=task_id, work=self.tasks[task_id].work, name=name, runner=runner)
        self.actions[call_id] = action
        return action

    def prune(self) -> list[Work]:
        """Forget what no open or recent Work needs, and return the Works forgotten."""
        forgotten = [self.works.pop(key) for key in _closed(self.works)[:-MAX_KEPT_CLOSED]]
        gone = {task.id for work in forgotten for task in work.tasks}
        for task_id in gone:
            del self.tasks[task_id]
        for records in (self.actions, self.steerings):
            for key in [key for key, record in records.items() if record.task in gone]:
                del records[key]
        carried = {delegation for work in self.works.values() for delegation in work.requests}
        _forget(self.delegations, carried)
        kept = carried & set(self.delegations)
        _forget(self.responses, {response for key in kept for response in self.delegations[key].responses})
        return forgotten


def _closed(records: dict[str, Record], needed: Collection[str] = ()) -> list[str]:
    """Return the ids of the closed records no kept Work needs, oldest first."""
    return [key for key, record in records.items() if record.ended and key not in needed]


def _forget(records: dict[str, Record], needed: Collection[str]) -> None:
    """Forget the closed records beyond the newest ``MAX_KEPT_CLOSED``, except those a kept Work needs."""
    for key in _closed(records, needed)[:-MAX_KEPT_CLOSED]:
        del records[key]
