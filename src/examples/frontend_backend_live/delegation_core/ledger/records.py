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
class Placement:
    """A source's exact target, private to the ledger and absent from ordinary model records."""

    work: str
    tasks: tuple[str, ...]


@dataclass(eq=False)
class Delegation(Record):
    """One delegation: one request and the responses that answer it; its id is ``metadata.live_delegation_id``."""

    # An unplaced request can fail if its unqualified native control prompt loses its attachment with effects unknown.
    MOVES = moves(
        queued=("running", "answered", "handed_over", "failed", "cancelled"),
        running=("waiting_client", "answered", "handed_over", "failed", "cancelled"),
        waiting_client=("running", "answered", "handed_over", "failed", "cancelled"),
        answered=(),
        handed_over=(),
        failed=(),
        cancelled=(),
    )
    START = "queued"

    input: list[dict] = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    work: str = ""
    tasks: tuple[str, ...] = ()
    responses: list[str] = field(default_factory=list)
    placements: tuple[Placement, ...] = ()
    call_generation: int = 0
    caller_excerpt: str | None = None


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
    # A later positive resource proof supplements the original bounded stop result; it never rewrites it.
    quiescent_later: bool = False


@dataclass(eq=False)
class Result(Record):
    """A task's result: ``final`` once answered, ``pushed`` once written, ``expired`` once forgotten."""

    MOVES = moves(none=("final",), final=("pushed",), pushed=("expired",), expired=())
    START = "none"


@dataclass(frozen=True)
class InputReceipt:
    """An actual task-local application of caller messages, not a model-authored revision."""

    id: str
    revision: int
    source_message_ids: tuple[str, ...]
    caller_order: tuple[tuple[str, int | None], ...] = ()


@dataclass(eq=False)
class Question:
    """A private question identity, its written evidence and its explicit resolution."""

    id: str
    text: str
    call_id: str
    input_revision: int
    dispatch: str = "pending"
    resolution: str = "open"
    input_ref: str | None = None
    creator_task_id: str = ""
    owner_task_id: str = ""
    response_id: str | None = None
    item_id: str | None = None
    publication_revision: int = 0
    publication_call_id: str = ""
    created_input_revision: int = 0
    previous_owners: tuple[str, ...] = ()
    input_caller_order: int = 0

    @property
    def open(self) -> bool:
        """Whether the task still owns this unresolved question."""
        return self.resolution == "open"


@dataclass(eq=False)
class Task(Record):
    """One task of a Work: one worker, one backend run; its id is its Work's and its number, such as ``W1.2``."""

    MOVES = moves(
        queued=("running", "pausing", "paused", "failed", "cancelled"),
        running=("waiting_client", "pausing", "paused", "answered", "failed", "cancelled"),
        waiting_client=("running", "pausing", "paused", "failed", "cancelled"),
        pausing=("paused", "queued", "waiting_client", "failed", "cancelled"),
        paused=("queued", "waiting_client", "pausing", "failed", "cancelled"),
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
    questions: list[Question] = field(default_factory=list)
    question_sequence: int = 0
    input_revision: int = 0
    input_receipts: dict[str, InputReceipt] = field(default_factory=dict)
    publications: dict[str, dict] = field(default_factory=dict)
    continues: str | None = None
    verifies: str | None = None
    # Canonical lineage survives pruning the successor's own records/history.
    successor_task_id: str | None = None
    successor_work_id: str | None = None
    # An explicit stop can close an ended thread's unresolved questions without changing its physical stop state.
    continuation_cancelled_by: str | None = None
    history_mode: str = "shared"
    answer: str | None = None
    answer_item_id: str | None = None
    answer_publication: str = "none"
    started_at: float | None = None
    ended_at: float | None = None
    stop: Stop = field(init=False)
    result: Result = field(init=False)

    def __post_init__(self) -> None:
        """Start queued, with no stop requested and no result yet."""
        super().__post_init__()
        self.stop, self.result = Stop(self.id), Result(self.id)

    @property
    def retained_answer(self) -> str | None:
        """Return the answer eligible for conversation context."""
        return None if self.answer_publication == "superseded" else self.answer

    def without_superseded_answer(self, items: list[dict]) -> list[dict]:
        """Exclude just the superseded final item, preserving other task evidence."""
        if self.answer_publication != "superseded" or self.answer_item_id is None:
            return items
        return [item for item in items if item.get("id") != self.answer_item_id]

    def close_questions(self, disposition: str) -> tuple[Question, ...]:
        """Resolve only this record's owned questions after their publication has settled."""
        owned = [q for q in self.questions if q.owner_task_id == self.id and q.open]
        closed = tuple(q for q in owned if disposition != "failed" or q.dispatch != "written")
        for question in closed:
            question.resolution = disposition
        self.question = "\n".join(q.text for q in owned if q.open) or None
        return closed


@dataclass(eq=False)
class Work(Record):
    """One unit of backend work: the requests it carries, run by its tasks; the newest to join carries it."""

    MOVES = moves(
        queued=("running", "pausing", "paused", "failed", "cancelled"),
        running=("queued", "waiting_client", "pausing", "paused", "answered", "failed", "cancelled"),
        waiting_client=("queued", "running", "pausing", "paused", "answered", "failed", "cancelled"),
        pausing=("queued", "running", "waiting_client", "paused", "answered", "failed", "cancelled"),
        paused=("queued", "running", "waiting_client", "pausing", "answered", "failed", "cancelled"),
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
    continues: str | None = None
    after_call: bool = False


@dataclass(eq=False)
class Steering(Record):
    """Requests that joined a running or waiting task and wait for its next safe point; its ids are S1, S2 ..."""

    # Unapplied when its task ends, it goes follow_up: its requests start a follow-up Work, unless their call has ended.
    MOVES = moves(
        waiting=("taken", "follow_up", "cancelled"),
        taken=("applied", "not_applied", "unknown", "follow_up", "cancelled"),
        not_applied=("taken", "follow_up", "cancelled"),
        applied=(),
        unknown=(),
        follow_up=(),
        cancelled=(),
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
    arguments: str | None = None
    app_published: bool = False
    output: str | list[dict] | None = None
    late_output: str | list[dict] | None = None
    late_outcome: str | None = None


@dataclass
class ControlAction:
    """A declared routing obligation, with host identity and independent execution evidence."""

    id: str
    kind: str
    sources: tuple[str, ...]
    state: str = "pending"
    receipt: dict = field(default_factory=dict)
    delivery: dict[str, str] = field(default_factory=dict)


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
        self.controls: dict[str, ControlAction] = {}
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

    def prune(self, needed: Collection[str] = (), sources: Collection[str] = ()) -> list[Work]:
        """Forget what no open or recent Work needs, and return the Works forgotten."""
        forgotten = [self.works.pop(key) for key in _closed(self.works, needed)[:-MAX_KEPT_CLOSED]]
        gone = {task.id for work in forgotten for task in work.tasks}
        for task_id in gone:
            del self.tasks[task_id]
        for records in (self.actions, self.steerings):
            for key in [key for key, record in records.items() if record.task in gone]:
                del records[key]
        carried = {delegation for work in self.works.values() for delegation in work.requests} | set(sources)
        _forget(self.delegations, carried)
        kept = carried & set(self.delegations)
        _forget(self.responses, {response for key in kept for response in self.delegations[key].responses})
        for key, control in list(self.controls.items()):
            if control.state != "pending" and not any(source in self.delegations for source in control.sources):
                del self.controls[key]
        return forgotten


def _closed(records: dict[str, Record], needed: Collection[str] = ()) -> list[str]:
    """Return the ids of the closed records no kept Work needs, oldest first."""
    return [key for key, record in records.items() if record.ended and key not in needed]


def _forget(records: dict[str, Record], needed: Collection[str]) -> None:
    """Forget the closed records beyond the newest ``MAX_KEPT_CLOSED``, except those a kept Work needs."""
    for key in _closed(records, needed)[:-MAX_KEPT_CLOSED]:
        del records[key]
