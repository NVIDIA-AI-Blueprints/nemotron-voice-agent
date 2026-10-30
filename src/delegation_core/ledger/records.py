# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The work ledger's records: what the interaction manager knows about one session's work.

Each record has one state field and a table of the moves it allows (``MOVES``); ``move`` refuses any other move with a
``TransitionError``, so a wrong move fails at once instead of drifting. Only the interaction manager moves the records,
and a ``WorkLedger`` keeps one session's records in memory only.
"""

from __future__ import annotations

import time
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Literal

MAX_KEPT_CLOSED = 32

# Who runs a tool call: the app (a function its request declared; the port's ``Route`` names it alike), the backend
# (its tool sources), or the IM (its own work tools, such as show and cancel_work).
Runner = Literal["app", "backend", "im"]


class TransitionError(ValueError):
    """Raised for a move that a record's table does not allow."""


def moves(**table: tuple[str, ...]) -> dict[str, frozenset[str]]:
    """A table of allowed moves: each state, and the states it may move to (none for a final state)."""
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
        """Move to the state ``to``.

        Raises:
            TransitionError: The table does not allow the move.
        """
        if to not in self.MOVES[self.state]:
            raise TransitionError(f"{type(self).__name__} {self.id} may not move from {self.state} to {to}")
        self.state = to

    @property
    def ended(self) -> bool:
        """Whether the record is in a final state, one that its table lets nothing leave."""
        return not self.MOVES[self.state]


@dataclass(eq=False)
class Session(Record):
    """One IM session: ``active`` from its start, ``ended`` once it closed."""

    MOVES = moves(active=("ended",), ended=())
    START = "active"


@dataclass(eq=False)
class Delegation(Record):
    """One delegation: one request, and the responses that answer it. Its id is ``metadata.live_delegation_id``.

    ``queued`` from its admission until the Work it joined starts or its steering lands; ``running ⇄ waiting_client``
    while it carries running work (``waiting_client`` while its response waits for the app's outputs). It ends with its
    Work (``answered``, ``failed`` or ``cancelled``), or ``handed_over`` when a newer request takes its Work over
    (``handed_to`` names that taker, which the IM names on the wire), also while it waits for the app, once the
    client-tool deadline passed. While queued it can also end ``answered`` (a conductor's ``Answer``, which touches no
    Work). It never fails while queued: an arrival no decision placed is placed by the steer rule.
    """

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

    input: list[dict] = field(default_factory=list)  # its request's input items, as sent (a conductor reads them)
    settings: dict = field(default_factory=dict)  # what its request set (model, instructions, tools ...), as sent
    work: str = ""  # the Work it joined, once placed
    handed_to: str | None = None  # the newer delegation that took its Work over
    responses: list[str] = field(default_factory=list)  # its responses, in order


@dataclass(eq=False)
class Response(Record):
    """One Responses API response of a delegation, which only the IM's response book writes.

    ``queued`` from the request's acceptance (``response.created`` with status ``queued``, then ``response.queued``);
    ``in_progress`` once work runs for it, a handover, a conductor's ``Answer`` and a late continuation's empty answer
    included; then exactly one terminal, ``completed``, ``failed`` or ``incomplete`` (with status ``cancelled`` in its
    snapshot when the IM stopped it at close). Only a close ends a queued response without running it. A one-shot
    request's response opens in progress: the IM moves it there before any event.
    """

    MOVES = moves(
        queued=("in_progress", "incomplete"),
        in_progress=("completed", "failed", "incomplete"),
        completed=(),
        failed=(),
        incomplete=(),
    )
    START = "queued"

    delegation: str = ""  # the delegation it answers


@dataclass(eq=False)
class Stop(Record):
    """How a task's stop stands (only the IM stops work, and it confirms the stop). Its id is its task's.

    ``none → stop_requested`` when the IM's stop door takes the task: nothing more of it is dispatched or written. Then
    ``stopped`` once nothing of it runs (a queued task never ran; a running one's ``stop()`` confirmed it), or
    ``unconfirmed`` when the run could not confirm it within the stop limit.
    """

    MOVES = moves(none=("stop_requested",), stop_requested=("stopped", "unconfirmed"), stopped=(), unconfirmed=())
    START = "none"

    outcome: str | None = None  # once it ended, whether anything changed: none, done or unclear


@dataclass(eq=False)
class Result(Record):
    """A task's result: ``none → final → pushed → expired``. Its id is its task's.

    ``final`` once the task's round committed its answer, with its ``items`` (the ``show`` items, then the
    ``final_answer`` message), which never change then; ``pushed`` once they are written into the carrier's response
    (they wait while that response waits for the app's outputs); ``expired`` when its Work's records are dropped. Each
    task's answer is its own item, which nothing merges or rewords.
    """

    MOVES = moves(none=("final",), final=("pushed",), pushed=("expired",), expired=())
    START = "none"

    items: list[dict] = field(default_factory=list)  # the answer's items, in order


@dataclass(eq=False)
class Task(Record):
    """One task of a Work: one worker, one backend run. Its id is its Work's and its number: ``W1.1``, ``W1.2`` ...

    ``queued`` until the IM's scheduler starts it, while fewer tasks run than the adapter's ``max_running``;
    ``running ⇄ waiting_client`` (``waiting_client`` while the app owes the outputs of its calls); it ends
    ``answered``, ``failed`` or ``cancelled`` (by the stop door; a queued task is cancelled without running). It holds
    its own ``Stop`` and ``Result``.
    """

    MOVES = moves(
        queued=("running", "cancelled"),
        running=("waiting_client", "answered", "failed", "cancelled"),
        waiting_client=("running", "failed", "cancelled"),
        answered=(),
        failed=(),
        cancelled=(),
    )
    START = "queued"

    work: str = ""  # its Work
    brief: str | None = None  # its conductor's brief; None when it works on the requests as they came
    acted: bool = False  # it dispatched a call, so something may have changed
    progress: str | None = None  # its newest progress note
    started_at: float | None = None  # when it started running
    ended_at: float | None = None  # when it ended
    outcome: str | None = None  # at its end, whether anything changed: none, done or unclear
    stop: Stop = field(init=False)
    result: Result = field(init=False)

    def __post_init__(self) -> None:
        """Start queued, with no stop requested and no result yet."""
        super().__post_init__()
        self.stop, self.result = Stop(self.id), Result(self.id)


@dataclass(eq=False)
class Work(Record):
    """One unit of backend work: the requests it carries, run by its tasks.

    ``queued`` from the decision that made it (a conductor's ``Start``, or a follow-up for steering that never landed)
    until a task of it starts; ``running ⇄ waiting_client`` (``waiting_client`` while its response waits for the app's
    outputs, its other tasks running on); it ends ``answered``, ``failed`` or ``cancelled`` (the stop door) once its
    last task has ended. It never goes back to the queue: a newer request joins it as steering, and the newest request
    carries it.
    """

    MOVES = moves(
        queued=("running", "cancelled"),
        running=("waiting_client", "answered", "failed", "cancelled"),
        waiting_client=("running", "failed", "cancelled"),
        answered=(),
        failed=(),
        cancelled=(),
    )
    START = "queued"

    carrier: str = ""  # the delegation whose response carries its output now, and gets its answer
    requests: list[str] = field(default_factory=list)  # the delegations it carries, oldest first
    # The settings its response is written with: its starting request's; an explicit continuation or typed run brings
    # its own, and a join keeps them.
    settings: dict = field(default_factory=dict)
    tasks: list[Task] = field(default_factory=list)  # its tasks, in order
    queued_at: float = field(default_factory=time.monotonic)  # when the decision that made it was applied


@dataclass(eq=False)
class Steering(Record):
    """Requests that joined a running or waiting task and wait for its next safe point. Ids: S1, S2 ...

    ``waiting`` until the task's adapter takes it at a safe point (``take_boundary_input``, which hands over the port's
    ``Steering`` with these ``items``); then ``applied`` when the backend took it in (the IM joins it to the
    conversation and the newest request carries the Work), or ``not_applied``, which the next safe point takes again.
    Still unapplied when its task ends, it goes ``follow_up``: its requests start a follow-up Work, so nothing the
    caller said is lost.
    """

    MOVES = moves(
        waiting=("taken", "follow_up"),
        taken=("applied", "not_applied", "follow_up"),
        not_applied=("taken", "follow_up"),
        applied=(),
        follow_up=(),
    )
    START = "waiting"

    task: str = ""  # the task it joins
    delegations: list[str] = field(default_factory=list)  # the requests it carries, oldest first
    items: list[dict] = field(default_factory=list)  # their input, with the records block and the closing line


@dataclass(eq=False)
class Action(Record):
    """One tool call: ``started →`` ``done``, ``failed``, ``unclear`` or ``not_run``. Its id is its ``call_id``.

    ``unclear`` when its task ended with no report of it, so whether it ran is not known; ``not_run`` when the IM
    refused it and its output says why: while newer input waits (the commit gate), after a stop was
    requested or the task's own ``cancel_work``, for a function the app and a work tool both name, or for an app call
    that repeats the newest call that ran, which its task made and the app answered, with no request since (its output
    is that answer).
    """

    MOVES = moves(started=("done", "failed", "unclear", "not_run"), done=(), failed=(), unclear=(), not_run=())
    START = "started"

    task: str = ""  # the task whose round made it
    work: str = ""  # that task's Work
    name: str = ""  # the function's name
    runner: Runner | None = None  # who runs it; ``WorkLedger.new_action`` always names it
    output: str | None = None  # its output, once it ended


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
        """Record a new queued Work for ``requests`` (oldest first; the newest carries it) and its queued tasks.

        It has one task per brief, or one on the requests as they came when there is none. A session numbers its Works
        W1, W2 ... and each Work its tasks: W1.1, W1.2 ...
        """
        self._works_made += 1
        work_id = f"W{self._works_made}"
        work = Work(work_id, carrier=requests[-1], requests=list(requests), settings=dict(settings))
        for number, brief in enumerate(briefs or (None,), start=1):
            task = self.tasks[f"{work_id}.{number}"] = Task(f"{work_id}.{number}", work=work_id, brief=brief)
            work.tasks.append(task)
        self.works[work_id] = work
        return work

    def new_steering(self, task_id: str, delegations: Sequence[str], items: Sequence[dict]) -> Steering:
        """Record new waiting steering of the task ``task_id``: the requests it carries, and the items it hands over."""
        self._steerings_made += 1
        steering_id = f"S{self._steerings_made}"
        steering = Steering(steering_id, task=task_id, delegations=list(delegations), items=list(items))
        self.steerings[steering_id] = steering
        return steering

    def new_action(self, call_id: str, task_id: str, name: str, runner: Runner) -> Action:
        """Record a started tool call that a round of the task ``task_id`` made, and who runs it."""
        action = Action(call_id, task=task_id, work=self.tasks[task_id].work, name=name, runner=runner)
        self.actions[call_id] = action
        return action

    def prune(self) -> list[Work]:
        """Forget what no open or recent Work needs, and return the Works forgotten.

        Every open Work and the newest ``MAX_KEPT_CLOSED`` closed ones are kept with their tasks, actions and steering,
        and so are the delegations a kept Work carries and their responses; of the other delegations and responses, the
        open ones and the newest ``MAX_KEPT_CLOSED`` closed ones are kept. The IM expires the forgotten Works' results.
        """
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
    """The ids of the closed records no kept Work needs, oldest first."""
    return [key for key, record in records.items() if record.ended and key not in needed]


def _forget(records: dict[str, Record], needed: Collection[str]) -> None:
    """Forget the closed records beyond the newest ``MAX_KEPT_CLOSED``, except those a kept Work needs."""
    for key in _closed(records, needed)[:-MAX_KEPT_CLOSED]:
        del records[key]
