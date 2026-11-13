# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The port between the interaction manager (IM) and a backend adapter: only Responses shapes and these types pass.

The IM imports no other backend module. The IM owns admission, placement, the records, the conversation, steering's
timing, stops and every response the client reads; the adapter owns model rounds, providers, tools, conductors and
worker sessions. :class:`AdapterSource` is one per core, :class:`BackendAdapter` one per IM session at a time, and
:class:`BackendRun` one per task.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, runtime_checkable

from examples.shared.delegation_contract import is_progress

if TYPE_CHECKING:
    from examples.frontend_backend_live.delegation_core.config import Texts
    from examples.shared.delegation_contract import ModelEndpoint

# Who runs a call: the app (a function the request declared), the backend, or none (refused: its output says why).
Route = Literal["app", "backend", "refused"]
# ``unclear``: the call may have taken effect, so the IM answers it with texts.owed_output and the backend checks first.
Outcome = Literal["done", "failed", "unclear"]
Delivery = Literal["applied", "not_applied"]
StopOutcome = Literal["stopped", "unconfirmed"]
OpenState = Literal["queued", "running", "waiting_client"]
EndedState = Literal["answered", "failed", "cancelled"]
WorkState = OpenState | EndedState
Phase = Literal["commentary", "final_answer"]


def function_call_item(raw: dict) -> dict | None:
    """Return the ``function_call`` item a ``response.output_item.done`` event finishes, or ``None``."""
    item = raw.get("item")
    finished = raw.get("type") == "response.output_item.done" and isinstance(item, dict)
    return item if finished and item.get("type") == "function_call" else None


def message_phase(item: dict) -> Phase:
    """Return a message item's phase; a message with no phase counts as the answer."""
    return "commentary" if is_progress(item) else "final_answer"


class ToolSource(Protocol):
    """Tools that run on the server side: a tool set the backend runs (``backend.tools``) or the IM's work tools."""

    # The server label of its calls' mcp_call items.
    label: str

    def tools(self) -> list[dict]:
        """Return the function tools it offers, in the Responses shape."""

    async def execute(self, call_id: str, name: str, arguments: str) -> str:
        """Run one call and return its output; an unknown name returns an error output instead of raising.

        A repeated ``call_id`` adds nothing a second time.
        """


# A fresh set of the server tool sources, made by the root for each IM session.
ToolSources = Callable[[], Sequence[ToolSource]]


def allowed_tools(choice: Any) -> list[dict] | None:
    """Return the tools an ``allowed_tools`` tool choice lists, or ``None`` for another tool choice."""
    if not (isinstance(choice, dict) and choice.get("type") == "allowed_tools"):
        return None
    tools = choice.get("tools")
    return [tool for tool in tools if isinstance(tool, dict)] if isinstance(tools, list) else []


def listed(tool: dict, allowed: list[dict] | None) -> bool:
    """Return whether a round may call ``tool``: always with no list, else when every field of an entry matches."""
    return allowed is None or any(all(tool.get(key) == value for key, value in entry.items()) for entry in allowed)


class AppOutputStep(Protocol):
    """A step the app's function outputs take before the backend reads them (``backend.app_output_steps``)."""

    def outputs(self, calls: Sequence[tuple[dict, dict]], tools: Sequence[Mapping] | None) -> dict[str, dict]:
        """Return the outputs of the app's ``calls`` (call, output item) as the backend reads them, by call id.

        ``tools`` are the app's functions of the request that brought the outputs; an output the step changes is a copy.
        """


# Makes one IM session's app output steps, given its trace hook.
AppOutputSteps = Callable[[Callable[..., Any]], Sequence[AppOutputStep]]
# A model name to its endpoint, resolved by the root from the catalog; ``None`` gives the configured default.
EndpointLookup = Callable[[str | None], "ModelEndpoint"]


@dataclass(frozen=True)
class Steering:
    """Input that joined a running task and waits for the adapter's safe point (one batch)."""

    id: str
    items: tuple[dict, ...]


@dataclass(frozen=True)
class BoundaryInput:
    """What a safe point takes: the app's outputs, joined and in call order, and at most one steering batch."""

    outputs: tuple[dict, ...]
    steering: Steering | None = None


class TaskContext(Protocol):
    """What the IM gives one task's run: what to read and where to report. The IM implements it, one per task."""

    work_id: str
    task_id: str
    texts: Texts
    # The Work or task (W1, W1.2) whose thread this task's Work continues, by the IM's placement; None for a new
    # thread. An adapter that keeps sessions serves the task only from an idle session of that thread, else from a
    # fresh one.
    continues: str | None
    # Whether the task's Work continues after the call (a task's continue_after_call was written), read when asked:
    # the call's per-turn limit does not apply to its turns.
    after_call: bool

    def request(self) -> dict:
        """Return the settings of the task's next model call: model, composed instructions and the app's functions.

        Every adapter puts ``instructions`` where its backend takes a system prompt: every round for an LLM, each
        worker session's system prompt for a harness.
        """

    def history(self) -> list[dict]:
        """Return this task's view of the conversation, as Responses input items within the IM's budget.

        The shared conversation, the Work's input, the task's brief, then its own rounds, outputs and steering. An ended
        Work's items join the shared conversation in order, so a later task's view begins with the view of that Work's
        first task, unless the budget trimmed it or other items joined in between. Each message carries an id, and each
        call and its output a call_id, the same in every task's view: a request's messages carry ids from its
        delegation's id, except a one-shot session's, which are as sent.
        """

    def tools(self) -> ToolSource:
        """Return the IM's work tools for this task, such as ``show``, ``cancel_work`` and ``ask_caller``.

        A backend asks the caller only through ``ask_caller`` or the task's answer.
        """

    def route(self, call: dict) -> Route:
        """Return who runs ``call``; the IM decides once per call, before it runs."""

    def refusal(self, call_id: str) -> str:
        """Return the output the IM gave a call it refused, which says why."""

    def tool_started(self, call: dict, source: str) -> None:
        """Report a call the backend runs as it starts; ``source`` is the label of the source that runs it.

        A call that started with no ``route()`` asked, such as an agent's own tool, is the backend's.
        """

    def tool_outcome(self, call_id: str, output: str, outcome: Outcome) -> None:
        """Report that a call the backend ran ended; an ``unclear`` call's ``output`` says why, for the logs only."""

    def progress(self, text: str) -> None:
        """Report activity that is not model text, such as a harness's plan: a progress note, never history."""

    def take_boundary_input(self) -> BoundaryInput:
        """Join the app's outputs to the conversation and hand over waiting steering: a safe point.

        Safe to call at any point, any number of times: a second call joins nothing twice.
        """

    def settle(self, steering: Steering, delivery: Delivery) -> None:
        """Report what became of a steering batch the adapter took, once per batch.

        ``applied``: the IM joins it to the conversation, and its newest request carries the Work. ``not_applied``: the
        IM keeps it for the next safe point, or for a follow-up Work once the task ends.
        """

    def trace(self, kind: str, **fields: Any) -> None:
        """Note a trace of this task for the logs; the IM adds the Work, the task and the carrier."""


class BackendRun(Protocol):
    """One task's run on a backend: one worker, whose events keep these rules.

    Each attempt opens with ``response.created`` and ends with one terminal; only ``response.completed`` ends a round.
    Each item is announced by ``output_item.added`` and finished by ``output_item.done``, which the IM reads.
    Every message has a ``phase``; a round's calls follow its messages, and a round with calls has no answer.
    The iteration ends after an answer, after calls the app runs (then :meth:`resume`), or by raising.
    Each backend call runs once, reported by ``tool_started`` and ``tool_outcome``; each steering batch by ``settle``.
    No attempt that dispatched a call or delivered an item is retried.
    A run ends within its adapter's limits, except the turn limit of a task whose Work continues after the call, and
    :meth:`stop` returns within its stop limit.
    """

    def events(self) -> AsyncIterator[dict]:
        """Yield the task's Responses events, attempt by attempt; called again after :meth:`resume`."""

    def steer(self) -> None:
        """Wake the run: steering waits, which the adapter takes with ``take_boundary_input()`` at its next safe point.

        The LLM run redoes a round still streaming with nothing dispatched; a harness worker steers as its agent class
        does (``steering.agent_steering``). A round that answers first ends the task.
        """

    async def resume(self) -> None:
        """Take the app's outputs with ``take_boundary_input()``; then ``events()`` goes on."""

    async def stop(self) -> StopOutcome:
        """Stop the task: return once nothing of it runs, or ``unconfirmed`` at the adapter's stop limit.

        It never cuts a running call that changes things, and nothing is dispatched after it.
        """


@dataclass(frozen=True)
class Request:
    """One delegation's request, as a conductor reads it: an arrival, or a request a Work carries."""

    delegation_id: str
    input: tuple[dict, ...]


@dataclass(frozen=True)
class TaskSummary:
    """One task of a Work, as a conductor and the records block read it."""

    task_id: str
    state: WorkState
    # None when the task works on the requests as they came.
    brief: str | None
    steps: tuple[str, ...]
    # A call was dispatched, so something may have changed.
    acted: bool
    progress: str | None
    seconds: float
    # The question it asked the caller and still waits on; None when it holds none.
    question: str | None = None
    # The start of its answer, only for a task whose Work continues after the call; None for any other task.
    answer: str | None = None


@dataclass(frozen=True)
class WorkSummary:
    """One Work, open or recently ended, as a conductor and the records block read it."""

    work_id: str
    state: WorkState
    # The delegation whose response carries the Work's output.
    carrier: str
    requests: tuple[Request, ...]
    tasks: tuple[TaskSummary, ...]
    seconds: float
    # A task's continue_after_call was written: the Work goes on after the call ends.
    after_call: bool = False


@dataclass(frozen=True)
class ConductorInput:
    """What one ``decide()`` call reads."""

    arrivals: tuple[Request, ...]
    # The open Works, then the newest ended ones in the order they ended, so the records answer a status question about
    # finished work.
    works: tuple[WorkSummary, ...]
    conversation: tuple[dict, ...]
    # The index in ``conversation`` of the first item this conductor was not given before.
    new_from: int = 0


class DecisionTarget(Protocol):
    """What applies a conductor's decisions, the IM session: each method returns the conductor's outcome text."""

    async def start(self, decision: Start) -> str:
        """Start a new Work for these requests, with its tasks; it runs when a task slot is free.

        Above one task at a time, it is refused when it would make more than ``max_running`` running and queued tasks.
        """

    async def steer(self, decision: Steer) -> str:
        """Join these requests to an open Work (one task, or every open task); the IM decides when they land.

        A Steer to an ended Work or task starts a new Work that continues it.
        """

    async def cancel(self, decision: Cancel) -> str:
        """Stop a Work, or one task of it, through the stop door; return once the stop is requested."""

    async def answer(self, decision: Answer) -> str:
        """End a delegation with this answer from the records; no Work changes."""


@dataclass(frozen=True)
class Start:
    """Start a new Work for these requests; the newest one carries the Work's response."""

    delegations: tuple[str, ...]
    # The conductor's brief for each task, which run in parallel; () is one task on the requests as they came.
    tasks: tuple[str, ...] = ()

    async def apply(self, target: DecisionTarget) -> str:
        """Apply this decision to ``target``."""
        return await target.start(self)


@dataclass(frozen=True)
class Steer:
    """These requests join an open Work: one task of it, or every open task when ``task_id`` is ``None``."""

    work_id: str
    delegations: tuple[str, ...]
    task_id: str | None = None

    async def apply(self, target: DecisionTarget) -> str:
        """Apply this decision to ``target``."""
        return await target.steer(self)


@dataclass(frozen=True)
class Cancel:
    """Stop a Work, or one task of it when ``task_id`` names one; the other tasks keep running."""

    work_id: str
    task_id: str | None = None

    async def apply(self, target: DecisionTarget) -> str:
        """Apply this decision to ``target``."""
        return await target.cancel(self)


@dataclass(frozen=True)
class Answer:
    """End this delegation with ``text`` as its answer, from the records; it touches no Work."""

    delegation_id: str
    text: str

    async def apply(self, target: DecisionTarget) -> str:
        """Apply this decision to ``target``."""
        return await target.answer(self)


Decision = Start | Steer | Cancel | Answer
Place = Callable[[Decision], Awaitable[str]]


class ControlTools(Protocol):
    """A model conductor's vocabulary from the core's texts: its instructions, control tools and two answers."""

    prompt: str
    # The outcome of a control call that comes before a decide() takes its calls.
    idle: str
    # The answer to a control call that raised: holds {name} and {error}.
    failed: str

    def instructions(self, max_running: int) -> str:
        """Return the conductor's instructions: ``prompt`` with the tasks its backend runs at once filled in."""

    def conductor_tools(self) -> list[dict]:
        """Return the four control tools, as Responses function tools."""

    def decision(self, name: str, arguments: str) -> Decision:
        """Return the decision one control call makes; ``ValueError`` for a malformed call, which is skipped."""


class Conductor(Protocol):
    """Places the IM session's requests: start Works and their tasks, steer, cancel, or answer from the records."""

    async def decide(self, given: ConductorInput, place: Place) -> None:
        """Place decisions through ``place``, each awaited for its outcome text, then return.

        The IM calls it one at a time within a call; an arrival may cancel it once before it placed anything, and the
        conductor may let its agent finish that turn before it ends. A call's end cancels it without waiting, so a later
        call's ``decide()`` calls may start while it still runs. Concurrent ``place()`` calls apply in order, and one
        after it ended is refused. What it leaves unplaced, also on a timeout or an error, the IM places by its fallback
        rule. The conductor never runs work itself.
        """


class BackendAdapter(Protocol):
    """One IM session's backend (never stacked): its conductor and the runs of its tasks."""

    max_running: int

    def conductor(self) -> Conductor:
        """Return the session's conductor; the IM calls this once per adapter."""

    def start(self, task: TaskContext) -> BackendRun:
        """Return one task's run, before any model call; the run starts with ``events()``."""

    def release_idle(self) -> None:
        """Release every session that no running task holds, and keep none until a task starts: the caller left."""

    async def close(self) -> None:
        """Release the session's connections and its conductor."""


class AdapterSource(Protocol):
    """One kind of backend, built once per delegation core from its own block of the core's settings."""

    @classmethod
    def from_settings(
        cls,
        block: Mapping[str, Any],
        *,
        endpoints: EndpointLookup,
        tools: ToolSources,
        vocabulary: ControlTools,
    ) -> Self:
        """Validate the block, refusing unknown keys and names, so a bad block fails the core's build.

        It connects nothing; heavy imports, such as a harness's protocol packages, happen here.
        """

    def describe(self) -> dict:
        """Return what it serves, as plain JSON with no credentials."""

    async def check_model(self, model: str) -> None:
        """Raise ``ValueError`` if it cannot serve ``model``."""

    def unsupported(self, settings: Mapping[str, Any]) -> list[tuple[str, str]]:
        """Return the request settings it cannot honor, as ``(param, message)`` pairs."""

    def new(self) -> BackendAdapter:
        """Return the backend of a new IM session."""

    async def aclose(self) -> None:
        """Release what the source holds, such as a runtime it started."""


@runtime_checkable
class OneShotSource(Protocol):
    """A source that can serve the official stateless use: one request, its whole context as input."""

    def new_one_shot(self) -> BackendAdapter:
        """Return the backend of one stateless request: it needs no conductor, and no round that opened runs again."""
