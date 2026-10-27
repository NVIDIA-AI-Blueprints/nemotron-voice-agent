# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The port between the interaction manager (IM) and a backend adapter: types and the shared event rules.

The IM imports no other backend module. The IM owns admission, placement, the records, the conversation, steering's
timing, stops and every response the client reads; the adapter owns model rounds, providers, tools, conductors and
worker sessions. Only Responses shapes and the types below pass between them. There are three levels:
:class:`AdapterSource` (one per delegation core, chosen by ``backend.adapter``), :class:`BackendAdapter` (one per IM
session, with its :class:`Conductor` and ``max_running``) and :class:`BackendRun` (one per task).

**Works and tasks.** One request, or several joined requests, is one Work. A Work holds one or more tasks, and a task
is one worker: one :class:`BackendRun`, which reads and reports through its own :class:`TaskContext`. A conductor may
split a request into independent tasks that run in parallel (:class:`Start`), and :class:`Steer` and :class:`Cancel`
may name one task. Results are never merged or reworded: each task's answer is its own ``final_answer`` item in the
Work's response, and the response ends when the Work's last task ends. When one task waits for the app's outputs, the
response ends with that task's calls (GPT-Live requires it) while the other tasks run on; the IM holds their items
until the continuation opens the next response, so no adapter sees the difference. The backend LLM, directly
connected, always has exactly one task per Work.

**The event rules**, which every :meth:`BackendRun.events` keeps (the contract suite checks them):

* **E1 Rounds and attempts.** Each attempt opens with its own ``response.created`` and ends with exactly one terminal
  (:data:`TERMINALS`). ``response.completed`` ends the round; ``response.failed`` or ``response.incomplete`` ends only
  the attempt. Nothing of an attempt follows its terminal.
* **E2 Items.** Each item is announced by ``response.output_item.added`` and finished by ``response.output_item.done``.
  The IM reads items only from ``output_item.done``; deltas are optional.
* **E3 Phase.** Every message carries ``phase``: ``commentary`` is progress and never the answer, ``final_answer`` is
  the task's answer (:func:`message_phase`; a message with no phase counts as ``final_answer``).
* **E4 Calls last.** A round's calls come after its messages, and a round with calls carries no ``final_answer``.
* **E5 How the iteration ends.** After a round whose ``final_answer`` completed (the task answered); after a round
  with calls the app runs (the task waits: once the outputs are in, the IM calls :meth:`BackendRun.resume`, then
  :meth:`BackendRun.events` again); or by raising (the task failed). After :meth:`BackendRun.stop` it yields nothing
  more.
* **E6 Every step the IM must see.** Each call the backend runs is reported by :meth:`TaskContext.tool_started`
  before it runs and :meth:`TaskContext.tool_outcome` when it ends, every outcome with an output, and a ``call_id``
  never runs twice. Steering reaches the IM only through :meth:`TaskContext.settle`. Nothing that read or changed the
  world stays out of the IM's sight.
* **E7 Reading and retries.** An adapter that owns its model rounds reads :meth:`TaskContext.history` and
  :meth:`TaskContext.request` at the start of each model call, and takes boundary input at each safe point. An adapter
  whose runtime owns the rounds (a harness) sends ``history()`` in the worker's first prompt and steering through
  :meth:`TaskContext.take_boundary_input`. No attempt that dispatched a call or delivered an item is retried; an
  attempt that did neither may be retried (an empty round once, never in a one-shot session).
* **E8 Time.** A task ends or raises within the adapter's own limits (first event, silence, tool and turn timeouts),
  and :meth:`BackendRun.stop` returns within the adapter's stop limit. The IM's ``delegation.work_timeout_seconds`` is
  only the backstop, through its stop door, which waits at most ``delegation.stop_timeout_seconds`` for a stop.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from delegation_core.config import ModelEndpoint

# Who runs a call: ``app`` for a function the request declared (the round ends and the task waits), ``backend`` for any
# other, ``refused`` for one the IM will not run (after a stop, on a name clash, or while newer input waits to land, so
# the backend re-plans with that input first); a refused call gets an output that says why.
Route = Literal["app", "backend", "refused"]
# How a call the backend ran ended; ``unclear`` means it may have taken effect without reporting in time, and the IM
# answers it with ``texts.owed_output`` so the backend checks before it tries again.
Outcome = Literal["done", "failed", "unclear"]
# What became of steering the adapter took: the backend took it in, or it did not.
Delivery = Literal["applied", "not_applied"]
# What a stop could confirm: nothing of the task runs any more, or the adapter's stop limit passed first.
StopOutcome = Literal["stopped", "unconfirmed"]
# Where a Work or a task stands: open (queued, running, or waiting for the app's outputs), or ended.
OpenState = Literal["queued", "running", "waiting_client"]
EndedState = Literal["answered", "failed", "cancelled"]
WorkState = OpenState | EndedState
# What a message is: progress beside the work, or the task's answer.
Phase = Literal["commentary", "final_answer"]

# The terminal events of a response (and of a backend attempt).
TERMINALS: Final = frozenset({"response.completed", "response.failed", "response.incomplete"})


def calls_a_function(raw: dict) -> bool:
    """Whether a backend event belongs to a function call: an item of that type, or its arguments."""
    kind = raw.get("type", "")
    return raw.get("item", {}).get("type") == "function_call" or kind.startswith("response.function_call")


def function_call_item(raw: dict) -> dict | None:
    """The ``function_call`` item that a ``response.output_item.done`` event finishes, or ``None`` for any other."""
    item = raw.get("item")
    finished = raw.get("type") == "response.output_item.done" and isinstance(item, dict)
    return item if finished and item.get("type") == "function_call" else None


def message_phase(item: dict) -> Phase:
    """A message item's phase: ``commentary`` or ``final_answer``; a message with no phase counts as the answer."""
    return "commentary" if item.get("phase") == "commentary" else "final_answer"


# --------------------------------------------------------------------------------------------- tools
class ToolSource(Protocol):
    """Tools that run on the server side: a tool set the backend runs (``backend.tools``) or the IM's work tools.

    One instance serves one IM session, so per-session state (a cart) is never shared. A source class builds an
    instance from its YAML block with ``from_settings(block)``; the root makes a fresh set for each IM session
    (:data:`ToolSources`). It must be idempotent per ``call_id``: a repeated call adds nothing a second time.
    """

    label: str  # the server label of its calls' ``mcp_call`` items: "cafe", "hermes", "voiceclaw" ...

    def tools(self) -> list[dict]:
        """The function tools it offers, in the Responses shape (``{"type": "function", "name", ...}``)."""

    async def execute(self, call_id: str, name: str, arguments: str) -> str:
        """Run one call and return its output; an unknown name returns an error output instead of raising."""


# A fresh set of the session's server tool sources, made by the root for each IM session.
ToolSources = Callable[[], Sequence[ToolSource]]
# A model name to its endpoint, resolved by the root from the catalog; ``None`` gives the configured default.
EndpointLookup = Callable[[str | None], "ModelEndpoint"]


# --------------------------------------------------------------------------------------------- a task
@dataclass(frozen=True)
class Steering:
    """Input that joined a running task and waits for the adapter's safe point (one batch)."""

    id: str
    items: tuple[dict, ...]  # the joined requests' input, the records block and closing line included


@dataclass(frozen=True)
class BoundaryInput:
    """What a safe point takes: the app's outputs, already joined, and at most one waiting steering batch."""

    outputs: tuple[dict, ...]  # one function_call_output per call, in call order
    steering: Steering | None = None


class TaskContext(Protocol):
    """What the IM gives one task's run: what to read and where to report. The IM implements it, one per task."""

    work_id: str
    task_id: str

    def request(self) -> dict:
        """The settings of the task's next model call: model, composed instructions and the app's functions.

        They come from the request whose response the task is writing into. An explicit request (the Work's start, a
        continuation with the app's outputs, a typed run) brings its own; an automatic successor (a steering join)
        keeps the Work's. The IM composes ``instructions``: the request's or the default, then the voice note, then
        the result format's note. Every adapter puts it where its backend takes a system prompt: every round for an
        LLM, each worker session's system prompt for a harness.
        """

    def history(self) -> list[dict]:
        """This task's view of the conversation, as Responses input items within the IM's budget.

        The shared conversation, then the Work's input (every request it carries, each with the records block), the
        task's brief as ``texts.task_line`` when the conductor wrote one, and the task's own rounds. Steering that was
        never applied is not in it.
        """

    def position(self) -> int:
        """The position after the last item of this task's view, for an adapter that keeps its own memory."""

    def since(self, position: int) -> list[dict]:
        """The items of this task's view from ``position`` on, also those trimmed from ``history()`` since."""

    def tools(self) -> ToolSource:
        """The IM's work tools for this task: ``show`` when the result format offers it, ``cancel_work`` if listed."""

    def route(self, call: dict) -> Route:
        """Who runs this call (:data:`Route`); asked once per call, before it runs."""

    def tool_started(self, call: dict, source: str) -> None:
        """Report a call the backend runs as it starts; ``source`` is the label of the source that runs it."""

    def tool_outcome(self, call_id: str, output: str, outcome: Outcome) -> None:
        """A call the backend ran has ended; ``output`` is kept for every outcome but ``unclear``.

        An ``unclear`` call's output, in the history and on the wire, is ``texts.owed_output``; ``output`` says why.
        """

    def progress(self, text: str) -> None:
        """Activity that is not model text (a harness's plan or tool activity): a progress note, never history."""

    def take_boundary_input(self) -> BoundaryInput:
        """A safe point: the app's outputs join the conversation, and waiting steering is handed over.

        Safe to call at any point, any number of times: a second call joins nothing twice and returns only steering
        that arrived since.
        """

    def settle(self, steering: Steering, delivery: Delivery) -> None:
        """Report what became of a steering batch the adapter took, once per batch.

        ``applied``: the backend took it in; the IM joins it to the conversation and the newest request carries the
        Work. ``not_applied``: the IM keeps it for the next safe point, or for a follow-up Work once the task ends.
        """

    def trace(self, kind: str, **fields: Any) -> None:
        """Note a trace of this task for the logs; the IM adds the Work, the task and the carrier."""


class BackendRun(Protocol):
    """One task's run on a backend: one worker."""

    round: int  # the current round, from 1; 0 before the first

    def events(self) -> AsyncIterator[dict]:
        """The task's Responses events, attempt by attempt, by the event rules; called again after resume()."""

    def steer(self) -> None:
        """A wake-up: steering waits. The adapter takes it with ``take_boundary_input()`` at its next safe point.

        The LLM run redoes a round still streaming with nothing dispatched, once per Work; otherwise the steering
        joins the next round. A harness worker uses the strategy its settings name (native steer, next prompt, or
        cancel and re-prompt).
        """

    async def resume(self) -> None:
        """The app's outputs are in: take them with ``take_boundary_input()``, then ``events()`` goes on."""

    async def stop(self) -> StopOutcome:
        """Stop the task: return once nothing of it runs, or ``unconfirmed`` at the adapter's stop limit.

        It never cuts a running call that changes things; a stop is level-triggered and never cleared, and
        nothing is dispatched after it.
        """


# --------------------------------------------------------------------------------------------- the conductor
@dataclass(frozen=True)
class Request:
    """One delegation's request, as a conductor reads it: an arrival, or a request a Work carries."""

    delegation_id: str
    input: tuple[dict, ...]  # its input items, as the delegation sent them


@dataclass(frozen=True)
class TaskSummary:
    """One task of a Work, as a conductor and the records block read it (the work records)."""

    task_id: str
    state: WorkState  # a task can end while the other tasks of its Work run on
    brief: str | None  # the conductor's brief for it; None when it works on the requests as they came
    steps: tuple[str, ...]  # one line per call it made, with the call's outcome
    acted: bool  # a call was dispatched, so something may have changed
    progress: str | None  # its newest progress note
    seconds: float  # how long it has run (or ran)


@dataclass(frozen=True)
class WorkSummary:
    """One Work, open or recently ended, as a conductor and the records block read it."""

    work_id: str
    state: WorkState  # queued until a task starts; waiting_client while its response waits for the app; then ended
    carrier: str  # the delegation whose response carries the Work's output (or carried it last)
    requests: tuple[Request, ...]  # the requests it carries, oldest first
    tasks: tuple[TaskSummary, ...]
    seconds: float  # how long since it was started


@dataclass(frozen=True)
class ConductorInput:
    """What one ``decide()`` call reads."""

    arrivals: tuple[Request, ...]  # the requests no decision has placed yet, oldest first
    # The open Works, then the last ``delegation.records_max_works`` ended ones, so a status question about finished
    # work can be answered from the records.
    works: tuple[WorkSummary, ...]
    conversation: tuple[dict, ...]  # the conversation within the IM's budget, oldest first
    new_from: int = 0  # the index in ``conversation`` of the first item this conductor was not given before


class DecisionTarget(Protocol):
    """What applies a conductor's decisions: the IM session. Each method returns the outcome text the conductor reads.

    The IM re-checks every decision against the current state before it acts:

    * a Steer to an ended Work, or to an ended task, becomes a Start;
    * a Cancel of an ended Work or task does nothing;
    * a delegation placed twice in one ``decide()`` keeps its first placement (two Starts, a Start and a Steer, an
      Answer for a delegation already placed in a Work);
    * an unknown Work or task is refused.

    The outcome texts come from ``texts.outcomes``.
    """

    async def start(self, decision: Start) -> str:
        """A new Work for these requests, with its tasks; it runs when a task slot is free."""

    async def steer(self, decision: Steer) -> str:
        """These requests join an open Work (one task, or every open task); the IM decides when they land."""

    async def cancel(self, decision: Cancel) -> str:
        """Stop a Work, or one task of it, through the one stop door; returns once the stop is requested."""

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


# Every decision a conductor may place.
Decision = Start | Steer | Cancel | Answer
# How a conductor places a decision: the IM applies it and returns the outcome text.
Place = Callable[[Decision], Awaitable[str]]


class ControlTools(Protocol):
    """A model conductor's vocabulary: its instructions and its control tools, from the core's texts.

    ``delegation_core.backend.control.ControlVocabulary`` implements it from ``texts.conductor_prompt`` and
    ``texts.control_tools``.
    """

    prompt: str  # the conductor's instructions (texts.conductor_prompt); a rule conductor reads none

    def conductor_tools(self) -> list[dict]:
        """The four control tools, as Responses function tools."""

    def decision(self, name: str, arguments: str) -> Decision:
        """The decision one control call makes; ``ValueError`` for a malformed call, which is skipped and logged."""


class Conductor(Protocol):
    """Places the IM session's requests: start Works and their tasks, steer, cancel, or answer from the records.

    The IM calls :meth:`decide` after admission, never inside it, one call at a time per IM session.

    * An arrival that comes while a ``decide()`` runs: if that call has placed nothing yet, the IM cancels it and calls
      ``decide()`` again with all arrivals, at most once (as a run restarts before it acts); otherwise the arrival
      waits for the next ``decide()``, which sees the earlier requests as Works and tasks. With no call running, an
      arrival goes straight in.
    * A ``decide()`` cancelled before its first ``place()`` leaves the conductor as it was. A conductor whose runtime
      keeps a cancelled turn accounts for that itself.
    * ``place()`` may be called while an earlier ``place()`` of the same ``decide()`` is still running (a harness's
      control calls can come in parallel); the IM applies the decisions one at a time, in the order the calls reach
      it, and answers each with its outcome text. A ``place()`` after its ``decide()`` ended is refused.
    * A conductor never runs work itself and never words a result. A timeout returns having placed nothing, and the IM
      then places the arrivals by the steer rule (they join the newest open Work, or start one); so does a
      ``decide()`` that raises.
    * While an arrival waits for its decision, the IM's commit gate refuses every open Work's new calls, since the
      conductor has not yet said which Work the arrival concerns; calls already running are never cut.

    The backend LLM's conductor is a rule (``backend.llm.conductor``: ``steer`` or ``start_each``); the LLM agent's is
    one model call per ``decide()``; the harness's is its main session.
    """

    async def decide(self, given: ConductorInput, place: Place) -> None:
        """Place decisions through ``place``, each awaited for its outcome text, then return."""


# --------------------------------------------------------------------------------------------- an adapter
class BackendAdapter(Protocol):
    """One IM session's backend (never stacked): its conductor and the runs of its tasks."""

    max_running: int  # how many tasks may run at once in this IM session (the backend LLM: 1)

    def conductor(self) -> Conductor:
        """The session's conductor; the IM calls this once per IM session."""

    def start(self, task: TaskContext) -> BackendRun:
        """Run one task. It returns before any model call; the run starts with ``events()``."""

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
        """Validate the block, refusing unknown keys and names so a bad block fails the core's build.

        It connects nothing; heavy imports (a harness's protocol packages) happen here, not at module import.
        ``vocabulary`` carries a model conductor's instructions and control tools.
        """

    def describe(self) -> dict:
        """What it serves, as plain JSON for ``GET /v1/live/info`` (no credentials)."""

    async def check_model(self, model: str) -> None:
        """Raise ``ValueError`` if it cannot serve ``model``."""

    def unsupported(self, settings: Mapping[str, Any]) -> list[tuple[str, str]]:
        """The request settings it cannot honor, as ``(param, message)``: each one a ``responses_validation_failed``."""

    def new(self) -> BackendAdapter:
        """The backend of a new IM session."""

    async def aclose(self) -> None:
        """Release what the source holds (a runtime it started)."""


@runtime_checkable
class OneShotSource(Protocol):
    """A source that can serve the official stateless use: one request, its whole context as input, one attempt.

    ``backend.one_shot_adapter`` names the source that serves stateless requests; the core refuses a name whose
    source has no ``new_one_shot``. The adapter it returns needs no conductor, and its runs make one attempt per round.
    """

    def new_one_shot(self) -> BackendAdapter:
        """The backend of one stateless request's one-shot session."""
