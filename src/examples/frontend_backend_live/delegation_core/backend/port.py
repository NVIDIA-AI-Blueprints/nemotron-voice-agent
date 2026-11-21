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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, runtime_checkable

from examples.shared.delegation_contract import is_progress

if TYPE_CHECKING:
    from examples.frontend_backend_live.delegation_core.config import Texts
    from examples.shared.delegation_contract import ModelEndpoint

# Who runs a call: the app (a function the request declared), the backend, or none (refused: its output says why).
Route = Literal["app", "backend", "refused"]
# ``unclear``: the call may have taken effect, so the IM answers it with texts.owed_output and the backend checks first.
Outcome = Literal["done", "failed", "unclear"]
Delivery = Literal["applied", "not_applied", "unknown"]
StopOutcome = Literal["stopped", "unconfirmed"]
OpenState = Literal["queued", "running", "waiting_client", "pausing", "paused"]
EndedState = Literal["answered", "failed", "cancelled"]
WorkState = OpenState | EndedState
Phase = Literal["commentary", "final_answer"]


class DecisionRejected(ValueError):
    """An invalid or incomplete decision attempt that failed before Core admitted an action."""


class PlacementExpired(str):
    """A placement callback has lost its authority without admitting this decision."""


class ToolNotExecuted(ValueError):
    """A trusted tool owner rejects an operation before its effectful callback starts."""


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
        """Run one call and return its output; a known pre-effect refusal can raise ``ToolNotExecuted``.

        A repeated ``call_id`` adds nothing a second time.
        Other exceptions after dispatch preserve uncertainty about effects already started.
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
class RefusalDetails:
    """One refused call's immutable gate snapshot; it is not permission or an input-delivery receipt."""

    work_id: str
    task_id: str
    call_id: str
    generation: int
    reason: str
    input_scope: Literal["owned", "global", "none"]
    caller_attached: bool

    @property
    def owned_input_pending(self) -> bool:
        """Whether this refusal saw actual owned input, rather than global placement or another restriction."""
        return self.reason == "not_run" and self.input_scope == "owned"


@dataclass(frozen=True)
class BoundaryInput:
    """What a safe point takes: the app's outputs, joined and in call order, and at most one steering batch."""

    outputs: tuple[dict, ...]
    steering: Steering | None = None


class TaskContext(Protocol):
    """What the IM gives one task's run: what to read and where to report. The IM implements it, one per task."""

    work_id: str
    task_id: str
    # Opaque correlation and native-owner namespace; never authority to attach a kept conversation.
    conversation_ref: str
    # Private execution ownership can differ for an isolated control invocation of the same logical task.
    execution_ref: str
    context_budget: int
    business_access: Literal["full", "read_only", "core_only"]
    texts: Texts
    continues: str | None
    after_call: bool
    history_mode: Literal["shared", "focused"]

    def request(self) -> dict:
        """Return the settings of the task's next model call: model, composed instructions and the app's functions.

        Every adapter puts ``instructions`` where its backend takes a system prompt: every round for an LLM, each
        worker session's system prompt for a harness.
        """

    def history(self, extra: Sequence[dict] = ()) -> list[dict]:
        """Return this task's view of the conversation, as Responses input items within the IM's budget.

        A focused task receives its exact thread, assigned sources and current responsibility. A shared task also
        receives the bounded conversation. Items retain actual message and call identities across views; private host
        bookkeeping describes current ownership and applied input independently of those conversation items.
        Required extra items count against the same budget without joining history or becoming applied input.
        """

    def refresh_history(self, *, unconfirmed_calls: Sequence[str] = ()) -> None:
        """Reconcile recorded ancestry and actual late tool results before a delayed native prompt.

        Preserve the task's own items, history mode, starting settings, and applied-input receipts. Publish exact
        thread execution facts through backend_state(), including calls absent from a failed model response and
        owned unconfirmed_calls whose results might not have reached the native model. This does not admit newer
        conversation input or grant permission; adapters still check the dispatch gate. Native display IDs without
        a Core execution receipt add no fact; known calls belonging to another thread are rejected.
        """

    def backend_state(self, extra: Sequence[dict] = ()) -> tuple[dict, ...]:
        """Return private host state with budget reserved for the same extra items as :meth:`history`."""

    def can_dispatch(self) -> bool:
        """Whether the task's current lifecycle permits a native prompt handoff."""

    def tools(self) -> ToolSource:
        """Return this task's display, cancellation, caller-message and question-resolution tools."""

    def tool_sources(self, sources: Sequence[ToolSource]) -> tuple[ToolSource, ...]:
        """Restrict trusted source capabilities for this execution before offering or running tools."""

    def route(self, call: dict) -> Route:
        """Return who runs ``call``; the IM decides once per call, before it runs."""

    def dispatch(self, call: dict) -> Route:
        """Recheck actual execution permission while retaining any recorded refusal."""

    def placement_pending(self, call: dict) -> bool:
        """Whether a tool call awaits frontend routing or global placement.

        Owned input and higher-priority refusals do not wait here. Recheck after
        ``wait_for_placement()`` and before routing or dispatching an unpublished call.
        """

    async def wait_for_placement(self) -> None:
        """Wait for the existing placement barrier without requiring attachment or consuming owned input.

        This grants no permission: the adapter must recheck the preflight and then ask ``route()``. Its existing
        stop and retirement lifecycle still owns any pending wait.
        """

    def refusal(self, call_id: str) -> str:
        """Return the output the IM gave a call it refused, which says why."""

    def refusal_details(self, call_id: str) -> RefusalDetails:
        """Return that refused call's original typed gate snapshot, without rechecking or rerouting it."""

    def pending_input_ready(self) -> bool:
        """Whether owned input still waits on a live, nonstopping task with current or eligible after-call access."""

    def tool_started(self, call: dict, source: str) -> None:
        """Report a call the backend runs as it starts; ``source`` is the label of the source that runs it.

        A call that started with no ``route()`` asked, such as an agent's own tool, is the backend's.
        """

    def tool_observed(self, call: dict, source: str) -> None:
        """Record an adapter-observed tool outside Core dispatch; this reports activity and grants no permission."""

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
        IM keeps it for the next safe point, or for a follow-up Work once the task ends. ``unknown`` records an
        uncertain handoff without automatic reapplication or an applied-input receipt.
        """

    def trace(self, kind: str, **fields: Any) -> None:
        """Note a trace of this task for the logs; the IM adds the Work, the task and the carrier."""


class ControlContext(TaskContext, Protocol):
    """An invocation-bound Core-only view and its canonical business context, without native implementation data."""

    original_context: TaskContext


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
        """Continue after real application outputs or a confirmed suspension; then ``events()`` goes on."""

    async def suspend(self) -> StopOutcome:
        """Quiesce this invocation without ending its logical task or inventing application results.

        Return ``stopped`` only after its model/tool activity is quiescent. Retain context and genuine pending
        application ownership for resumption; ``unconfirmed`` keeps the Core's business-effect fence closed.
        """

    async def stop(self) -> StopOutcome:
        """Stop the task: return once nothing of it runs, or ``unconfirmed`` at the adapter's stop limit.

        It never cuts a running call that changes things, and nothing is dispatched after it.
        """

    def on_quiescent(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Subscribe to positive late resource settlement; return an unsubscribe function.

        The callback does not revise a stop outcome or establish an external business result. It runs only after
        this invocation's reader and tracked physical activity have settled, without polling or another stop.
        """


class ControlRun(BackendRun, Protocol):
    """A control invocation that can retain or lazily establish its logical task's owning business run."""

    parent: BackendRun | None


@dataclass(frozen=True)
class Request:
    """One delegation's request, as a conductor reads it: an arrival, or a request a Work carries."""

    delegation_id: str
    input: tuple[dict, ...]
    # A placed steering request remains pending even after an adapter takes it, until it reports application.
    delivery: Literal["landed", "pending"] = "landed"
    caller_excerpt: str | None = None
    caller_lines: tuple[dict, ...] = ()


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
    question: str | None = None
    answer: str | None = None
    # A bounded ending of an ordinary answered task, bound to this exact task for continuation routing.
    answer_end: str | None = None
    questions: tuple[dict, ...] = ()
    stop: Literal["none", "stop_requested", "stopped", "unconfirmed"] = "none"
    answer_publication: Literal["none", "pending", "written", "superseded"] = "none"
    tool_results: tuple[dict, ...] = ()
    tool_results_omitted: int = 0
    continues: str | None = None
    successor_task_id: str | None = None
    successor_work_id: str | None = None
    steering: Literal["active", "continuable", "continued", "unavailable"] | None = None
    verifies: str | None = None
    stop_initial: Literal["none", "stop_requested", "stopped", "unconfirmed"] | None = None


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
    # Thread identity survives a zero-length presentation records window.
    last_ended_work: str | None = None
    clarifications: tuple[dict, ...] = ()
    attempt: int = 1
    # Private telemetry correlation, never part of the model's conversation or placement authority.
    conversation_ref: str | None = None
    last_ended_task: str | None = None
    control_actions: tuple[dict, ...] = ()
    # Proven pre-admission rejection from the previous bounded attempt; not a provider failure or new caller input.
    decision_rejection: str | None = None
    available_worker_slots: int | None = None

    def routing_targets(self) -> dict:
        """List the currently presented work that accepts a steer or can be continued."""
        steer = []
        continues = []
        for work in self.works:
            tasks = [task.task_id for task in work.tasks if task.steering in ("active", "continuable")]
            if tasks:
                steer.append({"work_id": work.work_id, "task_ids": tasks})
            continues.extend(task.task_id for task in work.tasks if task.steering == "continuable")
        return {"steer": steer, "continues": continues}


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

    async def pause(self, decision: Pause) -> str:
        """Fence selected tasks and retain them while their active invocation becomes quiescent."""

    async def resume_work(self, decision: Resume) -> str:
        """Authorize selected paused tasks to resume through their existing capacity and input boundaries."""

    async def caller_message(self, decision: CallerMessage) -> str:
        """Publish a routing question or factual status for its selected sources."""

    async def submit(self, decision: CompleteDecision) -> str:
        """Validate and apply a complete decision, returning its per-action receipts."""


@dataclass(frozen=True)
class CallerLine:
    """Select the whole stored caller line or an exact part of it."""

    source_message_id: str
    sequence: int
    excerpt: str | None = None


@dataclass(frozen=True)
class TaskSpec:
    """One model-placed task's brief and optional exact ended thread."""

    brief: str
    continues: str | None = None
    verifies: str | None = None
    evidence: tuple[str, ...] = ()
    caller_lines: tuple[CallerLine, ...] = ()


@dataclass(frozen=True)
class Start:
    """Start a new Work for these requests; the newest one carries the Work's response."""

    delegations: tuple[str, ...]
    # The conductor's brief for each task, which run in parallel; () is one task on the requests as they came.
    tasks: tuple[TaskSpec, ...] = ()
    clarifications: tuple[str, ...] = ()

    async def apply(self, target: DecisionTarget) -> str:
        """Apply this decision to ``target``."""
        return await target.start(self)


@dataclass(frozen=True)
class WorkTarget:
    """One exact Work and selected tasks; ``None`` selects the whole Work."""

    work_id: str
    task_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class TaskUpdate:
    """The conductor's scoped reading of an update for one exact task."""

    task_id: str
    text: str
    evidence: tuple[str, ...] = ()
    caller_lines: tuple[CallerLine, ...] = ()


@dataclass(frozen=True)
class Steer:
    """Send selected sources to the exact Work/task sets in one action."""

    delegations: tuple[str, ...]
    targets: tuple[WorkTarget, ...]
    clarifications: tuple[str, ...] = ()
    updates: tuple[TaskUpdate, ...] = ()

    async def apply(self, target: DecisionTarget) -> str:
        """Apply this decision to ``target``."""
        return await target.steer(self)


@dataclass(frozen=True)
class Cancel:
    """Stop the selected Work/task sets without reversing completed business effects."""

    delegations: tuple[str, ...]
    targets: tuple[WorkTarget, ...]
    clarifications: tuple[str, ...] = ()

    async def apply(self, target: DecisionTarget) -> str:
        """Apply this decision to ``target``."""
        return await target.cancel(self)


@dataclass(frozen=True)
class Pause:
    """Pause the selected logical tasks without cancelling their goals, questions, or recorded effects."""

    delegations: tuple[str, ...]
    targets: tuple[WorkTarget, ...]
    clarifications: tuple[str, ...] = ()

    async def apply(self, target: DecisionTarget) -> str:
        """Apply the Core-owned suspension request."""
        return await target.pause(self)


@dataclass(frozen=True)
class Resume:
    """Resume selected paused tasks with their retained parameters and actual pending input."""

    delegations: tuple[str, ...]
    targets: tuple[WorkTarget, ...]
    clarifications: tuple[str, ...] = ()

    async def apply(self, target: DecisionTarget) -> str:
        """Apply through normal capacity and dispatch admission."""
        return await target.resume_work(self)


@dataclass(frozen=True)
class CallerMessage:
    """Publish one routing question or factual status for selected source obligations."""

    delegations: tuple[str, ...]
    kind: Literal["question", "status"]
    message: str = ""
    clarifications: tuple[str, ...] = ()
    targets: tuple[WorkTarget, ...] = ()

    async def apply(self, target: DecisionTarget) -> str:
        """Publish through the owning Core session."""
        return await target.caller_message(self)


DecisionAction = Start | Steer | Cancel | Pause | Resume | CallerMessage


@dataclass(frozen=True)
class CompleteDecision:
    """All actions selected for one admitted placement request."""

    actions: tuple[DecisionAction, ...]
    model_authored: bool = field(default=False, compare=False, repr=False)

    async def apply(self, target: DecisionTarget) -> str:
        """Submit all obligations before the Core releases placement admission."""
        return await target.submit(self)


Decision = DecisionAction | CompleteDecision
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
        """Return the complete-decision function tool."""

    def decision(self, name: str, arguments: str) -> CompleteDecision:
        """Validate one complete control envelope before any action is applied."""


class Conductor(Protocol):
    """Selects placement actions while the Core owns their admission and execution."""

    async def decide(self, given: ConductorInput, place: Place) -> None:
        """Place decisions through ``place``, each awaited for its outcome text, then return.

        The IM calls it one at a time within a call; an arrival may cancel it once before it placed anything, and the
        conductor may let its agent finish that turn before it ends. A call's end cancels it without waiting, so a later
        call's ``decide()`` calls may start while it still runs. Concurrent ``place()`` calls apply in order, and one
        after it ended is refused. Core owns bounded recovery and truthful failure of unresolved obligations.
        The conductor never runs work itself.
        """


class BackendAdapter(Protocol):
    """One IM session's backend (never stacked): its conductor and the runs of its tasks."""

    max_running: int

    def conductor(self) -> Conductor:
        """Return the session's conductor; the IM calls this once per adapter."""

    def start(self, task: TaskContext) -> BackendRun:
        """Return one task's run, before any model call; the run starts with ``events()``."""

    def start_control(self, task: ControlContext, *, parent: BackendRun | None = None) -> ControlRun:
        """Return an isolated invocation of the same logical worker offering only Core controls.

        The Core supplies focused context, an immutable execution reference, and its inference permit. No business
        ToolSource or application function is offered; this neither resumes nor retires a pending business run.
        """

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
class BoundedRetentionSource(Protocol):
    """An adapter source that retains request evidence outside the Core history."""

    def set_context_budget(self, tokens: int) -> None:
        """Apply the configured budget before creating any sessions."""


@runtime_checkable
class OneShotSource(Protocol):
    """A source that can serve the official stateless use: one request, its whole context as input."""

    def new_one_shot(self) -> BackendAdapter:
        """Return the backend of one stateless request: it needs no conductor, and no round that opened runs again."""
