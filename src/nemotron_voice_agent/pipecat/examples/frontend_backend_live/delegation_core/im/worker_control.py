# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An isolated, business-tool-free worker context for caller control of one existing logical task."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import (
    BackendRun,
    BoundaryInput,
    Delivery,
    Outcome,
    RefusalDetails,
    Route,
    Steering,
    ToolNotExecuted,
    ToolSource,
    message_phase,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.conversation import (
    TaskHistory,
    cut_output,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.input import scoped_items
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.tools import (
    CallerMessage,
    Invocation,
    WorkTools,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.items import message_text
from nemotron_voice_agent.pipecat.shared.ids import uid

if TYPE_CHECKING:
    from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.config import Texts
    from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.run import TaskRun


@dataclass(frozen=True)
class ControlCallbacks:
    """Core-owned control admission and input settlement; none may consume the business worker's owed output."""

    take_boundary: Callable[[], BoundaryInput]
    settle: Callable[[Steering, Delivery], None]
    refuse: Callable[[dict, str], None]
    can_dispatch: Callable[[], bool]
    pending_input_ready: Callable[[], bool]


@dataclass
class _ControlHistory:
    view: TaskHistory
    outputs: dict[str, dict] = field(default_factory=dict)
    routes: dict[str, Route] = field(default_factory=dict)
    calls: dict[str, dict] = field(default_factory=dict)
    started: set[str] = field(default_factory=set)
    observed: set[str] = field(default_factory=set)
    taken: dict[str, Steering] = field(default_factory=dict)
    deliveries: dict[str, Delivery] = field(default_factory=dict)
    progress: str | None = None


def _reference_item(call: dict, fact: Mapping) -> dict:
    return {
        "type": "message",
        "role": "developer",
        "id": f"{call['call_id']}.control-reference",
        "content": [
            {
                "type": "input_text",
                "text": json.dumps(
                    {"private_control_reference": {"original_call": call, "recorded_action": dict(fact)}}
                ),
            }
        ],
    }


def _matches_fact(call: dict, fact: Mapping, owners: frozenset[str]) -> bool:
    return (
        fact.get("call_id") == call.get("call_id")
        and fact.get("name") == call.get("name")
        and fact.get("arguments") == call.get("arguments")
        and fact.get("task_id") in owners
    )


def _project_history(
    items: Sequence[dict],
    facts: Mapping[str, Mapping],
    owners: frozenset[str],
    late_placeholders: Mapping[str, tuple[Any, Any]] | None = None,
) -> list[dict]:
    """Keep complete call/output pairs and describe genuine unmatched operations without inventing outputs."""
    calls: dict[str, tuple[int, dict]] = {}
    outputs: dict[str, int] = {}
    for index, item in enumerate(items):
        kind = item.get("type")
        if kind not in ("function_call", "function_call_output"):
            continue
        identity = item.get("call_id")
        if not isinstance(identity, str) or not identity:
            raise ValueError("Control history requires exact nonempty tool call identities")
        if kind == "function_call":
            if identity in calls:
                raise ValueError("Control history contains a duplicate tool call identity")
            calls[identity] = index, item
        else:
            if identity in outputs:
                raise ValueError("Control history contains duplicate tool outputs")
            outputs[identity] = index
    if any(identity not in calls or calls[identity][0] >= index for identity, index in outputs.items()):
        raise ValueError("Control history contains an orphaned or out-of-order tool output")
    projected = []
    for item in items:
        identity = item.get("call_id")
        if item.get("type") == "function_call_output" and late_placeholders and identity in late_placeholders:
            fact = facts.get(identity)
            if (
                item.get("output") in late_placeholders[identity]
                and fact is not None
                and _matches_fact(calls[identity][1], fact, owners)
            ):
                item = {**item, "output": fact["output"]}
        if item.get("type") != "function_call" or identity in outputs:
            projected.append(deepcopy(item))
            continue
        fact = facts.get(identity)
        if (
            fact is None
            or not _matches_fact(item, fact, owners)
            or fact.get("state") not in ("started", "done", "failed", "unclear", "not_run")
        ):
            raise ValueError("An unmatched control reference requires its exact recorded task/action facts")
        if fact["state"] in ("done", "failed"):
            if not isinstance(fact.get("output"), str) or fact.get("has_result") is False:
                raise ValueError("A completed referenced operation is missing its actual recorded result")
            projected.extend(
                [
                    deepcopy(item),
                    {"type": "function_call_output", "call_id": identity, "output": fact["output"]},
                ]
            )
            continue
        if fact.get("has_result") is True:
            raise ValueError("An unresolved referenced operation cannot claim a completed result")
        projected.append(_reference_item(item, fact))
    return projected


@dataclass(eq=False, frozen=True, init=False)
class WorkerControlContext:
    """Immutable Task/tool/callback binding with its own provider transcript and exact control-call outputs.

    The supervisor supplies original history before fresh backend_state(), and the actual action snapshot for any
    unmatched calls. Completed control-round items join through append(); own_items() returns the separate transcript
    for a later safe business-worker join. No method drains or rewrites the business provider's history.
    """

    _task: TaskRun
    _callbacks: ControlCallbacks
    _tools: ToolSource
    _names: frozenset[str]
    _request: dict
    _initial: tuple[dict, ...]
    _facts: dict[str, dict]
    _owners: frozenset[str]
    _state: _ControlHistory
    work_id: str
    task_id: str
    conversation_ref: str
    execution_ref: str
    context_budget: int
    business_access: Literal["full", "read_only", "core_only"]
    texts: Texts
    continues: str | None
    after_call: bool
    history_mode: Literal["shared", "focused"]

    def __init__(
        self,
        task: TaskRun,
        *,
        initial_history: Sequence[dict],
        call_facts: Mapping[str, Mapping],
        tools: ToolSource,
        callbacks: ControlCallbacks,
    ):
        """Snapshot the task's history and bind the exact Core tool capability for this invocation."""
        initial = tuple(deepcopy(list(initial_history)))
        facts = deepcopy({key: dict(value) for key, value in call_facts.items()})
        owners = frozenset(task.execution_owners)
        names = [tool.get("name") for tool in tools.tools()]
        if (
            any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
            or not set(names).issubset(task.work_tools.names)
        ):
            raise ValueError("A control invocation may expose only a unique selection of its task's Core tools")
        request = deepcopy(task.request())
        request.update(tools=[], tool_choice="auto")
        request["instructions"] = "\n\n".join(
            part for part in (request.get("instructions"), task.texts.control_worker_note) if part
        )
        view = TaskHistory(
            _project_history(initial, facts, owners),
            task.context_budget,
            task.session.settings.tool_output_chars,
            task.texts.cut_output,
        )
        values = {
            "_task": task,
            "_callbacks": callbacks,
            "_tools": tools,
            "_names": frozenset(names),
            "_request": request,
            "_initial": initial,
            "_facts": facts,
            "_owners": owners,
            "_state": _ControlHistory(view),
            "work_id": task.work_id,
            "task_id": task.task_id,
            "conversation_ref": task.conversation_ref,
            "execution_ref": uid("worker_control"),
            "context_budget": task.context_budget,
            "business_access": "core_only",
            "texts": task.texts,
            "continues": None,
            "after_call": False,
            "history_mode": task.history_mode,
        }
        for key, value in values.items():
            object.__setattr__(self, key, value)

    def request(self) -> dict:
        """Return the original model/settings with no application tools or forced business-tool choice."""
        return deepcopy(self._request)

    @property
    def original_context(self) -> TaskRun:
        """Return this immutable invocation's canonical owner through the backend port contract."""
        return self._task

    def tools(self) -> ToolSource:
        """The injected Core tools own their original records and exact control-source publication callbacks."""
        return self._tools

    def tool_sources(self, sources: Sequence[ToolSource]) -> tuple[ToolSource, ...]:
        """Control invocations never acquire any business source capability."""
        return ()

    def append(self, items: Sequence[dict]) -> None:
        """Join only a completed control round, leaving the original provider's history unchanged."""
        self._state.view.join(list(items))

    def _join_outputs(self) -> tuple[dict, ...]:
        calls = {item.get("call_id") for item in self._state.view.items if item.get("type") == "function_call"}
        answered = {
            item.get("call_id") for item in self._state.view.items if item.get("type") == "function_call_output"
        }
        identities = [identity for identity in self._state.outputs if identity in calls and identity not in answered]
        outputs = tuple(self._state.outputs.pop(identity) for identity in identities)
        if outputs:
            self._state.view.join(list(outputs))
        return deepcopy(outputs)

    def own_items(self) -> list[dict]:
        """Return control items and exact host execution facts for a later safe canonical join."""
        for identity, call in self._state.calls.items():
            if identity not in self._state.started and identity not in self._task.session.ledger.actions:
                self._refuse(call, "control_not_started")
        self._join_outputs()
        own = self._state.view.own()
        calls = {item.get("call_id") for item in own if item.get("type") == "function_call"}
        identities = tuple(self._state.calls)
        facts = self._current_facts(identities)
        projected = _project_history(own, facts, self._owners, self._late_placeholders(identities))
        # A native tool can execute while its containing model response fails. Its actual Core action remains
        # evidence; it must not become an orphaned function output or a fabricated completed model response.
        for identity, call in self._state.calls.items():
            if identity not in calls:
                fact = facts.get(identity)
                if fact is None:
                    raise ValueError("An unrepresented control call lacks its canonical execution facts")
                projected.append(_reference_item(call, fact))
        return projected

    def evidence(self) -> dict:
        """Retain exact private control evidence even if provider-history projection itself cannot complete."""
        return deepcopy(
            {
                "execution_ref": self.execution_ref,
                "deliveries": self._state.deliveries,
                "taken_input_ids": list(self._state.taken),
                "started_call_ids": sorted(self._state.started),
                "calls": list(self._state.calls.values()),
                "outputs": list(self._state.outputs.values()),
                "items": self._state.view.own(),
            }
        )

    def history(self, extra: Sequence[dict] = ()) -> list[dict]:
        """Return only this isolated context and fresh canonical state within the existing context budget."""
        view = self._history_view()
        return view.history((*self._backend_state(view, extra), *extra))

    def _history_view(self) -> TaskHistory:
        identities = (*self._facts, *self._state.calls)
        facts = self._current_facts(identities)
        late_placeholders = self._late_placeholders(identities)
        view = TaskHistory(
            _project_history(self._initial, facts, self._owners, late_placeholders),
            self.context_budget,
            self._state.view.output_chars,
            self._state.view.marker,
        )
        view.join(_project_history(self._state.view.own(), facts, self._owners, late_placeholders))
        return view

    def _late_placeholders(self, identities: Sequence[str]) -> dict[str, tuple[Any, Any]]:
        """Identify outputs made obsolete by an actual late result."""
        placeholders = {}
        view = self._state.view
        for identity in identities:
            action = self._task.session.ledger.actions.get(identity)
            if action is not None and action.late_outcome in ("done", "failed") and action.late_output is not None:
                item = {"type": "function_call_output", "output": action.output}
                rendered = cut_output(item, view.output_chars, view.marker) if view.output_chars is not None else item
                placeholders[identity] = action.output, rendered["output"]
        return placeholders

    def _current_facts(self, identities: Sequence[str]) -> dict[str, dict]:
        facts = deepcopy(self._facts)
        for identity in identities:
            action = self._task.session.ledger.actions.get(identity)
            if action is None:
                continue
            if action.task not in self._owners:
                raise ValueError("Control execution facts must belong to the exact logical task thread")
            facts[identity] = action.execution_fact(outcome_key="state")
            reported = self._state.outputs.get(identity)
            if reported is not None and reported.get("output") != facts[identity]["output"]:
                facts[identity]["host_reported_output"] = deepcopy(reported.get("output"))
        return facts

    def refresh_history(self, *, unconfirmed_calls: Sequence[str] = ()) -> None:
        """Refresh recorded operation facts locally; never call the business worker's refresh/join path."""
        known = set(unconfirmed_calls) & self._task.session.ledger.actions.keys()
        if any(identity not in self._state.calls for identity in known):
            raise ValueError("Unconfirmed control calls must belong to this isolated execution")
        self.history()

    def backend_state(self, extra: Sequence[dict] = ()) -> tuple[dict, ...]:
        """Use canonical questions/input receipts plus private execution facts for this isolated capability."""
        return self._backend_state(self._history_view(), extra)

    def _backend_state(self, view: TaskHistory, extra: Sequence[dict]) -> tuple[dict, ...]:
        represented = {item.get("call_id") for item in view.items if item.get("type") == "function_call_output"}
        facts = self._current_facts(tuple(self._state.calls))
        evidence = {
            "private_worker_control_state": {
                "task_id": self.task_id,
                "execution_ref": self.execution_ref,
                "business_tools_available": False,
                "progress": self._state.progress,
                "execution_facts": [facts[key] for key in self._state.calls if key not in represented and key in facts],
            }
        }
        control = {
            "type": "message",
            "role": "developer",
            "id": f"{self.execution_ref}.state",
            "content": [{"type": "input_text", "text": json.dumps(evidence)}],
        }
        budget = view.extra_budget((*extra, control))
        return (*self._task.backend_state_for_budget(budget), control)

    def can_dispatch(self) -> bool:
        """Only the canonical supervisor can grant this still-attached control invocation permission."""
        return self._callbacks.can_dispatch()

    def _capture_output(self, identity: str) -> None:
        output = self._task.outputs.pop(identity, None)
        if output is not None:
            self._state.outputs[identity] = output

    def _refuse(self, call: dict, reason: str) -> Route:
        identity = call.get("call_id", "")
        self._callbacks.refuse(call, reason)
        self._capture_output(identity)
        self._state.routes[identity] = "refused"
        return "refused"

    def route(self, call: dict) -> Route:
        """Refuse every non-Core name before the original task can grant application/backend tool admission."""
        identity = call.get("call_id", "")
        if identity in self._state.observed:
            raise ValueError("An external observation cannot be reused as an executable control call")
        if not isinstance(identity, str) or not identity:
            raise ValueError("A control call requires its own exact nonempty operation identity")
        if identity in self._state.routes:
            prior = self._state.calls[identity]
            if (prior.get("name"), prior.get("arguments")) != (call.get("name"), call.get("arguments")):
                raise ValueError("A control operation identity cannot be reused for different arguments")
            return self._state.routes[identity]
        if identity in self._task.routes or identity in self._task.session.ledger.actions:
            raise ValueError("A control call cannot reuse another execution's canonical operation identity")
        self._state.calls[identity] = deepcopy(call)
        if call.get("name") not in self._names:
            return self._refuse(call, "control_only")
        if not self.can_dispatch():
            return self._refuse(call, "control_unavailable")
        route = self._task.route(call)
        if route == "app":
            return self._refuse(call, "control_only")
        self._state.routes[identity] = route
        self._capture_output(identity)
        return route

    def dispatch(self, call: dict, *, suspended: bool = False) -> Route:
        """Recheck the control capability and canonical stop/input gate at actual execution."""
        if self.route(call) == "refused":
            return "refused"
        if not self.can_dispatch():
            return self._refuse(call, "control_unavailable")
        route = self._task.dispatch(call, suspended=suspended)
        if route == "app":
            return self._refuse(call, "control_only")
        self._state.routes[call.get("call_id", "")] = route
        self._capture_output(call.get("call_id", ""))
        return route

    def placement_pending(self, call: dict) -> bool:
        """Only an eligible Core call can wait for shared placement; business names never enter that wait."""
        return bool(call.get("name") in self._names and self.can_dispatch() and self._task.placement_pending(call))

    async def wait_for_placement(self) -> None:
        """Wait for the underlying task's placement decision."""
        await self._task.wait_for_placement()

    def refusal(self, call_id: str) -> str:
        """Return the caller-safe refusal for an exact tool call."""
        return self._task.refusal(call_id)

    def refusal_details(self, call_id: str) -> RefusalDetails:
        """Return structured evidence for an exact refused call."""
        return self._task.refusal_details(call_id)

    def pending_input_ready(self) -> bool:
        """Indicate whether this active control invocation has ready input."""
        return self.can_dispatch() and self._callbacks.pending_input_ready()

    def tool_started(self, call: dict, source: str) -> None:
        """Record the start of a call after the control admission check."""
        if self.dispatch(call) != "refused":
            self._task.tool_started(call, source)
            self._state.started.add(call["call_id"])
            self._capture_output(call.get("call_id", ""))

    def tool_observed(self, call: dict, source: str) -> None:
        """Preserve a native adapter's actual observation without granting an execution capability."""
        identity = call.get("call_id")
        if not isinstance(identity, str) or not identity:
            raise ValueError("A tool observation requires its exact native/Core correlation")
        previous = self._state.calls.get(identity)
        if identity not in self._state.observed and (
            previous is not None or identity in self._task.routes or identity in self._task.session.ledger.actions
        ):
            raise ValueError("An external observation cannot replace another execution's operation")
        if previous is not None and previous.get("name") != call.get("name"):
            raise ValueError("An observed operation cannot change its tool identity")
        self._task.tool_observed(call, source)
        self._state.observed.add(identity)
        self._state.started.add(identity)
        self._state.calls[identity] = deepcopy(call)
        self._capture_output(identity)

    def tool_outcome(self, call_id: str, output: str, outcome: Outcome) -> None:
        """Record an outcome and retain this invocation's result."""
        if call_id not in self._state.calls:
            raise ValueError("A control outcome must identify this isolated execution's own call")
        self._task.tool_outcome(call_id, output, outcome)
        self._capture_output(call_id)

    def progress(self, text: str) -> None:
        """Retain native progress within the control invocation."""
        if text.strip():
            self._state.progress = text

    def take_boundary_input(self) -> BoundaryInput:
        """Take control input and join results owned by this invocation."""
        boundary = self._callbacks.take_boundary() if self.can_dispatch() else BoundaryInput(())
        if any(item.get("call_id") not in self._state.calls for item in boundary.outputs):
            raise ValueError("A control boundary cannot consume the business worker's tool output")
        if boundary.outputs:
            self._state.view.join(list(boundary.outputs))
        local_outputs = self._join_outputs()
        steering = deepcopy(boundary.steering)
        if steering is not None:
            if steering.id in self._state.taken:
                raise ValueError("A control input batch was taken twice without its original settlement")
            self._state.taken[steering.id] = steering
        return BoundaryInput((*deepcopy(boundary.outputs), *local_outputs), steering)

    def settle(self, steering: Steering, delivery: Delivery) -> None:
        """Settle the exact input batch and retain applied items in local history."""
        selected = self._state.taken.get(steering.id)
        if selected is None:
            return
        if selected != steering:
            raise ValueError("A control delivery receipt must identify its exact taken input")
        self._callbacks.settle(steering, delivery)
        self._state.taken.pop(steering.id)
        self._state.deliveries[steering.id] = delivery
        if delivery == "applied":
            self._state.view.join(list(steering.items))

    def trace(self, kind: str, **fields: Any) -> None:
        """Record an observation with control execution provenance."""
        self._task.trace(kind, **{**fields, "execution_ref": self.execution_ref, "control_only": True})


class WorkerControl:
    """Run an isolated control foreground without taking ownership of the business invocation."""

    def __init__(self, task: TaskRun):
        """Associate the controller with its business task."""
        self.task = task
        self.run: BackendRun | None = None
        self.items: list[dict] = []
        self.blocked: set[str] = set()
        self.unconfirmed = False

    def ready(self) -> bool:
        """Check whether queued control input can acquire an inference slot."""
        return (
            self.run is None
            and not self.unconfirmed
            and self.task not in self.task.session.running
            and self.has_input()
        )

    def has_input(self) -> bool:
        """Find unhandled input for the isolated interpreter used by the capacity-one steer rule."""
        task = self.task
        return (
            task.session.backend.max_running == 1
            and task.session.caller_attached
            and callable(getattr(task.session.backend, "start_control", None))
            and any(waiting.batch.control and waiting.record.id not in self.blocked for waiting in task.steering)
        )

    def _take_boundary(self) -> BoundaryInput:
        task = self.task
        if not task.steering or task.lifecycle.stop_reason is not None:
            return BoundaryInput(())
        return BoundaryInput((), task.take_steering())

    def _remember_uncertain(self, evidence: dict, **outcome: bool) -> None:
        task = self.task
        details = {key: evidence[key] for key in ("items", "calls", "outputs")}
        receipt = {key: value for key, value in evidence.items() if key not in details}
        task.remember_uncertain_input(
            {"control_evidence": receipt, **outcome}, {"execution_ref": evidence["execution_ref"], **details}
        )

    async def process_input(self) -> None:
        """Run this same worker with only Core controls while its original operation remains owned."""
        task = self.task
        if self.run is not None or self.unconfirmed or not self.has_input():
            return
        queued = tuple(task.steering)
        sources = dict.fromkeys(source for row in queued if row.batch.control for source in row.batch.delegations)
        frontier = dict.fromkeys(waiting.record.id for waiting in task.unapplied())
        call_generation = task.session.call_generation
        publication_source = next(reversed(sources), None)
        active = True
        publications: set[str] = set()
        published_sources: set[str] = set()
        names = ("cancel_work", "set_work_state", "caller_message", "resolve_questions")

        def can_dispatch() -> bool:
            return (
                active and task.permits_output(allow_suspended=True) and task.session.call_generation == call_generation
            )

        def publish(invocation: Invocation) -> str:
            if not can_dispatch():
                raise ToolNotExecuted("caller_message: control invocation is no longer current")
            publications.add(invocation.call_id)
            result = task.presentation.publish(
                CallerMessage.model_validate(invocation.arguments),
                invocation.call_id,
                source_id=publication_source,
            )
            if publication_source is not None and invocation.call_id in task.presentation.written_calls:
                published_sources.add(publication_source)
            return result

        def settle(steering: Steering, delivery: Delivery) -> None:
            nonlocal publication_source
            waiting = task.taken.get(steering.id)
            selected = tuple(waiting.batch.delegations) if waiting is not None and waiting.batch.control else ()
            task.settle_control(steering, delivery)
            if waiting is not None:
                sources.update(dict.fromkeys(selected))
                if delivery == "not_applied":
                    frontier[steering.id] = None
                else:
                    frontier.pop(steering.id, None)
            if delivery == "applied" and selected:
                publication_source = selected[-1]

        def refuse(call: dict, reason: str) -> None:
            text = task.texts.not_run_output if reason == "control_not_started" else task.texts.unlisted_output
            task.remember_refusal(call, reason, text)

        tools = WorkTools(
            names,
            task.texts,
            task.lifecycle.cancel_own,
            publish,
            task.continue_after_call,
            task.session.settings.display_label,
            resolve=task.presentation.resolve_questions,
            set_state=task.lifecycle.set_work_state,
        )
        facts = {}
        for action in task.session.ledger.actions.values():
            if action.task not in task.execution_owners:
                continue
            facts[action.id] = action.execution_fact(outcome_key="state")
        context = WorkerControlContext(
            task,
            initial_history=[
                *(
                    task.view.history()
                    if task.view is not None
                    else [
                        *deepcopy(task.session.retained_thread(task.continues) or []),
                        *deepcopy(scoped_items(task.initial_requested, task.initial_scopes)),
                        *([task.assignment_message()] if task.brief and not task.initial_scopes else []),
                        *deepcopy(scoped_items(task.pending, task.pending_scopes)),
                    ]
                ),
                *deepcopy(self.items),
            ],
            call_facts=facts,
            tools=tools,
            callbacks=ControlCallbacks(
                self._take_boundary,
                settle,
                refuse,
                can_dispatch,
                self.has_input,
            ),
        )
        items: list[dict] = []
        owned_run: BackendRun | None = None
        grant: asyncio.Future | None = None
        owns_permit = False
        events = None
        failure: BaseException | None = None
        try:
            # Pause can complete an unscheduled grant to wake the business driver. That is not evidence
            # that this isolated invocation received an inference permit; inspect the actual scheduler.
            while task.lifecycle.stop_reason is None:
                grant = task.session.request_slot(task)
                await grant
                if task.grant is not grant:
                    raise RuntimeError("The control inference permit belongs to another admission owner")
                if task in task.session.running:
                    owns_permit = True
                    break
            if not can_dispatch():
                return
            self.unconfirmed = True
            owned_run = self.run = task.session.backend.start_control(context, parent=task.run)
            if task.run is None:
                task.run = owned_run.parent
            events = owned_run.events()
            async for raw in events:
                kind = raw.get("type")
                if kind == "response.created":
                    items = []
                elif kind == "response.output_item.done" and isinstance(raw.get("item"), dict):
                    items.append(raw["item"])
                elif kind == "response.completed":
                    context.append(items)
                    task.trace_round_usage(raw, context=context)
                    finals = [
                        item
                        for item in items
                        if item.get("type") == "message" and message_phase(item) == "final_answer"
                    ]
                    calls = [item for item in items if item.get("type") == "function_call"]
                    if finals and not calls and not publications and can_dispatch():
                        publish(
                            Invocation(
                                uid("control_reply"),
                                "caller_message",
                                {
                                    "kind": "status",
                                    "message": message_text(finals[-1]),
                                },
                            )
                        )
                    items = []
                elif kind in ("response.failed", "response.incomplete"):
                    task.trace_round_usage(raw, context=context)
                    items = []
        except asyncio.CancelledError as exc:
            failure = exc
            raise
        except Exception as exc:
            failure = exc
            self.blocked.update(row.record.id for row in queued if row in task.steering)
            task.trace("backend.control_unavailable", error_type=type(exc).__name__, detail=str(exc)[:300])
        finally:
            active = False
            if grant is not None and task.grant is grant:
                # Cancellation can arrive after the scheduler grants but before the waiter resumes. Only
                # this exact grant may remove the queue entry or release that already-awarded permit.
                owns_permit = owns_permit or task in task.session.running
                task.session.unschedule(task)
                if not grant.done():
                    grant.cancel()

            async def finish() -> None:
                confirmed = owned_run is None
                try:
                    if owned_run is not None:
                        async with asyncio.timeout(task.session.settings.stop_timeout_seconds):
                            confirmed = await owned_run.stop() == "stopped"
                            close = getattr(events, "aclose", None)
                            if close is not None:
                                await close()
                except Exception as exc:
                    confirmed = False
                    task.trace("backend.control_stop_unconfirmed", error_type=type(exc).__name__)
                evidence = context.evidence()
                if failure is not None or not confirmed:
                    self._remember_uncertain(evidence, stop_confirmed=confirmed)
                if confirmed:
                    self._retain_history(context)
                self.unconfirmed = not confirmed
                if confirmed and self.run is owned_run:
                    self.run = None
                elif not confirmed and owned_run is not None:
                    self._observe_quiescence(owned_run, context)
                if (failure is not None or not confirmed) and task.lifecycle.stop_reason is None:
                    uncertain = bool(
                        evidence["started_call_ids"]
                        or any(value in ("applied", "unknown") for value in evidence["deliveries"].values())
                        or not confirmed
                    )
                    line = task.texts.control_uncertain_line if uncertain else task.texts.control_unavailable_line
                    task.work.control_diagnostic(
                        task,
                        tuple(source for source in sources if source not in published_sources),
                        line,
                        frontier=frozenset(frontier),
                        call_generation=call_generation,
                    )
                if owns_permit and task.grant is grant:
                    task.session.release(task)
                task.wake.set()

            cleanup = task.session.spawn(finish())
            await asyncio.shield(cleanup)

    def _retain_history(self, context: WorkerControlContext) -> None:
        """Join one settled control transcript, including input whose real dispatch receipt arrived late."""
        task = self.task
        try:
            self.items.extend(deepcopy(context.own_items()))
            if task.ended:
                task.session.task_ended(task)
        except Exception as exc:
            self._remember_uncertain(context.evidence(), history_unavailable=True)
            task.trace("backend.control_history_unavailable", error_type=type(exc).__name__)

    def _observe_quiescence(self, run: BackendRun, context: WorkerControlContext) -> None:
        """Retain an uncertain control foreground until its adapter provides positive settlement proof."""
        task = self.task

        def settled() -> None:
            if self.run is run:
                self._retain_history(context)
                self.run = None
                self.unconfirmed = False
                task.trace("backend.control_later_quiescent")
                task.session.release(task)
                task.wake.set()

        task.lifecycle.observe_quiescence(run.on_quiescent(settled))
