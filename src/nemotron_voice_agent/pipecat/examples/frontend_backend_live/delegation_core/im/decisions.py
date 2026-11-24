# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Admit complete routing decisions, then execute their independent Core-owned obligations."""

from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import (
    CallerMessage,
    Cancel,
    CompleteDecision,
    DecisionAction,
    DecisionRejected,
    Pause,
    Resume,
    Start,
    Steer,
    TaskSpec,
    WorkTarget,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.input import InputScope
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.stop import (
    BeforeStartFailure,
    ForNewerRequest,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.ledger.records import (
    ControlAction,
    Placement,
)
from nemotron_voice_agent.pipecat.shared.ids import approx_tokens, uid

if TYPE_CHECKING:
    from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.ownership import Edge
    from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.run import TaskRun, WorkRun
    from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.session import IMSession
    from nemotron_voice_agent.pipecat.examples.frontend_backend_live.work_provenance import Provenance


class WorkerCapacityUnavailable(DecisionRejected):
    """A worker allocation was refused before it could start."""

    outcome = "capacity_unavailable"

    def __init__(self, requested: int, available: int):
        """Report the slot counts at the time of refusal."""
        self.details = {"reason": self.outcome, "requested": requested, "available": available, "executed": False}
        super().__init__(json.dumps(self.details))


class RoutingTargetUnavailable(DecisionRejected):
    """The placement cannot be admitted against its selected targets."""

    outcome = "target_unavailable"


@dataclass(frozen=True)
class StagedInput:
    """The exact control and caller input proposed for existing tasks."""

    control: ControlAction
    action: Steer | Pause | Resume
    targets: list[tuple[WorkRun, list[TaskRun]]]
    extra: tuple[dict, ...]
    scopes: dict[str | int, InputScope]


@dataclass(frozen=True)
class _ValidatedDecision:
    decision: CompleteDecision
    sources: tuple[str, ...]
    pending_stops: dict[int, tuple[str, ...]]
    clarifications: tuple[tuple[dict, ...], ...]


@dataclass(frozen=True)
class _PlannedAction:
    action: DecisionAction
    control: ControlAction
    extra: tuple[dict, ...]
    scopes: dict[str | int, InputScope]
    refusal: str | None
    pending_stops: tuple[str, ...]


@dataclass(frozen=True)
class _DecisionPlan:
    identity: str
    sources: tuple[str, ...]
    actions: tuple[_PlannedAction, ...]
    prepared_inputs: dict[str, dict[str, Edge]]
    releasing: tuple[TaskRun, ...]
    status_sources: frozenset[str]


@dataclass
class _DecisionExecution:
    releasing: set[TaskRun]
    status_sources: frozenset[str]
    release_waits: list[asyncio.Task] = field(default_factory=list)
    inputs: list[StagedInput] = field(default_factory=list)
    unresolved_input_order: int = 0


class DecisionExecutor:
    """Admit and apply the conductor's source-bound actions."""

    def __init__(self, session: IMSession):
        """Bind this session's existing records, tools, and execution callbacks."""
        self.session = session
        self.submissions: dict[int, tuple[str, tuple[str, ...]]] = {}

    def _receipt(self, ids: tuple[str, ...]) -> str:
        controls = self.session.ledger.controls
        return json.dumps({"actions": [asdict(controls[key]) for key in ids]})

    def submit(self, decision: CompleteDecision, *, epoch: int, queued: tuple[str, ...]) -> str:
        """Validate, plan, then apply one decision with a replayable receipt for its invocation."""
        signature = json.dumps(asdict(decision), sort_keys=True)
        self.submissions = {key: value for key, value in self.submissions.items() if key == epoch}
        prior = self.submissions.get(epoch)
        if prior is not None:
            if prior[0] != signature:
                raise DecisionRejected("A placement invocation has already accepted a different complete decision")
            return self._receipt(prior[1])

        validated = self._validate(decision, queued, signature)
        plan = self._plan(validated)
        self._register(plan)
        ids = tuple(entry.control.id for entry in plan.actions)
        self.submissions[epoch] = signature, ids
        self._execute(plan)
        self.session.observe(
            "im.decision.accepted", decision_id=plan.identity, action_ids=list(ids), sources=list(plan.sources)
        )
        return self._receipt(ids)

    def _validate(self, decision: CompleteDecision, queued: tuple[str, ...], signature: str) -> _ValidatedDecision:
        """Check source coverage, action compatibility, and the envelope budget before planning effects."""
        session = self.session
        if not decision.actions:
            raise DecisionRejected("A complete decision must contain actions")
        selected_sources = {source for action in decision.actions for source in action.delegations}
        if not selected_sources or not selected_sources.issubset(queued):
            raise DecisionRejected("Every action must name only its bound pending caller sources")
        decision, pending_stops = self._resolve_all_stops(decision, queued)
        pending_targets = [target for targets in pending_stops.values() for target in targets]
        if len(set(pending_targets)) != len(pending_targets):
            raise DecisionRejected("A queued request can be stopped only once in a decision")
        if overlap := selected_sources.intersection(pending_targets):
            raise DecisionRejected(
                f"A stopped queued request cannot also be used as an action source: {', '.join(sorted(overlap))}. "
                "Use the caller's stop request as the source and leave the queued targets out of starts."
            )
        selected_sources.update(pending_targets)
        sources = tuple(source for source in queued if source in selected_sources)
        missing_sources = set(queued) - selected_sources
        if missing_sources:
            raise DecisionRejected(
                "A complete decision must cover every bound pending caller source; missing: "
                + ", ".join(sorted(missing_sources))
            )
        clarifications = []
        for action in decision.actions:
            if not action.delegations or len(set(action.delegations)) != len(action.delegations):
                raise DecisionRejected("Action sources must be a nonempty unique selection")
            if not isinstance(action, (Start, Steer, Cancel, Pause, Resume, CallerMessage)):
                raise DecisionRejected("Unsupported decision action")
            if isinstance(action, CallerMessage) and action.kind == "status":
                if decision.model_authored and not action.targets:
                    raise DecisionRejected("Operational status selects work targets for Core to report")
                if not session.ledger.works:
                    raise DecisionRejected("Operational status requires open or retained ended work records")
            extra = session.clarification_input(action.clarifications)
            if extra is None:
                raise DecisionRejected("Every action must bind only live, unique routing clarification identities")
            clarifications.append(tuple(extra))
            if isinstance(action, Start) and len(action.tasks) > 1:
                initial_sources = set(action.delegations).union(session.clarification_sources(action.clarifications))
                if len(initial_sources) > 1:
                    raise DecisionRejected(
                        "Use one start action per task when starting from several caller sources, including retained "
                        "clarification sources. Give each start only its applicable sources in this same decision."
                    )
        self._validate_targets(decision, pending_stops)
        if approx_tokens(signature) > session.settings.context_tokens:
            raise DecisionRejected("Complete decision exceeds the backend context budget")
        return _ValidatedDecision(decision, sources, pending_stops, tuple(clarifications))

    def _validate_targets(self, decision: CompleteDecision, pending_stops: dict[int, tuple[str, ...]]) -> None:
        """Reject conflicting controls and more than one successor for the same task."""
        session = self.session
        selected = {}
        continuations = set()
        for index, action in enumerate(decision.actions):
            if isinstance(action, Start):
                for spec in action.tasks:
                    if spec.continues is not None:
                        if spec.continues in continuations:
                            raise DecisionRejected("A thread can have only one successor in a decision")
                        continuations.add(spec.continues)
            if isinstance(action, (Steer, Cancel, Pause, Resume)):
                if not action.targets and index not in pending_stops:
                    raise DecisionRejected("A control action requires selected targets")
                action_targets = set()
                if len({target.work_id for target in action.targets}) != len(action.targets):
                    raise DecisionRejected("Select each Work once, listing its exact task IDs together")
                for target in action.targets:
                    work = session.ledger.works.get(target.work_id)
                    tasks = (
                        target.task_ids
                        if target.task_ids is not None
                        else tuple(t.id for t in work.tasks)
                        if work
                        else ("*",)
                    )
                    if not tasks or len(set(tasks)) != len(tasks):
                        raise DecisionRejected("A target requires a nonempty unique task selection")
                    for task_id in tasks:
                        key = (target.work_id, task_id)
                        prior_actions = selected.get(key, set())
                        lifecycle = {Cancel, Pause, Resume}
                        if (
                            key in action_targets
                            or type(action) in prior_actions
                            or (Cancel in prior_actions)
                            or (isinstance(action, Cancel) and prior_actions)
                            or (type(action) in lifecycle and prior_actions.intersection(lifecycle))
                        ):
                            raise DecisionRejected(
                                "A task permits one update and one pause or resume; a stop excludes other controls"
                            )
                        action_targets.add(key)
                if isinstance(action, Steer):
                    for _, task_id in action_targets:
                        task = session.ledger.tasks.get(task_id)
                        if task is not None and task.ended:
                            if task_id in continuations:
                                raise DecisionRejected("A thread can have only one successor in a decision")
                            continuations.add(task_id)
                for key in action_targets:
                    selected.setdefault(key, set()).add(type(action))
        starts = {
            spec.continues
            for action in decision.actions
            if isinstance(action, Start)
            for spec in action.tasks
            if spec.continues is not None
        }
        steered = {
            task_id
            for action in decision.actions
            if isinstance(action, Steer)
            for target in action.targets
            for task_id in (
                target.task_ids
                if target.task_ids is not None
                else tuple(task.id for task in session.ledger.works[target.work_id].tasks)
                if target.work_id in session.ledger.works
                else ()
            )
        }
        if starts.intersection(steered):
            raise DecisionRejected("A predecessor cannot be both continued and steered in one decision")
        if any(task_id in starts and Cancel in actions for (_, task_id), actions in selected.items()):
            raise DecisionRejected("A predecessor cannot be both continued and stopped in one decision")

    def _plan(self, validated: _ValidatedDecision) -> _DecisionPlan:
        """Budget all input before admitting work, allowing repair when every target is unavailable."""
        session = self.session
        decision, sources, pending_stops = validated.decision, validated.sources, validated.pending_stops
        input_scopes = self._plan_scopes(decision, validated.clarifications)
        continued = [self._continued_targets(action) for action in decision.actions]
        if all(continued):
            rejection = {
                "reason": "continued_targets",
                "executed": False,
                "sources": list(sources),
                "actions": [{"action_index": index, "targets": list(rows)} for index, rows in enumerate(continued)],
            }
            session.observe("im.decision.rejected", **rejection)
            raise RoutingTargetUnavailable(json.dumps(rejection))
        refusals = {
            index: error
            for index, action in enumerate(decision.actions)
            if (error := self._target_refusal(action)) is not None
        }
        if len(refusals) == len(decision.actions):
            rejection = {
                "reason": RoutingTargetUnavailable.outcome,
                "executed": False,
                "sources": list(sources),
                "actions": [{"action_index": index, "error": error} for index, error in refusals.items()],
            }
            session.observe("im.decision.rejected", **rejection)
            raise RoutingTargetUnavailable(json.dumps(rejection))
        releasing = {
            task
            for work in session.works.values()
            for task in work.tasks
            if not task.ended and task.stop_reason is not None
        }
        without_allocation = refusals.keys() | {index for index in pending_stops if not decision.actions[index].targets}
        eligible = [action for index, action in enumerate(decision.actions) if index not in without_allocation]
        if session.backend.max_running > 1 and all(isinstance(action, (Start, CallerMessage)) for action in eligible):
            requested = sum(len(action.tasks) or 1 for action in eligible if isinstance(action, Start))
            available = session.available_worker_slots()
            if requested > available and (not releasing or requested > session.backend.max_running):
                error = WorkerCapacityUnavailable(requested, available)
                session.observe("im.decision.rejected", **error.details, sources=list(sources))
                raise error
        identity = uid("decision")
        actions = []
        for index, (action, extra, scopes) in enumerate(
            zip(decision.actions, validated.clarifications, input_scopes, strict=True)
        ):
            action = replace(action, delegations=tuple(source for source in sources if source in action.delegations))
            control = ControlAction(f"{identity}.{index + 1}", type(action).__name__, action.delegations)
            actions.append(
                _PlannedAction(action, control, extra, scopes, refusals.get(index), pending_stops.get(index, ()))
            )
        status_sources = frozenset(
            source
            for action in decision.actions
            if isinstance(action, CallerMessage) and action.kind == "status" and not action.targets
            for source in action.delegations
        )
        prepared_inputs = self._plan_inputs(actions, sources)
        return _DecisionPlan(identity, sources, tuple(actions), prepared_inputs, tuple(releasing), status_sources)

    def _plan_inputs(self, actions: list[_PlannedAction], sources: tuple[str, ...]) -> dict[str, dict[str, Edge]]:
        session = self.session
        staged = []
        allocations = []
        for entry in actions:
            action = entry.action
            if entry.refusal is not None:
                continue
            if isinstance(action, Start):
                allocations.append(action.delegations)
            if not isinstance(action, (Steer, Pause, Resume)):
                continue
            try:
                targets = (
                    self._targets(action, stopping=False)
                    if isinstance(action, Steer)
                    else self._lifecycle_targets(action)
                )
            except ValueError:
                continue
            if isinstance(action, Steer) and any(
                task is None or task.ended for _, _, _, tasks in targets for task in tasks
            ):
                allocations.append(action.delegations)
            live = [
                (work, [task for task in tasks if task is not None and not task.ended]) for _, _, work, tasks in targets
            ]
            live = [(work, tasks) for work, tasks in live if tasks]
            if live:
                extra = entry.extra
                if isinstance(action, (Pause, Resume)):
                    extra = (*extra, self._lifecycle_fact(entry.control.id, action))
                staged.append(StagedInput(entry.control, action, live, extra, entry.scopes))
        projected = session.ownership.project_allocations(allocations) if staged else {}
        for entry in staged:
            session.ownership.validate_budget(
                entry.action.delegations, entry.targets, extra=entry.extra, projected=projected
            )
        return self._prepare_inputs(staged, sources, projected)

    def _register(self, plan: _DecisionPlan) -> None:
        """Install every source obligation before taking inputs or starting worker effects."""
        session = self.session
        placements = Counter(source for entry in plan.actions for source in entry.action.delegations)
        for entry in plan.actions:
            control, action = entry.control, entry.action
            session.ledger.controls[control.id] = control
            composed = isinstance(action, (Cancel, Pause, Resume, CallerMessage))
            if isinstance(action, Start):
                composed = len(action.tasks) > 1
            elif isinstance(action, Steer):
                composed = len(action.targets) > 1 or any(
                    len(session.ledger.works[target.work_id].tasks) > 1
                    for target in action.targets
                    if target.work_id in session.ledger.works
                )
            for source in control.sources:
                owner = session.ownership.owner(source)
                owner.composed = owner.composed or composed or placements[source] > 1
                owner.controls.add(control.id)

    def _execute(self, plan: _DecisionPlan) -> None:
        """Fence refused input, release resources, apply worker input, then publish caller messages."""
        session = self.session
        accepted = []
        refused = []
        for entry in plan.actions:
            (refused if entry.refusal is not None else accepted).append(entry)
        execution = _DecisionExecution(set(plan.releasing), plan.status_sources)
        with session.applying_decision(plan.sources):
            execution.unresolved_input_order = self._fence_refusals(entry.action for entry in refused)
            session.take_decision_sources(plan.sources)
            session.observe("im.routing.applying", delegation_ids=plan.sources)
            for entry in refused:
                self._settle(
                    entry.control,
                    "refused",
                    outcome=RoutingTargetUnavailable.outcome,
                    executed=False,
                    error=entry.refusal,
                )
            execution.release_waits.extend(session.spawn(task.confirm_stop()) for task in plan.releasing)
            self._release_resources(accepted, execution)
            self._apply_worker_actions(accepted, execution)
            self._flush_inputs(execution.inputs, plan.sources, plan.prepared_inputs)
            for entry in accepted:
                if isinstance(action := entry.action, CallerMessage):
                    try:
                        self._message(entry, action)
                    except (ValueError, RuntimeError) as error:
                        self._settle(entry.control, "refused", error=str(error))

    def _release_resources(self, accepted: list[_PlannedAction], execution: _DecisionExecution) -> None:
        for entry in accepted:
            action = entry.action
            if not isinstance(action, (Cancel, Pause)):
                continue
            try:
                confirmations = (
                    self._execute_stop(entry, action)
                    if isinstance(action, Cancel)
                    else self._execute_pause(entry, action, execution)
                )
                if confirmations:
                    execution.release_waits.extend(confirmations)
                    execution.releasing.update(
                        task
                        for _, _, _, tasks in self._targets(action, stopping=True)
                        for task in tasks
                        if task is not None and not task.ended
                    )
            except Exception as error:
                self._settle(entry.control, "unknown", error_type=type(error).__name__)

    def _apply_worker_actions(self, accepted: list[_PlannedAction], execution: _DecisionExecution) -> None:
        """Stage updates before resume intent, independent of action order in the submitted envelope."""
        for action_type in (Steer, Resume, Start):
            for entry in accepted:
                action = entry.action
                if not isinstance(action, action_type):
                    continue
                try:
                    if isinstance(action, Start):
                        self._start(entry, action, execution)
                    elif isinstance(action, Steer):
                        self._steer(entry, action, execution)
                    elif isinstance(action, Resume):
                        self._execute_resume(entry, action, execution)
                except WorkerCapacityUnavailable as error:
                    self._settle(entry.control, "refused", outcome=error.outcome, **error.details)
                except (ValueError, RuntimeError) as error:
                    self._settle(entry.control, "refused", error=str(error))
                except Exception as error:
                    self._settle(entry.control, "unknown", error_type=type(error).__name__)

    def _plan_scopes(
        self, decision: CompleteDecision, clarifications: tuple[tuple[dict, ...], ...]
    ) -> list[dict[str | int, InputScope]]:
        """Bind normalized worker input before taking any source or issuing a lifecycle effect."""
        session, actions = self.session, decision.actions
        plans: list[dict[str | int, InputScope]] = [{} for _ in actions]
        model_parallel = decision.model_authored and session.backend.max_running > 1
        sources = [
            set(action.delegations).union(session.clarification_sources(action.clarifications)) for action in actions
        ]
        for index, action in enumerate(actions):
            if not isinstance(action, (Start, Steer, Pause, Resume)):
                continue
            if not model_parallel and not (
                isinstance(action, Start)
                and any(spec.caller_lines for spec in action.tasks)
                or isinstance(action, Steer)
                and action.updates
            ):
                continue
            shared = model_parallel and any(
                sources[index] & other for offset, other in enumerate(sources) if offset != index
            )
            raw = [
                *clarifications[index],
                *(item for source in action.delegations for item in session.request_items(source)),
            ]
            if isinstance(action, Start):
                for task_index, spec in enumerate(action.tasks):
                    if shared or (model_parallel and len(action.tasks) > 1) or spec.caller_lines:
                        plans[index][task_index] = session.bind_worker_input(
                            spec.brief,
                            raw,
                            spec.continues,
                            selections=spec.caller_lines,
                        )
                continue
            try:
                targets = tuple(task for _, ids, _ in self._selected_tasks(action) for task in ids)
            except ValueError:
                continue
            mixed = shared or (model_parallel and len(targets) > 1)
            if isinstance(action, Steer):
                if not mixed and not action.updates:
                    continue
                updates = defaultdict(list)
                for update in action.updates:
                    updates[update.task_id].append(update)
                if updates.keys() != set(targets):
                    raise DecisionRejected("Provide a scoped steer update for every exact selected task")
                for task in targets:
                    fragments = updates[task]
                    plans[index][task] = session.bind_worker_input(
                        "\n".join(update.text for update in fragments),
                        raw,
                        task,
                        require_lines=mixed,
                        selections=tuple(line for update in fragments for line in update.caller_lines),
                        include_bound_lines=decision.model_authored and not mixed,
                    )
            elif mixed:
                verb = "pause" if isinstance(action, Pause) else "resume"
                for task in targets:
                    plans[index][task] = session.bind_worker_input(
                        f"The caller asked to {verb} this task. Core already handles that lifecycle action.",
                        raw,
                        task,
                    )
        if model_parallel:
            assigned: list[InputScope] = []
            for action, plan in zip(actions, plans, strict=True):
                if not isinstance(action, Start):
                    continue
                for index, spec in enumerate(action.tasks):
                    scope = plan.get(index)
                    if spec.continues is not None or scope is None:
                        continue
                    if any(scope.conflicts(previous) for previous in assigned):
                        raise DecisionRejected(
                            "A whole caller line or source was assigned to a worker while another worker uses it. "
                            "Select each worker's excerpt and put shared facts in their briefs."
                        )
                    assigned.append(scope)
        return plans

    def _settle(self, control: ControlAction, state: str, **receipt) -> None:
        control.state, control.receipt = state, receipt
        session = self.session
        session.observe(
            "im.decision.action",
            action_id=control.id,
            action_kind=control.kind,
            sources=control.sources,
            state=state,
            receipt=receipt,
        )
        session.control_settled(control.sources)

    def delivered(self, task_id: str, sources: tuple[str, ...], delivery: str) -> None:
        """Reconcile every selected task before completing its accepted input action."""
        for control in list(self.session.ledger.controls.values()):
            if task_id not in control.delivery or not set(control.sources).intersection(sources):
                continue
            control.delivery[task_id] = delivery
            self._delivery_receipt(control)

    def _delivery_receipt(self, control: ControlAction) -> None:
        """Keep a failed continuation separate from the same action's live input deliveries."""
        values = set(control.delivery.values()) - {"continued"}
        outcome = (
            "pending"
            if "pending" in values
            else "partial"
            if "applied" in values and values.intersection({"unknown", "failed", "cancelled"})
            else "unknown"
            if "unknown" in values
            else "failed"
            if "failed" in values
            else "cancelled"
            if "cancelled" in values
            else "applied"
        )
        receipt = dict(control.receipt)
        if outcome != "pending":
            receipt["outcome"] = outcome
        self._settle(control, outcome, **receipt)

    def transfer(self, predecessor: str, successor: str, sources: tuple[str, ...]) -> None:
        """The same accepted update can follow one exact task successor without duplicate admission."""
        for control in self.session.ledger.controls.values():
            if control.delivery.get(predecessor) == "pending" and set(control.sources).intersection(sources):
                control.delivery[predecessor] = "continued"
                control.delivery[successor] = "pending"
                control.receipt.setdefault("input_continuations", []).append({"from": predecessor, "to": successor})

    @staticmethod
    def _continued_record(task) -> dict:
        return {
            "reason": "continued",
            "work_id": task.work,
            "task_id": task.id,
            "successor_work_id": task.successor_work_id,
            "successor_task_id": task.successor_task_id,
        }

    def _continued_targets(self, action) -> tuple[dict, ...]:
        """Identify only proven superseded atomic actions; unrelated failures retain runtime outcomes."""
        if isinstance(action, Start):
            tasks = [
                self.session.ledger.tasks.get(spec.continues) for spec in action.tasks if spec.continues is not None
            ]
        elif isinstance(action, Steer):
            try:
                tasks = [task for _, _, selected in self._selected_tasks(action) for task in selected]
            except ValueError:
                return ()
        else:
            return ()
        continued = []
        for task in tasks:
            if task is None:
                return ()
            state = self.session.task_steering_state(task)
            if state == "continued":
                continued.append(self._continued_record(task))
            elif state not in ("active", "continuable"):
                return ()
        return tuple(continued)

    def _continuation(self, task_id: str) -> None:
        task = self.session.ledger.tasks.get(task_id)
        if task is not None and self.session.task_steering_state(task) == "continued":
            raise ValueError(json.dumps(self._continued_record(task)))
        if task is None or self.session.task_steering_state(task) != "continuable":
            raise ValueError(f"Task {task_id} is not an eligible ended thread")

    def _target_refusal(self, action) -> str | None:
        """Check target eligibility without consuming input or changing task state."""
        try:
            if isinstance(action, Start):
                for spec in action.tasks:
                    if spec.continues is not None:
                        self._continuation(spec.continues)
                    if spec.verifies is not None and spec.verifies not in self.session.thread_history:
                        raise ValueError("Verification needs an exact retained task")
            elif isinstance(action, Steer):
                self._targets(action, stopping=False)
            elif isinstance(action, (Pause, Resume)):
                self._lifecycle_targets(action)
            elif isinstance(action, Cancel) or (
                isinstance(action, CallerMessage) and action.kind == "status" and action.targets
            ):
                self._selected_tasks(action)
        except ValueError as error:
            return str(error)
        return None

    def _fence_refusals(self, actions: Iterable[DecisionAction]) -> int:
        """Protect refused targets and return the global input order for unidentified targets."""
        session = self.session
        unresolved = set()
        for action in actions:
            if isinstance(action, CallerMessage):
                continue
            try:
                if isinstance(action, Start):
                    tasks = [
                        session.ledger.tasks[reference]
                        for spec in action.tasks
                        if (reference := spec.continues or spec.verifies) is not None
                    ]
                else:
                    tasks = [task for _, _, selected in self._selected_tasks(action) for task in selected]
            except (KeyError, ValueError):
                unresolved.update(action.delegations)
            else:
                session.fence_unresolved_inputs(action.delegations, tasks=tasks)
        return session.fence_unresolved_inputs(unresolved)

    def _start(self, entry: _PlannedAction, action: Start, execution: _DecisionExecution) -> None:
        control, scopes = entry.control, entry.scopes
        worker_scopes = tuple((scopes[index],) if index in scopes else () for index in range(len(action.tasks)))
        work = self._allocate_work(
            action.delegations, action.tasks, entry.extra, execution, worker_scopes=worker_scopes
        )
        self.session.consume_clarifications(action.clarifications, control)
        receipt = {"work_id": work.id, "task_ids": [task.task_id for task in work.tasks]}
        if work.start_barrier is None:
            self._settle(control, "applied", **receipt)
        else:
            self._settle(control, "pending", **receipt, outcome="waiting_for_resource_release")
            self.session.spawn(self._start_after_release(control, work, tuple(execution.release_waits)))

    def _allocate_work(
        self,
        sources: tuple[str, ...],
        specs: tuple[TaskSpec, ...],
        extra: tuple[dict, ...],
        execution: _DecisionExecution,
        *,
        worker_scopes: tuple[tuple[InputScope, ...], ...] = (),
    ) -> WorkRun:
        """Allocate an assignment, allowing a single-worker start to wait in the scheduler's FIFO."""
        session = self.session
        if session.closed or not session.caller_attached:
            raise ValueError("The caller attachment is no longer eligible")
        requested = len(specs) or 1
        available = session.available_worker_slots()
        needs_release = requested > available and not (requested == session.backend.max_running == 1)
        if needs_release:
            occupied = (
                set(session.running)
                | set(session.ready)
                | {
                    task
                    for work in session.works.values()
                    for task in work.tasks
                    if task.record.state == "queued" and task.stop_reason is None
                }
            )
            retained = len(occupied - execution.releasing)
            if not execution.release_waits or retained + requested > session.backend.max_running:
                raise WorkerCapacityUnavailable(requested, available)
        for spec in specs:
            if spec.continues is not None:
                self._continuation(spec.continues)
        barrier = asyncio.get_running_loop().create_future() if needs_release else None
        return session.start_work(
            list(sources),
            specs,
            extra=extra,
            start_barrier=barrier,
            worker_scopes=worker_scopes,
            unresolved_input_order=execution.unresolved_input_order,
        )

    def _allocation_fits(self, work: WorkRun) -> bool:
        """Use real leases and earlier queued reservations without making pending sibling starts wait on each other."""
        session = self.session
        own = set(work.tasks)
        occupied = set(session.running) | set(session.ready)
        for earlier in session.works.values():
            if earlier is work:
                break
            occupied.update(
                task
                for task in earlier.tasks
                if not task.ended and task.record.state == "queued" and task.stop_reason is None
            )
        required = sum(not task.ended and task.stop_reason is None for task in work.tasks)
        return len(occupied - own) + required <= session.backend.max_running

    async def _wait_allocation(self, work: WorkRun, waits: tuple[asyncio.Task, ...]) -> None:
        """Wake on scheduler changes or settlement of the bounded stop and pause attempts."""
        pending = set(waits)
        while True:
            if work.ended or not any(not task.ended and task.stop_reason is None for task in work.tasks):
                raise asyncio.CancelledError("The assignment ended before dispatch")
            if self.session.closed or not self.session.caller_attached:
                raise ValueError("The caller attachment is no longer eligible")
            if self._allocation_fits(work):
                return
            pending = {wait for wait in pending if not wait.done()}
            if not pending:
                raise ValueError("Lifecycle attempts settled without releasing enough worker capacity")
            await asyncio.wait({*pending, self.session.capacity_change()}, return_when=asyncio.FIRST_COMPLETED)

    def _allocation_failed(self, work: WorkRun, error: BaseException) -> None:
        """Fail the never-started assignment without treating a setup error as a caller cancellation."""
        barrier = work.start_barrier
        work.stop_soon(BeforeStartFailure(str(error) or "The queued assignment could not start"))
        if barrier is not None and not barrier.done():
            if isinstance(error, asyncio.CancelledError):
                barrier.cancel()
            else:
                barrier.set_exception(error)
                barrier.exception()

    async def _start_after_release(
        self, control: ControlAction, work: WorkRun, waits: tuple[asyncio.Task, ...]
    ) -> None:
        receipt = {"work_id": work.id, "task_ids": [task.task_id for task in work.tasks]}
        try:
            await self._wait_allocation(work, waits)
            self._settle(
                control,
                "applied",
                **receipt,
                task_states={task.task_id: task.record.state for task in work.tasks},
            )
            if not work.start_barrier.done():
                work.start_barrier.set_result(None)
        except (ValueError, RuntimeError) as error:
            self._settle(control, "refused", **receipt, error=str(error), executed=False)
            self._allocation_failed(work, error)
        except asyncio.CancelledError as error:
            self._settle(control, "cancelled", **receipt, error="Pending start was cancelled before dispatch")
            self._allocation_failed(work, error)
            if not work.ended:
                raise
        except Exception as error:
            self._settle(control, "failed", **receipt, error_type=type(error).__name__, executed=False)
            self._allocation_failed(work, error)

    def _selected_tasks(self, action):
        """Resolve exact target membership without changing lifecycle, source, or lineage state."""
        selected = []
        for target in action.targets:
            record = self.session.ledger.works.get(target.work_id)
            if record is None:
                raise ValueError(f"Unknown Work {target.work_id}")
            ids = target.task_ids if target.task_ids is not None else tuple(task.id for task in record.tasks)
            known = {task.id: task for task in record.tasks}
            if not ids or any(task_id not in known for task_id in ids):
                raise ValueError(f"Unknown or empty task selection for {target.work_id}")
            selected.append((target.work_id, ids, tuple(known[task_id] for task_id in ids)))
        return selected

    def _resolve_all_stops(
        self, decision: CompleteDecision, queued: tuple[str, ...]
    ) -> tuple[CompleteDecision, dict[int, tuple[str, ...]]]:
        """Select current tasks and queued requests older than the stop instruction."""
        actions = list(decision.actions)
        pending = {}
        for index, action in enumerate(actions):
            if not isinstance(action, Cancel) or action.scope != "all":
                continue
            if action.targets:
                raise DecisionRejected("A stop with scope: all cannot also select targets")
            if len(action.delegations) != 1:
                raise DecisionRejected("For scope: all, use only the caller's stop request in delegations")
            cutoff = queued.index(action.delegations[0])
            pending[index] = tuple(
                source for source in queued[:cutoff] if self.session.ledger.delegations[source].state == "queued"
            )
            targets = []
            for work in self.session.ledger.works.values():
                task_ids = tuple(
                    task.id
                    for task in work.tasks
                    if not task.ended
                    or (
                        task.successor_task_id is None
                        and any(question.open and question.owner_task_id == task.id for question in task.questions)
                    )
                )
                if task_ids:
                    targets.append(WorkTarget(work.id, task_ids))
            actions[index] = replace(action, targets=tuple(targets))
        return replace(decision, actions=tuple(actions)), pending

    def _targets(self, action, *, stopping: bool):
        targets = []
        for work_id, ids, _ in self._selected_tasks(action):
            work = self.session.works.get(work_id)
            tasks = [
                next((task for task in work.tasks if task.task_id == task_id), None) if work else None
                for task_id in ids
            ]
            if not stopping:
                for task_id, task in zip(ids, tasks, strict=True):
                    if task is None or task.ended:
                        self._continuation(task_id)
                    elif task.stop_reason is not None or task.own_cancel:
                        raise ValueError(f"Task {task_id} is stopping")
            targets.append((work_id, ids, work, tasks))
        return targets

    def _lifecycle_targets(self, action: Pause | Resume):
        """Require exact live targets; lifecycle controls never create a successor for an ended thread."""
        targets = self._targets(action, stopping=True)
        for _work_id, ids, _work, tasks in targets:
            for task_id, task in zip(ids, tasks, strict=True):
                if task is None or task.ended:
                    raise ValueError(f"Task {task_id} has ended; its lifecycle cannot be changed")
                if task.stop_reason is not None or task.own_cancel:
                    raise ValueError(f"Task {task_id} is stopping")
                if isinstance(action, Resume) and task.record.state not in ("pausing", "paused"):
                    raise ValueError(f"Task {task_id} is not paused or pausing")
        return targets

    @staticmethod
    def _lifecycle_fact(identity: str, action: Pause | Resume) -> dict:
        """Retain host-interpreted intent as data; fresh backend state remains the current-state authority."""
        return {
            "type": "message",
            "role": "developer",
            "id": f"host.{identity}",
            "content": json.dumps(
                {
                    "applied_lifecycle_intent": {
                        "action_id": identity,
                        "intent_already_interpreted": True,
                        "requested_state": "paused" if isinstance(action, Pause) else "running",
                        "source_delegation_ids": list(action.delegations),
                        "targets": [asdict(target) for target in action.targets],
                    }
                }
            ),
        }

    def _lifecycle_message(
        self,
        control: ControlAction,
        text_key: str,
        *,
        status_sources: frozenset[str] = frozenset(),
        eligible: bool = True,
        retain_for_reconnect: bool = False,
    ) -> list[dict]:
        """Publish lifecycle updates without letting a write failure undo the control action."""
        session = self.session
        publications = []
        for source in control.sources:
            row = {"source_id": source, "kind": text_key, "state": "not_written"}
            if source in status_sources:
                row["reason"] = "explicit_status_action"
                publications.append(row)
                continue
            if not eligible or not session.can_publish_lifecycle(source, retain_for_reconnect=retain_for_reconnect):
                row["reason"] = "source_or_lifecycle_superseded"
                publications.append(row)
                continue
            try:
                session.book.start(source)
                writer = session.book.running(source)
                if writer is not None:
                    item = writer.message(getattr(session.texts, text_key), "final_answer", source_item=True)
                    row.update(state="written", response_id=writer.id, item_id=item["id"])
            except Exception as error:
                row.update(state="unknown", error_type=type(error).__name__)
            publications.append(row)
        return publications

    @staticmethod
    def _stage_input(
        execution: _DecisionExecution,
        entry: _PlannedAction,
        action: Steer | Pause | Resume,
        targets: list[tuple[WorkRun, list[TaskRun]]],
        extra: Iterable[dict],
    ) -> None:
        execution.inputs.append(StagedInput(entry.control, action, targets, tuple(extra), entry.scopes))

    @staticmethod
    def _input_rows(plan: list[StagedInput], source_order: tuple[str, ...]):
        """Merge accepted portions once per task, with sources in canonical caller order."""
        rows = {}
        for entry in plan:
            host_fact = entry.extra[-1] if isinstance(entry.action, (Pause, Resume)) else None
            caller_extra = entry.extra[:-1] if host_fact is not None else entry.extra
            for work, tasks in entry.targets:
                for task in tasks:
                    row = rows.setdefault(
                        task.task_id,
                        {
                            "work": work,
                            "task": task,
                            "sources": set(),
                            "extra": [],
                            "host": [],
                            "control": False,
                            "scopes": [],
                            "entries": [],
                        },
                    )
                    row["sources"].update(entry.action.delegations)
                    row["entries"].append(entry)
                    seen = {item.get("id") for item in row["extra"] if item.get("id")}
                    for item in caller_extra:
                        if not item.get("id") or item["id"] not in seen:
                            row["extra"].append(item)
                            if item.get("id"):
                                seen.add(item["id"])
                    if host_fact is not None:
                        row["host"].append(host_fact)
                    if scope := entry.scopes.get(task.task_id):
                        row["scopes"].append(scope)
                    if isinstance(entry.action, Steer):
                        row["control"] = True
        for row in rows.values():
            yield tuple(source for source in source_order if source in row["sources"]), row

    def _prepare_inputs(
        self, plan: list[StagedInput], source_order: tuple[str, ...], projected: dict[str, Provenance]
    ) -> dict[str, dict[str, Edge]]:
        """Budget exact merged bindings before committing any staged action."""
        prepared_rows = {}
        for sources, row in self._input_rows(plan, source_order):
            work, task = row["work"], row["task"]
            prepared = self.session.ownership.validate_budget(
                sources,
                [(work, [task])],
                extra=(*row["extra"], *row["host"], *(scope.item for scope in row["scopes"])),
                projected=projected,
            )
            prepared_rows[task.task_id] = {edge.source: edge for edge in prepared[work.id]}
        return prepared_rows

    def _flush_inputs(
        self, inputs: list[StagedInput], source_order: tuple[str, ...], prepared_inputs: dict[str, dict[str, Edge]]
    ) -> None:
        """Install admitted reservations and retain queued input even if the backend wake fails."""
        session = self.session
        for sources, row in self._input_rows(inputs, source_order):
            work, task = row["work"], row["task"]
            edges = ()
            refs = ()
            retained = False
            try:
                brings = next(
                    (
                        session.incoming[source].brings
                        for source in reversed(sources)
                        if session.incoming[source].brings is not None
                    ),
                    None,
                )
                edges = tuple(prepared_inputs[task.task_id][source] for source in sources)
                refs = session.ownership.reserve(sources, [(work, [task])], {work.id: edges})[work.id]
                work.steer(
                    list(sources),
                    None,
                    brings,
                    row["extra"],
                    refs,
                    selected=[task],
                    control=row["control"] and not row["host"],
                    scopes=tuple(row["scopes"]),
                )
                retained = True
                if row["host"]:
                    work.steer([], None, None, row["host"], selected=[task], control=False, caller_input=False)
            except Exception as error:
                retained = retained or (
                    bool(refs)
                    and (
                        set(sources).issubset(task.request_ids)
                        or any(waiting.batch.placement_refs == refs for waiting in task.steering)
                    )
                )
                if retained:
                    task.wake.set()
                else:
                    session.ownership.release_unqueued(edges)
                    session.fence_unresolved_inputs(sources, tasks=(task.record,))
                self._record_input_failure(task, row["entries"], error, retained=retained)
            if retained:
                placement = Placement(work.id, (task.task_id,))
                for source in sources:
                    record = session.ledger.delegations[source]
                    record.placements = (*record.placements, placement)
                    record.work, record.tasks = record.placements[0].work, record.placements[0].tasks

    def _record_input_failure(
        self, task: TaskRun, entries: list[StagedInput], error: Exception, *, retained: bool
    ) -> None:
        for entry in entries:
            control = entry.control
            control.receipt.setdefault("input_failures", []).append(
                {"task_id": task.task_id, "retained": retained, "error_type": type(error).__name__}
            )
            if isinstance(entry.action, Steer):
                if not retained and control.delivery.get(task.task_id) == "pending":
                    control.delivery[task.task_id] = "failed"
                    self._delivery_receipt(control)
            elif isinstance(entry.action, Resume):
                if not retained:
                    control.receipt["input_delivery"] = "incomplete"
                self._settle(control, "partial", **control.receipt)
            elif not retained:
                control.receipt["input_delivery"] = "incomplete"

    def _steer(self, entry: _PlannedAction, action: Steer, execution: _DecisionExecution) -> None:
        session = self.session
        targets = self._targets(action, stopping=False)
        control, extra = entry.control, entry.extra
        ended = [
            task_id
            for _, ids, _, tasks in targets
            for task_id, task in zip(ids, tasks, strict=True)
            if task is None or task.ended
        ]
        live = [
            (work, [task for task in tasks if task is not None and not task.ended]) for _, _, work, tasks in targets
        ]
        live = [(work, tasks) for work, tasks in live if tasks]
        successors = None
        if ended:
            specs = tuple(TaskSpec(session.ledger.tasks[task_id].brief or "", task_id) for task_id in ended)
            scopes = entry.scopes
            worker_scopes = tuple((scopes[task_id],) if task_id in scopes else () for task_id in ended)
            successors = self._allocate_work(action.delegations, specs, extra, execution, worker_scopes=worker_scopes)
        control.delivery = {task.task_id: "pending" for _, tasks in live for task in tasks}
        if successors is not None:
            control.delivery.update({task.task_id: "pending" for task in successors.tasks})
        self._stage_input(execution, entry, action, live, extra)
        self.session.consume_clarifications(action.clarifications, control)
        self._settle(
            control,
            "pending",
            outcome="accepted",
            targets=[{"work_id": work_id, "task_ids": list(ids)} for work_id, ids, _, _ in targets],
            continuations=[{"from": task.continues, "to": task.task_id} for task in successors.tasks]
            if successors
            else [],
        )
        if successors is not None and successors.start_barrier is not None:
            session.spawn(self._continuations_after_release(control, successors, tuple(execution.release_waits)))

    async def _continuations_after_release(
        self, control: ControlAction, work: WorkRun, waits: tuple[asyncio.Task, ...]
    ) -> None:
        """Release only the successor allocation, leaving this Steer's live target deliveries independent."""
        try:
            await self._wait_allocation(work, waits)
            if not work.start_barrier.done():
                work.start_barrier.set_result(None)
        except (Exception, asyncio.CancelledError) as error:
            affected = [task.task_id for task in work.tasks if control.delivery.get(task.task_id) == "pending"]
            for task_id in affected:
                control.delivery[task_id] = "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"
            control.receipt.setdefault("continuation_failures", []).append(
                {"task_ids": affected, "executed": False, "error_type": type(error).__name__, "error": str(error)}
            )
            self._delivery_receipt(control)
            self._allocation_failed(work, error)
            if isinstance(error, asyncio.CancelledError):
                raise

    def _execute_pause(self, entry: _PlannedAction, action: Pause, execution: _DecisionExecution):
        control = entry.control
        try:
            targets = self._lifecycle_targets(action)
            extra = [*entry.extra, self._lifecycle_fact(control.id, action)]
        except (ValueError, RuntimeError) as error:
            self._settle(control, "refused", error=str(error))
            return None
        tasks = [task for _, _, _, selected in targets for task in selected]
        changed = {task.task_id: task.request_pause() for task in tasks}
        retained = False
        try:
            self._stage_input(execution, entry, action, [(work, selected) for _, _, work, selected in targets], extra)
            self.session.consume_clarifications(action.clarifications, control)
        except (ValueError, RuntimeError) as error:
            self._settle(control, "partial", outcome="pause_requested_input_not_retained", error=str(error))
        else:
            retained = True
            self._settle(
                control,
                "pending",
                outcome="pausing",
                task_ids=[task.task_id for task in tasks],
                input_delivery="retained",
                changed=changed,
                publications=self._lifecycle_message(
                    control, "pause_requested_line", status_sources=execution.status_sources
                ),
            )

        confirmations = tuple(self.session.spawn(task.wait_paused()) for task in tasks)

        async def confirm():
            try:
                outcomes = await asyncio.gather(*confirmations)
                outcome = "unconfirmed" if "unconfirmed" in outcomes else "stopped"
            except Exception:
                outcome = "unconfirmed"
            except asyncio.CancelledError:
                self._settle(control, "unknown", outcome="pause_unconfirmed", task_ids=[t.task_id for t in tasks])
                raise
            receipt = {
                **control.receipt,
                "outcome": "paused" if outcome == "stopped" else "pause_unconfirmed",
                "task_ids": [task.task_id for task in tasks],
                "task_states": {task.task_id: task.record.state for task in tasks},
                "publications": [
                    *control.receipt.get("publications", ()),
                    *self._lifecycle_message(
                        control,
                        "paused_line" if outcome == "stopped" else "pause_unconfirmed_line",
                        eligible=all(task.want_paused and task.stop_reason is None for task in tasks),
                    ),
                ],
            }
            input_retained = retained and not control.receipt.get("input_failures")
            state = "unknown" if outcome != "stopped" else "applied" if input_retained else "partial"
            self._settle(control, state, **receipt)
            return outcome

        self.session.spawn(confirm())
        return confirmations

    def _execute_resume(self, entry: _PlannedAction, action: Resume, execution: _DecisionExecution) -> None:
        control = entry.control
        targets = self._lifecycle_targets(action)
        extra = [*entry.extra, self._lifecycle_fact(control.id, action)]
        tasks = [task for _, _, _, selected in targets for task in selected]
        changed = {task.task_id: task.request_resume() for task in tasks}
        accepted = [task.task_id for task in tasks]
        try:
            self._stage_input(execution, entry, action, [(work, selected) for _, _, work, selected in targets], extra)
        except (ValueError, RuntimeError) as error:
            self._settle(
                control,
                "partial",
                outcome="resume_accepted_input_not_retained",
                task_ids=accepted,
                error=str(error),
            )
            return
        self.session.consume_clarifications(action.clarifications, control)
        self._settle(
            control,
            "applied",
            outcome="resume_accepted",
            task_ids=accepted,
            task_states={task.task_id: task.record.state for task in tasks},
            input_delivery="retained",
            inference_started_at_acceptance=False,
            changed=changed,
            publications=self._lifecycle_message(
                control, "resume_requested_line", status_sources=execution.status_sources
            ),
        )

    def _execute_stop(self, entry: _PlannedAction, action: Cancel):
        """Stop selected work, retaining the outcome if a shared response closes on an app call."""
        control, pending = entry.control, entry.pending_stops
        try:
            targets = self._targets(action, stopping=True)
        except (ValueError, RuntimeError) as error:
            self._settle(control, "refused", error=str(error))
            return None
        report_on_work = any(
            self.session.ledger.controls[key].kind in ("Start", "Steer", "Pause", "Resume")
            for source in control.sources
            for key in self.session.ownership.owner(source).controls
        )
        reason = ForNewerRequest(report_on_work=report_on_work)
        asked = []
        already_ended = []
        closed_questions = {}
        for _work_id, ids, work, tasks in targets:
            for task_id, task in zip(ids, tasks, strict=True):
                if task is None or task.ended:
                    already_ended.append(task_id)
                    record = self.session.ledger.tasks.get(task_id)
                    if record is not None and (questions := self.session.cancel_ended_questions(record, control.id)):
                        closed_questions[task_id] = questions
                elif task.stop_reason is None:
                    asked.extend(work.ask_stop(reason, task))
                else:
                    asked.append(task)
        self.session.consume_clarifications(action.clarifications, control)
        if action.scope == "all" and not targets and not pending:
            self._settle(
                control,
                "applied",
                outcome="no_active_work",
                publications=self._lifecycle_message(control, "no_active_work_line"),
            )
            return None
        evidence = {"closed_questions": closed_questions} if closed_questions else {}
        if pending:
            for delegation in pending:
                self.session.book.cancelled(delegation)
                self.session.ledger.delegations[delegation].move("cancelled")
            evidence["cancelled_delegation_ids"] = list(pending)
        if asked or (closed_questions and not pending):
            for source in action.delegations:
                self.session.book.start(source)
                if not asked and (writer := self.session.book.running(source)) is not None:
                    writer.message(self.session.texts.control_requested_line, "final_answer", source_item=True)
        if not asked:
            if pending:
                self._settle(
                    control,
                    "applied",
                    outcome="stopped",
                    task_ids=[],
                    already_ended=already_ended,
                    publications=self._lifecycle_message(control, "queued_cancelled_line"),
                    **evidence,
                )
            else:
                if already_ended and not closed_questions:
                    evidence["publications"] = self._lifecycle_message(control, "already_ended_line")
                self._settle(control, "applied", outcome="already_ended", task_ids=already_ended, **evidence)
            return None
        control.receipt.update(evidence)

        confirmations = tuple(self.session.spawn(task.confirm_stop()) for task in asked)

        async def confirm():
            try:
                outcomes = await asyncio.gather(*confirmations)
                outcome = "unconfirmed" if "unconfirmed" in outcomes else "stopped"
            except Exception:
                outcome = "unconfirmed"
            except asyncio.CancelledError:
                self._settle(
                    control, "unknown", outcome="unconfirmed", task_ids=[task.task_id for task in asked], **evidence
                )
                raise
            confirmed = outcome == "stopped"
            line = "stop_confirmed_line" if confirmed else "stop_unconfirmed_line"
            self._settle(
                control,
                "applied" if confirmed else "unknown",
                outcome=outcome,
                task_ids=[task.task_id for task in asked],
                already_ended=already_ended,
                publications=self._lifecycle_message(control, line, retain_for_reconnect=not report_on_work),
                **evidence,
            )
            return outcome

        self.session.spawn(confirm())
        return confirmations

    def _status_text(self, action: CallerMessage) -> str:
        """Report execution state using the selected jobs' ledger records."""
        session = self.session
        states = {}
        for _, _, selected in self._selected_tasks(action):
            for target in selected:
                task = session.latest_task(target)
                if task.successor_task_id is not None:
                    state = "continued"
                elif task.stop.quiescent_later:
                    state = "stopped"
                elif task.stop.state != "none":
                    state = task.stop.state
                elif task.state in ("running", "answered") and any(
                    question.open and question.owner_task_id == task.id and question.dispatch == "written"
                    for question in task.questions
                ):
                    state = "waiting_for_caller"
                else:
                    state = task.state
                states[task.id] = ((task.brief or "").strip(), session.texts.task_status[state])
        if len(states) <= 1:
            return next((text for _, text in states.values()), "")
        brief_counts = Counter(brief for brief, _ in states.values())
        return "\n".join(
            session.texts.task_status_label.format(
                task=brief if brief and brief_counts[brief] == 1 and brief not in states else task_id,
                status=text,
            )
            for task_id, (brief, text) in states.items()
        )

    def _message(self, entry: _PlannedAction, action: CallerMessage) -> None:
        session = self.session
        control, extra = entry.control, entry.extra
        message = self._status_text(action) if action.kind == "status" and action.targets else action.message
        clarification_id = None
        carrier = action.delegations[-1]
        if action.kind == "question":
            clarification_id = session.open_clarification(action, control, extra)
        session.consume_clarifications(action.clarifications, control, superseded_by=clarification_id)
        previous = None
        for index, source in enumerate(action.delegations):
            if source != carrier and session.ownership.handover_message_source(
                source, action.delegations[index + 1], carrier, control.id, previous
            ):
                previous = source
        session.book.start(carrier, previous=previous)
        writer = session.book.running(carrier)
        if writer is None:
            raise RuntimeError("The caller-message response is unavailable")
        item = writer.message(message, "final_answer", source_item=True, routing_question=clarification_id is not None)
        session.conversation.join(
            [*[item for source in action.delegations for item in session.request_items(source)], item]
        )
        if clarification_id is not None:
            session.clarification_written(clarification_id, item)
        session.prune_clarifications()
        self._settle(
            control,
            "applied",
            kind=action.kind,
            clarification_id=clarification_id,
            publication="written",
            carrier=carrier,
        )
