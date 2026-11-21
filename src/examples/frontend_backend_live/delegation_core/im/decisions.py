# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Admit complete routing decisions, then execute their independent Core-owned obligations."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

from examples.frontend_backend_live.delegation_core.backend.port import (
    CallerMessage,
    Cancel,
    CompleteDecision,
    DecisionRejected,
    Pause,
    Resume,
    Start,
    Steer,
    TaskSpec,
)
from examples.frontend_backend_live.delegation_core.im.input import InputScope
from examples.frontend_backend_live.delegation_core.im.stop import BeforeStartFailure, ForNewerRequest
from examples.frontend_backend_live.delegation_core.ledger.records import ControlAction, Placement
from examples.shared.ids import approx_tokens, uid

if TYPE_CHECKING:
    from examples.frontend_backend_live.delegation_core.im.run import WorkRun
    from examples.frontend_backend_live.delegation_core.im.session import IMSession


class WorkerCapacityUnavailable(DecisionRejected):
    """A worker allocation was refused before it could start."""

    outcome = "capacity_unavailable"

    def __init__(self, requested: int, available: int):
        """Report the slot counts at the time of refusal."""
        self.details = {"reason": self.outcome, "requested": requested, "available": available, "executed": False}
        super().__init__(json.dumps(self.details))


class RoutingTargetUnavailable(DecisionRejected):
    """Every proposed action has an ineligible target. No action was admitted."""

    outcome = "target_unavailable"


class DecisionExecutor:
    """Use the existing ledger, Work branches and stop door; never infer semantic routing."""

    def __init__(self, session: IMSession):
        """Bind this session's existing records, tools, and execution callbacks."""
        self.session = session
        self.submissions: dict[int, tuple[str, tuple[str, ...]]] = {}
        self._extras: dict[str, list[dict] | None] = {}
        self._scopes: dict[str, dict[str | int, InputScope]] = {}
        self._input_plan: list[tuple] = []
        self._source_order: tuple[str, ...] = ()
        self._status_sources: dict[str, frozenset[str]] = {}

    def _receipt(self, ids: tuple[str, ...]) -> str:
        controls = self.session.ledger.controls
        return json.dumps({"actions": [asdict(controls[key]) for key in ids]})

    def submit(self, decision: CompleteDecision) -> str:
        """Validate source associations as one envelope before registering or performing effects."""
        session = self.session
        signature = json.dumps(asdict(decision), sort_keys=True)
        self.submissions = {
            epoch: value for epoch, value in self.submissions.items() if epoch == session._decision_epoch
        }
        prior = self.submissions.get(session._decision_epoch)
        if prior is not None:
            if prior[0] != signature:
                raise DecisionRejected("A placement invocation has already accepted a different complete decision")
            return self._receipt(prior[1])
        if not decision.actions:
            raise DecisionRejected("A complete decision must contain actions")
        selected_sources = {source for action in decision.actions for source in action.delegations}
        sources = tuple(source for source in session.arrivals if source in selected_sources)
        if not sources or selected_sources != set(sources) or any(source not in session._given for source in sources):
            raise DecisionRejected("Every action must name only its bound pending caller sources")
        missing_sources = (set(session.arrivals) & session._given) - selected_sources
        if missing_sources:
            raise DecisionRejected(
                "A complete decision must cover every bound pending caller source; missing: "
                + ", ".join(sorted(missing_sources))
            )
        for action in decision.actions:
            if not action.delegations or len(set(action.delegations)) != len(action.delegations):
                raise DecisionRejected("Action sources must be a nonempty unique selection")
            if not isinstance(action, (Start, Steer, Cancel, Pause, Resume, CallerMessage)):
                raise DecisionRejected("Unsupported decision action")
            if isinstance(action, CallerMessage) and action.kind == "status":
                if decision.model_authored and (action.message or not action.targets):
                    raise DecisionRejected("Operational status selects work targets for Core to report")
                if not session.ledger.works:
                    raise DecisionRejected("Operational status requires open or retained ended work records")
            if session._clarified_input(action.clarifications, consume=False) is None:
                raise DecisionRejected("Every action must bind only live, unique routing clarification identities")
            if isinstance(action, Start) and len(action.tasks) > 1:
                initial_sources = set(action.delegations)
                for clarification in action.clarifications:
                    initial_sources.update(session.clarifications[clarification]["source_delegation_ids"])
                if len(initial_sources) > 1:
                    raise DecisionRejected(
                        "Use one start action per task when starting from several caller sources, including retained "
                        "clarification sources. Give each start only its applicable sources in this same decision."
                    )
        selected = {}
        continuations = set()
        for action in decision.actions:
            if isinstance(action, Start):
                for spec in action.tasks:
                    if spec.continues is not None:
                        if spec.continues in continuations:
                            raise DecisionRejected("A thread can have only one successor in a decision")
                        continuations.add(spec.continues)
            if isinstance(action, (Steer, Cancel, Pause, Resume)):
                if not action.targets:
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
        if approx_tokens(signature) > session.settings.context_tokens:
            raise DecisionRejected("Complete decision exceeds the backend context budget")
        input_scopes = self._plan_scopes(decision)
        continued = [self._continued_targets(action) for action in decision.actions]
        if all(continued):
            # Every atomic action is already impossible for a known lineage reason. Admit nothing, so the
            # existing bounded corrective decision can read fresh facts without replaying accepted effects.
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
        eligible = [action for index, action in enumerate(decision.actions) if index not in refusals]
        if session.backend.max_running > 1 and all(isinstance(action, (Start, CallerMessage)) for action in eligible):
            requested = sum(len(action.tasks) or 1 for action in eligible if isinstance(action, Start))
            available = session.available_worker_slots()
            if requested > available and (not releasing or requested > session.backend.max_running):
                error = WorkerCapacityUnavailable(requested, available)
                session.observe("im.decision.rejected", **error.details, sources=list(sources))
                raise error
        # A budget rejection admits nothing. Mixed plans retain individual target refusals.
        identity = uid("decision")
        ids = tuple(f"{identity}.{index}" for index in range(1, len(decision.actions) + 1))
        budget_plan = []
        for index, (key, action, scopes) in enumerate(zip(ids, decision.actions, input_scopes, strict=True)):
            if index in refusals:
                continue
            if isinstance(action, (Steer, Pause, Resume)):
                try:
                    targets = (
                        self._targets(action, stopping=False)
                        if isinstance(action, Steer)
                        else self._lifecycle_targets(action)
                    )
                except ValueError:
                    continue
                live = [
                    (work, [task for task in tasks if task is not None and not task.ended])
                    for _, _, work, tasks in targets
                ]
                live = [(work, tasks) for work, tasks in live if tasks]
                if live:
                    extra = session._clarified_input(action.clarifications, consume=False) or ()
                    if isinstance(action, (Pause, Resume)):
                        extra = (*extra, self._lifecycle_fact(key, action))
                    session.ownership.validate_budget(
                        action.delegations,
                        live,
                        extra=extra,
                    )
                    budget_plan.append((action, live, extra, scopes))
        self._prepare_inputs(budget_plan, sources)
        # Install every obligation before taking sources, starting workers, or allowing completion callbacks.
        for key in list(self._extras):
            if key not in session.ledger.controls:
                del self._extras[key]
                self._scopes.pop(key, None)
                self._status_sources.pop(key, None)
        status_sources = {
            source
            for action in decision.actions
            if isinstance(action, CallerMessage) and action.kind == "status" and not action.targets
            for source in action.delegations
        }
        for key, action, scopes in zip(ids, decision.actions, input_scopes, strict=True):
            self._extras[key] = session._clarified_input(action.clarifications, consume=False)
            self._scopes[key] = scopes
            self._status_sources[key] = frozenset(set(action.delegations) & status_sources)
            control = session.ledger.controls[key] = ControlAction(key, type(action).__name__, action.delegations)
            for source in control.sources:
                owner = session.ownership.owner(source)
                placements = [item for item in decision.actions if source in item.delegations]
                composed = len(placements) > 1 or isinstance(action, (Cancel, Pause, Resume, CallerMessage))
                if isinstance(action, Start):
                    composed = composed or len(action.tasks) > 1
                elif isinstance(action, Steer):
                    composed = (
                        composed
                        or len(action.targets) > 1
                        or any(
                            len(session.ledger.works[target.work_id].tasks) > 1
                            for target in action.targets
                            if target.work_id in session.ledger.works
                        )
                    )
                owner.composed = owner.composed or composed
                owner.controls.add(key)
        self.submissions[session._decision_epoch] = signature, ids
        ordered_actions = tuple(
            replace(action, delegations=tuple(source for source in sources if source in action.delegations))
            for action in decision.actions
        )
        session._admitting_decision = True
        self._input_plan = []
        self._source_order = sources
        try:
            if refusals:
                session.fence_unresolved_inputs(
                    source
                    for index in refusals
                    if not isinstance(decision.actions[index], CallerMessage)
                    for source in decision.actions[index].delegations
                )
            session._take(sources, session._given)
            refused_ids = {ids[index] for index in refusals}
            for index, error in refusals.items():
                self._settle(
                    session.ledger.controls[ids[index]],
                    "refused",
                    outcome=RoutingTargetUnavailable.outcome,
                    executed=False,
                    error=error,
                )
            release_waits = [session.spawn(task.confirm_stop()) for task in releasing]
            for key, action in zip(ids, ordered_actions, strict=True):
                if key in refused_ids or not isinstance(action, (Cancel, Pause)):
                    continue
                control = session.ledger.controls[key]
                try:
                    confirmations = (
                        self._execute_stop(control, action)
                        if isinstance(action, Cancel)
                        else self._execute_pause(control, action)
                    )
                    if confirmations:
                        release_waits.extend(confirmations)
                        releasing.update(
                            task
                            for _, _, _, tasks in self._targets(action, stopping=True)
                            for task in tasks
                            if task is not None and not task.ended
                        )
                except Exception as error:
                    self._settle(control, "unknown", error_type=type(error).__name__)
            # Apply all supplied updates before recording resume intent, regardless of envelope member order.
            execution_order = (Steer, Resume, Start)
            for action_type in execution_order:
                for key, action in zip(ids, ordered_actions, strict=True):
                    if key in refused_ids or not isinstance(action, action_type):
                        continue
                    control = session.ledger.controls[key]
                    try:
                        if isinstance(action, Start):
                            self._start(control, action, waits=tuple(release_waits), releasing=frozenset(releasing))
                        elif isinstance(action, Steer):
                            self._steer(control, action, waits=tuple(release_waits), releasing=frozenset(releasing))
                        elif isinstance(action, Resume):
                            self._execute_resume(control, action)
                    except WorkerCapacityUnavailable as error:
                        self._settle(control, "refused", outcome=error.outcome, **error.details)
                    except (ValueError, RuntimeError) as error:
                        self._settle(control, "refused", error=str(error))
                    except Exception as error:
                        self._settle(control, "unknown", error_type=type(error).__name__)
            try:
                self._flush_inputs()
            except Exception as error:
                for control, action, _live, _extra in self._input_plan:
                    self._settle(
                        control,
                        "partial" if isinstance(action, (Pause, Resume)) else "unknown",
                        **{**control.receipt, "input_delivery": "not_retained", "error_type": type(error).__name__},
                    )
            for key, action in zip(ids, ordered_actions, strict=True):
                if isinstance(action, CallerMessage) and key not in refused_ids:
                    try:
                        self._message(session.ledger.controls[key], action)
                    except (ValueError, RuntimeError) as error:
                        self._settle(session.ledger.controls[key], "refused", error=str(error))
        finally:
            self._input_plan = []
            self._source_order = ()
            session._admitting_decision = False
            if not session.arrivals:
                session._decided.set()
            for source in sources:
                session.ownership.finish_owner(session.ownership.owner(source))
            session._prune()
        session.observe("im.decision.accepted", decision_id=identity, action_ids=list(ids), sources=list(sources))
        return self._receipt(ids)

    def _plan_scopes(self, decision: CompleteDecision) -> list[dict[str | int, InputScope]]:
        """Bind normalized worker input before taking any source or issuing a lifecycle effect."""
        session, actions = self.session, decision.actions
        plans: list[dict[str | int, InputScope]] = [{} for _ in actions]
        if session.backend.max_running <= 1:
            return plans
        sources = [
            set(action.delegations).union(
                source
                for key in action.clarifications
                for source in session.clarifications[key]["source_delegation_ids"]
            )
            for action in actions
        ]
        for index, action in enumerate(actions):
            if not isinstance(action, (Start, Steer, Pause, Resume)):
                continue
            shared = decision.model_authored and any(
                sources[index] & other for offset, other in enumerate(sources) if offset != index
            )
            raw = [
                *(session._clarified_input(action.clarifications, consume=False) or ()),
                *(item for source in action.delegations for item in session.request_items(source)),
            ]
            if isinstance(action, Start):
                for task_index, spec in enumerate(action.tasks):
                    if (
                        shared
                        or (decision.model_authored and len(action.tasks) > 1)
                        or spec.evidence
                        or spec.caller_lines
                    ):
                        plans[index][task_index] = InputScope.bind(
                            spec.brief,
                            spec.evidence,
                            raw,
                            session._caller_context,
                            session.playback_owners(spec.continues),
                            selections=spec.caller_lines,
                        )
                continue
            try:
                targets = tuple(task for _, ids, _ in self._selected_tasks(action) for task in ids)
            except ValueError:
                continue  # A bad target receives the existing per-action refusal.
            mixed = shared or (decision.model_authored and len(targets) > 1)
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
                    plans[index][task] = InputScope.bind(
                        "\n".join(update.text for update in fragments),
                        tuple(quote for update in fragments for quote in update.evidence),
                        raw,
                        session._caller_context,
                        session.playback_owners(task),
                        require_evidence=mixed,
                        selections=tuple(line for update in fragments for line in update.caller_lines),
                        include_bound_lines=decision.model_authored and not mixed,
                    )
            elif mixed:
                verb = "pause" if isinstance(action, Pause) else "resume"
                for task in targets:
                    plans[index][task] = InputScope.bind(
                        f"The caller asked to {verb} this task. Core already handles that lifecycle action.",
                        (),
                        raw,
                        session._caller_context,
                        session.playback_owners(task),
                    )
        if decision.model_authored:
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
                            "Use distinct source excerpts for each worker. Put shared facts in each brief or update."
                        )
                    assigned.append(scope)
        return plans

    def _extra(self, action, control):
        extra = self._extras[control.id]
        if extra is None:
            raise ValueError("Unknown or already consumed routing clarification")
        return extra

    def _consume(self, action, control):
        # All selected actions were registered against the same original clarification snapshot.
        for key in action.clarifications:
            row = self.session.clarifications[key]
            row["state"] = "consumed"
            row.setdefault("consumed_by_actions", []).append(control.id)
            self.session.observe(
                "im.clarification.consumed",
                clarification_id=key,
                action_id=control.id,
                delegation_ids=list(action.delegations),
                source_delegation_ids=list(row["source_delegation_ids"]),
            )

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
        for source in control.sources:
            owner = session.ownership.owner(source)
            owner.revision += 1
            if not session._admitting_decision:
                session.ownership.finish_owner(owner)

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

    def _start(
        self,
        control: ControlAction,
        action: Start,
        *,
        waits: tuple[asyncio.Task, ...] = (),
        releasing: frozenset = frozenset(),
    ) -> None:
        extra = self._extra(action, control)
        scopes = self._scopes[control.id]
        worker_scopes = tuple((scopes[index],) if index in scopes else () for index in range(len(action.tasks)))
        work = self._allocate_work(
            action.delegations, action.tasks, extra, waits=waits, releasing=releasing, worker_scopes=worker_scopes
        )
        self._consume(action, control)
        receipt = {"work_id": work.id, "task_ids": [task.task_id for task in work.tasks]}
        if work.start_barrier is None:
            self._settle(control, "applied", **receipt)
        else:
            self._settle(control, "pending", **receipt, outcome="waiting_for_resource_release")
            self.session.spawn(self._start_after_release(control, work, waits))

    def _allocate_work(
        self,
        sources: tuple[str, ...],
        specs: tuple[TaskSpec, ...],
        extra,
        *,
        waits: tuple[asyncio.Task, ...],
        releasing: frozenset,
        worker_scopes: tuple[tuple[InputScope, ...], ...] = (),
    ) -> WorkRun:
        """Allocate one exact assignment with the same capacity and continuation checks for Start and Steer."""
        session = self.session
        if session.closed or not session.caller_attached:
            raise ValueError("The caller attachment is no longer eligible")
        requested = len(specs) or 1
        available = session.available_worker_slots()
        needs_release = requested > available
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
            retained = len(occupied - releasing)
            if not waits or retained + requested > session.backend.max_running:
                raise WorkerCapacityUnavailable(requested, available)
        for spec in specs:
            if spec.continues is not None:
                self._continuation(spec.continues)
        barrier = asyncio.get_running_loop().create_future() if needs_release else None
        return session._start_work(
            list(sources), specs, extra=extra, start_barrier=barrier, worker_scopes=worker_scopes
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
        self, control: ControlAction, text_key: str, *, confirmation: bool = False, eligible: bool = True
    ) -> list[dict]:
        """Publish host lifecycle facts to their real source without applying its retained worker input."""
        session = self.session
        publications = []
        for source in control.sources:
            row = {"source_id": source, "kind": text_key, "state": "not_written"}
            if not confirmation and source in self._status_sources.get(control.id, ()):
                row["reason"] = "explicit_status_action"
                publications.append(row)
                continue
            record = session.ledger.delegations.get(source)
            if (
                not eligible
                or session.closed
                or not session.caller_attached
                or record is None
                or record.ended
                or record.call_generation != session._call_generation
            ):
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
                # Publication failure cannot erase an accepted lifecycle operation or skip its confirmation.
                row.update(state="unknown", error_type=type(error).__name__)
            publications.append(row)
        return publications

    def _attach_input(self, control: ControlAction, action, live, extra) -> None:
        """Defer input queueing until every envelope action has installed its obligations and fences."""
        self._input_plan.append((action, live, extra, self._scopes[control.id]))

    def _prepare_inputs(self, plan, source_order):
        """Budget the merged exact source/task bindings before retaining any of this envelope's input."""
        rows = {}
        for action, live, extra, scopes in plan:
            host_fact = extra[-1] if isinstance(action, (Pause, Resume)) else None
            caller_extra = extra[:-1] if host_fact is not None else extra
            for work, tasks in live:
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
                        },
                    )
                    row["sources"].update(action.delegations)
                    seen = {item.get("id") for item in row["extra"] if item.get("id")}
                    for item in caller_extra:
                        if not item.get("id") or item["id"] not in seen:
                            row["extra"].append(item)
                            if item.get("id"):
                                seen.add(item["id"])
                    if host_fact is not None:
                        row["host"].append(host_fact)
                    if scope := scopes.get(task.task_id):
                        row["scopes"].append(scope)
                    if isinstance(action, Steer):
                        row["control"] = True
        prepared_rows = []
        for row in rows.values():
            sources = tuple(source for source in source_order if source in row["sources"])
            work, task = row["work"], row["task"]
            prepared = self.session.ownership.validate_budget(
                sources, [(work, [task])], extra=(*row["extra"], *row["host"], *(scope.item for scope in row["scopes"]))
            )
            prepared_rows.append((sources, row, prepared))
        return prepared_rows

    def _flush_inputs(self) -> None:
        """Queue caller input in canonical source order, once per exact task, then separate host-only facts."""
        session = self.session
        for sources, row, prepared in self._prepare_inputs(self._input_plan, self._source_order):
            work, task = row["work"], row["task"]
            refs = session.ownership.reserve(sources, [(work, [task])], prepared)
            placement = Placement(work.id, (task.task_id,))
            for source in sources:
                record = session.ledger.delegations[source]
                record.placements = (*record.placements, placement)
                record.work, record.tasks = record.placements[0].work, record.placements[0].tasks
            brings = next(
                (
                    session.incoming[source].brings
                    for source in reversed(sources)
                    if session.incoming[source].brings is not None
                ),
                None,
            )
            work.steer(
                list(sources),
                None,
                brings,
                row["extra"],
                refs[work.id],
                selected=[task],
                control=row["control"] and not row["host"],
                scopes=tuple(row["scopes"]),
            )
            if row["host"]:
                work.steer([], None, None, row["host"], selected=[task], control=False, caller_input=False)

    def _steer(
        self,
        control: ControlAction,
        action: Steer,
        *,
        waits: tuple[asyncio.Task, ...] = (),
        releasing: frozenset = frozenset(),
    ) -> None:
        session = self.session
        targets = self._targets(action, stopping=False)
        extra = self._extra(action, control)
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
        if live:
            session.ownership.validate_budget(action.delegations, live, extra=extra)
        successors = None
        if ended:
            specs = tuple(TaskSpec(session.ledger.tasks[task_id].brief or "", task_id) for task_id in ended)
            scopes = self._scopes[control.id]
            worker_scopes = tuple((scopes[task_id],) if task_id in scopes else () for task_id in ended)
            successors = self._allocate_work(
                action.delegations, specs, extra, waits=waits, releasing=releasing, worker_scopes=worker_scopes
            )
        control.delivery = {task.task_id: "pending" for _, tasks in live for task in tasks}
        if successors is not None:
            control.delivery.update({task.task_id: "pending" for task in successors.tasks})
        self._attach_input(control, action, live, extra)
        self._consume(action, control)
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
            session.spawn(self._continuations_after_release(control, successors, waits))

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

    def _execute_pause(self, control: ControlAction, action: Pause):
        try:
            targets = self._lifecycle_targets(action)
            extra = [*self._extra(action, control), self._lifecycle_fact(control.id, action)]
        except (ValueError, RuntimeError) as error:
            self._settle(control, "refused", error=str(error))
            return None
        tasks = [task for _, _, _, selected in targets for task in selected]
        changed = {task.task_id: task.request_pause() for task in tasks}
        retained = False
        try:
            self._attach_input(control, action, [(work, selected) for _, _, work, selected in targets], extra)
            self._consume(action, control)
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
                publications=self._lifecycle_message(control, "pause_requested_line"),
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
                        confirmation=True,
                        eligible=all(task.want_paused and task.stop_reason is None for task in tasks),
                    ),
                ],
            }
            input_retained = retained and control.receipt.get("input_delivery") != "not_retained"
            state = "unknown" if outcome != "stopped" else "applied" if input_retained else "partial"
            self._settle(control, state, **receipt)
            return outcome

        self.session.spawn(confirm())
        return confirmations

    def _execute_resume(self, control: ControlAction, action: Resume) -> None:
        targets = self._lifecycle_targets(action)
        extra = [*self._extra(action, control), self._lifecycle_fact(control.id, action)]
        tasks = [task for _, _, _, selected in targets for task in selected]
        changed = {task.task_id: task.request_resume() for task in tasks}
        accepted = [task.task_id for task in tasks]
        try:
            self._attach_input(control, action, [(work, selected) for _, _, work, selected in targets], extra)
        except (ValueError, RuntimeError) as error:
            self._settle(
                control,
                "partial",
                outcome="resume_accepted_input_not_retained",
                task_ids=accepted,
                error=str(error),
            )
            return
        self._consume(action, control)
        self._settle(
            control,
            "applied",
            outcome="resume_accepted",
            task_ids=accepted,
            task_states={task.task_id: task.record.state for task in tasks},
            input_delivery="retained",
            inference_started_at_acceptance=False,
            changed=changed,
            publications=self._lifecycle_message(control, "resume_requested_line"),
        )

    def _execute_stop(self, control: ControlAction, action: Cancel):
        try:
            targets = self._targets(action, stopping=True)
            self._extra(action, control)
        except (ValueError, RuntimeError) as error:
            self._settle(control, "refused", error=str(error))
            return None
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
                    asked.extend(work.ask_stop(ForNewerRequest(), task))
                else:
                    asked.append(task)
        self._consume(action, control)
        evidence = {"closed_questions": closed_questions} if closed_questions else {}
        if asked or closed_questions:
            for source in action.delegations:
                self.session.book.start(source)
                writer = self.session.book.running(source)
                if writer is not None:
                    writer.message(self.session.texts.control_requested_line, "final_answer", source_item=True)
        if not asked:
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
            self._settle(
                control,
                "unknown" if outcome == "unconfirmed" else "applied",
                outcome=outcome,
                task_ids=[task.task_id for task in asked],
                already_ended=already_ended,
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
            for task in selected:
                while task.successor_task_id in session.ledger.tasks:
                    task = session.ledger.tasks[task.successor_task_id]
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
                states[task.id] = (task.brief, session.texts.task_status[state])
        return "\n".join(
            session.texts.task_status_label.format(task=brief, status=text) if brief and len(states) > 1 else text
            for brief, text in states.values()
        )

    def _message(self, control: ControlAction, action: CallerMessage) -> None:
        session = self.session
        message = self._status_text(action) if action.kind == "status" and action.targets else action.message
        extra = self._extra(action, control)
        clarification_id = None
        carrier = action.delegations[-1]
        if action.kind == "question":
            waiting = sum(row["state"] == "waiting" for row in session.clarifications.values())
            if waiting - len(action.clarifications) >= session.settings.max_pending_delegations:
                raise ValueError("Too many unresolved routing questions")
            clarification_id = uid("clarification")
            prior_sources = [
                source
                for key in action.clarifications
                for source in session.clarifications[key]["source_delegation_ids"]
            ]
            accepted_ids = {
                key
                for source in action.delegations
                for key in session.ownership.owner(source).controls
                if key != control.id
            }
            accepted_ids.update(
                key
                for old in action.clarifications
                for key in session.clarifications[old].get("accepted_action_ids", ())
            )
            row = {
                "id": clarification_id,
                "action_id": control.id,
                "state": "waiting",
                "question": action.message,
                "source_delegation_ids": list(dict.fromkeys([*prior_sources, *action.delegations])),
                "reply_delegation_ids": list(action.delegations),
                "supersedes": list(action.clarifications),
                "source_epoch": session._decision_epoch,
                "source_placement": session._placement_serial,
                "carrier": carrier,
                "subjects": session._decision_subjects,
                "input": [*extra, *[item for source in action.delegations for item in session.request_items(source)]],
                "accepted_action_ids": sorted(accepted_ids),
            }
            if approx_tokens(json.dumps(row)) > session.settings.context_tokens:
                raise ValueError("Routing clarification exceeds context budget")
            session.clarifications[clarification_id] = row
        self._consume(action, control)
        if clarification_id is not None:
            for old in action.clarifications:
                session.clarifications[old].update(state="superseded", superseded_by=clarification_id)
        # Several sources can ask one shared question. Only message-only older sources hand over here.
        previous = None
        for index, source in enumerate(action.delegations):
            owner = session.ownership.owner(source)
            if source != carrier and not owner.edges and owner.controls == {control.id}:
                session.book.start(source, previous=previous)
                taker = action.delegations[index + 1]
                writer = session.book.running(source)
                if writer is not None:
                    writer.message(session.texts.handover_line.format(delegation=taker), "commentary", source_item=True)
                owner.disposition = "handed_over"
                owner.revision += 1
                record = session.ledger.delegations[source]
                if not record.ended:
                    record.move("handed_over")
                session.book.end(source, "response.completed")
                session.ownership._remember_settlement(
                    session.ownership.owner(carrier), session.ownership._settlement(owner)
                )
                previous = source
        session.book.start(carrier, previous=previous)
        writer = session.book.running(carrier)
        if writer is None:
            raise RuntimeError("The caller-message response is unavailable")
        item = writer.message(message, "final_answer", source_item=True)
        session.conversation.join(
            [*[item for source in action.delegations for item in session.request_items(source)], item]
        )
        if clarification_id is not None:
            session.clarifications[clarification_id]["input"].append(item)
            session.observe(
                "im.clarification.created",
                clarification_id=clarification_id,
                delegation_ids=list(action.delegations),
                source_delegation_ids=row["source_delegation_ids"],
                supersedes=list(action.clarifications),
                source_epoch=session._decision_epoch,
                source_placement=session._placement_serial,
            )
        session._prune_clarifications()
        self._settle(
            control,
            "applied",
            kind=action.kind,
            clarification_id=clarification_id,
            publication="written",
            carrier=carrier,
        )
