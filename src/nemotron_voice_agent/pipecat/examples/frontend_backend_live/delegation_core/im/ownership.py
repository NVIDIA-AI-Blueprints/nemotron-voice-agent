# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-source response obligations, separate from a Work's computation and input delivery."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import (
    DecisionRejected,
    StopOutcome,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.decisions import (
    RoutingTargetUnavailable,
    WorkerCapacityUnavailable,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.work_provenance import (
    Attribution,
    Binding,
    BranchSources,
    Obligation,
    OwnerSettlement,
    Provenance,
    encode_provenance,
)
from nemotron_voice_agent.pipecat.shared.ids import approx_tokens, uid

if TYPE_CHECKING:
    from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.run import (
        Batch,
        Parked,
        TaskRun,
        WorkRun,
    )
    from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.session import IMSession


@dataclass
class Edge:
    """One canonical source's selected target and its actual per-task delivery state."""

    ref: str
    source: str
    branch: str
    pending: set[str]
    task_scope: frozenset[str] | None = field(init=False)
    state: str = "pending"
    outcome: str | None = None
    stop: StopOutcome | None = None
    to_owner: str | None = None
    transferred_revision: int | None = None
    continuation_of: str | None = None

    def __post_init__(self) -> None:
        """Keep the selected tasks after delivery. Publication-only edges cover their branch."""
        self.task_scope = frozenset(self.pending) if self.pending else None

    def record_stops(self, outcomes: Iterable[StopOutcome | None]) -> None:
        """Preserve uncertainty among the stops that apply to this source."""
        stops = set(outcomes) | {self.stop}
        self.stop = "unconfirmed" if "unconfirmed" in stops else "stopped" if "stopped" in stops else None

    def snapshot(self, publication_pending: bool = False) -> Obligation:
        """Return host facts without promoting a reservation into an applied input or hearing."""
        return Obligation(
            self.ref,
            self.branch,
            self.state,
            bool(self.pending),
            publication_pending and self.state not in ("transferred", "settled"),
            self.outcome,
            self.stop,
            self.to_owner,
        )


@dataclass
class Branch:
    """A Work's stable output identity and current binding, not its source-input identities."""

    ref: str
    work: WorkRun
    owner: str
    revision: int = 0
    composed: bool = False


@dataclass
class Owner:
    """The one lifecycle owner of a real Delegation's current response frame."""

    ref: str
    order: int
    edges: dict[str, Edge] = field(default_factory=dict)
    revision: int = 0
    composed: bool = False
    disposition: str | None = None
    settlements: dict[str, OwnerSettlement] = field(default_factory=dict)
    settlement_bindings: dict[str, tuple[Binding, ...]] = field(default_factory=dict)
    flushing: bool = False
    controls: set[str] = field(default_factory=set)


class ResponseOwnership:
    """Track owned output obligations without duplicating callers, writers, actions or native runtimes."""

    def __init__(self, session: IMSession):
        """Use the session's existing response book, limits and lifecycle."""
        self.session = session
        self.owners: dict[str, Owner] = {}
        self.branches: dict[str, Branch] = {}
        self.edges: dict[str, Edge] = {}
        self.expired: dict[str, Parked] = {}
        self.expired_consumed: set[Parked] = set()
        self._placement_waits: dict[str, asyncio.Task] = {}
        self._owner_order = 0

    def owner(self, source: str) -> Owner:
        """Return the bookkeeping owner of one real source."""
        if source not in self.owners:
            self._owner_order += 1
            self.owners[source] = Owner(source, self._owner_order)
        return self.owners[source]

    def register(self, work: WorkRun) -> str:
        """Give a genuinely new Work its own output branch; retain unfinished input-edge identity."""
        ref = uid("branch")
        branch = self.branches[ref] = Branch(ref, work, work.carrier)
        task_ids = tuple(task.task_id for task in work.tasks)
        for edge in self.edges.values():
            if edge.continuation_of is None or edge.continuation_of != work.record.continues:
                continue
            branch.owner = edge.source
            branch.composed = True
            work.record.carrier = edge.source
            edge.branch, edge.continuation_of = ref, None
            if edge.task_scope is not None:
                edge.task_scope = frozenset(task_ids)
            self.owner(edge.source).revision += 1
        for source in work.requests:
            previous = next(
                (
                    edge
                    for edge in self.edges.values()
                    if edge.source == source
                    and edge.pending
                    and edge.state != "settled"
                    and (old := self.branches.get(edge.branch)) is not None
                    and (
                        old.work.ended
                        or any(task.task_id == work.record.continues and task.ended for task in old.work.tasks)
                    )
                    and work.record.continues in (old.work.id, *(task.task_id for task in old.work.tasks))
                ),
                None,
            )
            if previous is None:
                self._edge(source, ref, task_ids)
            else:
                old_branch = self.branches[previous.branch]
                if self.owner(old_branch.owner).order > self.owner(branch.owner).order:
                    branch.owner = old_branch.owner
                    work.record.carrier = old_branch.owner
                    branch.composed = True
                    self.owner(branch.owner).composed = True
                predecessor = work.record.continues
                if predecessor in previous.pending and len(previous.pending) > 1:
                    previous.pending.remove(predecessor)
                    if previous.task_scope is not None:
                        previous.task_scope = previous.task_scope - {predecessor}
                    self._edge(source, ref, task_ids)
                    self.owner(source).revision += 1
                    continue
                previous.branch, previous.pending = ref, set(task_ids)
                previous.task_scope = frozenset(task_ids)
                if previous.state == "transferred":
                    previous.to_owner, previous.transferred_revision = branch.owner, branch.revision
                else:
                    previous.state = "pending"
                branch.composed = self.owner(source).composed
                self.owner(source).revision += 1
        if branch.owner not in work.requests and not any(
            edge.branch == ref for edge in self.owner(branch.owner).edges.values()
        ):
            self._edge(branch.owner, ref, ()).state = "active"
        return ref

    def _edge(self, source: str, branch: str, tasks: Sequence[str]) -> Edge:
        edge = Edge(uid("placement"), source, branch, set(tasks))
        self.edges[edge.ref] = self.owner(source).edges[edge.ref] = edge
        self.owner(source).revision += 1
        return edge

    def reserve(
        self,
        sources: Sequence[str],
        targets: Sequence[tuple[WorkRun, Sequence[TaskRun]]],
        prepared: dict[str, tuple[Edge, ...]],
    ) -> dict[str, tuple[str, ...]]:
        """Reserve every atomic target after validation, before any selected input can land."""
        refs: dict[str, list[str]] = {}
        composed = (
            len(targets) > 1
            or any(self.owner(source).composed for source in sources)
            or any(self.managed(work) for work, _ in targets)
        )
        for source in sources:
            self.owner(source).composed = self.owner(source).composed or composed
            for work, _tasks in targets:
                branch = self.branches[work.branch_ref]
                branch.composed = branch.composed or composed
                self.owner(branch.owner).composed = self.owner(branch.owner).composed or composed
                writer = self.session.book.writer(source)
                if writer is not None and composed:
                    writer.usage = None
                edge = next(edge for edge in prepared[work.id] if edge.source == source)
                self.edges[edge.ref] = self.owner(source).edges[edge.ref] = edge
                self.owner(source).revision += 1
                refs.setdefault(work.id, []).append(edge.ref)
        return {work: tuple(edges) for work, edges in refs.items()}

    def validate_budget(
        self,
        sources: Sequence[str],
        targets: Sequence[tuple[WorkRun, Sequence[TaskRun]]],
        *,
        extra: Sequence[dict] = (),
        projected: Mapping[str, Provenance] | None = None,
    ) -> dict[str, tuple[Edge, ...]]:
        """Charge complete target/provenance identities through the existing context budget before mutation."""
        prepared = {
            work.id: tuple(
                Edge(uid("placement"), source, work.branch_ref, {task.task_id for task in tasks}) for source in sources
            )
            for work, tasks in targets
        }
        owners = {
            source: projected[source]
            if projected is not None and source in projected
            else self.provenance(source, self.session.book.response_id(source) or source, None)
            for source in dict.fromkeys([*sources, *(work.carrier for work, _ in targets)])
        }
        source_provenance = []
        final_owner = sources[-1]
        settlements = tuple(
            OwnerSettlement(
                source,
                self.owner(source).revision + 1,
                "handed_over",
                tuple(
                    Obligation(edge.ref, edge.branch, "transferred", False, False, edge.outcome, edge.stop, final_owner)
                    for edge in self.owner(source).edges.values()
                ),
            )
            for source in dict.fromkeys(work.carrier for work, _ in targets)
        )
        for source in sources:
            existing = owners[source]
            planned = tuple(edge.snapshot() for edges in prepared.values() for edge in edges if edge.source == source)
            bindings = tuple(
                Binding(work.branch_ref, source, self.branches[work.branch_ref].revision + 1) for work, _ in targets
            )
            source_provenance.append(
                encode_provenance(
                    Provenance(
                        existing.conversation_ref,
                        existing.response_id,
                        source,
                        existing.owner_revision + len(planned),
                        bindings,
                        (*existing.obligations, *planned),
                        owner_settlements=settlements,
                        request_sources=self._request_sources(
                            tuple(binding.branch_ref for binding in bindings), sources
                        ),
                    )
                )
            )
        required = {
            "sources": [{"id": source, "input": self.session.request_items(source)} for source in sources],
            "targets": [
                {"work": work.id, "branch": work.branch_ref, "tasks": [task.task_id for task in tasks]}
                for work, tasks in targets
            ],
            "owners": [encode_provenance(owner) for owner in owners.values()],
            "complete_planned_owner_metadata": source_provenance,
            "planned_edges": [
                {"ref": edge.ref, "source": edge.source, "branch": edge.branch, "pending": sorted(edge.pending)}
                for edges in prepared.values()
                for edge in edges
            ],
        }
        if extra:
            required["retained_clarification_input"] = extra
        if approx_tokens(json.dumps(required, ensure_ascii=False)) > self.session.settings.context_tokens:
            raise DecisionRejected("Backend context budget exceeded for atomic work placement")
        return prepared

    def project_allocations(self, allocations: Sequence[tuple[str, ...]]) -> dict[str, Provenance]:
        """Account for the new branch records of sibling allocations before they can start."""
        projected = {}
        for sources in allocations:
            branch = uid("branch")
            final_owner = sources[-1]
            for source in sources:
                owner = projected.get(source)
                if owner is None:
                    owner = self.provenance(source, self.session.book.response_id(source) or source, None)
                edge = Edge(uid("placement"), source, branch, set())
                if source != final_owner:
                    edge.state, edge.to_owner = "transferred", final_owner
                allocation = replace(edge.snapshot(), delivery_pending=True)
                request_sources = (
                    (*owner.request_sources, BranchSources(branch, sources))
                    if source == final_owner
                    else owner.request_sources
                )
                projected[source] = replace(
                    owner,
                    owner_revision=owner.owner_revision + 1,
                    bindings=(*owner.bindings, Binding(branch, final_owner, len(sources) - 1)),
                    obligations=(*owner.obligations, allocation),
                    request_sources=request_sources,
                )
        return projected

    def release_unqueued(self, edges: Sequence[Edge]) -> None:
        """Release the exact reservations whose staged input was never queued."""
        for edge in edges:
            if self.edges.pop(edge.ref, None) is not None:
                owner = self.owner(edge.source)
                del owner.edges[edge.ref]
                owner.revision += 1

    def initial_applied(self, task: TaskRun) -> None:
        """Record actual initial application independently for each task."""
        self.session.executor.delivered(task.task_id, tuple(task.request_ids), "applied")
        for edge in self.edges.values():
            if edge.branch == task.work.branch_ref and task.task_id in edge.pending:
                edge.pending.remove(task.task_id)
                if edge.state != "transferred":
                    edge.state = "active"
                self.owner(edge.source).revision += 1
                self.finish_owner(self.owner(edge.source))

    def delivered(self, task: TaskRun, batch: Batch) -> None:
        """Settle only this batch's actual task receipt, not another branch's reservation."""
        self.session.executor.delivered(task.task_id, tuple(batch.delegations), "applied")
        for ref in batch.placement_refs:
            edge = self.edges[ref]
            edge.pending.discard(task.task_id)
            if edge.state != "transferred":
                edge.state = "active"
            self.owner(edge.source).revision += 1
            self.finish_owner(self.owner(edge.source))

    def delivery_unknown(self, task: TaskRun, batch: Batch) -> None:
        """An uncertain handoff is retained as evidence and never automatically resent."""
        for ref in batch.placement_refs:
            edge = self.edges[ref]
            edge.pending.discard(task.task_id)
            edge.outcome = "failed"
            if edge.state == "pending" and not edge.pending:
                edge.state = "settled"
            self.owner(edge.source).revision += 1
        self.session.executor.delivered(task.task_id, tuple(batch.delegations), "unknown")

    def pending_sources(self) -> set[str]:
        """Admission budget only: these sources are never put back behind the GLOBAL placement gate."""
        return {
            edge.source
            for edge in self.edges.values()
            if edge.pending and edge.state != "settled" and self.owner(edge.source).composed
        }

    def publication_changed(self, work: WorkRun) -> None:
        """Version an owner's held-publication facts before another event can expose them."""
        branch = self.branches[work.branch_ref]
        owner = self.owner(branch.owner)
        if work.held:
            branch.composed = owner.composed = True
        owner.revision += 1

    def input_joined(self, work: WorkRun) -> None:
        """Version the current writer's source list even when an older input cannot take over."""
        self.owner(self.branches[work.branch_ref].owner).revision += 1

    def older_input(self, work: WorkRun, source: str) -> bool:
        """Applying an older queued input never takes output back from a newer admitted source."""
        return self.owner(source).order < self.owner(work.carrier).order

    def managed(self, work: WorkRun, taker: str | None = None) -> bool:
        """Whether a Work participates in scoped ownership, permanently after its first composition."""
        branch = self.branches[work.branch_ref]
        return branch.composed or self.owner(branch.owner).composed or bool(taker and self.owner(taker).composed)

    def attribution(self, work: WorkRun, *, task_ref: str | None = None) -> Attribution:
        """Capture a publication binding; later ownership moves never alter this object."""
        branch = self.branches[work.branch_ref]
        focused = any(task.record.history_mode == "focused" for task in work.tasks)
        return Attribution(branch.ref, branch.owner, branch.revision, focused=focused, task_ref=task_ref)

    def hand_over(self, work: WorkRun, taker: str) -> bool:
        """Move one Work's future output; only return true when the scoped path owns the lifecycle."""
        branch, old = self.branches[work.branch_ref], work.carrier
        managed = self.managed(work, taker)
        if old == taker:
            return managed
        previous, current = self.owner(old), self.owner(taker)
        if not any(edge.branch == branch.ref for edge in current.edges.values()):
            edge = self._edge(
                taker, branch.ref, [task.task_id for task in work.tasks] if work.started_at is None else ()
            )
            if not edge.pending:
                edge.state = "active"
        transferred_owners = set()
        for owner in self.owners.values():
            changed = False
            for edge in owner.edges.values():
                if edge.branch != branch.ref or edge.state in ("settled", "transferred"):
                    continue
                if owner is previous or owner.order < current.order:
                    edge.state, edge.to_owner = "transferred", taker
                    edge.transferred_revision = branch.revision + 1
                    transferred_owners.add(owner.ref)
                changed = True
            if changed and owner is not previous and owner is not current:
                owner.revision += 1
        previous.revision += 1
        current.revision += 1
        branch.owner, branch.revision = taker, branch.revision + 1
        if not managed:
            return False
        previous.composed = current.composed = branch.composed = True
        self.session.notes.drop(old, branch_ref=branch.ref)
        work.record.carrier = taker
        self.session.book.start(taker)
        record = self.session.ledger.delegations[taker]
        if record.state == "queued":
            record.move("running")
        for source in transferred_owners | {old}:
            owner = self.owner(source)
            self.finish_owner(owner)
            if owner.disposition is not None:
                self._remember_settlement(current, self._settlement(owner))
        self.session.observe("backend.branch_handed_over", delegation_id=old, taker=taker, work_id=work.id)
        return True

    def provenance(self, source: str, response: str, attribution: Attribution | None) -> Provenance:
        """Produce private facts on an event envelope, never model-visible item fields."""
        owner = self.owner(source)
        refs = {edge.branch for edge in owner.edges.values()}
        if attribution is not None:
            refs.add(attribution.branch_ref)
        bindings = {}
        for values in owner.settlement_bindings.values():
            for binding in values:
                previous = bindings.get(binding.branch_ref)
                if previous is None or binding.revision > previous.revision:
                    bindings[binding.branch_ref] = binding
        for ref in sorted(refs):
            if (binding := self._binding(owner, ref)) is None:
                continue
            previous = bindings.get(ref)
            if previous is None or binding.revision >= previous.revision:
                bindings[ref] = binding
        return Provenance(
            self.session.conversation_ref,
            response,
            source,
            owner.revision,
            tuple(bindings[ref] for ref in sorted(bindings)),
            tuple(self._snapshot(edge) for edge in owner.edges.values()),
            attribution,
            tuple(owner.settlements.values()),
            request_sources=self._request_sources(tuple(sorted(bindings)), owner_ref=source),
        )

    def _request_sources(
        self, refs: Sequence[str], adding: Sequence[str] = (), *, owner_ref: str | None = None
    ) -> tuple[BranchSources, ...]:
        """Read original input identity from the Work, where publication-only edges never add a request."""
        rows = []
        for ref in refs:
            branch = self.branches.get(ref)
            if branch is not None and (owner_ref is None or branch.owner == owner_ref):
                sources = tuple(dict.fromkeys((*branch.work.requests, *adding)))
                if sources:
                    rows.append(BranchSources(ref, sources))
        return tuple(rows)

    def _binding(self, owner: Owner, ref: str) -> Binding | None:
        branch = self.branches.get(ref)
        if branch is None:
            return None
        transferred = next(
            (edge for edge in owner.edges.values() if edge.branch == ref and edge.state == "transferred"), None
        )
        return (
            Binding(ref, transferred.to_owner, transferred.transferred_revision)
            if transferred is not None
            else Binding(ref, branch.owner, branch.revision)
        )

    def _snapshot(self, edge: Edge) -> Obligation:
        branch = self.branches.get(edge.branch)
        held = bool(branch and branch.owner == edge.source and branch.work.held)
        return edge.snapshot(held)

    def handover_message_source(
        self, source: str, taker: str, carrier: str, control_id: str, previous: str | None
    ) -> bool:
        """Hand over a message-only source that has no separate branch or control obligation."""
        owner = self.owner(source)
        if owner.edges or owner.controls != {control_id}:
            return False
        session = self.session
        session.book.start(source, previous=previous)
        writer = session.book.running(source)
        if writer is not None:
            writer.message(session.texts.handover_line.format(delegation=taker), "commentary", source_item=True)
        owner.disposition = "handed_over"
        owner.revision += 1
        record = session.ledger.delegations[source]
        if not record.ended:
            record.move("handed_over")
        session.book.end(source, "response.completed")
        self._remember_settlement(self.owner(carrier), self._settlement(owner))
        return True

    def _settlement(self, owner: Owner) -> OwnerSettlement:
        return OwnerSettlement(
            owner.ref, owner.revision, owner.disposition, tuple(self._snapshot(edge) for edge in owner.edges.values())
        )

    def _remember_settlement(self, owner: Owner, settlement: OwnerSettlement) -> None:
        if owner.settlements.get(settlement.owner_ref) != settlement:
            owner.settlements[settlement.owner_ref] = settlement
            source = self.owner(settlement.owner_ref)
            owner.settlement_bindings[settlement.owner_ref] = tuple(
                binding
                for ref in sorted({row.branch_ref for row in settlement.obligations if row.branch_ref is not None})
                if (binding := self._binding(source, ref)) is not None
            )
            owner.revision += 1

    def work_finished(self, work: WorkRun, state: str) -> bool:
        """Retain sibling and held-output obligations when one Work ends."""
        branch = self.branches[work.branch_ref]
        continuing = (
            not self.session.closed
            and not self.session.detached
            and not (work.stop_reason is not None and work.stop_reason.discard_pending_requests)
            and any(
                edge.branch == branch.ref
                and edge.pending
                and edge.state == "transferred"
                and edge.source not in work.requests
                for edge in self.edges.values()
            )
        )
        for edge in self.edges.values():
            if edge.branch != branch.ref or edge.state == "settled":
                continue
            if edge.pending and edge.source not in work.requests:
                continue  # An unlanded canonical request still needs its existing follow-up.
            if edge.state == "transferred":
                if edge.pending:
                    edge.pending.clear()
                    self.owner(edge.source).revision += 1
                continue
            edge.pending.clear()
            edge.outcome = "failed" if "failed" in (state, edge.outcome) else state
            edge.record_stops(
                task.stopped for task in work.tasks if edge.task_scope is None or task.task_id in edge.task_scope
            )
            if not work.held:
                edge.state = "settled"
            self.owner(edge.source).revision += 1
        if continuing:
            for edge in self.owner(branch.owner).edges.values():
                if edge.branch == branch.ref and edge.state != "transferred":
                    edge.state, edge.continuation_of = "active", work.id
                    self.owner(edge.source).revision += 1
        if not self.managed(work):
            return False
        self.finish_owner(self.owner(branch.owner))
        return True

    def discard_pending(self, work: WorkRun, sources: Sequence[str], *, tasks: Sequence[str] | None = None) -> None:
        """An explicit stop retires only this Work's unapplied edges, not its source's sibling branch."""
        owners = set()
        selected = set(tasks) if tasks is not None else None
        for edge in self.edges.values():
            if edge.branch != work.branch_ref or edge.source not in sources or not edge.pending:
                continue
            removed = set(edge.pending) if selected is None else edge.pending & selected
            if not removed:
                continue
            edge.pending.difference_update(removed)
            if edge.state != "transferred" and not edge.pending:
                edge.state, edge.outcome = "settled", "cancelled"
                edge.continuation_of = None
            edge.record_stops(task.stopped for task in work.tasks if task.task_id in removed)
            self.owner(edge.source).revision += 1
            owners.add(edge.source)
        for source in owners:
            self.finish_owner(self.owner(source))

    def finish_owner(self, owner: Owner) -> None:
        """Write one logical terminal only after actual branch obligations and held publications settle."""
        if owner.disposition is not None or not (owner.edges or owner.controls):
            return
        controls = [self.session.ledger.controls[key] for key in owner.controls]
        if any(control.state == "pending" for control in controls):
            return
        if any(edge.pending or edge.state not in ("settled", "transferred") for edge in owner.edges.values()):
            return
        book = self.session.book
        outcomes = [edge.outcome for edge in owner.edges.values() if edge.state == "settled"]
        failures = [control for control in controls if control.state in ("refused", "failed", "unknown")]
        capacity_refused = sum(
            control.state == "refused" and control.receipt.get("outcome") == WorkerCapacityUnavailable.outcome
            for control in failures
        )
        target_refused = sum(
            control.state == "refused" and control.receipt.get("outcome") == RoutingTargetUnavailable.outcome
            for control in failures
        )
        routing_refused = capacity_refused + target_refused
        disposition = (
            "failed"
            if "failed" in outcomes or len(failures) > routing_refused
            else "handed_over"
            if not routing_refused and owner.edges and all(edge.state == "transferred" for edge in owner.edges.values())
            else "answered"
            if routing_refused or "answered" in outcomes or (controls and not owner.edges)
            else "cancelled"
        )
        record = self.session.ledger.delegations.get(owner.ref)
        if record is not None and record.ended:
            disposition = record.state
        owner.disposition = disposition
        owner.revision += 1
        settlement = self._settlement(owner)
        for source in {
            branch.owner
            for edge in owner.edges.values()
            if edge.state == "transferred" and (branch := self.branches.get(edge.branch)) is not None
        } - {owner.ref}:
            self._remember_settlement(self.owner(source), settlement)
        if record is not None and not record.ended:
            record.move(disposition)
        writer = book.writer(owner.ref)
        if writer is None:
            return  # An app frame already ended; never manufacture a second response stream.
        book.start(owner.ref)
        if disposition == "cancelled":
            book.cancelled(owner.ref)
        elif disposition == "failed":
            failure = next(
                (
                    task.failure
                    for edge in owner.edges.values()
                    if edge.state == "settled" and (branch := self.branches.get(edge.branch)) is not None
                    for task in branch.work.tasks
                    if task.failure is not None
                ),
                None,
            )
            if failure is not None:
                book.end_after_calls(owner.ref, failure.kind, fields=failure.fields)
            else:
                book.end_after_calls(
                    owner.ref, "response.failed", fields={"error": {"code": "server_error", "message": "Work failed."}}
                )
        else:
            if routing_refused:
                texts = self.session.texts
                self.session.routing_notice(
                    owner.ref,
                    texts.capacity_unavailable_output if capacity_refused else texts.routing_failed_output,
                    remember_input=not owner.edges and len(controls) == routing_refused,
                )
            book.end_after_calls(owner.ref, "response.completed")

    def members(self, source: str) -> list[WorkRun]:
        """Include ended members whose genuine output is still held, not only scheduler-live Works."""
        return [branch.work for branch in self.branches.values() if branch.owner == source]

    def has_unsettled_sibling(self, source: str, work_id: str) -> bool:
        """Include selected siblings whose input has yet to land."""
        return any(
            edge.state not in ("settled", "transferred") and self.branches[edge.branch].work.id != work_id
            for edge in self.owner(source).edges.values()
        )

    def flush(self, source: str) -> None:
        """Drain every current member until another app batch closes the owner's frame."""
        owner = self.owner(source)
        if owner.flushing:
            return
        owner.flushing = True
        try:
            for work in self.members(source):
                work._flush()
                if self.session.book.running(source) is None:
                    return
                if work.ended and not work.held:
                    for edge in owner.edges.values():
                        if edge.branch == work.branch_ref and edge.outcome is not None and edge.state != "transferred":
                            changed = edge.state != "settled" or bool(edge.pending)
                            edge.state = "settled"
                            edge.pending.clear()
                            if changed:
                                owner.revision += 1
            if owner.composed:
                self.finish_owner(owner)
            self.session.notes.resume(source)
        finally:
            owner.flushing = False

    def wait_to_flush(self, source: str) -> None:
        """Retain a GLOBAL-only commit wait without blocking the response reader or adding a timer."""
        if source in self._placement_waits:
            return

        async def waiting() -> None:
            try:
                await self.session.dispatch_placement()
            finally:
                self._placement_waits.pop(source, None)
            if not self.session.closed:
                self.flush(source)

        self._placement_waits[source] = self.session.spawn(waiting())

    def protected_works(self) -> set[str]:
        """Keep only actual outstanding owners reachable through normal ledger pruning."""
        calls = self.session.book.pending_calls()
        protected = {action.work for call_id, action in self.session.ledger.actions.items() if call_id in calls}
        protected.update(
            branch.work.id
            for owner in self.owners.values()
            if owner.composed and owner.disposition is None
            for values in owner.settlement_bindings.values()
            for binding in values
            if (branch := self.branches.get(binding.branch_ref)) is not None
        )
        return protected | {
            branch.work.id
            for branch in self.branches.values()
            if branch.work.held
            or any(
                edge.branch == branch.ref
                and (
                    edge.pending
                    or edge.state not in ("settled", "transferred")
                    or self.owner(edge.source).composed
                    and self.owner(edge.source).disposition is None
                )
                for edge in self.edges.values()
            )
        }

    def expire(self, parked: Parked) -> None:
        """Retain a non-runnable app-frame tombstone while a composite owner still owes output."""
        if self.owner(parked.delegation_id).composed:
            for call_id in parked.calls:
                self.expired[call_id] = parked

    def owed_output(self, source: str) -> bool:
        """Whether publication or independent computation remains, not whether an expired task may resume."""
        return any(work.held or not work.ended for work in self.members(source)) or any(
            self.session.ledger.controls[key].state == "pending" for key in self.owner(source).controls
        )

    def listening(self, source: str) -> bool:
        """A completed app frame does not end a still-attached composite source's ownership."""
        owner, record = self.owner(source), self.session.ledger.delegations.get(source)
        return bool(
            owner.composed
            and owner.disposition is None
            and record is not None
            and not record.ended
            and self.session.caller_attached
            and not self.session.detached
        )

    def end_call(self, kept: set[str]) -> None:
        """Discard unheard publication for stopped call-owned branches, never a kept branch's computation."""
        for branch in self.branches.values():
            if branch.work.id in kept:
                continue
            branch.work.drop_held("call_ended")
            for edge in self.edges.values():
                if edge.branch != branch.ref or edge.state == "settled":
                    continue
                if edge.state == "transferred":
                    if edge.pending:
                        edge.pending.clear()
                        self.owner(edge.source).revision += 1
                else:
                    edge.pending.clear()
                    edge.state = "settled"
                    edge.continuation_of = None
                    edge.outcome = branch.work.record.state if branch.work.record.ended else "cancelled"
                    self.owner(edge.source).revision += 1
        # The call has already closed its streams. Retire only logical owners whose genuine obligations settled;
        # kept branches remain active, and no replacement response is created for discarded speech.
        for owner in tuple(self.owners.values()):
            if owner.composed:
                self.finish_owner(owner)

    def prune(self) -> None:
        """Release closed owner graphs under the ledger's existing reachability and retention policy."""
        needed = self.protected_works()
        for ref, branch in tuple(self.branches.items()):
            if branch.work.id not in self.session.ledger.works and branch.work.id not in needed:
                del self.branches[ref]
        for ref, edge in tuple(self.edges.items()):
            if edge.source not in self.session.ledger.delegations or edge.branch not in self.branches:
                self.owner(edge.source).edges.pop(ref, None)
                self.owner(edge.source).revision += 1
                del self.edges[ref]
        for source, owner in tuple(self.owners.items()):
            if owner.disposition is not None:
                retained = {
                    ref: settlement
                    for ref, settlement in owner.settlements.items()
                    if ref in self.session.ledger.delegations
                }
                if retained != owner.settlements:
                    owner.settlements = retained
                    owner.settlement_bindings = {
                        ref: values for ref, values in owner.settlement_bindings.items() if ref in retained
                    }
                    owner.revision += 1
            if source not in self.session.ledger.delegations and not owner.edges:
                del self.owners[source]
        for call_id, frame in tuple(self.expired.items()):
            if frame.work.id not in self.session.ledger.works and not self.owed_output(frame.delegation_id):
                del self.expired[call_id]
        self.expired_consumed.intersection_update(self.expired.values())
