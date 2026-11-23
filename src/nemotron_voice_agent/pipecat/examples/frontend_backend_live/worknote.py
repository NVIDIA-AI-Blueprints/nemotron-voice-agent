# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The work note: the frontend's record of its open delegations, and the line the talker reads about them."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import TYPE_CHECKING

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.prompts import PromptSet
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.work_provenance import Attribution

if TYPE_CHECKING:
    from nemotron_voice_agent.pipecat.examples.frontend_backend_live.ownership import WorkOwnership

REQUEST_CHARS = 140


class WorkState(StrEnum):
    """Where an open delegation stands."""

    WAITING = "waiting"
    RUNNING = "running"


@dataclass
class ToolCall:
    """One tool call of a delegation's latest response that had some."""

    name: str
    status: str | None = None


@dataclass
class OpenWork:
    """One open delegation, as the note knows it."""

    id: str
    request: str
    created_at: float
    state: WorkState = WorkState.WAITING
    started_at: float | None = None
    calls: dict[str, ToolCall] = field(default_factory=dict)
    calls_response: str | None = None
    parts: tuple[tuple[int, str], ...] = ()
    carries: tuple[str, ...] = ()
    progress: str | None = None
    news: bool = False
    revision: int = 0
    progress_offers: int = 0
    ready: int = 0
    ended: bool = False
    waits_for_app: bool = False
    reserved: bool = False
    retired: bool = False
    placeholders: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ProgressKey:
    """A real transport delegation and one opaque Work branch, never a synthetic delegation ID."""

    delegation_id: str
    branch_ref: str


@dataclass(kw_only=True)
class BranchWork(OpenWork):
    """One Work's progress state, whose inherited ID is always its actual current transport delegation."""

    branch_ref: str
    owner_revision: int = 0
    request_known: bool = False
    predecessor_refs: frozenset[str] = field(default_factory=frozenset)

    @property
    def progress_key(self) -> ProgressKey:
        """The scoped selector for this branch's current progress."""
        return ProgressKey(self.id, self.branch_ref)


@dataclass(frozen=True)
class ProgressSource:
    """The identity and evidence revision offered to one progress wording call."""

    entry: OpenWork
    revision: int
    delegation_id: str = field(kw_only=True)
    branch_ref: str | None = field(default=None, kw_only=True)
    owner_revision: int = field(default=0, kw_only=True)

    @property
    def id(self) -> str:
        """The delegation whose evidence was offered."""
        return self.delegation_id


def humanize(tool_name: str) -> str:
    """Return a tool name as words: underscores and hyphens become spaces."""
    return re.sub(r"[_\-]+", " ", tool_name).strip()


def shorten(request: str) -> str:
    """Return a request as the note quotes it: at most ``REQUEST_CHARS`` characters."""
    return request if len(request) <= REQUEST_CHARS else request[:REQUEST_CHARS] + "..."


class WorkNote:
    """The frontend's one record of its open work, and the line the talker reads from it."""

    def __init__(self, prompts: PromptSet, *, tracks: bool, clock: Callable[[], float], observe: Callable[..., None]):
        """Start empty; only a part that reports events (``tracks``) gets entries, since nothing else ends one."""
        self.prompts, self.tracks, self.clock, self.observe = prompts, tracks, clock, observe
        self.entries: dict[str, OpenWork] = {}
        self.branches: dict[str, BranchWork] = {}
        self._confirmed_requests: dict[str, tuple[int, str]] = {}
        self._scoped: set[str] = set()
        self._obligations: dict[str, BranchWork] = {}
        self._ready_items: dict[tuple[str, str], tuple[OpenWork, BranchWork | None]] = {}
        self._settled_items: set[tuple[str, str]] = set()
        self._made = 0
        self.on_answer: Callable[[], None] = lambda: None
        self.on_settled: Callable[[], None] = lambda: None
        self.on_confirm: Callable[[str], None] = lambda delegation_id: None

    def sync_ownership(self, ownership: WorkOwnership) -> None:
        """Apply host branch bindings without moving already published ready items between delegations."""
        for branch_ref, binding in ownership.bindings.items():
            entry = self.branches.get(branch_ref)
            if entry is not None and entry.owner_revision != binding.revision:
                entry.owner_revision = binding.revision
                entry.progress, entry.news, entry.calls, entry.calls_response = None, False, {}, None
                entry.retired = False
                entry.revision += 1
            delegation_id = ownership.current_transport(branch_ref)
            if delegation_id is None:
                continue
            parent = self.entries.get(delegation_id)
            if parent is None:
                continue
            self._scoped.add(delegation_id)
            if entry is None:
                values = {item.name: getattr(parent, item.name) for item in fields(OpenWork)}
                values.update(
                    calls=dict(parent.calls),
                    placeholders=dict(parent.placeholders),
                    ready=0,
                    ended=False,
                    retired=False,
                )
                entry = BranchWork(
                    **values,
                    branch_ref=branch_ref,
                    owner_revision=binding.revision,
                    request_known=branch_ref not in ownership.request_sources
                    and not ownership.is_composite(delegation_id),
                )
                self.branches[branch_ref] = entry
            elif entry.id != delegation_id or entry.owner_revision != binding.revision:
                entry.created_at = min(entry.created_at, parent.created_at)
                starts = [start for start in (entry.started_at, parent.started_at) if start is not None]
                entry.started_at = min(starts) if starts else None
                entry.id, entry.owner_revision = delegation_id, binding.revision
                entry.progress, entry.news, entry.calls, entry.calls_response = None, False, {}, None
                entry.retired = False
                entry.revision += 1
        retained = set()
        for owner in ownership.owners.values():
            for obligation in owner.obligations:
                retained.add(obligation.obligation_ref)
                entry = self.branches.get(obligation.branch_ref)
                if entry is None:
                    continue
                previous = self._obligations.get(obligation.obligation_ref)
                if previous is not None and previous is not entry:
                    if entry.branch_ref in previous.predecessor_refs:
                        continue
                    if obligation.state == "transferred":
                        self._inherit_progress(
                            previous, entry, inherit_request=previous.branch_ref not in ownership.request_sources
                        )
                        continue
                    self._inherit_progress(
                        entry, previous, inherit_request=entry.branch_ref not in ownership.request_sources
                    )
                self._obligations[obligation.obligation_ref] = entry
        self._obligations = {key: value for key, value in self._obligations.items() if key in retained}
        self._sync_request_sources(ownership)
        for branch_ref, entry in tuple(self.branches.items()):
            if branch_ref not in ownership.bindings and entry.ready == 0:
                self.branches.pop(branch_ref)
        for entry in self.branches.values():
            entry.predecessor_refs = entry.predecessor_refs.intersection(self.branches)
        transports = {ownership.transport(owner) for owner in ownership.owners} | set(self.entries)
        private_transports = set(self._scoped)
        retained_items = {
            (ownership.frames[response_id][1], item_id)
            for response_id, item_id in ownership.items
            if response_id in ownership.frames
        }
        self._scoped.intersection_update(transports)
        self._settled_items = {
            key
            for key in self._settled_items
            if key in retained_items or (key[0] in transports and key[0] not in private_transports)
        }

    def _sync_request_sources(self, ownership: WorkOwnership) -> None:
        """Join only confirmed frontend wording for sources Core actually assigned to the Work."""
        retained = self.entries.keys() | ownership.transports.values()
        for branch_ref, sources in ownership.request_sources.items():
            entry = self.branches.get(branch_ref)
            if entry is None:
                continue
            retained.update(sources)
            parts = tuple(
                sorted(self._confirmed_requests[source] for source in sources if source in self._confirmed_requests)
            )
            if entry.request_known != bool(parts) or parts != entry.parts:
                entry.parts, entry.request_known = parts, bool(parts)
                entry.request = " / ".join(words for _, words in parts)
                entry.revision += 1
        self._confirmed_requests = {
            source: part for source, part in self._confirmed_requests.items() if source in retained
        }

    @staticmethod
    def _inherit_progress(entry: BranchWork, previous: BranchWork, *, inherit_request: bool) -> None:
        """Carry one exact obligation's conservative budget and reachable predecessor references."""
        entry.progress_offers = max(entry.progress_offers, previous.progress_offers)
        entry.created_at = min(entry.created_at, previous.created_at)
        starts = [start for start in (entry.started_at, previous.started_at) if start is not None]
        started_at = min(starts, default=None)
        if entry.started_at != started_at:
            entry.started_at = started_at
            entry.revision += 1
        if inherit_request and previous.request_known:
            entry.request, entry.parts, entry.request_known = previous.request, previous.parts, True
        entry.predecessor_refs |= previous.predecessor_refs | frozenset((previous.branch_ref,))

    def _branch(self, delegation_id: str, attribution: Attribution | None = None) -> BranchWork | None:
        if attribution is not None:
            return self.branches.get(attribution.branch_ref)
        matching = [entry for entry in self.branches.values() if entry.id == delegation_id]
        return matching[0] if len(matching) == 1 else None

    def _fact_entry(self, delegation_id: str, attribution: Attribution | None = None) -> OpenWork | None:
        entry = self._branch(delegation_id, attribution)
        if attribution is not None:
            if entry is None or entry.id != delegation_id or entry.owner_revision != attribution.binding_revision:
                return None
            return entry
        return entry or (self.entries.get(delegation_id) if delegation_id not in self._scoped else None)

    def publication_current(self, delegation_id: str, attribution: Attribution | None = None) -> bool:
        """Whether the publication's exact source/binding is still known; not a relevance judgment."""
        return self._fact_entry(delegation_id, attribution) is not None

    def progress_key(self, entry: OpenWork) -> str | ProgressKey:
        """Select a branch with a typed scope while retaining the legacy string API."""
        return entry.progress_key if isinstance(entry, BranchWork) else entry.id

    def attachment_progress(self) -> list[dict]:
        """Keep spent allowances for open work and retained branches that can still transfer them."""
        entries = {id(entry): (entry, True) for entry in self.open()}
        obligations: dict[str, list[str]] = {}
        for reference, entry in self._obligations.items():
            obligations.setdefault(entry.branch_ref, []).append(reference)
        for entry in self.branches.values():
            if entry.progress_offers:
                entries.setdefault(id(entry), (entry, False))
        rows = []
        for entry, opened in entries.values():
            row = {
                "delegation_id": entry.id,
                "branch_ref": None,
                "progress_offers": entry.progress_offers,
            }
            if isinstance(entry, BranchWork):
                row.update(branch_ref=entry.branch_ref, binding_revision=entry.owner_revision)
                if entry.progress_offers and entry.branch_ref in obligations:
                    row["obligation_refs"] = obligations[entry.branch_ref]
            if opened:
                row.update(progress=entry.progress, news=entry.news)
            rows.append(row)
        return rows

    def restore_attachment_progress(self, rows: list[dict]) -> None:
        """Restore counts through exact replayed ownership while leaving wording with its original binding."""
        received: dict[str, int] = {}
        for row in rows:
            branch, delegation_id = row["branch_ref"], row["delegation_id"]
            spent = row["progress_offers"]
            if branch is None:
                entry = self.entries.get(delegation_id)
                if entry is not None:
                    entry.progress_offers = max(entry.progress_offers, spent)
            else:
                entry = self.branches.get(branch)
                received[branch] = max(received.get(branch, 0), spent)
                for reference in row.get("obligation_refs", ()):
                    target = self._obligations.get(reference)
                    if target is not None:
                        received[target.branch_ref] = max(received.get(target.branch_ref, 0), spent)
                if entry is not None and row.get("binding_revision", entry.owner_revision) != entry.owner_revision:
                    continue
            if (
                entry is not None
                and entry.id == delegation_id
                and "progress" in row
                and entry.progress == row["progress"]
            ):
                entry.news = entry.news or row["news"]
        for branch, entry in self.branches.items():
            spent = max(received.get(ref, 0) for ref in entry.predecessor_refs | {branch})
            entry.progress_offers = max(entry.progress_offers, spent)

    def restore_request_wording(self, confirmed: dict[str, str], ownership: WorkOwnership) -> None:
        """Restore exact accepted wording after replay establishes its Core scope."""
        for source, wording in confirmed.items():
            if source not in self._confirmed_requests:
                self._made += 1
                self._confirmed_requests[source] = (self._made, wording)
        self._sync_request_sources(ownership)

    def branch_ended(self, branch_ref: str) -> None:
        """End only one branch's progress; original published results retain their ready counts."""
        entry = self.branches.get(branch_ref)
        if entry is not None:
            entry.ended, entry.news = True, False
            entry.revision += 1

    def __contains__(self, delegation_id: str) -> bool:
        """Return whether the delegation is open in the note."""
        return delegation_id in self.entries

    def reserve(self, request: str) -> str | None:
        """Reserve an entry for a request about to be delegated, so no turn decided meanwhile reads "none"."""
        if not self.tracks:
            return None
        self._made += 1
        reservation = f"reserved-{self._made}"
        self.entries[reservation] = OpenWork(
            reservation, request, self.clock(), parts=((self._made, request),), reserved=True
        )
        return reservation

    def confirm(self, reservation: str | None, delegation_id: str) -> None:
        """Give the reserved entry the accepted ``delegation_id``, in its place."""
        entry = self.entries.get(reservation) if reservation else None
        if entry is None:
            return
        self._confirmed_requests[delegation_id] = entry.parts[0]
        existing = self.carrier(delegation_id)
        if existing is not None and existing is not entry:
            # The transport may have already bound results or transferred work before acceptance.
            placeholder = existing.placeholders.get(delegation_id)
            matched = [existing]
            if placeholder is not None:
                matched.extend(
                    branch
                    for branch in self.branches.values()
                    if branch.request_known and any(order == placeholder for order, _ in branch.parts)
                )
            for target in matched:
                target.placeholders.pop(delegation_id, None)
                parts = (part for part in target.parts if part[0] != placeholder)
                target.parts = tuple(sorted(set((*parts, *entry.parts))))
                target.request = " / ".join(words for _, words in target.parts)
                target.created_at = min(target.created_at, entry.created_at)
                target.reserved = False
                target.revision += 1
            self.entries.pop(reservation, None)
            self.on_confirm(delegation_id)
            return
        entry.id, entry.reserved = delegation_id, False
        self.entries = {(delegation_id if key == reservation else key): value for key, value in self.entries.items()}
        self.on_confirm(delegation_id)

    def release(self, reservation: str | None) -> None:
        """Remove the reserved entry of a refused request."""
        if reservation:
            self.entries.pop(reservation, None)

    def typed(self, delegation_id: str) -> None:
        """Add an entry for a delegation the frontend did not start: the application's typed run."""
        if self.tracks and delegation_id not in self.entries:
            self._made += 1
            label = self.prompts.typed_request_label
            self.entries[delegation_id] = OpenWork(
                delegation_id,
                label,
                self.clock(),
                parts=((self._made, label),),
                placeholders={delegation_id: self._made},
            )

    def reopen(self, delegation_id: str) -> None:
        """Start a new trusted logical request on the same transport ID without losing old published results."""
        if not self.tracks:
            return
        self.typed(delegation_id)
        entry = self.entries[delegation_id]
        self._made += 1
        entry.request = self.prompts.typed_request_label
        entry.parts = ((self._made, entry.request),)
        entry.placeholders = {delegation_id: self._made}
        entry.created_at, entry.started_at, entry.state = self.clock(), None, WorkState.WAITING
        entry.progress, entry.news, entry.calls, entry.calls_response = None, False, {}, None
        entry.ended, entry.retired, entry.waits_for_app, entry.progress_offers = False, False, False, 0
        entry.revision += 1
        indexed = 0
        for key, (_, branch) in tuple(self._ready_items.items()):
            if key[0] == delegation_id:
                self._ready_items[key] = (entry, branch)
                indexed += 1
        entry.ready = max(entry.ready, indexed)

    def running(self, delegation_id: str) -> None:
        """Mark work as running and invalidate progress wording based on its waiting state."""
        entries = (
            self.entries.get(delegation_id),
            *(branch for branch in self.branches.values() if branch.id == delegation_id),
        )
        for entry in entries:
            if entry is not None and entry.state != WorkState.RUNNING:
                entry.state = WorkState.RUNNING
                if entry.started_at is None:
                    entry.started_at = self.clock()
                entry.revision += 1

    def call_started(
        self,
        delegation_id: str,
        response_id: str | None,
        item_id: str,
        name: str,
        *,
        attribution: Attribution | None = None,
    ) -> None:
        """Note a tool call of response ``response_id`` that runs now; a new response starts the calls anew."""
        calls = self._calls(delegation_id, response_id, attribution=attribution)
        if calls is not None and calls.get(item_id) != ToolCall(name):
            calls[item_id] = ToolCall(name)
            self._fact_entry(delegation_id, attribution).revision += 1

    def call_ended(
        self,
        delegation_id: str,
        response_id: str | None,
        item_id: str,
        name: str,
        status: str,
        *,
        attribution: Attribution | None = None,
    ) -> None:
        """Note a tool call of response ``response_id`` that ended with the official ``status``."""
        calls = self._calls(delegation_id, response_id, attribution=attribution)
        if calls is not None:
            calls[item_id] = ToolCall(name, status)
            self._fact_entry(delegation_id, attribution).revision += 1

    def _calls(
        self,
        delegation_id: str,
        response_id: str | None,
        *,
        attribution: Attribution | None = None,
    ) -> dict[str, ToolCall] | None:
        """Return the delegation's calls, started anew for a response other than theirs."""
        entry = self._fact_entry(delegation_id, attribution)
        if entry is None:
            return None
        if entry.calls_response != response_id:
            entry.calls, entry.calls_response = {}, response_id
            entry.revision += 1
        return entry.calls

    def progress(self, delegation_id: str, line: str, *, attribution: Attribution | None = None) -> None:
        """Note the delegation's newest progress line."""
        entry = self._fact_entry(delegation_id, attribution)
        if entry is not None and line.strip():
            entry.progress, entry.news = line.strip(), True
            entry.revision += 1

    def restore_progress(self, delegation_id: str, line: str, *, attribution: Attribution | None = None) -> None:
        """Restore retained factual context without new evidence, a budget charge, or replacing fresher news."""
        entry = self._fact_entry(delegation_id, attribution)
        if entry is not None and line.strip() and not entry.news:
            entry.progress = line.strip()

    def waits_for_app(self, delegation_id: str, waiting: bool, *, branch_refs: tuple[str, ...] = ()) -> None:
        """Note whether the delegation waits for the application's outputs; once they came, its calls are done."""
        entries = (
            [self.branches[ref] for ref in branch_refs if ref in self.branches]
            if branch_refs
            else [
                self.entries.get(delegation_id),
                *(entry for entry in self.branches.values() if entry.id == delegation_id),
            ]
        )
        for entry in entries:
            if entry is None:
                continue
            changed = entry.waits_for_app != waiting or (not waiting and bool(entry.calls or entry.calls_response))
            entry.waits_for_app = waiting
            if not waiting:
                entry.calls, entry.calls_response = {}, None
            if changed:
                entry.revision += 1

    def answered(
        self,
        delegation_id: str,
        *,
        attribution: Attribution | None = None,
        item_id: str | None = None,
    ) -> None:
        """Count a result of the delegation that waits to be said."""
        entry = self._entry(delegation_id)
        if item_id is not None and (
            (delegation_id, item_id) in self._ready_items or (delegation_id, item_id) in self._settled_items
        ):
            return
        branch = self._branch(delegation_id, attribution)
        if entry is not None:
            entry.ready += 1
            if branch is None:
                entry.news = False
                entry.revision += 1
            if item_id is not None:
                self._ready_items[delegation_id, item_id] = (entry, branch)
        if branch is not None:
            branch.ready += 1
            if attribution is None or branch.owner_revision == attribution.binding_revision:
                branch.news = False
                branch.revision += 1
        self.on_answer()

    def said(
        self,
        delegation_id: str,
        *,
        attribution: Attribution | None = None,
        item_id: str | None = None,
    ) -> None:
        """Count a result as heard or dropped; an ended delegation leaves once none waits."""
        settled = False
        if item_id is not None:
            counted = self._ready_items.pop((delegation_id, item_id), None)
            if counted is None:
                if (delegation_id, item_id) in self._settled_items or attribution is not None:
                    return
                entry, branch = self._entry(delegation_id), self._branch(delegation_id)
            else:
                entry, branch = counted
                settled = True
            self._settled_items.add((delegation_id, item_id))
        else:
            entry, branch = self._entry(delegation_id), self._branch(delegation_id, attribution)
        if entry is not None:
            settled = settled or entry.ready > 0
            entry.ready = max(0, entry.ready - 1)
            self._leave_if_done(entry)
        if branch is not None:
            settled = settled or branch.ready > 0
            branch.ready = max(0, branch.ready - 1)
        if settled:
            self.on_settled()

    def _entry(self, delegation_id: str) -> OpenWork | None:
        """Return the delegation's entry, or the entry of the delegation that carries its work."""
        entry = self.entries.get(delegation_id)
        if entry is None:
            entry = next((each for each in self.entries.values() if delegation_id in each.carries), None)
        return entry

    def carrier(self, delegation_id: str) -> OpenWork | None:
        """Resolve one exact current carrier; ambiguous ancestry never selects an arbitrary job."""
        matching = [
            entry for entry in self.entries.values() if entry.id == delegation_id or delegation_id in entry.carries
        ]
        return matching[0] if len(matching) == 1 else None

    def ended(self, delegation_id: str) -> None:
        """Mark the delegation ended: it leaves once nothing of it waits to be heard."""
        entry = self.entries.get(delegation_id)
        if entry is not None:
            entry.ended = True
            self._leave_if_done(entry)
        for branch in self.branches.values():
            if branch.id == delegation_id:
                self.branch_ended(branch.branch_ref)

    def retire(self, delegation_id: str) -> None:
        """Retire logical ownership while retaining the parent for its still-draining published frame."""
        entry = self.entries.get(delegation_id)
        retired = [entry, *(branch for branch in self.branches.values() if branch.id == delegation_id)]
        for entry in retired:
            if entry is not None and not entry.retired:
                entry.retired, entry.news = True, False
                entry.revision += 1

    def hand_over(self, delegation_id: str, taker: str) -> str | None:
        """Move the delegation's words, clocks, line count and waiting results to ``taker``, which carries its work."""
        entry, carrier = self.entries.get(delegation_id), self.carrier(taker)
        if carrier is None:
            return None
        if entry is None:
            return carrier.id if self.carrier(delegation_id) is carrier else None
        if entry is carrier or {entry.id, *entry.carries}.intersection((carrier.id, *carrier.carries)):
            return None
        self.entries.pop(delegation_id)
        carrier.parts = tuple(sorted(entry.parts + carrier.parts))
        carrier.request = " / ".join(words for _, words in carrier.parts)
        carrier.created_at = min(carrier.created_at, entry.created_at)
        starts = [start for start in (carrier.started_at, entry.started_at) if start is not None]
        carrier.started_at = min(starts) if starts else None
        carrier.progress_offers += entry.progress_offers
        scoped_ready = sum(1 for parent, branch in self._ready_items.values() if parent is entry and branch is not None)
        carrier.ready += max(0, entry.ready - scoped_ready)
        for key, (parent, branch) in tuple(self._ready_items.items()):
            if parent is entry and branch is None:
                self._ready_items[key] = (carrier, None)
        carrier.carries += (entry.id, *entry.carries)
        carrier.placeholders.update(entry.placeholders)
        carrier.revision += 1
        for branch in self.branches.values():
            if branch.id == carrier.id:
                branch.request, branch.parts, branch.request_known = carrier.request, carrier.parts, True
        self.observe("worknote.joined", delegation_id=delegation_id, taker=carrier.id)
        return carrier.id

    def _leave_if_done(self, entry: OpenWork) -> None:
        if entry.ended and entry.ready == 0 and self.entries.get(entry.id) is entry:
            self.entries.pop(entry.id, None)
            if entry.id not in self._scoped:
                closed = {entry.id, *entry.carries}
                self._settled_items = {key for key in self._settled_items if key[0] not in closed}

    def open(self) -> list[OpenWork]:
        """Return the delegations still open, oldest first."""
        opened = []
        for parent in self.entries.values():
            if parent.retired and not parent.ready:
                continue
            branches = [entry for entry in self.branches.values() if entry.id == parent.id]
            if branches:
                if any(
                    delegation_id == parent.id and all(branch is not current for current in branches)
                    for (delegation_id, _), (_, branch) in self._ready_items.items()
                ):
                    opened.append(parent)
                opened.extend(
                    entry for entry in branches if not (entry.ended or entry.retired) or self._published_ready(entry)
                )
            elif parent.id not in self._scoped or parent.ready:
                opened.append(parent)
        return opened

    def answer_ready(self) -> bool:
        """Return whether an answer of an open delegation waits to be heard."""
        return any(entry.ready for entry in self.entries.values()) or any(
            entry.ready for entry in self.branches.values()
        )

    def pending_publications(self) -> frozenset[tuple[str, str]]:
        """Return only the original transport/item identities of results that still await settlement."""
        return frozenset(self._ready_items)

    def newer_requests(self, delegation_ids: tuple[str, ...], work_refs: tuple[str, ...] = ()) -> tuple[dict, ...]:
        """Supply exact known relationships; later chronology never asserts semantic supersession."""
        rows = []
        for entry in self.open():
            branch = entry.branch_ref if isinstance(entry, BranchWork) else None
            if branch in work_refs:
                relation = "same_branch"
            elif branch is None and entry.id in delegation_ids:
                relation = "same_source"
            else:
                relation = "other_open_source_relevance_unknown"
            if entry.request and (not isinstance(entry, BranchWork) or entry.request_known):
                rows.append(
                    {
                        "delegation_id": entry.id,
                        "branch_ref": branch,
                        "relationship": relation,
                        "caller_requests": [words for _, words in entry.parts] or [entry.request],
                    }
                )
        return tuple(rows)

    def quiet_candidates(self, max_lines: int) -> tuple[OpenWork, ...]:
        """Return every quiet job with fresh backend evidence and room under its own cap."""
        return tuple(
            entry for entry in self.open() if self._quiet(entry) and entry.news and entry.progress_offers < max_lines
        )

    def prepare_progress(
        self, delegation_ids: tuple[str | ProgressKey, ...], max_offers: int
    ) -> tuple[tuple[ProgressSource, ...], str]:
        """Freeze evidence for a batch, without presenting caller requests as progress facts."""
        sources, items = [], []
        for delegation_id in dict.fromkeys(delegation_ids):
            entry = (
                self.branches.get(delegation_id.branch_ref)
                if isinstance(delegation_id, ProgressKey)
                else self._fact_entry(delegation_id)
            )
            if isinstance(delegation_id, ProgressKey) and entry is not None and entry.id != delegation_id.delegation_id:
                continue
            if entry is None or not self._quiet(entry) or not entry.news or entry.progress_offers >= max_offers:
                continue
            source = ProgressSource(
                entry,
                entry.revision,
                delegation_id=entry.id,
                branch_ref=entry.branch_ref if isinstance(entry, BranchWork) else None,
                owner_revision=entry.owner_revision if isinstance(entry, BranchWork) else 0,
            )
            sources.append(source)
            state = self.prompts.work_texts.waiting
            if entry.state == WorkState.RUNNING:
                state = self.prompts.work_texts.running.format(
                    seconds=round(self.clock() - (entry.started_at or self.clock()))
                ) + self._calls_text(entry)
            update = self.prompts.work_texts.update.format(progress=entry.progress) if entry.progress else ""
            row = self.prompts.work_texts.progress_item.format(number=len(items) + 1, state=state, update=update)
            scope = (
                {"work_refs": [source.branch_ref]} if source.branch_ref is not None else {"delegation_ids": [source.id]}
            )
            items.append(f"{row} {json.dumps(scope, ensure_ascii=False)}")
        return tuple(sources), self.prompts.work_in_progress.format(items="\n".join(items))

    @staticmethod
    def progress_offered(sources: tuple[ProgressSource, ...]) -> None:
        """Charge only the jobs in a progress request admitted for wording."""
        for source in sources:
            source.entry.progress_offers += 1

    def progress_current(self, sources: tuple[ProgressSource, ...]) -> bool:
        """Whether all offered jobs still have exactly the evidence and ownership the writer read."""
        return all(
            (
                self.branches.get(source.branch_ref) is source.entry
                and source.entry.id == source.id
                and source.entry.owner_revision == source.owner_revision
                if source.branch_ref is not None
                else self.entries.get(source.id) is source.entry
            )
            and source.entry.revision == source.revision
            and source.entry.news
            and self._quiet(source.entry)
            for source in sources
        )

    def progress_worded(self, sources: tuple[ProgressSource, ...]) -> None:
        """Consume only the revisions actually worded, preserving any newer backend note."""
        for source in sources:
            if self.progress_current((source,)):
                source.entry.news = False

    @staticmethod
    def _quiet(entry: OpenWork) -> bool:
        return not (entry.reserved or entry.ready or entry.ended or entry.retired or entry.waits_for_app)

    def line(self) -> str:
        """Return what the talker reads about open work: the note, the "none" line when tracking, or nothing."""
        open_entries = self.open()
        if open_entries:
            items = "\n".join(self._item(number, entry) for number, entry in enumerate(open_entries, 1))
            return self.prompts.work_in_progress.format(items=items)
        return self.prompts.no_work_in_progress if self.tracks else ""

    def _item(self, number: int, entry: OpenWork) -> str:
        """Return one entry of the note in the prompt set's words: its state, request and newest progress line."""
        texts = self.prompts.work_texts
        states = []
        if not (entry.ended or entry.retired):
            execution = texts.waiting
            if entry.state == WorkState.RUNNING:
                execution = texts.running.format(seconds=round(self.clock() - (entry.started_at or self.clock())))
                execution += self._calls_text(entry)
            states.append(execution)
        if self._published_ready(entry):
            states.append(self.prompts.answer_ready_label)
        # Each joined request is shortened on its own, so a long older one never hides the newer words.
        request = " / ".join(shorten(words) for _, words in entry.parts) or shorten(entry.request)
        if isinstance(entry, BranchWork) and not entry.request_known:
            request = ""
        progress = texts.update.format(progress=entry.progress) if entry.progress and not entry.ready else ""
        return texts.item.format(number=number, state=". ".join(states), request=request, update=progress)

    def _published_ready(self, entry: OpenWork) -> int:
        """Exclude another transport's original publications from a current branch's displayed ready count."""
        if not isinstance(entry, BranchWork):
            return entry.ready
        foreign = sum(
            1
            for (delegation_id, _), (_, branch) in self._ready_items.items()
            if branch is entry and delegation_id != entry.id
        )
        return max(0, entry.ready - foreign)

    def _calls_text(self, entry: OpenWork) -> str:
        """Return the entry's calls in the set's words: the ended ones with their status, then the running ones."""
        texts = self.prompts.work_texts
        named = [(humanize(call.name), call.status) for call in entry.calls.values() if humanize(call.name)]
        ended = [texts.ended_call.format(name=name, status=status) for name, status in named if status is not None]
        running = [name for name, status in named if status is None]
        text = texts.ended.format(steps=", ".join(ended)) if ended else ""
        return text + (texts.steps.format(steps=", ".join(running)) if running else "")
