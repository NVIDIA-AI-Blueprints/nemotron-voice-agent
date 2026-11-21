# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private host work identities, separate from a stream's public delegation identity."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

from examples.shared.work_provenance import (
    Attribution,
    Binding,
    Obligation,
    Provenance,
    decode_provenance,
)


@dataclass(frozen=True)
class OwnerView:
    """The newest host snapshot of one canonical Core output owner."""

    revision: int
    obligations: tuple[Obligation, ...]
    disposition: str | None = None

    @property
    def branches(self) -> frozenset[str]:
        """Return the real output branches this owner has owed, including transferred ones."""
        return frozenset(row.branch_ref for row in self.obligations if row.branch_ref is not None)

    @property
    def pending(self) -> bool:
        """Whether a delivery, computation, or publication obligation remains."""
        return any(
            row.state in ("pending", "active") or row.delivery_pending or row.publication_pending
            for row in self.obligations
        )


@dataclass(frozen=True)
class ItemScope:
    """The output type and original Work attribution of an exact published item."""

    kind: str | None
    attribution: Attribution | None


class WorkOwnership:
    """Apply authenticated host snapshots without relabeling previously published items.

    The transport delegation is supplied by the actual stream. Core owner references
    are opaque and may differ from it, notably in the legacy late-message path.
    """

    def __init__(self) -> None:
        """Start without trusting a conversation, frame, or branch claim."""
        self.conversation_ref: str | None = None
        self.frames: dict[str, tuple[str, str]] = {}
        self.transports: dict[str, str] = {}
        self.bindings: dict[str, Binding] = {}
        self.request_sources: dict[str, frozenset[str]] = {}
        self.owners: dict[str, OwnerView] = {}
        self.items: dict[tuple[str, str], ItemScope] = {}
        self.composite: set[str] = set()

    @staticmethod
    def _owner(owners: dict[str, OwnerView], owner_ref: str, proposed: OwnerView) -> None:
        if proposed.disposition and proposed.pending:
            raise ValueError("A settled work owner cannot retain pending obligations")
        current = owners.get(owner_ref)
        if current is None:
            owners[owner_ref] = proposed
            return
        if current.disposition and proposed.disposition and current.disposition != proposed.disposition:
            raise ValueError("Contradictory work owner settlement")
        disposition = current.disposition or proposed.disposition
        newest = proposed if proposed.revision > current.revision else current
        if disposition and newest.pending:
            raise ValueError("A settled work owner cannot regain pending obligations")
        if proposed.revision > current.revision:
            owners[owner_ref] = OwnerView(proposed.revision, proposed.obligations, disposition)
        elif proposed.revision == current.revision:
            if proposed.obligations != current.obligations:
                raise ValueError("Contradictory work owner snapshot at the same revision")
            if proposed.disposition and not current.disposition:
                owners[owner_ref] = proposed
        elif proposed.disposition and not current.disposition:
            # A settlement can overtake an older stream, or itself arrive late.
            # Its immutable disposition does not replace newer obligation facts.
            owners[owner_ref] = OwnerView(current.revision, current.obligations, disposition)

    def apply(self, delegation_id: str, event: Mapping, response_id: str | None) -> Provenance | None:
        """Validate a complete event before changing any remembered ownership fact."""
        provenance = decode_provenance(event)
        item = event.get("item")
        item_id = item.get("id") if isinstance(item, Mapping) else event.get("item_id")
        key = (response_id, item_id) if response_id and isinstance(item_id, str) and item_id else None
        output_item = event.get("type") in ("response.output_item.added", "response.output_item.done")
        kind = item.get("type") if output_item and isinstance(item, Mapping) else None
        previous_item = self.items.get(key) if key is not None else None
        if previous_item is not None and previous_item.kind and kind and kind != previous_item.kind:
            raise ValueError("A published work item changed its output type")
        needs_scope = kind in ("message", "function_call", "mcp_call") and (
            previous_item is None or previous_item.kind is None
        )
        if provenance is None:
            frame = self.frames.get(response_id)
            owner = frame[0] if frame is not None else self.owner_for(delegation_id)
            if needs_scope and owner in self.composite:
                raise ValueError("A shared response item lacks its publication scope")
            return None
        if not response_id or provenance.response_id != response_id:
            raise ValueError("Work provenance does not identify the actual response")
        if self.conversation_ref is not None and provenance.conversation_ref != self.conversation_ref:
            raise ValueError("Work provenance belongs to another Core conversation")
        frame = (provenance.owner_ref, delegation_id)
        if provenance.response_id in self.frames and self.frames[provenance.response_id] != frame:
            raise ValueError("Work provenance changed a response's owner or transport")
        transport = self.transports.get(provenance.owner_ref)
        if transport is not None and transport != delegation_id:
            raise ValueError("A Core work owner appeared on another delegation stream")

        bindings = dict(self.bindings)
        owners = dict(self.owners)
        for proposed in provenance.bindings:
            current = bindings.get(proposed.branch_ref)
            if current is None or proposed.revision > current.revision:
                bindings[proposed.branch_ref] = proposed
            elif proposed.revision == current.revision and proposed != current:
                raise ValueError("Contradictory work binding at the same revision")
        self._owner(owners, provenance.owner_ref, OwnerView(provenance.owner_revision, provenance.obligations))
        for settlement in provenance.owner_settlements:
            self._owner(
                owners,
                settlement.owner_ref,
                OwnerView(settlement.owner_revision, settlement.obligations, settlement.disposition),
            )
        relevant = (provenance.obligations, *(row.obligations for row in provenance.owner_settlements))
        if any(row.branch_ref is not None and row.branch_ref not in bindings for rows in relevant for row in rows):
            raise ValueError("A work obligation names an unannounced output branch")
        own_bindings = {row.branch_ref for row in provenance.bindings if row.owner_ref == provenance.owner_ref}
        if any(row.branch_ref not in own_bindings for row in provenance.request_sources):
            raise ValueError("Work request sources lack a binding to their reporting owner")

        attribution = provenance.item_attribution
        composite = {owner for owner, view in owners.items() if len(view.branches) > 1}
        if len({row.branch_ref for row in provenance.obligations if row.branch_ref is not None}) > 1:
            composite.add(provenance.owner_ref)
        has_scope = attribution is not None or provenance.source_item
        if (
            needs_scope
            and not has_scope
            and (provenance.owner_ref in self.composite or provenance.owner_ref in composite)
        ):
            raise ValueError("A shared response item lacks its publication scope")
        if provenance.source_item and kind != "message":
            raise ValueError("Source-level publication must identify a message item")
        if has_scope and key is None:
            raise ValueError("Work item scope has no actual item identity")
        if has_scope and previous_item is not None and previous_item.attribution != attribution:
            raise ValueError("A published work item changed its immutable attribution")
        if attribution is not None:
            binding = bindings.get(attribution.branch_ref)
            if binding is None or attribution.binding_revision > binding.revision:
                raise ValueError("A work item names an unannounced branch revision")
            if attribution.owner_ref != provenance.owner_ref:
                raise ValueError("A work item's publication owner differs from its actual writer")
            if attribution.binding_revision == binding.revision and attribution.owner_ref != binding.owner_ref:
                raise ValueError("A work item contradicts its binding")

        self.conversation_ref = provenance.conversation_ref
        self.frames[response_id] = frame
        self.transports[provenance.owner_ref] = delegation_id
        self.bindings, self.owners = bindings, owners
        for row in provenance.request_sources:
            previous = self.request_sources.get(row.branch_ref, frozenset())
            self.request_sources[row.branch_ref] = previous.union(row.source_refs)
        self.composite.update(composite)
        if has_scope and key is not None:
            item_kind = kind if isinstance(kind, str) and kind else previous_item.kind if previous_item else None
            self.items[key] = ItemScope(item_kind, attribution)
        return provenance

    def transport(self, owner_ref: str) -> str | None:
        """Resolve only an owner actually observed on a stream; never guess from its spelling."""
        return self.transports.get(owner_ref)

    def owner_for(self, delegation_id: str) -> str | None:
        """Return the latest observed canonical owner of this transport delegation."""
        return next(
            (owner for owner, transport in reversed(self.transports.items()) if transport == delegation_id), None
        )

    def view(self, delegation_id: str) -> OwnerView | None:
        """Return the observed owner snapshot, if this stream speaks the private protocol."""
        owner = self.owner_for(delegation_id)
        return self.owners.get(owner) if owner is not None else None

    def is_composite(self, delegation_id: str) -> bool:
        """Whether the source has ever owed several Work branches, including after a split."""
        return self.owner_for(delegation_id) in self.composite

    def attribution(self, response_id: str, item_id: str) -> Attribution | None:
        """Return an item's original attribution, independently of the branch's current owner."""
        item = self.items.get((response_id, item_id))
        return item.attribution if item is not None else None

    def current_transport(self, branch_ref: str) -> str | None:
        """Return a branch's current observed transport without moving old item identities."""
        binding = self.bindings.get(branch_ref)
        return self.transport(binding.owner_ref) if binding is not None else None

    def prune(
        self,
        delegation_ids: Collection[str],
        branch_refs: Collection[str] = (),
        *,
        response_ids: Collection[str] = (),
        max_closed: int | None = None,
    ) -> None:
        """Retain only current/retained frontend owners and their reachable branch facts."""
        kept = {owner for owner, transport in self.transports.items() if transport in delegation_ids}
        if max_closed is not None:
            active = {self.frames[key][0] for key in response_ids if key in self.frames}
            active.update(self.bindings[branch].owner_ref for branch in branch_refs if branch in self.bindings)
            closed = [owner for owner in self.owners if owner in kept and owner not in active]
            kept = active | set(closed[-max_closed:])
        # An owner announced ahead of its stream remains reachable through a kept
        # owner's exact outgoing/pending obligations, not an arbitrary transfer log.
        pending = list(kept)
        visited = set()
        while pending:
            owner = pending.pop()
            if owner in visited:
                continue
            visited.add(owner)
            view = self.owners.get(owner)
            if view is not None:
                successors = {row.to_owner_ref for row in view.obligations if row.to_owner_ref is not None}
                kept.update(successors)
                pending.extend(successors - visited)
        branches = set(branch_refs)
        branches.update(
            branch for owner in kept if (view := self.owners.get(owner)) is not None for branch in view.branches
        )
        self.owners = {owner: view for owner, view in self.owners.items() if owner in kept}
        self.bindings = {branch: binding for branch, binding in self.bindings.items() if branch in branches}
        self.request_sources = {
            branch: sources for branch, sources in self.request_sources.items() if branch in branches
        }
        self.transports = {owner: value for owner, value in self.transports.items() if owner in kept}
        self.frames = {key: value for key, value in self.frames.items() if value[0] in kept}
        if max_closed is not None:
            closed_frames = [key for key in self.frames if key not in response_ids]
            kept_frames = set(response_ids) | set(closed_frames[-max_closed:])
            self.frames = {key: value for key, value in self.frames.items() if key in kept_frames}
        self.items = {key: value for key, value in self.items.items() if key[0] in self.frames}
        self.composite.intersection_update(kept)
