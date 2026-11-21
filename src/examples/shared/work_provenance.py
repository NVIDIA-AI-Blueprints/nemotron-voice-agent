# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private Core event attribution, separate from model items and public Live events."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

WORK_PROVENANCE_KEY = "_nva_work"


@dataclass(frozen=True)
class Binding:
    """A Work's stable output branch and its current owner revision."""

    branch_ref: str
    owner_ref: str
    revision: int


@dataclass(frozen=True)
class BranchSources:
    """The canonical request sources that have joined this Work, independent of its current writer."""

    branch_ref: str
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class Obligation:
    """One real input-placement or publication obligation of a response owner."""

    obligation_ref: str
    branch_ref: str | None
    state: str
    delivery_pending: bool
    publication_pending: bool
    terminal_outcome: str | None
    stop_outcome: str | None
    to_owner_ref: str | None


@dataclass(frozen=True)
class Attribution:
    """The branch and owner that actually published an item, unaffected by later moves."""

    branch_ref: str
    owner_ref: str
    binding_revision: int
    focused: bool = False
    task_ref: str | None = None


@dataclass(frozen=True)
class OwnerSettlement:
    """A logical owner settlement, which does not fabricate an old response terminal."""

    owner_ref: str
    owner_revision: int
    disposition: str
    obligations: tuple[Obligation, ...]


@dataclass(frozen=True)
class Provenance:
    """Host-only metadata on an existing event envelope, never on an output item."""

    conversation_ref: str
    response_id: str
    owner_ref: str
    owner_revision: int
    bindings: tuple[Binding, ...]
    obligations: tuple[Obligation, ...]
    item_attribution: Attribution | None = None
    owner_settlements: tuple[OwnerSettlement, ...] = ()
    version: int = 1
    request_sources: tuple[BranchSources, ...] = ()
    source_item: bool = False


def _object(value: Any, fields: tuple[str, ...]) -> dict:
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError("Invalid private work-provenance fields")
    return value


def _text(value: Any) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError("Work-provenance references must be nonempty strings")
    return value


def _integer(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("Work-provenance revisions must be nonnegative integers")
    return value


def _boolean(value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError("Work-provenance flags must be booleans")
    return value


def _choice(value: Any, choices: tuple[str, ...], *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if type(value) is not str or value not in choices:
        raise ValueError("Invalid work-provenance disposition")
    return value


def _nullable_text(value: Any) -> str | None:
    return None if value is None else _text(value)


def _rows[T](value: Any, parse: Callable[[Any], T], identity: str) -> tuple[T, ...]:
    if type(value) is not list:
        raise ValueError("Work-provenance collections must be arrays")
    rows = tuple(parse(row) for row in value)
    keys = [getattr(row, identity) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate private work-provenance identity")
    return rows


def _binding(value: Any) -> Binding:
    row = _object(value, ("branch_ref", "owner_ref", "revision"))
    return Binding(_text(row["branch_ref"]), _text(row["owner_ref"]), _integer(row["revision"]))


def _branch_sources(value: Any) -> BranchSources:
    row = _object(value, ("branch_ref", "source_refs"))
    if type(row["source_refs"]) is not list:
        raise ValueError("Work request sources must be an array")
    refs = tuple(_text(ref) for ref in row["source_refs"])
    if not refs or len(set(refs)) != len(refs):
        raise ValueError("Work request sources must be nonempty and distinct")
    return BranchSources(_text(row["branch_ref"]), refs)


def _obligation(value: Any) -> Obligation:
    row = _object(value, tuple(Obligation.__dataclass_fields__))
    result = Obligation(
        _text(row["obligation_ref"]),
        _nullable_text(row["branch_ref"]),
        _choice(row["state"], ("pending", "active", "settled", "transferred")),
        _boolean(row["delivery_pending"]),
        _boolean(row["publication_pending"]),
        _choice(row["terminal_outcome"], ("answered", "failed", "cancelled"), nullable=True),
        _choice(row["stop_outcome"], ("stopped", "unconfirmed"), nullable=True),
        _nullable_text(row["to_owner_ref"]),
    )
    if result.state == "settled" and (
        result.delivery_pending or result.publication_pending or result.terminal_outcome is None
    ):
        raise ValueError("A settled work obligation cannot retain pending delivery or publication")
    if result.state == "transferred" and result.to_owner_ref is None:
        raise ValueError("A transferred work obligation must name its destination")
    return result


def _attribution(value: Any) -> Attribution:
    if type(value) is not dict:
        raise ValueError("Invalid private work-provenance fields")
    row = _object({"focused": False, "task_ref": None, **value}, tuple(Attribution.__dataclass_fields__))
    return Attribution(
        _text(row["branch_ref"]),
        _text(row["owner_ref"]),
        _integer(row["binding_revision"]),
        _boolean(row["focused"]),
        _nullable_text(row["task_ref"]),
    )


def _settlement(value: Any) -> OwnerSettlement:
    row = _object(value, ("owner_ref", "owner_revision", "disposition", "obligations"))
    return OwnerSettlement(
        _text(row["owner_ref"]),
        _integer(row["owner_revision"]),
        _choice(row["disposition"], ("answered", "failed", "cancelled", "handed_over")),
        _rows(row["obligations"], _obligation, "obligation_ref"),
    )


def decode_provenance(event: Mapping[str, object]) -> Provenance | None:
    """Decode the reserved envelope; legacy absence is ``None``, malformed data raises ``ValueError``.

    Conversation/writer authorization and monotonic ownership are consumer state, not structural decoding.
    In particular, a canonical Core owner and its gateway transport delegation need not have equal IDs.
    """
    if WORK_PROVENANCE_KEY not in event:
        return None
    raw = event[WORK_PROVENANCE_KEY]
    if type(raw) is not dict:
        raise ValueError("Invalid private work-provenance fields")
    row = _object({"request_sources": [], "source_item": False, **raw}, tuple(Provenance.__dataclass_fields__))
    version = _integer(row["version"])
    if version != 1:
        raise ValueError("Unsupported private work-provenance version")
    source_item = _boolean(row["source_item"])
    if source_item and row["item_attribution"] is not None:
        raise ValueError("A source-level item cannot also claim a Work branch")
    return Provenance(
        _text(row["conversation_ref"]),
        _text(row["response_id"]),
        _text(row["owner_ref"]),
        _integer(row["owner_revision"]),
        _rows(row["bindings"], _binding, "branch_ref"),
        _rows(row["obligations"], _obligation, "obligation_ref"),
        None if row["item_attribution"] is None else _attribution(row["item_attribution"]),
        _rows(row["owner_settlements"], _settlement, "owner_ref"),
        version,
        _rows(row["request_sources"], _branch_sources, "branch_ref"),
        source_item,
    )


def encode_provenance(value: Provenance) -> dict:
    """Return a validated JSON-safe envelope; it contains no model item or runtime object."""
    try:
        result = json.loads(json.dumps(asdict(value)))
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid private work-provenance value") from error
    if result["request_sources"] == []:
        result.pop("request_sources")
    if result["source_item"] is False:
        result.pop("source_item")
    attribution = result["item_attribution"]
    if attribution is not None:
        if attribution["focused"] is False:
            attribution.pop("focused")
        if attribution["task_ref"] is None:
            attribution.pop("task_ref")
    decode_provenance({WORK_PROVENANCE_KEY: result})
    return result


def strip_provenance(event: Mapping[str, object]) -> dict:
    """Copy an event without its reserved envelope, leaving opaque business content and the journal intact."""
    return {key: value for key, value in event.items() if key != WORK_PROVENANCE_KEY}
