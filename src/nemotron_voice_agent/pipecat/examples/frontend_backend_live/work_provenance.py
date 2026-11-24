# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private Core event attribution, separate from model items and public Live events."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Annotated, Literal

from pydantic import AfterValidator, BeforeValidator, ConfigDict, Field, StrictBool, StrictInt, StrictStr, TypeAdapter

from nemotron_voice_agent.pipecat.shared.delegation_contract import WORK_PROVENANCE_KEY


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("Work-provenance references must be nonempty")
    return value


def _array(value: object) -> object:
    if not isinstance(value, list):
        raise ValueError("Work-provenance collections must be arrays")
    return value


def _object(value: object) -> object:
    if not isinstance(value, dict):
        raise ValueError("Work-provenance rows must be objects")
    return value


class _WireRecord:
    __pydantic_config__ = ConfigDict(extra="forbid")


type _WireRow[T] = Annotated[T, BeforeValidator(_object)]
type _Rows[T] = Annotated[tuple[T, ...], BeforeValidator(_array)]
_Ref = Annotated[StrictStr, AfterValidator(_nonblank)]
_Revision = Annotated[StrictInt, Field(ge=0)]


@dataclass(frozen=True)
class Binding(_WireRecord):
    """A Work's stable output branch and its current owner revision."""

    branch_ref: _Ref
    owner_ref: _Ref
    revision: _Revision


@dataclass(frozen=True)
class BranchSources(_WireRecord):
    """The canonical request sources that have joined this Work, independent of its current writer."""

    branch_ref: _Ref
    source_refs: Annotated[_Rows[_Ref], Field(min_length=1)]


@dataclass(frozen=True)
class Obligation(_WireRecord):
    """One real input-placement or publication obligation of a response owner."""

    obligation_ref: _Ref
    branch_ref: _Ref | None
    state: Literal["pending", "active", "settled", "transferred"]
    delivery_pending: StrictBool
    publication_pending: StrictBool
    terminal_outcome: Literal["answered", "failed", "cancelled"] | None
    stop_outcome: Literal["stopped", "unconfirmed"] | None
    to_owner_ref: _Ref | None


def _validate_obligation(value: Obligation) -> Obligation:
    if value.state == "settled" and (
        value.delivery_pending or value.publication_pending or value.terminal_outcome is None
    ):
        raise ValueError("A settled work obligation cannot retain pending delivery or publication")
    if value.state == "transferred" and value.to_owner_ref is None:
        raise ValueError("A transferred work obligation must name its destination")
    return value


_ObligationRow = Annotated[_WireRow[Obligation], AfterValidator(_validate_obligation)]


@dataclass(frozen=True)
class Attribution(_WireRecord):
    """The branch and owner that actually published an item, unaffected by later moves."""

    branch_ref: _Ref
    owner_ref: _Ref
    binding_revision: _Revision
    focused: StrictBool = False
    task_ref: _Ref | None = None


@dataclass(frozen=True)
class OwnerSettlement(_WireRecord):
    """A logical owner settlement, which does not fabricate an old response terminal."""

    owner_ref: _Ref
    owner_revision: _Revision
    disposition: Literal["answered", "failed", "cancelled", "handed_over"]
    obligations: _Rows[_ObligationRow]


@dataclass(frozen=True)
class Provenance(_WireRecord):
    """Host-only metadata on an existing event envelope, never on an output item."""

    conversation_ref: _Ref
    response_id: _Ref
    owner_ref: _Ref
    owner_revision: _Revision
    bindings: _Rows[_WireRow[Binding]]
    obligations: _Rows[_ObligationRow]
    item_attribution: _WireRow[Attribution] | None = None
    owner_settlements: _Rows[_WireRow[OwnerSettlement]] = ()
    version: StrictInt = 1
    request_sources: _Rows[_WireRow[BranchSources]] = ()
    source_item: StrictBool = False
    routing_question: StrictBool = False


_PROVENANCE = TypeAdapter(_WireRow[Provenance])
_REQUIRED_WIRE_FIELDS = {"version", "item_attribution", "owner_settlements"}


def _distinct(values) -> None:
    values = tuple(values)
    if len(values) != len(set(values)):
        raise ValueError("Duplicate private work-provenance identity")


def decode_provenance(event: Mapping[str, object]) -> Provenance | None:
    """Decode the reserved envelope; legacy absence is ``None``, malformed data raises ``ValueError``.

    Conversation/writer authorization and monotonic ownership are consumer state.
    """
    if WORK_PROVENANCE_KEY not in event:
        return None
    raw = event[WORK_PROVENANCE_KEY]
    if not isinstance(raw, dict) or not raw.keys() >= _REQUIRED_WIRE_FIELDS:
        raise ValueError("Missing required private work-provenance fields")
    value = _PROVENANCE.validate_python(raw)
    if value.version != 1:
        raise ValueError("Unsupported private work-provenance version")
    if value.source_item and value.item_attribution is not None:
        raise ValueError("A source-level item cannot also claim a Work branch")
    if value.routing_question and not value.source_item:
        raise ValueError("A routing question must be a source-level item")
    _distinct(binding.branch_ref for binding in value.bindings)
    _distinct(obligation.obligation_ref for obligation in value.obligations)
    _distinct(settlement.owner_ref for settlement in value.owner_settlements)
    _distinct(branch.branch_ref for branch in value.request_sources)
    for settlement in value.owner_settlements:
        _distinct(obligation.obligation_ref for obligation in settlement.obligations)
    for branch in value.request_sources:
        _distinct(branch.source_refs)
    return value


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
    if result["routing_question"] is False:
        result.pop("routing_question")
    attribution = result["item_attribution"]
    if attribution is not None:
        if attribution["focused"] is False:
            attribution.pop("focused")
        if attribution["task_ref"] is None:
            attribution.pop("task_ref")
    decode_provenance({WORK_PROVENANCE_KEY: result})
    return result
