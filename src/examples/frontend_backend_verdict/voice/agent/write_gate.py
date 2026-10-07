# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hold consequential tool calls until the caller confirms the exact call (``write_gate``).

The backend's gated calls never reach the client directly. Each one becomes a
:class:`ConfirmationProposal` whose summary is generated here from the canonical
arguments (never written by a model), and the held call is answered internally
with a ``confirmation_required`` result, so the backend history stays valid. The
agent speaks the summary and a fixed question. Only the first caller turn after
the summary played completely can confirm it, and only the frontend's explicit
``yes`` does. After that, a re-issued call that equals the proposal is the one
gated call that goes out; anything else is held again.

Everything here is domain-agnostic: which tools are gated comes from the generic
read/write classifier (``schema_rules.is_read_tool``) and from configuration.
The gate checks consent and argument consistency, not whether the action is
allowed by policy; that stays with the model.

State is an immutable :class:`GateState`; the runner commits a new one only when
an agent call completes, like the session state, so a cancelled turn leaves no
trace.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from examples.frontend_backend_verdict.text.messages import ToolCall
from examples.frontend_backend_verdict.text.tools import ToolSpec
from examples.frontend_backend_verdict.voice.errors import VoiceConfigError
from examples.frontend_backend_verdict.voice.normalization.schema_rules import is_read_tool

#: The fixed question that closes every spoken summary.
CONFIRM_QUESTION = "Shall I go ahead?"

#: Frontend verdicts for the bound confirmation turn; only ``yes`` confirms.
CONFIRM_YES = "yes"
CONFIRMATIONS = (CONFIRM_YES, "partial", "no", "unclear")

#: Event kinds written to the session's event log.
WRITE_PROPOSED = "write_proposed"
WRITE_PRESENTED = "write_presented"
WRITE_CONFIRMATION = "write_confirmation"
WRITE_CONFIRMED = "write_confirmed"
WRITE_INVALIDATED = "write_invalidated"
WRITE_HELD = "write_held"
WRITE_GATE_ERROR = "write_gate_error"
WRITE_GATE_CONFIG_PROBLEM = "write_gate_config_problem"

HELD_INSTRUCTION = (
    "Nothing was executed. The caller will hear this summary and be asked to confirm. Reply with one short "
    "sentence that introduces it; do not repeat the details or ask for confirmation yourself. If they confirm, "
    "call the same tool again with exactly these arguments."
)


@dataclass(frozen=True, slots=True)
class ToolGateSettings:
    """Per-tool overrides (``write_gate.tools.<name>``)."""

    #: Leaf paths (``a.b``, ``items[].id``) whose values do not change what the action does.
    not_consequential: tuple[str, ...] = ()
    #: ``{path: spoken label}``; a path without one is spoken as its keys in words.
    labels: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WriteGateSettings:
    """The ``write_gate`` configuration section."""

    enabled: bool = False
    default: str = "non_read"
    exempt: tuple[str, ...] = ()
    include: tuple[str, ...] = ()
    tools: Mapping[str, ToolGateSettings] = field(default_factory=dict)
    max_summary_chars: int = 600
    max_unconfirmed_reissues: int = 2


@dataclass(frozen=True, slots=True)
class ConfirmationProposal:
    """One held call, as it would be sent, and the summary generated from it."""

    proposal_id: str
    tool: str
    #: Canonical: after argument normalization, exactly what would be sent.
    arguments: dict[str, Any]
    #: ``(path, rendered value)`` for every consequential leaf, in schema order.
    fields: tuple[tuple[str, str], ...]
    #: Generated from ``fields``, not written by the model.
    summary_text: str
    #: Set when the summary is handed to the agent's speech.
    presentation_id: str | None = None
    #: True only when that audio finished playing with no caller speech before its end.
    presented_complete: bool = False


@dataclass(frozen=True, slots=True)
class GateState:
    """Proposals and the confirmation window, swapped only when an agent call completes."""

    proposals: tuple[ConfirmationProposal, ...] = ()
    #: The proposal whose summary was spoken last, awaiting the next caller turn.
    presented: str | None = None
    #: The proposal the bound caller turn confirmed, awaiting the backend's re-issue.
    confirmed: str | None = None
    #: ``not_confirmed`` records and confirmations for the next backend request.
    notes: tuple[str, ...] = ()
    #: Gated calls held again in the current caller turn.
    reissues: int = 0
    serial: int = 0

    def proposal(self, proposal_id: str | None) -> ConfirmationProposal | None:
        """The pending proposal with ``proposal_id``, if any."""
        return next((item for item in self.proposals if item.proposal_id == proposal_id), None)


@dataclass(frozen=True, slots=True)
class Screened:
    """One backend step's calls after the gate."""

    state: GateState
    allowed: tuple[ToolCall, ...]
    #: Internal results of the held calls, keyed by call id (JSON text).
    held: dict[str, str]
    #: Too many re-issues before a confirmation: stop the backend loop for this caller turn.
    stop: bool = False


Emit = Callable[..., None]


# -- configuration --------------------------------------------------------------------


def resolve_schema(node: Any, root: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """``node`` with a local ``$ref`` followed, and the first object branch of ``anyOf``/``oneOf``/``allOf``."""
    for _ in range(16):  # bounded: a reference cycle ends here
        if not isinstance(node, Mapping):
            return None
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/"):
            target: Any = root
            for part in ref[2:].split("/"):
                target = target.get(part) if isinstance(target, Mapping) else None
            node = target
            continue
        for key in ("anyOf", "oneOf", "allOf"):
            branches = node.get(key)
            if isinstance(branches, list) and branches and "properties" not in node:
                resolved = [resolve_schema(branch, root) for branch in branches]
                node = next((item for item in resolved if item and "properties" in item), resolved[0])
                break
        else:
            return node
    return None


def _schema_child(schema: Mapping[str, Any], segment: str, root: Mapping[str, Any]) -> Mapping[str, Any] | None:
    key, is_list = (segment[:-2], True) if segment.endswith("[]") else (segment, False)
    properties = schema.get("properties")
    if not isinstance(properties, Mapping) or key not in properties:
        return None
    child = resolve_schema(properties[key], root)
    if child is None:
        return None
    if is_list:
        return resolve_schema(child.get("items"), root)
    return child


def schema_has_path(schema: Mapping[str, Any], path: str) -> bool:
    """Whether ``path`` (``a.b``, ``items[].id``) names a property of a JSON schema (local refs followed)."""
    node: Mapping[str, Any] | None = resolve_schema(schema, schema)
    for segment in path.split("."):
        if node is None or not segment:
            return False
        node = _schema_child(node, segment, schema)
    return node is not None


@dataclass(frozen=True, slots=True)
class ResolvedGate:
    """The gate for one session's tools."""

    settings: WriteGateSettings
    gated: frozenset[str]
    specs: Mapping[str, ToolSpec]

    def is_gated(self, name: str) -> bool:
        """Whether a call to ``name`` is held; a tool the session did not offer is classified by its name."""
        if name in self.specs:
            return name in self.gated
        settings = self.settings
        if name in settings.exempt:
            return False
        return name in settings.include or (settings.default == "non_read" and not is_read_tool(name, ""))


def resolve_gate(settings: WriteGateSettings, specs: Sequence[ToolSpec]) -> ResolvedGate | None:
    """Select the gated tools for a session; ``None`` when the gate is off.

    Raises ``VoiceConfigError`` for a tool name in ``exempt``, ``include`` or
    ``tools`` that the session does not offer, or a ``not_consequential`` path
    that is not in that tool's schema. Before the session offers any tools (the
    agent is built before ``session.update``) there is nothing to check against.
    """
    if not settings.enabled:
        return None
    by_name = {spec.name: spec for spec in specs}
    if not by_name:
        return ResolvedGate(settings=settings, gated=frozenset(), specs={})
    for key, names in (("exempt", settings.exempt), ("include", settings.include), ("tools", settings.tools)):
        unknown = sorted(name for name in names if name not in by_name)
        if unknown:
            raise VoiceConfigError(f"write_gate.{key} names tools this session does not offer: {unknown}")
    for name, tool in settings.tools.items():
        for path in (*tool.not_consequential, *tool.labels):
            if not schema_has_path(by_name[name].parameters, path):
                raise VoiceConfigError(f"write_gate.tools.{name}: {path!r} is not in the tool's parameter schema")
    gated = {
        spec.name for spec in specs if settings.default == "non_read" and not is_read_tool(spec.name, spec.description)
    }
    gated = (gated | set(settings.include)) - set(settings.exempt)
    return ResolvedGate(settings=settings, gated=frozenset(gated), specs=by_name)


# -- the generated summary --------------------------------------------------------------


def _ordered_keys(value: Mapping[str, Any], schema: Mapping[str, Any] | None) -> list[str]:
    properties = schema.get("properties") if isinstance(schema, Mapping) else None
    order = [key for key in properties if key in value] if isinstance(properties, Mapping) else []
    return order + [key for key in value if key not in order]


def flatten(
    value: Any, schema: Mapping[str, Any] | None, path: str = "", root: Mapping[str, Any] | None = None
) -> list[tuple[str, Any]]:
    """Every leaf of ``value`` as ``(path, value)``, in schema order; list items are ``name[i]``."""
    root = root if root is not None else (schema or {})
    schema = resolve_schema(schema, root) if schema is not None else None
    if isinstance(value, Mapping) and value:
        leaves: list[tuple[str, Any]] = []
        properties = schema.get("properties") if isinstance(schema, Mapping) else None
        for key in _ordered_keys(value, schema):
            child = properties.get(key) if isinstance(properties, Mapping) else None
            leaves.extend(flatten(value[key], child, f"{path}.{key}" if path else key, root))
        return leaves
    if isinstance(value, list) and value:
        items = schema.get("items") if isinstance(schema, Mapping) else None
        leaves = []
        for index, item in enumerate(value):
            leaves.extend(flatten(item, items, f"{path}[{index}]", root))
        return leaves
    return [(path, value)]


def pattern_of(path: str) -> str:
    """``passengers[0].first_name`` -> ``passengers[].first_name`` (the configured form)."""
    out, depth = [], 0
    for char in path:
        if char == "[":
            depth += 1
            out.append("[")
        elif char == "]":
            depth -= 1
            out.append("]")
        elif depth == 0:
            out.append(char)
    return "".join(out)


def render_value(value: Any) -> str:
    """Fixed rendering: strings verbatim, numbers as numbers, booleans as yes/no."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None or value == [] or value == {}:
        return "none"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, int | float):
        return str(value)
    return str(value)


def spoken_label(path: str, labels: Mapping[str, str]) -> str:
    """The configured label for ``path``, or its keys in words (list positions counted from 1)."""
    pattern = pattern_of(path)
    if pattern in labels:
        return labels[pattern]
    words: list[str] = []
    for segment in path.split("."):
        name, _, rest = segment.partition("[")
        words.append(name.replace("_", " "))
        while rest:
            index, _, rest = rest.partition("]")
            words.append(str(int(index) + 1))
            rest = rest.removeprefix("[")
    return " ".join(word for word in words if word)


def summarize(
    tool: str, arguments: Mapping[str, Any], spec: ToolSpec | None, settings: ToolGateSettings
) -> tuple[tuple[tuple[str, str], ...], str]:
    """The consequential fields and the summary text of one call."""
    skipped = set(settings.not_consequential)
    leaves = flatten(dict(arguments), spec.parameters if spec is not None else None)
    fields = tuple((path, render_value(value)) for path, value in leaves if path and pattern_of(path) not in skipped)
    parts = [f"{spoken_label(path, settings.labels)}: {value}" for path, value in fields]
    action = tool.replace("_", " ")
    summary = f"{action}, with {'; '.join(parts)}." if parts else f"{action}."
    return fields, summary[:1].upper() + summary[1:]


def _differences(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    old, new = dict(flatten(dict(before), None)), dict(flatten(dict(after), None))
    return {
        path: {"before": old.get(path), "after": new.get(path)}
        for path in [*old, *(path for path in new if path not in old)]
        if old.get(path) != new.get(path)
    }


def _result(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(", ", ": "))


def spoken_presentation(framing: str, proposal: ConfirmationProposal) -> str:
    """What the agent says: the model's framing sentence, the generated summary, the fixed question."""
    return " ".join(part for part in (framing.strip(), proposal.summary_text, CONFIRM_QUESTION) if part)


# -- the gate ---------------------------------------------------------------------------------


class WriteGate:
    """Pure transitions over :class:`GateState` for one session's resolved gate."""

    def __init__(self, resolved: ResolvedGate, emit: Emit) -> None:
        """Bind the session's gated tools and the event-log emitter."""
        self.resolved = resolved
        self._emit = emit

    @property
    def settings(self) -> WriteGateSettings:
        """The configured section."""
        return self.resolved.settings

    def screen(self, calls: Sequence[ToolCall], state: GateState) -> Screened:
        """Pass every call through the gate before anything is sent (fails closed)."""
        allowed: list[ToolCall] = []
        held: dict[str, str] = {}
        created: set[str] = set()
        for call in calls:
            if not self.resolved.is_gated(call.name):
                allowed.append(call)
                continue
            try:
                state, outcome = self._screen_one(call, state, created)
            except Exception as exc:  # noqa: BLE001 - the gate fails closed: hold the call
                self._emit(WRITE_GATE_ERROR, call_id=call.id, tool=call.name, error=f"{type(exc).__name__}: {exc}")
                held[call.id] = _result(
                    {"status": "confirmation_required", "tool": call.name, "error": "write_gate_error"}
                )
                continue
            if outcome is None:
                allowed.append(call)
            else:
                held[call.id] = outcome
        stop = bool(held) and not allowed and state.reissues >= self.settings.max_unconfirmed_reissues
        return Screened(state=state, allowed=tuple(allowed), held=held, stop=stop)

    def _screen_one(self, call: ToolCall, state: GateState, created: set[str]) -> tuple[GateState, str | None]:
        arguments = call.arguments
        confirmed = state.proposal(state.confirmed)
        if confirmed is not None:
            if confirmed.tool == call.name and confirmed.arguments == arguments:
                self._emit(WRITE_CONFIRMED, proposal_id=confirmed.proposal_id, call_id=call.id, tool=call.name)
                remaining = tuple(item for item in state.proposals if item is not confirmed)
                return replace(state, proposals=remaining, confirmed=None), None
            state = self.invalidate(state, confirmed.proposal_id, "superseded")
            return self._propose(call, state, created, previous=confirmed, status="arguments_mismatch")
        earlier = [item for item in state.proposals if item.proposal_id not in created]
        same = next((item for item in earlier if item.tool == call.name and item.arguments == arguments), None)
        if same is not None:
            state = replace(state, reissues=state.reissues + 1)
            self._emit(
                WRITE_HELD, proposal_id=same.proposal_id, call_id=call.id, tool=call.name, reissue=state.reissues
            )
            return state, _result(self._required(same))
        rival = next((item for item in earlier if item.tool == call.name), None)
        if rival is not None:
            state = replace(state, reissues=state.reissues + 1)
            state = self.invalidate(state, rival.proposal_id, "superseded")
            return self._propose(call, state, created, previous=rival, status="arguments_mismatch")
        for item in earlier:
            state = self.invalidate(state, item.proposal_id, "superseded")
        return self._propose(call, state, created)

    def _propose(
        self,
        call: ToolCall,
        state: GateState,
        created: set[str],
        *,
        previous: ConfirmationProposal | None = None,
        status: str = "confirmation_required",
    ) -> tuple[GateState, str]:
        tool_settings = self.settings.tools.get(call.name, ToolGateSettings())
        fields, summary = summarize(call.name, call.arguments, self.resolved.specs.get(call.name), tool_settings)
        if len(summary) > self.settings.max_summary_chars:
            # Never truncated: an incomplete summary cannot be consented to.
            self._emit(
                WRITE_GATE_CONFIG_PROBLEM,
                call_id=call.id,
                tool=call.name,
                problem="summary_too_long",
                chars=len(summary),
                max_summary_chars=self.settings.max_summary_chars,
            )
            return state, _result(
                {
                    "status": "summary_too_long",
                    "tool": call.name,
                    "instruction": (
                        "Nothing was executed. This action has too many details to confirm in one summary. "
                        "Split it into smaller calls if the tool allows it; otherwise tell the caller it "
                        "cannot be done in this call."
                    ),
                }
            )
        serial = state.serial + 1
        proposal = ConfirmationProposal(
            proposal_id=f"P{serial}",
            tool=call.name,
            arguments=call.arguments,
            fields=fields,
            summary_text=summary,
        )
        created.add(proposal.proposal_id)
        state = replace(state, proposals=(*state.proposals, proposal), serial=serial)
        self._emit(
            WRITE_PROPOSED,
            proposal_id=proposal.proposal_id,
            call_id=call.id,
            tool=call.name,
            arguments=call.arguments,
            fields=[list(item) for item in fields],
            summary=summary,
        )
        if previous is None:
            return state, _result(self._required(proposal))
        return state, _result(
            {
                **self._required(proposal),
                "status": status,
                "previous_proposal_id": previous.proposal_id,
                "differences": _differences(previous.arguments, call.arguments),
            }
        )

    @staticmethod
    def _required(proposal: ConfirmationProposal) -> dict[str, Any]:
        return {
            "status": "confirmation_required",
            "proposal_id": proposal.proposal_id,
            "tool": proposal.tool,
            "fields": [list(item) for item in proposal.fields],
            "instruction": HELD_INSTRUCTION,
        }

    def invalidate(self, state: GateState, proposal_id: str, reason: str) -> GateState:
        """Drop a proposal for good; the next backend request is told why."""
        if state.proposal(proposal_id) is None:
            return state
        self._emit(WRITE_INVALIDATED, proposal_id=proposal_id, reason=reason)
        note = _result({"status": "not_confirmed", "proposal_id": proposal_id, "reason": reason})
        return replace(
            state,
            proposals=tuple(item for item in state.proposals if item.proposal_id != proposal_id),
            presented=None if state.presented == proposal_id else state.presented,
            confirmed=None if state.confirmed == proposal_id else state.confirmed,
            notes=(*state.notes, note),
        )

    def unpresented(self, state: GateState) -> ConfirmationProposal | None:
        """The oldest pending proposal not yet spoken (it is spoken next)."""
        return next((item for item in state.proposals if item.presentation_id is None), None)

    def present(self, state: GateState, proposal: ConfirmationProposal) -> GateState:
        """Record that ``proposal``'s summary is being spoken."""
        presented = replace(proposal, presentation_id=f"pres_{proposal.proposal_id}")
        self._emit(WRITE_PRESENTED, proposal_id=proposal.proposal_id, presentation_id=presented.presentation_id)
        proposals = tuple(presented if item is proposal else item for item in state.proposals)
        return replace(state, proposals=proposals, presented=proposal.proposal_id)

    def begin_caller_turn(
        self, state: GateState, heard: Mapping[str, bool]
    ) -> tuple[GateState, ConfirmationProposal | None]:
        """A new caller turn starts: close the last window and bind it to a completely heard summary.

        Returns the proposal this turn may confirm, if any. Every other pending
        proposal is invalidated: a confirmation binds only to the first caller
        turn after a complete presentation.
        """
        state = replace(state, reissues=0)
        bound: ConfirmationProposal | None = None
        for item in state.proposals:
            if item.proposal_id == state.presented and state.confirmed is None and heard.get(item.proposal_id):
                bound = replace(item, presented_complete=True)
                continue
            if item.proposal_id == state.confirmed:
                reason = "expired"
            elif item.proposal_id == state.presented:
                reason = "interrupted"
            else:
                reason = "superseded"
            state = self.invalidate(state, item.proposal_id, reason)
        if bound is not None:
            proposals = tuple(bound if item.proposal_id == bound.proposal_id else item for item in state.proposals)
            state = replace(state, proposals=proposals, presented=None)
        return state, bound

    def judge(self, state: GateState, bound: ConfirmationProposal, verdict: str) -> GateState:
        """Apply the frontend's verdict for the bound turn; anything but ``yes`` invalidates."""
        verdict = verdict if verdict in CONFIRMATIONS else "unclear"
        self._emit(WRITE_CONFIRMATION, proposal_id=bound.proposal_id, confirmation=verdict)
        if verdict != CONFIRM_YES:
            return self.invalidate(state, bound.proposal_id, verdict)
        note = _result(
            {
                "confirmation": CONFIRM_YES,
                "proposal_id": bound.proposal_id,
                "tool": bound.tool,
                "arguments": bound.arguments,
            }
        )
        return replace(state, confirmed=bound.proposal_id, notes=(*state.notes, note))

    @staticmethod
    def take_notes(state: GateState) -> tuple[GateState, tuple[str, ...]]:
        """The notes for the next backend request, removed from the state."""
        return replace(state, notes=()), state.notes
