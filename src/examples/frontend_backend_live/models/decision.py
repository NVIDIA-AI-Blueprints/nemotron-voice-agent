# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The talker's output contract: one ``{action, speech}`` decision, and the spec of the call that asks for it."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from examples.shared.json_parsing import extract_json_object

TalkerAction = Literal["speak", "delegate", "wait"]
ACTIONS = get_args(TalkerAction)
# The parser's reasons, so a caller can tell an unusable decision from the model's own delegation.
UNUSABLE = frozenset({"empty_output", "invalid_json", "invalid_decision", "incomplete_output"})


@dataclass(frozen=True)
class CallSpec:
    """What one model call is, stated by its caller: nothing about a call is inferred from its actions."""

    role: str
    actions: tuple[str, ...]
    hint: str


@dataclass(frozen=True)
class TalkerDecision:
    """One routing or commentary decision: an action and the words to speak."""

    action: TalkerAction
    speech: str = ""
    diagnostics: dict = field(default_factory=dict)
    call_id: str = ""


def unusable_decision(metadata: dict, reason: str) -> TalkerDecision:
    """Return a ``delegate`` decision for unusable output: no tool ran yet, so delegating never repeats an action."""
    return TalkerDecision("delegate", "", {**metadata, "recovery": reason})


def decision_from_output_text(text: str, metadata: dict) -> TalkerDecision:
    """Return the decision in completed output: its first JSON object, whatever text follows it."""
    if not (text or "").strip():
        return unusable_decision(metadata, "empty_output")
    data = extract_json_object(text)
    if not data:
        return unusable_decision(metadata, "invalid_json")
    if set(data) == {"action", "speech"} and data["action"] in ACTIONS and isinstance(data["speech"], str):
        return TalkerDecision(data["action"], data["speech"], metadata)
    return unusable_decision(metadata, "invalid_decision")


def delegated(arguments: Any, diagnostics: dict) -> TalkerDecision:
    """Return the decision of a ``delegate`` call, its ``lead_in`` argument the lead-in when it is text."""
    lead_in = arguments.get("lead_in") if isinstance(arguments, Mapping) else None
    return TalkerDecision("delegate", lead_in if isinstance(lead_in, str) else "", diagnostics)


def waited(arguments: Any, diagnostics: dict) -> TalkerDecision:
    """Return the decision of a ``wait`` call: nothing is said and nothing is handed over."""
    return TalkerDecision("wait", "", diagnostics)


# The routing tools, in the order they are offered and win when a reply calls more than one.
TOOL_DECISIONS: dict[str, Callable[[Any, dict], TalkerDecision]] = {
    "delegate": delegated,
    "wait": waited,
}


def decision_schema(spec: CallSpec) -> dict:
    """Return the JSON schema of the decision ``spec`` asks for: its actions, and its hint for ``speech``."""
    speech = {"type": "string"}
    if spec.hint:
        speech["description"] = spec.hint
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(spec.actions)},
            "speech": speech,
        },
        "required": ["action", "speech"],
        "additionalProperties": False,
    }
