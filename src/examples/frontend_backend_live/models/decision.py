# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The talker's output contract: one `{action, speech}` decision, and the spec of the call that asks for it.

The parser accepts every action of `TalkerAction`; what a call allows is its caller's to enforce (the talker's floor,
the relay's checks). Anything unusable delegates the original request: no tool has run at the talker
stage, so delegating can never repeat an action.

The tool-calling talker's routing call makes the same decision with a call of a routing tool (`TOOL_DECISIONS`), or
with text and no call for a `speak`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from examples.shared.json_parsing import extract_json_object

# Routing speaks, delegates or waits; the writer speaks.
TalkerAction = Literal["speak", "delegate", "wait"]
ACTIONS = get_args(TalkerAction)
# Why the parser could not use a decision, so a caller can tell such a delegation from the model's own choice.
UNUSABLE = frozenset({"empty_output", "invalid_json", "invalid_decision", "incomplete_output"})


@dataclass(frozen=True)
class CallSpec:
    """What one model call is, stated by its caller: nothing about a call is inferred from its actions."""

    role: str  # who calls: talker or writer; it names the call in the logs
    actions: tuple[str, ...]  # the actions the decision may take: the JSON schema's enum
    hint: str  # the description of ``speech`` in the JSON schema


@dataclass(frozen=True)
class TalkerDecision:
    """One routing or commentary decision: an action and the words to speak."""

    action: TalkerAction
    speech: str = ""
    diagnostics: dict = field(default_factory=dict)
    call_id: str = ""  # the routing tool's call that made it, whose handler Pipecat runs; empty: none


def unusable_decision(metadata: dict, reason: str) -> TalkerDecision:
    """Delegate the original request for unusable output; ``reason`` is one of ``UNUSABLE``."""
    return TalkerDecision("delegate", "", {**metadata, "recovery": reason})


def decision_from_output_text(text: str, metadata: dict) -> TalkerDecision:
    """Validate completed output: the first JSON object in it. Shared by every call of the talker.

    The object is found by the repository's shared extractor (``examples.shared.json_parsing``), which reads the first
    complete object, so a valid decision followed by stray text is read as the decision.
    """
    if not (text or "").strip():
        return unusable_decision(metadata, "empty_output")
    data = extract_json_object(text)
    if not data:
        return unusable_decision(metadata, "invalid_json")
    if set(data) == {"action", "speech"} and data["action"] in ACTIONS and isinstance(data["speech"], str):
        return TalkerDecision(data["action"], data["speech"], metadata)
    return unusable_decision(metadata, "invalid_decision")


def delegated(arguments: Mapping[str, Any], diagnostics: dict) -> TalkerDecision:
    """A ``delegate`` call: the turn goes to the backend, its ``lead_in`` the lead-in when it is text, else none."""
    lead_in = arguments.get("lead_in")
    return TalkerDecision("delegate", lead_in if isinstance(lead_in, str) else "", diagnostics)


def waited(arguments: Mapping[str, Any], diagnostics: dict) -> TalkerDecision:
    """A ``wait`` call: nothing is said and nothing is handed over now."""
    return TalkerDecision("wait", "", diagnostics)


# The tool-calling talker's routing tools, by name, in the order they are offered and win when a reply calls more than
# one: the decision a call of each one makes, from its arguments.
TOOL_DECISIONS: dict[str, Callable[[Mapping[str, Any], dict], TalkerDecision]] = {
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
