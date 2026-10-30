# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The frontend model's output contract: one `{action, speech}` decision.

The parser accepts every action of `TalkerAction`; what a call allows is its caller's to enforce (the talker's floor,
the writer's wording). Anything unusable delegates the original request: no tool has run at the talker
stage, so delegating can never repeat an action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, get_args

from examples.shared.json_parsing import extract_json_object

if TYPE_CHECKING:
    from .base import CallSpec

# Routing speaks, delegates or waits; the writer speaks.
TalkerAction = Literal["speak", "delegate", "wait"]
ACTIONS = get_args(TalkerAction)
# Why the parser could not use a decision, so a caller can tell such a delegation from the model's own choice.
UNUSABLE = frozenset({"empty_output", "invalid_json", "invalid_decision", "incomplete_output"})


@dataclass(frozen=True)
class TalkerDecision:
    """One routing or commentary decision: an action and the words to speak."""

    action: TalkerAction
    speech: str = ""
    diagnostics: dict = field(default_factory=dict)


def unusable_decision(metadata: dict, reason: str) -> TalkerDecision:
    """Delegate the original request for unusable output; ``reason`` is one of ``UNUSABLE``."""
    return TalkerDecision("delegate", "", {**metadata, "recovery": reason})


def decision_from_output_text(text: str, metadata: dict) -> TalkerDecision:
    """Validate completed output: the first JSON object in it. Shared by every model provider.

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
