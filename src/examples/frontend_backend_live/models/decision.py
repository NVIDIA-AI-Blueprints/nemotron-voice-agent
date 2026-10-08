# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The frontend model's output contract: one `{action, speech}` decision.

Routing actions are `speak` and `delegate`; commentary phrasing may also `listen` (stay silent).
Anything unusable delegates the original request: no tool has run at the talker stage, so
delegating can never repeat an action.
"""

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TalkerDecision:
    """One routing or commentary decision: an action and the words to speak."""

    action: str  # speak | delegate | listen
    speech: str = ""
    diagnostics: dict = field(default_factory=dict)


def unusable_decision(metadata: dict, reason: str) -> TalkerDecision:
    """Unusable routing output delegates the original request; no tool has run yet."""
    return TalkerDecision("delegate", "", {**metadata, "recovery": reason})


def decision_from_output_text(text: str, metadata: dict) -> TalkerDecision:
    """Validate completed talker output. Shared by every model provider."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return unusable_decision(metadata, "empty_output" if not (text or "").strip() else "invalid_json")
    if (
        isinstance(data, dict)
        and set(data) == {"action", "speech"}
        and data["action"] in ("speak", "delegate", "listen")
        and isinstance(data["speech"], str)
    ):
        return TalkerDecision(data["action"], data["speech"], metadata)
    return unusable_decision(metadata, "invalid_decision")


# `speech` means something different per action and per call site (routing has three actions,
# commentary two, a forced answer one); a bare `{"type": "string"}` leaves that to prose alone.
# Keyed by the exact actions tuple so every `decision_schema(actions)` call picks up the right hint.
SPEECH_HINTS = {
    ("speak", "delegate"): (
        "For delegate: a brief acknowledgment that work is starting -- never its outcome, "
        "even a likely one -- or empty when the instructions say not to speak. For speak: the "
        "complete, already-verified answer or a question; never empty."
    ),
    ("speak", "listen"): "A concise spoken paraphrase of the update, or empty to stay silent.",
    ("speak",): "The answer to speak aloud, kept brief.",
}


# Shared by providers: JSON schema for the routing and commentary decisions.
def decision_schema(actions=("speak", "delegate")) -> dict:
    """Return the JSON schema of a decision whose action is one of ``actions``."""
    speech = {"type": "string"}
    hint = SPEECH_HINTS.get(tuple(actions))
    if hint:
        speech["description"] = hint
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(actions)},
            "speech": speech,
        },
        "required": ["action", "speech"],
        "additionalProperties": False,
    }
