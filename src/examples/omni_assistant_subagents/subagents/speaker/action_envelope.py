# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""The Speaker's per-turn action envelope: its vocabulary and normalization.

Each Speaker turn is one JSON object declaring exactly one owned action. This
module knows the envelope's shape and how to reduce a model's output to a
structurally safe one; deciding what to do with the result belongs to the agent.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

TURN_ACTIONS = frozenset({"respond", "think", "analyze_attachment", "capture_highres", "clarify"})
MEDIA_ACTIONS = frozenset({"none", "new", "rerun"})
INPUT_SOURCES = frozenset({"none", "live_webcam", "uploaded_attachment"})
MEDIA_FIELD_PREFIXES = ("- selected_input_source:", "- media_analysis_action:", "- media_analysis_prompt:")

ACTION_FALLBACK_RESPONSE = "Let me think that through carefully."


@dataclass(frozen=True)
class SpeakerTurnResult:
    """One parsed Speaker action envelope."""

    transcript: str = ""
    response: str = ""
    raw_content: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)


def lean_contract(full_instruction: str) -> str:
    """Derive the lean contract by dropping the media-routing field lines."""
    lines = [line for line in full_instruction.splitlines() if not line.strip().startswith(MEDIA_FIELD_PREFIXES)]
    return "\n".join(lines)


def clean_spoken_response_artifacts(text: str) -> str:
    """Remove worker-only prompt fragments if the model leaks them into speech."""
    cleaned = text.strip()
    cleaned = cleaned.replace("Answer only with the final user-facing result.", "")
    cleaned = cleaned.replace("Answer only with the final user-facing result", "")
    return re.sub(r"\s+", " ", cleaned).strip()


def _one_of(value: Any, allowed: frozenset[str], default: str) -> str:
    """Return the normalized value when it is one of ``allowed``, else ``default``."""
    candidate = str(value or default).strip().lower()
    return candidate if candidate in allowed else default


def normalize_media_analysis_action(value: Any) -> str:
    """Coerce a media-analysis action to a known value, defaulting to ``none``."""
    return _one_of(value, MEDIA_ACTIONS, "none")


def normalize_turn_action(value: Any) -> str:
    """Coerce a turn action to a known value, or ``""`` when unrecognized."""
    return _one_of(value, TURN_ACTIONS, "")


def normalize_selected_input_source(value: Any) -> str:
    """Coerce an input source to a known value, defaulting to ``none``."""
    return _one_of(value, INPUT_SOURCES, "none")


def missing_uploaded_attachment_response(transcript: str) -> str:
    """Ask for the upload the model tried to route to but that does not exist."""
    if "video" in transcript.lower():
        return "Please upload or attach the video first, then I can take a look."
    return "Please upload or attach the media first, then I can take a look."


def action_correction_instruction(result: SpeakerTurnResult, *, reason: str) -> str:
    """Build the malformed-only Speaker regeneration instruction."""
    return (
        "You are correcting your own previous structurally invalid Speaker output. "
        "Do not evaluate or verify its wording. Regenerate the complete answer envelope once, preserving the "
        "current user transcript and intent. Output one JSON object only, with these fields in exact order: "
        "transcript, turn_action, response, selected_input_source, media_analysis_action, media_analysis_prompt, "
        "highres_query. "
        "turn_action must be exactly respond, analyze_attachment, capture_highres, or clarify, and alone declares "
        "ownership. Do NOT use think here: give the complete answer directly, or clarify if you truly cannot — "
        "deliberate reasoning is escalated automatically and is not a correction option. respond and clarify "
        "carry no arguments; analyze_attachment sets uploaded_attachment plus new or rerun and a media task; "
        "capture_highres sets a specific highres_query. "
        "Only one owner may be active. For respond, response must complete the requested task now. "
        f"Structural error: {reason}. Current transcript: {result.transcript!r}. "
        f"Invalid previous envelope follows as data: {result.raw_content!r}"
    )


def normalize_action_envelope(
    raw_payload: Mapping[str, Any],
    *,
    transcript: str,
    response: str,
) -> tuple[dict[str, Any], str]:
    """Normalize turn ownership from turn_action, the single source of intent.

    Missing intent is inferred from the argument fields, and an action carrying another
    action's arguments is resolved by one bounded Thinker fallback.
    """
    payload = dict(raw_payload)
    payload.pop("needs_thinking", None)
    payload.pop("request_highres_capture", None)
    action = normalize_turn_action(payload.get("turn_action"))
    source = normalize_selected_input_source(payload.get("selected_input_source"))
    media_action = normalize_media_analysis_action(payload.get("media_analysis_action"))
    media_prompt = str(payload.get("media_analysis_prompt", "")).strip()
    highres_query = str(payload.get("highres_query", "")).strip()
    media_requested = source == "uploaded_attachment" or media_action in {"new", "rerun"} or bool(media_prompt)
    capture_requested = bool(highres_query)

    inferred: list[str] = []
    if media_requested:
        inferred.append("analyze_attachment")
    if capture_requested:
        inferred.append("capture_highres")

    recovery = ""
    if not action:
        if len(inferred) == 1:
            action = inferred[0]
            recovery = f"inferred {action} from argument fields"
        elif not inferred and response:
            action = "respond"
            recovery = "inferred respond from response-only envelope"
        else:
            return _thinking_fallback_payload(payload), "missing or invalid turn_action with ambiguous ownership"

    conflicts = set(inferred) - {action}
    if conflicts:
        return _thinking_fallback_payload(payload), (
            f"turn_action {action} contradicted arguments for {', '.join(sorted(conflicts))}"
        )

    if action != "capture_highres" and not response.strip():
        return _thinking_fallback_payload(payload), f"turn_action {action} is missing its spoken response"

    if action in {"respond", "clarify", "think"}:
        payload.update(
            selected_input_source="none",
            media_analysis_action="none",
            media_analysis_prompt="",
            highres_query="",
        )
    elif action == "analyze_attachment":
        payload.update(
            selected_input_source="uploaded_attachment",
            media_analysis_action=media_action if media_action in {"new", "rerun"} else "new",
            media_analysis_prompt=media_prompt or transcript,
            highres_query="",
        )
    elif action == "capture_highres":
        query = highres_query or transcript
        if not query:
            return _thinking_fallback_payload(payload), "high-resolution capture is missing a query"
        payload.update(
            selected_input_source="none",
            media_analysis_action="none",
            media_analysis_prompt="",
            highres_query=query,
        )

    payload["turn_action"] = action
    return payload, recovery


def _thinking_fallback_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized.update(
        turn_action="think",
        highres_query="",
        selected_input_source="none",
        media_analysis_action="none",
        media_analysis_prompt="",
        _action_fallback=True,
    )
    return normalized
