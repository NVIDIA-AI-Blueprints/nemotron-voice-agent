# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Deterministic, TTS-safe formatting for trusted generic-tool results."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from examples.frontend_backend_agent.src.protocol import is_speakable_payload, response_hint, tool_result
from examples.frontend_backend_agent.src.tools import ToolSpec

_UNSPEAKABLE_RE = re.compile(r"(?:</?(?:think|tool_call|function|parameter)[^>]*>|```|[*#]{2,})", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_MAX_SPOKEN_RESULT_CHARS = 450


def _speech_text(value: object, *, max_length: int = 1200) -> str:
    return _SPACE_RE.sub(" ", _UNSPEAKABLE_RE.sub(" ", str(value or ""))).strip()[:max_length]


def _bounded_speech(value: object, *, max_sentences: int) -> str:
    """Return TTS-safe prose with deterministic sentence and character ceilings."""
    text = _speech_text(value, max_length=5000)
    if not text:
        return ""
    sentences = [sentence for sentence in _SENTENCE_BOUNDARY_RE.split(text) if sentence]
    bounded = " ".join(sentences[:max_sentences])
    if len(bounded) <= _MAX_SPOKEN_RESULT_CHARS:
        return bounded
    prefix = bounded[: _MAX_SPOKEN_RESULT_CHARS - 1]
    if " " in prefix:
        prefix = prefix.rsplit(" ", 1)[0]
    return prefix.rstrip(" ,;:-.") + "."


def missing_parameters(spec: ToolSpec, names: list[str]) -> dict[str, Any]:
    """Return a deterministic clarification for missing required fields."""
    readable = [spec.params[name].label or name.replace("_", " ") for name in names]
    requested = readable[0] if len(readable) == 1 else f"{', '.join(readable[:-1])} and {readable[-1]}"
    return response_hint(
        reason="params_missing",
        action="req_params",
        params_needed=names,
        response_text=f"Please tell me {requested}.",
        context=spec.name,
    )


def missing_client_parameters(tool: str, labels: Sequence[str], names: Sequence[str]) -> dict[str, Any]:
    """Return a deterministic clarification for a client-owned tool's missing fields.

    Labels come from the caller's declared schema, never from the planner's own
    sentence, so the spoken words stay trusted even though the tool is not ours.
    """
    readable = list(labels)
    requested = readable[0] if len(readable) == 1 else f"{', '.join(readable[:-1])} and {readable[-1]}"
    return response_hint(
        reason="params_missing",
        action="req_params",
        params_needed=list(names),
        response_text=f"Please tell me {requested}.",
        context=tool,
    )


#: Values longer than this are not spoken in full, so they cannot be approved by a bare "yes".
_MAX_CONFIRMATION_VALUE_CHARS = 80
_MAX_CONFIRMATION_CHARS = 450
_FIRST_CLAUSE_RE = re.compile(r"^(.*?)(?:[.;:!?](?:\s|$)|$)", re.DOTALL)


def confirmation_text(
    tool: str,
    description: str,
    labels: Mapping[str, str],
    arguments: Mapping[str, Any],
) -> tuple[str, bool]:
    """Render the fixed consent question for one call, and whether any value had to be shortened.

    The action is the opening clause of the tool's own description; each
    argument is said as "label: value" with the value exactly as it will be
    sent. A value that is a list, an object, empty or very long is not spoken
    in full, and the question is marked summarized: a bare "yes" to it can
    never approve the call on its own.
    """
    clause = _FIRST_CLAUSE_RE.match(_speech_text(description, max_length=300))
    action = (clause.group(1).strip() if clause else "") or tool.replace("_", " ").strip() or "that action"
    action = action[0].lower() + action[1:] if action[:1].isupper() and action[1:2].islower() else action
    parts: list[str] = []
    summarized = False
    for name, value in arguments.items():
        label = labels.get(name) or name.replace("_", " ")
        if isinstance(value, bool):
            rendered = "yes" if value else "no"
        elif isinstance(value, int | float):
            rendered = str(value)
        elif isinstance(value, str) and 0 < len(value.strip()) <= _MAX_CONFIRMATION_VALUE_CHARS:
            rendered = _speech_text(value)
            summarized = summarized or rendered != value.strip()
        else:
            summarized = True
            continue
        parts.append(f"{label}: {rendered}")
    details = f", with {', '.join(parts)}" if parts else ""
    text = f"Just to confirm, I will {action}{details}. Shall I go ahead?"
    if len(text) > _MAX_CONFIRMATION_CHARS:
        summarized = True
        text = f"Just to confirm, I will {action}. Shall I go ahead?"
    return text, summarized


def confirmation_request(
    tool: str,
    arguments: Mapping[str, Any],
    *,
    description: str = "",
    labels: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Ask for consent before one validated action runs.

    The sentence comes from a fixed template over the tool's own schema and the
    exact arguments (:func:`confirmation_text`), never from the planner. The
    Talker speaks it word for word; ``summarized`` tells the backend whether a
    bare "yes" to it may approve the call.
    """
    text, summarized = confirmation_text(tool, description, labels or {}, arguments)
    payload = response_hint(
        reason="confirmation_needed",
        action="req_confirmation",
        params_resolved=dict(arguments),
        response_text=text,
        context=tool,
    )
    payload["summarized"] = summarized
    return payload


def nothing_further() -> dict[str, Any]:
    """Close a turn that needed no tool, without claiming a failure."""
    return response_hint(
        reason="no_action_needed",
        action="answer_directly",
        response_text="Nothing further was needed for that.",
        context="general",
    )


def unspecified_clarification(tool: str) -> dict[str, Any]:
    """Ask for more detail when a client schema names no field we can speak."""
    return response_hint(
        reason="params_missing",
        action="req_params",
        response_text="I need a little more information to do that. What details should I use?",
        context=tool,
    )


def invalid_parameters(tool: str) -> dict[str, Any]:
    """Return a safe clarification without relaying validator internals."""
    return response_hint(
        reason="params_invalid",
        action="req_params",
        response_text="I couldn't use those details. Could you restate them with the units or range you want?",
        context=tool,
    )


def disabled_tool(tool: str) -> dict[str, Any]:
    """Explain a disabled capability without exposing implementation details."""
    return response_hint(
        reason="tool_disabled",
        action="unsupported",
        response_text="That capability is not enabled for this session.",
        context=tool,
    )


def unsupported_request(specs: Sequence[ToolSpec], *, suppress_capabilities: bool = False) -> dict[str, Any]:
    """Describe only the capabilities enabled for this session.

    A session whose real tools are client-owned has no server capability worth
    naming: listing the built-in ones would advertise weather and BMI on, say,
    a call about the caller's own account. Suppress the list there and let the Talker, which holds the
    caller's own instructions, phrase the refusal in its domain.
    """
    capabilities = [] if suppress_capabilities else [spec.capability for spec in specs if spec.capability]
    if not capabilities:
        text = "That isn't something I can do in this session."
    elif len(capabilities) == 1:
        text = f"I can {capabilities[0]}."
    elif len(capabilities) == 2:
        text = f"I can {capabilities[0]} or {capabilities[1]}."
    else:
        text = f"I can {', '.join(capabilities[:-1])}, or {capabilities[-1]}."
    return response_hint(
        reason="unsupported_request",
        action="answer_directly",
        response_text=text,
        context="general",
    )


def planner_failure() -> dict[str, Any]:
    """Return a non-sensitive fallback for malformed or failed plans."""
    return response_hint(
        reason="planner_error",
        action="retry",
        response_text="I couldn't complete that request reliably. Please say it again and I'll retry.",
        context="general",
    )


def timeout_failure() -> dict[str, Any]:
    """Return a bounded-deadline fallback."""
    return response_hint(
        reason="timeout",
        action="retry",
        response_text="That check took too long, so I stopped it. Would you like me to try again?",
        context="general",
    )


def format_tool_result(spec: ToolSpec, arguments: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Build speech only from validated arguments and returned service data."""
    if spec.speak is None:
        if is_speakable_payload(data):
            return data
        raise ValueError(f"Tool {spec.name} returned no speakable protocol payload")
    status = str(data.get("status") or "error")
    if status == "unavailable":
        text = _speech_text(data.get("assistant_should_say")) or "I couldn't complete that check right now."
    elif status == "not_found":
        text = _speech_text(data.get("message")) or "I couldn't find a matching result."
    elif status != "success":
        text = "I couldn't complete that check right now. Would you like me to try again?"
    else:
        text = spec.speak(arguments, data)
    if spec.name == "web_search":
        text = _bounded_speech(text, max_sentences=2)
    return tool_result(
        tool=spec.name,
        status=status,
        data={"arguments": arguments, "result": data},
        response_text=_speech_text(text),
        context=spec.name,
    )


def combine_tool_results(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine payloads in planner order without introducing model-authored facts."""
    text = _bounded_speech(" ".join(str(payload.get("response_text") or "") for payload in payloads), max_sentences=3)
    statuses = [str(payload.get("status") or "error") for payload in payloads]
    status = "success" if statuses and all(item == "success" for item in statuses) else "partial"
    return tool_result(
        tool="multi_tool",
        status=status,
        data={"results": payloads},
        response_text=text or "I finished checking those details.",
        context="multi_tool",
    )
