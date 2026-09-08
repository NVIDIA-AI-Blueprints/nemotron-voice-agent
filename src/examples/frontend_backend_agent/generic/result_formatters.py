# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Deterministic, TTS-safe formatting for trusted generic-tool results."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from examples.frontend_backend_agent.src.protocol import is_speakable_payload, response_hint, tool_result
from examples.frontend_backend_agent.src.tools import ToolSpec

_UNSPEAKABLE_RE = re.compile(r"(?:</?(?:think|tool_call|function|parameter)[^>]*>|```|[*#]{2,})", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")


def _speech_text(value: object, *, max_length: int = 1200) -> str:
    return _SPACE_RE.sub(" ", _UNSPEAKABLE_RE.sub(" ", str(value or ""))).strip()[:max_length]


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


def unsupported_request(specs: Sequence[ToolSpec]) -> dict[str, Any]:
    """Describe only the capabilities enabled for this session."""
    capabilities = [spec.capability for spec in specs if spec.capability]
    if not capabilities:
        text = "No live-data or calculation capabilities are enabled for this session."
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
    return tool_result(
        tool=spec.name,
        status=status,
        data={"arguments": arguments, "result": data},
        response_text=_speech_text(text),
        context=spec.name,
    )


def combine_tool_results(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine payloads in planner order without introducing model-authored facts."""
    text = _speech_text(" ".join(str(payload.get("response_text") or "") for payload in payloads), max_length=1800)
    statuses = [str(payload.get("status") or "error") for payload in payloads]
    status = "success" if statuses and all(item == "success" for item in statuses) else "partial"
    return tool_result(
        tool="multi_tool",
        status=status,
        data={"results": payloads},
        response_text=text or "I finished checking those details.",
        context="multi_tool",
    )
