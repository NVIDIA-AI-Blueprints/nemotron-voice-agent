# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured protocol between the Talker and the internal Thinker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ThinkerPayloadType = Literal["response_hint", "tool_result"]
LifecycleMarker = Literal["ThinkerStarted", "IntermediateResponse", "ThinkerCompleted", "ThinkerAborted"]


@dataclass(slots=True, frozen=True)
class ThinkerLifecycleEvent:
    """Internal-only lifecycle/history event for the Talker context/debug log."""

    marker: LifecycleMarker
    call_id: str
    query: str = ""
    payload: dict[str, Any] | None = None
    reason: str = ""
    speakable: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable event representation."""
        data: dict[str, Any] = {
            "marker": self.marker,
            "call_id": self.call_id,
            "speakable": self.speakable,
        }
        if self.query:
            data["query"] = self.query
        if self.payload is not None:
            data["payload"] = self.payload
        if self.reason:
            data["reason"] = self.reason
        return data


def response_hint(
    *,
    reason: str,
    action: str,
    response_text: str,
    context: str,
    params_needed: list[str] | None = None,
    params_resolved: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a user-facing intermediate Thinker payload."""
    payload: dict[str, Any] = {
        "type": "response_hint",
        "reason": reason,
        "action": action,
        "response_text": response_text,
        "context": context,
    }
    if params_needed:
        payload["params_needed"] = params_needed
    if params_resolved:
        payload["params_resolved"] = params_resolved
    payload.update({key: value for key, value in extra.items() if value is not None})
    return payload


def tool_result(
    *,
    tool: str,
    status: str,
    data: dict[str, Any],
    response_text: str,
    context: str,
    **extra: Any,
) -> dict[str, Any]:
    """Build a user-facing final Thinker payload."""
    payload: dict[str, Any] = {
        "type": "tool_result",
        "tool": tool,
        "status": status,
        "data": data,
        "response_text": response_text,
        "context": context,
    }
    payload.update({key: value for key, value in extra.items() if value is not None})
    return payload


def is_speakable_payload(payload: dict[str, Any]) -> bool:
    """Return whether ``payload`` is allowed to drive Talker speech."""
    return payload.get("type") in {"response_hint", "tool_result"} and bool(payload.get("response_text"))
