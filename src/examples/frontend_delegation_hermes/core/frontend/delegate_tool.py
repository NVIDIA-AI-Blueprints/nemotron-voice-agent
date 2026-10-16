# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The frontend's only tool: ``delegate(delegate, filler_text, request)`` (V1, RC1).

``request`` is optional and defaults to ``"task"``; the backend reads it only while
it is WORKING. :func:`parse_decision` turns a model reply into a :class:`Decision`
or a :class:`ContractError` describing why it broke the contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from examples.frontend_delegation_hermes.core.backend.protocol import REQUEST_STATUS, REQUEST_TASK, REQUESTS
from examples.frontend_delegation_hermes.core.frontend.prompts import PromptCatalog

TOOL_NAME = "delegate"


@dataclass(frozen=True, slots=True)
class Decision:
    """One frontend decision for one user turn."""

    delegate: bool
    filler_text: str
    request: str = REQUEST_TASK
    #: How the decision was reached: "" (clean), text_as_direct, repaired, contract_fallback, timeout, guard.
    repair: str = ""
    latency_ms: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)


class ContractError(ValueError):
    """A reply that is not exactly one well-formed ``delegate`` call."""


def delegate_tool(prompts: PromptCatalog) -> dict[str, Any]:
    """The OpenAI chat-completions tool schema (descriptions come from the prompt catalog)."""
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": prompts.render("delegate_tool_description"),
            "parameters": {
                "type": "object",
                "properties": {
                    "delegate": {"type": "boolean", "description": prompts.render("delegate_param_delegate")},
                    "filler_text": {"type": "string", "description": prompts.render("delegate_param_filler_text")},
                    "request": {
                        "type": "string",
                        "enum": list(REQUESTS),
                        "description": prompts.render("delegate_param_request"),
                    },
                },
                "required": ["delegate", "filler_text"],
            },
        },
    }


def parse_arguments(arguments: str | dict[str, Any]) -> Decision:
    """Validate the arguments of one ``delegate`` call; raises :class:`ContractError`."""
    if isinstance(arguments, str):
        try:
            data = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ContractError(f"arguments are not JSON: {exc}") from exc
    else:
        data = arguments
    if not isinstance(data, dict):
        raise ContractError("arguments must be a JSON object")
    delegate = data.get("delegate")
    if isinstance(delegate, str) and delegate.strip().lower() in ("true", "false"):
        delegate = delegate.strip().lower() == "true"
    if not isinstance(delegate, bool):
        raise ContractError("'delegate' must be a boolean")
    filler = data.get("filler_text")
    if filler is None:
        raise ContractError("'filler_text' is required")
    if not isinstance(filler, str):
        raise ContractError("'filler_text' must be a string")
    request = data.get("request", REQUEST_TASK)
    request = request.strip().lower() if isinstance(request, str) else REQUEST_TASK
    if request not in REQUESTS:
        request = REQUEST_TASK  # RC1: anything but "status" is a task
    return Decision(delegate=delegate, filler_text=" ".join(filler.split()), request=request)


def parse_reply(tool_calls: list[tuple[str, str]], content: str | None) -> Decision:
    """A model reply -> decision. ``tool_calls`` is ``[(name, arguments_json), ...]``.

    Plain text without a tool call is taken as a direct reply (``delegate=false``,
    ``repair="text_as_direct"``); anything else that is not exactly one ``delegate``
    call raises :class:`ContractError`.
    """
    calls = [call for call in tool_calls if call[0] == TOOL_NAME]
    if not tool_calls:
        text = " ".join((content or "").split())
        if text:
            return Decision(delegate=False, filler_text=text, repair="text_as_direct")
        raise ContractError("no tool call and no text")
    if len(calls) != 1 or len(tool_calls) != 1:
        raise ContractError(f"expected exactly one {TOOL_NAME} call, got {[name for name, _ in tool_calls]}")
    return parse_arguments(calls[0][1])


__all__ = [
    "REQUEST_STATUS",
    "REQUEST_TASK",
    "TOOL_NAME",
    "ContractError",
    "Decision",
    "delegate_tool",
    "parse_arguments",
    "parse_reply",
]
