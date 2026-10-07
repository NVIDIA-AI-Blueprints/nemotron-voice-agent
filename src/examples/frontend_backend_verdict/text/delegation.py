# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The frontend's only tool: ``call_backend``.

There is deliberately no cancel/discard tool. Under a turn-based API there is
no reachable path on which the frontend could fire one: internal execution
blocks until the backend turn completes, and in external execution a user
message arriving while tool results are outstanding is intercepted by the
driver-level ``on_user_message_while_pending`` policy before the frontend LLM
is ever called.
"""

from __future__ import annotations

import copy
from typing import Any

CALL_BACKEND = "call_backend"

#: Values of the in-progress ``task`` field (see :data:`FRONTEND_TOOLS_IN_PROGRESS`).
TASK_CONTINUE = "continue"
TASK_NEW = "new"

CALL_BACKEND_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": CALL_BACKEND,
        "description": (
            "Send one detailed, self-contained natural-language request to the backend agent, which "
            "owns all task execution. Do not write normal assistant text in the same turn as this call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The complete current request, restated so it stands alone: include every "
                        "detail already established in the conversation plus the latest user turn. "
                        "Never send only the change ('add a window seat'); send the whole request."
                    ),
                },
                "filler_text": {
                    "type": "string",
                    "description": (
                        "Always provide a short, generic holding phrase that would be safe to say "
                        "aloud while the backend works, such as 'Let me take a look.' Never put "
                        "results, identifiers, names, or guesses in it."
                    ),
                },
            },
            "required": ["query", "filler_text"],
            "additionalProperties": False,
        },
    },
}

#: The complete tool surface offered to the frontend LLM.
FRONTEND_TOOLS: tuple[dict[str, Any], ...] = (CALL_BACKEND_TOOL,)


def _with_task_field(tool: dict[str, Any]) -> dict[str, Any]:
    extended = copy.deepcopy(tool)
    parameters = extended["function"]["parameters"]
    parameters["properties"]["task"] = {
        "type": "string",
        "enum": [TASK_CONTINUE, TASK_NEW],
        "description": (
            f"Only while a request is in progress: '{TASK_CONTINUE}' if the user's latest words don't change "
            f"the request in progress, otherwise '{TASK_NEW}'."
        ),
    }
    parameters["required"] = [*parameters["required"], "task"]
    return extended


#: ``call_backend`` with the required ``task`` field, offered only while a request is in progress
#: (a caller passes an in-progress note). ``FRONTEND_TOOLS`` is unchanged.
CALL_BACKEND_TOOL_IN_PROGRESS: dict[str, Any] = _with_task_field(CALL_BACKEND_TOOL)
FRONTEND_TOOLS_IN_PROGRESS: tuple[dict[str, Any], ...] = (CALL_BACKEND_TOOL_IN_PROGRESS,)

#: Values of the pending-confirmation ``confirmation`` field; only ``yes`` confirms.
CONFIRMATION_VALUES = ("yes", "partial", "no", "unclear")


def _with_confirmation_field(tool: dict[str, Any]) -> dict[str, Any]:
    extended = copy.deepcopy(tool)
    parameters = extended["function"]["parameters"]
    parameters["properties"]["confirmation"] = {
        "type": "string",
        "enum": list(CONFIRMATION_VALUES),
        "description": (
            "Your reading of the user's reply to the action summary they were asked to confirm: 'yes' only if "
            "the user approves the whole summary and adds or changes nothing; 'partial' if they approve some "
            "parts or change anything; 'no' if they decline; 'unclear' if you cannot tell."
        ),
    }
    parameters["required"] = [*parameters["required"], "confirmation"]
    return extended


#: ``call_backend`` with the required ``confirmation`` field, offered (and forced) only on the first
#: user turn after the agent read out an action summary and asked for confirmation (``write_gate``).
CALL_BACKEND_TOOL_CONFIRMATION: dict[str, Any] = _with_confirmation_field(CALL_BACKEND_TOOL)
FRONTEND_TOOLS_CONFIRMATION: tuple[dict[str, Any], ...] = (CALL_BACKEND_TOOL_CONFIRMATION,)
#: ``tool_choice`` that makes the model call ``call_backend`` (it cannot answer directly).
FORCE_CALL_BACKEND: dict[str, Any] = {"type": "function", "function": {"name": CALL_BACKEND}}

PENDING_CONFIRMATION_NOTE = """PENDING CONFIRMATION
You just read this action to the user and asked "Shall I go ahead?":
  {summary}
The latest user message is their reply. For this turn, always call call_backend: restate the request in
query, include the user's exact words, and set "confirmation" to "yes" only if they approve the whole
action and add or change nothing; "partial" if they approve only some of it or change anything; "no" if
they decline; "unclear" if you cannot tell."""
