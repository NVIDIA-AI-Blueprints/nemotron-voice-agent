# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend-visible tool schema for the Frontend/Backend Agent example."""

from __future__ import annotations

from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema

CALL_BACKEND_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "call_backend",
        "description": (
            "Send a single detailed, context-enriched query to the backend agent. "
            "The backend agent infers whether to search flights, continue booking, or check PNR status. "
            "Do not speak in the same turn as this tool call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A detailed rephrased query containing the current user request and any resolved "
                        "flight search, selection, preference, confirmation, or PNR details from context."
                    ),
                },
                "filler_text": {
                    "type": "string",
                    "description": (
                        "A brief, natural filler sentence to speak only if internal booking/search/PNR work "
                        "takes longer than the configured latency threshold. Do not include final answers or "
                        "specific results."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

CANCEL_BACKEND_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "cancel_backend",
        "description": (
            "Cancel any in-progress backend-agent work when the user says to stop, cancel, "
            "ignore, abandon, or never mind a pending flight task, or when they interrupt pending "
            "flight work by switching to unrelated small talk or a non-flight topic. Do not speak "
            "in the same turn as this tool call."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}

TOOLS_SCHEMA = ToolsSchema(
    standard_tools=[], custom_tools={AdapterType.OPENAI: [CALL_BACKEND_TOOL, CANCEL_BACKEND_TOOL]}
)
