# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Project canonical Realtime function tools onto Pipecat's OpenAI adapter."""

from __future__ import annotations

import copy
from typing import Any

from openai import NOT_GIVEN
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema


def project_function_tools(tools: list[dict[str, Any]]) -> ToolsSchema | Any:
    """Convert validated flat Realtime functions to Pipecat provider tools."""
    if not tools:
        return NOT_GIVEN
    provider_tools: list[dict[str, Any]] = []
    for tool in tools:
        function: dict[str, Any] = {
            "name": tool["name"],
            "parameters": copy.deepcopy(tool.get("parameters", {})),
        }
        if "description" in tool:
            function["description"] = tool["description"]
        provider_tools.append({"type": "function", "function": function})
    return ToolsSchema(
        standard_tools=[],
        custom_tools={AdapterType.OPENAI: provider_tools},
    )
