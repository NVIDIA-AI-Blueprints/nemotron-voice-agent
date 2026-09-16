# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Tool schema loading for the Webex BYOVA customer-care pipeline."""

from pathlib import Path

from loguru import logger
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema

from examples.webex_byova.tool_handlers import TOOL_HANDLERS as TOOL_HANDLERS  # noqa: F401
from utils import load_tools_catalog


def build_tools_schema(module_file: str | Path, tool_names: list[str]) -> tuple[ToolsSchema | None, list[str]]:
    """Build a validated OpenAI-compatible schema for the requested tools."""
    if not tool_names:
        return None, []

    catalog = load_tools_catalog(module_file)
    if not catalog:
        logger.warning("Tools catalog not found beside example module; tool calling disabled")
        return None, []

    tools: list[dict] = []
    registered: list[str] = []
    for name in tool_names:
        entry = catalog.get(name)
        if not isinstance(entry, dict):
            logger.warning(f"Tool '{name}' not found in tools.yaml; skipping")
            continue
        function = entry.get("function") if isinstance(entry.get("function"), dict) else {}
        schema_name = function.get("name")
        if schema_name != name:
            logger.warning(f"Tool '{name}' has mismatched function.name={schema_name!r} in tools.yaml; skipping")
            continue
        if name not in TOOL_HANDLERS:
            logger.warning(f"Tool '{name}' has no registered handler; skipping")
            continue
        tools.append(entry)
        registered.append(name)

    if not tools:
        return None, []
    return ToolsSchema(standard_tools=[], custom_tools={AdapterType.OPENAI: tools}), registered
