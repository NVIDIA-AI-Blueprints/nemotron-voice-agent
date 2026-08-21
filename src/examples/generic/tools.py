# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Tool schema loading and handlers for the generic cascaded pipeline."""

from pathlib import Path

from loguru import logger
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema

from examples.generic.tool_handlers import TOOL_HANDLERS as TOOL_HANDLERS  # noqa: F401
from utils import load_tools_catalog


def build_tools_schema(
    module_file: str | Path,
    tool_names: list[str],
    *,
    client_tools: list[dict] | None = None,
) -> tuple[ToolsSchema | None, list[str]]:
    """Build one schema containing server-catalog and client-owned tools.

    A tool is included only when (a) the YAML entry exists, (b) the entry's
    ``function.name`` equals the catalog key, and (c) a handler is registered
    in ``TOOL_HANDLERS``. This prevents the schema from advertising callable
    tools that the pipeline cannot fulfil. Failed entries are logged and
    skipped.

    Returns ``(schema, registered_names)`` so callers can register handlers
    for exactly the tools that made it into the schema.
    """
    catalog = load_tools_catalog(module_file) if tool_names else {}
    if tool_names and not catalog:
        logger.warning("Tools catalog not found beside example module; tool calling disabled")

    tools: list[dict] = [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            },
        }
        for tool in (client_tools or [])
    ]
    registered: list[str] = []
    for name in tool_names:
        entry = catalog.get(name)
        if not isinstance(entry, dict):
            logger.warning(f"Tool '{name}' not found in tools.yaml; skipping")
            continue
        fn = entry.get("function") if isinstance(entry.get("function"), dict) else {}
        schema_name = fn.get("name")
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
