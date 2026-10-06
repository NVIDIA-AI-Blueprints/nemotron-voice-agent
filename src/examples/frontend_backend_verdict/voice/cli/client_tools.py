# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-executed tools for the Realtime clients (the tau3 flow, run locally).

The client registers the tools in ``session.update``, executes each
``response.function_call_arguments.done`` itself, and answers with
``function_call_output`` items followed by one ``response.create``, exactly as
tau2 does. Tool lists are the text prototype's ``ToolSpec`` lists.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Sequence
from typing import Any

from examples.frontend_backend_verdict.text.messages import ToolCall
from examples.frontend_backend_verdict.text.tools import ToolRegistry, ToolSpec


def load_tools(reference: str) -> list[ToolSpec]:
    """Import ``module:attr`` returning a list of ``ToolSpec``."""
    module_name, _, attr = reference.partition(":")
    if not module_name or not attr:
        raise ValueError(f"expected 'module:attr', got {reference!r}")
    tools = getattr(importlib.import_module(module_name), attr)
    if not isinstance(tools, list | tuple) or not all(isinstance(tool, ToolSpec) for tool in tools):
        raise ValueError(f"{reference} is not a list of ToolSpec")
    return list(tools)


def realtime_schema(spec: ToolSpec) -> dict[str, Any]:
    """Flat GA Realtime function-tool schema (no ``function`` wrapper)."""
    return {"type": "function", "name": spec.name, "description": spec.description, "parameters": spec.parameters}


class ClientToolbox:
    """Executes function calls locally and formats outputs the way tau2 does."""

    def __init__(self, tools: Sequence[ToolSpec] = ()) -> None:
        """Index ``tools`` by name."""
        self._registry = ToolRegistry(tools)
        self._specs = list(tools)

    def schemas(self) -> list[dict[str, Any]]:
        """Tools for ``session.update``."""
        return [realtime_schema(spec) for spec in self._specs]

    async def execute(self, call_id: str, name: str, arguments: str) -> str:
        """Run one call; failures become ``"Error: ..."`` strings, as tau2 sends them."""
        try:
            json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return f"Error: invalid JSON arguments: {exc}"
        spec = self._registry.get(name)
        if spec is None or spec.callable is None:
            return f"Error: unknown tool {name!r}"
        result = await self._registry.execute(ToolCall(id=call_id, name=name, arguments_json=arguments or "{}"))
        return f"Error: {result.content}" if result.is_error else result.content
