# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The browser profile's tools as the agent's ``ToolSpec`` list (``tools.config_tools``).

``profiles/default.yaml`` sets ``tools.source: config`` and
``tools.config_tools: examples.frontend_backend_agent.tools:TOOLS``, so the backend agent runs
these tools in this process. The schemas come from ``tools.yaml`` next to this module (read by
its path, so ``TOOLS_FILE_PATH`` cannot swap them) and the handlers from ``tool_handlers.py``.
OpenAI Realtime sessions use the ``eval`` profile, whose client sends its own tools.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

from examples.frontend_backend_agent import tool_handlers
from examples.frontend_backend_agent.core.tools.tools import ToolSpec

TOOLS_FILE = Path(__file__).resolve().parent / "tools.yaml"

Handler = Callable[[Mapping[str, Any]], Awaitable[dict[str, Any]]]

#: Tool name -> handler; every tool in ``tools.yaml`` must have one.
HANDLERS: dict[str, Handler] = {
    "get_weather": tool_handlers.get_weather,
    "get_stock_price": tool_handlers.get_stock_price,
    "web_search": tool_handlers.web_search,
    "calculate_bmi": tool_handlers.calculate_bmi,
    "generate_random_number": tool_handlers.generate_random_number,
}


def _callable(handler: Handler) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Adapt a ``handler(arguments)`` to the ``callable(**arguments)`` the agent's registry calls."""

    async def run(**arguments: Any) -> dict[str, Any]:
        return await handler(arguments)

    return run


def load_tools(path: Path = TOOLS_FILE) -> list[ToolSpec]:
    """The tools of ``path`` (OpenAI function schemas) with their handlers, in file order."""
    catalog = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    specs: list[ToolSpec] = []
    for name, entry in catalog.items():
        function = entry["function"]
        if function["name"] != name:
            raise ValueError(f"{path}: tool {name!r} declares function name {function['name']!r}")
        if name not in HANDLERS:
            raise ValueError(f"{path}: tool {name!r} has no handler in tools.py")
        specs.append(
            ToolSpec(
                name=name,
                description=" ".join(str(function.get("description", "")).split()),
                parameters=dict(function.get("parameters") or {"type": "object", "properties": {}}),
                callable=_callable(HANDLERS[name]),
            )
        )
    return specs


TOOLS: list[ToolSpec] = load_tools()
