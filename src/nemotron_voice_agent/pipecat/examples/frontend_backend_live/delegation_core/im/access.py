# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restrict verification to tool effects explicitly declared read-only by their trusted server source."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import ToolSource


class ReadOnlySource:
    """A capability restriction over an existing owner, preserving its execution and idempotency identity."""

    def __init__(self, source: ToolSource, names: frozenset[str]):
        """Wrap the original execution owner with an explicit name restriction."""
        self.source, self.names, self.label = source, names, source.label

    def tools(self) -> list[dict]:
        """Expose only the permitted schemas."""
        return [deepcopy(tool) for tool in self.source.tools() if tool.get("name") in self.names]

    async def execute(self, call_id: str, name: str, arguments: str) -> str:
        """Delegate a permitted call to its original tool owner."""
        if name not in self.names:
            raise ValueError("This verification capability does not permit that tool")
        return await self.source.execute(call_id, name, arguments)


def read_only_sources(sources: Sequence[ToolSource]) -> tuple[tuple[ToolSource, ...], frozenset[str]]:
    """Use host declarations only, retaining original first-owner name precedence; unknown effects are denied."""
    selected = []
    seen: set[str] = set()
    allowed: set[str] = set()
    for source in sources:
        names = {tool.get("name") for tool in source.tools() if isinstance(tool.get("name"), str)}
        declared = getattr(source, "read_only_tools", frozenset())
        if not isinstance(declared, frozenset) or any(not isinstance(name, str) for name in declared):
            raise ValueError("A trusted read_only_tools declaration must be a frozenset of tool names")
        permitted = frozenset((names - seen) & declared)
        seen.update(names)
        if permitted:
            selected.append(ReadOnlySource(source, permitted))
            allowed.update(permitted)
    return tuple(selected), frozenset(allowed)
