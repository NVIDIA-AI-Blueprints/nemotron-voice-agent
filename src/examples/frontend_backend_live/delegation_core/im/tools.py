# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The IM's work tools: the tools a task gets from the IM, not from its backend.

What an app function does (:func:`declared_tool_kind`): its own MCP annotations, then the operator's
``delegation.tool_kinds``; a tool neither declares is undeclared, and the write gate holds it on its path.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any

from examples.shared.delegation_contract import TOOL_ANNOTATIONS

if TYPE_CHECKING:
    from examples.frontend_backend_live.delegation_core.config import Texts, ToolKindSettings


class ToolKind(StrEnum):
    """What an app function does to the app's data."""

    READ = "read"  # changes nothing the app keeps
    WRITE = "write"  # may change something


class KindSource(StrEnum):
    """Where a tool's kind came from, for the traces."""

    ANNOTATIONS = "annotations"  # the tool's own MCP annotations
    OPERATOR = "operator"  # delegation.tool_kinds.read_tools or write_tools


@dataclass(frozen=True)
class DeclaredKind:
    """What the app or the operator declares a tool does, and who declared it."""

    kind: ToolKind
    source: KindSource
    reversible: bool = False  # a write the app declares easily undone (destructiveHint false)


def _hint(annotations: Mapping[str, Any], name: str) -> bool | None:
    """An MCP hint's value; ``None`` when it is missing or not a boolean (no hint)."""
    value = annotations.get(name)
    return value if isinstance(value, bool) else None


def declared_tool_kind(tool: Mapping[str, Any], settings: ToolKindSettings) -> DeclaredKind | None:
    """What the app or the operator declares ``tool`` does; ``None`` when neither does (undeclared).

    The tool's annotations first, in the MCP ``ToolAnnotations`` form: ``readOnlyHint`` true is a read; ``readOnlyHint``
    false, or ``destructiveHint`` true, is a write; a write with ``destructiveHint`` false is reversible. A hint that is
    not a boolean is no hint. Then the operator's lists: a name or an ``fnmatch`` pattern (case-sensitive) in
    ``settings.write_tools`` makes the tool a write, else one in ``settings.read_tools`` makes it a read.
    """
    annotations = tool.get(TOOL_ANNOTATIONS)
    if isinstance(annotations, Mapping):
        read_only, destructive = _hint(annotations, "readOnlyHint"), _hint(annotations, "destructiveHint")
        if read_only is True:
            return DeclaredKind(ToolKind.READ, KindSource.ANNOTATIONS)
        if read_only is False or destructive is True:
            return DeclaredKind(ToolKind.WRITE, KindSource.ANNOTATIONS, reversible=destructive is False)
    name = str(tool.get("name", ""))
    if any(fnmatchcase(name, pattern) for pattern in settings.write_tools):
        return DeclaredKind(ToolKind.WRITE, KindSource.OPERATOR)
    if any(fnmatchcase(name, pattern) for pattern in settings.read_tools):
        return DeclaredKind(ToolKind.READ, KindSource.OPERATOR)
    return None


def is_declared_read(tool: Mapping[str, Any], settings: ToolKindSettings) -> bool:
    """Whether the app or the operator declares that ``tool`` changes nothing.

    ``declared_tool_kind`` gives a ``DeclaredKind`` whose ``kind`` is ``ToolKind.READ``. An undeclared tool (``None``)
    and a write are both false.
    """
    declared = declared_tool_kind(tool, settings)
    return declared is not None and declared.kind is ToolKind.READ


class WorkTools:
    """One task's work tools (``port.ToolSource``)."""

    def __init__(self, names: tuple[str, ...], texts: Texts, cancel: Callable[[], str], label: str):
        """Offer ``names`` as ``texts.work_tools`` words them; ``cancel`` stops the task, ``label`` tags the display."""
        self.names, self.texts, self.cancel, self.label = names, texts, cancel, label
        self.schemas = texts.work_tools
        self._run = {"show": self._show, "cancel_work": self._cancel_work}

    def tools(self) -> list[dict]:
        """Return the tools it offers, as Responses function tools."""
        return [
            {
                "type": "function",
                "name": name,
                "description": self.schemas[name]["description"],
                "parameters": dict(self.schemas[name]["parameters"]),
            }
            for name in self.names
        ]

    async def execute(self, call_id: str, name: str, arguments: str) -> str:
        """Run one call; a tool it does not offer gives an error output, and bad arguments raise ``ValueError``."""
        run = self._run.get(name) if name in self.names else None
        if run is None:
            offered = ", ".join(self.names) or self.texts.none
            return json.dumps({"error": self.texts.unknown_tool_output.format(name=name, offered=offered)})
        try:
            given = json.loads(arguments or "{}")
        except ValueError:
            raise ValueError(f"{name}: the arguments are not JSON") from None
        if not isinstance(given, dict):
            raise ValueError(f"{name}: the arguments must be a JSON object")
        return run(given)

    def _show(self, given: dict) -> str:
        display = given.get("display")
        if not isinstance(display, str) or not display.strip():
            raise ValueError("show: display must be non-empty text")
        return display

    def _cancel_work(self, given: dict) -> str:
        return self.cancel()
