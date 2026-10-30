# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The IM's work tools: the tools a task gets from the IM, not from its backend.

``show`` (offered by the ``speak_display`` result format) returns the display it was given as an ``mcp_call`` item, and
``cancel_work`` (when ``delegation.work_tools`` lists it) stops the task after its current step. Their descriptions and
parameters are the core's texts (``texts.work_tools``), and so is the output of a call to a name they do not offer
(``texts.unknown_tool_output``).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from delegation_core.config import Texts


class WorkTools:
    """One task's work tools (``port.ToolSource``)."""

    def __init__(self, names: tuple[str, ...], texts: Texts, cancel: Callable[[], str], label: str):
        """Offer ``names``, worded by ``texts`` (``texts.work_tools``); ``cancel`` stops the task after its step.

        ``label`` is the server label of the display's ``mcp_call`` items (``delegation.display_label``).
        """
        self.names, self.texts, self.cancel, self.label = names, texts, cancel, label
        self.schemas = texts.work_tools
        self._run = {"show": self._show, "cancel_work": self._cancel_work}

    def tools(self) -> list[dict]:
        """The tools it offers, as Responses function tools."""
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
        """Run one call; a tool it does not offer gives an error output.

        Raises:
            ValueError: The arguments are not what the tool takes (the call failed).
        """
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
