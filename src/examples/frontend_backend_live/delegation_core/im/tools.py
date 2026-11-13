# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The IM's work tools: the tools a task gets from the IM, not from its backend."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from examples.frontend_backend_live.delegation_core.config import Texts


class WorkTools:
    """One task's work tools (``port.ToolSource``)."""

    def __init__(
        self,
        names: tuple[str, ...],
        texts: Texts,
        cancel: Callable[[], str],
        ask: Callable[[str], str],
        carry_on: Callable[[str], str],
        label: str,
    ):
        """Offer ``names`` as ``texts.work_tools`` words them.

        ``cancel`` stops the task, ``ask`` hands its question for the caller to the IM, ``carry_on`` hands the IM the
        message that the work goes on after the call, and ``label`` tags the display.
        """
        self.names, self.texts, self.cancel, self.ask, self.carry_on = names, texts, cancel, ask, carry_on
        self.label, self.schemas = label, texts.work_tools
        self._run = {
            "show": self._show,
            "cancel_work": self._cancel_work,
            "ask_caller": self._ask_caller,
            "continue_after_call": self._continue_after_call,
        }

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

    def _ask_caller(self, given: dict) -> str:
        question = given.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("ask_caller: question must be non-empty text")
        return self.ask(question.strip())

    def _continue_after_call(self, given: dict) -> str:
        message = given.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("continue_after_call: message must be non-empty text")
        return self.carry_on(message.strip())
