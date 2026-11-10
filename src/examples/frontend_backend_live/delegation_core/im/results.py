# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How results and backend calls reach the wire: the result formats, chosen by name, and the call reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from examples.frontend_backend_live.delegation_core.backend.port import Outcome

if TYPE_CHECKING:
    from examples.frontend_backend_live.delegation_core.im.run import TaskRun


class ResultFormat:
    """``plain``: no note, no work tool; the default, since a voice-only client never sees display-only content."""

    def __init__(self, note: str = ""):
        """Carry the format's note (``texts.result_formats[name]``), which the IM adds to every task's instructions."""
        self.note = note

    def tools(self) -> tuple[str, ...]:
        """Return the names of the work tools this format offers every task."""
        return ()


class SpeakDisplay(ResultFormat):
    """``speak_display``: the envelope. Its note asks for a ``show`` call, then the answer to speak."""

    def tools(self) -> tuple[str, ...]:
        """Return ``show``, which puts the display on the screen."""
        return ("show",)


class CallReports:
    """A call the backend runs, as an official ``mcp_call`` item from its start to its outcome."""

    def started(self, task: TaskRun, call: dict, label: str) -> None:
        """Write the call's item as it starts, ``in_progress``, labeled with its tool source."""
        task.work.write(lambda writer: writer.mcp_started(call, label))

    def ended(self, task: TaskRun, call_id: str, output: str, outcome: Outcome) -> None:
        """Write the call's end into the response that holds it: its event, then its item with the output or error."""
        if not task.session.book.mcp_ended(call_id, output, outcome):
            task.work.write(lambda writer: writer.mcp_ended(call_id, output, outcome))


class ControlCalls(CallReports):
    """A control call of the IM's own, such as ``cancel_work``: it changes nothing in the world, so it has no item."""

    def started(self, task: TaskRun, call: dict, label: str) -> None:
        """Write nothing."""

    def ended(self, task: TaskRun, call_id: str, output: str, outcome: Outcome) -> None:
        """Write nothing."""


RESULT_FORMATS: Mapping[str, type[ResultFormat]] = {"plain": ResultFormat, "speak_display": SpeakDisplay}
