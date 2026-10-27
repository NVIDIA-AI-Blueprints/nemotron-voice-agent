# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How results and backend-run calls reach the wire: the result formats and call reports, chosen by name.

The IM's own work tools reach the wire by what they are, whatever ``delegation.backend_calls`` says: ``show`` is always
an ``mcp_call`` item, so the display reaches the screen, and a control call such as ``cancel_work`` never is.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from delegation_core.backend.port import Outcome

if TYPE_CHECKING:
    from delegation_core.im.run import TaskRun


class ResultFormat:
    """``plain``: no note, no work tool; the default, since a voice-only client never sees display-only content."""

    def __init__(self, note: str = ""):
        """Carry the format's note (``texts.result_formats[name]``), which the IM adds to every task's instructions."""
        self.note = note

    def tools(self) -> tuple[str, ...]:
        """The names of the work tools this format offers every task."""
        return ()


class SpeakDisplay(ResultFormat):
    """``speak_display``: the envelope. Its note asks for a ``show`` call, then the answer to speak."""

    def tools(self) -> tuple[str, ...]:
        """``show``: what the screen shows."""
        return ("show",)


class CallReports:
    """``items``: a call the backend runs is an official ``mcp_call`` item, from its start to its outcome."""

    def started(self, task: TaskRun, call: dict, label: str) -> None:
        """The call starts: its item, ``in_progress``, labeled with its tool source."""
        task.work.write(lambda writer: writer.mcp_started(call, label))

    def ended(self, task: TaskRun, call_id: str, output: str, outcome: Outcome) -> None:
        """The call ended: its event, then its item with the output or the error."""
        task.work.write(lambda writer: writer.mcp_ended(call_id, output, outcome))


class ControlCalls(CallReports):
    """A control call of the IM's own (the worker's ``cancel_work``): it changes nothing in the world, so no item.

    The IM records its action and its outcome all the same; the task's answer says what became of the work.
    """

    def started(self, task: TaskRun, call: dict, label: str) -> None:
        """Nothing on the wire."""

    def ended(self, task: TaskRun, call_id: str, output: str, outcome: Outcome) -> None:
        """Nothing on the wire."""


class CallNotes(CallReports):
    """``notes``: a call the backend runs shows only as a progress note, its step line, when it ends."""

    def started(self, task: TaskRun, call: dict, label: str) -> None:
        """Nothing on the wire yet."""

    def ended(self, task: TaskRun, call_id: str, output: str, outcome: Outcome) -> None:
        """The call's step line as a note."""
        task.progress(task.step_line(call_id))


# The result formats by the name ``delegation.result_format`` gives; a composition root picks one at build.
RESULT_FORMATS: Mapping[str, type[ResultFormat]] = {"plain": ResultFormat, "speak_display": SpeakDisplay}
# How backend-run calls reach the wire, by the name ``delegation.backend_calls`` gives; picked at build.
CALL_REPORTS: Mapping[str, type[CallReports]] = {"items": CallReports, "notes": CallNotes}
