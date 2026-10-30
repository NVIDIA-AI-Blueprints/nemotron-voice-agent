# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts between the cascade's parts: ready results, their wording, and the display.

The reader (``delegations.py``) reads each ``final_answer`` item as a :class:`ReadyResult` and hands display items to
the :class:`DisplaySink` at once. The presenter words every ready result in one :class:`ResultWording` call as it
arrives; at idle it records the line in the talker's memory and the heard log in one step, and speaks it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


@dataclass(frozen=True)
class ReadyResult:
    """One ready result: one ``final_answer`` item of a delegation's response. A Work with tasks gives several."""

    delegation_id: str
    item_id: str  # the final_answer message item's id, so a result unheard after a barge-in is requeued exactly
    speak: str  # the text to say, as the backend wrote it

    @property
    def unworded(self) -> str:
        """What is said when the result is not worded: its speak part."""
        return self.speak


@dataclass(frozen=True)
class WordingNotes:
    """What the wording must know besides the results."""

    told: Mapping[str, str]  # by delegation: what the caller was already told about it (a lead-in, a progress line)
    newer_requests: tuple[str, ...]  # requests still open that were made after these results' delegations


class ResultWording(Protocol):
    """Words every ready result into one line to say: ``writer`` (a model call) or ``as_written``.

    One call covers every result ready when it words. It drops no fact: when the model fails or stays silent, the
    results are said as written. It never repeats what ``notes.told`` says the caller already heard. A progress line
    holds no fact the caller needs: when it cannot be worded, nothing is said.
    """

    async def words(self, results: Sequence[ReadyResult], notes: WordingNotes) -> str:
        """The one line to say for ``results``, in the order given."""

    async def progress(self, update: str, told: str | None, latest: str | None) -> str:
        """A progress line for ``update``, the work note's line about one delegation, or none (empty)."""


class Ending(StrEnum):
    """How a delegation ended, as the reader reads it from the official events."""

    ANSWERED = "answered"  # its results came, a cancel's line for a newer request included; the response completed
    HANDED_OVER = "handed_over"  # only a commentary line, then completed: the named taker carries its work
    LATE = "late"  # a continuation that came after its deadline: completed with no item
    FAILED = "failed"  # response.failed (an error object), or incomplete for a reason such as max_output_tokens
    CANCELLED = "cancelled"  # closed: incomplete with status cancelled and no reason, at the session's close
    LOST = "lost"  # the stream ended with no terminal (DelegationEvents.on_delegation_lost)


class DisplaySink(Protocol):
    """Where the display part of each delegation goes: ``NullDisplay`` (Live) or ``RtviTaskCards`` (the UI bot).

    Plain calls that never wait. Nothing given here is ever spoken, and a display item never enters the talker's
    memory.
    """

    def started(self, delegation_id: str, request: str) -> None:
        """A delegation started; ``request`` is what the caller asked."""

    def progress(self, delegation_id: str, line: str) -> None:
        """The delegation's newest progress line."""

    def shown(self, delegation_id: str, item: dict) -> None:
        """A display item (the ``show`` work tool's ``mcp_call``), at once."""

    def answered(self, result: ReadyResult) -> None:
        """A ready result, at once, before it is said."""

    def spoken(self, delegation_ids: Sequence[str], words: str) -> None:
        """The one line said for these delegations' results, when it is handed to speech."""

    def ended(self, delegation_id: str, how: Ending) -> None:
        """The delegation ended."""
