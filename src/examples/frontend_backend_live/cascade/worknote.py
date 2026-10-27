# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The work note: the frontend's record of its open delegations, and the line the talker reads about them.

Entries move only on the frontend's own calls and the official events the reader reads (``cascade/delegations.py``).
A request is reserved before it is sent, so no turn decided meanwhile reads "none" while the hand-off is in flight. An
entry reads "answer ready" while a result of it waits to be said, so the talker never answers a result about to be
said. Only a delegation part that reports events gets entries (``tracks``): no event ever ends a client delegation, so
in Live client mode the talker reads no work line at all.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from examples.frontend_backend_live.prompts import PromptSet

REQUEST_CHARS = 140


class WorkState(StrEnum):
    """Where an open delegation stands."""

    WAITING = "waiting"  # reserved or queued: waiting to start
    RUNNING = "running"


@dataclass
class OpenWork:
    """One open delegation, as the note knows it."""

    id: str  # the delegation id, or the reservation until the request is accepted
    request: str  # what the caller asked, in their words; after a handover every joined request, "older / newer"
    created_at: float
    state: WorkState = WorkState.WAITING
    started_at: float | None = None
    step: tuple[str, ...] = ()  # the names of the tool calls of its latest response that had some, in order
    step_response: str | None = None  # that response's id
    parts: tuple[tuple[int, str], ...] = ()  # the requests it carries, oldest first: (the order they were made, words)
    progress: str | None = None  # its newest progress line
    progress_said: bool = False  # whether the newest progress line was said
    lines_said: int = 0  # progress lines said for it (joined work shares the count)
    ready: int = 0  # its results waiting in the presenter's inbox
    ended: bool = False  # its delegation ended: it leaves once nothing of it waits to be said
    waits_for_app: bool = False
    reserved: bool = False  # its request is still being handed over: no delegation id yet


def humanize(tool_name: str) -> str:
    """A tool name as words: ``search_direct_flight`` -> ``search direct flight``."""
    return re.sub(r"[_\-]+", " ", tool_name).strip()


def shorten(request: str) -> str:
    """One request as the note quotes it: at most ``REQUEST_CHARS`` characters."""
    return request if len(request) <= REQUEST_CHARS else request[:REQUEST_CHARS] + "..."


class WorkNote:
    """The frontend's one record of its open work, and the line the talker reads from it."""

    def __init__(self, prompts: PromptSet, *, tracks: bool, clock: Callable[[], float], observe: Callable[..., None]):
        """Start with no entries.

        Args:
            prompts: The prompt set whose texts the note's line uses.
            tracks: Whether the delegation part reports each delegation's events (``DelegationPort.reports_events``).
            clock: The session's clock, which the entries' times are read on.
            observe: ``observe(kind, **fields)`` hook for traces.
        """
        self.prompts, self.tracks, self.clock, self.observe = prompts, tracks, clock, observe
        self.entries: dict[str, OpenWork] = {}  # the open delegations, oldest first
        self._made = 0  # how many entries were made: each entry's place in the order of requests

    def __contains__(self, delegation_id: str) -> bool:
        """Whether the delegation is open in the note."""
        return delegation_id in self.entries

    def reserve(self, request: str) -> str | None:
        """Note a request about to be delegated, waiting to start; return its reservation (``None``: not tracking)."""
        if not self.tracks:
            return None
        self._made += 1
        reservation = f"reserved-{self._made}"
        self.entries[reservation] = OpenWork(
            reservation, request, self.clock(), parts=((self._made, request),), reserved=True
        )
        return reservation

    def confirm(self, reservation: str | None, delegation_id: str) -> None:
        """The reserved request was accepted as ``delegation_id``: its entry carries that id, in its place."""
        entry = self.entries.get(reservation) if reservation else None
        if entry is None:
            return
        entry.id, entry.reserved = delegation_id, False
        self.entries = {(delegation_id if key == reservation else key): value for key, value in self.entries.items()}

    def release(self, reservation: str | None) -> None:
        """The reserved request was refused: its entry goes."""
        if reservation:
            self.entries.pop(reservation, None)

    def typed(self, delegation_id: str) -> None:
        """A delegation the frontend did not start appeared: the application's typed run."""
        if self.tracks and delegation_id not in self.entries:
            self._made += 1
            label = self.prompts.typed_request_label
            self.entries[delegation_id] = OpenWork(delegation_id, label, self.clock(), parts=((self._made, label),))

    def running(self, delegation_id: str) -> None:
        """The delegation's work runs; one that runs already keeps its start."""
        entry = self.entries.get(delegation_id)
        if entry is not None and entry.state != WorkState.RUNNING:
            entry.state, entry.started_at = WorkState.RUNNING, self.clock()

    def called(self, delegation_id: str, response_id: str | None, name: str) -> None:
        """A tool call of the delegation's response ``response_id`` was made: its name joins the step.

        The step lists every call of the latest response that had some, in order.
        """
        entry = self.entries.get(delegation_id)
        if entry is None:
            return
        if entry.step_response != response_id:
            entry.step, entry.step_response = (), response_id
        entry.step += (name,)

    def progress(self, delegation_id: str, line: str) -> None:
        """The delegation's newest progress line."""
        entry = self.entries.get(delegation_id)
        if entry is not None and line.strip():
            entry.progress, entry.progress_said = line.strip(), False

    def waits_for_app(self, delegation_id: str, waiting: bool) -> None:
        """Whether the delegation waits for the application's function outputs.

        Once the outputs came, the calls the app ran are done: the step goes, so the line never says one still runs.
        """
        entry = self.entries.get(delegation_id)
        if entry is not None:
            entry.waits_for_app = waiting
            if not waiting:
                entry.step, entry.step_response = (), None

    def answered(self, delegation_id: str) -> None:
        """A result of the delegation waits to be said."""
        entry = self.entries.get(delegation_id)
        if entry is not None:
            entry.ready += 1

    def said(self, delegation_id: str) -> None:
        """A result of the delegation was handed to speech; an ended delegation leaves once nothing else waits."""
        entry = self.entries.get(delegation_id)
        if entry is not None:
            entry.ready = max(0, entry.ready - 1)
            self._leave_if_done(entry)

    def ended(self, delegation_id: str) -> None:
        """The delegation ended: it leaves once nothing of it waits to be said."""
        entry = self.entries.get(delegation_id)
        if entry is not None:
            entry.ended = True
            self._leave_if_done(entry)

    def remove(self, delegation_id: str) -> None:
        """The delegation leaves the note at once."""
        self.entries.pop(delegation_id, None)

    def hand_over(self, delegation_id: str, taker: str) -> None:
        """The IM named ``taker`` as the delegation that carries this one's work: words, clocks and count go to it.

        The taker's entry reads "older words / newer words" and keeps the older clocks and progress-line count.
        """
        entry = self.entries.pop(delegation_id, None)
        carrier = self.entries.get(taker)
        if entry is None or carrier is None:
            return
        carrier.parts = tuple(sorted(entry.parts + carrier.parts))
        carrier.request = " / ".join(words for _, words in carrier.parts)
        carrier.created_at = min(carrier.created_at, entry.created_at)
        starts = [start for start in (carrier.started_at, entry.started_at) if start is not None]
        carrier.started_at = min(starts) if starts else None
        carrier.lines_said += entry.lines_said
        self.observe("worknote.joined", delegation_id=delegation_id, taker=taker)

    def _leave_if_done(self, entry: OpenWork) -> None:
        if entry.ended and entry.ready == 0:
            self.entries.pop(entry.id, None)

    def open(self) -> list[OpenWork]:
        """The delegations still open, oldest first."""
        return list(self.entries.values())

    def newer_requests(self, delegation_ids: list[str]) -> tuple[str, ...]:
        """What the caller asked in the requests still open that were made after the oldest of ``delegation_ids``.

        Results are worded knowing them, so an answer a newer request may change is said as such.
        """
        ids = list(self.entries)
        known = [ids.index(delegation_id) for delegation_id in delegation_ids if delegation_id in self.entries]
        if not known:
            return ()
        later = [self.entries[each] for each in ids[min(known) + 1 :] if each not in delegation_ids]
        return tuple(entry.request for entry in later if entry.ready == 0)

    def quiet_candidate(self, max_lines: int) -> OpenWork | None:
        """The oldest open delegation a quiet spell may speak about: sent, not said next, not waiting for the app."""
        for entry in self.entries.values():
            quiet = not (entry.reserved or entry.ready or entry.ended or entry.waits_for_app)
            if quiet and entry.lines_said < max_lines:
                return entry
        return None

    def offer_progress(self, delegation_id: str) -> str | None:
        """A line is said about the delegation after a quiet spell: its newest progress line not said yet, or ``None``.

        Either way the line counts against the delegation's cap.
        """
        entry = self.entries.get(delegation_id)
        if entry is None:
            return None
        entry.lines_said += 1
        if entry.progress and not entry.progress_said:
            entry.progress_said = True
            return entry.progress
        return None

    def line(self) -> str:
        """What the talker reads about open work: the note, the "none" line when tracking, or nothing."""
        open_entries = self.open()
        if open_entries:
            items = "\n".join(self._item(number, entry) for number, entry in enumerate(open_entries, 1))
            return self.prompts.work_in_progress.format(items=items)
        return self.prompts.no_work_in_progress if self.tracks else ""

    def _item(self, number: int, entry: OpenWork) -> str:
        """One entry of the note: its state, its request, and its newest progress line."""
        if entry.ready:
            state = self.prompts.answer_ready_label
        elif entry.state == WorkState.RUNNING:
            state = f"running for {round(self.clock() - (entry.started_at or self.clock()))} s"
            step = ", ".join(humanize(name) for name in entry.step if humanize(name))
            if step:
                state += f", now using {step}"
        else:
            state = "waiting to start"
        # Each joined request is shortened on its own, so a long older one never hides the newer words.
        request = " / ".join(shorten(words) for _, words in entry.parts) or shorten(entry.request)
        progress = f'; latest update: "{entry.progress}"' if entry.progress and not entry.ready else ""
        return f'{number}. {state}: "{request}"{progress}'
