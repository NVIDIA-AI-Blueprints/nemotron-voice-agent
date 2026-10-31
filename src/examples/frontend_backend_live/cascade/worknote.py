# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The work note: the frontend's record of its open delegations, and the line the talker reads about them.

Entries move only on the frontend's own calls and the official events the reader reads (``cascade/delegations.py``).
A request is reserved before it is sent, so no turn decided meanwhile reads "none" while the hand-off is in flight. An
entry reads as answered while a result of it waits to be heard, so the talker never requests or answers a result
about to be said. Only a delegation part that reports events gets entries (``tracks``): no event ever ends a client
delegation, so in Live client mode the talker reads no work line at all.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from examples.frontend_backend_live.prompts import PromptSet

REQUEST_CHARS = 140


class WorkState(StrEnum):
    """Where an open delegation stands."""

    WAITING = "waiting"  # reserved or queued: its work has not started
    RUNNING = "running"


@dataclass
class ToolCall:
    """One tool call of a delegation's latest response that had some."""

    name: str
    status: str | None = None  # the official status it ended with; ``None`` while it runs


@dataclass
class OpenWork:
    """One open delegation, as the note knows it."""

    id: str  # the delegation id, or the reservation until the request is accepted
    request: str  # what the caller asked, in their words; after a handover every joined request, "older / newer"
    created_at: float
    state: WorkState = WorkState.WAITING
    started_at: float | None = None
    calls: dict[str, ToolCall] = field(default_factory=dict)  # its latest response's tool calls, by item id, in order
    calls_response: str | None = None  # that response's id
    parts: tuple[tuple[int, str], ...] = ()  # the requests it carries, oldest first: (the order they were made, words)
    carries: tuple[str, ...] = ()  # the delegations whose work it took over, and those they took over
    progress: str | None = None  # its newest progress line
    news: bool = True  # something new for a progress line: its start, a backend step that ended, a new progress line
    lines_said: int = 0  # progress lines offered for it, said or not (joined work shares the count)
    ready: int = 0  # its results waiting to be heard: in the presenter's inbox, or on a line not settled yet
    ended: bool = False  # its delegation ended: it leaves once nothing of it waits to be heard
    waits_for_app: bool = False
    reserved: bool = False  # its request is still being handed over: no delegation id yet


def humanize(tool_name: str) -> str:
    """A tool name as words: underscores and hyphens become spaces."""
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
        self.on_answer: Callable[[], None] = lambda: None  # the caller turns hear of each result that arrives

    def __contains__(self, delegation_id: str) -> bool:
        """Whether the delegation is open in the note."""
        return delegation_id in self.entries

    def reserve(self, request: str) -> str | None:
        """Note a request about to be delegated, not yet started; return its reservation (``None``: not tracking)."""
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

    def call_started(self, delegation_id: str, response_id: str | None, item_id: str, name: str) -> None:
        """A tool call of the delegation's response ``response_id`` runs now.

        The note keeps every call of the latest response that had some, in order.
        """
        calls = self._calls(delegation_id, response_id)
        if calls is not None:
            calls[item_id] = ToolCall(name)

    def call_ended(self, delegation_id: str, response_id: str | None, item_id: str, name: str, status: str) -> None:
        """A tool call of the delegation's response ``response_id`` ended with the official ``status``."""
        calls = self._calls(delegation_id, response_id)
        if calls is not None:
            calls[item_id] = ToolCall(name, status)
            self.entries[delegation_id].news = True

    def _calls(self, delegation_id: str, response_id: str | None) -> dict[str, ToolCall] | None:
        """The calls the note keeps for the delegation, started anew for a response other than theirs."""
        entry = self.entries.get(delegation_id)
        if entry is None:
            return None
        if entry.calls_response != response_id:
            entry.calls, entry.calls_response = {}, response_id
        return entry.calls

    def progress(self, delegation_id: str, line: str) -> None:
        """The delegation's newest progress line."""
        entry = self.entries.get(delegation_id)
        if entry is not None and line.strip():
            entry.progress, entry.news = line.strip(), True

    def waits_for_app(self, delegation_id: str, waiting: bool) -> None:
        """Whether the delegation waits for the application's function outputs.

        Once the outputs came, the calls the app ran are done: the calls go, so the line never says one still runs.
        """
        entry = self.entries.get(delegation_id)
        if entry is not None:
            entry.waits_for_app = waiting
            if not waiting:
                entry.calls, entry.calls_response = {}, None

    def answered(self, delegation_id: str) -> None:
        """A result of the delegation waits to be said."""
        entry = self._entry(delegation_id)
        if entry is not None:
            entry.ready += 1
        self.on_answer()

    def said(self, delegation_id: str) -> None:
        """The caller heard a result of the delegation, or it is not said; an ended one leaves once none waits."""
        entry = self._entry(delegation_id)
        if entry is not None:
            entry.ready = max(0, entry.ready - 1)
            self._leave_if_done(entry)

    def _entry(self, delegation_id: str) -> OpenWork | None:
        """The delegation's entry, or the entry of the delegation that carries its work."""
        entry = self.entries.get(delegation_id)
        if entry is None:
            entry = next((each for each in self.entries.values() if delegation_id in each.carries), None)
        return entry

    def ended(self, delegation_id: str) -> None:
        """The delegation ended: it leaves once nothing of it waits to be heard."""
        entry = self.entries.get(delegation_id)
        if entry is not None:
            entry.ended = True
            self._leave_if_done(entry)

    def hand_over(self, delegation_id: str, taker: str) -> None:
        """The IM named ``taker`` as the delegation that carries this one's work: words, clocks and count go to it.

        The taker's entry reads "older words / newer words" and keeps the older clocks and progress-line count, and the
        results of the delegation that wait to be heard. Its ``carries`` gains the delegation and every delegation that
        one carried.
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
        carrier.ready += entry.ready
        carrier.carries += (entry.id, *entry.carries)
        self.observe("worknote.joined", delegation_id=delegation_id, taker=taker)

    def _leave_if_done(self, entry: OpenWork) -> None:
        if entry.ended and entry.ready == 0:
            self.entries.pop(entry.id, None)

    def open(self) -> list[OpenWork]:
        """The delegations still open, oldest first."""
        return list(self.entries.values())

    def answer_ready(self) -> bool:
        """Whether an answer of an open delegation waits to be heard: its entry reads as answered."""
        return any(entry.ready for entry in self.entries.values())

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
        """The oldest open delegation a quiet spell may speak about: quiet, with news, and under the cap."""
        for entry in self.entries.values():
            if self._quiet(entry) and entry.news and entry.lines_said < max_lines:
                return entry
        return None

    def quiet(self, delegation_id: str) -> bool:
        """Whether the delegation is open and quiet: sent, no answer ready, not ended, not waiting for the app."""
        entry = self.entries.get(delegation_id)
        return entry is not None and self._quiet(entry)

    @staticmethod
    def _quiet(entry: OpenWork) -> bool:
        return not (entry.reserved or entry.ready or entry.ended or entry.waits_for_app)

    def offer(self, delegation_id: str) -> str:
        """A progress line about the delegation is worded now: it counts against the cap.

        Returns the note's line with that entry alone, which the writer words.
        """
        entry = self.entries[delegation_id]
        entry.lines_said += 1
        return self.prompts.work_in_progress.format(items=self._item(1, entry))

    def worded(self, delegation_id: str) -> None:
        """A progress line about it was worded while nothing moved, said or left silent: its news is taken."""
        entry = self.entries.get(delegation_id)
        if entry is not None:
            entry.news = False

    def line(self) -> str:
        """What the talker reads about open work: the note, the "none" line when tracking, or nothing."""
        open_entries = self.open()
        if open_entries:
            items = "\n".join(self._item(number, entry) for number, entry in enumerate(open_entries, 1))
            return self.prompts.work_in_progress.format(items=items)
        return self.prompts.no_work_in_progress if self.tracks else ""

    def _item(self, number: int, entry: OpenWork) -> str:
        """One entry of the note, in the prompt set's words: its state, its request, and its newest progress line."""
        texts = self.prompts.work_texts
        if entry.ready:
            state = self.prompts.answer_ready_label
        elif entry.state == WorkState.RUNNING:
            state = texts.running.format(seconds=round(self.clock() - (entry.started_at or self.clock())))
            state += self._calls_text(entry)
        else:
            state = texts.waiting
        # Each joined request is shortened on its own, so a long older one never hides the newer words.
        request = " / ".join(shorten(words) for _, words in entry.parts) or shorten(entry.request)
        progress = texts.update.format(progress=entry.progress) if entry.progress and not entry.ready else ""
        return texts.item.format(number=number, state=state, request=request, update=progress)

    def _calls_text(self, entry: OpenWork) -> str:
        """The entry's calls, in the prompt set's words: those that ended, each with its status, then those running."""
        texts = self.prompts.work_texts
        named = [(humanize(call.name), call.status) for call in entry.calls.values() if humanize(call.name)]
        ended = [texts.ended_call.format(name=name, status=status) for name, status in named if status is not None]
        running = [name for name, status in named if status is None]
        text = texts.ended.format(steps=", ".join(ended)) if ended else ""
        return text + (texts.steps.format(steps=", ".join(running)) if running else "")
