# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How the frontend learns about its delegations: only from the official events it is handed, read by structure.

The delegation part hands every event to :meth:`DelegationReader.on_delegation_event` by a plain call that only records
or queues; the reader never reads a text to decide anything. Items are read from ``response.output_item.done``, since
the forwarded lifecycle snapshots carry no output; a backend call is also read at ``response.output_item.added``, where
it starts running, and its ``done`` gives the official status it ended with.

A ``previous_response_id`` on ``response.in_progress`` that names another delegation's response makes this delegation
the taker of that one's work; the IM names the taker on every join, so the reader never guesses. Each response has a
stream of its own, so a naming may arrive before the response it names, or after that response ended. While another
delegation may still name a response, the reader holds its newest progress line (a handover's line is the last item
before its response's end) and its ending.

Each ending is one :class:`Ending`. A failed, late or lost delegation says the voice's own line for it. A delegation
still waiting for the application after ``app_wait_seconds`` says the ``app_wait_line`` and stays watched, so a late
result is still said.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from examples.frontend_backend_live.cascade.lines import FixedLines
from examples.frontend_backend_live.cascade.ports import DisplaySink, Ending, ReadyResult
from examples.frontend_backend_live.cascade.worknote import WorkNote

TERMINALS = frozenset({"response.completed", "response.failed", "response.incomplete"})
# How many ended delegations are remembered, so their late events are ignored: the newest ones.
MAX_KEPT_ENDED = 32


def message_text(item: dict) -> str:
    """The text of a message item: its ``output_text`` parts, joined (a plain string content is taken whole)."""
    content = item.get("content")
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") for part in content or [] if isinstance(part, dict) and part.get("type") == "output_text"
    )


def is_progress(item: dict) -> bool:
    """Whether a message item is progress: phase ``commentary``. A message with no phase is a final answer."""
    return item.get("phase") == "commentary"


class Results(Protocol):
    """What the reader hands on, in order; the presenter's inbox implements it."""

    def ready(self, result: ReadyResult) -> None:
        """A ready result."""

    def notice(self, delegation_id: str | None, text: str) -> None:
        """A fixed line about a delegation."""

    def ended(self, delegation_id: str) -> None:
        """The delegation ended with no result to come: what the caller was told about it is forgotten."""

    def handed_over(self, delegation_id: str, taker: str) -> None:
        """The delegation's work went to ``taker``: what the caller was told goes with it."""


@dataclass
class Seen:
    """What the reader saw of one response."""

    id: str
    delegation_id: str
    finals: int = 0
    calls: int = 0
    running: bool = False  # it went in progress, where a taker names the response it takes over


@dataclass
class Watched:
    """One open delegation."""

    id: str
    response: Seen | None = None  # its current response
    tool_ran: bool = False  # a function call or a backend tool ran for it
    waits: bool = False  # it waits for the application's function outputs
    taker: str | None = None  # the delegation the IM named as carrying its work
    awaits_taker: bool = False  # its response ended, no taker named, while another delegation may still name it
    held_line: str | None = None  # a progress line waiting for its response's next event: a handover's line if an end


class DelegationReader:
    """Reads the official events of one call's delegations, by structure, and moves the work note and the presenter."""

    def __init__(
        self,
        worknote: WorkNote,
        results: Results,
        display: DisplaySink,
        lines: FixedLines,
        *,
        display_label: str,
        app_wait_seconds: float | None,
        observe: Callable[..., None],
        outcome_arrived: Callable[[str], None] = lambda delegation_id: None,
    ):
        """Wire what the events move.

        Args:
            worknote: The frontend's record of its open work.
            results: Where results, fixed lines and endings go (the presenter's inbox).
            display: Where the display part, progress, results and endings are shown at once.
            lines: The voice's fixed lines.
            display_label: The server label of the display item (the core's ``delegation.display_label``).
            app_wait_seconds: How long a delegation waits for the application before the app-wait line; ``None``: off.
            observe: ``observe(kind, **fields)`` hook for traces.
            outcome_arrived: Called with a delegation's id when a result of it arrives or it ends: what is said next
                about it makes its pending acknowledgement moot, so the acknowledgement is dropped.
        """
        self.worknote, self.results, self.display, self.lines = worknote, results, display, lines
        self.display_label, self.app_wait_seconds, self.observe = display_label, app_wait_seconds, observe
        self.outcome_arrived = outcome_arrived
        self.open: dict[str, Watched] = {}
        self.responses: dict[str, Seen] = {}  # every response of an open delegation, by id
        self.named: dict[str, str] = {}  # takers named for responses not seen yet: response id -> taker
        self.ended: dict[str, None] = {}  # the delegations that ended, newest last
        self.app_waits: dict[str, asyncio.TimerHandle] = {}
        self.closed = False

    def close(self) -> None:
        """The cascade closed: anything handed from now on is ignored, and every app wait stops."""
        self.closed = True
        for wait in self.app_waits.values():
            wait.cancel()
        self.app_waits.clear()

    def on_delegation_event(self, delegation_id: str, event: dict) -> None:
        """Take one event of a delegation, exactly as the core wrote it. A plain call: it only records or queues."""
        if self.closed or delegation_id in self.ended:
            return
        kind = event.get("type", "")
        response = event.get("response") if isinstance(event.get("response"), dict) else {}
        watched = self.open.get(delegation_id)
        if kind == "response.created":
            self._opened(watched or self._watch(delegation_id), response)
        elif watched is None or watched.response is None:
            return
        elif kind == "response.in_progress":
            self._runs(watched, response.get("previous_response_id"))
        elif kind == "response.output_item.added":
            self._release_line(watched)  # its response went on: the line it kept was progress
            self._started(watched, event.get("item") or {})
        elif kind == "response.output_item.done":
            self._release_line(watched)  # its response went on: the line it kept was progress
            self._item(watched, event.get("item") or {})
        elif kind in TERMINALS:
            watched.held_line = None  # the end came right after it: the handover's line (or one the end makes moot)
            self._terminal(watched, kind, response)

    def on_delegation_lost(self, delegation_id: str) -> None:
        """The delegation's stream ended with no ending: the ``lost`` line, and it ends."""
        if self.closed or delegation_id in self.ended:
            return
        self._finish(self.open.get(delegation_id) or Watched(delegation_id), Ending.LOST, self.lines.lost)

    def _watch(self, delegation_id: str) -> Watched:
        """A delegation's first response: the frontend's own delegation, or the application's typed run."""
        watched = self.open[delegation_id] = Watched(delegation_id)
        if delegation_id not in self.worknote:
            self.worknote.typed(delegation_id)
            self.display.started(delegation_id, "")
            self.observe("delegation.typed", delegation_id=delegation_id)
        return watched

    def _opened(self, watched: Watched, response: dict) -> None:
        response_id = response.get("id") or ""
        watched.response = self.responses[response_id] = Seen(response_id, watched.id)
        if watched.waits:  # a continuation: the application answered
            watched.waits = False
            self.worknote.waits_for_app(watched.id, False)
            self._stop_app_wait(watched.id)
        taker = self.named.pop(response_id, None)
        if taker is not None and taker != watched.id:  # its taker's naming came first, on the taker's own stream
            self._hand_over(watched, taker)
        if response.get("status") == "in_progress":  # a response that opens running (a one-shot request)
            self.worknote.running(watched.id)

    def _runs(self, watched: Watched, previous: str | None) -> None:
        self.worknote.running(watched.id)
        watched.response.running = True
        older = self.responses.get(previous or "")
        if older is None and previous:  # the response it names has not reached the reader yet
            self.named[previous] = watched.id
            while len(self.named) > MAX_KEPT_ENDED:
                del self.named[next(iter(self.named))]
        elif older is not None and older.delegation_id != watched.id and older.delegation_id in self.open:
            self._hand_over(self.open[older.delegation_id], watched.id)
        self._settle_awaiting()

    def _hand_over(self, older: Watched, taker: str) -> None:
        """The IM named ``taker`` as carrying ``older``'s work: its words, clock and what was told go to the taker."""
        if older.taker is not None:
            return
        older.taker, older.held_line = taker, None  # a line it kept was the handover's
        self.observe("delegation.taker", delegation_id=older.id, taker=taker)
        carrier = self.open.get(taker)
        if carrier is not None and older.tool_ran:  # a tool that ran for the work ran for the work the taker carries
            carrier.tool_ran = True
        self.worknote.hand_over(older.id, taker)
        self.results.handed_over(older.id, taker)
        if older.awaits_taker or older.waits:  # its response already ended (with the app's calls): the naming ends it
            self._completed(older)

    def _may_be_named(self, watched: Watched) -> bool:
        """Whether another open delegation has not started yet, so its start may still name ``watched``'s response."""
        return any(
            other is not watched and (other.response is None or not other.response.running)
            for other in self.open.values()
        )

    def _settle_awaiting(self) -> None:
        """Once no delegation can name a response any more: an ended one with no taker ends, a kept line is progress."""
        for watched in list(self.open.values()):
            if watched.id not in self.open or watched.taker is not None or self._may_be_named(watched):
                continue
            if watched.awaits_taker:
                self._completed(watched)
            else:
                self._release_line(watched)

    def _started(self, watched: Watched, item: dict) -> None:
        """An item starts: a backend call, other than the display, runs now."""
        if item.get("type") == "mcp_call" and item.get("server_label") != self.display_label:
            self.worknote.call_started(watched.id, watched.response.id, item.get("id", ""), item.get("name", ""))

    def _item(self, watched: Watched, item: dict) -> None:
        seen, delegation_id, item_type = watched.response, watched.id, item.get("type")
        if item_type == "message" and item.get("status") == "completed" and is_progress(item):
            self._progress(watched, message_text(item))
        elif item_type == "message" and item.get("status") == "completed":
            seen.finals += 1
            result = ReadyResult(delegation_id, item.get("id", ""), message_text(item))
            self.outcome_arrived(delegation_id)
            self.display.answered(result)
            self.worknote.answered(delegation_id)
            self.results.ready(result)
        elif item_type == "function_call":
            seen.calls += 1
            watched.tool_ran = True
            # The app runs it: it reads as running until the app's outputs come.
            self.worknote.call_started(delegation_id, seen.id, item.get("id", ""), item.get("name", ""))
        elif item_type == "mcp_call" and item.get("server_label") == self.display_label:
            if item.get("status") == "completed":  # a display that failed is none, and no tool step either
                self.display.shown(delegation_id, item)
        elif item_type == "mcp_call":
            watched.tool_ran = True
            self.worknote.call_ended(
                delegation_id, seen.id, item.get("id", ""), item.get("name", ""), item.get("status", "")
            )
            self.observe("delegation.backend_step", delegation_id=delegation_id, name=item.get("name", ""))

    def _progress(self, watched: Watched, text: str) -> None:
        """A progress line. Once its taker is named, the delegation's commentary is the handover's line: none.

        While another delegation may still name its response, the line waits for the response's next event: the
        handover's line is the last item before its response's end.
        """
        if watched.taker is not None:
            return
        if self._may_be_named(watched):
            watched.held_line = text
            return
        self._show_progress(watched, text)

    def _release_line(self, watched: Watched) -> None:
        """The line a delegation kept is progress after all: its response went on, or no delegation can name it."""
        line, watched.held_line = watched.held_line, None
        if line is not None and watched.taker is None:
            self._show_progress(watched, line)

    def _show_progress(self, watched: Watched, text: str) -> None:
        self.worknote.progress(watched.id, text)
        self.display.progress(watched.id, text)

    def _terminal(self, watched: Watched, kind: str, response: dict) -> None:
        """A response of the delegation ended: a wait for the application, or one of the endings."""
        seen, delegation_id = watched.response, watched.id
        if kind == "response.completed" and seen.calls:
            watched.waits = True
            self.worknote.waits_for_app(delegation_id, True)
            self._start_app_wait(delegation_id)
            self.observe("delegation.waits_for_app", delegation_id=delegation_id)
        elif kind == "response.completed":
            if seen.finals:
                self.worknote.ended(delegation_id)  # it leaves once nothing of it waits to be heard
            if watched.taker is None and self._may_be_named(watched):
                watched.awaits_taker = True  # the taker's naming may still come, on a stream of its own
            else:
                self._completed(watched)
        elif kind == "response.incomplete" and response.get("status") == "cancelled":
            self._finish(watched, Ending.CANCELLED)
        else:
            self._finish(watched, Ending.FAILED, self.lines.unclear if watched.tool_ran else self.lines.apology)

    def _completed(self, watched: Watched) -> None:
        """Its response completed with no calls, and its taker is named or none can be any more: how it ended.

        Answered with a final answer (a taker named for it carries the rest of its work); handed over with a taker and
        no answer; with neither, a continuation came too late.
        """
        if watched.response.finals:
            self._end(watched, Ending.ANSWERED)
        elif watched.taker is not None:
            self._end(watched, Ending.HANDED_OVER)  # its work went to the taker when the IM named it
        else:
            self._finish(watched, Ending.LATE, self.lines.unclear)

    def _finish(self, watched: Watched, how: Ending, line: str = "") -> None:
        """The delegation ended with no result to come: its line, if any, is said, and what it told is forgotten."""
        if line:
            self.results.notice(watched.id, line)
        self.worknote.ended(watched.id)  # it leaves once nothing of it waits to be heard
        self.results.ended(watched.id)
        self._end(watched, how)

    def _end(self, watched: Watched, how: Ending) -> None:
        """The delegation ended: it leaves the reader, and the display hears how."""
        delegation_id = watched.id
        self.open.pop(delegation_id, None)
        self._stop_app_wait(delegation_id)
        self.outcome_arrived(delegation_id)  # its ending, or its taker, speaks for it now
        self.responses = {key: seen for key, seen in self.responses.items() if seen.delegation_id != delegation_id}
        self.ended[delegation_id] = None
        while len(self.ended) > MAX_KEPT_ENDED:
            del self.ended[next(iter(self.ended))]
        self.display.ended(delegation_id, how)
        self.observe("delegation.ended", delegation_id=delegation_id, how=how.value)
        self._settle_awaiting()  # a delegation that could have named a taker is gone

    def _start_app_wait(self, delegation_id: str) -> None:
        self._stop_app_wait(delegation_id)
        if self.app_wait_seconds is not None:
            loop = asyncio.get_running_loop()
            self.app_waits[delegation_id] = loop.call_later(
                self.app_wait_seconds, self._app_wait_ran_out, delegation_id
            )

    def _stop_app_wait(self, delegation_id: str) -> None:
        wait = self.app_waits.pop(delegation_id, None)
        if wait is not None:
            wait.cancel()

    def _app_wait_ran_out(self, delegation_id: str) -> None:
        """The application has not answered the delegation's calls yet: say so once; the delegation stays watched."""
        self.app_waits.pop(delegation_id, None)
        watched = self.open.get(delegation_id)
        if watched is not None and watched.waits:
            self.observe("delegation.app_wait", delegation_id=delegation_id)
            self.results.notice(delegation_id, self.lines.app_wait_line)
