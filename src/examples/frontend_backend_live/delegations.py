# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How the frontend learns about its delegations: from the official events it is handed, read by structure."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from examples.frontend_backend_live.worknote import WorkNote
from examples.shared.delegation_contract import TERMINALS, is_progress

if TYPE_CHECKING:
    from examples.frontend_backend_live.display import DisplaySink
    from examples.frontend_backend_live.prompts import FixedLines

MAX_KEPT_ENDED = 32


@dataclass(frozen=True)
class ReadyResult:
    """One ready result: one ``final_answer`` item of a delegation's response."""

    delegation_id: str
    item_id: str
    speak: str

    @property
    def unworded(self) -> str:
        """The text said when the result is not worded: its speak part."""
        return self.speak


class Ending(StrEnum):
    """How a delegation ended, as the reader reads it from the official events."""

    ANSWERED = "answered"
    HANDED_OVER = "handed_over"
    LATE = "late"
    FAILED = "failed"
    CANCELLED = "cancelled"
    LOST = "lost"


class Results(Protocol):
    """What the reader hands on, in order; the voice implements it."""

    def ready(self, result: ReadyResult) -> None:
        """Owe a ready result."""

    def notice(self, delegation_id: str | None, text: str) -> None:
        """Owe a fixed line about a delegation."""

    def ended(self, delegation_id: str) -> None:
        """Forget what the caller was told about a delegation that ended with no result to come."""

    def handed_over(self, delegation_id: str, taker: str) -> None:
        """Move what the caller was told about a delegation to ``taker``, which carries its work."""


@dataclass
class Seen:
    """What the reader saw of one response."""

    id: str
    delegation_id: str
    finals: int = 0
    calls: int = 0
    running: bool = False


@dataclass
class Watched:
    """One open delegation."""

    id: str
    response: Seen | None = None
    tool_ran: bool = False
    waits: bool = False
    taker: str | None = None
    awaits_taker: bool = False
    held_line: str | None = None
    timed_out: bool = False


class DelegationReader:
    """Reads the official events of one call's delegations, by structure, and moves the work note and the voice."""

    def __init__(
        self,
        worknote: WorkNote,
        results: Results,
        display: DisplaySink,
        lines: FixedLines,
        *,
        display_label: str,
        app_wait_seconds: float | None,
        item_text: Callable[[dict], str],
        observe: Callable[..., None],
        outcome_arrived: Callable[[str], None] = lambda delegation_id: None,
        client_tool_timeout_seconds: float | None = None,
    ):
        """Wire what the events move; ``outcome_arrived`` hears of each delegation whose result or end arrives."""
        self.worknote, self.results, self.display, self.lines = worknote, results, display, lines
        self.display_label, self.app_wait_seconds, self.observe = display_label, app_wait_seconds, observe
        self.item_text = item_text
        self.outcome_arrived = outcome_arrived
        self.open: dict[str, Watched] = {}
        self.responses: dict[str, Seen] = {}
        self.named: dict[str, str] = {}
        self.ended: dict[str, None] = {}
        self.app_waits: dict[str, asyncio.TimerHandle] = {}
        self.client_tool_timeout_seconds = client_tool_timeout_seconds
        self.deadlines: dict[str, asyncio.TimerHandle] = {}
        self.closed = False

    def close(self) -> None:
        """Stop reading: later events are ignored, and every app wait stops."""
        self.closed = True
        for wait in (*self.app_waits.values(), *self.deadlines.values()):
            wait.cancel()
        self.app_waits.clear()
        self.deadlines.clear()

    def on_delegation_event(self, delegation_id: str, event: dict) -> None:
        """Take one event of a delegation as the core wrote it; a plain call that only records or queues."""
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
            self._release_line(watched)
            self._started(watched, event.get("item") or {})
        elif kind == "response.output_item.done":
            self._release_line(watched)
            self._item(watched, event.get("item") or {})
        elif kind in TERMINALS:
            # A line right before the end is the handover's, or one the end makes moot.
            watched.held_line = None
            self._terminal(watched, kind, response)

    def on_delegation_lost(self, delegation_id: str) -> None:
        """End the delegation whose stream ended with no ending, with the ``lost`` line."""
        if self.closed or delegation_id in self.ended:
            return
        self._finish(self.open.get(delegation_id) or Watched(delegation_id), Ending.LOST, self.lines.lost)

    def _watch(self, delegation_id: str) -> Watched:
        """Watch a delegation's first response: the frontend's own delegation, or the application's typed run."""
        watched = self.open[delegation_id] = Watched(delegation_id)
        if delegation_id not in self.worknote:
            self.worknote.typed(delegation_id)
            self.display.started(delegation_id, "")
            self.observe("delegation.typed", delegation_id=delegation_id)
        return watched

    def _opened(self, watched: Watched, response: dict) -> None:
        response_id = response.get("id") or ""
        watched.response = self.responses[response_id] = Seen(response_id, watched.id)
        if watched.waits:
            watched.waits = False
            self.worknote.waits_for_app(watched.id, False)
            self._stop_app_wait(watched.id)
        taker = self.named.pop(response_id, None)
        # The taker's naming came first, on the taker's own stream.
        if taker is not None and taker != watched.id:
            self._hand_over(watched, taker)
        if response.get("status") == "in_progress":
            self.worknote.running(watched.id)

    def _runs(self, watched: Watched, previous: str | None) -> None:
        self.worknote.running(watched.id)
        watched.response.running = True
        older = self.responses.get(previous or "")
        if older is None and previous:
            self.named[previous] = watched.id
            while len(self.named) > MAX_KEPT_ENDED:
                del self.named[next(iter(self.named))]
        elif older is not None and older.delegation_id != watched.id and older.delegation_id in self.open:
            self._hand_over(self.open[older.delegation_id], watched.id)
        self._settle_awaiting()

    def _hand_over(self, older: Watched, taker: str) -> None:
        """Hand ``older``'s words, clock and what it told to ``taker``, which the IM named as carrying its work."""
        if older.taker is not None:
            return
        older.taker, older.held_line = taker, None
        self.observe("delegation.taker", delegation_id=older.id, taker=taker)
        carrier = self.open.get(taker)
        if carrier is not None and older.tool_ran:
            carrier.tool_ran = True
        self.worknote.hand_over(older.id, taker)
        self.results.handed_over(older.id, taker)
        if older.awaits_taker or older.waits:
            self._completed(older)

    def _may_be_named(self, watched: Watched) -> bool:
        """Return whether another open delegation has not started yet, so it may still name ``watched``'s response."""
        return any(
            other is not watched and (other.response is None or not other.response.running)
            for other in self.open.values()
        )

    def _settle_awaiting(self) -> None:
        """Settle what waited for a naming once no delegation can name a response any more.

        An ended one with no taker ends, one past the app's deadline fails, and a kept line is progress.
        """
        for watched in list(self.open.values()):
            if watched.id not in self.open or watched.taker is not None or self._may_be_named(watched):
                continue
            if watched.timed_out:
                self._finish(watched, Ending.FAILED, self.lines.unclear)
            elif watched.awaits_taker:
                self._completed(watched)
            else:
                self._release_line(watched)

    def _started(self, watched: Watched, item: dict) -> None:
        """Note an item that starts: a backend call, other than the display, runs now."""
        if item.get("type") == "mcp_call" and item.get("server_label") != self.display_label:
            self.worknote.call_started(watched.id, watched.response.id, item.get("id", ""), item.get("name", ""))

    def _item(self, watched: Watched, item: dict) -> None:
        seen, delegation_id, item_type = watched.response, watched.id, item.get("type")
        if item_type == "message" and item.get("status") == "completed" and is_progress(item):
            self._progress(watched, self.item_text(item))
        elif item_type == "message" and item.get("status") == "completed":
            seen.finals += 1
            result = ReadyResult(delegation_id, item.get("id", ""), self.item_text(item))
            self.outcome_arrived(delegation_id)
            self.display.answered(result)
            self.worknote.answered(delegation_id)
            self.results.ready(result)
        elif item_type == "function_call":
            seen.calls += 1
            watched.tool_ran = True
            self.worknote.call_started(delegation_id, seen.id, item.get("id", ""), item.get("name", ""))
        elif item_type == "mcp_call" and item.get("server_label") == self.display_label:
            if item.get("status") == "completed":
                self.display.shown(delegation_id, item)
        elif item_type == "mcp_call":
            watched.tool_ran = True
            self.worknote.call_ended(
                delegation_id, seen.id, item.get("id", ""), item.get("name", ""), item.get("status", "")
            )
            self.observe("delegation.backend_step", delegation_id=delegation_id, name=item.get("name", ""))

    def _progress(self, watched: Watched, text: str) -> None:
        """Note a progress line; hold it while the response may still be named, as a handover's line comes last."""
        if watched.taker is not None:
            return
        if self._may_be_named(watched):
            watched.held_line = text
            return
        self._show_progress(watched, text)

    def _release_line(self, watched: Watched) -> None:
        """Show the line a delegation kept as progress: its response went on, or no delegation can name it."""
        line, watched.held_line = watched.held_line, None
        if line is not None and watched.taker is None:
            self._show_progress(watched, line)

    def _show_progress(self, watched: Watched, text: str) -> None:
        self.worknote.progress(watched.id, text)
        self.display.progress(watched.id, text)

    def _terminal(self, watched: Watched, kind: str, response: dict) -> None:
        """Note the end of a delegation's response: a wait for the application, or one of the endings."""
        seen, delegation_id = watched.response, watched.id
        if kind == "response.completed" and seen.calls:
            watched.waits = True
            self.worknote.waits_for_app(delegation_id, True)
            self._start_app_wait(delegation_id)
            self.observe("delegation.waits_for_app", delegation_id=delegation_id)
        elif kind == "response.completed":
            if seen.finals:
                self.worknote.ended(delegation_id)
            if watched.taker is None and self._may_be_named(watched):
                watched.awaits_taker = True
            else:
                self._completed(watched)
        elif kind == "response.incomplete" and response.get("status") == "cancelled":
            self._finish(watched, Ending.CANCELLED)
        else:
            self._finish(watched, Ending.FAILED, self.lines.unclear if watched.tool_ran else self.lines.apology)

    def _completed(self, watched: Watched) -> None:
        """End a delegation whose response completed with no calls: answered, handed over, or too late."""
        if watched.response.finals:
            self._end(watched, Ending.ANSWERED)
        elif watched.taker is not None:
            self._end(watched, Ending.HANDED_OVER)
        else:
            self._finish(watched, Ending.LATE, self.lines.unclear)

    def _finish(self, watched: Watched, how: Ending, line: str = "") -> None:
        """End the delegation with no result to come: say its line, if any, and forget what it told."""
        if line:
            self.results.notice(watched.id, line)
        self.worknote.ended(watched.id)
        self.results.ended(watched.id)
        self._end(watched, how)

    def _end(self, watched: Watched, how: Ending) -> None:
        """Remove the ended delegation from the reader and show how it ended."""
        delegation_id = watched.id
        self.open.pop(delegation_id, None)
        self._stop_app_wait(delegation_id)
        self.outcome_arrived(delegation_id)
        self.responses = {key: seen for key, seen in self.responses.items() if seen.delegation_id != delegation_id}
        self.ended[delegation_id] = None
        while len(self.ended) > MAX_KEPT_ENDED:
            del self.ended[next(iter(self.ended))]
        self.display.ended(delegation_id, how)
        self.observe("delegation.ended", delegation_id=delegation_id, how=how.value)
        self._settle_awaiting()

    def _start_app_wait(self, delegation_id: str) -> None:
        self._stop_app_wait(delegation_id)
        loop = asyncio.get_running_loop()
        if self.app_wait_seconds is not None:
            self.app_waits[delegation_id] = loop.call_later(
                self.app_wait_seconds, self._app_wait_ran_out, delegation_id
            )
        if self.client_tool_timeout_seconds is not None:
            self.deadlines[delegation_id] = loop.call_later(
                self.client_tool_timeout_seconds, self._app_deadline, delegation_id
            )

    def _stop_app_wait(self, delegation_id: str) -> None:
        for waits in (self.app_waits, self.deadlines):
            wait = waits.pop(delegation_id, None)
            if wait is not None:
                wait.cancel()

    def _app_wait_ran_out(self, delegation_id: str) -> None:
        """Say the app-wait line once for a delegation still waiting for the application; it stays watched."""
        self.app_waits.pop(delegation_id, None)
        watched = self.open.get(delegation_id)
        if watched is not None and watched.waits:
            self.observe("delegation.app_wait", delegation_id=delegation_id)
            self.results.notice(delegation_id, self.lines.app_wait_line)

    def _app_deadline(self, delegation_id: str) -> None:
        """Fail a delegation still waiting at the core's deadline for the app's outputs, unless a taker carries it."""
        self.deadlines.pop(delegation_id, None)
        watched = self.open.get(delegation_id)
        if watched is not None and watched.waits:
            watched.timed_out = True
            self._settle_awaiting()
