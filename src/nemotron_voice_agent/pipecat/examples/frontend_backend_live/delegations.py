# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How the frontend learns about its delegations: from the official events it is handed, read by structure."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.ownership import WorkOwnership
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.work_provenance import (
    Attribution,
    Provenance,
    decode_provenance,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.worknote import WorkNote
from nemotron_voice_agent.pipecat.shared.delegation_contract import (
    APP_RESULT_WINDOW,
    TERMINALS,
    AppResultState,
    is_progress,
)

if TYPE_CHECKING:
    from nemotron_voice_agent.pipecat.examples.frontend_backend_live.display import DisplaySink
    from nemotron_voice_agent.pipecat.examples.frontend_backend_live.prompts import FixedLines
    from nemotron_voice_agent.pipecat.runtime.gateways.live.ports import DelegationRequestState

MAX_KEPT_ENDED = 32


@dataclass(frozen=True)
class ReadyResult:
    """One ready result: one ``final_answer`` item of a delegation's response."""

    delegation_id: str
    item_id: str
    speak: str
    attribution: Attribution | None = field(default=None, kw_only=True)
    routing_question: bool = field(default=False, kw_only=True)

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


@runtime_checkable
class WorkResults(Protocol):
    """Optional private branch bookkeeping; ordinary Responses listeners remain compatible."""

    def work_binding(self, branch_ref: str, delegation_id: str) -> None:
        """Associate a genuine source with a Work without moving past hearing."""

    def work_ended(self, branch_ref: str, delegation_id: str) -> None:
        """Forget only this branch's progress after its queued results settle."""

    def work_notice(self, delegation_id: str, text: str, attribution: Attribution) -> None:
        """Owe a notice about one exact Work, preserving its publication identity."""


@dataclass
class Seen:
    """What the reader saw of one response."""

    id: str
    delegation_id: str
    finals: int = 0
    calls: int = 0
    running: bool = False
    ended: bool = False
    app_branches: set[str] = field(default_factory=set)
    previous: str | None = None


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
    held_attribution: Attribution | None = None
    timed_out: bool = False
    waiting_branches: set[str] = field(default_factory=set)
    retired: Ending | None = None
    retired_replay: bool = False
    silent_terminal: bool = False


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
        self.requests: dict[str, str | None] = {}
        self.responses: dict[str, Seen] = {}
        self.named: dict[str, str] = {}
        self.ended: dict[str, None] = {}
        self.app_waits: dict[str, asyncio.TimerHandle] = {}
        self.client_tool_timeout_seconds = client_tool_timeout_seconds
        self.deadlines: dict[str, asyncio.TimerHandle] = {}
        self.closed = False
        self.ownership = WorkOwnership()
        self._ended_branches: set[str] = set()
        self._failed_branches: set[str] = set()
        self._branch_tools: set[str] = set()
        self._handled_frames: set[str] = set()
        self._handled_owners: dict[str, str] = {}
        self._replayed_settlements: set[str] = set()
        self.worknote.on_settled = self._prune_ownership
        self.worknote.on_confirm = self._confirmed

    def _confirmed(self, delegation_id: str) -> None:
        """Join acceptance to the existing reader lifecycle without reviving an already-ended source."""
        if any(delegation_id in sources for sources in self.ownership.request_sources.values()):
            self.worknote.sync_ownership(self.ownership)
        if self.closed or delegation_id in self.ended:
            self.worknote.ended(delegation_id)
        else:
            carrier = self.worknote.carrier(delegation_id)
            if carrier is not None and carrier.id == delegation_id:
                self.open.setdefault(delegation_id, Watched(delegation_id))

    def close(self) -> None:
        """Stop reading: later events are ignored, and every app wait stops."""
        self.closed = True
        for wait in (*self.app_waits.values(), *self.deadlines.values()):
            wait.cancel()
        self.app_waits.clear()
        self.deadlines.clear()
        self.requests.clear()

    def on_delegation_request(
        self, request_ref: str, state: DelegationRequestState, delegation_id: str | None = None
    ) -> None:
        """Retain actual in-flight submissions independently of parked reservations and first response events."""
        if self.closed:
            return
        if state == "lost" and delegation_id is not None:
            self.requests.pop(request_ref, None)
            self.requests = {key: source for key, source in self.requests.items() if source != delegation_id}
            watched = self.open.get(delegation_id)
            if watched is not None and watched.response is None:
                self.open.pop(delegation_id, None)
                self.worknote.ended(delegation_id)
            self._settle_awaiting()
            return
        if state in ("submitted", "accepted"):
            if delegation_id is not None and (
                delegation_id in self.ended
                or (watched := self.open.get(delegation_id)) is not None
                and watched.response is not None
            ):
                self.requests.pop(request_ref, None)
                self._settle_awaiting()
                return
            self.requests[request_ref] = delegation_id
            if state == "accepted" and delegation_id is not None:
                self.open.setdefault(delegation_id, Watched(delegation_id))
            return
        previous = self.requests.pop(request_ref, None)
        watched = self.open.get(previous) if previous is not None else None
        if watched is not None and watched.response is None and previous not in self.requests.values():
            self.open.pop(previous, None)
        self._settle_awaiting()

    def on_delegation_event(self, delegation_id: str, event: dict) -> None:
        """Take one event of a delegation as the core wrote it; a plain call that only records or queues."""
        self._event(delegation_id, event)

    def restore_delegation_event(self, delegation_id: str, event: dict) -> None:
        """Rebuild state from an event already handed to a previous attachment."""
        self._event(delegation_id, event)

    def _event(self, delegation_id: str, event: dict) -> None:
        if self.closed:
            return
        kind = event.get("type", "")
        response = event.get("response") if isinstance(event.get("response"), dict) else {}
        watched = self.open.get(delegation_id)
        previous_owner = self._handled_owners.get(delegation_id)
        try:
            actual_response_id = self._event_response_id(delegation_id, event)
        except ValueError:
            self.observe("delegation.provenance_refused", delegation_id=delegation_id)
            return
        observed_frame = actual_response_id in self._handled_frames
        if not self._restore_work(delegation_id, event, actual_response_id):
            return
        if kind == "response.created":
            self.requests = {key: source for key, source in self.requests.items() if source != delegation_id}
        frame = self.ownership.frames.get(actual_response_id)
        new_core_request = (
            kind in ("response.created", "response.in_progress")
            and not observed_frame
            and previous_owner is not None
            and frame is not None
            and frame[0] != previous_owner
        )
        if new_core_request:
            # Legacy late typed messages create a genuinely new Core request on
            # the existing transport stream. Its actual response proves that
            # identity; an old/duplicate lifecycle cannot reopen an ended job.
            self.ended.pop(delegation_id, None)
            self._stop_app_wait(delegation_id)
            self.worknote.reopen(delegation_id)
            self.open.pop(delegation_id, None)
            watched = self._watch(delegation_id)
            self.worknote.sync_ownership(self.ownership)
            if kind == "response.in_progress":
                self._opened(watched, response)
        elif delegation_id in self.ended:
            self._prune_ownership()
            return
        if kind == "response.created" and observed_frame:
            return
        if kind == "response.created":
            watched = self._watch(delegation_id)
            self.worknote.sync_ownership(self.ownership)
            self._opened(watched, response)
        elif watched is None or watched.response is None:
            return
        elif actual_response_id is not None and actual_response_id != watched.response.id:
            # A closing old MCP frame can finish after its genuine continuation
            # opened. Its historical items belong to that frame, and its terminal
            # must never close the newer response.
            old = self.responses.get(actual_response_id)
            if old is not None and old.delegation_id == delegation_id:
                if kind == "response.output_item.added":
                    self._started(watched, event.get("item") or {}, seen=old)
                elif kind == "response.output_item.done":
                    self._item(watched, event.get("item") or {}, seen=old)
                elif kind in TERMINALS:
                    old.ended = True
                    self._finish_retired(watched)
        elif kind == "response.in_progress":
            if not watched.response.running:
                self._runs(watched, response.get("previous_response_id"))
            elif watched.response.previous != response.get("previous_response_id"):
                self.observe("delegation.handover_refused", delegation_id=watched.id, reason="conflicting_predecessor")
        elif kind == "response.output_item.added":
            self._release_line(watched)
            self._started(watched, event.get("item") or {})
        elif kind == "response.output_item.done":
            self._release_line(watched)
            self._item(watched, event.get("item") or {})
        elif kind in TERMINALS:
            # A line right before the end is the handover's, or one the end makes moot.
            watched.held_line = None
            watched.held_attribution = None
            self._terminal(watched, kind, response, event)

    def _event_response_id(self, delegation_id: str, event: dict) -> str | None:
        response = event.get("response")
        if isinstance(response, dict) and isinstance(response.get("id"), str):
            return response["id"]
        if isinstance(event.get("response_id"), str):
            return event["response_id"]
        provenance = decode_provenance(event)
        if provenance is not None:
            frame = self.ownership.frames.get(provenance.response_id)
            if frame is not None and frame[1] == delegation_id:
                return provenance.response_id
        watched = self.open.get(delegation_id)
        return watched.response.id if watched is not None and watched.response is not None else None

    def restore_work_provenance(self, delegation_id: str, event: dict) -> None:
        """Restore private host facts idempotently, with no old speech or failure notice."""
        if self.closed:
            return
        try:
            actual_id = self._event_response_id(delegation_id, event)
        except ValueError:
            return
        self._restore_work(delegation_id, event, actual_id, replay=True)

    def _restore_work(self, delegation_id: str, event: dict, response_id: str | None, *, replay: bool = False) -> bool:
        """Apply a complete private snapshot before reading its ordinary event."""
        try:
            provenance = self.ownership.apply(delegation_id, event, response_id)
        except ValueError as error:
            self.observe("delegation.provenance_refused", delegation_id=delegation_id, reason=str(error))
            return False
        if provenance is None:
            return True
        self.worknote.sync_ownership(self.ownership)
        if isinstance(self.results, WorkResults):
            for owner, view in self.ownership.owners.items():
                if transport := self.ownership.transport(owner):
                    for branch in view.branches:
                        self.results.work_binding(branch, transport)
        if not replay and event.get("type") == "response.in_progress":
            response = event.get("response") or {}
            watched = self.open.get(delegation_id)
            previous = self.responses.get(response.get("previous_response_id", ""))
            if (
                watched is not None
                and watched.response is not None
                and watched.response.id == response_id
                and previous is not None
                and previous.delegation_id != delegation_id
                and previous.delegation_id in self.open
            ):
                self._hand_over(self.open[previous.delegation_id], delegation_id)
        self._owner_settlements(provenance, replay=replay)
        return True

    def _owner_settlements(self, provenance: Provenance, *, replay: bool) -> None:
        for settlement in provenance.owner_settlements:
            if replay:
                self._replayed_settlements.add(settlement.owner_ref)
                self._failed_branches.update(
                    row.branch_ref
                    for row in settlement.obligations
                    if row.branch_ref is not None and row.terminal_outcome == "failed"
                )
            if settlement.owner_ref == provenance.owner_ref:
                continue  # The actual current terminal still follows its ordinary event path.
            transport = self.ownership.transport(settlement.owner_ref)
            watched = self.open.get(transport) if transport is not None else None
            if watched is None:
                continue
            active_frame = self.ownership.frames.get(watched.response.id) if watched.response is not None else None
            if (
                transport == self.ownership.transport(provenance.owner_ref)
                or active_frame is not None
                and active_frame[0] != settlement.owner_ref
            ):
                # A legacy late request can reuse its old public transport DID.
                # Settling that old Core owner does not settle the new request.
                if not replay and settlement.disposition == "failed":
                    self._failed_work_notices(watched, owner_ref=settlement.owner_ref)
                self._settle_branch_ends(watched, owner_ref=settlement.owner_ref)
                continue
            self._retire(watched, settlement.owner_ref, replay=replay)

    def _retire(self, watched: Watched, owner_ref: str, *, replay: bool = False) -> None:
        """Retire logical ownership without discarding older frames' published tail."""
        view = self.ownership.owners.get(owner_ref)
        if view is None or view.disposition is None:
            return
        if watched.retired is None:
            watched.retired = Ending(view.disposition)
            watched.retired_replay = replay
            if not replay and view.disposition == "failed":
                self._failed_work_notices(watched, owner_ref=owner_ref)
            watched.held_line, watched.held_attribution = None, None
            self._stop_app_wait(watched.id)
            self.worknote.retire(watched.id)
        self._finish_retired(watched)

    def _finish_retired(self, watched: Watched) -> None:
        """End presentation only after every genuinely observed frame has drained."""
        if watched.retired is None or any(
            seen.delegation_id == watched.id and not seen.ended for seen in self.responses.values()
        ):
            return
        self.worknote.ended(watched.id)
        self._settle_branch_ends(watched)
        if not watched.retired_replay:
            self.results.ended(watched.id)
        self._end(watched, watched.retired, settle=not watched.retired_replay)

    def _attribution(self, watched: Watched, item: dict, seen: Seen | None = None) -> Attribution | None:
        seen = seen or watched.response
        if seen is None:
            return None
        return self.ownership.attribution(seen.id, item.get("id", ""))

    def _work_notice(self, watched: Watched, text: str, branch: str) -> None:
        binding = self.ownership.bindings.get(branch)
        if binding is not None and isinstance(self.results, WorkResults):
            self.results.work_notice(watched.id, text, Attribution(branch, binding.owner_ref, binding.revision))
        else:
            self.results.notice(watched.id, text)

    def _failed_work_notices(self, watched: Watched, *, owner_ref: str | None = None) -> None:
        view = self.ownership.owners.get(owner_ref) if owner_ref is not None else self.ownership.view(watched.id)
        if view is None:
            return
        for row in view.obligations:
            branch = row.branch_ref
            if branch is None or row.terminal_outcome != "failed" or branch in self._failed_branches:
                continue
            self._failed_branches.add(branch)
            self._work_notice(
                watched, self.lines.unclear if branch in self._branch_tools else self.lines.apology, branch
            )

    def on_delegation_lost(self, delegation_id: str) -> None:
        """End the delegation whose stream ended with no ending, with the ``lost`` line."""
        if self.closed or delegation_id in self.ended:
            return
        self.requests = {key: source for key, source in self.requests.items() if source != delegation_id}
        watched = self.open.get(delegation_id)
        if watched is not None and watched.retired is not None:
            for seen in self.responses.values():
                if seen.delegation_id == delegation_id:
                    seen.ended = True
            self._finish_retired(watched)
        else:
            self._finish(watched or Watched(delegation_id), Ending.LOST, self.lines.lost)

    def restore_final(self, delegation_id: str) -> None:
        """Count a previously forwarded final during private journal replay, without speaking it again."""
        watched = self.open.get(delegation_id)
        if not self.closed and watched is not None and watched.response is not None:
            watched.response.finals += 1

    def restore_progress(self, delegation_id: str, item: dict) -> None:
        """Restore forwarded commentary as private evidence, without replaying speech or display events."""
        if not self.closed and delegation_id in self.open and is_progress(item):
            watched = self.open[delegation_id]
            self.worknote.restore_progress(
                delegation_id, self.item_text(item), attribution=self._attribution(watched, item)
            )

    def restore_terminal(self, delegation_id: str, event: dict) -> None:
        """Close a previously forwarded terminal silently while the host replays its canonical journal."""
        watched = self.open.get(delegation_id)
        kind = event.get("type")
        if self.closed or watched is None or kind not in TERMINALS:
            return
        response = event.get("response") or {}
        actual_id = response.get("id")
        if actual_id is not None and watched.response is not None and actual_id != watched.response.id:
            old = self.responses.get(actual_id)
            if old is not None and old.delegation_id == delegation_id:
                old.ended = True
                if watched.retired is not None:
                    watched.retired_replay = True
                    self._finish_retired(watched)
            return
        if watched.response is not None:
            watched.response.ended = True
        if watched.retired is not None:
            watched.retired_replay = True
            self._finish_retired(watched)
            return
        if (
            kind == "response.completed"
            and watched.response is not None
            and not watched.response.calls
            and self._may_be_named(watched)
        ):
            watched.awaits_taker, watched.silent_terminal = True, True
            return
        if event.get("_nva_lost"):
            how = Ending.LOST
        elif kind == "response.incomplete" and response.get("status") == "cancelled":
            how = Ending.CANCELLED
        elif kind == "response.completed":
            how = self._completed_ending(watched)
        else:
            how = Ending.FAILED
        watched.held_line = None
        self.worknote.ended(delegation_id)
        self._end(watched, how, settle=False)
        while settled := next(
            (
                other
                for other in self.open.values()
                if (other.awaits_taker or other.timed_out)
                and other.taker is None
                and not self._may_be_named(other)
                and not self._composite_pending(other)
            ),
            None,
        ):
            ending = Ending.FAILED if settled.timed_out else self._completed_ending(settled)
            self.worknote.ended(settled.id)
            self._end(settled, ending, settle=False)

    def _watch(self, delegation_id: str) -> Watched:
        """Watch a delegation's first response: the frontend's own delegation, or the application's typed run."""
        watched = self.open.setdefault(delegation_id, Watched(delegation_id))
        if delegation_id not in self.worknote:
            self.worknote.typed(delegation_id)
            self.display.started(delegation_id, "")
            self.observe("delegation.typed", delegation_id=delegation_id)
        return watched

    def _opened(self, watched: Watched, response: dict) -> None:
        response_id = response.get("id") or ""
        watched.response = self.responses[response_id] = Seen(response_id, watched.id)
        self._handled_frame(watched)
        if watched.waits:
            self._clear_app_wait(watched)
        taker = self.named.get(response_id)
        # The taker's naming came first, on the taker's own stream.
        if taker is not None and taker != watched.id and self._hand_over(watched, taker):
            self.named.pop(response_id, None)
        if response.get("status") == "in_progress":
            self.worknote.running(watched.id)
        if (frame := self.ownership.frames.get(response_id)) is not None:
            self._retire(watched, frame[0], replay=frame[0] in self._replayed_settlements)

    def _runs(self, watched: Watched, previous: str | None) -> None:
        self._handled_frame(watched)
        self.worknote.running(watched.id)
        watched.response.running = True
        watched.response.previous = previous
        older = self.responses.get(previous or "")
        if older is None and previous:
            if previous in self.named and self.named[previous] != watched.id:
                self.observe("delegation.handover_refused", delegation_id=watched.id, reason="conflicting_taker")
            else:
                self.named[previous] = watched.id
        elif older is not None and older.delegation_id != watched.id and older.delegation_id in self.open:
            self._hand_over(self.open[older.delegation_id], watched.id)
        self._settle_awaiting()

    def _handled_frame(self, watched: Watched) -> None:
        if watched.response is not None and (frame := self.ownership.frames.get(watched.response.id)) is not None:
            self._handled_frames.add(watched.response.id)
            self._handled_owners[watched.id] = frame[0]

    def _hand_over(self, older: Watched, taker: str) -> bool:
        """Hand ``older``'s words, clock and what it told to ``taker``, which the IM named as carrying its work."""
        if older.id == taker:
            return False
        if older.taker is not None:
            current = self.worknote.carrier(older.taker)
            return older.taker == taker or current is not None and current is self.worknote.carrier(taker)
        destination = self.worknote.carrier(taker)
        composite = (
            self.ownership.is_composite(older.id)
            or self.ownership.is_composite(taker)
            or destination is not None
            and self.ownership.is_composite(destination.id)
        )
        if not composite:
            transferred = self.worknote.hand_over(older.id, taker)
            if transferred is None:
                self.observe("delegation.handover_refused", delegation_id=older.id, reason="unavailable_carrier")
                return False
            taker = transferred
        older.taker, older.held_line, older.held_attribution = taker, None, None
        carrier = self.open.get(taker)
        if carrier is not None and older.tool_ran:
            carrier.tool_ran = True
        if not composite:
            self.results.handed_over(older.id, taker)
        self.observe("delegation.taker", delegation_id=older.id, taker=taker)
        if (older.awaits_taker or older.waits) and older.retired is None:
            self._completed(older)
        return True

    def _may_be_named(self, watched: Watched) -> bool:
        """Return whether another open delegation has not started yet, so it may still name ``watched``'s response."""
        return any(source is None or source != watched.id for source in self.requests.values()) or any(
            other is not watched and (other.response is None or not (other.response.running or other.response.ended))
            for other in self.open.values()
        )

    def _settle_awaiting(self) -> None:
        """Settle what waited for a naming once no delegation can name a response any more.

        An ended one with no taker ends, one past the app's deadline fails, and a kept line is progress.
        """
        for watched in list(self.open.values()):
            if watched.retired is not None:
                continue
            if watched.id not in self.open or watched.taker is not None or self._may_be_named(watched):
                continue
            if self._composite_pending(watched):
                continue
            if watched.timed_out:
                self._finish(watched, Ending.FAILED, self.lines.unclear)
            elif watched.awaits_taker:
                self._completed(watched)
            else:
                self._release_line(watched)

    def _composite_pending(self, watched: Watched) -> bool:
        view = self.ownership.view(watched.id)
        return self.ownership.is_composite(watched.id) and view is not None and view.pending

    def _started(self, watched: Watched, item: dict, *, seen: Seen | None = None) -> None:
        """Note an item that starts: a backend call, other than the display, runs now."""
        seen = seen or watched.response
        if item.get("type") == "mcp_call" and item.get("server_label") != self.display_label:
            attribution = self._attribution(watched, item, seen)
            if attribution is not None:
                self._branch_tools.add(attribution.branch_ref)
            self.worknote.call_started(
                watched.id, seen.id, item.get("id", ""), item.get("name", ""), attribution=attribution
            )

    def _item(self, watched: Watched, item: dict, *, seen: Seen | None = None) -> None:
        seen, delegation_id, item_type = seen or watched.response, watched.id, item.get("type")
        attribution = self._attribution(watched, item, seen)
        if item_type == "message" and item.get("status") == "completed" and is_progress(item):
            self._progress(watched, self.item_text(item), attribution)
        elif item_type == "message" and item.get("status") == "completed":
            seen.finals += 1
            item_id = item.get("id", "")
            scope = self.ownership.items.get((seen.id, item_id))
            result = ReadyResult(
                delegation_id,
                item_id,
                self.item_text(item),
                attribution=attribution,
                routing_question=scope is not None and scope.routing_question,
            )
            self.outcome_arrived(delegation_id)
            self.display.answered(result)
            self.worknote.answered(delegation_id, attribution=attribution, item_id=result.item_id)
            self.results.ready(result)
        elif item_type == "function_call":
            seen.calls += 1
            if seen is watched.response:
                watched.tool_ran = True
            if attribution is not None:
                self._branch_tools.add(attribution.branch_ref)
                seen.app_branches.add(attribution.branch_ref)
            self.worknote.call_started(
                delegation_id, seen.id, item.get("id", ""), item.get("name", ""), attribution=attribution
            )
        elif item_type == "mcp_call" and item.get("server_label") == self.display_label:
            if item.get("status") == "completed":
                self.display.shown(delegation_id, item)
        elif item_type == "mcp_call":
            if seen is watched.response:
                watched.tool_ran = True
            self.worknote.call_ended(
                delegation_id,
                seen.id,
                item.get("id", ""),
                item.get("name", ""),
                item.get("status", ""),
                attribution=attribution,
            )
            self.observe("delegation.backend_step", delegation_id=delegation_id, name=item.get("name", ""))

    def _progress(self, watched: Watched, text: str, attribution: Attribution | None = None) -> None:
        """Note a progress line; hold it while the response may still be named, as a handover's line comes last."""
        if watched.taker is not None or watched.retired is not None:
            return
        if self._may_be_named(watched):
            watched.held_line = text
            watched.held_attribution = attribution
            return
        self._show_progress(watched, text, attribution)

    def _release_line(self, watched: Watched) -> None:
        """Show the line a delegation kept as progress: its response went on, or no delegation can name it."""
        line, watched.held_line = watched.held_line, None
        attribution, watched.held_attribution = watched.held_attribution, None
        if line is not None and watched.taker is None and watched.retired is None:
            self._show_progress(watched, line, attribution)

    def _show_progress(self, watched: Watched, text: str, attribution: Attribution | None = None) -> None:
        if attribution is not None and not self.worknote.publication_current(watched.id, attribution):
            return
        self.worknote.progress(watched.id, text, attribution=attribution)
        self.display.progress(watched.id, text)

    def _terminal(self, watched: Watched, kind: str, response: dict, event: dict) -> None:
        """Note the end of a delegation's response: a wait for the application, or one of the endings."""
        seen, delegation_id = watched.response, watched.id
        seen.ended = True
        if watched.retired is not None:
            self._finish_retired(watched)
            return
        if kind == "response.completed" and seen.calls:
            watched.waits = True
            watched.waiting_branches = set(seen.app_branches)
            self.worknote.waits_for_app(delegation_id, True, branch_refs=tuple(watched.waiting_branches))
            if APP_RESULT_WINDOW not in event:
                self._start_app_wait(delegation_id)
            elif (state := event[APP_RESULT_WINDOW]) is not None:
                self.on_app_result_window(delegation_id, seen.id, state)
            if watched.waits:
                self.observe("delegation.waits_for_app", delegation_id=delegation_id)
        elif kind == "response.completed":
            if watched.taker is None and self._may_be_named(watched):
                watched.awaits_taker = True
            else:
                self._completed(watched)
        elif kind == "response.incomplete" and response.get("status") == "cancelled":
            self._settle_branch_ends(watched)
            self._finish(watched, Ending.CANCELLED)
        elif self._transferred(watched, disposition=Ending.FAILED):
            if not seen.finals:
                self.results.notice(delegation_id, self.lines.unclear if watched.tool_ran else self.lines.apology)
            self._finish(watched, Ending.FAILED)
        elif self.ownership.is_composite(delegation_id):
            self._failed_work_notices(watched)
            self._settle_branch_ends(watched)
            self.worknote.ended(delegation_id)
            self.results.ended(delegation_id)
            self._end(watched, Ending.FAILED)
        else:
            self._finish(watched, Ending.FAILED, self.lines.unclear if watched.tool_ran else self.lines.apology)

    def _transferred(self, watched: Watched, *, disposition: Ending = Ending.HANDED_OVER) -> bool:
        """Check the response's exact Core owner for a fully settled transfer."""
        frame = self.ownership.frames.get(watched.response.id) if watched.response is not None else None
        view = self.ownership.owners.get(frame[0]) if frame is not None and frame[1] == watched.id else None
        return bool(
            view is not None
            and view.disposition in (None, disposition)
            and view.obligations
            and not view.pending
            and all(row.branch_ref is not None and row.state == "transferred" for row in view.obligations)
        )

    def _completed_ending(self, watched: Watched) -> Ending:
        """Classify a completed response without replaying its speech or ownership changes."""
        if watched.response is not None and watched.response.finals:
            return Ending.ANSWERED
        return Ending.HANDED_OVER if watched.taker is not None or self._transferred(watched) else Ending.LATE

    def _completed(self, watched: Watched) -> None:
        """End a delegation whose response completed with no calls: answered, handed over, or too late."""
        view = self.ownership.view(watched.id)
        self._settle_branch_ends(watched)
        if watched.silent_terminal:
            self.worknote.ended(watched.id)
            self._end(watched, self._completed_ending(watched), settle=False)
            return
        handed = self._transferred(watched)
        if view is not None and (self.ownership.is_composite(watched.id) or handed and not watched.response.finals):
            self.worknote.ended(watched.id)
            self.results.ended(watched.id)
            self._end(watched, Ending.HANDED_OVER if handed else Ending.ANSWERED)
            return
        if watched.response.finals:
            self.worknote.ended(watched.id)
            self._end(watched, Ending.ANSWERED)
        elif watched.taker is not None:
            self._end(watched, Ending.HANDED_OVER)
        elif (
            view is not None
            and view.disposition in (None, "answered")
            and len(view.branches) == 1
            and view.obligations
            and all(
                row.state == "settled"
                and row.terminal_outcome == "answered"
                and not (row.delivery_pending or row.publication_pending)
                for row in view.obligations
            )
        ):
            # A held final can be intentionally superseded by an exact continuation. The canonical
            # source is still answered; missing obsolete wording is not an unknown or failed action.
            self.worknote.ended(watched.id)
            self.results.ended(watched.id)
            self._end(watched, Ending.ANSWERED)
        else:
            self._finish(watched, Ending.LATE, self.lines.unclear)

    def _settle_branch_ends(self, watched: Watched, *, owner_ref: str | None = None) -> None:
        view = self.ownership.owners.get(owner_ref) if owner_ref is not None else self.ownership.view(watched.id)
        if view is None or not isinstance(self.results, WorkResults):
            return
        for branch in view.branches:
            rows = [row for row in view.obligations if row.branch_ref == branch]
            if (
                branch not in self._ended_branches
                and rows
                and all(
                    row.state == "settled" and not (row.delivery_pending or row.publication_pending) for row in rows
                )
            ):
                self._ended_branches.add(branch)
                self.results.work_ended(branch, watched.id)

    def _finish(self, watched: Watched, how: Ending, line: str = "") -> None:
        """End the delegation with no result to come: say its line, if any, and forget what it told."""
        if line:
            view = self.ownership.view(watched.id)
            if view is not None and len(view.branches) == 1:
                (branch,) = view.branches
                self._failed_branches.add(branch)
                self._work_notice(watched, line, branch)
            else:
                self.results.notice(watched.id, line)
        self.worknote.ended(watched.id)
        self._settle_branch_ends(watched)
        self.results.ended(watched.id)
        self._end(watched, how)

    def _end(self, watched: Watched, how: Ending, *, settle: bool = True) -> None:
        """Remove the ended delegation from the reader and show how it ended."""
        delegation_id = watched.id
        self.open.pop(delegation_id, None)
        self.requests = {key: source for key, source in self.requests.items() if source != delegation_id}
        self._stop_app_wait(delegation_id)
        self.outcome_arrived(delegation_id)
        self.responses = {key: seen for key, seen in self.responses.items() if seen.delegation_id != delegation_id}
        self.ended[delegation_id] = None
        while len(self.ended) > MAX_KEPT_ENDED:
            del self.ended[next(iter(self.ended))]
        self.display.ended(delegation_id, how)
        self.observe("delegation.ended", delegation_id=delegation_id, how=how.value)
        self._prune_ownership()
        if settle:
            self._settle_awaiting()

    def _prune_ownership(self) -> None:
        named = {}
        for key, taker in self.named.items():
            carrier = self.worknote.carrier(taker)
            current = carrier.id if carrier is not None else taker
            if current in self.open:
                named[key] = current
        self.named = named
        retained = {*self.open, *self.ended, *self.worknote.entries}
        known_frames = set(self.ownership.frames)
        active = {
            self.ownership.frames[watched.response.id][0]
            for watched in self.open.values()
            if watched.response is not None and watched.response.id in self.ownership.frames
        }
        responses = {
            key
            for key, seen in self.responses.items()
            if key not in self.ownership.frames
            or self.ownership.frames[key][0] in active
            or seen.delegation_id in self.open
            and not seen.ended
        }
        pending = self.worknote.pending_publications()
        responses.update(
            response_id
            for response_id, item_id in self.ownership.items
            if (self.ownership.frames[response_id][1], item_id) in pending
        )
        ready = {branch for branch, value in self.worknote.branches.items() if value.ready}
        self.ownership.prune(retained, ready, response_ids=responses, max_closed=MAX_KEPT_ENDED)
        self.responses = {
            key: seen for key, seen in self.responses.items() if key not in known_frames or key in self.ownership.frames
        }
        self.worknote.sync_ownership(self.ownership)
        self._handled_frames.intersection_update(self.ownership.frames)
        self._handled_owners = {key: value for key, value in self._handled_owners.items() if key in retained}
        self._replayed_settlements.intersection_update(self.ownership.owners)
        self._ended_branches.intersection_update(self.ownership.bindings)
        self._failed_branches.intersection_update(self.ownership.bindings)
        self._branch_tools.intersection_update(self.ownership.bindings)

    def on_app_result_window(self, delegation_id: str, response_id: str, state: AppResultState) -> None:
        """Start timers for this response's result window, or settle work that stopped before it opened."""
        watched = self.open.get(delegation_id)
        if self.closed or watched is None or not watched.waits or watched.response.id != response_id:
            return
        if state == "cancelled":
            self._clear_app_wait(watched)
            if not self._composite_pending(watched):
                self._finish(watched, Ending.CANCELLED)
        elif state == "closed":
            self._stop_app_wait(delegation_id)
            self._app_deadline(delegation_id)
        else:
            self._start_app_wait(delegation_id, state)

    def _clear_app_wait(self, watched: Watched) -> None:
        watched.waits = False
        self.worknote.waits_for_app(watched.id, False, branch_refs=tuple(watched.waiting_branches))
        watched.waiting_branches.clear()
        self._stop_app_wait(watched.id)

    def _start_app_wait(self, delegation_id: str, started: float | None = None) -> None:
        self._stop_app_wait(delegation_id)
        loop = asyncio.get_running_loop()
        start = loop.time() if started is None else started
        if self.app_wait_seconds is not None:
            self.app_waits[delegation_id] = loop.call_at(
                start + self.app_wait_seconds, self._app_wait_ran_out, delegation_id
            )
        if self.client_tool_timeout_seconds is not None:
            self.deadlines[delegation_id] = loop.call_at(
                start + self.client_tool_timeout_seconds, self._app_deadline, delegation_id
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
            if self.ownership.is_composite(delegation_id) and watched.waiting_branches:
                for branch in watched.waiting_branches:
                    self._work_notice(watched, self.lines.app_wait_line, branch)
            else:
                self.results.notice(delegation_id, self.lines.app_wait_line)

    def _app_deadline(self, delegation_id: str) -> None:
        """Fail a delegation still waiting at the core's deadline for the app's outputs, unless a taker carries it."""
        self.deadlines.pop(delegation_id, None)
        watched = self.open.get(delegation_id)
        if watched is not None and watched.waits:
            watched.timed_out = True
            if self.ownership.is_composite(delegation_id) and watched.waiting_branches:
                for branch in watched.waiting_branches - self._failed_branches:
                    self._failed_branches.add(branch)
                    self._work_notice(watched, self.lines.unclear, branch)
                return
            self._settle_awaiting()
