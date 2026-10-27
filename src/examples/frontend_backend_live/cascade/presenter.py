# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The presenter's inbox: everything the voice says on its own, worded and said together when the caller is idle.

Items are queued by plain calls, so backend work never waits on the voice. When the floor lets it prepare, the
presenter takes everything queued and words every result and commentary in one ``ResultWording`` call; if the talker's
memory moved or more items came before the floor is idle, it words again with everything. A delegation's end or
handover is applied in its turn, in order with its results. A delivery that raises is requeued once; a line nobody
heard is requeued by the utterances.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar

from loguru import logger

from examples.frontend_backend_live.cascade.lines import FixedLines
from examples.frontend_backend_live.cascade.memory import TalkerMemory
from examples.frontend_backend_live.cascade.ports import DisplaySink, ReadyResult, ResultWording, WordingNotes
from examples.frontend_backend_live.cascade.settings import CascadeSettings
from examples.frontend_backend_live.cascade.utterances import Line, LineKind, Utterances
from examples.frontend_backend_live.cascade.wording import RESULT_WORDINGS
from examples.frontend_backend_live.cascade.worknote import WorkNote
from examples.frontend_backend_live.tables import pick

if TYPE_CHECKING:
    from examples.frontend_backend_live.cascade.agent import LiveAgent


@dataclass(frozen=True)
class Item:
    """One thing the inbox holds to say: its delegation (``""`` for the session), its item id, and its text."""

    said: ReadyResult
    tries: int = 0  # deliveries of it that raised

    line_kind: ClassVar[LineKind]  # the kind of a line it is said in (the first of a mixed line's)
    worded: ClassVar[bool] = False  # worded with the batch's other worded items, or said as written
    answers: ClassVar[bool] = False  # a delegation's result: "said next" in the work note until it is said

    def record(self, memory: TalkerMemory, line: Line) -> tuple[str, ...]:
        """Write what it adds to the talker's memory with ``line``; return the delegations it tells the caller about."""
        return ()


class Result(Item):
    """A ready result: its speak part enters the talker's memory with the line that says it."""

    line_kind = LineKind.RESULT
    worded = True
    answers = True

    def record(self, memory: TalkerMemory, line: Line) -> tuple[str, ...]:
        """The result's speak part, as reference data."""
        memory.result(line.id, self.said.speak)
        return ()


class Commentary(Item):
    """The application's commentary (a fact to say) or the greeting: it enters the memory as application context."""

    line_kind = LineKind.COMMENTARY
    worded = True

    def record(self, memory: TalkerMemory, line: Line) -> tuple[str, ...]:
        """The commentary, as the application's context."""
        memory.app_context(self.said.speak, line=line.id)
        return ()


class Status(Item):
    """A progress line about a delegation, said after a quiet spell."""

    line_kind = LineKind.STATUS

    def record(self, memory: TalkerMemory, line: Line) -> tuple[str, ...]:
        """Nothing in memory besides the line; the delegation's ``told`` gains it."""
        return (self.said.delegation_id,)


class Notice(Item):
    """A fixed line, said word for word."""

    line_kind = LineKind.NOTICE


# A mixed line takes the first of these kinds it carries.
LINE_KIND_ORDER = (LineKind.RESULT, LineKind.COMMENTARY, LineKind.STATUS, LineKind.NOTICE)


@dataclass(frozen=True)
class Turnover:
    """The end of a delegation, or its handover to ``taker``, applied in its turn."""

    delegation_id: str
    taker: str | None = None


class Presenter:
    """The inbox of one call: everything said on its own, worded and said together at idle (module docstring)."""

    def __init__(
        self,
        wording: ResultWording,
        memory: TalkerMemory,
        utterances: Utterances,
        worknote: WorkNote,
        display: DisplaySink,
        lines: FixedLines,
        settings: CascadeSettings,
        *,
        observe: Callable[..., None],
    ):
        """Wire the wording and the records a delivery writes; nothing is presented until :meth:`start`."""
        self.wording, self.memory, self.utterances = wording, memory, utterances
        self.floor = utterances.floor
        self.worknote, self.display, self.lines, self.settings = worknote, display, lines, settings
        self.observe = observe
        self.inbox: deque[Item | Turnover] = deque()
        self.utterances.requeue = self.requeue
        self._arrived = asyncio.Event()
        self._handling = False
        self._said: list[str] = []  # the delegations the delivery under way marked said
        self._tasks: list[asyncio.Task] = []
        self._closed = False
        self._made = 0

    @classmethod
    def from_settings(
        cls,
        settings: CascadeSettings,
        *,
        agent: LiveAgent,
        memory: TalkerMemory,
        utterances: Utterances,
        worknote: WorkNote,
        display: DisplaySink,
        lines: FixedLines,
        observe: Callable[..., None],
    ) -> Presenter:
        """Build the presenter with the result wording the settings name (``result_wording``)."""
        wording = pick(RESULT_WORDINGS, settings.result_wording, "result wording")(agent, memory, observe)
        return cls(wording, memory, utterances, worknote, display, lines, settings, observe=observe)

    @property
    def busy(self) -> bool:
        """Whether anything waits in the inbox, or a delivery is under way."""
        return self._handling or any(isinstance(entry, Item) for entry in self.inbox)

    def ready(self, result: ReadyResult) -> None:
        """A ready result (a ``final_answer`` item of a delegation)."""
        self._queue(Result(result))

    def notice(self, delegation_id: str | None, text: str) -> None:
        """A fixed line, said word for word; an empty line says nothing."""
        if text.strip():
            self._queue(Notice(ReadyResult(delegation_id or "", self._own_id("notice"), text)))

    def commentary(self, text: str, delegation_id: str | None = None) -> None:
        """The application's commentary (a fact to say), or the greeting; ``delegation_id``: a client delegation."""
        self._queue(Commentary(ReadyResult(delegation_id or "", self._own_id("commentary"), text)))

    def ended(self, delegation_id: str) -> None:
        """The delegation ended with nothing more to say: in its turn, what the caller was told is forgotten."""
        self._queue(Turnover(delegation_id))

    def handed_over(self, delegation_id: str, taker: str) -> None:
        """The delegation's work went to ``taker``: in its turn, what the caller was told goes with the taker."""
        self._queue(Turnover(delegation_id, taker))

    def requeue(self, line: Line) -> None:
        """Nobody heard ``line``: what it delivered goes back to the front of the inbox, to be said next."""
        items = [item for item in line.items if isinstance(item, Item)]
        self._back(items, items)
        if items:
            self.observe("answer.requeued", line=line.id, items=len(items))

    def _queue(self, entry: Item | Turnover) -> None:
        if self._closed:
            return
        self.inbox.append(entry)
        self._arrived.set()

    def _back(self, items: list[Item], said: list[Item]) -> None:
        """``items`` go back to the front of the inbox; the results of ``said`` wait to be said again."""
        for item in reversed(items):
            self.inbox.appendleft(item)
        for item in said:
            if item.answers and item.said.delegation_id:
                self.worknote.answered(item.said.delegation_id)
        if items:
            self._arrived.set()

    def _own_id(self, kind: str) -> str:
        self._made += 1
        return f"{kind}_{self._made}"

    def start(self) -> None:
        """Start the delivery task and the quiet-spell task. Once only."""
        if not self._tasks and not self._closed:
            self._tasks = [asyncio.create_task(self._run(), name="result-presenter")]
            if self.settings.progress_after_seconds is not None:
                self._tasks.append(asyncio.create_task(self._quiet(), name="delegation-progress"))

    async def close(self) -> None:
        """Stop both tasks, cutting the delivery under way, and drop the inbox. Safe twice; later items are ignored."""
        self._closed = True
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self.inbox.clear()

    async def _run(self) -> None:
        """Deliver everything queued, together, each time the floor allows."""
        while True:
            await self._arrived.wait()
            self._arrived.clear()
            if not self.inbox:
                continue
            await self.floor.wait_may_prepare()
            batch = self._take()
            if not batch:
                continue
            self._handling, self._said = True, []
            try:
                await self._present(batch)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._failed(batch, exc)
            finally:
                self._handling = False

    def _take(self, batch: list[Item] | None = None) -> list[Item]:
        """Everything queued, oldest first, after ``batch``; ends and handovers are applied in their turn.

        An end or a handover of a delegation whose worded item the batch already holds needs nothing: delivering the
        item uses what the caller was told about it, and forgets it.
        """
        batch = list(batch or [])
        while self.inbox:
            entry = self.inbox.popleft()
            if isinstance(entry, Item):
                batch.append(entry)
            elif any(item.worded and item.said.delegation_id == entry.delegation_id for item in batch):
                continue
            elif entry.taker is None:
                self.utterances.forget(entry.delegation_id)
            else:
                self.utterances.carry_told(entry.delegation_id, entry.taker)
        return batch

    async def _present(self, batch: list[Item]) -> None:
        """Word the batch, wait for idle (wording again if anything moved), then deliver it."""
        while True:
            version = self.memory.version
            text = await self._word(batch)
            await self.floor.wait_idle()
            if self.memory.version == version and not self.inbox:
                break
            batch = self._take(batch)  # the memory or the inbox moved while it was worded: word everything again
        await self._deliver(batch, text)

    async def _word(self, batch: list[Item]) -> str:
        """One line for the batch: the worded items in one wording call, at the first one's place; others as written."""
        worded = [item.said for item in batch if item.worded]
        words = ""
        if worded:
            delegations = list(dict.fromkeys(said.delegation_id for said in worded if said.delegation_id))
            told = {each: heard for each in delegations if (heard := self.utterances.told_text(each))}
            notes = WordingNotes(told, self.worknote.newer_requests(delegations))
            words = await self.wording.words(worded, notes)
        parts: list[str] = []
        placed = False
        for item in batch:
            if not item.worded:
                parts.append(item.said.speak)
            elif not placed:
                parts.append(words)
                placed = True
        return " ".join(part.strip() for part in parts if part.strip())

    async def _deliver(self, batch: list[Item], text: str) -> None:
        """In one step, write the records and hand the line to speech."""
        kinds = {type(item).line_kind for item in batch}
        kind = next(each for each in LINE_KIND_ORDER if each in kinds)
        ids = tuple(dict.fromkeys(item.said.delegation_id for item in batch if item.said.delegation_id))
        results = any(item.answers for item in batch)
        line = self.utterances.line(text, kind, delegation_ids=ids, items=batch, results=results)
        for item in batch:
            delegation_id = item.said.delegation_id
            if item.worded and delegation_id:
                self.utterances.forget(delegation_id)  # what the caller was told was worded into this line
            if item.answers and delegation_id:
                self.worknote.said(delegation_id)
                self._said.append(delegation_id)
        if not text:
            return
        tells = [each for item in batch for each in item.record(self.memory, line) if each]
        self.display.spoken(ids, text)
        self.observe("answer.said", line=line.id, items=len(batch), delegations=list(ids))
        await self.utterances.hand_off(line, tells=tells)

    def _failed(self, batch: list[Item], exc: Exception) -> None:
        """A delivery raised: its items go back once; a second failure drops them."""
        logger.warning(f"Delivering {len(batch)} item(s) failed: {type(exc).__name__}: {exc}")
        self.observe("answer.failed", items=len(batch), error_type=type(exc).__name__)
        again = [replace(item, tries=item.tries + 1) for item in batch if item.tries == 0]
        said = [item for item in again if item.said.delegation_id in self._said]
        self._back(again, said)

    async def _quiet(self) -> None:
        """Offer a progress line when work has run through a quiet spell."""
        after = self.settings.progress_after_seconds
        tick = max(0.05, min(1.0, after / 4))
        while True:
            await asyncio.sleep(tick)
            if self.busy or not self.floor.idle:
                continue
            entry = self.worknote.quiet_candidate(self.settings.progress_max_per_delegation)
            if entry is None or self.floor.clock() - max(self.floor.idle_since, entry.created_at) < after:
                continue
            line = self.worknote.offer_progress(entry.id) or self.lines.still_working
            self.observe("progress.offered", delegation_id=entry.id)
            if line.strip():
                self._queue(Status(ReadyResult(entry.id, self._own_id("status"), line)))
