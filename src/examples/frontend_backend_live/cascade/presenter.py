# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The presenter's inbox: everything the voice says on its own, worded as it arrives and said together at idle.

Items are queued by plain calls, so backend work never waits on the voice. As soon as items are queued, even while the
caller speaks or a turn is decided, the presenter takes everything queued and words every result and commentary in one
``ResultWording`` call. At idle it says that line while its stamp holds: no item came since, and the notes (what the
caller was told about the batch's delegations, the requests opened after them) and the talker's memory, but for the
turns the talker decides, are as they were when it was worded. A caller turn alone keeps the stamp, so after a turn the
talker waited on, the line is said at once; a reply, a lead-in, an instruction, the application's context, a new
request or a new item moves it, and everything is worded again once no caller turn waits or is decided (a decision
moves it again): after a reply or a lead-in, while that line plays. A move cuts a wording under way short. A
delegation's end or handover is applied in its turn, in order with its results. A delivery that raises is requeued once;
a line nobody heard is requeued by the utterances. A result reads as answered in the work note until the caller
hears it, even in part. After a quiet spell, the quiet task words a short progress line about open work from the work
note, beside the inbox, and says it only if nothing moved while it was worded.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, NamedTuple

from loguru import logger

from examples.frontend_backend_live.cascade.lines import FixedLines
from examples.frontend_backend_live.cascade.memory import TalkerMemory
from examples.frontend_backend_live.cascade.ports import DisplaySink, ReadyResult, ResultWording, WordingNotes
from examples.frontend_backend_live.cascade.settings import CascadeSettings
from examples.frontend_backend_live.cascade.utterances import Line, LineKind, Utterances
from examples.frontend_backend_live.cascade.wording import RESULT_WORDINGS
from examples.frontend_backend_live.cascade.worknote import OpenWork, WorkNote
from examples.frontend_backend_live.cascade.writer.wording import Instruction
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
    answers: ClassVar[bool] = False  # a delegation's result: answered in the work note until it is heard

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


class Notice(Item):
    """A fixed line, said word for word."""

    line_kind = LineKind.NOTICE


# A mixed line takes the first of these kinds it carries.
LINE_KIND_ORDER = (LineKind.RESULT, LineKind.COMMENTARY, LineKind.NOTICE)


class Stamp(NamedTuple):
    """What a wording of a batch is worded on: its notes, and the talker memory's version."""

    notes: WordingNotes
    version: int


@dataclass(frozen=True)
class Turnover:
    """The end of a delegation, or its handover to ``taker``, applied in its turn."""

    delegation_id: str
    taker: str | None = None


class Presenter:
    """The inbox of one call: everything said on its own, worded as it arrives and said at idle (module docstring)."""

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
        self.utterances.requeue, self.utterances.heard = self.requeue, self.heard
        self._arrived = asyncio.Event()
        self._handling = False
        self._tasks: list[asyncio.Task] = []
        self._closed = False
        self._made = 0
        self._offered_at = float("-inf")  # when the quiet task last offered a progress line

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
        """The application's commentary (a fact to say); ``delegation_id``: a client delegation."""
        self._queue(Commentary(ReadyResult(delegation_id or "", self._own_id("commentary"), text)))

    def greeting(self, instructions: str) -> None:
        """The greeting, worded from ``instructions``, or the ``greeting`` line when they cannot be worded."""
        self._queue(Commentary(Instruction("", self._own_id("commentary"), instructions, self.lines.greeting)))

    def ended(self, delegation_id: str) -> None:
        """The delegation ended with nothing more to say: in its turn, what the caller was told is forgotten."""
        self._queue(Turnover(delegation_id))

    def handed_over(self, delegation_id: str, taker: str) -> None:
        """The delegation's work went to ``taker``: in its turn, what the caller was told goes with the taker."""
        self._queue(Turnover(delegation_id, taker))

    def requeue(self, line: Line) -> None:
        """Nobody heard ``line``: what it delivered goes back to the front of the inbox, to be said next."""
        items = [item for item in line.items if isinstance(item, Item)]
        self._back(items)
        if items:
            self.observe("answer.requeued", line=line.id, items=len(items))

    def heard(self, line: Line) -> None:
        """The caller heard ``line``, even in part: what it delivered is not said again."""
        self._done([item for item in line.items if isinstance(item, Item)])

    def _queue(self, entry: Item | Turnover) -> None:
        if self._closed:
            return
        self.inbox.append(entry)
        self._arrived.set()

    def _back(self, items: list[Item]) -> None:
        """``items`` go back to the front of the inbox; their results still wait in the work note."""
        for item in reversed(items):
            self.inbox.appendleft(item)
        if items:
            self._arrived.set()

    def _done(self, items: list[Item]) -> None:
        """``items`` are not said again: the work note counts each result among them as said."""
        for item in items:
            if item.answers and item.said.delegation_id:
                self.worknote.said(item.said.delegation_id)

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
        """Take everything queued as it arrives, word it at once, and deliver it together at idle."""
        while True:
            await self._arrived.wait()
            self._arrived.clear()
            if not self.inbox:
                continue
            batch = self._take()
            if not batch:
                continue
            self._handling = True
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
        """Word the batch, again with everything each time its stamp moves, and deliver it at idle.

        The batch grows in place, so a delivery that raises fails with every item it held.
        """
        while True:
            stamp = self._stamp(batch)
            text = await self._worded(batch, stamp)
            if text is not None and await self._held(batch, stamp):
                break
            while self.floor.turns_busy:  # the stamp or the inbox moved: word everything again once the turn is decided
                await self.floor.wait_change()
            batch[:] = self._take(batch)
        await self._deliver(batch, text)

    def _stamp(self, batch: list[Item]) -> Stamp:
        """What a wording of the batch is worded on now."""
        return Stamp(self._notes(batch), self.memory.version)

    async def _worded(self, batch: list[Item], stamp: Stamp) -> str | None:
        """The batch's line worded on ``stamp``; ``None`` when an item comes or the stamp moves before it is worded.

        A move cuts the wording under way short, so the batch is worded again at once, not after a call out of date.
        """
        wording = asyncio.ensure_future(self._word(batch, stamp.notes))
        try:
            return wording.result() if await self._held(batch, stamp, until=wording) else None
        finally:
            wording.cancel()

    async def _held(self, batch: list[Item], stamp: Stamp, until: asyncio.Future | None = None) -> bool:
        """Wait while no item comes and the batch's ``stamp`` holds: for ``until`` if given, else for idle.

        Return whether the stamp holds then. Each change of the floor and each item queued wakes it, so a stamp that
        moves is seen as it moves, not at idle.
        """

        def holds() -> bool:
            return not self.inbox and self._stamp(batch) == stamp

        def waiting() -> bool:
            return not until.done() if until is not None else not self.floor.idle

        while holds() and waiting():
            self._arrived.clear()  # the inbox is empty: an item queued from now on wakes the wait
            waits = [asyncio.ensure_future(self.floor.wait_change()), asyncio.ensure_future(self._arrived.wait())]
            try:
                await asyncio.wait([*waits, until] if until is not None else waits, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for wait in waits:
                    wait.cancel()
        return holds()

    def _notes(self, batch: list[Item]) -> WordingNotes:
        """What the wording of the batch knows: what the caller was told about its delegations, and newer requests."""
        worded = [item.said for item in batch if item.worded]
        delegations = list(dict.fromkeys(said.delegation_id for said in worded if said.delegation_id))
        told = {each: heard for each in delegations if (heard := self.utterances.told_text(each))}
        return WordingNotes(told, self.worknote.newer_requests(delegations))

    async def _word(self, batch: list[Item], notes: WordingNotes) -> str:
        """One line for the batch: the worded items in one wording call, at the first one's place; others as written."""
        worded = [item.said for item in batch if item.worded]
        words = await self.wording.words(worded, notes) if worded else ""
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
        if not text:
            self._done(batch)
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
        self._back(again)
        self._done([item for item in batch if item.tries])

    async def _quiet(self) -> None:
        """Offer a progress line when work has run through a quiet spell (:meth:`_progress`)."""
        after = self.settings.progress_after_seconds
        tick = max(0.05, min(1.0, after / 4))
        while True:
            await asyncio.sleep(tick)
            if self.busy or not self.floor.idle:
                continue
            entry = self.worknote.quiet_candidate(self.settings.progress_max_per_delegation)
            if entry is None:
                continue
            if self.floor.clock() - max(self.floor.idle_since, entry.created_at, self._offered_at) < after:
                continue
            try:
                await self._progress(entry)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"A progress line failed: {type(exc).__name__}: {exc}")
                self.observe("progress.failed", delegation_id=entry.id, error_type=type(exc).__name__)

    async def _progress(self, entry: OpenWork) -> None:
        """Word a progress line about ``entry`` from the work note; say it only if nothing moved while it was worded."""
        delegation_id = entry.id
        self._offered_at, version = self.floor.clock(), self.memory.version
        update = self.worknote.offer(delegation_id)
        self.observe("progress.offered", delegation_id=delegation_id)
        told = self.utterances.told_text(delegation_id)
        wording = asyncio.ensure_future(self.wording.progress(update, told, entry.progress))
        try:
            finished = await self._undisturbed(wording)
        finally:
            wording.cancel()
        if not (finished and self.memory.version == version and self.worknote.quiet(delegation_id)):
            self.observe("progress.dropped", delegation_id=delegation_id)  # its news stays for the next quiet spell
            return
        self.worknote.worded(delegation_id)
        text = wording.result()
        if not text:
            self.observe("progress.silent", delegation_id=delegation_id)
            return
        line = self.utterances.line(text, LineKind.STATUS, delegation_ids=(delegation_id,), again=False)
        self.display.spoken((delegation_id,), text)
        self.observe("progress.said", line=line.id, delegation_id=delegation_id)
        await self.utterances.hand_off(line, tells=(delegation_id,))

    async def _undisturbed(self, task: asyncio.Future) -> bool:
        """Wait for ``task`` while the floor stays idle and no item is queued; return whether it finished so.

        Any change of the floor and any item queued while it waits disturbs it, even one undone before it wakes: a
        result can be said and heard in that time.
        """
        if self.busy or not self.floor.idle:
            return False
        self._arrived.clear()  # set() already woke the delivery task for whatever was queued before now
        waits = [asyncio.ensure_future(self.floor.wait_change()), asyncio.ensure_future(self._arrived.wait())]
        try:
            done, _ = await asyncio.wait([task, *waits], return_when=asyncio.FIRST_COMPLETED)
        finally:
            for wait in waits:
                wait.cancel()
        return done == {task}
