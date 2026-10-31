# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The caller's turns: hear each committed turn, decide the turns not decided yet, then carry out the decision.

Turns are taken from the host's turn buffer by count, both when the caller stops speaking and when a turn is to be
decided, so either frame order notes each turn once. One decision runs at a time, for caller turns and instructions
alike. A new turn cuts off the caller decision in flight (Pipecat cancels it), so its words wait on and the next
decision covers them too: every caller turn is noted once and decided once. A new turn that began before the decision
did leaves no interruption to cut it off, so a decision that completes after the caller spoke again past the turns it
covers is discarded before it says or sends anything, and those turns wait on the same way. Once the caller is silent
again, they are decided with no new turn, unless a decision runs then: speech with no words may bring no decision of
its own. While turns wait or a decision runs, the floor is busy, so nothing the voice says on its own goes before the
reply. A reply or a lead-in never starts while the caller speaks: it waits for the floor, and a lead-in is dropped once
the caller starts speaking after its delegation was decided. A result that arrives while a decision on caller turns
runs cuts that decision off at once, since it read the work note before the result, and its turns wait the same way.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING

from loguru import logger

from examples.frontend_backend_live.cascade.agent import LiveAgent
from examples.frontend_backend_live.cascade.floor import Floor
from examples.frontend_backend_live.cascade.heard_log import HeardLog
from examples.frontend_backend_live.cascade.lines import FixedLines
from examples.frontend_backend_live.cascade.memory import (
    TalkerMemory,
    TurnBuffer,
    caller_turns,
    split_messages,
)
from examples.frontend_backend_live.cascade.ports import DisplaySink
from examples.frontend_backend_live.cascade.utterances import Line, LineKind, Utterances
from examples.frontend_backend_live.cascade.worknote import WorkNote
from examples.frontend_backend_live.models.decision import TalkerDecision

if TYPE_CHECKING:
    from live.ports import DelegationPort


class Acknowledgements:
    """The spoken acknowledgement of a delegation: the lead-in the talker wrote in the decision that delegated it.

    GPT-Live requests an acknowledgement for each delegation and lets its model word it; the talker words it in the
    same decision. A delegation with no lead-in says nothing. The lead-in is said after the part's delay (0: at once),
    once the caller is silent; until then it is pending. It is dropped when the caller started speaking after its
    delegation was decided, since the next decision answers their newer words. A pending one is also dropped when its
    delegation's result arrives or the delegation ends (its own line, or its taker, speaks for it), when the caller
    speaks, or when the application appends commentary or thinking, which drops every delegation's.
    """

    def __init__(self, utterances: Utterances, *, observe: Callable[..., None]):
        """Say through ``utterances``; nothing is pending."""
        self.utterances, self.observe = utterances, observe
        self.floor: Floor = utterances.floor
        self.pending: dict[str, asyncio.Task] = {}

    async def acknowledge(self, delegation_id: str, text: str, delay_ms: int, *, decided_at: int) -> None:
        """Acknowledge the delegation with the talker's lead-in ``text``, now or after ``delay_ms``; empty: none.

        ``decided_at`` is the floor's count of the caller's starts when the delegation was decided.
        """
        if not text.strip():
            return
        if self.floor.caller_starts != decided_at:
            self.observe("acknowledgement.dropped", delegation_id=delegation_id)
            return
        if delay_ms <= 0 and not self.floor.caller_speaking:
            await self.utterances.say_lead_in(delegation_id, text)
            return
        task = asyncio.create_task(self._later(delegation_id, delay_ms / 1000, text), name="acknowledgement")
        self.pending[delegation_id] = task

    async def _later(self, delegation_id: str, delay: float, text: str) -> None:
        await asyncio.sleep(delay)
        await self.floor.wait_caller_silent()
        self.pending.pop(delegation_id, None)
        await self.utterances.say_lead_in(delegation_id, text)

    def drop(self, delegation_id: str) -> None:
        """Drop the delegation's pending acknowledgement."""
        task = self.pending.pop(delegation_id, None)
        if task is not None:
            task.cancel()
            self.observe("acknowledgement.dropped", delegation_id=delegation_id)

    def drop_all(self) -> None:
        """Drop every pending acknowledgement."""
        for delegation_id in list(self.pending):
            self.drop(delegation_id)


class CallerTurns:
    """Hears each caller turn of one call, decides the turns not decided yet, and carries the decision out."""

    def __init__(
        self,
        agent: LiveAgent,
        memory: TalkerMemory,
        buffer: TurnBuffer,
        *,
        heard_log: HeardLog,
        utterances: Utterances,
        worknote: WorkNote,
        delegation: DelegationPort,
        acknowledgements: Acknowledgements,
        lines: FixedLines,
        display: DisplaySink,
        resume_after_noise: bool,
        observe: Callable[..., None],
    ):
        """Wire the talker and the parts a decision reaches.

        Args:
            agent: The models; turns use the talker and the normalizer.
            memory: The talker's memory.
            buffer: The host's turn buffer, which the caller's turns are taken from.
            heard_log: Notes the caller's words and builds each delegation's input.
            utterances: Says replies and lead-ins.
            worknote: The open work the talker sees.
            delegation: The session's delegation part each delegation is handed to.
            acknowledgements: The automatic acknowledgements.
            lines: The voice's fixed lines (the refusal).
            display: Hears each delegation start.
            resume_after_noise: Whether a turn with no words, or one the talker waits on, right after a cut says the cut
                line's unheard rest once.
            observe: ``observe(kind, **fields)`` hook for traces.
        """
        self.agent, self.memory, self.buffer = agent, memory, buffer
        self.heard_log, self.utterances, self.worknote, self.delegation = heard_log, utterances, worknote, delegation
        self.acknowledgements, self.lines, self.display = acknowledgements, lines, display
        self.resume_after_noise, self.observe = resume_after_noise, observe
        self.floor: Floor = utterances.floor
        # The action table: what each of the talker's actions does with a decided turn.
        self.actions = {"speak": self._speak, "delegate": self._delegate, "wait": self._wait}
        # The caller's turns heard and not decided yet, oldest first; the next decision covers them all.
        self.waiting: list[str] = []
        self.cuts: list[Line | None] = []  # beside each waiting turn, the line its words cut, if any
        # How many of the buffer's caller turns were taken. Those it holds when the call starts are history.
        self._taken = len(caller_turns(buffer.messages()))
        self._lock = asyncio.Lock()  # one decision at a time: caller turns and instructions
        self._deciding = False
        self.handoffs: set[asyncio.Task] = set()  # hand-offs still running, whoever waited for them
        self.redecisions: set[asyncio.Task] = set()  # decisions of the turns a dropped decision left waiting
        self._arrival: asyncio.Future | None = None  # set when a result arrives during a decision on caller turns
        worknote.on_answer = self.answer_arrived

    def hear(self) -> None:
        """Take each caller turn the buffer gained: note it, and keep it waiting for a decision. Nothing awaits."""
        for text in caller_turns(self.buffer.messages())[self._taken :]:
            if text.strip():
                self.heard_log.caller(text)
                self.memory.caller(text)
                self.waiting.append(text)
                self.cuts.append(self.utterances.caller_spoke())
        self.buffer.trim(0)  # every turn is taken: the buffer keeps its prompt only, so it stays bounded
        self._taken = len(caller_turns(self.buffer.messages()))
        self._busy()

    def _busy(self) -> None:
        self.floor.set_turns_busy(bool(self.waiting) or self._deciding)

    async def on_turn(self) -> None:
        """Hear the caller's new turns, then decide every turn not decided yet; with none, resume a cut line."""
        self.hear()
        if not self.waiting:
            await self._resume()
            return
        async with self._decision():
            if self.waiting:  # unless an instruction's decision covered them meanwhile
                await self._decide_turns()

    async def decide_instruction(self, text: str) -> None:
        """Decide an instruction append, together with any caller turn waiting (one decision)."""
        async with self._decision():
            await self._act(" ".join(self.waiting) or text)

    @contextlib.asynccontextmanager
    async def _decision(self) -> AsyncIterator[None]:
        """One decision at a time; the floor stays busy while it runs."""
        async with self._lock:
            self._deciding = True
            self._busy()
            try:
                yield
            finally:
                self._deciding = False
                self._busy()

    async def _decide_turns(self) -> None:
        await self._act(await self._normalize())

    async def _resume(self) -> None:
        """A turn with no words: right after a cut, say the cut line's unheard rest, once."""
        if self.resume_after_noise and await self.utterances.resume():
            self.observe("line.resumed")

    async def _normalize(self) -> str:
        """Return the waiting turns as one text, the newest in written form; its rewrite replaces it where noted."""
        newest = len(self.waiting) - 1  # a turn heard while the normalizer runs comes after it
        heard = self.waiting[newest]
        normalizer = self.agent.normalizer
        if normalizer is not None:
            _, history = split_messages(self.memory.messages())
            normalized, diagnostics = await normalizer.normalize(history, heard)
            self.observe("normalizer.completed", heard=heard, normalized=normalized, **diagnostics)
            if normalized != heard:
                self.memory.replace_caller(heard, normalized)
                self.heard_log.replace_caller(heard, normalized)
                self.waiting[newest] = normalized
        return " ".join(self.waiting)

    async def _act(self, text: str) -> None:
        """Run the talker's decision and carry it out; it covers the turns waiting when the talker reads the memory.

        A decision that covers turns is discarded when the caller spoke again after them (:meth:`_spoke_after`): it
        says, sends and records nothing, and those turns wait for the next decision, which covers them with the newer
        words, or comes once the caller is silent (:meth:`_redecide`). A result that arrives while the talker decides
        on turns cuts the decision off, and those turns wait the same way (:meth:`_decide_on_latest`). A turn heard
        while the talker decides an instruction alone is not in what it read, so it stays waiting for the next
        decision: its stop frame can come during a decision no interruption cuts off.
        """
        instructions, history = split_messages(self.memory.labeled())
        covered, started = len(self.waiting), self.floor.caller_starts
        open_work = bool(self.worknote.open())
        decision = await self._decide_on_latest(covered, instructions, history, work_open=open_work, turn=text)
        if decision is None:
            self.observe("talker.cut_off")
            self.floor.when_caller_silent(self._redecide)
            return
        if covered and self._spoke_after(covered, started):
            self.observe("talker.discarded", action=decision.action, **decision.diagnostics)
            self.floor.when_caller_silent(self._redecide)
            return
        cut = next((line for line in self.cuts[:covered] if line is not None), None)  # joined words: the first cut
        del self.waiting[:covered], self.cuts[:covered]
        self._busy()
        self.observe("talker.completed", action=decision.action, **decision.diagnostics)
        # An action the table does not know would be spoken, as a reply is.
        action = self.actions.get(decision.action, self._speak)
        await action(decision, text, cut)

    async def _decide_on_latest(
        self, covered: int, instructions: str, history: list[dict], *, work_open: bool, turn: str
    ) -> TalkerDecision | None:
        """The talker's decision on the work note as it is now; ``None`` when a result arrives before it is made.

        Only a decision that covers caller turns is cut off: the talker reads the work note when its call starts, so a
        result that arrives during the call would leave the decision on the note before it. Such a decision names
        ``turn``, the words of the turns it covers.
        """
        if not covered:
            return await self._decide(instructions, history, self.work_line(), work_open=work_open)
        self._arrival = arrival = asyncio.get_running_loop().create_future()
        work_line = self.work_line()
        deciding = asyncio.ensure_future(self._decide(instructions, history, work_line, work_open=work_open, turn=turn))
        try:
            await asyncio.wait([deciding, arrival], return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            deciding.cancel()
            raise
        finally:
            self._arrival = None
        if deciding.done():
            return deciding.result()
        deciding.cancel()
        return None

    def answer_arrived(self) -> None:
        """A result arrived: the decision on caller turns under way, if any, is cut off (:meth:`_decide_on_latest`)."""
        if self._arrival is not None and not self._arrival.done():
            self._arrival.set_result(None)

    def _spoke_after(self, covered: int, started: int) -> bool:
        """Whether the caller spoke again after a decision's ``covered`` turns; ``started``: the starts counted then.

        They speak now, they started since the talker read the turns, or a newer turn was heard. The host can commit a
        turn after the caller's next one began, so that start comes before the decision and only the caller still
        speaking shows it; a start that ended with no words before the decision began leaves nothing to join.
        """
        floor = self.floor
        return floor.caller_speaking or floor.caller_starts != started or len(self.waiting) > covered

    def _redecide(self) -> None:
        """The caller is silent after a dropped decision: decide the turns it left waiting, unless a decision runs then.

        Speech with no words may bring no decision of its own, so they would wait for the caller's next words. The
        decision runs in a task of its own, once the dropping decision has ended; a newer turn heard meanwhile drops it
        in turn, and the decision after it covers both.
        """
        task = asyncio.create_task(self._decide_dropped(), name="turns-redecided")
        self.redecisions.add(task)
        task.add_done_callback(self.redecisions.discard)

    async def _decide_dropped(self) -> None:
        if not self.waiting or self._deciding:
            return
        try:
            await self.on_turn()  # turns wait, so it decides them
        except Exception as exc:
            logger.exception(f"Live turn failed: {exc}")

    def work_line(self) -> str:
        """What the talker reads about open work (the work note's line)."""
        return self.worknote.line()

    async def _answer(self, text: str, kind: LineKind) -> None:
        """Say a line that answers the decision, once the caller is silent."""
        await self.floor.wait_caller_silent()
        await self.utterances.say(text, kind)

    async def _speak(self, decision: TalkerDecision, text: str, cut: Line | None) -> None:
        await self._answer(decision.speech, LineKind.REPLY)

    async def _wait(self, decision: TalkerDecision, text: str, cut: Line | None) -> None:
        """Say nothing for the turn; a line its words cut says its unheard rest once, as after a turn with no words."""
        self.observe("talker.waited")
        if cut is not None and self.resume_after_noise and await self.utterances.resume(cut):
            self.observe("line.resumed")

    async def _delegate(self, decision: TalkerDecision, text: str, cut: Line | None) -> None:
        reservation = self.worknote.reserve(text)  # before any await: no decision meanwhile reads "none"
        decided_at = self.floor.caller_starts
        handoff = asyncio.create_task(self._hand_off(reservation, text), name="delegation-hand-off")
        self.handoffs.add(handoff)
        handoff.add_done_callback(_retrieved)
        handoff.add_done_callback(self.handoffs.discard)
        try:
            delegation_id = await asyncio.shield(handoff)  # a cut-off turn leaves the hand-off running to its end
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"The delegation was refused: {type(exc).__name__}: {exc}")
            self.observe("delegation.refused", error_type=type(exc).__name__)
            await self._answer(self.lines.refusal, LineKind.NOTICE)
            return
        delay_ms = self.delegation.acknowledgement_delay_ms
        await self.acknowledgements.acknowledge(delegation_id, decision.speech.strip(), delay_ms, decided_at=decided_at)

    async def _hand_off(self, reservation: str | None, text: str) -> str:
        """Send the delegation with the heard log's input; confirm the reserved entry, or release it when refused.

        The input notes an answer that waits to be heard as it is built (:meth:`WorkNote.answer_ready`).
        """
        try:
            delegation_id = await self.heard_log.delegate(self.delegation, unheard=self.worknote.answer_ready())
        except BaseException:
            self.worknote.release(reservation)
            raise
        self.worknote.confirm(reservation, delegation_id)
        self.display.started(delegation_id, text)
        self.observe("delegation.handed", delegation_id=delegation_id)
        return delegation_id

    async def _decide(
        self, instructions: str, history: list[dict], work_in_progress: str, *, work_open: bool, turn: str = ""
    ) -> TalkerDecision:
        try:
            talker = self.agent.talker
            return await talker.decide(instructions, history, work_in_progress, work_open=work_open, turn=turn)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"Talker failed ({type(exc).__name__}: {exc}); recovering the turn")
            self.observe("talker.recovered", error_type=type(exc).__name__)
            return self.agent.talker.after_failure({"recovery": "talker_failed"}, work_open=work_open)


def _retrieved(task: asyncio.Task) -> None:
    """Read a finished hand-off's failure, which a turn cut off while it waited never reads."""
    if not task.cancelled():
        task.exception()
