# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The caller's turns: hear each committed turn, decide the turns not decided yet, then carry out the decision.

Turns are taken from the host's turn buffer by count, both when the caller stops speaking and when a turn is to be
decided, so either frame order notes each turn once. One decision runs at a time, for caller turns and instructions
alike. A new turn cuts off the caller decision in flight (Pipecat cancels it), so its words wait on and the next
decision covers them too: every caller turn is noted once and decided once. While turns wait or a decision runs, the
floor is busy, so nothing the voice says on its own goes before the reply.
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
    last_assistant_text,
    split_messages,
)
from examples.frontend_backend_live.cascade.ports import DisplaySink
from examples.frontend_backend_live.cascade.talker.repeat_guard import repeated_speech
from examples.frontend_backend_live.cascade.talker.router import TurnRouter
from examples.frontend_backend_live.cascade.utterances import LineKind, Utterances
from examples.frontend_backend_live.cascade.worknote import WorkNote
from examples.frontend_backend_live.cascade.writer.guards import safe_lead_in
from examples.frontend_backend_live.config_manager.schema import LiveConfig
from examples.frontend_backend_live.models.decision import TalkerDecision

if TYPE_CHECKING:
    from live.ports import DelegationPort


class Acknowledgements:
    """The automatic spoken acknowledgement of a delegation whose talker decision brought no lead-in.

    The fixed ``lead_in`` line is said after the part's delay (0: at once; an empty line: none). A pending one is
    dropped when its delegation's result arrives or the delegation ends (its own line, or its taker, speaks for it),
    when the caller speaks, or when the application appends commentary or thinking, which drops every delegation's.
    """

    def __init__(self, utterances: Utterances, lines: FixedLines, *, observe: Callable[..., None]):
        """Say through ``utterances``; nothing is pending."""
        self.utterances, self.lines, self.observe = utterances, lines, observe
        self.pending: dict[str, asyncio.Task] = {}

    async def acknowledge(self, delegation_id: str, delay_ms: int) -> None:
        """Acknowledge the delegation with the ``lead_in`` line, now or after ``delay_ms``."""
        text = self.lines.lead_in
        if not text.strip():
            return
        if delay_ms <= 0:
            await self.utterances.say_lead_in(delegation_id, text)
            return
        task = asyncio.create_task(self._later(delegation_id, delay_ms / 1000, text), name="acknowledgement")
        self.pending[delegation_id] = task

    async def _later(self, delegation_id: str, delay: float, text: str) -> None:
        await asyncio.sleep(delay)
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
        config: LiveConfig,
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
            config: The validated example configuration (the guards, the router, the recovery switch).
            memory: The talker's memory.
            buffer: The host's turn buffer, which the caller's turns are taken from.
            heard_log: Notes the caller's words and builds each delegation's input.
            utterances: Says replies and lead-ins.
            worknote: The open work the talker sees.
            delegation: The session's delegation part each delegation is handed to.
            acknowledgements: The automatic acknowledgements.
            lines: The voice's fixed lines (the refusal).
            display: Hears each delegation start.
            resume_after_noise: Whether a turn with no words right after a cut says the unheard rest once.
            observe: ``observe(kind, **fields)`` hook for traces.
        """
        self.agent, self.config, self.memory, self.buffer = agent, config, memory, buffer
        self.heard_log, self.utterances, self.worknote, self.delegation = heard_log, utterances, worknote, delegation
        self.acknowledgements, self.lines, self.display = acknowledgements, lines, display
        self.resume_after_noise, self.observe = resume_after_noise, observe
        self.floor: Floor = utterances.floor
        self.router = (
            TurnRouter(config.guards.continuers)
            if config.guards.turn_router or config.guards.turn_router_shadow
            else None
        )
        # The action table: what each of the talker's actions does with a decided turn.
        self.actions = {"speak": self._speak, "delegate": self._delegate, "wait": self._wait}
        # The caller's turns heard and not decided yet, oldest first; the next decision covers them all.
        self.waiting: list[str] = []
        # How many of the buffer's caller turns were taken. Those it holds when the call starts are history.
        self._taken = len(caller_turns(buffer.messages()))
        self._lock = asyncio.Lock()  # one decision at a time: caller turns and instructions
        self._deciding = False
        self.handoffs: set[asyncio.Task] = set()  # hand-offs still running, whoever waited for them

    def hear(self) -> None:
        """Take each caller turn the buffer gained: note it, and keep it waiting for a decision. Nothing awaits."""
        for text in caller_turns(self.buffer.messages())[self._taken :]:
            if text.strip():
                self.heard_log.caller(text)
                self.memory.caller(text)
                self.waiting.append(text)
                self.utterances.caller_spoke()
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
        text = " ".join(self.waiting)
        if self.router and self._dropped_by_router(text):
            return
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
        if normalizer is not None and normalizer.wanted(heard):
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

        A turn heard while the talker decides is not in what it read, so it stays waiting for the next decision: its
        stop frame can come during a decision no interruption cuts off, such as an instruction's.
        """
        instructions, history = split_messages(self.memory.messages())
        covered = len(self.waiting)
        open_work = bool(self.worknote.open())
        decision = await self._decide(instructions, history, self.work_line(), work_open=open_work)
        del self.waiting[:covered]
        self._busy()
        if not open_work:
            decision = self._apply_repeat_guard(decision, text, history)
        self.observe("talker.completed", action=decision.action, **decision.diagnostics)
        # An action the table does not know would be spoken, as a reply is.
        action = self.actions.get(decision.action, self._speak)
        await action(decision, text)

    def work_line(self) -> str:
        """What the talker reads about open work (the work note's line)."""
        return self.worknote.line()

    async def _speak(self, decision: TalkerDecision, text: str) -> None:
        await self.utterances.say(decision.speech, LineKind.REPLY)

    async def _wait(self, decision: TalkerDecision, text: str) -> None:
        self.observe("talker.waited")

    async def _delegate(self, decision: TalkerDecision, text: str) -> None:
        reservation = self.worknote.reserve(text)  # before any await: no decision meanwhile reads "none"
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
            await self.utterances.say(self.lines.refusal, LineKind.NOTICE)
            return
        speech = safe_lead_in(decision.speech) if decision.speech else ""
        if speech != decision.speech:
            self.observe("talker.lead_in_replaced", original=decision.speech)
        if speech:
            await self.utterances.say_lead_in(delegation_id, speech)
        else:
            await self.acknowledgements.acknowledge(delegation_id, self.delegation.acknowledgement_delay_ms)

    async def _hand_off(self, reservation: str | None, text: str) -> str:
        """Send the delegation with the heard log's input; confirm the reserved entry, or release it when refused."""
        try:
            delegation_id = await self.heard_log.delegate(self.delegation)
        except BaseException:
            self.worknote.release(reservation)
            raise
        self.worknote.confirm(reservation, delegation_id)
        self.display.started(delegation_id, text)
        self.observe("delegation.handed", delegation_id=delegation_id)
        return delegation_id

    def _dropped_by_router(self, text: str) -> bool:
        verdict = self.router.verdict(text, last_assistant_text(self.memory.messages()))
        if verdict.keep:
            return False
        if self.config.guards.turn_router:
            self.observe("router.dropped", text=text, reason=verdict.reason)
            self.waiting.clear()
            return True
        self.observe("router.would_drop", text=text, reason=verdict.reason)
        return False

    async def _decide(
        self, instructions: str, history: list[dict], work_in_progress: str, *, work_open: bool
    ) -> TalkerDecision:
        try:
            return await self.agent.talker.decide(instructions, history, work_in_progress, work_open=work_open)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"Talker failed ({type(exc).__name__}: {exc}); recovering the turn")
            self.observe("talker.recovered", error_type=type(exc).__name__)
            return self.agent.talker.after_failure({"recovery": "talker_failed"}, work_open=work_open)

    def _apply_repeat_guard(self, decision: TalkerDecision, text: str, history: list[dict]) -> TalkerDecision:
        if not self.config.guards.repeat_guard or decision.action != "speak":
            return decision
        similarity = repeated_speech(decision.speech, text, history)
        if not similarity:
            return decision
        self.observe("talker.repeat_replaced", similarity=similarity)
        return TalkerDecision("delegate", "", {**decision.diagnostics, "recovery": "repeat_guard"})


def _retrieved(task: asyncio.Task) -> None:
    """Read a finished hand-off's failure, which a turn cut off while it waited never reads."""
    if not task.cancelled():
        task.exception()
