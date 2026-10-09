# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One call's orchestration: decide each turn, delegate, phrase the answer and deliver it.

This class knows nothing about Pipecat frames or the wire protocol. It reads and writes a :class:`Conversation`
and speaks through an injected ``say`` coroutine, so the same logic runs in tests and in the voice pipeline.

Behavior:

* Every turn gets a reply or a delegation (``never_silent``); a failed talker call delegates.
* A delegation's lead-in never asserts an outcome or asks a question (``safe_lead_in``), and a ``speak``
  that nearly repeats an earlier reply becomes a delegation (repeat guard).
* A backend result enters the history in time order. The phrasing request for that answer carries what the
  caller was already told, and only that request does.
* Every delegation's answer is spoken, even after the caller said "okay" in the meantime. Phrasing that fails or
  stays silent still speaks the verified text. An answer that a newer request in progress changes is spoken as
  being updated, not as final.
* The talker sees the work still in progress. A turn that only asks for an update is answered from it, and a
  turn that corrects work in flight replaces it (see the worker). Work that runs silently for a while gets a
  short status line.
* The delegate receives only the voice turns since its previous delegation, never backend results.
* An optional turn router drops acknowledgment-only turns (or, in shadow mode, only traces them).

Two delegation modes exist. In ``responses`` mode a server-owned backend does the work. In ``client`` mode the
client owns the work: the session only announces each delegation, and the client reports back with commentary.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from loguru import logger

from examples.frontend_backend_live.cascade.agent import LiveAgent
from examples.frontend_backend_live.cascade.application_input import ApplicationInput
from examples.frontend_backend_live.cascade.commentary.guards import plain_speech, safe_lead_in
from examples.frontend_backend_live.cascade.conversation import (
    Conversation,
    last_assistant_text,
    last_caller_text,
    split_messages,
)
from examples.frontend_backend_live.cascade.talker.repeat_guard import repeated_speech
from examples.frontend_backend_live.cascade.talker.router import TurnRouter
from examples.frontend_backend_live.common.history import BACKEND_RESULT_PREFIX
from examples.frontend_backend_live.config_manager.schema import LiveConfig
from examples.frontend_backend_live.delegation.coordinator import DelegationCoordinator, Observe, log_observation
from examples.frontend_backend_live.delegation.ledger import STATUS_LINES, asks_for_status
from examples.frontend_backend_live.delegation.tasks import CLIENT, RESPONSES, BackendTask, DelegatedTask
from examples.frontend_backend_live.models.decision import TalkerDecision

Say = Callable[[str], Awaitable[None]]
ANSWER_HOLD_SECONDS = 10.0

__all__ = ["ANSWER_HOLD_SECONDS", "CLIENT", "RESPONSES", "CascadeSession", "log_observation"]


class CascadeSession:
    """Runs the cascade frontend for one call: the talker, the commentary writer and the guards.

    Delegated work belongs to the :class:`DelegationCoordinator`; this class decides when to delegate and phrases
    what comes back.
    """

    def __init__(
        self,
        agent: LiveAgent,
        config: LiveConfig,
        conversation: Conversation,
        *,
        backend_task: BackendTask | None,
        tools,
        say: Say,
        mode: str = RESPONSES,
        on_event: Callable[[DelegatedTask, dict], Awaitable[None]] | None = None,
        on_client_delegation: Callable[[str], Awaitable[None]] | None = None,
        interrupt: Callable[[], Awaitable[None]] | None = None,
        observe: Observe = log_observation,
    ):
        """Wire the agent, conversation and speech output.

        Args:
            agent: The talker, commentary writer and thinker.
            config: The validated example configuration.
            conversation: The shared message list.
            backend_task: The model, instructions and tool schemas every backend round uses (``None`` in
                ``client`` mode).
            tools: A ``ToolExecutor`` for the thinker's function calls (unused in ``client`` mode).
            say: Awaited with the text to speak.
            mode: ``responses`` (server-owned backend) or ``client`` (the client owns delegated work).
            on_event: Optional awaited hook with ``(task, raw_event)`` for every backend event.
            on_client_delegation: Awaited with the id of each delegation announced in ``client`` mode.
            interrupt: Awaited to cut off current speech (instruction appends).
            observe: Optional ``observe(kind, **fields)`` hook for traces.
        """
        self.agent, self.config, self.conversation = agent, config, conversation
        self.say_text, self.observe, self.mode, self.interrupt = say, observe, mode, interrupt
        self.router = (
            TurnRouter(config.guards.continuers)
            if config.guards.turn_router or config.guards.turn_router_shadow
            else None
        )
        self.coordinator = DelegationCoordinator(
            agent.thinker,
            config,
            backend_task=backend_task,
            tools=tools,
            sink=self,
            mode=mode,
            on_event=on_event,
            on_client_delegation=on_client_delegation,
            observe=observe,
        )
        self.application = ApplicationInput(
            conversation,
            self.coordinator,
            phrase=self._phrase,
            wait_for_caller=self._wait_for_caller,
            say=self._say,
            act=self._act,
            interrupt=interrupt,
        )
        self.user_speaking = False
        self._user_idle = asyncio.Event()
        self._user_idle.set()
        self.last_spoke = self.last_progress = 0.0
        self._status_count = 0
        self._progress: asyncio.Task | None = None

    @property
    def worker(self):
        """The backend worker (``responses`` mode), else ``None``."""
        return self.coordinator.worker

    @property
    def client_delegations(self) -> set[str]:
        """Delegation ids issued in ``client`` mode."""
        return self.coordinator.client_delegations

    @property
    def backend_busy(self) -> bool:
        """Whether a backend task is running or queued."""
        return self.coordinator.backend_busy

    def start(self) -> None:
        """Start the delegation worker (``responses`` mode) and the progress announcer."""
        self.coordinator.start()
        if self.coordinator.worker and self.config.delegation.progress_speech:
            self._progress = asyncio.create_task(self._announce_progress(), name="delegation-progress")

    async def close(self) -> None:
        """Stop the worker and release the models."""
        if self._progress:
            self._progress.cancel()
            await asyncio.gather(self._progress, return_exceptions=True)
        await self.coordinator.close()
        await self.agent.close()

    def set_user_speaking(self, speaking: bool) -> None:
        """Record whether the caller is speaking; answers wait for them to finish."""
        self.user_speaking = speaking
        if speaking:
            self._user_idle.clear()
        else:
            self._user_idle.set()

    # ------------------------------------------------------------------ the caller's turn
    async def on_turn(self) -> None:
        """Decide the newest caller turn: speak, or delegate to the backend."""
        self.conversation.trim(self.config.chat_history_recent_turns)
        messages = self.conversation.messages()
        text = last_caller_text(messages)
        if not text.strip():
            return
        if self.router and await self._dropped_by_router(text, messages):
            return
        if self._asks_for_status(text):
            self._remember_voice("user", text)
            self.observe("status.answered", open=len(self.coordinator.ledger.open()))
            await self._say(self._status_line())
            return
        await self._act(text)

    def _asks_for_status(self, text: str) -> bool:
        return (
            self.config.delegation.status_from_ledger and bool(self.coordinator.ledger.open()) and asks_for_status(text)
        )

    def _status_line(self) -> str:
        line = STATUS_LINES[self._status_count % len(STATUS_LINES)]
        self._status_count += 1
        return line

    async def _announce_progress(self) -> None:
        """Say a short status line when work has run for a while without the assistant saying anything."""
        settings = self.config.delegation
        tick = max(0.05, min(1.0, settings.progress_after_seconds / 4))
        while True:
            await asyncio.sleep(tick)
            oldest = self.coordinator.ledger.oldest_open()
            if oldest is None or self.user_speaking:
                continue
            since = oldest.created_at
            if oldest.progress_lines >= settings.progress_max_per_delegation:
                continue
            now = time.monotonic()
            if now - max(self.last_spoke, since) < settings.progress_after_seconds:
                continue
            if now - self.last_progress < settings.progress_interval_seconds:
                continue
            self.last_progress = now
            oldest.progress_lines += 1
            self.observe("progress.spoken", open=len(self.coordinator.ledger.open()))
            await self._say(self._status_line())

    async def _act(self, text: str, *, instruction: bool = False) -> None:
        """Run the routing decision for ``text`` and carry it out."""
        instructions, history = split_messages(self.conversation.messages())
        work = self.coordinator.ledger.describe()
        decision = await self._decide(instructions, history, work)
        if not work:
            decision = self._apply_repeat_guard(decision, text, history)
        self.observe("talker.completed", action=decision.action, **decision.diagnostics)
        if decision.action == "delegate":
            await self._delegate(decision, text, instruction=instruction)
        else:
            if not instruction:
                self._remember_voice("user", text)
            await self._say(decision.speech)

    async def _dropped_by_router(self, text: str, messages: list[dict]) -> bool:
        verdict = self.router.verdict(text, last_assistant_text(messages))
        if verdict.keep:
            return False
        if self.config.guards.turn_router:
            self.observe("router.dropped", text=text, reason=verdict.reason)
            self._remember_voice("user", text)
            return True
        self.observe("router.would_drop", text=text, reason=verdict.reason)
        return False

    async def _decide(self, instructions: str, history: list[dict], work_in_progress: str) -> TalkerDecision:
        try:
            return await self.agent.talker.decide(instructions, history, work_in_progress)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self.config.reliability.recover_failures:
                raise
            # A failed routing call delegates the turn: no tool has run, so nothing can repeat.
            logger.warning(f"Talker failed ({type(exc).__name__}: {exc}); delegating the turn")
            self.observe("talker.recovered", error_type=type(exc).__name__)
            return TalkerDecision("delegate", "", {"recovery": "talker_failed"})

    def _apply_repeat_guard(self, decision: TalkerDecision, text: str, history: list[dict]) -> TalkerDecision:
        if not self.config.guards.repeat_guard or decision.action != "speak":
            return decision
        similarity = repeated_speech(decision.speech, text, history)
        if not similarity:
            return decision
        self.observe("talker.repeat_replaced", similarity=similarity)
        return TalkerDecision("delegate", "", {**decision.diagnostics, "recovery": "repeat_guard"})

    # ---------------------------------------------------------------------- delegation
    async def _delegate(self, decision: TalkerDecision, text: str, *, instruction: bool = False) -> None:
        delegation_id = await self.coordinator.delegate(text, instruction=instruction)
        speech = safe_lead_in(decision.speech) if decision.speech else ""
        if speech != decision.speech:
            self.observe("talker.lead_in_replaced", original=decision.speech)
        if speech:
            # Recorded at submit time: the answer can arrive and be phrased before this lead-in finishes
            # playing, so waiting for playback could miss the "already told" note.
            earlier = self.coordinator.told.get(delegation_id)  # a request that joined a waiting delegation
            self.coordinator.told[delegation_id] = f"{earlier} {speech}" if earlier else speech
            await self._say(speech)

    async def submit_items(self, items: list[dict]) -> str:
        """Start a backend task from typed input items (``response.create`` with queued messages)."""
        return await self.coordinator.submit_items(items)

    # --------------------------------------------------------------------- the answer
    async def on_result(self, text: str, task: DelegatedTask, told: str | None, *, newer_requests: list[str]) -> None:
        """Record a backend result, then phrase and speak it (as being updated if a newer request may change it)."""
        self.conversation.add({"role": "user", "content": BACKEND_RESULT_PREFIX + text})
        speech = await self._phrase(BACKEND_RESULT_PREFIX, text, told, answer=True, newer_requests=newer_requests)
        await self._wait_for_caller()
        await self._say(speech)

    async def _phrase(
        self,
        prefix: str,
        text: str,
        already_said: str | None,
        *,
        answer: bool,
        must_speak: bool = False,
        newer_requests: list[str] | None = None,
    ) -> str:
        """Return the words to speak for an update.

        A verified answer is spoken even if phrasing fails or stays silent. So is an update the application asked
        to have said aloud (``must_speak``): the writer may choose its wording, but not to drop it.
        """
        note = (
            f'\n(you already told the caller: "{already_said}" -- do not repeat that, report only the result)'
            if already_said
            else ""
        )
        if must_speak and not answer:
            note += "\n(the application asked for this to be said to the caller, so speak it)"
        if newer_requests:
            asked = "; ".join(f'"{request}"' for request in newer_requests)
            note += (
                f"\n(the caller has since asked something else, still in progress: {asked}. If it changes this result, "
                "say what was found, briefly, and that you are updating it; otherwise report this result normally)"
            )
        instructions, history = split_messages(self.conversation.messages())
        # The phrasing request carries the update once, as the commentary update; leave its history copy out.
        positions = [i for i, m in enumerate(history) if m["content"] == prefix + text]
        if positions:
            history = history[: positions[-1]] + history[positions[-1] + 1 :]
        try:
            decision = await self.agent.writer.render(instructions, history, text, answer=answer, note=note)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self.config.reliability.recover_failures:
                raise
            logger.warning(f"Commentary failed ({type(exc).__name__}: {exc}); speaking the update as written")
            self.observe("commentary.recovered", error_type=type(exc).__name__)
            return plain_speech(text) if answer or must_speak else ""
        if decision.action == "speak" and decision.speech:
            return decision.speech
        if answer or must_speak:
            # The caller is waiting on this answer, or the application asked for it to be said: it must be heard,
            # even only as written.
            self.observe("commentary.answer_forced" if answer else "commentary.forced", talker_action=decision.action)
            return plain_speech(text)
        return ""

    async def _wait_for_caller(self) -> None:
        """Hold an answer while the caller is mid-sentence, then speak it (bounded)."""
        if not self.user_speaking:
            return
        self.observe("commentary.held_for_caller")
        try:
            await asyncio.wait_for(self._user_idle.wait(), ANSWER_HOLD_SECONDS)
        except TimeoutError:
            self.observe("commentary.hold_expired")

    # ----------------------------------------------------------- client commands (appends)
    async def on_commentary(self, text: str, delegation_id: str | None) -> None:
        """Say a client-supplied fact to the caller (see :class:`ApplicationInput`)."""
        await self.application.commentary(text, delegation_id)

    def on_thinking(self, text: str) -> None:
        """Retain silent context (see :class:`ApplicationInput`)."""
        self.application.thinking(text)

    async def on_instruction(self, text: str) -> None:
        """Steer behavior (see :class:`ApplicationInput`)."""
        await self.application.instruction(text)

    # ------------------------------------------------------------------------- speech
    def _remember_voice(self, role: str, text: str) -> None:
        self.coordinator.remember_voice(role, text)

    async def _say(self, text: str) -> None:
        if not text.strip():
            return
        self._remember_voice("assistant", text)
        await self.say_text(text)
        self.last_spoke = time.monotonic()
