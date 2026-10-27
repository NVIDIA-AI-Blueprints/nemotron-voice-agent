# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The harness's conductor (``port.Conductor``): an agent session that places requests through the control tools.

It runs in a runtime process of its own, so workers never take its turns, and its door serves exactly the four control
tools. Its instructions are ``texts.conductor_prompt``, placed by the agent class. One ``decide()`` is one prompt of the
rendered input, bounded by ``conductor_timeout_seconds``: each control call is a decision the IM applies, the IM's
outcome line is the call's result, and the model's text is ignored.

When the turn fails, times out or skips a request, the ``SteerOpenWork`` rule (``backend/conductors.py``) places what is
left: each unplaced arrival joins the newest open Work it was shown, or starts one when none is open, so a request never
starts a second Work beside an open one. When a newer request arrives, the IM cancels a ``decide()`` that placed
nothing; its session is retired, never reused (an agent may carry a cancelled turn into the next prompt), and a fresh
one opens in the background. So is a session whose turn ended any way but ``end_turn``.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

from loguru import logger

from delegation_core.backend.conductors import SteerOpenWork
from delegation_core.backend.port import Answer, Cancel, ConductorInput, Decision, Place, Start, Steer

from .bridge import ConductorDoor
from .protocol import HarnessSession, Mailbox, PromptEnded, Runtime
from .runtime import give_back

if TYPE_CHECKING:
    from .adapter import HarnessAdapter


class _Turn:
    """Reads the conductor session's events until its prompt ends; everything else it ignores."""

    def __init__(self, prompt: int):
        self.prompt = prompt
        self.ended: PromptEnded | None = None

    def prompt_ended(self, event: PromptEnded) -> None:
        if event.prompt == self.prompt:
            self.ended = event

    def __getattr__(self, name: str):
        """Every other event is ignored (the conductor's text is never used)."""
        return _ignore


def _ignore(event: object) -> None:
    """Ignore an event."""


class _Placed:
    """One ``decide()``'s ``place``: it records what the decisions placed and the Works they stopped whole.

    It reads each decision as a ``DecisionTarget`` does (double dispatch, no type switch), after the IM applied it. A
    Steer to a Work or task the conductor was not shown places nothing, as the IM refuses it.
    """

    def __init__(self, place: Place, given: ConductorInput):
        self.place = place
        self.shown = {work.work_id: {task.task_id for task in work.tasks} for work in given.works}
        self.placed: set[str] = set()
        self.cancelled: set[str] = set()

    async def __call__(self, decision: Decision) -> str:
        outcome = await self.place(decision)
        await decision.apply(self)
        return outcome

    async def start(self, decision: Start) -> str:
        self.placed.update(decision.delegations)
        return ""

    async def steer(self, decision: Steer) -> str:
        tasks = self.shown.get(decision.work_id)
        if tasks is not None and (decision.task_id is None or decision.task_id in tasks):
            self.placed.update(decision.delegations)
        return ""

    async def cancel(self, decision: Cancel) -> str:
        if decision.task_id is None:
            self.cancelled.add(decision.work_id)
        return ""

    async def answer(self, decision: Answer) -> str:
        self.placed.add(decision.delegation_id)
        return ""


class HarnessConductor:
    """The conductor of one IM session on the harness (``port.Conductor``)."""

    def __init__(self, host: HarnessAdapter):
        """Place through a session of ``host``'s conductor process; nothing opens until :meth:`warm`."""
        self.host, self.settings = host, host.settings
        self.runtime: Runtime | None = None
        self.session: HarnessSession | None = None
        self.door: ConductorDoor | None = None
        self.mailbox: Mailbox = asyncio.Queue()
        self.blocks: tuple[str, ...] = ()  # what opens a fresh session's first prompt, as the agent class gave it
        self.fresh = True  # the session has read nothing of the conversation
        self._opening: asyncio.Task | None = None

    def warm(self) -> None:
        """Open the session in the background, so the first request does not wait for the agent's start."""
        if self._opening is None or self._opening.done():
            self._opening = asyncio.ensure_future(self._open())
            self._opening.add_done_callback(_retrieved)

    async def decide(self, given: ConductorInput, place: Place) -> None:
        """One prompt to the session; each control call is placed; whatever is left joins or starts by the rule."""
        placed = _Placed(place, given)
        try:
            await self._turn(given, placed)
        except asyncio.CancelledError:
            await self._retire(cancel=True)
            raise
        except Exception as exc:  # a failed turn places nothing more; the rule places what is left
            logger.warning(f"harness.conductor_failed {type(exc).__name__}: {exc}")
            await self._retire(cancel=True)
        rest = tuple(arrival for arrival in given.arrivals if arrival.delegation_id not in placed.placed)
        if rest:
            works = tuple(work for work in given.works if work.work_id not in placed.cancelled)
            logger.info(
                f"harness.conductor_left_unplaced {[arrival.delegation_id for arrival in rest]}: the rule places them"
            )
            await SteerOpenWork().decide(ConductorInput(rest, works, given.conversation, given.new_from), place)

    async def _turn(self, given: ConductorInput, placed: _Placed) -> None:
        session = await self._session()
        self.door.bind(placed)
        text = self.host.renderer.conductor_input(given, fresh=self.fresh)
        if self.fresh and self.blocks:
            text = "\n\n".join([*self.blocks, text])
        turn = _Turn(session.prompt(text))
        try:
            async with asyncio.timeout(self.settings.conductor_timeout_seconds):
                while turn.ended is None:
                    (await self.mailbox.get()).apply(turn)
        except TimeoutError:
            logger.warning(f"harness.conductor_timed_out after {self.settings.conductor_timeout_seconds:g} s")
            await self._retire(cancel=True)
            return
        self.fresh = False
        if turn.ended.error is not None or turn.ended.stop_reason != "end_turn":
            logger.warning(f"harness.conductor_turn_ended stop={turn.ended.stop_reason} error={turn.ended.error}")
            await self._retire(cancel=False)  # never reused: a refusal lost it, a cancel would join the next prompt

    async def _session(self) -> HarnessSession:
        """The session: the open one, the one opening (pre-warmed), or a new one when it was lost.

        A lost one is retired, which opens a fresh one in the background; that opening is the one awaited, so two
        sessions never race to become the conductor's.
        """
        if self.session is not None and not self.runtime.client.alive:
            await self._retire(cancel=False)
        if self._opening is not None:
            with contextlib.suppress(Exception):
                await asyncio.shield(self._opening)  # a cancelled decide() leaves the opening to finish
            self._opening = None
        if self.session is None:
            await self._open()  # the opening failed (or none ran): open here, and let its error end the turn
        return self.session

    async def _open(self) -> None:
        """A fresh session (its runtime's place taken); a failed or cancelled opening leaves nothing behind."""
        host, prompt = self.host, host_prompt(self.host)
        runtime = await host.shared(host.conductor_role)
        session = door = None
        try:
            name = runtime.server_name(host.conductor_role.prefix)
            door = ConductorDoor(name, host.vocabulary, self.settings.prompts.no_placement)
            server = host.bridge.open(door)
            folder = runtime.session_folder(name)
            blocks = self.settings.agent.prepare(folder, prompt)
            mailbox: Mailbox = asyncio.Queue()
            session = await runtime.client.new_session(folder, server, mailbox)
            try:
                async with asyncio.timeout(self.settings.ready_timeout_seconds):
                    await door.listed.wait()
            except TimeoutError:
                raise RuntimeError("The agent did not list the control tools: its MCP connection failed") from None
        except BaseException:
            if session is not None:
                session.retire()
            if door is not None:
                host.bridge.close(door, "")
            give_back(runtime, session)
            raise
        self.runtime, self.session, self.door, self.mailbox = runtime, session, door, mailbox
        self.blocks, self.fresh = blocks, True
        logger.debug(f"harness.conductor_opened server={name} session={session.id}")

    async def _retire(self, *, cancel: bool) -> None:
        """Never use the session again (cancelled first while its prompt runs); open a fresh one in the background."""
        session, door, runtime = self.session, self.door, self.runtime
        self.session = self.door = self.runtime = None
        if session is not None:
            if cancel and session.outstanding:
                with contextlib.suppress(Exception):
                    await session.cancel()
            session.retire()
        if door is not None:
            self.host.bridge.close(door, "")
        if runtime is not None:
            give_back(runtime, session)
        if session is not None and not self.host.closed:
            self.warm()

    async def close(self) -> None:
        """Stop opening, retire the session."""
        if self._opening is not None:
            self._opening.cancel()
            await asyncio.gather(self._opening, return_exceptions=True)
            self._opening = None
        await self._retire(cancel=True)


def host_prompt(host: HarnessAdapter) -> str:
    """The conductor's instructions: ``texts.conductor_prompt``."""
    return host.vocabulary.prompt


def _retrieved(task: asyncio.Future) -> None:
    """Read a background opening's error, so it is logged once and never left unretrieved."""
    if not task.cancelled() and task.exception() is not None:
        logger.warning(f"harness.conductor_open_failed {type(task.exception()).__name__}: {task.exception()}")
