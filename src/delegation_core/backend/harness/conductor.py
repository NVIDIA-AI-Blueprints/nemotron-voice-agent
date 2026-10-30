# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The harness's conductor (``port.Conductor``): an agent session that places requests through the control tools.

It runs in a runtime process of its own, so workers never take its turns, and its door serves exactly the four control
tools. Its instructions are ``texts.conductor_prompt``, placed by the agent class. One ``decide()`` is one prompt of the
rendered input, bounded by ``conductor_timeout_seconds``: each control call is a decision the IM applies, the IM's
outcome line is the call's result, and the model's text is ignored.

Requests the turn leaves unplaced (a failed or timed-out turn, a refused decision, or no call naming them) are left to
the IM, whose ``steer`` rule places them against the Works as the turn left them: a request that came with one the turn
started joins that Work. When a newer request arrives, the IM cancels a ``decide()`` that placed nothing.

A turn past its limit, or of a cancelled ``decide()``, is cancelled while it runs, and the ``decide()`` reads on until
the turn ends, for at most ``stop_timeout_seconds``. A session whose turn ended ``cancelled`` takes the next
``decide()``; the agent may carry the cancelled prompt into that one (Hermes puts it before the new prompt's text). A
turn that ended ``end_turn`` after the cancel went out may have ended before the cancel reached the agent, and an idle
agent takes a cancel for its next prompt (Hermes ends that one unprocessed): that session is retired, as is one whose
turn did not end by then or ended any other way, and a fresh one opens in the background.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

from loguru import logger

from delegation_core.backend.port import ConductorInput, Place

from .bridge import ConductorDoor
from .protocol import HarnessSession, Mailbox, PromptEnded, Runtime
from .runtime import give_back

if TYPE_CHECKING:
    from .adapter import HarnessAdapter

# How a turn ends that keeps its session for the next decide(), by whether a cancel went out for it: it ran to its end,
# or it ended for that cancel. After a cancel, end_turn is not kept: the cancel may have crossed the turn's end.
KEPT = {False: "end_turn", True: "cancelled"}


class _Turn:
    """Reads the conductor session's events until its prompt ends; everything else it ignores."""

    def __init__(self, prompt: int):
        self.prompt = prompt
        self.ended: PromptEnded | None = None
        self.cancel_sent = False  # a cancel went out while it ran

    def prompt_ended(self, event: PromptEnded) -> None:
        if event.prompt == self.prompt:
            self.ended = event

    def __getattr__(self, name: str):
        """Every other event is ignored (the conductor's text is never used)."""
        return _ignore


def _ignore(event: object) -> None:
    """Ignore an event."""


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
        """One prompt to the session; each control call is placed; the IM's steer rule places what the turn leaves."""
        try:
            await self._turn(given, place)
        except Exception as exc:  # a failed turn places nothing more; the IM's steer rule places what is left
            logger.warning(f"harness.conductor_failed {type(exc).__name__}: {exc}")
            await self._retire(cancel=True)

    async def _turn(self, given: ConductorInput, place: Place) -> None:
        session = await self._session()
        self.door.bind(place)
        text = self.host.renderer.conductor_input(given, fresh=self.fresh)
        if self.fresh and self.blocks:
            text = "\n\n".join([*self.blocks, text])
        turn = _Turn(session.prompt(text))
        try:
            await self._read(turn, self.settings.conductor_timeout_seconds)
        except asyncio.CancelledError:  # the IM cancelled the decide() for a newer request
            await self._stop(session, turn)
            raise
        if turn.ended is None:
            logger.warning(f"harness.conductor_timed_out after {self.settings.conductor_timeout_seconds:g} s")
            await self._stop(session, turn)
            return
        self.fresh = False
        await self._keep_or_retire(turn)

    async def _read(self, turn: _Turn, seconds: float) -> None:
        """Read the session's events until ``turn`` has ended, for at most ``seconds``."""
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(seconds):
                while turn.ended is None:
                    (await self.mailbox.get()).apply(turn)

    async def _stop(self, session: HarnessSession, turn: _Turn) -> None:
        """Cancel the running turn and read on until it ends, for at most ``stop_timeout_seconds``."""
        try:
            turn.cancel_sent = await session.cancel()
            await self._read(turn, self.settings.stop_timeout_seconds)
        finally:  # a decide() cancelled again while it waits leaves the turn running: the session is retired
            await self._keep_or_retire(turn)

    async def _keep_or_retire(self, turn: _Turn) -> None:
        """Keep the session for the next ``decide()`` if its turn ended as ``KEPT`` names; else retire it."""
        ended = turn.ended
        if ended is not None and ended.error is None and ended.stop_reason == KEPT[turn.cancel_sent]:
            return
        if ended is None:
            logger.warning("harness.conductor_turn_not_ended after its cancel")
        else:
            logger.warning(f"harness.conductor_turn_ended stop={ended.stop_reason} error={ended.error}")
        await self._retire(cancel=False)

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
            door = ConductorDoor(name, host.vocabulary)
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
    """The conductor's instructions: ``texts.conductor_prompt``, with the tasks the harness runs at once."""
    return host.vocabulary.instructions(host.max_running)


def _retrieved(task: asyncio.Future) -> None:
    """Read a background opening's error, so it is logged once and never left unretrieved."""
    if not task.cancelled() and task.exception() is not None:
        logger.warning(f"harness.conductor_open_failed {type(task.exception()).__name__}: {task.exception()}")
