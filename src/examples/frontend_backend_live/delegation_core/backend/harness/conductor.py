# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The harness's conductor (``port.Conductor``): an agent session that places requests through the control tools."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

from loguru import logger

from examples.frontend_backend_live.delegation_core.backend.port import ConductorInput, Place

from .bridge import ConductorDoor
from .protocol import HarnessSession, Mailbox, PromptEnded, Runtime, Wake, usage_total
from .runtime import cancel_in_background, give_back

if TYPE_CHECKING:
    from .adapter import HarnessAdapter


class _Turn:
    """One conductor turn: it reads the session's events until its prompt ends, and ignores the rest."""

    def __init__(self, prompt: int, total_before: int | None, started: float):
        self.prompt = prompt
        # The session's token total before the prompt, for the agent class's check of a model call.
        self.total_before = total_before
        # The loop time the prompt went out, and the backstop that fires turn_timeout_seconds after it.
        self.started = started
        self.timer: asyncio.TimerHandle | None = None
        self.ended: PromptEnded | None = None

    def prompt_ended(self, event: PromptEnded) -> None:
        if event.prompt == self.prompt:
            self.ended = event

    def __getattr__(self, name: str):
        """Ignore every other event: the conductor's text is never used."""
        return _ignore


def _ignore(event: object) -> None:
    """Ignore an event."""


class HarnessConductor:
    """The conductor of one IM session, in a runtime process of its own so workers never take its turns."""

    def __init__(self, host: HarnessAdapter):
        """Place through a session of ``host``'s conductor process; nothing opens until :meth:`warm`."""
        self.host, self.settings = host, host.settings
        self.runtime: Runtime | None = None
        self.session: HarnessSession | None = None
        self.door: ConductorDoor | None = None
        self.mailbox: Mailbox = asyncio.Queue()
        self.blocks: tuple[str, ...] = ()
        self.fresh = True
        # The session's turn until its end is judged, and the token total the session last reported.
        self.turn: _Turn | None = None
        self.total: int | None = None
        self._opening: asyncio.Task | None = None
        # One reader of the session's events at a time, so two decide() calls never split a turn's events.
        self._reading = asyncio.Lock()

    def warm(self) -> None:
        """Open the session in the background, so the first request does not wait for the agent's start."""
        if self._opening is None or self._opening.done():
            self._opening = asyncio.ensure_future(self._open())
            self._opening.add_done_callback(_retrieved)

    async def decide(self, given: ConductorInput, place: Place) -> None:
        """Prompt the session with the rendered input; each control call is placed, and the IM places what is left.

        It reads its turn to its end. A cancelled ``decide()`` also reads the running turn to its end before it ends,
        so a restarted one reads records built once the agent is free; one cancelled again leaves the turn running,
        and ``close()`` ends any reading and cancels the turn. A ``decide()`` never prompts a busy agent: a turn still
        running, such as one a hang-up left, is read to its end first. A turn that fails, in which the agent class
        finds no model call, or that still runs ``turn_timeout_seconds`` after its prompt (cancelled then) retires the
        session, as a failed ``decide()`` does.
        """
        try:
            await self._turn(given, place)
        except Exception as exc:
            logger.warning(f"harness.conductor_failed {type(exc).__name__}: {exc}")
            await self._retire(cancel=True)

    async def _turn(self, given: ConductorInput, place: Place) -> None:
        try:
            session = await self._free()
            self.door.bind(place)
            text = self.host.renderer.conductor_input(given, fresh=self.fresh)
            if self.fresh and self.blocks:
                text = "\n\n".join([*self.blocks, text])
            loop = asyncio.get_running_loop()
            turn = self.turn = _Turn(session.prompt(text), self.total, loop.time())
            turn.timer = loop.call_later(self.settings.turn_timeout_seconds, self._overran, turn)
            self.fresh = False
            await self._read(turn)
        except asyncio.CancelledError:
            # The agent's turn runs on; on a restart its calls count, and the IM builds the next input after it ends.
            while self.turn is not None:
                await self._read(self.turn)
            raise

    async def _free(self) -> HarnessSession:
        """Return the session once its agent is free: a turn still running is read to its end and judged first.

        Such a turn is one a ``decide()`` cancelled twice left, such as at a hang-up; when judging it retires the
        session, a fresh one opens.
        """
        session = await self._session()
        while self.turn is not None:
            await self._read(self.turn)
            session = await self._session()
        return session

    async def _read(self, turn: _Turn) -> None:
        """Read the turn's events until it ends, then judge it; a drop (backstop or ``close()``) ends the reading.

        Another ``decide()`` still reading the turn reads it first, and this one judges nothing it judged.
        """
        async with self._reading:
            mailbox = self.mailbox
            while turn.ended is None and turn is self.turn:
                (await mailbox.get()).apply(turn)
            if turn is self.turn:
                await self._judge(turn)

    async def _judge(self, turn: _Turn) -> None:
        """Keep the session if its turn ended ``end_turn`` with no error, unless its agent class finds no model call."""
        ended, total = turn.ended, usage_total(turn.ended.usage)
        turn.timer.cancel()
        self.turn = None
        if total is not None:
            self.total = total
        if ended.error is None and ended.stop_reason == "end_turn":
            reason = self.settings.agent.no_answer(total, turn.total_before)
            if reason is None:
                return
            logger.warning(f"harness.conductor_turn_ended stop={ended.stop_reason}: {reason}")
        else:
            logger.warning(f"harness.conductor_turn_ended stop={ended.stop_reason} error={ended.error}")
        await self._retire(cancel=False)

    def _overran(self, turn: _Turn) -> None:
        """Cancel a turn still running ``turn_timeout_seconds`` after its prompt, once, and retire its session."""
        session = self.session
        if turn is not self.turn or turn.prompt not in session.outstanding:
            return
        seconds = asyncio.get_running_loop().time() - turn.started
        logger.warning(f"harness.conductor_turn_overran seconds={seconds:.3f}")
        cancel_in_background(session, self.settings.stop_timeout_seconds)
        self._drop()

    async def _session(self) -> HarnessSession:
        """Return the open session, else the one opening, else a new one.

        A lost session is retired, starting a fresh opening, which is the one awaited so two sessions never race.
        """
        if self.session is not None and not self.runtime.client.alive:
            await self._retire(cancel=False)
        if self._opening is not None:
            with contextlib.suppress(Exception):
                # Shielded: a cancelled decide() leaves the opening to finish.
                await asyncio.shield(self._opening)
            self._opening = None
        if self.session is None:
            await self._open()
        return self.session

    async def _open(self) -> None:
        """Open a fresh session in a conductor process; a failed or cancelled opening leaves nothing behind."""
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
        self.turn, self.total = None, 0
        logger.debug(f"harness.conductor_opened server={name} session={session.id}")

    async def _retire(self, *, cancel: bool) -> None:
        """Retire the session, cancelling a running prompt if ``cancel`` is set, and warm a fresh one."""
        session = self.session
        self._drop()
        if cancel and session is not None and session.outstanding:
            with contextlib.suppress(Exception):
                await session.cancel()

    def _drop(self) -> None:
        """Retire the session, its turn and the turn's backstop at once, and warm a fresh one; it sends no cancel."""
        session, door, runtime, turn = self.session, self.door, self.runtime, self.turn
        self.session = self.door = self.runtime = self.turn = self.total = None
        if turn is not None:
            turn.timer.cancel()
            # Wakes a decide() still reading the turn, so it stops.
            self.mailbox.put_nowait(Wake())
        if session is not None:
            session.retire()
        if door is not None:
            self.host.bridge.close(door, "")
        if runtime is not None:
            give_back(runtime, session)
        if session is not None and not self.host.closed:
            self.warm()

    async def close(self) -> None:
        """Stop any opening and retire the session."""
        if self._opening is not None:
            self._opening.cancel()
            await asyncio.gather(self._opening, return_exceptions=True)
            self._opening = None
        await self._retire(cancel=True)


def host_prompt(host: HarnessAdapter) -> str:
    """Return the conductor's instructions: ``texts.conductor_prompt`` for ``max_running`` tasks."""
    return host.vocabulary.instructions(host.max_running)


def _retrieved(task: asyncio.Future) -> None:
    """Log a background opening's error, so it is never left unretrieved."""
    if not task.cancelled() and task.exception() is not None:
        logger.warning(f"harness.conductor_open_failed {type(task.exception()).__name__}: {task.exception()}")
