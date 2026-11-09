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
from .protocol import HarnessSession, Mailbox, PromptEnded, Runtime
from .runtime import give_back

if TYPE_CHECKING:
    from .adapter import HarnessAdapter

# The stop reason that keeps the session, by whether a cancel went out: after a cancel, end_turn may have crossed it,
# and an idle agent takes a stray cancel for its next prompt. A cancelled prompt may join the session's next one.
KEPT = {False: "end_turn", True: "cancelled"}


class _Turn:
    """One conductor turn: it reads the session's events until its prompt ends, and ignores the rest."""

    def __init__(self, prompt: int):
        self.prompt = prompt
        self.ended: PromptEnded | None = None
        self.cancel_sent = False

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
        self._opening: asyncio.Task | None = None

    def warm(self) -> None:
        """Open the session in the background, so the first request does not wait for the agent's start."""
        if self._opening is None or self._opening.done():
            self._opening = asyncio.ensure_future(self._open())
            self._opening.add_done_callback(_retrieved)

    async def decide(self, given: ConductorInput, place: Place) -> None:
        """Prompt the session with the rendered input; each control call is placed, and the IM places what is left.

        A turn is bounded by ``conductor_timeout_seconds``; a failed turn retires the session.
        """
        try:
            await self._turn(given, place)
        except Exception as exc:
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
        except asyncio.CancelledError:
            await self._stop(session, turn)
            raise
        if turn.ended is None:
            logger.warning(f"harness.conductor_timed_out after {self.settings.conductor_timeout_seconds:g} s")
            await self._stop(session, turn)
            return
        self.fresh = False
        await self._keep_or_retire(turn)

    async def _read(self, turn: _Turn, seconds: float) -> None:
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(seconds):
                while turn.ended is None:
                    (await self.mailbox.get()).apply(turn)

    async def _stop(self, session: HarnessSession, turn: _Turn) -> None:
        """Cancel the running turn and read on until it ends, for at most ``stop_timeout_seconds``."""
        try:
            turn.cancel_sent = await session.cancel()
            await self._read(turn, self.settings.stop_timeout_seconds)
        finally:
            # A decide() cancelled again during the wait leaves the turn running, and the session is retired.
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
        logger.debug(f"harness.conductor_opened server={name} session={session.id}")

    async def _retire(self, *, cancel: bool) -> None:
        """Retire the session, cancelling a running prompt if ``cancel`` is set, and warm a fresh one."""
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
