# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime processes: each a client of the agent's protocol with its own work folder, opened from the settings."""

from __future__ import annotations

import asyncio
import os
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from loguru import logger

from .protocol import CancelDispatch, HarnessClient, HarnessSession, NativeCleanup, Role, Runtime, RuntimeLaunch
from .settings import HarnessSettings
from .workdir import NATIVE_OWNER, WorkFolder

# The waits for retired sessions' last prompts, kept so none is collected while it waits.
_ENDINGS: set[asyncio.Future] = set()
_CANCELS: set[asyncio.Future] = set()
# Physical runtimes may drain after the logical close returns. Keep ownership and the folder lock until they end.
_CLEANUPS: set[asyncio.Task] = set()


def _cleanup(client: HarnessClient, folder: WorkFolder, *, close_error: BaseException | None = None) -> asyncio.Task:
    if close_error is not None and not (folder.path / NATIVE_OWNER).exists():
        folder.protect_native_owner()

    async def finish() -> NativeCleanup:
        try:
            receipt = await client.wait_closed()
        except Exception as error:
            folder.quarantine()
            logger.error(f"harness.native_cleanup_unconfirmed folder={folder.path} error_type={type(error).__name__}")
            raise
        if receipt.confirmed and receipt.workspace_safe:
            folder.remove()
        else:
            folder.quarantine()
            logger.info(
                f"harness.workspace_quarantined folder={folder.path} scope={receipt.scope} "
                f"confirmed={receipt.confirmed} workspace_safe={receipt.workspace_safe}"
            )
        return receipt

    task = asyncio.create_task(finish(), name="harness-runtime-cleanup")
    _CLEANUPS.add(task)
    task.add_done_callback(_CLEANUPS.discard)
    task.add_done_callback(lambda done: done.cancelled() or done.exception())
    return task


def give_back(
    runtime: Runtime, session: HarnessSession | None, *, pending_calls: Sequence[asyncio.Future] = ()
) -> asyncio.Future[bool]:
    """Return physical capacity only after the retired native invocation and its admitted bridge calls settle."""
    pending = tuple(call for call in pending_calls if not call.done())
    if not pending and (session is None or session.quiescent):
        runtime.leave()
        receipt: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        receipt.set_result(True)
        return receipt

    async def release() -> bool:
        if pending:
            await asyncio.gather(*(asyncio.shield(call) for call in pending), return_exceptions=True)
        if session is not None:
            try:
                await session.idle()
            except Exception as error:
                logger.info(f"harness.native_place_retained session={session.id} error_type={type(error).__name__}")
                return False
        if session is None or session.quiescent:
            runtime.leave()
            return True
        else:
            logger.info(f"harness.native_place_retained session={session.id} reason=uncertain_native_terminal")
            return False

    ending = asyncio.ensure_future(release())
    _ENDINGS.add(ending)
    ending.add_done_callback(_ENDINGS.discard)
    ending.add_done_callback(lambda done: done.cancelled() or done.exception())
    return ending


def cancel_in_background(session: HarnessSession, seconds: float) -> asyncio.Task[CancelDispatch]:
    """Cancel ``session``'s running turn in the background, within ``seconds``, and retrieve the cancel's error."""
    cancel = asyncio.ensure_future(asyncio.wait_for(session.cancel(), seconds))
    _CANCELS.add(cancel)
    cancel.add_done_callback(_CANCELS.discard)
    cancel.add_done_callback(lambda done: done.cancelled() or done.exception())
    return cancel


class RuntimeProcess:
    """One runtime process: its client, its work folder, its sessions' server names and the places they hold."""

    def __init__(self, client: HarnessClient, folder: WorkFolder):
        """Hold an open ``client`` whose sessions work in ``folder``."""
        self.client, self.work = client, folder
        # Sessions holding one of its turns, also while they wait for the app, when their task holds no IM slot.
        self.places = 0
        self._names: Counter[str] = Counter()
        self._cleanup: asyncio.Task | None = None
        self._closing: asyncio.Task | None = None
        self.cleanup_receipt = NativeCleanup("connection", False, False)

    def leave(self) -> None:
        """Give back the place of a session whose prompts have ended."""
        self.places = max(self.places - 1, 0)

    @property
    def folder(self) -> Path:
        """The process's work folder."""
        return self.work.path

    def server_name(self, prefix: str) -> str:
        """Return a new MCP server name for this process, since an agent may key MCP servers by name per process."""
        self._names[prefix] += 1
        return f"{prefix}{self._names[prefix]}"

    def session_folder(self, name: str) -> Path:
        """Make a new folder for one session, for the agent class to ready (``Agent.prepare``)."""
        return self.work.session_folder(name)

    async def close(self) -> None:
        """Fence the process promptly; retain its workspace until physical cleanup is confirmed."""
        if self._closing is None:
            self._closing = asyncio.create_task(self._close(), name="harness-runtime-close")
            self._closing.add_done_callback(lambda task: task.cancelled() or task.exception())
        await asyncio.shield(self._closing)

    async def _close(self) -> None:
        close_error = None
        try:
            await self.client.close()
        except BaseException as error:
            close_error = error
            raise
        finally:
            if self._cleanup is None:
                self._cleanup = _cleanup(self.client, self.work, close_error=close_error)

    async def wait_closed(self) -> NativeCleanup:
        """Wait for confirmed physical shutdown and workspace removal, without cancelling either."""
        results = await asyncio.gather(self.close(), return_exceptions=True)
        if self._cleanup is not None:
            physical = await asyncio.gather(asyncio.shield(self._cleanup), return_exceptions=True)
            if not isinstance(physical[0], BaseException):
                self.cleanup_receipt = physical[0]
            results.extend(physical)
            if self.cleanup_receipt.confirmed and self.cleanup_receipt.workspace_safe:
                self.places = 0
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return self.cleanup_receipt


class RuntimeOpener:
    """Opens runtime processes from the settings."""

    def __init__(self, settings: HarnessSettings):
        """Open processes as ``settings`` say."""
        self.settings = settings
        self._cleanups: set[asyncio.Task] = set()

    @property
    def has_cleanup(self) -> bool:
        """Whether this opener still owns unfinished cleanup or an unjoined cleanup failure."""
        return bool(self._cleanups)

    async def wait_closed(self) -> None:
        """Join this opener's failed-start cleanups without cancelling them or losing earlier failures."""
        while self._cleanups:
            cleanups = tuple(self._cleanups)
            results = await asyncio.gather(*(asyncio.shield(task) for task in cleanups), return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            self._cleanups.difference_update(cleanups)

    def _own_cleanup(self, task: asyncio.Task) -> None:
        self._cleanups.add(task)

        def settled(done: asyncio.Task) -> None:
            if not done.cancelled() and done.exception() is None:
                self._cleanups.discard(done)

        task.add_done_callback(settled)

    def environment(self) -> dict[str, str]:
        """Return a started process's whole environment: the ``pass_env`` variables, the agent's needs, then ``env``."""
        env = {name: os.environ[name] for name in self.settings.pass_env if name in os.environ}
        env.update(self.settings.agent.environment())
        env.update(self.settings.env)
        return env

    async def open(self, role: Role) -> RuntimeProcess:
        """Make a work folder, start or reach a process for ``role`` and initialize it, leaving nothing on failure."""
        settings = self.settings
        folder = WorkFolder.make(settings.work_root)
        launch = RuntimeLaunch(
            settings.command,
            self.environment(),
            settings.url,
            folder.path,
        )
        client = settings.protocol.client(
            launch,
            permission=settings.permission,
            start_timeout_seconds=settings.start_timeout_seconds,
            stop_grace_seconds=settings.stop_grace_seconds,
        )
        folder.protect_native_owner()
        try:
            await client.open()
        except BaseException:
            close_error = None
            try:
                await client.close()
            except BaseException as error:
                close_error = error
                raise
            finally:
                cleanup = _cleanup(client, folder, close_error=close_error)
                self._own_cleanup(cleanup)
            raise
        logger.info(f"harness.runtime_opened role={role.name} folder={folder.path}")
        return RuntimeProcess(client, folder)
