# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime processes: each a client of the agent's protocol with its own work folder, opened from the settings."""

from __future__ import annotations

import asyncio
import os
from collections import Counter
from pathlib import Path

from loguru import logger

from .protocol import HarnessClient, HarnessSession, Role, Runtime, RuntimeLaunch
from .settings import HarnessSettings
from .workdir import WorkFolder

# The waits for retired sessions' last prompts, kept so none is collected while it waits.
_ENDINGS: set[asyncio.Future] = set()
# The cancels sent in the background, kept until sent or timed out so none is collected before it is sent.
_CANCELS: set[asyncio.Future] = set()


def give_back(runtime: Runtime, session: HarnessSession | None) -> None:
    """Give back a retired session's place in ``runtime`` once none of its prompts runs (an unconfirmed stop's may)."""
    if session is None or not session.outstanding:
        runtime.leave()
        return
    ending = asyncio.ensure_future(session.idle())
    _ENDINGS.add(ending)
    ending.add_done_callback(_ENDINGS.discard)
    ending.add_done_callback(lambda _: runtime.leave())


def cancel_in_background(session: HarnessSession, seconds: float) -> None:
    """Cancel ``session``'s running turn in the background, within ``seconds``, and retrieve the cancel's error."""
    cancel = asyncio.ensure_future(asyncio.wait_for(session.cancel(), seconds))
    _CANCELS.add(cancel)
    cancel.add_done_callback(_CANCELS.discard)
    cancel.add_done_callback(lambda done: done.cancelled() or done.exception())


class RuntimeProcess:
    """One runtime process: its client, its work folder, its sessions' server names and the places they hold."""

    def __init__(self, client: HarnessClient, folder: WorkFolder):
        """Hold an open ``client`` whose sessions work in ``folder``."""
        self.client, self.work = client, folder
        # Sessions holding one of its turns, also while they wait for the app, when their task holds no IM slot.
        self.places = 0
        self._names: Counter[str] = Counter()

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
        """End the process, then delete its work folder."""
        try:
            await self.client.close()
        finally:
            self.work.remove()


class RuntimeOpener:
    """Opens runtime processes from the settings."""

    def __init__(self, settings: HarnessSettings):
        """Open processes as ``settings`` say."""
        self.settings = settings

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
        launch = RuntimeLaunch(settings.command, self.environment(), settings.url, folder.path)
        client = settings.protocol.client(
            launch,
            permission=settings.permission,
            start_timeout_seconds=settings.start_timeout_seconds,
            stop_grace_seconds=settings.stop_grace_seconds,
        )
        try:
            await client.open()
        except BaseException:
            try:
                await client.close()
            finally:
                folder.remove()
            raise
        logger.info(f"harness.runtime_opened role={role.name} folder={folder.path}")
        return RuntimeProcess(client, folder)
