# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime processes: each one a client of the agent's protocol with a fresh work folder, opened from the settings.

The conductor's process runs only its session, so workers never take its turns; the workers' processes run the IM
session's worker sessions. Each session of a shared process holds one of its ``turns_per_process`` places from its
opening until its prompts have ended (:func:`give_back`), also while it waits for the app (a task waiting for the app
holds no IM slot). So the places, not ``max_running``, keep a process within its turns, and a full role opens another
process. A started process's environment holds only the allowlist, the agent's own needs and ``env``; the agent's
model, key and configuration are its operator's.
"""

from __future__ import annotations

import asyncio
import os
from collections import Counter
from pathlib import Path

from loguru import logger

from .protocol import HarnessClient, HarnessSession, Role, Runtime, RuntimeLaunch
from .settings import HarnessSettings
from .workdir import WorkFolder

# The waits for a retired session's last prompts to end (kept, so none is collected while it waits).
_ENDINGS: set[asyncio.Future] = set()


def give_back(runtime: Runtime, session: HarnessSession | None) -> None:
    """Give back a retired session's place in ``runtime`` once none of its prompts runs (an unconfirmed stop's may)."""
    if session is None or not session.outstanding:
        runtime.leave()
        return
    ending = asyncio.ensure_future(session.idle())
    _ENDINGS.add(ending)
    ending.add_done_callback(_ENDINGS.discard)
    ending.add_done_callback(lambda _: runtime.leave())


class RuntimeProcess:
    """One runtime process: its client, its work folder, the names its sessions' MCP servers take, its places taken."""

    def __init__(self, client: HarnessClient, folder: WorkFolder):
        """Hold an open ``client`` whose sessions work in ``folder``."""
        self.client, self.work = client, folder
        self.places = 0  # sessions that hold one of its turns (``HarnessAdapter.shared`` takes them)
        self._names: Counter[str] = Counter()

    def leave(self) -> None:
        """A session gives back its place (its prompts have ended)."""
        self.places = max(self.places - 1, 0)

    @property
    def folder(self) -> Path:
        """The process's work folder."""
        return self.work.path

    def server_name(self, prefix: str) -> str:
        """A new MCP server name, never used before in this process (an agent may key MCP servers by name)."""
        self._names[prefix] += 1
        return f"{prefix}{self._names[prefix]}"

    def session_folder(self, name: str) -> Path:
        """A new folder for one session, which the agent class readies (``Agent.prepare``)."""
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
        """A started process's environment: the allowlisted variables of ours, the agent's own needs, then ``env``."""
        env = {name: os.environ[name] for name in self.settings.pass_env if name in os.environ}
        env.update(self.settings.agent.environment())
        env.update(self.settings.env)
        return env

    async def open(self, role: Role) -> RuntimeProcess:
        """Make a work folder, start or reach a process for ``role``, initialize the protocol.

        Raises:
            Exception: The agent did not start or answer; nothing is left behind.
        """
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
