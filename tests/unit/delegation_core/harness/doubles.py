# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""Doubles of the harness's own seams, and the recorded Hermes wire in ``fixtures/``."""

from __future__ import annotations

import asyncio
import atexit
import itertools
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from examples.frontend_backend_live.delegation_core.backend.control import ControlVocabulary
from examples.frontend_backend_live.delegation_core.backend.harness.acp.hermes import HermesAgent
from examples.frontend_backend_live.delegation_core.backend.harness.acp.updates import SessionUpdates
from examples.frontend_backend_live.delegation_core.backend.harness.adapter import HarnessAdapter
from examples.frontend_backend_live.delegation_core.backend.harness.bridge import Door
from examples.frontend_backend_live.delegation_core.backend.harness.protocol import (
    McpServer,
    MessageChunk,
    PromptEnded,
    Role,
    ThoughtChunk,
)
from examples.frontend_backend_live.delegation_core.backend.harness.render import Renderer
from examples.frontend_backend_live.delegation_core.backend.harness.settings import HarnessSettings
from tests.unit.delegation_core.helpers import CORE_SETTINGS, TEXTS

# The shipped prompts and agent blocks: the prompts have no default, and neither has Hermes's command guard.
PROMPTS = dict(CORE_SETTINGS.backend.blocks["harness"]["prompts"])
AGENT_BLOCKS = {name: dict(block) for name, block in CORE_SETTINGS.backend.blocks["harness"]["agents"].items()}
WIRE = Path(__file__).parent / "fixtures" / "hermes_wire.json"


def vocabulary() -> ControlVocabulary:
    return ControlVocabulary(TEXTS.control_tools, TEXTS.conductor_prompt, TEXTS.refusal("idle"), TEXTS.failed_output)


def hermes(**block: Any) -> HermesAgent:
    return HermesAgent.from_settings({**AGENT_BLOCKS["hermes"], **block})


def wire(scenario: str) -> list[dict]:
    """Return a recorded scenario of the real Hermes wire: one message per line of ``fixtures/hermes_wire.json``."""
    return json.loads(WIRE.read_text(encoding="utf-8"))[scenario]


def settings(**block: Any) -> HarnessSettings:
    """Build settings of the shipped prompts and agents: Hermes, which steers mid-turn, unless ``block`` says so."""
    base = {"prompts": PROMPTS, "agents": AGENT_BLOCKS, "agent": "hermes", "command": ["runtime", "acp"]}
    return HarnessSettings.model_validate({**base, "ready_timeout_seconds": 1, "stop_timeout_seconds": 0.5, **block})


def plain(**block: Any) -> HarnessSettings:
    """Build settings for a plain ACP agent, which has no mid-turn steer: a newer request cancels and re-prompts."""
    return settings(**{"agent": "acp", **block})


_SCRATCH: list[Path] = []


def scratch(prefix: str) -> Path:
    """Make a fresh folder under one root per test process, which is removed when the process ends."""
    if not _SCRATCH:
        _SCRATCH.append(Path(tempfile.mkdtemp(prefix="harness-tests-")))
        atexit.register(shutil.rmtree, _SCRATCH[0], True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=_SCRATCH[0]))


async def until(condition, timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.005)


class FakeSession:
    """A ``HarnessSession``: what the run sends is recorded; the test ends prompts and puts events in the mailbox."""

    def __init__(self, mailbox: asyncio.Queue, number: int = 1):
        self.id = f"session-{number}"
        self.mailbox = mailbox
        self.sent: list[tuple[int, str, str]] = []
        self.cancels = 0
        self.retired = False
        self._numbers = itertools.count(1)
        self._outstanding: set[int] = set()
        self._ended: dict[int, asyncio.Event] = {}

    @property
    def outstanding(self) -> frozenset[int]:
        return frozenset(self._outstanding)

    def prompt(self, text: str) -> int:
        return self._send("prompt", text)

    def steer(self, text: str) -> int:
        return self._send("steer", text)

    def _send(self, kind: str, text: str) -> int:
        number = next(self._numbers)
        self._outstanding.add(number)
        self._ended[number] = asyncio.Event()
        self.sent.append((number, kind, text))
        return number

    async def cancel(self) -> bool:
        if not self._outstanding:
            return False
        self.cancels += 1
        return True

    async def ended(self, prompt: int) -> None:
        event = self._ended.get(prompt)
        if event is not None:
            await event.wait()

    async def idle(self) -> None:
        while self._outstanding:
            await asyncio.sleep(0.01)

    def retire(self) -> None:
        self.retired = True

    def end(self, prompt: int, stop: str | None = "end_turn", usage: dict | None = None, error: str | None = None):
        self._outstanding.discard(prompt)
        self.mailbox.put_nowait(PromptEnded(prompt, stop, usage, error))
        self._ended.pop(prompt).set()

    def say(self, message_id: str, *chunks: str, thought: bool = True) -> None:
        if thought:
            self.mailbox.put_nowait(ThoughtChunk(message_id))
        for chunk in chunks:
            self.mailbox.put_nowait(MessageChunk(message_id, chunk))

    def put(self, event: Any) -> None:
        self.mailbox.put_nowait(event)

    def replay(self, update: dict, server: str = "client") -> None:
        """Put the event a recorded Hermes update carries, read as the Hermes agent class reads it."""
        parsed = SessionUpdates(hermes(), server).read(update)
        if parsed is not None:
            self.mailbox.put_nowait(parsed)

    def texts(self, kind: str) -> list[str]:
        return [text for _, sent_kind, text in self.sent if sent_kind == kind]

    def last_prompt(self) -> int:
        return self.sent[-1][0]


class FakeClient:
    def __init__(self):
        self.sessions: list[FakeSession] = []
        self.servers: list[McpServer] = []
        self.folders: list[Path] = []
        self.alive = True

    async def new_session(self, cwd: Path, server: McpServer, mailbox: asyncio.Queue) -> FakeSession:
        self.sessions.append(FakeSession(mailbox, len(self.sessions) + 1))
        self.servers.append(server)
        self.folders.append(cwd)
        return self.sessions[-1]

    async def close(self) -> None:
        self.alive = False


class FakeRuntime:
    def __init__(self):
        self.client = FakeClient()
        self.closed = False
        self.places = 0
        self.folder = scratch("work-")
        self._names: dict[str, int] = {}

    def leave(self) -> None:
        self.places -= 1

    def server_name(self, prefix: str) -> str:
        self._names[prefix] = self._names.get(prefix, 0) + 1
        return f"{prefix}{self._names[prefix]}"

    def session_folder(self, name: str) -> Path:
        folder = self.folder / "sessions" / name
        folder.mkdir(parents=True)
        return folder

    async def close(self) -> None:
        self.closed = True


class FakeBridge:
    """The bridge's door book with no HTTP: opening a door counts as the runtime listing its tools (ready)."""

    def __init__(self, *, lists: bool = True):
        self.doors: dict[str, Door] = {}
        self.closed: list[tuple[Door, str]] = []
        self.lists = lists

    def open(self, door: Door) -> McpServer:
        self.doors[door.token] = door
        if self.lists:
            door.listed.set()
        return McpServer(door.name, "http://bridge.test/mcp", door.token)

    def close(self, door: Door, text: str) -> None:
        self.doors.pop(door.token, None)
        self.closed.append((door, text))
        door.retire(text)

    async def start(self) -> None:
        pass


class FakeHost:
    """The adapter as a worker run and a conductor see it: settings, renderer, bridge, tools, roles, runtimes."""

    def __init__(self, harness: HarnessSettings | None = None, *, tools=(), bridge: FakeBridge | None = None):
        self.settings = harness or settings()
        self.renderer = Renderer(self.settings.prompts)
        self.bridge = bridge or FakeBridge()
        self.tools = tuple(tools)
        self.vocabulary = vocabulary()
        self.max_running = self.settings.max_running
        self.workers = Role("workers", "w")
        self.conductor_role = Role("conductor", "c")
        self.runtime = FakeRuntime()
        self.closed = False
        self.kept = None

    async def shared(self, role: Role) -> FakeRuntime:
        self.runtime.places += 1
        return self.runtime

    def take_place(self, runtime: FakeRuntime) -> bool:
        limit = self.settings.turns_per_process
        if (limit is not None and runtime.places >= limit) or not runtime.client.alive:
            return False
        runtime.places += 1
        return True

    # The adapter's own methods, so that a session is kept and taken as the adapter does.
    take_kept = HarnessAdapter.take_kept
    keep = HarnessAdapter.keep
