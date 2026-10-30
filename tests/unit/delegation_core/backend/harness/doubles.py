# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""Doubles of the harness's own seams for its unit tests; the real runtime runs in ``test_harness_real_hermes``.

Session events come from the real Hermes wire logs in ``fixtures/``, read through the Hermes agent class.
"""

from __future__ import annotations

import asyncio
import atexit
import itertools
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from delegation_core.backend.control import ControlVocabulary
from delegation_core.backend.harness.acp.hermes import HermesAgent
from delegation_core.backend.harness.acp.updates import SessionUpdates
from delegation_core.backend.harness.bridge import Door
from delegation_core.backend.harness.protocol import McpServer, MessageChunk, PromptEnded, Role, ThoughtChunk
from delegation_core.backend.harness.render import Renderer
from delegation_core.backend.harness.settings import HarnessSettings
from delegation_core.config import CoreSettings

FIXTURES = Path(__file__).parent / "fixtures"
SHIPPED = CoreSettings.load(Path(__file__).resolve().parents[5] / "src/examples/frontend_backend_live/core.yaml")
# The harness's prompts as core.yaml words them: every harness block carries them.
PROMPTS = dict(SHIPPED.backend.blocks["harness"]["prompts"])
# The agent classes' blocks as core.yaml words them: Hermes's holds its command guard, which has no default.
AGENT_BLOCKS = {name: dict(block) for name, block in SHIPPED.backend.blocks["harness"]["agents"].items()}

# The core's own control tools and conductor prompt, as core.yaml words them.
CONTROL_TOOLS = {
    "start_work": {
        "description": "Start new work.",
        "parameters": {
            "type": "object",
            "properties": {"delegations": {"type": "array", "items": {"type": "string"}}, "tasks": {"type": "array"}},
        },
    },
    "steer_work": {
        "description": "Join open work.",
        "parameters": {
            "type": "object",
            "properties": {
                "work_id": {"type": "string"},
                "delegations": {"type": "array", "items": {"type": "string"}},
                "task_id": {"type": "string"},
            },
        },
    },
    "cancel_work": {
        "description": "Stop open work.",
        "parameters": {"type": "object", "properties": {"work_id": {"type": "string"}, "task_id": {"type": "string"}}},
    },
    "answer_status": {
        "description": "Answer from the records.",
        "parameters": {
            "type": "object",
            "properties": {"delegation_id": {"type": "string"}, "text": {"type": "string"}},
        },
    },
}
CONDUCTOR_PROMPT = "You place the caller's requests, {max_running} tasks at once. Never do the work yourself."


def vocabulary() -> ControlVocabulary:
    texts = SHIPPED.texts
    return ControlVocabulary(CONTROL_TOOLS, CONDUCTOR_PROMPT, texts.refusal("idle"), texts.failed_output)


def hermes(**block: Any) -> HermesAgent:
    """The Hermes agent class with core.yaml's block, ``block`` over it."""
    return HermesAgent.from_settings({**AGENT_BLOCKS["hermes"], **block})


def wire(scenario: str) -> list[dict]:
    """A recorded scenario of the real Hermes wire log (``fixtures/hermes_wire.json``)."""
    return json.loads((FIXTURES / "hermes_wire.json").read_text(encoding="utf-8"))[scenario]


def settings(**block: Any) -> HarnessSettings:
    """Test settings: core.yaml's prompts and agent blocks; Hermes and its steer unless ``block`` names others."""
    base = {
        "prompts": PROMPTS,
        "agents": AGENT_BLOCKS,
        "agent": "hermes",
        "steer": "native_command",
        "command": ["runtime", "acp"],
        "ready_timeout_seconds": 1,
        "stop_timeout_seconds": 0.5,
    }
    return HarnessSettings.parse({**base, **block})


def plain(**block: Any) -> HarnessSettings:
    """Test settings for a plain ACP agent: no mid-turn steer of its own."""
    return settings(**{"agent": "acp", "steer": "next_prompt", **block})


_SCRATCH: list[Path] = []


def scratch(prefix: str) -> Path:
    """A fresh folder for a test, under one root per test process that is removed when the process ends."""
    if not _SCRATCH:
        root = Path(tempfile.mkdtemp(prefix="harness-tests-"))
        atexit.register(shutil.rmtree, root, True)
        _SCRATCH.append(root)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=_SCRATCH[0]))


class FakeSession:
    """A ``HarnessSession``: what the run sends is recorded; the test ends prompts and puts events in the mailbox."""

    def __init__(self, mailbox: asyncio.Queue, number: int = 1):
        self.id = f"session-{number}"
        self.mailbox = mailbox
        self.sent: list[tuple[int, str, str]] = []  # (prompt number, "prompt" | "steer", text as the run gave it)
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

    # ------------------------------------------------------------------------------------------- the test's side
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
        session = FakeSession(mailbox, len(self.sessions) + 1)
        self.sessions.append(session)
        self.servers.append(server)
        self.folders.append(cwd)
        return session

    async def close(self) -> None:
        self.alive = False


class FakeRuntime:
    def __init__(self):
        self.client = FakeClient()
        self.closed = False
        self.places = 0  # sessions holding one of its turns
        self._folder = scratch("work-")
        self._names: dict[str, int] = {}

    def leave(self) -> None:
        self.places -= 1

    @property
    def folder(self) -> Path:
        return self._folder

    def server_name(self, prefix: str) -> str:
        self._names[prefix] = self._names.get(prefix, 0) + 1
        return f"{prefix}{self._names[prefix]}"

    def session_folder(self, name: str) -> Path:
        folder = self._folder / "sessions" / name
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
        self.kept = None  # the session the session policy keeps for a later task

    async def shared(self, role: Role) -> FakeRuntime:
        self.runtime.places += 1
        return self.runtime

    def take_place(self, runtime: FakeRuntime) -> bool:
        limit = self.settings.turns_per_process
        if (limit is not None and runtime.places >= limit) or not runtime.client.alive:
            return False
        runtime.places += 1
        return True


async def until(condition, timeout: float = 2.0) -> None:
    """Wait (on the loop) until ``condition()`` is true."""
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.005)
