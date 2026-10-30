# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The cascade's own test doubles: a cascade on a scripted delegation port, with no delegation core."""

from __future__ import annotations

import asyncio
import json
import unittest

from examples.frontend_backend_live.cascade.agent import LiveAgent
from examples.frontend_backend_live.cascade.lines import catalog_lines
from examples.frontend_backend_live.cascade.memory import trim_messages
from examples.frontend_backend_live.cascade.session import CascadeSession
from examples.frontend_backend_live.cascade.settings import CascadeSettings
from examples.frontend_backend_live.cascade.talker.talker import Talker
from examples.frontend_backend_live.cascade.writer.writer import CommentaryWriter
from examples.frontend_backend_live.config_manager.schema import LiveConfig, ModelEndpoint, ReliabilityConfig
from examples.frontend_backend_live.models import Frontend
from examples.frontend_backend_live.prompts import PromptSet, get_prompt_set
from tests.unit.contracts.delegation_port import ScriptedDelegationPort


def prompt_set(version: str = "default") -> PromptSet:
    return get_prompt_set(version)


# The default set, whose words the cascade writes in these tests, and its words for the work note, which tests build
# the lines they expect with.
PROMPTS = prompt_set()
WORK = PROMPTS.work_texts
WORK_IN_PROGRESS, NO_WORK_IN_PROGRESS = PROMPTS.work_in_progress, PROMPTS.no_work_in_progress
ANSWER_READY_LABEL = PROMPTS.answer_ready_label


def work_item(number, state, request, update=""):
    """One entry of the work note, as the default set words it; ``update`` is its worded progress line, if any."""
    return WORK.item.format(number=number, state=state, request=request, update=update)


def running_state(seconds, steps="", ended=()):
    """The state of running work, as the default set words it: the calls that ``ended`` (name, status), then ``steps``.

    ``steps`` are the tools it uses now, when it uses some.
    """
    finished = ", ".join(WORK.ended_call.format(name=name, status=status) for name, status in ended)
    state = WORK.running.format(seconds=seconds) + (WORK.ended.format(steps=finished) if finished else "")
    return state + (WORK.steps.format(steps=steps) if steps else "")


def progress_update(progress):
    """An entry's newest progress line, as the default set words it."""
    return WORK.update.format(progress=progress)


def runtime_data(call: dict) -> str:
    """The runtime data a talker call sends after its conversation (the work line and the cue): its last entry."""
    return call["history"][-1]["content"]


def conversation(call: dict) -> list[dict]:
    """The conversation a talker call sends, without the runtime data after it."""
    return call["history"][:-1]


ENDPOINT = ModelEndpoint(model="test/model", base_url="http://localhost:1/v1")
FAST = ReliabilityConfig(provider_retries=1, talker_attempt_timeout_seconds=1.0, client_timeout_seconds=2.0)
CONFIG = LiveConfig(provider="scripted", slot="llm", reliability=FAST)
QUIET = CascadeSettings.parse({"delegation": {"progress_after_seconds": None}})


class ScriptedFrontend(Frontend):
    """A model that plays a script: each call pops the next decision, raises it, or waits for it (a future)."""

    def __init__(self, script, reliability=FAST):
        super().__init__(ENDPOINT, reliability)
        self.script = list(script)
        self.calls: list[dict] = []

    async def complete_once(self, instructions, history, spec):
        self.calls.append(
            {
                "instructions": instructions,
                "history": json.loads(json.dumps(history)),
                "actions": spec.actions,
                "role": spec.role,
            }
        )
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, asyncio.Future):
            return await step
        return step


def agent(frontend: Frontend, version: str = "default") -> LiveAgent:
    prompts = prompt_set(version)
    return LiveAgent(frontend, prompts, Talker(frontend, prompts), CommentaryWriter(frontend, prompts))


class Buffer:
    """The host's turn buffer: what Pipecat's user aggregator writes (the seed, then each committed caller turn)."""

    def __init__(self, messages=()):
        self.items = list(messages)
        # The seed's system messages are the prompt that every trim keeps, as in ``ContextConversation``.
        self.prompt = next(
            (i for i, m in enumerate(self.items) if m["role"] not in ("system", "developer")), len(self.items)
        )

    def messages(self):
        return list(self.items)

    def trim(self, limit):
        self.items = trim_messages(self.items, self.prompt, limit)

    def turn(self, text):
        self.items.append({"role": "user", "content": text})


class RecordingDisplay:
    def __init__(self):
        self.calls: list[tuple] = []

    def started(self, delegation_id, request):
        self.calls.append(("started", delegation_id, request))

    def progress(self, delegation_id, line):
        self.calls.append(("progress", delegation_id, line))

    def shown(self, delegation_id, item):
        self.calls.append(("shown", delegation_id, item.get("name")))

    def answered(self, result):
        self.calls.append(("answered", result.delegation_id, result.speak))

    def spoken(self, delegation_ids, words):
        self.calls.append(("spoken", tuple(delegation_ids), words))

    def ended(self, delegation_id, how):
        self.calls.append(("ended", delegation_id, how.value))

    def kinds(self):
        return [call[0] for call in self.calls]


class RecordingResults:
    """The reader's output, as the presenter takes it."""

    def __init__(self):
        self.calls: list[tuple] = []

    def ready(self, result):
        self.calls.append(("ready", result.delegation_id, result.speak))

    def notice(self, delegation_id, text):
        self.calls.append(("notice", delegation_id, text))

    def ended(self, delegation_id):
        self.calls.append(("ended", delegation_id))

    def handed_over(self, delegation_id, taker):
        self.calls.append(("handed_over", delegation_id, taker))


# ----------------------------------------------------------------------------------------------- official events
def response(response_id, status, **fields):
    return {"id": response_id, "object": "response", "status": status, "output": [], **fields}


def created(response_id, status="queued"):
    return {"type": "response.created", "response": response(response_id, status)}


def queued(response_id):
    return {"type": "response.queued", "response": response(response_id, "queued")}


def in_progress(response_id, previous=None):
    extra = {"previous_response_id": previous} if previous else {}
    return {"type": "response.in_progress", "response": response(response_id, "in_progress", **extra)}


def message(text, phase="final_answer", item_id="msg_1", status="completed"):
    part = {"type": "output_text", "text": text, "annotations": []}
    item = {"id": item_id, "type": "message", "role": "assistant", "status": status, "content": [part]}
    if phase:
        item["phase"] = phase
    return {"type": "response.output_item.done", "item": item}


def function_call(name, call_id="call_1"):
    item = {"id": f"fc_{call_id}", "type": "function_call", "call_id": call_id, "name": name, "arguments": "{}"}
    return {"type": "response.output_item.done", "item": {**item, "status": "completed"}}


def mcp_call(label, name, item_id="mcp_1", output="{}", status="completed"):
    item = {"id": item_id, "type": "mcp_call", "server_label": label, "name": name, "arguments": "{}"}
    return {"type": "response.output_item.done", "item": {**item, "output": output, "status": status}}


def mcp_started(label, name, item_id="mcp_1"):
    """A backend call starts, as the IM writes it: its item ``in_progress``, with no output yet."""
    item = {"id": item_id, "type": "mcp_call", "server_label": label, "name": name, "arguments": "{}"}
    return {"type": "response.output_item.added", "item": {**item, "output": None, "status": "in_progress"}}


def completed(response_id):
    return {"type": "response.completed", "response": response(response_id, "completed")}


def failed(response_id, code="server_error"):
    error = {"code": code, "message": "The backend failed."}
    return {"type": "response.failed", "response": response(response_id, "failed", error=error)}


def incomplete(response_id, reason=None):
    if reason is None:
        return {"type": "response.incomplete", "response": response(response_id, "cancelled", incomplete_details=None)}
    details = {"reason": reason}
    return {"type": "response.incomplete", "response": response(response_id, "incomplete", incomplete_details=details)}


def answered_events(response_id, *texts, start=True):
    """A response that opens queued, runs, and answers with each of ``texts``."""
    events = [created(response_id), queued(response_id), in_progress(response_id)] if start else []
    events += [message(text, item_id=f"msg_{response_id}_{n}") for n, text in enumerate(texts)]
    return [*events, completed(response_id)]


async def settle(predicate, timeout=2.0):
    """Wait until ``predicate()`` holds."""
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


# ----------------------------------------------------------------------------------------------- a whole call
class Call(unittest.IsolatedAsyncioTestCase):
    """A cascade session on a scripted delegation port: the test speaks for the caller and for the IM."""

    async def call(
        self,
        script=(),
        *,
        seed=(),
        port=None,
        settings=QUIET,
        config=CONFIG,
        reports_playout=False,
        say=None,
        display=None,
        lines=None,
        version="default",
    ):
        self.frontend = ScriptedFrontend(script)
        self.port = port or ScriptedDelegationPort()
        self.buffer = Buffer([{"role": "system", "content": "You are Remy."}, *seed])
        self.spoken: list[str] = []
        self.events: list[tuple[str, dict]] = []
        self.display = display or RecordingDisplay()

        async def spoken(text):
            self.spoken.append(text)

        def observe(kind, **fields):
            self.events.append((kind, fields))

        self.session = CascadeSession(
            agent(self.frontend, version),
            config,
            self.buffer,
            delegation=self.port,
            say=say or spoken,
            observe=observe,
            display=self.display,
            settings=settings,
            lines=lines,
            reports_playout=reports_playout,
        )
        self.session.start()
        self.addAsyncCleanup(self.session.close)
        return self.session

    async def turn(self, text):
        """The caller says ``text``: the stop takes it, the context frame decides it."""
        self.buffer.turn(text)
        self.session.caller_started()
        self.session.caller_stopped()
        await self.session.on_turn()

    def emit(self, delegation_id, *events):
        for event in events:
            self.port.emit(delegation_id, event)

    def kinds(self):
        return [kind for kind, _ in self.events]

    async def said(self, count, timeout=2.0):
        await settle(lambda: len(self.spoken) >= count, timeout)
        return self.spoken

    def memory(self):
        return [(m["role"], m["content"]) for m in self.session.memory.messages()]

    @property
    def lines(self):
        return catalog_lines()
