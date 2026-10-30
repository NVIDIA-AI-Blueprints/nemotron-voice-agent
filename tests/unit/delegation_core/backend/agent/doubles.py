# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""Doubles for the backend LLM agent's tests: a scripted model, control-tool rounds and agent builders."""

from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from delegation_core.backend.agent import AgentSettings, AgentSource, ConductorSettings, LlmConductor
from delegation_core.backend.agent.settings import WHEN_IDLE
from delegation_core.backend.control import ControlVocabulary
from delegation_core.backend.llm import LlmSettings
from delegation_core.backend.port import Request, TaskSummary, WorkSummary
from delegation_core.backend.providers import Backend, ProviderLimits
from delegation_core.backend.providers.translation import item_text
from delegation_core.config import CoreSettings, ModelEndpoint
from tests.unit.contracts.backend import a_round, call_events

FBL = Path(__file__).resolve().parents[5] / "src/examples/frontend_backend_live"
SHIPPED = CoreSettings.load(FBL / "core.yaml")
TEXTS = SHIPPED.texts
VOCABULARY = ControlVocabulary(TEXTS.control_tools, TEXTS.conductor_prompt, TEXTS.refusal("idle"), TEXTS.failed_output)
ENDPOINT = ModelEndpoint(model="test/model", base_url="http://localhost:1/v1", api_key="secret-key")
OTHER = ModelEndpoint(model="other/model", base_url="http://localhost:2/v1")
LIMITS = ProviderLimits(provider_retries=0, first_event_timeout_seconds=2.0, idle_timeout_seconds=2.0)
# Workers that give up fast, so a test never waits on a limit.
FAST_WORKER = LlmSettings(
    first_event_timeout_seconds=0.5,
    idle_timeout_seconds=0.5,
    server_tool_timeout_seconds=1.0,
    stop_timeout_seconds=0.5,
    provider_retries=0,
)
# The model conductor places every request, also while no Work is open, so the tests read its rounds; the idle rule
# (conductor.when_idle) has tests of its own.
CONDUCTOR = ConductorSettings(when_idle=WHEN_IDLE["decide"])
SETTINGS = AgentSettings(conductor=CONDUCTOR, worker=FAST_WORKER)


def lookup(model: str | None) -> ModelEndpoint:
    """The catalog: the default model and one other; any other model is not listed."""
    listed = {ENDPOINT.model: ENDPOINT, OTHER.model: OTHER}
    if model is None:
        return ENDPOINT
    if model not in listed:
        raise LookupError(f"Model {model!r} is not in the catalog")
    return listed[model]


# --------------------------------------------------------------------------------------------- the model
class Model(Backend):
    """A scripted model: each ``stream()`` call plays the next scripted round, or ``respond(payload)``'s."""

    name = "scripted"

    def __init__(self, *rounds: list, respond: Callable[[dict], list] | None = None):
        super().__init__(ENDPOINT, LIMITS)
        self.rounds, self.respond = list(rounds), respond
        self.payloads: list[dict] = []
        self.closed = 0

    async def stream(self, payload):
        self.payloads.append(json.loads(json.dumps(payload)))
        steps = self.respond(payload) if self.respond is not None else self.rounds.pop(0)
        for step in steps:
            if isinstance(step, BaseException):
                raise step
            if isinstance(step, asyncio.Event):
                await step.wait()
                continue
            yield step

    async def close(self):
        self.closed += 1


# --------------------------------------------------------------------------------------------- the conductor's model
_rounds = itertools.count(1)


def decisions(*calls: tuple[str, dict]) -> list[dict]:
    """One conductor round whose model calls the control tools ``(name, arguments)`` at once, in this order."""
    number = next(_rounds)
    parts = [
        call_events(name, arguments, call_id=f"c{number}_{index}", index=index)
        for index, (name, arguments) in enumerate(calls)
    ]
    return a_round(*parts, response_id=f"resp_c{number}")


def start(*delegations: str, tasks: Sequence[str] | None = None) -> tuple[str, dict]:
    arguments: dict[str, Any] = {"delegations": list(delegations)}
    if tasks is not None:
        arguments["tasks"] = list(tasks)
    return "start_work", arguments


def steer(work_id: str, *delegations: str, task_id: str | None = None) -> tuple[str, dict]:
    arguments: dict[str, Any] = {"work_id": work_id, "delegations": list(delegations)}
    if task_id is not None:
        arguments["task_id"] = task_id
    return "steer_work", arguments


def cancel(work_id: str, task_id: str | None = None) -> tuple[str, dict]:
    return "cancel_work", {"work_id": work_id, **({"task_id": task_id} if task_id is not None else {})}


def answer_status(delegation_id: str, text: str) -> tuple[str, dict]:
    return "answer_status", {"delegation_id": delegation_id, "text": text}


def round_data(payload: dict) -> dict:
    """What a conductor round's input carries: the JSON of its last message."""
    return json.loads(payload["input"][-1]["content"][0]["text"])


def start_each(payload: dict) -> list[dict]:
    """A conductor model that starts one Work for each new request it is given."""
    return decisions(*(start(request["delegation_id"]) for request in round_data(payload)["new_requests"]))


def conductor_on(model: Model, **settings: Any) -> LlmConductor:
    """An ``LlmConductor`` on ``model``, with the shipped vocabulary and ``settings`` over ``CONDUCTOR``."""
    conductor = replace(CONDUCTOR, **settings)
    return LlmConductor(lambda: model, VOCABULARY, conductor, ENDPOINT.model, SETTINGS.max_running)


# --------------------------------------------------------------------------------------------- the workers' model
def texts_of(payload: dict) -> list[str]:
    """The text of each message of a request's input, oldest first."""
    return [item_text(item.get("content")) for item in payload["input"] if item.get("type", "message") == "message"]


def by_brief(scripts: dict[str, list[list]]) -> Callable[[dict], list]:
    """A workers' model that plays each task the next round of its brief's script (the newest brief it reads)."""
    lines = {TEXTS.task_line.format(task=brief): brief for brief in scripts}

    def respond(payload: dict) -> list:
        brief = next(lines[text] for text in reversed(texts_of(payload)) if text in lines)
        return scripts[brief].pop(0)

    return respond


# --------------------------------------------------------------------------------------------- the agent
def source(conductor: Model, workers: Model, *, settings: AgentSettings = SETTINGS, tools=None) -> AgentSource:
    """An ``AgentSource`` whose conductor runs on ``conductor`` and whose workers run on ``workers``."""
    return AgentSource(
        settings,
        endpoints=lookup,
        tools=lambda: list(tools or ()),
        vocabulary=VOCABULARY,
        make_provider=lambda endpoint: workers,
        make_conductor_provider=lambda endpoint: conductor,
    )


# --------------------------------------------------------------------------------------------- what a conductor reads
def a_request(delegation_id: str, text: str) -> Request:
    return Request(delegation_id, ({"type": "message", "role": "user", "content": text},))


def a_work(work_id: str, *task_states: str, state: str = "running", delegation_id: str = "D1") -> WorkSummary:
    """A Work carrying ``delegation_id``, with one task per state (``W1.1``, ``W1.2`` ...)."""
    tasks = tuple(
        TaskSummary(f"{work_id}.{number}", task_state, None, ("get_menu: done",), True, "Looking.", 2.25)
        for number, task_state in enumerate(task_states or (state,), start=1)
    )
    return WorkSummary(work_id, state, delegation_id, (a_request(delegation_id, "Find teas."),), tasks, 3.04)
