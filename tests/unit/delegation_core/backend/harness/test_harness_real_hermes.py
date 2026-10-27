# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Tests of the harness against a real ``hermes acp``, deployed as its operator would deploy it.

The contracts and the core's flows run on a scripted model endpoint (``scripted_model.py``); one small run uses the
real model of an operator-configured Hermes home, and nothing here reads its key.

Skipped unless ``HARNESS_HERMES_BIN`` names a ``hermes`` executable; the real-model run also needs
``HARNESS_HERMES_HOME`` (the configured home) and writes its evidence to ``HARNESS_EVIDENCE`` when set. Set
``HARNESS_TRANSPORT=websocket`` to serve Hermes over WebSocket (``served_agent.py``) instead of ``stdio``.
"""

import asyncio
import dataclasses
import json
import os
import re
import subprocess
import time
import unittest
from pathlib import Path
from types import MappingProxyType

from delegation_core.app import build_core
from delegation_core.backend.harness import HarnessSource
from delegation_core.config import CoreSettings, ModelEndpoint
from delegation_core.im.input import BackendTask
from tests.unit.contracts.backend import (
    MENU_FUNCTION,
    BackendAdapterContract,
    ConductorContract,
    check_events,
)
from tests.unit.delegation_core.backend.harness.doubles import scratch, vocabulary
from tests.unit.delegation_core.backend.harness.scripted_model import (
    MODEL,
    Call,
    Fail,
    Hold,
    Say,
    ScriptedModel,
    after_tool,
    offers,
    user_text,
)
from tests.unit.delegation_core.backend.harness.served_agent import ServedAgent

REPO = Path(__file__).resolve().parents[5]
FBL = REPO / "src" / "examples" / "frontend_backend_live"
HERMES = os.environ.get("HARNESS_HERMES_BIN", "")
REAL_HOME = os.environ.get("HARNESS_HERMES_HOME", "")
EVIDENCE = os.environ.get("HARNESS_EVIDENCE", "")
SERVED = os.environ.get("HARNESS_TRANSPORT", "stdio") == "websocket"
HAVE_HERMES = bool(HERMES) and os.access(HERMES, os.X_OK)
SKIP_HERMES = (
    "the real Hermes is not available: set HARNESS_HERMES_BIN to a `hermes` executable at af26aca with its ACP and "
    "MCP extras"
)
SKIP_MODEL = (
    "the real model needs its operator's Hermes: set HARNESS_HERMES_BIN to a `hermes` executable at af26aca, and "
    "HARNESS_HERMES_HOME to a Hermes home configured with the real model and its key"
)
# The model name the core serves on the real model; empty: the model the home runs.
REAL_MODEL = os.environ.get("HARNESS_REAL_MODEL", "")
GET_TIME = {
    "type": "function",
    "name": "get_time",
    "description": "The current local time on the caller's device.",
    "parameters": {"type": "object", "properties": {}},
}
# A worker model's private reasoning, which must never reach the caller.
REASONING = "I should check the account balance first, then answer."
# The operator's Hermes configuration for the scripted model, as the architecture page's "Deploy Hermes" gives it: only
# the session's MCP server reaches a session, nothing of the host's logins, no background model calls, and an MCP tool
# timeout above the IM's client-tool deadline.
HERMES_CONFIG = """\
model:
  provider: nvidia
  default: {model}
  base_url: {base_url}
  context_length: 131072
platform_toolsets:
  acp: [no_mcp]
tools:
  tool_search:
    enabled: "off"
agent:
  coding_context: "off"
  task_completion_guidance: false
  parallel_tool_call_guidance: false
  tool_use_enforcement: false
  execution_guidance: false
memory:
  memory_enabled: false
  user_profile_enabled: false
auxiliary:
  title_generation:
    enabled: false
  background_review:
    enabled: false
auth:
  adopt_external_logins: false
timeouts:
  mcp:
    tool_call: 3600
"""
# The home's identity (SOUL.md), which every session reads first: without one, Hermes writes its own persona there.
HERMES_SOUL = (
    "You work behind a voice assistant, for its callers. Follow the instructions of your project context exactly, use "
    "only the tools you are given, and do not narrate your tool calls.\n"
)


def hermes_home(base_url: str) -> Path:
    """A Hermes home as its operator would set it up for the scripted model, removed when the test process ends."""
    home = scratch("hermes-home-")
    config = HERMES_CONFIG.format(model=json.dumps(MODEL), base_url=json.dumps(base_url))
    (home / "config.yaml").write_text(config, encoding="utf-8")
    (home / "SOUL.md").write_text(HERMES_SOUL, encoding="utf-8")
    return home


class Operator:
    """The operator's side: Hermes with a home of its own on the scripted model, started by command or served."""

    def __init__(self, model: ScriptedModel):
        self.home = hermes_home(model.base_url)
        # The deployment's own environment: its home and a key for the scripted endpoint.
        self.env = {"HERMES_HOME": str(self.home), "NVIDIA_API_KEY": "not-needed"}
        self.served: ServedAgent | None = None
        self.url: str | None = None

    async def start(self) -> "Operator":
        if SERVED:
            # The operator's server starts Hermes itself, so it also skips the configured MCP servers itself.
            environment = {"PATH": os.environ.get("PATH", ""), **self.env, "HERMES_ACP_SKIP_CONFIGURED_MCP": "1"}
            self.served = ServedAgent([HERMES, "acp"], environment, self.home)
            self.url = await self.served.start()
        return self

    async def stop(self) -> None:
        if self.served is not None:
            await self.served.stop()

    def reach(self) -> dict:
        """How the harness reaches this Hermes: the URL it is served on, or the command and its environment."""
        if self.url is not None:
            return {"transport": "websocket", "url": self.url, "command": [], "env": {}}
        return {"transport": "stdio", "command": [HERMES, "acp"], "env": dict(self.env)}

    def block(self, **over) -> dict:
        """The harness block of a test: Hermes's agent class, the way in, a work root, the bridge, the limits."""
        return {
            "agent": "hermes",
            "steer": "native_command",
            **self.reach(),
            "work_root": str(scratch("harness-real-")),
            "bridge": {"bind": "127.0.0.1:0", "advertise": None},
            "start_timeout_seconds": 120,
            "ready_timeout_seconds": 30,
            "turn_timeout_seconds": 120,
            "conductor_timeout_seconds": 90,
            **over,
        }


def scripted_endpoint(model: ScriptedModel) -> ModelEndpoint:
    return ModelEndpoint(model=MODEL, base_url=model.base_url, api_key="not-needed")


class Shop:
    """A tool source whose ``lookup`` answers (or fails)."""

    label = "shop"

    def __init__(self, fail=()):
        self.fail = set(fail)

    def tools(self):
        return [
            {
                "type": "function",
                "name": "lookup",
                "description": "Look up a record.",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

    async def execute(self, call_id, name, arguments):
        if name in self.fail:
            raise RuntimeError("the lookup service is down")
        return json.dumps({"record": "found"})


def section(text: str, start: str, end: str) -> str:
    head = text.rsplit(start, 1)[-1] if start in text else ""
    return head.split(end, 1)[0]


def conductor_model(body: dict):
    """The conductor's scripted model: one control call per new request, by what the request says; then a word."""
    if after_tool(body):
        return Say("Placed.")
    text = user_text(body)
    new = section(text, "New requests to place:", "Work records:")
    ids = re.findall(r"\[(\w+)\]", new)
    works = [
        work
        for work, _ in re.findall(
            r"\[(W\d+)\] (queued|running|waiting_client)", section(text, "Work records:", "Conversation since")
        )
    ]
    if "Stop that" in new:
        return Call(
            [
                ("__cancel_work", {"work_id": works[0]}),
                ("__answer_status", {"delegation_id": ids[0], "text": "Stopping that."}),
            ]
        )
    if "How is it going" in new:
        return Call([("__answer_status", {"delegation_id": ids[0], "text": "Still looking it up."})])
    if "the hours, please" in new and works:  # a model that names a Work it was not shown: the IM refuses it
        return Call([("__steer_work", {"work_id": "W9", "delegations": ids})])
    if "opening hours" in new:
        return Call([("__start_work", {"delegations": ids, "tasks": ["find teas", "find the opening hours"]})])
    if works:
        return Call([("__steer_work", {"work_id": works[0], "delegations": ids})])
    return Call([("__start_work", {"delegations": ids})])


class Workers:
    """The workers' scripted model, by what each request says; ``hold`` is the next plain request's reply."""

    def __init__(self):
        self.hold: Hold | None = None
        self.holds: list[Hold] = []  # every hold given, released at the test's end

    def holding(self, then: Say | Call) -> Hold:
        """Hold the next plain request until the test releases it, then play ``then``; the test keeps the hold."""
        self.hold = Hold(then=then)
        self.holds.append(self.hold)
        return self.hold

    def __call__(self, body: dict):
        if offers(body, "__start_work"):
            return conductor_model(body)
        text = user_text(body)
        if "Make it two" in text:
            return Say("Two teas: green and black.")
        if "Task: find the opening hours" in text:
            return Say("Open from eight to six.")
        if "Task: find teas" in text:
            return Say("Green tea and black tea.")
        if "Check my account" in text:
            return Fail(403)
        if "Check my balance" in text:  # a reasoning model that never writes its answer
            return Say("", reasoning=REASONING)
        if "What teas" in text:
            return (
                Say("We have green tea and black tea.")
                if after_tool(body)
                else Call([("__get_menu", {})], text="Checking the menu.")
            )
        if self.hold is not None:
            hold, self.hold = self.hold, None
            return hold
        return Say("Done.")


# --------------------------------------------------------------------------------------------- the contracts
@unittest.skipUnless(HAVE_HERMES, SKIP_HERMES)
class RealHermesAdapterContract(BackendAdapterContract, unittest.IsolatedAsyncioTestCase):
    """The backend adapter contract on a real ``hermes acp`` whose model endpoint is scripted per scenario."""

    async def asyncSetUp(self):
        self.model = await ScriptedModel().start()
        self.operator = await Operator(self.model).start()
        self.built: list = []
        self.hold = Hold(then=Say("Old."))

    async def asyncTearDown(self):
        self.hold.event.set()
        for source, adapter in self.built:
            await adapter.close()
            await source.aclose()
        await self.operator.stop()
        await self.model.stop()

    def make_adapter(self, scenario):
        script = {
            "answer": [Say("Done.")],
            "own_call": [Call([("__lookup", {})]), Say("Done.")],
            "tool_fails": [Call([("__lookup", {})]), Say("Done.")],
            "app_call": [Call([("__get_menu", {})], text="Let me look."), Say("Done.")],
            "slow": [self.hold, Say("Done.")],
            "tail_error": [Say("Done.")],
        }[scenario]
        self.model.add(*script)
        tools = {"own_call": [Shop()], "tool_fails": [Shop(fail=["lookup"])]}.get(scenario, [])
        source = HarnessSource.from_settings(
            self.operator.block(),
            endpoints=lambda name: scripted_endpoint(self.model),
            tools=lambda: tools,
            vocabulary=vocabulary(),
        )
        adapter = source.new()
        self.built.append((source, adapter))
        return adapter

    def release(self):
        self.hold.event.set()

    def sent_instructions(self):
        # The runtime builds its own system prompt around a session's instructions (its identity, the context file,
        # its environment lines): a model call was sent the composed instructions when its system prompt holds them
        # whole, under the context file's heading.
        marker = "## .hermes.md\n\n"
        return [
            prompt.split(marker, 1)[1].split("\n\n", 1)[0] if marker in prompt else prompt
            for prompt in self.model.system_prompts()
        ]

    # The bridge mints each call's id (unique in the IM session: the runtime's MCP call carries none); the contract
    # pins a model provider's own "call_1", so these two read the id the task was told.
    async def test_own_calls_are_reported_before_and_after(self):
        task, events = await self.run_task("own_call")
        self.assertEqual(check_events(events), [])
        [(call_id, label)] = task.started
        self.assertEqual(label, "shop")
        self.assertEqual(task.order, [f"started:{call_id}", f"outcome:{call_id}"])
        self.assertEqual(task.outcomes[call_id][1], "done")

    async def test_a_failed_tool_keeps_its_output_and_the_task_goes_on(self):
        task, events = await self.run_task("tool_fails")
        [(call_id, _)] = task.started
        output, outcome = task.outcomes[call_id]
        self.assertEqual(outcome, "failed")
        self.assertIn("the lookup service is down", output)
        self.assertEqual(events[-1]["type"], "response.completed")


@unittest.skipUnless(HAVE_HERMES, SKIP_HERMES)
class RealHermesConductorContract(ConductorContract, unittest.IsolatedAsyncioTestCase):
    """The conductor contract on a real Hermes session whose model places by what each request says."""

    async def asyncSetUp(self):
        self.model = await ScriptedModel(policy=conductor_model).start()
        self.operator = await Operator(self.model).start()
        self.built: list = []

    async def asyncTearDown(self):
        for source, adapter in self.built:
            await adapter.close()
            await source.aclose()
        await self.operator.stop()
        await self.model.stop()

    def make_conductor(self):
        source = HarnessSource.from_settings(
            self.operator.block(),
            endpoints=lambda name: scripted_endpoint(self.model),
            tools=lambda: [],
            vocabulary=vocabulary(),
        )
        adapter = source.new()
        self.built.append((source, adapter))
        return adapter.conductor()

    async def test_the_conductors_tools_are_exactly_the_control_tools_and_its_text_never_the_envelope(self):
        conductor = self.make_conductor()
        await asyncio.wait_for(conductor.decide(self.given(*self._requests()), _Target().place), 120)
        [first, *_] = [body for body in self.model.requests]
        names = sorted(tool["function"]["name"] for tool in first["tools"])
        self.assertEqual(
            names, [f"mcp__c1__{name}" for name in ("answer_status", "cancel_work", "start_work", "steer_work")]
        )
        system = first["messages"][0]["content"]
        self.assertIn(vocabulary().prompt, system)  # its instructions: the conductor prompt, from its own folder

    def _requests(self):
        from tests.unit.contracts.backend import a_request

        return (a_request("d_1", "Find teas."),)


class _Target:
    def __init__(self):
        self.decisions = []

    async def place(self, decision):
        self.decisions.append(decision)
        return "Started W1 with W1.1."


# --------------------------------------------------------------------------------------------- the core's flows
def build(model: ScriptedModel | None, *, block: dict, defaults=None, observe=None, endpoint=None):
    """A delegation core from ``core.harness.yaml`` on the real Hermes; only the harness block's test keys change.

    On the scripted model, the core serves its name; on the real model (``model`` None), ``endpoint`` names it.
    """
    settings = CoreSettings.load(FBL / "core.harness.yaml")
    harness = {**settings.backend.blocks["harness"], **block}
    blocks = MappingProxyType({**settings.backend.blocks, "harness": MappingProxyType(harness)})
    settings = dataclasses.replace(settings, backend=dataclasses.replace(settings.backend, blocks=blocks))
    endpoint = scripted_endpoint(model) if model is not None else endpoint
    kwargs = {"observe": observe} if observe is not None else {}
    return build_core(
        settings,
        endpoints=lambda name: endpoint,
        defaults=defaults or BackendTask(MODEL, "You help the callers of a tea shop.", [MENU_FUNCTION]),
        **kwargs,
    )


def home_model(home: str) -> dict:
    """What a Hermes home runs: its ``model.default`` and ``model.base_url``, as Hermes itself reads them.

    ``hermes config get`` prints the home's resolved setting, credentials masked; nothing here reads the key.
    """
    printed = subprocess.run(
        [HERMES, "config", "get", "model", "--json"],
        env={"PATH": os.environ.get("PATH", ""), "HERMES_HOME": home},
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    ).stdout
    model = json.loads(next(line for line in printed.splitlines() if line.strip()))
    model = model if isinstance(model, dict) else {"default": model}
    return {"model": str(model.get("default") or ""), "base_url": str(model.get("base_url") or "")}


async def read(stream, limit=180):
    """A response's events to its terminal."""
    events = []
    async with asyncio.timeout(limit):
        async for event in stream:
            events.append(event)
    return events


def done_items(events, kind):
    return [e["item"] for e in events if e["type"] == "response.output_item.done" and e["item"]["type"] == kind]


def spoken(events, phase="final_answer"):
    return [i["content"][0]["text"] for i in done_items(events, "message") if i.get("phase") == phase]


def message(text):
    return [{"type": "message", "role": "user", "content": text}]


@unittest.skipUnless(HAVE_HERMES, SKIP_HERMES)
class RealHermesCoreFlows(unittest.IsolatedAsyncioTestCase):
    """The IM, unchanged, on the harness: ``backend.adapter: harness`` is the only switch."""

    async def asyncSetUp(self):
        self.workers = Workers()
        self.model = await ScriptedModel(policy=self.workers).start()
        self.operator = await Operator(self.model).start()
        self.core = build(self.model, block=self.operator.block())
        self.session = self.core.session()
        self.session.start()

    async def asyncTearDown(self):
        for hold in self.workers.holds:
            hold.event.set()
        await self.session.close()
        await self.core.aclose()
        await self.operator.stop()
        await self.model.stop()

    async def test_a_delegation_starts_work_whose_app_call_goes_to_the_app_and_whose_answer_follows(self):
        first = await read(self.session.admit(message("What teas do you have?"), delegation_id="d1").stream)
        [call] = done_items(first, "function_call")
        self.assertEqual(call["name"], "get_menu")
        self.assertEqual(spoken(first, "commentary"), ["Checking the menu."])
        self.assertEqual(first[-1]["type"], "response.completed")
        output = [
            {"type": "function_call_output", "call_id": call["call_id"], "output": '{"teas": ["green", "black"]}'}
        ]
        rest = await read(self.session.admit_client_request(output, delegation_id="d1").stream)
        self.assertEqual(spoken(rest), ["We have green tea and black tea."])
        self.assertEqual(rest[-1]["type"], "response.completed")

    async def until_running(self):
        async with asyncio.timeout(120):
            while not any(
                work.record.state == "running" and work.tasks[0].run is not None for work in self.session.works.values()
            ):
                await asyncio.sleep(0.05)

    async def test_a_newer_request_joins_the_running_work_and_carries_its_answer(self):
        hold = self.workers.holding(Say("Old answer."))
        d1 = self.session.admit(message("Look up the teas."), delegation_id="d1")
        await self.until_running()
        await asyncio.sleep(1)
        d2 = self.session.admit(message("Make it two."), delegation_id="d2")
        async with asyncio.timeout(120):
            while not self.session.ledger.delegations["d2"].work:  # "" until a decision places it
                await asyncio.sleep(0.05)
        hold.event.set()
        older, newer = await read(d1.stream), await read(d2.stream)
        self.assertEqual(spoken(older), [])  # the older request handed its work over
        # The answer that crossed the newer request is only commentary; the handover line ends the older response.
        self.assertEqual(spoken(older, "commentary")[-1], self.core.settings.texts.handover_line)
        self.assertEqual(spoken(newer), ["Two teas: green and black."])
        taker = next(e for e in newer if e["type"] == "response.in_progress")
        self.assertEqual(taker["response"]["previous_response_id"], older[0]["response"]["id"])

    async def test_no_call_is_dispatched_while_a_newer_request_waits_to_join(self):
        # The commit gate on the real runtime: the model calls the app's get_menu while the newer request waits to land
        # in the Work. The call is not run (Hermes gets the not-run text), nothing reaches the app, and the Work answers
        # with the newer request taken in.
        hold = self.workers.holding(Call([("__get_menu", {})]))
        d1 = self.session.admit(message("Look up the teas."), delegation_id="d1")
        await self.until_running()
        await asyncio.sleep(1)
        d2 = self.session.admit(message("Make it two."), delegation_id="d2")
        async with asyncio.timeout(120):
            while not self.session.ledger.delegations["d2"].work:  # "" until a decision places it
                await asyncio.sleep(0.05)
        hold.event.set()
        older, newer = await read(d1.stream), await read(d2.stream)
        self.assertEqual(done_items(older, "function_call") + done_items(newer, "function_call"), [])
        [action] = [action for action in self.session.ledger.actions.values() if action.name == "get_menu"]
        self.assertEqual(action.state, "not_run")
        self.assertEqual(spoken(newer), ["Two teas: green and black."])

    async def test_a_stop_is_delegated_the_conductor_cancels_and_the_im_confirms(self):
        self.workers.holding(Say("Still counting."))
        d1 = self.session.admit(message("Count the stars."), delegation_id="d1")
        await self.until_running()
        d2 = self.session.admit(message("Stop that."), delegation_id="d2")
        older, newer = await read(d1.stream), await read(d2.stream)
        self.assertEqual(spoken(older), [self.core.settings.texts.cancelled_line])  # stopped, and confirmed
        self.assertEqual(spoken(newer), ["Stopping that."])

    async def test_a_request_split_into_two_tasks_answers_with_each_tasks_own_answer(self):
        events = await read(self.session.admit(message("Find teas and the opening hours."), delegation_id="d1").stream)
        self.assertEqual(sorted(spoken(events)), ["Green tea and black tea.", "Open from eight to six."])
        self.assertEqual(events[-1]["type"], "response.completed")

    async def test_a_status_question_is_answered_from_the_records_while_the_work_runs(self):
        hold = self.workers.holding(Say("Record ABC is a green tea."))
        d1 = self.session.admit(message("Look up record ABC."), delegation_id="d1")
        await self.until_running()
        status = await read(self.session.admit(message("How is it going?"), delegation_id="d2").stream)
        self.assertEqual(spoken(status), ["Still looking it up."])
        hold.event.set()
        self.assertEqual(spoken(await read(d1.stream)), ["Record ABC is a green tea."])

    async def test_the_runtimes_own_error_text_is_never_spoken(self):
        events = await read(self.session.admit(message("Check my account."), delegation_id="d1").stream)
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertEqual(events[-1]["response"]["error"]["code"], "server_error")
        self.assertEqual(done_items(events, "message"), [])  # neither as the answer nor as progress

    async def test_the_models_reasoning_alone_is_never_spoken(self):
        # The model reasons and writes no answer. Hermes then ends the turn with that reasoning as its reply, one chunk
        # under the reasoning's message id ("Reasoning-only clean stop ... returning the reasoning as the final
        # response"). It is the model's private text, never the caller's answer: the task fails instead.
        events = await read(self.session.admit(message("Check my balance."), delegation_id="d1").stream)
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertEqual(done_items(events, "message"), [])
        self.assertNotIn(REASONING, json.dumps(events))

    async def test_a_request_the_conductor_steers_into_work_it_was_not_shown_still_joins_the_open_work(self):
        # The IM refuses the conductor's Steer to an unknown Work; the request must still join the open Work by the
        # steer rule, never start a second Work beside it.
        hold = self.workers.holding(Say("Old answer."))
        d1 = self.session.admit(message("Look up the teas."), delegation_id="d1")
        await self.until_running()
        await asyncio.sleep(1)
        d2 = self.session.admit(message("Also the hours, please."), delegation_id="d2")
        async with asyncio.timeout(120):
            while not self.session.ledger.delegations["d2"].work:  # "" until a decision places it
                await asyncio.sleep(0.05)
        self.assertEqual(self.session.ledger.delegations["d2"].work, "W1")
        self.assertEqual(sorted(self.session.ledger.works), ["W1"])
        hold.event.set()
        older, newer = await read(d1.stream), await read(d2.stream)
        self.assertEqual(spoken(older), [])  # handed over to the newer request
        self.assertEqual(spoken(newer), ["Done."])

    async def test_close_stops_running_work_ends_its_response_and_removes_every_work_folder(self):
        self.workers.holding(Say("Late."))
        d1 = self.session.admit(message("Count the stars."), delegation_id="d1")
        await self.until_running()
        root = Path(self.core.source.settings.work_root)
        self.assertEqual(len([folder for folder in root.iterdir() if folder.name.startswith("work-")]), 2)
        await self.session.close()
        events = await read(d1.stream)
        self.assertEqual(events[-1]["type"], "response.incomplete")
        self.assertEqual(events[-1]["response"]["status"], "cancelled")
        self.assertEqual([folder for folder in root.iterdir() if folder.name.startswith("work-")], [])


# --------------------------------------------------------------------------------------------- the real model
@unittest.skipUnless(HAVE_HERMES and REAL_HOME, SKIP_MODEL)
class RealModelRun(unittest.IsolatedAsyncioTestCase):
    """One small delegation on the real model, on the Hermes home its operator configured (Hermes reads its key)."""

    async def test_a_delegation_with_an_app_call_on_the_real_model(self):
        traces: list = []
        block = {
            "transport": "stdio",
            "command": [HERMES, "acp"],
            "env": {"HERMES_HOME": REAL_HOME},
            "work_root": str(scratch("harness-real-")),
            "bridge": {"bind": "127.0.0.1:0", "advertise": None},
            "start_timeout_seconds": 120,
            "ready_timeout_seconds": 30,
            "turn_timeout_seconds": 120,
            "conductor_timeout_seconds": 90,
        }
        ran = home_model(REAL_HOME)  # what Hermes runs: the model and the route of the operator's home
        served = REAL_MODEL or ran["model"]
        defaults = BackendTask(
            served, "You help the callers of a tea shop. Use the functions you are given.", [GET_TIME]
        )
        core = build(
            None,
            block=block,
            defaults=defaults,
            endpoint=ModelEndpoint(served, ran["base_url"], "not-read"),
            observe=lambda kind, **fields: traces.append((round(time.monotonic(), 3), kind, fields)),
        )
        session = core.session()
        session.start()
        evidence: dict = {**ran, "served_model": served}
        try:
            await asyncio.sleep(15)  # the conductor's and the workers' runtimes start
            started = time.monotonic()
            text = "What time is it on my device? Use the get_time function, then tell me in one short sentence."
            first = await read(session.admit(message(text), delegation_id="d1").stream)
            evidence["first_response"] = [summary(e) for e in first]
            evidence["first_seconds"] = round(time.monotonic() - started, 2)
            calls = done_items(first, "function_call")
            if calls:
                output = [
                    {"type": "function_call_output", "call_id": calls[0]["call_id"], "output": '{"time": "14:07"}'}
                ]
                rest = await read(session.admit_client_request(output, delegation_id="d1").stream)
                evidence["continuation"] = [summary(e) for e in rest]
                answer = spoken(rest)
            else:
                answer = spoken(first)
            evidence["answer"] = answer
            evidence["seconds"] = round(time.monotonic() - started, 2)
            self.assertEqual([call["name"] for call in calls], ["get_time"])
            self.assertEqual(len(answer), 1)
            self.assertRegex(answer[0], r"14:07|2:07")
        finally:
            evidence["traces"] = [
                (t, kind, {k: v for k, v in fields.items() if k != "text"}) for t, kind, fields in traces
            ]
            await session.close()
            await core.aclose()
            if EVIDENCE:
                Path(EVIDENCE).write_text(json.dumps(evidence, indent=1, default=str) + "\n", encoding="utf-8")


def summary(event: dict) -> dict:
    """One event as the evidence keeps it: its type, its item's kind, phase, name and text."""
    item = event.get("item") or {}
    out = {"type": event["type"]}
    if item:
        out |= {"item": item.get("type"), "phase": item.get("phase"), "name": item.get("name")}
        if item.get("type") == "message" and item.get("content"):
            out["text"] = item["content"][0].get("text")
    response = event.get("response") or {}
    if response.get("previous_response_id"):
        out["previous_response_id"] = response["previous_response_id"]
    if response.get("error"):
        out["error"] = response["error"]
    if response.get("usage"):
        out["usage"] = response["usage"]
    return out


if __name__ == "__main__":
    unittest.main()
