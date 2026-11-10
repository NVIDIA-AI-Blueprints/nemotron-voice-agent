# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The harness's settings, source and adapter: every name resolved by a table, every refusal at build, the pool."""

import asyncio
import json
import unittest
from pathlib import Path

from loguru import logger

from examples.frontend_backend_live.delegation_core.backend.conductors import SteerOpenWork
from examples.frontend_backend_live.delegation_core.backend.harness import (
    HarnessAdapter,
    HarnessConductor,
    HarnessSettings,
    HarnessSource,
    WorkerRun,
)
from examples.frontend_backend_live.delegation_core.backend.harness.acp import AcpAgent, AcpProtocol
from examples.frontend_backend_live.delegation_core.backend.harness.acp.hermes import HermesAgent
from examples.frontend_backend_live.delegation_core.backend.harness.acp.transports import StdioTransport
from examples.frontend_backend_live.delegation_core.backend.harness.permissions import Reject
from examples.frontend_backend_live.delegation_core.backend.harness.steering import CancelReprompt, NativeCommand
from examples.shared.delegation_contract import ModelEndpoint
from tests.unit.delegation_core.harness.doubles import (
    AGENT_BLOCKS,
    PROMPTS,
    FakeBridge,
    FakeRuntime,
    scratch,
    until,
    vocabulary,
)
from tests.unit.delegation_core.helpers import CORE_SETTINGS, FakeTaskContext

EXTRA = "{}\n.*Extra inputs are not permitted"
ENDPOINT = ModelEndpoint(model="scripted/model", base_url="http://127.0.0.1:1/v1", api_key="secret-key-value")
UNSTARTABLE = {"command": ["/nonexistent/runtime"], "bridge": {"bind": "127.0.0.1:0", "advertise": None}}


def parse(block):
    """Parse ``block`` with the shipped prompts and agent blocks, which a harness block carries."""
    return HarnessSettings.model_validate({"prompts": PROMPTS, "agents": AGENT_BLOCKS, **block})


def build(block=None):
    """Build the source of ``block`` (Hermes over stdio by default), its work folders under a scratch folder."""
    block = {"prompts": PROMPTS, "agents": AGENT_BLOCKS, **(block or {"command": ["hermes", "acp"], "agent": "hermes"})}
    # A source sweeps its work root when it is built.
    block["work_root"] = str(scratch("work-root-"))
    return HarnessSource.from_settings(
        block, endpoints=lambda model: ENDPOINT, tools=lambda: [], vocabulary=vocabulary()
    )


class SourceTests(unittest.IsolatedAsyncioTestCase):
    def test_the_defaults_are_a_plain_acp_agent_over_stdio(self):
        settings = parse({"command": ["some-agent", "--acp"]})
        self.assertIsInstance(settings.protocol, AcpProtocol)
        self.assertIs(settings.protocol.transport, StdioTransport)
        self.assertIs(type(settings.agent), AcpAgent)
        self.assertIsInstance(settings.steering, CancelReprompt)
        self.assertIsInstance(settings.permission, Reject)
        self.assertEqual((settings.max_running, settings.turns_per_process), (3, None))
        self.assertEqual(settings.bridge.url(8791), "http://127.0.0.1:8791")
        self.assertEqual(settings.work_root, Path("~/.cache/delegation-core/harness").expanduser())
        self.assertEqual((settings.env, settings.pass_env), ({}, ("PATH", "LANG", "LC_ALL", "TZ")))

    def test_the_description_shows_the_url_without_credentials(self):
        settings = parse({"transport": "websocket", "url": "wss://u:p@h:9/acp?token=t#f"})
        self.assertEqual(settings.describe()["url"], "wss://h:9/acp")
        self.assertEqual(settings.url, "wss://u:p@h:9/acp?token=t#f")
        self.assertIsNone(parse({"command": ["x"]}).describe()["url"])
        advertised = parse({"command": ["x"], "bridge": {"advertise": "http://u:p@h:8791/mcp?token=t#f"}})
        self.assertEqual(advertised.describe()["bridge"], "http://h:8791/mcp")

    def test_the_shipped_harness_block_parses(self):
        settings = HarnessSettings.model_validate(CORE_SETTINGS.backend.blocks["harness"])
        self.assertIsInstance(settings.agent, HermesAgent)
        self.assertIsInstance(settings.steering, NativeCommand)
        self.assertEqual(settings.command, ("hermes", "acp"))
        self.assertEqual(settings.bridge.advertise, "http://127.0.0.1:8791")
        self.assertEqual(settings.runtime_label, "hermes")
        self.assertIn("HERMES_HOME", settings.pass_env)

    def test_a_bad_block_fails_when_the_core_is_built(self):
        hermes = {"command": ["x"], "agent": "hermes", "agents": AGENT_BLOCKS}
        cases = {
            "an unknown key": ({"command": ["x"], "yolo": True}, EXTRA.format("yolo")),
            "an unknown agent": ({"command": ["x"], "agent": "hermez"}, "Unknown harness agent 'hermez'"),
            "a block for no known agent": ({"command": ["x"], "agents": {"hermez": {}}}, "names no known agent"),
            "an agent block's unknown key": ({**hermes, "agents": {"hermes": {"soul": "x"}}}, EXTRA.format("soul")),
            "an empty steer": ({**hermes, "agents": {"hermes": {"steer_command": " "}}}, "non-empty text"),
            "a steer with no space": ({**hermes, "agents": {"hermes": {"steer_command": "/steer"}}}, "with a space"),
            "a guard Hermes would run": ({**hermes, "agents": {"hermes": {"command_guard": " /x "}}}, "command_guard"),
            "no guard": ({**hermes, "agents": {"hermes": {}}}, "agents.hermes: .*\ncommand_guard\n.*Field required"),
            "a path Hermes never reads": ({**hermes, "agents": {"hermes": {"instructions_file": ".."}}}, "reads"),
            "agents not a mapping": ({"command": ["x"], "agents": ["hermes"]}, "agents\n.*valid dict"),
            "no command to start": ({}, "must name the agent to start"),
            "a command as one text": ({"command": "hermes acp"}, "command\n.*valid tuple"),
            "a URL with stdio": ({"command": ["x"], "url": "ws://h:1/acp"}, "transport websocket"),
            "a URL that is not text": ({"transport": "websocket", "url": 8790}, "url\n.*valid string"),
            "a URL it cannot read": ({"transport": "websocket", "url": "ws://[::1/acp"}, "URL or null"),
            "a bridge URL it cannot read": ({"command": ["x"], "bridge": {"advertise": "http://[::1"}}, "URL or null"),
            "no URL to connect to": ({"transport": "websocket"}, "ws:// or wss:// URL"),
            "an HTTP URL": ({"transport": "websocket", "url": "http://h:1/acp"}, "ws:// or wss:// URL"),
            "more tasks than a process runs": ({**hermes, "max_running": 5}, "not be over turns_per_process"),
            "a fraction of tasks": ({"command": ["x"], "max_running": 2.5}, "max_running\n.*valid integer"),
            "no turn at all": ({"command": ["x"], "turns_per_process": 0}, "turns_per_process\n.*equal to 1"),
            "a negative wait": ({"command": ["x"], "turn_timeout_seconds": -1}, "turn_timeout_seconds\n.*than 0"),
            "a bridge without a port": ({"command": ["x"], "bridge": {"bind": "localhost"}}, "host:port"),
            "a bridge with no host in its brackets": ({"command": ["x"], "bridge": {"bind": "[]:8791"}}, "host:port"),
            "a relative work root": ({"command": ["x"], "work_root": "work"}, "absolute path"),
            "an unknown template field": ({"command": ["x"], "prompts": {**PROMPTS, "user": "{who}"}}, "names a field"),
            "a field read past its text": (
                {"command": ["x"], "prompts": {**PROMPTS, "user": "{text.x}"}},
                "names a field",
            ),
            "a prompt left out": (
                {"command": ["x"], "prompts": {key: text for key, text in PROMPTS.items() if key != "unnamed_call"}},
                "prompts.unnamed_call\n.*Field required",
            ),
            "an unknown prompt": ({"command": ["x"], "prompts": {"hello": "x"}}, EXTRA.format("prompts.hello")),
            "an env value that is not text": ({"command": ["x"], "env": {"A": 1}}, "env.A\n.*valid string"),
            "pass_env as one text": ({"command": ["x"], "pass_env": "PATH"}, "pass_env\n.*valid tuple"),
            "an empty runtime label": ({"command": ["x"], "runtime_label": " "}, "runtime_label\n.*non-empty text"),
            "not a mapping": ("hermes", "Input should be a valid dictionary"),
        }
        for name, (block, message) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                HarnessSettings.model_validate({"prompts": PROMPTS, **block} if isinstance(block, dict) else block)
        with self.assertRaisesRegex(ValueError, "prompts\n.*Field required"):
            HarnessSettings.model_validate({"command": ["x"]})

    async def test_building_starts_nothing_and_runs_its_own_model(self):
        source, logged = build(), []
        self.assertFalse(hasattr(source, "new_one_shot"))
        adapter = source.new()
        self.assertIsInstance(adapter, HarnessAdapter)
        self.assertEqual(adapter.max_running, 3)
        self.assertIsNone(source.bridge.base_url)
        self.assertEqual(adapter._shared, {})
        self.addCleanup(logger.remove, logger.add(logged.append, level="INFO", format="{message}"))
        for model in ("scripted/model", "other/model"):
            task = FakeTaskContext(model=model, declared=())
            self.assertIsInstance(adapter.start(task), WorkerRun)
        lines = [line.strip() for line in logged if "harness.other_model" in line]
        self.assertEqual(lines, ["harness.other_model task=W1.1 model='other/model': the agent runs its own model"])
        await adapter.close()
        await source.aclose()

    def test_it_describes_what_it_serves_without_credentials(self):
        described = build({"command": ["hermes", "acp"], "agent": "hermes", "env": {"KEY": "a-secret"}}).describe()
        self.assertEqual(described["adapter"], "harness")
        protocol = (described["protocol"], described["agent"], described["transport"])
        self.assertEqual(protocol, ("acp", "hermes", "stdio"))
        self.assertNotIn("a-secret", json.dumps(described))
        self.assertEqual(described["model"], "scripted/model")
        self.assertEqual(described["max_running"], 3)
        self.assertNotIn("secret-key-value", json.dumps(described))
        self.assertNotIn("env", described)

    async def test_it_serves_only_the_model_its_runtime_runs(self):
        source = build()
        await source.check_model("scripted/model")
        with self.assertRaisesRegex(ValueError, "cannot serve the model 'other/model'"):
            await source.check_model("other/model")

    def test_it_reports_the_settings_it_cannot_honor(self):
        source = build()
        client_body = {
            "model": "scripted/model",
            "instructions": "Be brief.",
            "tools": [{"type": "function", "name": "get_menu", "parameters": {}}],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
        }
        self.assertEqual(source.unsupported(client_body), [])  # a client's session body
        settings = {"max_output_tokens": 100, "text": {"format": {"type": "json_schema"}}, "tool_choice": "required"}
        settings |= {"tools": [{"type": "web_search"}], "reasoning": {"effort": "high"}, "service_tier": "flex"}
        problems = [param for param, _ in source.unsupported(settings)]
        self.assertEqual(problems, ["max_output_tokens", "text", "tool_choice", "tools", "reasoning", "service_tier"])

    async def test_the_conductor_is_made_once_and_both_runtimes_warm(self):
        source = build(UNSTARTABLE)
        adapter = source.new()
        conductor = adapter.conductor()
        self.assertIsInstance(conductor, HarnessConductor)
        self.assertIs(adapter.conductor(), conductor)
        # The runtime cannot start: the warm-up fails and is retried at the next use.
        await asyncio.sleep(0.2)
        self.assertEqual(sorted(adapter._shared), ["conductor", "workers"])
        await adapter.close()
        await source.aclose()

    async def test_with_one_task_the_steer_rule_conducts_alone(self):
        source = build({**UNSTARTABLE, "max_running": 1})
        adapter = source.new()
        self.assertIsInstance(adapter.conductor(), SteerOpenWork)
        self.assertIsInstance(adapter.conductor(), SteerOpenWork)
        await asyncio.sleep(0.2)
        self.assertEqual(sorted(adapter._shared), ["workers"])
        self.assertEqual(len(adapter._shared["workers"]), 1)
        await adapter.close()
        await source.aclose()


class Opener:
    """Fake runtime processes as the adapter's ``opener`` opens real ones: one per role, or a new one at every open."""

    def __init__(self, new_each_open: bool):
        self.new_each_open, self.opened, self.runtimes = new_each_open, [], {}

    async def open(self, role):
        if self.new_each_open or role.name not in self.runtimes:
            self.opened.append(FakeRuntime())
            self.runtimes[role.name] = self.opened[-1]
        return self.runtimes[role.name]


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    """An adapter of the source of ``block``: its opener opens its runtime processes, and a fake bridge its doors."""

    async def make(self, block, *, new_each_open=True):
        source = build(block)
        self.adapter = source.new()
        self.adapter.opener, self.adapter.bridge = Opener(new_each_open), FakeBridge()
        self.addAsyncCleanup(source.aclose)
        self.addAsyncCleanup(self.adapter.close)

    async def test_the_warm_up_session_serves_nothing_and_retires(self):
        # The workers' process pays its cold first session/new at the session's start, not in the first task.
        await self.make({**UNSTARTABLE, "agent": "hermes"}, new_each_open=False)
        self.adapter.conductor()
        workers = await self.adapter.shared(self.adapter.workers)
        (session,) = workers.client.sessions
        (door,) = [door for door, _ in self.adapter.bridge.closed if door.name == "w1"]
        self.assertTrue(session.retired)
        self.assertEqual((workers.client.servers[0].name, door.tools()), ("w1", []))
        self.assertEqual([path.name for path in workers.client.folders[0].iterdir()], [".git"])
        self.assertEqual(workers.server_name("w"), "w2")
        runtimes = self.adapter.opener.runtimes
        await until(lambda: "conductor" in runtimes and runtimes["conductor"].client.sessions)
        await asyncio.sleep(0.05)
        self.assertEqual(len(runtimes["conductor"].client.sessions), 1)

    async def test_a_warm_up_session_that_fails_leaves_the_process_usable(self):
        await self.make({**UNSTARTABLE, "agent": "hermes"}, new_each_open=False)
        self.adapter.settings = self.adapter.settings.model_copy(update={"ready_timeout_seconds": 0.05})
        self.adapter.bridge = FakeBridge(lists=False)
        self.adapter._warm(self.adapter.workers, first_session=True)
        workers = await self.adapter.shared(self.adapter.workers)
        self.assertTrue(workers.client.sessions[0].retired)
        self.assertEqual([door.name for door, _ in self.adapter.bridge.closed], ["w1"])
        self.assertFalse(workers.closed)

    async def test_a_close_during_the_warm_up_closes_the_process(self):
        await self.make({**UNSTARTABLE, "agent": "hermes"}, new_each_open=False)
        # The warm-up waits for the listing when the session closes.
        self.adapter.bridge = FakeBridge(lists=False)
        self.adapter._warm(self.adapter.workers, first_session=True)
        await until(lambda: "workers" in self.adapter.opener.runtimes)
        workers = self.adapter.opener.runtimes["workers"]
        await until(lambda: workers.client.sessions)
        await self.adapter.close()
        self.assertTrue(workers.closed)
        self.assertTrue(workers.client.sessions[0].retired)

    async def test_a_full_process_makes_the_next_session_open_another(self):
        # The IM's max_running counts running tasks only, and a task waiting for the app holds its Hermes turn for the
        # whole turn: the pool, not max_running, bounds the turns of a process.
        await self.make({"command": ["/nonexistent/runtime"], "max_running": 2, "turns_per_process": 2})
        workers = self.adapter.workers
        first, second, third = [await self.adapter.shared(workers) for _ in range(3)]
        self.assertIs(first, second)
        self.assertIsNot(first, third)
        self.assertEqual((first.places, third.places), (2, 1))
        first.leave()
        self.assertIs(await self.adapter.shared(workers), first)
        self.assertEqual(len(self.adapter.opener.opened), 2)

    async def test_with_no_turn_limit_one_process_serves_a_role(self):
        await self.make({"command": ["/nonexistent/runtime"], "max_running": 2, "turns_per_process": 2})
        self.adapter.settings = self.adapter.settings.model_copy(update={"turns_per_process": None})
        processes = {id(await self.adapter.shared(self.adapter.workers)) for _ in range(5)}
        self.assertEqual((len(processes), len(self.adapter.opener.opened)), (1, 1))

    async def test_a_kept_session_takes_only_a_free_turn_of_a_live_process(self):
        await self.make({"command": ["/nonexistent/runtime"], "max_running": 2, "turns_per_process": 2})
        workers = self.adapter.workers
        process = await self.adapter.shared(workers)
        # A kept session holds no turn while it waits for a later task.
        process.leave()
        self.assertTrue(self.adapter.take_place(process))
        self.assertIs(await self.adapter.shared(workers), process)
        self.assertFalse(self.adapter.take_place(process))
        self.assertEqual(process.places, 2)
        process.leave()
        process.client.alive = False
        self.assertFalse(self.adapter.take_place(process))
        self.assertEqual(process.places, 1)
        fresh = await self.adapter.shared(workers)
        self.assertIsNot(fresh, process)
        self.assertTrue(process.closed)

    async def test_the_close_retires_the_kept_session(self):
        await self.make({"command": ["/nonexistent/runtime"], "agent": "hermes"})
        run = self.adapter.start(FakeTaskContext(declared=()))
        events = run.events()
        await anext(events)
        [process] = self.adapter.opener.opened
        session, door = process.client.sessions[-1], run.door
        session.say("m1", "Do", "ne.")
        session.end(1, usage={"totalTokens": 100})
        [event async for event in events]
        self.assertEqual((self.adapter.kept.session, process.places, session.retired), (session, 0, False))
        await self.adapter.close()
        self.assertIsNone(self.adapter.kept)
        self.assertTrue(session.retired)
        self.assertIn(door, [closed for closed, _ in self.adapter.bridge.closed])
