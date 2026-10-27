# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The harness's settings and source: every name resolved by a table, every refusal at build, the YAML switch."""

import asyncio
import dataclasses
import json
import unittest
from pathlib import Path

from delegation_core.app import build_core
from delegation_core.backend.adapters import ADAPTERS
from delegation_core.backend.harness import HarnessAdapter, HarnessConductor, HarnessSettings, HarnessSource
from delegation_core.backend.harness.acp import AcpAgent, AcpProtocol
from delegation_core.backend.harness.acp.agents import AGENTS
from delegation_core.backend.harness.acp.hermes import HermesAgent
from delegation_core.backend.harness.acp.transports import StdioTransport, WebSocketTransport
from delegation_core.backend.harness.permissions import Reject
from delegation_core.backend.harness.steering import CancelReprompt, NativeCommand, NextPrompt
from delegation_core.config import CoreSettings, ModelEndpoint
from delegation_core.im.input import BackendTask
from delegation_core.tables import pick
from tests.unit.delegation_core.backend.harness.doubles import FakeBridge, FakeRuntime, scratch, until, vocabulary

REPO = Path(__file__).resolve().parents[5]
FBL = REPO / "src" / "examples" / "frontend_backend_live"
ENDPOINT = ModelEndpoint(model="scripted/model", base_url="http://127.0.0.1:1/v1", api_key="secret-key-value")


def lookup(model):
    return ENDPOINT


def scratch_root(block):
    """``block`` with its work folders under a scratch folder: a source sweeps its work root when it is built."""
    return {**block, "work_root": str(scratch("work-root-"))}


class SettingsTests(unittest.TestCase):
    def test_the_defaults_are_a_plain_acp_agent_started_over_stdio(self):
        settings = HarnessSettings.parse({"command": ["some-agent", "--acp"]})
        self.assertIsInstance(settings.protocol, AcpProtocol)
        self.assertIs(settings.protocol.transport, StdioTransport)
        self.assertIs(type(settings.agent), AcpAgent)
        self.assertIsInstance(settings.steering, NextPrompt)  # a plain agent takes no steer in the middle of a turn
        self.assertIsInstance(settings.permission, Reject)
        self.assertEqual((settings.max_running, settings.turns_per_process), (3, None))
        self.assertEqual(settings.bridge.url(8791), "http://127.0.0.1:8791")
        self.assertEqual(settings.work_root, Path("~/.cache/voiceclaw/harness").expanduser())
        self.assertEqual((settings.env, settings.pass_env), ({}, ("PATH", "LANG", "LC_ALL", "TZ")))

    def test_each_name_picks_its_class(self):
        for name, cls in {"next_prompt": NextPrompt, "cancel_reprompt": CancelReprompt}.items():
            with self.subTest(name):
                self.assertIsInstance(HarnessSettings.parse({"command": ["x"], "steer": name}).steering, cls)
        websocket = HarnessSettings.parse({"transport": "websocket", "url": "ws://agent.example:8790/acp"})
        self.assertIs(websocket.protocol.transport, WebSocketTransport)
        self.assertEqual(sorted(AGENTS), ["acp", "hermes"])

    def test_the_hermes_agent_class_is_chosen_by_name_with_its_own_block(self):
        settings = HarnessSettings.parse({"command": ["hermes", "acp"], "agent": "hermes", "steer": "native_command"})
        self.assertIsInstance(settings.agent, HermesAgent)
        self.assertIsInstance(settings.steering, NativeCommand)
        self.assertEqual(settings.turns_per_process, 4)  # Hermes's own limit
        self.assertEqual((settings.agent.instructions_file, settings.agent.steer_command), (".hermes.md", "/steer "))
        block = {"instructions_file": "AGENTS.md", "steer_command": "/queue "}
        tuned = HarnessSettings.parse({"command": ["x"], "agent": "hermes", "agents": {"hermes": block}})
        self.assertEqual((tuned.agent.instructions_file, tuned.agent.steer_command), ("AGENTS.md", "/queue "))
        lower = HarnessSettings.parse({"command": ["x"], "agent": "hermes", "max_running": 2, "turns_per_process": 2})
        self.assertEqual(lower.turns_per_process, 2)

    def test_hermes_guards_a_steers_text_too(self):
        # On an idle session Hermes runs a steer's text as a prompt of its own, so a text that starts with "/" would
        # run as one of its commands there: the steer's text gets the guard a prompt's gets.
        agent = HarnessSettings.parse({"command": ["x"], "agent": "hermes", "steer": "native_command"}).agent
        self.assertEqual(agent.steer_text("/reset the order to tea"), "/steer Request: /reset the order to tea")
        self.assertEqual(agent.steer_text("Make it two."), "/steer Make it two.")

    def test_the_description_shows_the_url_without_what_may_carry_a_credential(self):
        settings = HarnessSettings.parse({"transport": "websocket", "url": "wss://u:p@h:9/acp?token=t#f"})
        self.assertEqual(settings.describe()["url"], "wss://h:9/acp")
        self.assertEqual(settings.url, "wss://u:p@h:9/acp?token=t#f")  # the transport connects to the URL as written
        self.assertIsNone(HarnessSettings.parse({"command": ["x"]}).describe()["url"])

    def test_core_yamls_harness_block_parses(self):
        block = CoreSettings.load(FBL / "core.yaml").backend.blocks["harness"]
        settings = HarnessSettings.parse(block)
        self.assertIsInstance(settings.agent, HermesAgent)
        self.assertIsInstance(settings.steering, NativeCommand)
        self.assertEqual(settings.command, ("hermes", "acp"))
        self.assertEqual(settings.bridge.advertise, "http://127.0.0.1:8791")
        self.assertEqual(settings.runtime_label, "hermes")
        self.assertIn("HERMES_HOME", settings.pass_env)  # the operator's Hermes home, when this server names one

    def test_the_agents_own_configuration_is_no_key_of_the_block(self):
        # The operator gives the agent its model, key, toolsets, context window and home; an agent class's own values
        # are in its agents.<name> block. The generic block holds neither.
        keys = (
            "home_template",
            "hermes_toolsets",
            "context_tokens",
            "reasoning_effort",
            "runtime_root",
            "instructions",
            "context_file",
            "steer_command",
            "own_tool_prefix",
        )
        for key in keys:
            with self.subTest(key), self.assertRaisesRegex(ValueError, f"unknown keys: {key}"):
                HarnessSettings.parse({"command": ["x"], key: None})

    def test_a_bad_block_fails_when_the_core_is_built(self):
        hermes = {"command": ["x"], "agent": "hermes"}
        cases = {
            "an unknown key": ({"command": ["x"], "yolo": True}, "unknown keys: yolo"),
            "an unknown protocol": ({"command": ["x"], "protocol": "a2a"}, "Unknown harness protocol 'a2a'"),
            "an unknown transport": ({"command": ["x"], "transport": "grpc"}, "Unknown harness transport"),
            "an unknown agent": ({"command": ["x"], "agent": "hermez"}, "Unknown harness agent 'hermez'"),
            "a block for no known agent": ({"command": ["x"], "agents": {"hermez": {}}}, "names no known agent"),
            "an agent block's unknown key": ({**hermes, "agents": {"hermes": {"soul": "x"}}}, "unknown keys: soul"),
            "an empty steer": ({**hermes, "agents": {"hermes": {"steer_command": " "}}}, "non-empty text"),
            "a steer with no space": ({**hermes, "agents": {"hermes": {"steer_command": "/steer"}}}, "with a space"),
            "a guard Hermes would run": ({**hermes, "agents": {"hermes": {"command_guard": " /x "}}}, "command_guard"),
            "a file Hermes never reads": ({**hermes, "agents": {"hermes": {"instructions_file": "notes.md"}}}, "reads"),
            "a folder for a file": ({**hermes, "agents": {"hermes": {"instructions_file": ".git"}}}, "reads"),
            "the parent folder": ({**hermes, "agents": {"hermes": {"instructions_file": ".."}}}, "reads"),
            "agents not a mapping": ({"command": ["x"], "agents": ["hermes"]}, "map agent names to their blocks"),
            "a steer a plain agent cannot take": ({"command": ["x"], "steer": "native_command"}, "takes none"),
            "an unknown steering": ({"command": ["x"], "steer": "redirect"}, "Unknown harness steering"),
            "an unknown permission": ({"command": ["x"], "permission": "yolo"}, "Unknown permission answer"),
            "no command to start": ({}, "must name the agent to start"),
            "a command as one text": ({"command": "hermes acp"}, "command must be a list"),
            "a URL with stdio": ({"command": ["x"], "url": "ws://h:1/acp"}, "transport websocket"),
            "a URL that is not text": ({"transport": "websocket", "url": 8790}, "URL or null"),
            "a URL it cannot read": ({"transport": "websocket", "url": "ws://[::1/acp"}, "URL or null"),
            "no URL to connect to": ({"transport": "websocket"}, "ws:// or wss:// URL"),
            "more tasks than a process runs": ({**hermes, "max_running": 5}, "not be over turns_per_process"),
            "a fraction of tasks": ({"command": ["x"], "max_running": 2.5}, "whole number"),
            "no turn at all": ({"command": ["x"], "turns_per_process": 0}, "whole number"),
            "a negative wait": ({"command": ["x"], "turn_timeout_seconds": -1}, "positive"),
            "a bridge without a port": ({"command": ["x"], "bridge": {"bind": "localhost"}}, "host:port"),
            "a relative work root": ({"command": ["x"], "work_root": "work"}, "absolute path"),
            "an unknown template field": ({"command": ["x"], "prompts": {"user": "{who}"}}, "names a field"),
            "an unknown prompt": ({"command": ["x"], "prompts": {"hello": "x"}}, "unknown keys: hello"),
            "an env value that is not text": ({"command": ["x"], "env": {"A": 1}}, "variable names to text"),
            "pass_env as one text": ({"command": ["x"], "pass_env": "PATH"}, "pass_env must be a list"),
            "an empty runtime label": ({"command": ["x"], "runtime_label": " "}, "runtime_label must be non-empty"),
            "not a mapping": ("hermes", "must be a mapping"),
        }
        for name, (block, message) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                HarnessSettings.parse(block)


class SourceTests(unittest.IsolatedAsyncioTestCase):
    def build(self, block=None, tools=lambda: []):
        return HarnessSource.from_settings(
            scratch_root(block or {"command": ["hermes", "acp"], "agent": "hermes"}),
            endpoints=lookup,
            tools=tools,
            vocabulary=vocabulary(),
        )

    def test_the_adapter_table_loads_the_source(self):
        self.assertIs(pick(ADAPTERS, "harness", "backend adapter")(), HarnessSource)

    async def test_building_starts_nothing_and_an_adapter_opens_nothing_before_its_conductor(self):
        source = self.build()
        adapter = source.new()
        self.assertIsInstance(adapter, HarnessAdapter)
        self.assertEqual(adapter.max_running, 3)
        self.assertIsNone(source.bridge.base_url)  # no port bound
        self.assertEqual(adapter._shared, {})
        await adapter.close()
        await source.aclose()

    def test_it_describes_what_it_serves_without_credentials(self):
        described = self.build({"command": ["hermes", "acp"], "agent": "hermes", "env": {"KEY": "a-secret"}}).describe()
        self.assertEqual(described["adapter"], "harness")
        self.assertEqual(
            (described["protocol"], described["agent"], described["transport"]), ("acp", "hermes", "stdio")
        )
        self.assertNotIn("a-secret", json.dumps(described))
        self.assertEqual(described["model"], "scripted/model")
        self.assertNotIn("secret-key-value", json.dumps(described))
        self.assertNotIn("env", described)

    async def test_it_serves_only_the_model_its_runtime_runs(self):
        source = self.build()
        await source.check_model("scripted/model")
        with self.assertRaisesRegex(ValueError, "cannot serve the model 'other/model'"):
            await source.check_model("other/model")

    def test_it_reports_the_settings_a_runtime_that_owns_its_calls_cannot_honor(self):
        source = self.build()
        tau = {
            "model": "scripted/model",
            "instructions": "Policy.",
            "tools": [{"type": "function", "name": "get_user", "parameters": {}}],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
        }
        self.assertEqual(source.unsupported(tau), [])  # tau2's session body
        problems = source.unsupported(
            {
                "max_output_tokens": 100,
                "text": {"format": {"type": "json_schema"}},
                "tool_choice": "required",
                "tools": [{"type": "web_search"}],
                "reasoning": {"effort": "high"},
                "service_tier": "flex",
            }
        )
        self.assertEqual(
            [param for param, _ in problems],
            ["max_output_tokens", "text", "tool_choice", "tools", "reasoning", "service_tier"],
        )

    def test_it_never_serves_one_shot_sessions(self):
        self.assertFalse(hasattr(self.build(), "new_one_shot"))


class CoreSwitchTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_harness_profile_switches_the_core_to_the_harness_with_no_other_change(self):
        # It builds on the agent's profile, so both model conductors read one prompt and its envelope.
        self.assertIn("\nextends: core.agent.yaml\n", (FBL / "core.harness.yaml").read_text(encoding="utf-8"))
        settings = CoreSettings.load(FBL / "core.harness.yaml")
        agent = CoreSettings.load(FBL / "core.agent.yaml")
        self.assertEqual((settings.backend.adapter, settings.backend.one_shot_adapter), ("harness", "llm"))
        self.assertEqual(settings.delegation, agent.delegation)
        self.assertEqual(settings.texts.conductor_prompt, agent.texts.conductor_prompt)
        self.assertEqual(settings.delegation.result_format, "speak_display")
        blocks = {**settings.backend.blocks, "harness": scratch_root(settings.backend.blocks["harness"])}
        settings = dataclasses.replace(settings, backend=dataclasses.replace(settings.backend, blocks=blocks))
        core = build_core(settings, endpoints=lookup, defaults=BackendTask("scripted/model", "Help.", []))
        self.assertIsInstance(core.source, HarnessSource)
        self.assertEqual(core.describe()["adapter"], "harness")
        session = core.session()
        self.assertIsInstance(session.backend, HarnessAdapter)
        self.assertIsNone(session.conductor)  # nothing starts until the session starts
        await session.backend.close()
        await core.aclose()

    async def test_the_im_asks_the_conductor_once_and_both_runtimes_start_in_the_background(self):
        source = HarnessSource.from_settings(
            scratch_root({"command": ["/nonexistent/runtime"], "bridge": {"bind": "127.0.0.1:0", "advertise": None}}),
            endpoints=lookup,
            tools=lambda: [],
            vocabulary=vocabulary(),
        )
        adapter = source.new()
        conductor = adapter.conductor()
        self.assertIsInstance(conductor, HarnessConductor)
        self.assertIs(adapter.conductor(), conductor)
        await asyncio.sleep(0.2)  # the runtime cannot start: the warm-up fails and is retried at the next use
        self.assertEqual(sorted(adapter._shared), ["conductor", "workers"])
        await adapter.close()
        await source.aclose()


class FakeOpener:
    """One fake runtime process per role, as the adapter's ``opener`` makes real ones."""

    def __init__(self):
        self.runtimes: dict[str, FakeRuntime] = {}

    async def open(self, role):
        return self.runtimes.setdefault(role.name, FakeRuntime())


class WarmUpTests(unittest.IsolatedAsyncioTestCase):
    """The workers' process pays its cold first ``session/new`` at the session's start, not in the first task."""

    async def asyncSetUp(self):
        self.source = HarnessSource.from_settings(
            scratch_root(
                {
                    "command": ["/nonexistent/runtime"],
                    "agent": "hermes",
                    "bridge": {"bind": "127.0.0.1:0", "advertise": None},
                }
            ),
            endpoints=lookup,
            tools=lambda: [],
            vocabulary=vocabulary(),
        )
        self.adapter = self.source.new()
        self.adapter.opener, self.adapter.bridge = FakeOpener(), FakeBridge()

    async def asyncTearDown(self):
        await self.adapter.close()
        await self.source.aclose()

    async def test_the_workers_process_opens_one_session_that_serves_nothing_and_retires_it(self):
        self.adapter.conductor()
        workers = await self.adapter.shared(self.adapter.workers)
        (session,) = workers.client.sessions
        (door,) = [door for door, _ in self.adapter.bridge.closed if door.name == "w1"]
        self.assertTrue(session.retired)
        self.assertEqual((workers.client.servers[0].name, door.tools()), ("w1", []))
        self.assertEqual(
            [path.name for path in workers.client.folders[0].iterdir()], [".git"]
        )  # pinned, no instructions
        self.assertEqual(workers.server_name("w"), "w2")  # the first task's session is a fresh one
        runtimes = self.adapter.opener.runtimes
        await until(lambda: "conductor" in runtimes and runtimes["conductor"].client.sessions)
        await asyncio.sleep(0.05)
        self.assertEqual(len(runtimes["conductor"].client.sessions), 1)  # the conductor's own session, no warm-up

    async def test_a_warm_up_session_that_fails_leaves_the_process_usable(self):
        self.adapter.settings = dataclasses.replace(self.adapter.settings, ready_timeout_seconds=0.05)
        self.adapter.bridge = FakeBridge(lists=False)  # the runtime never lists the door's tools
        self.adapter._warm(self.adapter.workers, first_session=True)
        workers = await self.adapter.shared(self.adapter.workers)
        self.assertTrue(workers.client.sessions[0].retired)
        self.assertEqual([door.name for door, _ in self.adapter.bridge.closed], ["w1"])
        self.assertFalse(workers.closed)

    async def test_a_session_closed_during_the_warm_up_closes_the_workers_process(self):
        self.adapter.bridge = FakeBridge(lists=False)  # the warm-up waits for the listing when the session closes
        self.adapter._warm(self.adapter.workers, first_session=True)
        await until(lambda: "workers" in self.adapter.opener.runtimes)
        workers = self.adapter.opener.runtimes["workers"]
        await until(lambda: workers.client.sessions)
        await self.adapter.close()
        self.assertTrue(workers.closed)
        self.assertTrue(workers.client.sessions[0].retired)


class NewProcessEachOpen:
    """A fake opener that starts a new runtime process at every open, as the real one does."""

    def __init__(self):
        self.opened: list[FakeRuntime] = []

    async def open(self, role):
        self.opened.append(FakeRuntime())
        return self.opened[-1]


class PoolTests(unittest.IsolatedAsyncioTestCase):
    """A role's sessions take turns of its runtime processes; a full process never gets one more."""

    async def asyncSetUp(self):
        self.source = HarnessSource.from_settings(
            scratch_root({"command": ["/nonexistent/runtime"], "max_running": 2, "turns_per_process": 2}),
            endpoints=lookup,
            tools=lambda: [],
            vocabulary=vocabulary(),
        )
        self.adapter = self.source.new()
        self.adapter.opener, self.adapter.bridge = NewProcessEachOpen(), FakeBridge()

    async def asyncTearDown(self):
        await self.adapter.close()
        await self.source.aclose()

    async def test_a_full_process_makes_the_next_session_open_another_and_a_given_back_turn_is_reused(self):
        # The IM's max_running counts running tasks only: a task waiting for the app holds no slot of the IM, but its
        # Hermes turn holds its place in the process for the whole turn. So the pool, not max_running, bounds turns.
        workers = self.adapter.workers
        first, second, third = [await self.adapter.shared(workers) for _ in range(3)]
        self.assertIs(first, second)
        self.assertIsNot(first, third)
        self.assertEqual((first.places, third.places), (2, 1))
        first.leave()
        self.assertIs(await self.adapter.shared(workers), first)
        self.assertEqual(len(self.adapter.opener.opened), 2)

    async def test_an_agent_with_no_turn_limit_serves_every_session_of_a_role_from_one_process(self):
        self.adapter.settings = dataclasses.replace(self.adapter.settings, turns_per_process=None)
        workers = self.adapter.workers
        processes = {id(await self.adapter.shared(workers)) for _ in range(5)}
        self.assertEqual((len(processes), len(self.adapter.opener.opened)), (1, 1))

    async def test_a_lost_process_is_closed_and_its_role_opens_a_new_one(self):
        workers = self.adapter.workers
        lost = await self.adapter.shared(workers)
        lost.client.alive = False
        fresh = await self.adapter.shared(workers)
        self.assertIsNot(fresh, lost)
        self.assertTrue(lost.closed)


if __name__ == "__main__":
    unittest.main()
