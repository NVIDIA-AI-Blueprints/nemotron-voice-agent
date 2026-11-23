# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The harness's settings, source and adapter: every name resolved by a table, every refusal at build, the pool."""

import asyncio
import json
import unittest
from pathlib import Path

from loguru import logger

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.conductors import SteerOpenWork
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.harness import (
    HarnessAdapter,
    HarnessConductor,
    HarnessSettings,
    HarnessSource,
    WorkerRun,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.harness.acp import (
    AcpAgent,
    AcpProtocol,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.harness.acp.hermes import (
    HermesAgent,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.harness.acp.transports import (
    StdioTransport,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.harness.permissions import (
    Reject,
)
from nemotron_voice_agent.pipecat.shared.delegation_contract import ModelEndpoint
from tests.unit.pipecat.delegation_core.harness.doubles import (
    AGENT_BLOCKS,
    PROMPTS,
    FakeBridge,
    FakeRuntime,
    FakeTaskContext,
    scratch,
    until,
    vocabulary,
)
from tests.unit.pipecat.delegation_core.helpers import (
    CORE_SETTINGS,
    MENU,
    IMCase,
    RecordingTools,
    ScriptedProvider,
    answers,
    app_outputs,
    ask,
    call_round,
    read_all,
    text_round,
)

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
        self.assertEqual(settings.command, ("hermes", "acp"))
        self.assertEqual(settings.bridge.advertise, "http://127.0.0.1:8791")
        self.assertEqual(settings.runtime_label, "hermes")
        self.assertIn("HERMES_HOME", settings.pass_env)

    def test_a_bad_block_fails_when_the_core_is_built(self):
        hermes = {"command": ["x"], "agent": "hermes", "agents": AGENT_BLOCKS}
        cases = {
            "an unknown key": ({"command": ["x"], "yolo": True}, EXTRA.format("yolo")),
            "an unknown agent": (
                {"command": ["x"], "agent": "hermes-gateway"},
                "Unknown harness agent 'hermes-gateway'",
            ),
            "a block for no known agent": (
                {"command": ["x"], "agents": {"hermes-gateway": {}}},
                "names no known agent",
            ),
            "an agent block's unknown key": ({**hermes, "agents": {"hermes": {"soul": "x"}}}, EXTRA.format("soul")),
            "a removed steer override": (
                {**hermes, "agents": {"hermes": {"steer_command": "/steer "}}},
                EXTRA.format("steer_command"),
            ),
            "a guard Hermes would run": ({**hermes, "agents": {"hermes": {"command_guard": " /x "}}}, "command_guard"),
            "no guard": (
                {**hermes, "agents": {"hermes": {}}},
                r"backend\.harness\.agents\.hermes: .*\ncommand_guard\n.*Field required",
            ),
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
        self.assertFalse(adapter.can_run_without_attachment())
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
    """Fake runtime processes as the adapter's ``opener`` opens real ones: one per role, or a new one at every open.

    ``closes``: their agent closes each session it retires.
    """

    def __init__(self, new_each_open: bool, closes: bool = False):
        self.new_each_open, self.closes, self.opened, self.runtimes = new_each_open, closes, [], {}

    async def open(self, role):
        if self.new_each_open or role.name not in self.runtimes:
            self.opened.append(FakeRuntime())
            self.opened[-1].client.closes = self.closes
            self.runtimes[role.name] = self.opened[-1]
        return self.runtimes[role.name]

    async def wait_closed(self):
        pass


class RestrictedExecutionTests(IMCase):
    def harness(self, provider, tools):
        source = build({**UNSTARTABLE, "agent": "hermes", "max_running": 1, "stop_timeout_seconds": 1})
        source.tools = lambda: (tools,)
        native_factory = source.new_restricted

        def with_scripted_provider(tool_sources):
            restricted = native_factory(tool_sources)
            restricted.source.make_provider = lambda endpoint: provider
            return restricted

        source.new_restricted = with_scripted_provider
        adapter = source.new()
        adapter.opener, adapter.bridge = Opener(False), FakeBridge()
        self.addAsyncCleanup(source.aclose)
        return source, adapter

    async def start_app_wait(self, provider, tools):
        source, adapter = self.harness(provider, tools)
        im = self.session(adapter)
        initial = im.admit(ask("Look up the menu."), delegation_id="D1", settings={"tools": [MENU]})
        await until(
            lambda: im.works and (run := im.works["W1"].tasks[0].run) is not None and run.foreground.turn is not None
        )
        task = im.works["W1"].tasks[0]
        return source, im, task, initial

    async def test_held_application_stop_runs_only_core_tools_and_preserves_the_original_result_owner(self):
        provider = ScriptedProvider(call_round("cancel_work", {}, "stop_control"), endpoint=ENDPOINT)
        tools = RecordingTools(("lookup", "modify"))
        source, im, task, first = await self.start_app_wait(provider, tools)
        run = task.run
        worker = run.session
        pending_tool = asyncio.create_task(run.door.call("get_menu", {}))
        events = await read_all(first.stream, timeout=2)
        [result] = app_outputs(events, "Menu read completed.")
        stop = im.admit(ask("Stop that task."), delegation_id="D2")
        await until(lambda: worker.cancels == 1)
        self.assertEqual(provider.opened, 0)
        self.assertIs(task.run, run)
        worker.end(1, "cancelled")
        await read_all(stop.stream, timeout=3)
        await asyncio.wait_for(pending_tool, 2)
        self.assertEqual(provider.opened, 1)
        offered = {tool["name"] for tool in provider.payloads[0]["tools"]}
        self.assertIn("cancel_work", offered)
        self.assertTrue(offered.isdisjoint({"lookup", "modify", "get_menu"}))
        self.assertEqual(tools.calls, [])
        self.assertEqual(worker.cancels, 1)
        self.assertEqual(task.record.stop.state, "stopped")
        owner = im.ledger.actions[result["call_id"]].task
        self.assertEqual(owner, task.task_id)
        late = im.admit_client_request([result], delegation_id="D1")
        await read_all(late.stream, timeout=2)
        action = im.ledger.actions[result["call_id"]]
        self.assertEqual(action.task, owner)
        self.assertEqual(action.late_output, "Menu read completed.")
        self.assertEqual(source.describe()["restricted_execution"], "llm")
        self.assertTrue(any(row["backend"] == "llm" for row in self.traced("backend.round.usage")))

    async def test_read_only_verification_has_no_native_session_and_refuses_unapproved_tool_calls(self):
        provider = ScriptedProvider(
            call_round("modify", {}, "unapproved_write"),
            call_round("lookup", {}, "permitted_read"),
            text_round("Verified."),
            endpoint=ENDPOINT,
        )
        tools = RecordingTools(("lookup", "modify"))
        tools.read_only_tools = frozenset({"lookup"})
        _, adapter = self.harness(provider, tools)
        task = FakeTaskContext(declared=(), model=ENDPOINT.model)
        task.business_access = "read_only"
        from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.access import (
            read_only_sources,
        )

        task.tool_sources = lambda tool_sources: read_only_sources(tool_sources)[0]
        result = await read_all(adapter.start(task).events(), timeout=2)
        self.assertEqual(answers(result), ["Verified."])
        self.assertEqual([name for _, name, _ in tools.calls], ["lookup"])
        self.assertTrue(all("modify" not in {tool["name"] for tool in row["tools"]} for row in provider.payloads))
        self.assertEqual(adapter.opener.opened, [])

    async def test_status_while_waiting_preserves_the_session_and_feeds_back_the_real_application_result(self):
        provider = ScriptedProvider(text_round("The request is pending."), endpoint=ENDPOINT)
        _, im, task, initial = await self.start_app_wait(provider, RecordingTools())
        run = task.run
        worker, initial_prompt = run.session, run.foreground.turn.main
        pending_tool = asyncio.create_task(run.door.call("get_menu", {}))
        run.door.pending_calls.add(pending_tool)
        pending_tool.add_done_callback(run.door.pending_calls.discard)
        [result] = app_outputs(await read_all(initial.stream, timeout=2), "The app found green tea.")
        status = im.admit(ask("Any update on the menu?"), delegation_id="D2")
        await until(lambda: worker.cancels == 1)
        worker.end(initial_prompt, "cancelled")
        await until(lambda: provider.opened == 1 and task.control_run is None)
        self.assertFalse(pending_tool.done())
        self.assertEqual(len(worker.texts("prompt")), 1)
        resumed = im.admit_client_request([result], delegation_id="D1")
        await until(lambda: len(worker.texts("prompt")) == 2)
        self.assertEqual(await asyncio.wait_for(pending_tool, 2), "The app found green tea.")
        self.assertIs(run.session, worker)
        self.assertIn("The app found green tea.", worker.texts("prompt")[-1])
        worker.answer(run.foreground.turn.main, "updated", "The menu includes green tea.")
        reported = await asyncio.wait_for(asyncio.gather(read_all(status.stream), read_all(resumed.stream)), 2)
        self.assertTrue(any("The menu includes green tea." in answers(events) for events in reported))
        self.assertEqual(worker.cancels, 1)


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    """An adapter of the source of ``block``: its opener opens its runtime processes, and a fake bridge its doors."""

    async def make(self, block, *, new_each_open=True, closes=False):
        source = build(block)
        self.adapter = source.new()
        self.adapter.opener, self.adapter.bridge = Opener(new_each_open, closes), FakeBridge()
        self.addAsyncCleanup(source.aclose)
        self.addAsyncCleanup(self.adapter.close)

    def warm_door(self):
        """Return the warm-up session's door, while the bridge serves it."""
        (door,) = [door for door in self.adapter.bridge.doors.values() if door.name == "w1"]
        return door

    async def test_warmup_retires_each_session_and_keeps_an_inert_door_until_close(self):
        for closes in (False, True):
            with self.subTest(closes=closes):
                await self.make({**UNSTARTABLE, "agent": "hermes"}, new_each_open=False, closes=closes)
                self.adapter.conductor()
                workers = await self.adapter.shared(self.adapter.workers)
                [session] = workers.client.sessions
                door = self.warm_door()
                self.assertTrue(session.retired)
                self.assertEqual((workers.client.servers[0].name, door.tools()), ("w1", []))
                self.assertEqual([p.name for p in workers.client.folders[0].iterdir()], [".git"])
                self.assertEqual(workers.server_name("w"), "w2")
                runtimes = self.adapter.opener.runtimes
                await until(lambda runtimes=runtimes: "conductor" in runtimes and runtimes["conductor"].client.sessions)
                self.assertEqual(len(runtimes["conductor"].client.sessions), 1)
                self.adapter.release_idle()
                self.assertTrue(door.retired)
                self.assertIn(door.token, self.adapter.bridge.doors)
                self.assertEqual(door.tools(), [])
                await self.adapter.close()
                await self.adapter.wait_closed()
                self.assertNotIn(door.token, self.adapter.bridge.doors)

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
        self.adapter.bridge = FakeBridge(lists=False)
        self.adapter._warm(self.adapter.workers, first_session=True)
        await until(lambda: "workers" in self.adapter.opener.runtimes)
        workers = self.adapter.opener.runtimes["workers"]
        await until(lambda: workers.client.sessions)
        await self.adapter.close()
        self.assertTrue(workers.closed)
        self.assertTrue(workers.client.sessions[0].retired)
        with self.assertRaisesRegex(RuntimeError, "harness session is closing"):
            await self.adapter.shared(self.adapter.workers)

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
        session.answer(1, "m1", "Do", "ne.", usage={"totalTokens": 100})
        [event async for event in events]
        [kept] = self.adapter.pool.idle
        self.assertEqual((kept.session, process.places, session.retired), (session, 0, False))
        await self.adapter.close()
        self.assertEqual(self.adapter.pool.idle, [])
        self.assertTrue(session.retired)
        self.assertIn(door, [closed for closed, _ in self.adapter.bridge.closed])

    async def test_release_idle_retires_the_idle_sessions_and_pools_none_until_a_task_starts(self):
        # The caller left: no task of the call is left to take a session. A later call's first task opens the pool.
        await self.make({"command": ["/nonexistent/runtime"], "agent": "hermes"})
        totals = iter(range(100, 1000, 100))

        async def answer(run, events):
            """Have the run's session answer, and return it."""
            session = run.session
            session.answer(session.last_prompt(), "m1", "Do", "ne.", usage={"totalTokens": next(totals)})
            [event async for event in events]
            return session

        tasks = [FakeTaskContext(declared=(), work_id=f"W{n}", task_id=f"W{n}.1") for n in (1, 2, 3)]
        runs = [self.adapter.start(task) for task in tasks[:2]]
        events = [run.events() for run in runs]
        for each in events:
            await anext(each)
        idle = await answer(runs[0], events[0])
        [kept] = self.adapter.pool.idle
        self.adapter.release_idle()
        self.assertEqual((idle.retired, self.adapter.pool.idle), (True, []))
        late = await answer(runs[1], events[1])
        [traced] = tasks[1].traced("harness.session_retired")
        self.assertEqual((late.retired, self.adapter.pool.idle, traced["why"]), (True, [], "the pool closed"))
        run = self.adapter.start(tasks[2])
        events = run.events()
        await anext(events)
        later = await answer(run, events)
        self.assertEqual([kept.session for kept in self.adapter.pool.idle], [later])
