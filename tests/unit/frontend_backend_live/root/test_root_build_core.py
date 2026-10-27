# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107
"""The delegation core built from its own settings: ``build_core`` and its profiles."""

import asyncio
import json
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from delegation_core import app
from delegation_core.app import DelegationCore, build_core, log_trace
from delegation_core.backend.control import ControlVocabulary
from delegation_core.backend.port import Start
from delegation_core.config import BackendChoice, CoreSettings, DelegationSettings, ModelEndpoint
from delegation_core.im.input import BackendTask
from tests.unit.contracts.base import ContractCase

FBL = Path(__file__).resolve().parents[4] / "src/examples/frontend_backend_live"
SHIPPED = CoreSettings.load(FBL / "core.yaml")
DEFAULT = ModelEndpoint(model="test/model", base_url="http://localhost:1/v1")
DEFAULTS = BackendTask("test/model", "Take orders.", [])


def endpoints(model):
    return DEFAULT if model in (None, DEFAULT.model) else ModelEndpoint(model=model, base_url="http://other/v1")


def settings(**backend) -> CoreSettings:
    """The shipped core settings with another ``backend`` section (by default one ``scripted`` adapter, no tools)."""
    raw = {"slot": "thinker-llm", "prompt": "thinker", "adapter": "scripted", "one_shot_adapter": "scripted"}
    return replace(SHIPPED, backend=BackendChoice.parse({**raw, **backend}))


# --------------------------------------------------------------------------------------------- test sources
class ScriptedRun:
    """A task's run that answers ``Done.`` in one round: one attempt that completes with a final answer."""

    round = 0

    def __init__(self, task):
        self.task = task

    async def events(self):
        self.round += 1
        item = {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed", "phase": "final_answer"}
        item["content"] = [{"type": "output_text", "text": "Done.", "annotations": []}]
        yield {"type": "response.created", "response": {"id": "attempt_1", "status": "in_progress"}}
        yield {"type": "response.output_item.added", "output_index": 0, "item": {**item, "status": "in_progress"}}
        yield {"type": "response.output_item.done", "output_index": 0, "item": item}
        yield {"type": "response.completed", "response": {"id": "attempt_1", "status": "completed", "output": [item]}}

    def steer(self):
        pass

    async def resume(self):
        pass

    async def stop(self):
        return "stopped"


class StartEach:
    """A rule conductor: each arrival starts a Work of its own."""

    async def decide(self, given, place):
        for arrival in given.arrivals:
            await place(Start((arrival.delegation_id,)))


class ScriptedAdapter:
    max_running = 1

    def __init__(self, source):
        self.source, self.closed = source, False

    def conductor(self):
        return StartEach()

    def start(self, task):
        return ScriptedRun(task)

    async def close(self):
        self.closed = True


class ServesNoOneShot:
    """An adapter source with no ``new_one_shot``: it cannot serve stateless requests."""

    def __init__(self, block, endpoints, tools, vocabulary):
        self.block, self.endpoints, self.tools, self.vocabulary = block, endpoints, tools, vocabulary
        self.adapters, self.closed = [], 0

    @classmethod
    def from_settings(cls, block, *, endpoints, tools, vocabulary):
        unknown = set(block) - {"answer"}
        if unknown:
            raise ValueError(f"backend.scripted has unknown keys: {', '.join(sorted(unknown))}")
        return cls(dict(block), endpoints, tools, vocabulary)

    def describe(self):
        return {"provider": "scripted"}

    async def check_model(self, model):
        if model != "test/model":
            raise ValueError(f"Model {model!r} is not served")

    def unsupported(self, settings):
        return [("service_tier", "not honored")] if "service_tier" in settings else []

    def new(self):
        self.adapters.append(ScriptedAdapter(self))
        return self.adapters[-1]

    async def aclose(self):
        self.closed += 1


class ScriptedSource(ServesNoOneShot):
    def new_one_shot(self):
        self.adapters.append(ScriptedAdapter(self))
        return self.adapters[-1]


ADAPTERS = {"scripted": lambda: ScriptedSource, "other": lambda: ScriptedSource, "no-one-shot": lambda: ServesNoOneShot}


class Recorder:
    """Stands in for a class the builder hands arguments to; it keeps them."""

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs


class Format(Recorder):
    pass


class Items(Recorder):
    pass


class Notes(Recorder):
    pass


class CafeLike(Recorder):
    label = "cafe"

    @classmethod
    def from_settings(cls, block):
        return cls(block)


def module(name: str, **attributes) -> dict:
    """A stand-in module for ``patch.dict(sys.modules, ...)``: the builder imports from it when it is used."""
    stand_in = ModuleType(name)
    stand_in.__dict__.update(attributes)
    return {name: stand_in}


def im_results(**tables) -> dict:
    """A stand-in of the IM's tables (``im/results.py``), as the builder reads them: ``plain``, ``items`` by default."""
    return module(
        "delegation_core.im.results",
        **{"RESULT_FORMATS": {"plain": Format}, "CALL_REPORTS": {"items": Items}, **tables},
    )


class BuilderCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # The IM's tables of result formats and of backend-call reports, as the builder reads them.
        self.enterContext(patch.dict(sys.modules, im_results()))

    def build(self, configured=None, **kwargs) -> DelegationCore:
        kwargs.setdefault("adapters", ADAPTERS)
        return build_core(configured or settings(), endpoints=endpoints, defaults=DEFAULTS, **kwargs)


# ------------------------------------------------------------------------------------------------ the choices
class AdapterChoiceTests(BuilderCase):
    def test_the_adapter_is_picked_by_name_and_built_from_its_block(self):
        core = self.build(settings(scripted={"answer": "Done."}))
        source = core.source
        self.assertIsInstance(source, ScriptedSource)
        self.assertEqual(source.block, {"answer": "Done."})
        self.assertIs(source.endpoints, endpoints)
        self.assertIsInstance(source.vocabulary, ControlVocabulary)
        self.assertEqual(source.vocabulary.prompt, SHIPPED.texts.conductor_prompt)
        self.assertEqual(
            [tool["name"] for tool in source.vocabulary.conductor_tools()],
            ["start_work", "steer_work", "cancel_work", "answer_status"],
        )
        self.assertEqual(source.tools(), ())  # backend.tools names no source

    def test_one_name_builds_one_source_and_two_names_two(self):
        one = self.build()
        self.assertIs(one.source, one.one_shot_source)
        two = self.build(settings(one_shot_adapter="other", other={"answer": "Hi."}))
        self.assertIsNot(two.source, two.one_shot_source)
        self.assertEqual((two.source.block, two.one_shot_source.block), ({}, {"answer": "Hi."}))

    def test_an_unknown_adapter_and_a_block_for_no_adapter_are_refused(self):
        known = "no-one-shot, other, scripted"
        with self.assertRaisesRegex(ValueError, rf"^Unknown backend adapter 'nope'; known: {known}"):
            self.build(settings(adapter="nope"))
        with self.assertRaisesRegex(ValueError, "a block for an adapter it does not know: harness"):
            self.build(settings(harness={}))

    def test_a_block_its_source_refuses_fails_the_build(self):
        with self.assertRaisesRegex(ValueError, "backend.scripted has unknown keys: tempo"):
            self.build(settings(scripted={"tempo": 1}))

    def test_a_one_shot_source_that_cannot_serve_one_shot_sessions_is_refused(self):
        with self.assertRaisesRegex(ValueError, "one_shot_adapter 'no-one-shot' cannot serve one-shot sessions"):
            self.build(settings(one_shot_adapter="no-one-shot"))
        core = self.build(settings(adapter="no-one-shot"))  # as the main adapter it is fine
        self.assertIsInstance(core.source, ServesNoOneShot)

    def test_the_result_format_is_picked_by_name_with_its_note(self):
        formats = {"plain": Format, "speak_display": Format}
        with patch.dict(sys.modules, im_results(RESULT_FORMATS=formats)):
            core = self.build(replace(settings(), delegation=DelegationSettings(result_format="speak_display")))
        self.assertIsInstance(core.result_format, Format)
        self.assertEqual(core.result_format.args, (SHIPPED.texts.result_formats["speak_display"],))
        with self.assertRaisesRegex(ValueError, "Unknown result format 'speak_display'"):
            self.build(replace(settings(), delegation=DelegationSettings(result_format="speak_display")))

    def test_the_backend_call_reports_are_picked_by_name(self):
        reports = {"items": Items, "notes": Notes}
        with patch.dict(sys.modules, im_results(CALL_REPORTS=reports)):
            self.assertIsInstance(self.build().call_reports, Items)  # the default: official mcp_call items
            core = self.build(replace(settings(), delegation=DelegationSettings(backend_calls="notes")))
        self.assertIsInstance(core.call_reports, Notes)
        with self.assertRaisesRegex(ValueError, "Unknown backend call report 'logs'; known: items"):
            self.build(replace(settings(), delegation=DelegationSettings(backend_calls="logs")))

    def test_tool_sources_are_named_by_the_profile_and_fresh_for_each_session(self):
        tables = module("delegation_core.backend.tools", TOOL_SOURCES={"cafe": CafeLike})
        with patch.dict(sys.modules, tables):
            core = self.build(settings(tools={"cafe": {"currency": "USD"}}))
            first, second = core.source.tools(), core.source.tools()
            self.assertEqual([type(source) for source in first], [CafeLike])
            self.assertIsNot(first[0], second[0])
            self.assertEqual(first[0].args, ({"currency": "USD"},))
            with self.assertRaisesRegex(ValueError, "Unknown tool source 'crm'; known: cafe"):
                self.build(settings(tools={"crm": {}}))


# ------------------------------------------------------------------------------------------------ the core
class CoreTests(BuilderCase):
    def test_each_conversation_gets_an_im_session_on_an_adapter_of_its_own(self):
        core = self.build()
        with patch.object(app, "IMSession", Recorder):
            first, second = core.session(), core.session()
        self.assertEqual(len(core.source.adapters), 2)
        self.assertEqual([session.args[0] for session in (first, second)], core.source.adapters)
        self.assertEqual(first.args[1:], (DEFAULTS, core.settings.delegation, core.settings.texts))
        self.assertEqual(
            first.kwargs,
            {"result_format": core.result_format, "call_reports": core.call_reports, "observe": log_trace},
        )

    def test_each_stateless_request_gets_a_one_shot_session_on_the_one_shot_source(self):
        core = self.build(settings(one_shot_adapter="other", other={}))
        with patch.dict(sys.modules, module("delegation_core.im.oneshot", OneShotSession=Recorder)):
            session = core.one_shot_session()
        self.assertEqual((core.source.adapters, core.one_shot_source.adapters), ([], [session.args[0]]))
        self.assertEqual(session.args[1:], (DEFAULTS, core.settings.delegation, core.settings.texts))
        self.assertEqual(session.kwargs, {"observe": log_trace})

    async def test_the_model_rule_and_the_settings_rule_are_the_backend_sources(self):
        core = self.build()
        await core.check_model("test/model")
        with self.assertRaisesRegex(ValueError, "'org/other' is not served"):
            await core.check_model("org/other")
        self.assertEqual(core.unsupported({"service_tier": "flex"}), [("service_tier", "not honored")])
        self.assertEqual(core.unsupported({}), [])

    def test_the_close_budget_is_the_close_grace_then_the_stop_wait(self):
        # How long closing a conversation may take: running work's grace, then the stop door's wait for each stop. The
        # gateway sizes its close of the delegation part from it.
        delegation = SHIPPED.delegation
        self.assertEqual(self.build().close_seconds, delegation.close_grace_seconds + delegation.stop_timeout_seconds)
        no_grace = replace(SHIPPED, delegation=replace(delegation, close_grace_seconds=None))
        self.assertEqual(  # with no grace, running work may finish, within its backstop
            self.build(replace(no_grace, backend=settings().backend)).close_seconds,
            delegation.work_timeout_seconds + delegation.stop_timeout_seconds,
        )
        unbounded = replace(no_grace, delegation=replace(no_grace.delegation, work_timeout_seconds=None))
        self.assertIsNone(self.build(replace(unbounded, backend=settings().backend)).close_seconds)

    def test_describe_gives_the_choices_and_the_sources_own_description(self):
        core = self.build(settings(one_shot_adapter="other", other={}))
        self.assertEqual(core.describe(), {"adapter": "scripted", "one_shot_adapter": "other", "provider": "scripted"})
        json.dumps(core.describe())  # plain JSON

    async def test_aclose_releases_each_source_once(self):
        one = self.build()
        await one.aclose()
        self.assertEqual(one.source.closed, 1)
        two = self.build(settings(one_shot_adapter="other", other={}))
        await two.aclose()
        self.assertEqual((two.source.closed, two.one_shot_source.closed), (1, 1))

    def test_the_core_traces_with_its_own_hook(self):
        core = self.build()
        self.assertIs(core.observe, log_trace)
        lines = []
        handler = app.logger.add(lines.append, level="DEBUG", format="{message}")
        try:
            log_trace("work.started", work="W1")
        finally:
            app.logger.remove(handler)
        self.assertEqual([line.strip() for line in lines], ["core.work.started {'work': 'W1'}"])


# ---------------------------------------------------------------------------------------- crossings
class CrossingTests(unittest.TestCase):
    def test_the_builder_takes_no_apology_and_no_frontend_hook(self):
        self.assertNotIn("apology", build_core.__kwdefaults__ or {})
        self.assertNotIn("apology", build_core.__code__.co_varnames)
        self.assertIs(build_core.__kwdefaults__["observe"], log_trace)

    def test_the_core_builds_with_no_module_of_the_frontend_loaded(self):
        # A fresh interpreter: build a core from the shipped file with a test table, then list what got imported.
        script = f"""
import sys
from dataclasses import replace
from pathlib import Path
from delegation_core.app import build_core
from delegation_core.config import BackendChoice, CoreSettings, ModelEndpoint
from delegation_core.im.input import BackendTask
from tests.unit.frontend_backend_live.root.test_root_build_core import ADAPTERS
settings = CoreSettings.load(Path({str(FBL / "core.yaml")!r}))
raw = {{"slot": "s", "prompt": "p", "adapter": "scripted", "one_shot_adapter": "scripted"}}
settings = replace(settings, backend=BackendChoice.parse(raw))
endpoint = ModelEndpoint(model="m", base_url="http://localhost:1/v1")
build_core(settings, endpoints=lambda model: endpoint, defaults=BackendTask("m", "x", []), adapters=ADAPTERS)
print(sorted(name for name in sys.modules if name.startswith(("examples", "live", "pipecat"))))
"""
        repo = Path(__file__).resolve().parents[4]
        done = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo,
            env={"PYTHONPATH": f"{repo / 'src'}:{repo}", "PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(), "[]")


# ------------------------------------------------------------------------------------------------ the profiles
class ProfileTests(unittest.TestCase):
    def test_the_ui_bot_profile_runs_the_cafe_tools_and_the_envelope(self):
        ui_bot = CoreSettings.load(FBL / "core.ui-bot.yaml")
        self.assertEqual({name: dict(block) for name, block in ui_bot.backend.tools.items()}, {"cafe": {}})
        self.assertEqual(ui_bot.delegation.result_format, "speak_display")
        self.assertEqual(replace(ui_bot.delegation, result_format="plain"), SHIPPED.delegation)
        self.assertEqual(ui_bot.texts, SHIPPED.texts)
        self.assertEqual(ui_bot.backend.blocks, SHIPPED.backend.blocks)

    def test_the_benchmark_profile_waits_for_the_app_writes_no_notes_and_runs_unlisted_models(self):
        bench = CoreSettings.load(FBL / "core.bench.yaml")
        delegation = bench.delegation
        self.assertIsNone(delegation.client_tool_timeout_seconds)
        self.assertEqual((delegation.progress_note_seconds, delegation.result_format), (0.0, "plain"))
        llm = dict(bench.backend.blocks["llm"])
        self.assertEqual((llm["conductor"], llm["unlisted_models"]), ("steer", "fallback"))
        self.assertEqual({**dict(SHIPPED.backend.blocks["llm"]), "unlisted_models": "fallback"}, llm)

    def test_the_shipped_core_reports_an_unlisted_model(self):
        self.assertEqual(SHIPPED.backend.blocks["llm"]["unlisted_models"], "report")
        self.assertEqual((SHIPPED.backend.adapter, SHIPPED.backend.one_shot_adapter), ("llm", "llm"))

    def test_the_voice_note_defaults_to_v2(self):
        self.assertEqual(DelegationSettings().voice_note, "v2")


# ------------------------------------------------------------------------------------------------ the real core
class RealCoreTests(ContractCase, unittest.IsolatedAsyncioTestCase):
    """A core built from the shipped file, end to end, on the real IM and backend LLM."""

    async def ask(self, client, params):
        stream = await client.create_response({"stream": True, **params})
        events = []
        async with asyncio.timeout(10):
            async for event in stream:
                events.append(event)
        return events

    def client(self, core):
        from delegation_core.api.local import LocalResponsesClient
        from delegation_core.api.service import ResponsesService

        service = ResponsesService(core.session, stateless_session=core.one_shot_session, backend=core)
        return LocalResponsesClient(service)

    async def test_the_backend_llm_builds_from_the_shipped_file(self):
        core = build_core(SHIPPED, endpoints=endpoints, defaults=DEFAULTS)
        self.addAsyncCleanup(core.aclose)
        self.assertEqual(type(core.source).__name__, "LlmSource")
        self.assertIs(core.source, core.one_shot_source)

    async def test_a_conversation_is_answered_through_the_core(self):
        core = build_core(settings(), endpoints=endpoints, defaults=DEFAULTS, adapters=ADAPTERS)
        client = self.client(core)
        conversation = (await client.create_conversation({}))["id"]
        self.addAsyncCleanup(client.delete_conversation, conversation)
        user = {"type": "message", "role": "user", "content": "A latte, please."}
        params = {"conversation": conversation, "input": [user], "metadata": {"live_delegation_id": "d1"}}
        events = await self.ask(client, params)
        self.assertEqual(events[-1]["type"], "response.completed")
        answers = [e["item"] for e in events if e["type"] == "response.output_item.done" and e["item"].get("phase")]
        self.assertEqual([item["phase"] for item in answers], ["final_answer"])
        self.assertEqual((await client.retrieve_model("test/model"))["id"], "test/model")

    async def test_a_stateless_request_is_answered_by_a_one_shot_session(self):
        core = build_core(settings(), endpoints=endpoints, defaults=DEFAULTS, adapters=ADAPTERS)
        user = {"type": "message", "role": "user", "content": "What teas do you have?"}
        events = await self.ask(self.client(core), {"input": [user]})
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(len(core.one_shot_source.adapters), 1)


if __name__ == "__main__":
    unittest.main()
