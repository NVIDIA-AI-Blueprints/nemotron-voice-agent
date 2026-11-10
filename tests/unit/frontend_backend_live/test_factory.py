# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""The Live root: its settings read once, each session's adapter, its one core, and the catalog's endpoints."""

import asyncio
import json
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from pipecat.transports.smallwebrtc.connection import IceServer

import examples_registry
from examples.frontend_backend_live import factory, pipeline
from examples.frontend_backend_live.config_manager.schema import ExampleConfig
from examples.frontend_backend_live.delegation_core.app import build_core
from examples.frontend_backend_live.delegation_core.backend.control import ControlVocabulary
from examples.frontend_backend_live.delegation_core.backend.providers import base as providers
from examples.frontend_backend_live.delegation_core.config import BackendChoice, CoreSettings, DelegationSettings
from examples.frontend_backend_live.delegation_core.im.input import BackendTask, compose
from examples.frontend_backend_live.models.endpoints import ExampleCatalog, resolve_endpoint
from examples.frontend_backend_live.prompts import load_prompts
from examples.shared.delegation_contract import ModelEndpoint
from gateways.live.protocol import ProtocolError, SessionConfig
from tests.unit.core_helpers import CORE_BLOCK, CORE_TEXTS, EXAMPLE_SETTINGS, FakeChat, chunk, core_block
from tests.unit.frontend_helpers import (
    CATALOG,
    RootCase,
    StubAdapter,
    StubReadiness,
    StubSettings,
    module,
    reset_stubs,
    stub_root,
)
from utils import NVCF_LLM_BASE_URL

SESSION = SessionConfig.model_validate({"model": "live-1"})
SHIPPED = CoreSettings.parse(CORE_BLOCK, CORE_TEXTS)
DEFAULT = ModelEndpoint(model="test/model", base_url="http://localhost:1/v1")
ENTRY = {"model_id": "org/model", "base_url": "http://host:8000/v1"}
NEMOTRON = {"id": "self-hosted:nemotron", "key": "nemotron", "model_id": "org/nemotron", "base_url": "http://nemo/v1"}
OTHER = {"id": "cloud:other", "key": "other", "model_id": "org/other", "base_url": "https://other.example/v1"}
OTHER |= {"api_key_env": "OTHER_KEY"}
KEYS = {"NVIDIA_API_KEY": "nvapi-secret", "OTHER_KEY": "other-key"}
PICKED = ExampleCatalog(
    {
        "llm": [{"model_id": "org/a", "base_url": "http://talker/v1", "extra_params": '{"temperature": 0}'}],
        "thinker-llm": [
            {"model_id": "org/b", "base_url": "http://thinker/v1"},
            {"model_id": "org/a", "base_url": "http://thinker/v1", "extra_params": '{"temperature": 0.7}'},
        ],
        "asr": [{"server": "asr:50052", "model": "parakeet"}],
    }
)


def endpoints(model):
    return DEFAULT if model in (None, DEFAULT.model) else ModelEndpoint(model=model, base_url="http://other/v1")


def settings(**backend) -> CoreSettings:
    """Return the shipped core settings with another ``backend`` section, by default one ``scripted`` adapter."""
    raw = {"slot": "thinker-llm", "prompt": "thinker", "adapter": "scripted", "one_shot_adapter": "scripted"}
    return SHIPPED.model_copy(update={"backend": BackendChoice.model_validate({**raw, **backend})})


def handed(core):
    """Return what the root hands each frontend adapter's block: the core's display label and its app deadline."""
    delegation = core.delegation
    return {
        "display_label": delegation.display_label,
        "client_tool_timeout_seconds": delegation.client_tool_timeout_seconds,
    }


class ScriptedSource:
    """An adapter source by the backend's contract that keeps how it was built and counts its closes."""

    def __init__(self, block, endpoints, tools, vocabulary):
        self.block, self.endpoints, self.tools, self.vocabulary, self.closed = block, endpoints, tools, vocabulary, 0

    @classmethod
    def from_settings(cls, block, *, endpoints, tools, vocabulary):
        return cls(dict(block), endpoints, tools, vocabulary)

    def describe(self):
        return {"provider": "scripted"}

    def new_one_shot(self):
        raise AssertionError("no session runs here")

    async def aclose(self):
        self.closed += 1


class Recorder:
    """Stand-in for a class the builder hands arguments to, which keeps them."""

    label = "cafe"

    def __init__(self, *args):
        self.args = args

    @classmethod
    def from_settings(cls, block):
        return cls(block)


class BuildCoreTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.formats = {"plain": Recorder}
        results = module("examples.frontend_backend_live.delegation_core.im.results", RESULT_FORMATS=self.formats)
        self.enterContext(patch.dict(sys.modules, results))

    def build(self, configured=None):
        adapters = {"scripted": lambda: ScriptedSource, "other": lambda: ScriptedSource}
        return build_core(
            configured or settings(),
            endpoints=endpoints,
            defaults=BackendTask("test/model", "Take orders.", []),
            adapters=adapters,
        )

    def test_the_core_builds_the_named_adapter_from_its_block(self):
        source = self.build(settings(scripted={"answer": "Done."})).source
        self.assertIsInstance(source, ScriptedSource)
        self.assertEqual(source.block, {"answer": "Done."})
        self.assertIs(source.endpoints, endpoints)
        self.assertIsInstance(source.vocabulary, ControlVocabulary)
        self.assertEqual(source.vocabulary.prompt, SHIPPED.texts.conductor_prompt)
        answers = (source.vocabulary.idle, source.vocabulary.failed)
        self.assertEqual(answers, (SHIPPED.texts.refusal("idle"), SHIPPED.texts.failed_output))
        self.assertEqual(source.tools(), ())
        with self.assertRaisesRegex(ValueError, "a block for an adapter it does not know: harness"):
            self.build(settings(harness={}))

    async def test_each_named_source_is_built_and_released_once(self):
        one = self.build()
        self.assertIs(one.source, one.one_shot_source)
        await one.aclose()
        self.assertEqual(one.source.closed, 1)
        two = self.build(settings(one_shot_adapter="other", other={"answer": "Hi."}))
        self.assertIsNot(two.source, two.one_shot_source)
        self.assertEqual((two.source.block, two.one_shot_source.block), ({}, {"answer": "Hi."}))
        described = {"adapter": "scripted", "one_shot_adapter": "other", "provider": "scripted"}
        self.assertEqual(json.loads(json.dumps(two.describe())), described)
        await two.aclose()
        self.assertEqual((two.source.closed, two.one_shot_source.closed), (1, 1))

    def test_the_result_format_is_picked_by_name_with_its_note(self):
        speak_display = settings().model_copy(update={"delegation": DelegationSettings(result_format="speak_display")})
        with self.assertRaisesRegex(ValueError, "Unknown result format 'speak_display'"):
            self.build(speak_display)
        self.formats["speak_display"] = Recorder
        core = self.build(speak_display)
        self.assertIsInstance(core.result_format, Recorder)
        self.assertEqual(core.result_format.args, (SHIPPED.texts.result_formats["speak_display"],))

    def test_tool_sources_are_named_and_fresh_for_each_session(self):
        tables = module("examples.frontend_backend_live.delegation_core.backend.tools", TOOL_SOURCES={"cafe": Recorder})
        with patch.dict(sys.modules, tables):
            core = self.build(settings(tools={"cafe": {"currency": "USD"}}))
            first, second = core.source.tools(), core.source.tools()
            self.assertEqual([type(source) for source in first], [Recorder])
            self.assertIsNot(first[0], second[0])
            self.assertEqual(first[0].args, ({"currency": "USD"},))
            with self.assertRaisesRegex(ValueError, "Unknown tool source 'crm'; known: cafe"):
                self.build(settings(tools={"crm": {}}))

    def test_the_close_budget_is_the_grace_then_the_stop_wait(self):
        delegation = SHIPPED.delegation
        self.assertEqual(self.build().close_seconds, delegation.close_grace_seconds + delegation.stop_timeout_seconds)
        no_grace = delegation.model_copy(update={"close_grace_seconds": None})
        backend = settings().backend
        self.assertEqual(
            self.build(SHIPPED.model_copy(update={"delegation": no_grace, "backend": backend})).close_seconds,
            delegation.work_timeout_seconds + delegation.stop_timeout_seconds,
        )
        unbounded = no_grace.model_copy(update={"work_timeout_seconds": None})
        self.assertIsNone(
            self.build(SHIPPED.model_copy(update={"delegation": unbounded, "backend": backend})).close_seconds
        )

    def test_the_ui_bots_core_stays_on_the_backend_llm(self):
        def ui_bot_core(example: dict) -> CoreSettings:
            return CoreSettings.parse(ExampleConfig.model_validate(example).ui_bot_core, CORE_TEXTS)

        for adapter in ("llm-agent", "harness"):
            with self.subTest(adapter):
                example = {**EXAMPLE_SETTINGS, "core": core_block(backend={"adapter": adapter})}
                self.assertEqual(CoreSettings.parse(example["core"], CORE_TEXTS).backend.adapter, adapter)
                self.assertEqual(ui_bot_core(example).backend, ui_bot_core(EXAMPLE_SETTINGS).backend)


class RootTests(RootCase):
    def test_the_root_reads_its_settings_once(self):
        with patch.object(examples_registry, "example_config", wraps=examples_registry.example_config) as reads:
            first, second = factory.root(), factory.root()
        self.assertIs(first, second)
        self.assertEqual((reads.call_count, factory.ExampleCatalog.read.call_count), (1, 1))
        self.assertEqual(first.core.backend.slot, "thinker-llm")
        self.assertEqual(len(StubSettings.parsed), 1)
        self.assertEqual(first.settings.block["display_label"], first.core.delegation.display_label)

    def test_concurrent_first_calls_read_the_settings_once(self):
        reads, read = [], examples_registry.example_config

        def slow_read(example_key):
            reads.append(example_key)
            time.sleep(0.1)
            return read(example_key)

        results = []
        with patch.object(examples_registry, "example_config", side_effect=slow_read):
            threads = [threading.Thread(target=lambda: results.append(factory.root())) for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(5)
        self.assertEqual((len(reads), len({id(result) for result in results})), (1, 1))

    def test_malformed_settings_fail_and_are_read_again(self):
        self.with_settings({**EXAMPLE_SETTINGS, "core": core_block(delegation={"nope": 1})})
        with self.assertRaisesRegex(ValueError, "the core's settings"):
            factory.root()
        with self.assertRaisesRegex(ValueError, "the core's settings"):
            factory.root()  # not cached: corrected settings are read on the next use

    def test_the_named_class_parses_its_block_with_the_cores_values(self):
        state = factory.root()
        self.assertIs(state.adapter, StubAdapter)
        self.assertEqual(StubSettings.parsed, [{**EXAMPLE_SETTINGS["frontend"]["cascade"], **handed(state.core)}])
        self.assertIsInstance(state.readiness, StubReadiness)
        factory.root.cache_clear()
        with (
            patch.object(factory, "frontend_adapters", return_value={"other": StubAdapter}),
            self.assertRaisesRegex(ValueError, "Unknown frontend adapter 'cascade'; known: other"),
        ):
            factory.root()

    def test_one_core_is_built_from_its_settings_and_defaults(self):
        built, threads = [], []

        def build_core(settings, *, endpoints, defaults):
            threads.append(threading.current_thread())
            built.append((settings, endpoints, defaults))
            return SimpleNamespace(session=None, one_shot_session=None)

        with (
            patch.object(factory, "build_core", build_core),
            patch.object(factory, "core_client", side_effect=lambda core: ("client of", core)),
        ):
            first, second = asyncio.run(factory.core()), asyncio.run(factory.core())
        self.assertIs(first, second)
        self.assertIsNot(threads[0], threading.main_thread())
        [(settings, endpoints, defaults)] = built
        self.assertEqual(settings, SHIPPED)
        self.assertEqual((defaults.model, defaults.tools), ("org/back", []))
        self.assertEqual((endpoints(None).base_url, endpoints("org/other").base_url), ("http://back/v1", "http://o/v1"))

    def test_the_live_root_loads_the_tracing_setup(self):
        # The root imports it for the setup it runs on import (.env, OpenTelemetry); the adapter reads only the switch.
        self.assertIs(vars(factory).get("tracing"), sys.modules["tracing"])


class SessionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        reset_stubs()

    async def test_each_session_asks_readiness_and_builds_its_adapter(self):
        state = stub_root()
        no_core = AssertionError("a session's factory built a delegation core")
        with (
            patch.object(factory, "root", return_value=state),
            patch.object(factory, "delegation_core", side_effect=no_core),
            patch.object(factory, "responses_client", side_effect=no_core),
        ):
            for delegation in (None, {"type": "responses", "responses": {"model": "org/back"}}):
                config = SessionConfig.model_validate({"model": "live-1", "delegation": delegation})
                parts = await factory.create_frontend_adapter(config, "websocket", None)
                self.assertEqual(parts.frontend.args, (config, "websocket", None, state.settings, CATALOG))
                self.assertEqual(parts, factory.SessionParts(frontend=parts.frontend))
        self.assertEqual(state.readiness.checks, 2)

    async def test_a_webrtc_peer_is_made_on_the_gateways_ice_servers(self):
        turn = {"urls": ["turn:turn.example:3478"], "username": "user", "credential": "secret"}
        state = stub_root()
        with patch.object(factory, "root", return_value=state):
            parts = await factory.create_frontend_adapter(SESSION, "webrtc", [turn, {"urls": ["stun:s:3478"]}])
        self.addAsyncCleanup(parts.connection.close_after_drain)
        self.assertEqual(parts.frontend.args, (SESSION, "webrtc", parts.connection, state.settings, CATALOG))
        servers = [IceServer(urls=turn["urls"], username="user", credential="secret"), IceServer(urls=["stun:s:3478"])]
        self.assertEqual(parts.connection.connection.ice_servers, servers)

    async def test_a_service_still_starting_refuses_the_session(self):
        state = stub_root(readiness=StubReadiness(down=("asr",)))
        with patch.object(factory, "root", return_value=state), self.assertRaises(ProtocolError) as refused:
            await factory.create_frontend_adapter(SESSION, "websocket", None)
        self.assertEqual(refused.exception.code, "server_error")
        self.assertIn("asr is still starting", str(refused.exception))
        self.assertEqual(StubAdapter.built, [])

    async def test_slow_reads_and_builds_stay_off_the_event_loop(self):
        ticks, during = [], []

        def slow(result):
            before = len(ticks)
            time.sleep(0.2)
            during.append(len(ticks) - before)
            return result

        class SlowAdapter(StubAdapter):
            @classmethod
            def from_config(cls, *args):
                return slow(super().from_config(*args))

        async def tick():
            while True:
                ticks.append(1)
                await asyncio.sleep(0.01)

        state = stub_root(adapter=SlowAdapter)
        ticker = asyncio.create_task(tick())
        try:
            with patch.object(factory, "root", lambda: slow(state)):
                parts = await factory.create_frontend_adapter(SESSION, "websocket", None)
        finally:
            ticker.cancel()
        self.assertIsInstance(parts.frontend, SlowAdapter)
        self.assertEqual(len(during), 2)
        self.assertTrue(all(count > 5 for count in during), during)


class EndpointTests(unittest.TestCase):
    def test_an_entry_uses_its_own_key_and_api_else_the_defaults(self):
        named = {**ENTRY, "api_key_env": "ACME_API_KEY"}
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-secret", "ACME_API_KEY": "acme-key"}):
            self.assertEqual(resolve_endpoint(named, {}, slot="llm").api_key, "acme-key")
            self.assertEqual(resolve_endpoint(ENTRY, {}, slot="llm").api_key, "nvapi-secret")
            for empty in ("", "   "):
                with (
                    self.subTest(value=empty),
                    patch.dict(os.environ, {"ACME_API_KEY": empty}),
                    self.assertRaisesRegex(ValueError, "ACME_API_KEY is empty: the 'llm' slot's entry"),
                ):
                    resolve_endpoint(named, {}, slot="llm")
        named = {**ENTRY, "api": "openai-responses"}
        self.assertEqual(resolve_endpoint(named, {}, slot="thinker-llm").api, "openai-responses")
        # The UI bot's body repeats the selected entry's model and URL.
        self.assertEqual(resolve_endpoint(named, ENTRY, slot="thinker-llm").api, "openai-responses")
        cloud = {"model_id": "org/model", "api": "openai-responses"}
        self.assertEqual(resolve_endpoint(cloud, {}, slot="thinker-llm").api, "openai-responses")
        self.assertEqual(resolve_endpoint(ENTRY, {}, slot="thinker-llm").api, "chat-completions")
        for body in ({"base_url": "http://mine/v1"}, {"model_id": "org/mine"}):
            with self.subTest(body=body):
                self.assertEqual(resolve_endpoint(named, body, slot="thinker-llm").api, "chat-completions")
        catalog = ExampleCatalog({"thinker-llm": [named]})
        self.assertEqual(factory.endpoint_lookup(catalog, "thinker-llm")(None).api, "openai-responses")

    def test_a_slot_with_no_model_is_filled_in_or_named(self):
        for entry in ({}, {"base_url": "http://host:8000/v1"}):
            with self.subTest(entry=entry), self.assertRaisesRegex(ValueError, "'thinker-llm' slot has no model"):
                resolve_endpoint(entry, {}, slot="thinker-llm")
        body = {"model_id": "org/custom", "base_url": "http://c/v1"}
        custom = resolve_endpoint({}, body, slot="thinker-llm")
        self.assertEqual((custom.model, custom.base_url), ("org/custom", "http://c/v1"))
        self.assertEqual(resolve_endpoint({"model_id": "org/model"}, {}, slot="llm").base_url, NVCF_LLM_BASE_URL)

    def test_an_endpoint_carries_its_entrys_services_key(self):
        listed = {**ENTRY, "id": "cloud:nemotron-lightning", "key": "nemotron-lightning"}
        self.assertEqual(resolve_endpoint(listed, {}, slot="llm").service_key, "nemotron-lightning")
        self.assertEqual(resolve_endpoint(ENTRY, {}, slot="llm").service_key, "")

    def test_a_named_model_keeps_the_settings_of_its_slot(self):
        self.assertEqual(PICKED.model("thinker-llm").model, "org/b")
        self.assertEqual(PICKED.model("llm", {"model_id": "org/custom"}).model, "org/custom")
        self.assertEqual(
            (PICKED.service("asr")["server"], PICKED.models("thinker-llm")), ("asr:50052", ["org/b", "org/a"])
        )
        with self.assertRaisesRegex(ValueError, "'tts' slot has no model"):
            PICKED.model("tts")
        self.assertEqual(PICKED.named("thinker-llm", "org/a").extra_params, {"temperature": 0.7})
        self.assertEqual(PICKED.named("llm", "org/a").extra_params, {"temperature": 0})
        self.assertEqual(PICKED.named("llm", "org/b").base_url, "http://thinker/v1")
        with self.assertRaisesRegex(LookupError, "'org/z' is not in the catalog"):
            PICKED.named("thinker-llm", "org/z")

    def test_a_session_selects_an_entry_by_its_exact_key(self):
        first = {**ENTRY, "id": "cloud:first", "key": "first"}
        second = {**ENTRY, "id": "local:second", "key": "second", "response_format": "none"}
        catalog = ExampleCatalog({"llm": [first, second]})
        for selected, key in {"local:second": "second", "second": "second", "custom-1": "first", "": "first"}.items():
            with self.subTest(selected=selected):
                self.assertEqual(catalog.entry("llm", selected)["key"], key)
                self.assertEqual(catalog.model("llm", {"llm_id": selected}).service_key, key)
        self.assertEqual(catalog.model("llm", {"llm_id": "second"}).response_format, "none")
        self.assertEqual(catalog.model("llm").service_key, "first")

    def test_the_sessions_backend_choice_reaches_the_lookup(self):
        body = {"thinker_model_id": "org/picked", "thinker_base_url": "http://picked/v1", "model_id": "org/front"}
        overrides = pipeline.backend_overrides(body)
        self.assertEqual(overrides, {"model_id": "org/picked", "base_url": "http://picked/v1"})
        endpoints = factory.endpoint_lookup(PICKED, "thinker-llm", overrides)
        self.assertEqual((endpoints(None).model, endpoints(None).base_url), ("org/picked", "http://picked/v1"))
        self.assertEqual(endpoints("org/a").base_url, "http://thinker/v1")
        with self.assertRaisesRegex(LookupError, "'org/unlisted' is not in the catalog"):
            endpoints("org/unlisted")


def selecting(entry: dict, selected: str | None = None) -> dict:
    """Return the UI bot's body for both roles, as the shared runtime fills it from the selected entry."""
    fields = {"llm_id": selected or entry["id"], "model_id": entry["model_id"], "base_url": entry["base_url"]}
    return {**fields, **{f"thinker_{key}": value for key, value in fields.items()}}


class SelectionTests(unittest.TestCase):
    def keys(self, catalog: ExampleCatalog, body: dict) -> list[tuple[str, str]]:
        """Return each role's URL and key: the talker's, then the backend's default model's."""
        talker = catalog.model("llm", body)
        backend = factory.endpoint_lookup(catalog, "thinker-llm", pipeline.backend_overrides(body))(None)
        return [(talker.base_url, talker.api_key), (backend.base_url, backend.api_key)]

    def test_each_role_uses_the_key_of_the_selected_entry(self):
        cases = (([OTHER, NEMOTRON], NEMOTRON, "nvapi-secret"), ([NEMOTRON, OTHER], OTHER, "other-key"))
        with patch.dict(os.environ, KEYS):
            for listed, entry, key in cases:
                catalog = ExampleCatalog({"llm": listed, "thinker-llm": listed})
                for selected in (entry["id"], entry["key"]):
                    with self.subTest(selected=selected, default=listed[0]["key"]):
                        self.assertEqual(self.keys(catalog, selecting(entry, selected)), [(entry["base_url"], key)] * 2)

    def test_a_custom_url_gets_the_nvidia_key_never_an_entrys_own(self):
        catalog = ExampleCatalog({"llm": [OTHER], "thinker-llm": [OTHER]})
        custom = {"model_id": "org/mine", "base_url": "http://mine/v1", "id": "custom-1"}
        with patch.dict(os.environ, KEYS):
            unselected = {key: value for key, value in selecting(custom).items() if not key.endswith("llm_id")}
            for body in (selecting(custom), unselected):
                with self.subTest(selected=body.get("llm_id")):
                    self.assertEqual(self.keys(catalog, body), [("http://mine/v1", "nvapi-secret")] * 2)
            self.assertEqual(catalog.model("llm").api_key, "other-key")


class DefaultPromptTests(RootCase, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp()
        pipeline.bot_root.cache_clear()
        self.addCleanup(pipeline.bot_root.cache_clear)
        self.enterContext(patch.object(pipeline, "frontend_adapters", return_value={"cascade": StubAdapter}))

    async def system_message(self, build) -> str:
        """Return the backend model's instructions for a request that sets none, on the core ``build`` makes."""
        chat = FakeChat([chunk("Done.", finish="stop")])
        with patch.object(providers, "AsyncOpenAI", return_value=chat):
            core = build()
            self.addAsyncCleanup(core.aclose)
            client = factory.core_client(core)
            conversation = (await client.create_conversation({}))["id"]
            self.addAsyncCleanup(client.delete_conversation, conversation)
            user = {"type": "message", "role": "user", "content": "What can you do for me?"}
            params = {"conversation": conversation, "input": [user], "metadata": {"live_delegation_id": "d1"}}
            stream = await client.create_response({"stream": True, **params})
            async with asyncio.timeout(5):
                events = [event async for event in stream]
        self.assertEqual(events[-1]["type"], "response.completed")
        (request,) = chat.requests
        self.assertEqual(request["messages"][0]["role"], "system")
        return request["messages"][0]["content"]

    def ui_bot_core(self):
        """Build the UI bot's delegation core as ``pipeline.py`` does."""
        state, prompts = pipeline.bot_root(), load_prompts()
        instructions = factory.ui_bot_instructions(prompts, prompts.text("Channel", state.channel))
        endpoints = factory.endpoint_lookup(state.catalog, state.core.backend.slot)
        defaults = factory.backend_defaults(state.core, endpoints, instructions)
        return build_core(state.core, endpoints=endpoints, defaults=defaults)

    async def test_no_instructions_get_the_generic_or_the_cafe_prompt(self):
        groups = yaml.safe_load(Path(factory.__file__).with_name("prompts.yaml").read_text(encoding="utf-8"))
        groups = groups["agent_prompts"]
        channel, core = groups["Channel"]["cafe"]["content"].strip(), groups["Core"]
        notes = (core["texts.caller_note"]["content"], core["texts.voice_note"]["content"])
        live = await self.system_message(factory.delegation_core)
        self.assertEqual(live, compose(groups["Backend"]["generic"]["content"].strip(), *notes))
        self.assertNotIn(channel, live)
        self.assertNotIn("Bluebird", live)
        ui_bot = await self.system_message(self.ui_bot_core)
        cafe = f"{groups['Backend']['cafe']['content'].strip()}\n\n{channel}"
        self.assertEqual(ui_bot, compose(cafe, *notes, core["texts.result_formats.speak_display"]["content"]))
