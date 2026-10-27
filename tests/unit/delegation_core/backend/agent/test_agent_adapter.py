# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Tests for the backend LLM agent's adapter: its source, its settings and the backend adapter contract."""

import asyncio
import json
import unittest
from dataclasses import replace

from delegation_core.app import build_core
from delegation_core.backend.adapters import ADAPTERS
from delegation_core.backend.agent import AgentAdapter, AgentSettings, AgentSource, ConductorSettings, LlmConductor
from delegation_core.backend.llm import LlmRun, LlmSettings
from delegation_core.backend.port import ConductorInput, OneShotSource
from delegation_core.backend.providers import ChatCompletionsBackend, OpenAIResponsesBackend
from delegation_core.backend.tools import CafeTools
from delegation_core.config import BackendChoice, ModelEndpoint
from delegation_core.im.input import BackendTask
from delegation_core.tables import pick
from tests.unit.contracts.backend import (
    MENU_FUNCTION,
    BackendAdapterContract,
    FakeTaskContext,
    call_round,
    text_round,
)
from tests.unit.contracts.base import collect
from tests.unit.delegation_core.backend.agent.doubles import (
    ENDPOINT,
    LIMITS,
    OTHER,
    SETTINGS,
    SHIPPED,
    VOCABULARY,
    Model,
    a_request,
    decisions,
    lookup,
    source,
    start,
)
from tests.unit.delegation_core.backend.llm.doubles import FakeChat, RecordingTools, chunk, tool_fragment

# What tau2 sends: functions, tool_choice auto, one call per round, maybe an effort.
TAU2 = {
    "model": "test/model",
    "instructions": "Policy.",
    "tools": [{"type": "function", "name": "get_menu", "parameters": {"type": "object", "properties": {}}}],
    "tool_choice": "auto",
    "parallel_tool_calls": False,
    "reasoning": {"effort": "low"},
}


class AgentAdapterContractTests(BackendAdapterContract, unittest.IsolatedAsyncioTestCase):
    """The backend adapter contract on the backend LLM agent, with no expected failure."""

    def make_adapter(self, scenario):
        self.gate = asyncio.Event()
        slow = text_round("Old.")
        slow.insert(1, self.gate)
        rounds = {
            "answer": [text_round("Done.")],
            "own_call": [call_round("lookup", {}), text_round("Done.", "resp_2")],
            "tool_fails": [call_round("lookup", {}), text_round("Done.", "resp_2")],
            "app_call": [call_round("get_menu", {}, said="Let me look."), text_round("Done.", "resp_2")],
            "slow": [slow, text_round("Done.", "resp_2")],
            "tail_error": [[*text_round("Done."), RuntimeError("after the terminal")], text_round("Again.", "resp_2")],
        }[scenario]
        self.workers = Model(*rounds)
        tools = {"own_call": [RecordingTools()], "tool_fails": [RecordingTools(fail=["lookup"])]}.get(scenario)
        return source(Model(), self.workers, tools=tools).new()

    def release(self):
        self.gate.set()

    def sent_instructions(self):
        return [payload.get("instructions") for payload in self.workers.payloads]


def build(block, *, tools=lambda: [], endpoints=lookup) -> AgentSource:
    return AgentSource.from_settings(block, endpoints=endpoints, tools=tools, vocabulary=VOCABULARY)


class SettingsTests(unittest.TestCase):
    def test_the_adapter_table_loads_the_source(self):
        self.assertIs(pick(ADAPTERS, "llm-agent", "backend adapter")(), AgentSource)

    def test_the_shipped_block_parses_to_the_defaults(self):
        settings = build(SHIPPED.backend.blocks["llm-agent"]).settings
        self.assertEqual(settings, AgentSettings())
        self.assertEqual((settings.max_running, settings.conductor.timeout_seconds), (3, 20.0))
        self.assertIs(settings.conductor.provider, ChatCompletionsBackend)
        self.assertEqual(settings.worker, LlmSettings())

    def test_every_key_parses_and_names_become_classes(self):
        block = {
            "max_running": 4,
            "conductor": {
                "provider": "openai-responses",
                "model": "other/model",
                "timeout_seconds": 4,
                "memory_tokens": 2000,
                "max_output_tokens_per_request": 128,
                "temperature": None,
                "reasoning_effort": "low",
            },
            "worker": {"provider": "openai-responses", "unlisted_models": "fallback", "max_tool_rounds": 5},
        }
        settings = build(block).settings
        conductor, worker = settings.conductor, settings.worker
        self.assertEqual(settings.max_running, 4)
        self.assertIs(conductor.provider, OpenAIResponsesBackend)
        self.assertEqual(
            (conductor.model, conductor.timeout_seconds, conductor.memory_tokens), ("other/model", 4.0, 2000)
        )
        self.assertEqual((conductor.max_output_tokens_per_request, conductor.temperature), (128, None))
        self.assertEqual(conductor.reasoning_effort, "low")
        self.assertIs(worker.provider, OpenAIResponsesBackend)
        self.assertEqual((worker.unlisted_models.name, worker.max_tool_rounds), ("fallback", 5))

    def test_a_bad_block_fails_when_the_core_is_built(self):
        cases = {
            "not a mapping": ("fast", "backend.llm-agent must be a mapping"),
            "an unknown key": ({"workers": 3}, "backend.llm-agent has unknown keys: workers"),
            "no task may run": ({"max_running": 0}, "max_running must be a whole number of at least 1"),
            "a fraction of tasks": ({"max_running": 2.5}, "whole number"),
            "a boolean count": ({"max_running": True}, "whole number"),
            "an unknown conductor key": ({"conductor": {"tool_choice": "auto"}}, "conductor has unknown keys"),
            "an unknown provider": ({"conductor": {"provider": "hermes"}}, "Unknown backend provider 'hermes'"),
            "a negative wait": ({"conductor": {"timeout_seconds": -1}}, "timeout_seconds must be a positive number"),
            "no memory": ({"conductor": {"memory_tokens": 0}}, "memory_tokens must be a whole number of at least 1"),
            "no output": ({"conductor": {"max_output_tokens_per_request": 1.5}}, "whole number"),
            "a negative temperature": (
                {"conductor": {"temperature": -0.1}},
                "temperature must be a non-negative number",
            ),
            "an empty effort": ({"conductor": {"reasoning_effort": " "}}, "reasoning_effort must be non-empty text"),
            "an empty model": ({"conductor": {"model": ""}}, "model must be non-empty text"),
            "a worker's conductor": ({"worker": {"conductor": "steer"}}, "worker takes no conductor"),
            "a worker's unknown key": ({"worker": {"fold": True}}, "backend.llm-agent.worker: .*unknown keys: fold"),
            "a worker's bad provider": ({"worker": {"provider": "x"}}, "backend.llm-agent.worker: Unknown backend"),
            "a worker that is no mapping": ({"worker": []}, "worker must be a mapping"),
        }
        for name, (block, message) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                build(block)

    def test_the_conductors_model_is_resolved_when_the_core_is_built(self):
        self.assertEqual(build({}).conductor_endpoint.model, ENDPOINT.model)
        self.assertEqual(build({"conductor": {"model": "other/model"}}).conductor_endpoint.base_url, OTHER.base_url)
        with self.assertRaisesRegex(ValueError, r"conductor.model 'unlisted/model' is not in the model catalog"):
            build({"conductor": {"model": "unlisted/model"}})

    def test_the_conductor_runs_on_a_copy_of_its_endpoint_with_its_own_temperature(self):
        thinking = {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}, "temperature": 0.7}
        endpoint = ModelEndpoint("test/model", "http://localhost:1/v1", extra_params=thinking)
        built = build({}, endpoints=lambda model: endpoint)
        self.assertEqual(built.conductor_endpoint.extra_params["temperature"], 0.0)
        self.assertEqual(endpoint.extra_params["temperature"], 0.7)  # the catalog's entry is unchanged
        self.assertIs(
            build({"conductor": {"temperature": None}}, endpoints=lambda model: endpoint).conductor_endpoint, endpoint
        )
        provider = built.conductor_provider()
        self.assertIsInstance(provider, ChatCompletionsBackend)
        request = provider.request({"model": "test/model", "input": [], "reasoning": {"effort": "none"}})
        self.assertEqual(request["temperature"], 0.0)
        self.assertFalse(request["extra_body"]["chat_template_kwargs"]["enable_thinking"])  # thinking off


class SourceTests(unittest.IsolatedAsyncioTestCase):
    def test_it_serves_no_one_shot_session_and_the_core_refuses_it_there(self):
        built = build({})
        self.assertNotIsInstance(built, OneShotSource)
        settings = replace(
            SHIPPED,
            backend=BackendChoice.parse(
                {"slot": "s", "prompt": "p", "adapter": "llm-agent", "one_shot_adapter": "llm-agent"}
            ),
        )
        with self.assertRaisesRegex(ValueError, "cannot serve one-shot sessions"):
            build_core(settings, endpoints=lookup, defaults=BackendTask("test/model", "x", []))

    async def test_the_model_check_and_the_settings_check_are_the_workers(self):
        built = build({})
        await built.check_model("other/model")
        with self.assertRaisesRegex(ValueError, "does not serve the model 'unlisted/model'"):
            await built.check_model("unlisted/model")
        await build({"worker": {"unlisted_models": "fallback"}}).check_model("unlisted/model")
        self.assertEqual(built.unsupported(TAU2), [])  # tau2's session body
        self.assertEqual([param for param, _ in built.unsupported({"tools": [{"type": "web_search"}]})], ["tools"])

    def test_it_describes_what_it_serves_without_credentials(self):
        described = build({"max_running": 2}, tools=lambda: [CafeTools()]).describe()
        self.assertEqual(
            {key: described[key] for key in ("adapter", "model", "max_running")},
            {"adapter": "llm-agent", "model": "test/model", "max_running": 2},
        )
        self.assertEqual(
            (described["conductor"]["model"], described["conductor"]["provider"]), ("test/model", "chat-completions")
        )
        self.assertEqual(
            described["worker"], {"provider": "chat-completions", "unlisted_models": "report", "max_tool_rounds": 8}
        )
        self.assertEqual([tool["name"] for tool in described["tools"]], ["get_menu", "manage_cart", "place_order"])
        self.assertNotIn("secret-key", json.dumps(described))


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    def test_each_session_gets_its_adapter_its_conductor_and_its_tool_sources(self):
        settings = replace(SETTINGS, max_running=5)
        built = AgentSource(settings, endpoints=lookup, tools=lambda: [CafeTools()], vocabulary=VOCABULARY)
        first, second = built.new(), built.new()
        self.assertIsInstance(first, AgentAdapter)
        self.assertEqual(first.max_running, 5)
        self.assertIs(first.conductor(), first.conductor())  # one conductor per session, with its memory
        self.assertIsInstance(first.conductor(), LlmConductor)
        self.assertIsNot(first.conductor(), second.conductor())
        self.assertIsNot(first.workers.tools[0], second.workers.tools[0])
        self.assertEqual(first.workers.attempts, 2)

    async def test_a_run_is_an_llm_run_and_makes_no_model_call_before_its_events(self):
        conductor, workers = Model(), Model(text_round("Done."))
        adapter = source(conductor, workers).new()
        adapter.conductor()
        run = adapter.start(object())
        self.assertIsInstance(run, LlmRun)
        self.assertEqual((conductor.payloads, workers.payloads, run.round), ([], [], 0))

    async def test_close_releases_the_workers_and_the_conductors_providers(self):
        conductor, workers = Model(decisions(start("D1"))), Model()
        adapter = source(conductor, workers).new()
        adapter.workers.provider("test/model")
        await adapter.conductor().decide(_given("D1"), _place)
        await adapter.close()
        self.assertEqual((conductor.closed, workers.closed), (1, 1))

    async def test_a_conductor_made_by_default_runs_on_its_settings_provider(self):
        settings = replace(SETTINGS, conductor=ConductorSettings(provider=OpenAIResponsesBackend))
        built = AgentSource(settings, endpoints=lookup, tools=lambda: [], vocabulary=VOCABULARY)
        provider = built.conductor_provider()
        self.assertIsInstance(provider, OpenAIResponsesBackend)
        self.assertEqual(provider.limits, SETTINGS.worker.limits)
        await provider.close()

    async def test_workers_keep_one_call_per_round_and_the_requests_tool_choice(self):
        # tau2 asks for one call per round (parallel_tool_calls: false) with tool_choice auto: the workers' chat
        # provider keeps the reply's first call and sends the request's choice.
        two_calls = [tool_fragment(0, "p1", "get_menu", "{}"), tool_fragment(1, "p2", "lookup", "{}")]
        chat = FakeChat([chunk(tool_calls=two_calls), chunk(finish="tool_calls")])
        built = AgentSource(
            SETTINGS,
            endpoints=lookup,
            tools=lambda: [],
            vocabulary=VOCABULARY,
            make_provider=lambda endpoint: ChatCompletionsBackend(endpoint, LIMITS, chat),
        )
        events = await collect(built.new().start(OneCallTask(declared=[MENU_FUNCTION])).events())
        calls = [e["item"]["name"] for e in events if e["type"] == "response.output_item.done"]
        self.assertEqual(calls, ["get_menu"])
        self.assertEqual(chat.requests[0]["tool_choice"], "auto")


class OneCallTask(FakeTaskContext):
    """A task whose request asks for one call per round, as tau2's session body does."""

    def request(self):
        return {**super().request(), "parallel_tool_calls": False}


def _given(*delegations):
    return ConductorInput(tuple(a_request(d, "Teas?") for d in delegations), (), (), 0)


async def _place(decision):
    return "Placed."


if __name__ == "__main__":
    unittest.main()
