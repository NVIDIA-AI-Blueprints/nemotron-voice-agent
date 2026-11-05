# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Tests for the ``llm`` source and its sessions: its block, what it serves, its models, its rule conductors."""

import dataclasses
import json
import unittest

from examples.frontend_backend_live.delegation_core.backend.conductors import StartEach, SteerOpenWork
from examples.frontend_backend_live.delegation_core.backend.llm import LlmSettings, LlmSource
from examples.frontend_backend_live.delegation_core.backend.llm.run import RoundFailedError
from examples.frontend_backend_live.delegation_core.backend.port import (
    ConductorInput,
    Start,
    Steer,
    TaskSummary,
    WorkSummary,
)
from examples.frontend_backend_live.delegation_core.backend.tools import TOOL_SOURCES, CafeTools
from examples.shared.delegation_contract import ModelEndpoint
from tests.unit.contracts.backend import (
    ConductorContract,
    FakeDecisionTarget,
    a_request,
    a_round,
    an_ended_work,
    call_round,
    text_round,
)
from tests.unit.delegation_core.backend.llm.doubles import FAST, IMTask, RecordingTools, ScriptedProvider, drive, source

DEFAULT = ModelEndpoint("test/model", "http://default:1/v1", api_key="secret-key", extra_params={"temperature": 0.2})
OTHER = ModelEndpoint("other/model", "http://other:2/v1")
MENU_OUTPUT = [{"type": "function_call_output", "call_id": "call_1", "output": "{}"}]


def lookup(model):
    return {None: DEFAULT, DEFAULT.model: DEFAULT, OTHER.model: OTHER}[model]  # a KeyError is a LookupError


def build(block, tools=lambda: [], make_provider=None):
    if make_provider is None:
        return LlmSource.from_settings(block, endpoints=lookup, tools=tools, vocabulary=None)
    return LlmSource(LlmSettings.parse(block), endpoints=lookup, tools=tools, make_provider=make_provider)


class ModelTask(IMTask):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def request(self):
        return {**super().request(), "model": self.model}


class SourceTests(unittest.IsolatedAsyncioTestCase):
    def test_the_block_names_what_it_serves_and_it_describes_that_without_credentials(self):
        names = {"provider": "openai-responses", "conductor": "start_each", "unlisted_models": "fallback"}
        block = {**names, "max_tool_rounds": 4, "provider_retries": 1, "idle_timeout_seconds": 10.5}
        described = build(block, tools=lambda: [TOOL_SOURCES["cafe"].from_settings({})]).describe()
        expected = {**names, "adapter": "llm", "model": "test/model", "max_running": 1, "max_tool_rounds": 4}
        self.assertEqual({key: described[key] for key in expected}, expected)
        self.assertEqual([tool["name"] for tool in described["tools"]], ["get_menu", "manage_cart", "place_order"])
        self.assertNotIn("secret-key", json.dumps(described))
        with self.assertRaisesRegex(ValueError, "takes no settings"):
            TOOL_SOURCES["cafe"].from_settings({"url": "http://example"})
        described["tools"][0]["name"] = "changed"  # the source's schemas are a copy
        self.assertEqual(CafeTools().tools()[0]["name"], "get_menu")
        defaults = build({}).describe()
        self.assertEqual(
            [defaults[key] for key in ("provider", "conductor", "unlisted_models")],
            ["chat-completions", "steer", "report"],
        )

    def test_a_bad_block_fails_when_the_core_is_built(self):
        cases = {
            "an unknown provider": ({"provider": "hermes"}, "Unknown backend provider 'hermes'"),
            "an unknown conductor": ({"conductor": "fold"}, "Unknown llm conductor 'fold'"),
            "an unknown policy": ({"unlisted_models": "guess"}, "Unknown unlisted-model policy"),
            "an unknown key": ({"merge_pending": True}, "unknown keys: merge_pending"),
            "a fraction of rounds": ({"max_tool_rounds": 2.5}, "whole number"),
            "a boolean count": ({"provider_retries": True}, "whole number"),
            "no rounds": ({"max_tool_rounds": 0}, "at least 1"),
            "a negative wait": ({"idle_timeout_seconds": -1}, "positive"),
            "not a mapping": ("fast", "mapping"),
        }
        for name, (block, message) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                build(block)

    def test_the_settings_its_provider_cannot_honor_are_named(self):
        settings = {"tools": [{"type": "web_search"}], "tool_choice": "auto", "service_tier": "priority"}
        self.assertEqual([param for param, _ in build({}).unsupported(settings)], ["tools", "service_tier"])
        self.assertEqual(build({"provider": "openai-responses"}).unsupported(settings), [])  # it passes them on
        hosted = [{"type": "web_search"}, {"type": "web_search"}, {"type": "file_search"}]
        problems = build({}).unsupported({"tools": hosted, "service_tier": "flex", "text": {"verbosity": "low"}})
        self.assertEqual([param for param, _ in problems], ["tools", "tools", "service_tier", "text.verbosity"])
        self.assertIn("web_search", problems[0][1])
        self.assertEqual(build({}).unsupported({"service_tier": "auto"}), [])

    async def test_each_model_runs_on_its_own_endpoint_with_one_provider_per_session(self):
        rounds = {
            DEFAULT.model: [text_round("Done.")],
            OTHER.model: [call_round("get_menu", {}), text_round("Done.", "resp_2"), text_round("Done.", "resp_3")],
        }
        made = []

        def make(endpoint):
            made.append(ScriptedProvider(*rounds[endpoint.model], endpoint=endpoint))
            return made[-1]

        session = build({}, make_provider=make).new()
        for model in (None, "other/model", "other/model"):
            task = ModelTask(model)
            run = session.start(task)
            await drive(run, task)
            if task.calls_seen:  # the app's call: the session keeps its model's provider over the resume
                task.give_outputs(MENU_OUTPUT)
                await run.resume()
                await drive(run, task)
        self.assertEqual([(provider.endpoint, len(provider.payloads)) for provider in made], [(DEFAULT, 1), (OTHER, 3)])
        self.assertEqual({payload["model"] for payload in made[1].payloads}, {"other/model"})

    async def test_an_unlisted_model_is_refused_or_runs_on_the_default_endpoint_as_the_policy_says(self):
        made = []

        def make(endpoint):
            made.append(ScriptedProvider(text_round("Done."), endpoint=endpoint))
            return made[-1]

        report = build({}, make_provider=make)
        await report.check_model("other/model")
        with self.assertRaisesRegex(ValueError, "not in the model catalog"):
            await report.check_model("nvidia/unlisted")
        task = ModelTask("nvidia/unlisted")
        with self.assertRaisesRegex(ValueError, "not in the model catalog"):
            await drive(report.new().start(task), task)
        self.assertEqual(made, [])
        fallback = build({"unlisted_models": "fallback"}, make_provider=make)
        await fallback.check_model("nvidia/unlisted")
        task = ModelTask("nvidia/unlisted")
        await drive(fallback.new().start(task), task)
        self.assertEqual(made[0].endpoint, dataclasses.replace(DEFAULT, model="nvidia/unlisted"))
        self.assertEqual(made[0].payloads[0]["model"], "nvidia/unlisted")

    async def test_each_session_gets_fresh_tool_sources_starts_no_call_early_and_closes_its_providers(self):
        providers = [
            ScriptedProvider(call_round("manage_cart", {"action": "add", "item_id": "lat"}), text_round("Added."))
            for _ in range(2)
        ]
        made = list(providers)
        built = LlmSource(FAST, endpoints=lookup, tools=lambda: [CafeTools()], make_provider=lambda _: made.pop(0))
        for provider in providers:
            session, task = built.new(), IMTask()
            run = session.start(task)
            self.assertEqual((provider.payloads, run.round), ([], 0))  # no model call before events()
            await drive(run, task)
            self.assertEqual(json.loads(task.outcomes["call_1"][0])["cart"]["item_count"], 1)  # a cart of its own
            await session.close()
            self.assertEqual(provider.closed, 1)

    async def test_a_one_shot_session_offers_no_server_tool_and_never_retries(self):
        provider = ScriptedProvider(a_round(), text_round("Done."))
        task = IMTask()
        with self.assertRaisesRegex(RoundFailedError, "answered nothing"):
            await drive(source(provider, tools=[RecordingTools()]).new_one_shot().start(task), task)
        self.assertEqual(provider.opened, 1)
        self.assertEqual([tool["name"] for tool in provider.payloads[0]["tools"]], ["get_menu"])  # the app's only


class SteerOpenWorkContractTests(ConductorContract, unittest.IsolatedAsyncioTestCase):
    def make_conductor(self):
        return SteerOpenWork()


class StartEachContractTests(ConductorContract, unittest.IsolatedAsyncioTestCase):
    def make_conductor(self):
        return StartEach()


def a_work(work_id: str, state: str, seconds: float) -> WorkSummary:
    task = TaskSummary(f"{work_id}.1", state, None, (), False, None, seconds)
    return WorkSummary(work_id, state, f"d_{work_id}", (a_request(f"d_{work_id}", "Earlier."),), (task,), seconds)


async def decide(conductor, *works: WorkSummary, arrivals=("d_new",)) -> list:
    target = FakeDecisionTarget()
    await conductor.decide(
        ConductorInput(tuple(a_request(d, "Also this.") for d in arrivals), works, (), 0), target.place
    )
    return target.decisions


class RuleTests(unittest.IsolatedAsyncioTestCase):
    async def test_steer_joins_the_arrivals_together_to_the_newest_open_work_or_starts_one(self):
        running, queued = a_work("W1", "running", 9.0), a_work("W2", "queued", 0.0)
        waiting = a_work("W3", "waiting_client", 4.0)
        just_ended = a_work("W5", "answered", 0.5)  # newer than the open one, and still never joined
        cases = [
            ((an_ended_work("W0"), running, queued, an_ended_work("W4")), Steer("W2", ("d_1", "d_2"))),
            ((a_work("W1", "queued", 0.0), queued), Steer("W2", ("d_1", "d_2"))),  # a tie: the later one
            ((waiting,), Steer("W3", ("d_1", "d_2"))),  # waiting for the app's outputs is open too
            ((running, just_ended), Steer("W1", ("d_1", "d_2"))),
            ((just_ended, an_ended_work("W0")), Start(("d_1", "d_2"))),
        ]
        for works, decision in cases:
            with self.subTest(decision=decision, works=[work.work_id for work in works]):
                self.assertEqual(await decide(SteerOpenWork(), *works, arrivals=("d_1", "d_2")), [decision])
        self.assertEqual(await decide(SteerOpenWork(), running, arrivals=()), [])

    async def test_start_each_starts_a_work_for_each_arrival(self):
        decisions = await decide(StartEach(), a_work("W1", "running", 3.0), arrivals=("d_1", "d_2"))
        self.assertEqual(decisions, [Start(("d_1",)), Start(("d_2",))])


if __name__ == "__main__":
    unittest.main()
