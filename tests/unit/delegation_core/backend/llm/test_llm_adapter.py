# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Tests for the ``llm`` adapter: its source, its adapter and the backend adapter contract."""

import asyncio
import json
import unittest

from delegation_core.backend.adapters import ADAPTERS
from delegation_core.backend.conductors import StartEach, SteerOpenWork
from delegation_core.backend.llm import LlmAdapter, LlmRun, LlmSettings, LlmSource
from delegation_core.backend.port import OneShotSource
from delegation_core.backend.providers import ChatCompletionsBackend, OpenAIResponsesBackend
from delegation_core.backend.tools import CafeTools
from delegation_core.config import ModelEndpoint
from delegation_core.tables import pick
from tests.unit.contracts.backend import BackendAdapterContract, ScriptedBackend, call_round, check_events, text_round
from tests.unit.delegation_core.backend.llm.doubles import (
    ENDPOINT,
    FAST,
    FakeChat,
    IMTask,
    RecordingTools,
    adapter,
    chunk,
    drive,
    items_of,
    source,
    tool_fragment,
)


class LlmAdapterContractTests(BackendAdapterContract, unittest.IsolatedAsyncioTestCase):
    """The backend adapter contract on the backend LLM, with no expected failure."""

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
        self.backend = ScriptedBackend(rounds)
        tools = {"own_call": [RecordingTools()], "tool_fails": [RecordingTools(fail=["lookup"])]}.get(scenario)
        return adapter(self.backend, tools=tools)

    def release(self):
        self.gate.set()

    def sent_instructions(self):
        return [payload.get("instructions") for payload in self.backend.payloads]


def lookup(model):
    listed = {"test/model": ENDPOINT, "other/model": ModelEndpoint("other/model", "http://localhost:2/v1")}
    if model is None:
        return ENDPOINT
    return listed[model]


class SourceTests(unittest.IsolatedAsyncioTestCase):
    def build(self, block, tools=lambda: []):
        return LlmSource.from_settings(block, endpoints=lookup, tools=tools, vocabulary=None)

    def test_the_adapter_table_loads_the_source(self):
        self.assertIs(pick(ADAPTERS, "llm", "backend adapter")(), LlmSource)

    def test_the_frozen_block_parses_and_its_names_become_classes(self):
        block = {
            "provider": "openai-responses",
            "conductor": "start_each",
            "max_tool_rounds": 4,
            "provider_retries": 1,
            "first_event_timeout_seconds": 20,
            "idle_timeout_seconds": 10.5,
            "server_tool_timeout_seconds": 30,
            "unlisted_models": "fallback",
            "round_cap_note": "Answer now.",
        }
        settings = self.build(block).settings
        self.assertIs(settings.provider, OpenAIResponsesBackend)
        self.assertIs(settings.conductor, StartEach)
        self.assertEqual(settings.unlisted_models.name, "fallback")
        self.assertEqual((settings.max_tool_rounds, settings.idle_timeout_seconds), (4, 10.5))
        self.assertEqual((settings.limits.provider_retries, settings.limits.first_event_timeout_seconds), (1, 20.0))
        defaults = self.build({}).settings
        self.assertEqual((defaults.provider, defaults.conductor), (ChatCompletionsBackend, SteerOpenWork))
        self.assertEqual(defaults.unlisted_models.name, "report")

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
            "an empty note": ({"round_cap_note": " "}, "non-empty"),
            "not a mapping": ("fast", "mapping"),
        }
        for name, (block, message) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                self.build(block)

    def test_it_serves_one_shot_sessions_with_one_attempt_and_no_server_tools(self):
        built = self.build({}, tools=lambda: [CafeTools()])
        self.assertIsInstance(built, OneShotSource)
        one_shot = built.new_one_shot()
        self.assertEqual((one_shot.attempts, one_shot.tools, one_shot.max_running), (1, (), 1))
        session = built.new()
        self.assertEqual((session.attempts, [tool.label for tool in session.tools]), (2, ["cafe"]))

    async def test_two_sessions_get_two_carts(self):
        built = self.build({}, tools=lambda: [CafeTools()])
        first, second = built.new(), built.new()
        await first.tools[0].execute("c1", "manage_cart", json.dumps({"action": "add", "item_id": "lat"}))
        self.assertEqual((first.tools[0].snapshot()["item_count"], second.tools[0].snapshot()["item_count"]), (1, 0))

    def test_it_describes_what_it_serves_without_credentials(self):
        described = self.build({"conductor": "start_each"}, tools=lambda: [CafeTools()]).describe()
        self.assertEqual(
            {key: described[key] for key in ("adapter", "provider", "conductor", "model", "max_running")},
            {
                "adapter": "llm",
                "provider": "chat-completions",
                "conductor": "start_each",
                "model": "test/model",
                "max_running": 1,
            },
        )
        self.assertEqual([tool["name"] for tool in described["tools"]], ["get_menu", "manage_cart", "place_order"])
        self.assertNotIn("not-needed", json.dumps(described))  # no key
        json.dumps(described)

    def test_the_adapter_gives_its_rule_and_runs(self):
        session = source(ScriptedBackend([text_round("Done.")]), settings=LlmSettings(conductor=StartEach)).new()
        self.assertIsInstance(session, LlmAdapter)
        self.assertIsInstance(session.conductor(), StartEach)
        self.assertIsInstance(session.conductor(), StartEach)
        self.assertIsInstance(adapter(ScriptedBackend([])).conductor(), SteerOpenWork)
        self.assertEqual(LlmAdapter.max_running, 1)

    async def test_a_run_starts_no_model_call_and_close_releases_the_providers(self):
        provider = ScriptedBackend([text_round("Done.")])
        closed = []

        async def close():
            closed.append(True)

        provider.close = close
        session = adapter(provider, settings=FAST)
        run = session.start(object())
        self.assertIsInstance(run, LlmRun)
        self.assertEqual((provider.payloads, run.round), ([], 0))
        session.provider("test/model")
        await session.close()
        self.assertEqual(closed, [True])


class ChatEndToEndTests(unittest.IsolatedAsyncioTestCase):
    """A task on the real chat provider (template, translator and run), its client scripted."""

    async def test_a_call_round_then_an_answer_on_the_chat_provider(self):
        chat = FakeChat(
            [
                chunk("Let me check."),
                chunk(tool_calls=[tool_fragment(0, "p_1", "lookup", "{}")]),
                chunk(finish="tool_calls"),
            ],
            [chunk("A latte is "), chunk("$4.50."), chunk(finish="stop")],
        )
        shop, task = RecordingTools(), IMTask()
        session = source(settings=FAST, tools=[shop]).new()
        session.source.make_provider = lambda endpoint: ChatCompletionsBackend(endpoint, FAST.limits, chat)
        events = await drive(session.start(task), task)
        self.assertEqual(check_events(events), [])
        [call] = items_of(events, "function_call")
        self.assertNotEqual(call["call_id"], "p_1")  # our own id
        self.assertEqual(task.outcomes[call["call_id"]][1], "done")
        self.assertEqual(
            [(m["content"][0]["text"], m["phase"]) for m in items_of(events, "message")],
            [("Let me check.", "commentary"), ("A latte is $4.50.", "final_answer")],
        )
        second = chat.requests[1]["messages"]
        self.assertEqual([message["role"] for message in second], ["system", "user", "assistant", "tool"])
        self.assertEqual(second[-1]["tool_call_id"], call["call_id"])


if __name__ == "__main__":
    unittest.main()
