# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Tests that every shipped provider and the recipe's example provider pass ``ProviderContract``."""

import unittest

from delegation_core.backend.providers import Backend
from delegation_core.backend.providers.chat_completions import ChatCompletionsBackend
from delegation_core.backend.providers.openai_responses import OpenAIResponsesBackend
from tests.unit.contracts.backend import ProviderContract, a_round, call_events, message_events
from tests.unit.delegation_core.backend.llm.doubles import (
    ENDPOINT,
    LIMITS,
    FakeChat,
    FakeResponses,
    chunk,
    tool_fragment,
)


class EchoBackend(Backend):
    """The recipe in ``providers/__init__.py``, followed step by step: a provider that answers by its own rule."""

    name = "echo"

    def __init__(self, endpoint, limits, *, calls=False):
        super().__init__(endpoint, limits)
        self.calls = calls

    async def _events(self, payload):
        if self.calls:
            part = call_events(payload["tools"][0]["name"], {}, call_id="call_echo")
        else:
            part = message_events("Hello.", item_id="msg_echo")
        for event in a_round(part, response_id="resp_echo"):
            yield event


class ChatCompletionsProviderTests(ProviderContract, unittest.IsolatedAsyncioTestCase):
    def make_provider(self, script):
        rounds = {
            "text": [chunk("Hel"), chunk("lo."), chunk(finish="stop")],
            "call": [chunk(tool_calls=[tool_fragment(0, "c1", "get_menu", "{}")]), chunk(finish="tool_calls")],
        }
        return ChatCompletionsBackend(ENDPOINT, LIMITS, FakeChat(rounds[script]))


class OpenAIResponsesProviderTests(ProviderContract, unittest.IsolatedAsyncioTestCase):
    def make_provider(self, script):
        rounds = {"text": a_round(message_events("Hello.")), "call": a_round(call_events("get_menu", {}))}
        return OpenAIResponsesBackend(ENDPOINT, LIMITS, FakeResponses(rounds[script]))


class RecipeProviderTests(ProviderContract, unittest.IsolatedAsyncioTestCase):
    def make_provider(self, script):
        return EchoBackend(ENDPOINT, LIMITS, calls=script == "call")


if __name__ == "__main__":
    unittest.main()
