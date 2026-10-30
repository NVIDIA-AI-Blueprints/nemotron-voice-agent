# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D107

"""The repository UI bot's composition, as its root (``pipeline.py``) builds it, with no Live gateway."""

import asyncio
import json
import unittest
from unittest.mock import patch

from delegation_core.config import CoreSettings
from examples.frontend_backend_live import pipeline
from examples.frontend_backend_live.factory import backend_defaults, core_client, endpoint_lookup
from examples.frontend_backend_live.models.endpoints import ExampleCatalog
from tests.unit.contracts.base import ContractCase
from tests.unit.frontend_backend_live.root.root_helpers import StubAdapter

ENTRY = {"model_id": "test/model", "base_url": "http://localhost:1/v1"}


class BotRootTests(unittest.TestCase):
    def test_the_sessions_backend_choice_reaches_the_cores_lookup(self):
        body = {"thinker_model_id": "org/picked", "thinker_base_url": "http://picked/v1", "model_id": "org/front"}
        self.assertEqual(pipeline.backend_overrides(body), {"model_id": "org/picked", "base_url": "http://picked/v1"})
        catalog = ExampleCatalog({"thinker-llm": [{"model_id": "org/default", "base_url": "http://default/v1"}]})
        endpoints = endpoint_lookup(catalog, "thinker-llm", pipeline.backend_overrides(body))
        self.assertEqual((endpoints(None).model, endpoints(None).base_url), ("org/picked", "http://picked/v1"))
        self.assertEqual(endpoints("org/default").base_url, "http://default/v1")  # a listed model keeps its entry
        with self.assertRaisesRegex(LookupError, "'org/unlisted' is not in the catalog"):
            endpoints("org/unlisted")


class CafeOrderTests(ContractCase, unittest.IsolatedAsyncioTestCase):
    async def test_a_cafe_order_flows_through_the_bots_core_and_the_real_tools(self):
        from delegation_core.app import build_core
        from delegation_core.backend.providers import base as providers
        from examples.frontend_backend_live.cascade.agent import build_agent
        from examples.frontend_backend_live.cascade.session import CascadeSession
        from live.server_delegation import ServerDelegation
        from tests.unit.cascade_helpers import CASCADE_ENDPOINT, ListConversation, _decision_chunks, _messages
        from tests.unit.core_helpers import FakeChat, chunk, tool_fragment

        # One fake endpoint serves every model call, in order: routing, tool round, answer round, phrasing.
        client = FakeChat(
            _decision_chunks('{"action":"delegate","speech":"Let me add that."}'),
            [
                chunk(
                    tool_calls=[
                        tool_fragment(0, "call_1", "manage_cart", '{"action":"add","item_id":"lat","size":"large"}')
                    ]
                ),
                chunk(finish="tool_calls"),
            ],
            [chunk("Added one large Latte. Your total is $5.97 with tax.", finish="stop")],
            _decision_chunks(
                '{"action":"speak","speech":"One large latte is in your cart, five ninety-seven with tax."}'
            ),
        )
        with (
            patch.object(pipeline.ExampleCatalog, "read", return_value=ExampleCatalog({"thinker-llm": [ENTRY]})),
            patch.object(pipeline, "frontend_adapters", return_value={"cascade": StubAdapter}),
        ):
            pipeline.bot_root.cache_clear()
            state = pipeline.bot_root()
        self.addCleanup(pipeline.bot_root.cache_clear)
        settings: CoreSettings = state.core
        endpoints = endpoint_lookup(state.catalog, settings.backend.slot)
        spoken: list[str] = []

        async def say(text):
            spoken.append(text)

        with patch.object(providers, "AsyncOpenAI", return_value=client):  # every provider makes its client there
            core = build_core(settings, endpoints=endpoints, defaults=backend_defaults(settings, endpoints))
            delegation = ServerDelegation(core_client(core))
            agent = build_agent(state.cascade, CASCADE_ENDPOINT, frontend_client=client)
            conversation = ListConversation(_messages())
            session = CascadeSession(agent, state.cascade, conversation, delegation=delegation, say=say)
            conversation.add({"role": "user", "content": "A large latte, please."})  # the caller's turn arrives
            await delegation.open(None)
            session.start()
            try:
                await session.on_turn()
                async with asyncio.timeout(2):
                    while len(spoken) < 2:
                        await asyncio.sleep(0.005)
            finally:
                await session.close()
                await delegation.close()
                await core.aclose()
        self.assertEqual(spoken[0], "Let me add that.")
        tool_result = next(m for m in client.requests[2]["messages"] if m["role"] == "tool")
        self.assertEqual(json.loads(tool_result["content"])["status"], "added")


if __name__ == "__main__":
    unittest.main()
