# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""What the backend reads when a request sets no instructions: the Live core's generic prompt, the UI bot's cafe one."""

import asyncio
import os
import unittest
from unittest.mock import patch

from delegation_core.app import build_core
from delegation_core.backend.providers import base as providers
from examples.frontend_backend_live import factory, pipeline
from examples.frontend_backend_live.prompts.catalog import CHANNEL_PROMPT_KEY, catalog_prompt, role_instructions
from tests.unit.core_helpers import FakeChat, chunk
from tests.unit.frontend_backend_live.root.root_helpers import CATALOG, StubAdapter, clear_root_caches

USER = {"type": "message", "role": "user", "content": "What can you do for me?"}


def ui_bot_core():
    """The UI bot's delegation core, as its root (``pipeline.py``) builds it."""
    state = pipeline.bot_root()
    endpoints = factory.endpoint_lookup(state.catalog, state.core.backend.slot)
    return build_core(state.core, endpoints=endpoints, defaults=factory.backend_defaults(state.core, endpoints))


class DefaultPromptTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        clear_root_caches(self)
        pipeline.bot_root.cache_clear()
        self.addCleanup(pipeline.bot_root.cache_clear)
        self.enterContext(patch.object(factory.ExampleCatalog, "read", return_value=CATALOG))
        for root in (factory, pipeline):
            self.enterContext(patch.object(root, "frontend_adapters", return_value={"cascade": StubAdapter}))
        self.enterContext(patch.dict(os.environ))
        for name in ("CORE_CONFIG_PATH", "LIVE_CONFIG_PATH"):
            os.environ.pop(name, None)

    async def system_message(self, build) -> str:
        """The instructions the backend model gets for one request that sets none, on the core that ``build`` makes."""
        chat = FakeChat([chunk("Done.", finish="stop")])
        with patch.object(providers, "AsyncOpenAI", return_value=chat):
            core = build()
            self.addAsyncCleanup(core.aclose)
            client = factory.core_client(core)
            conversation = (await client.create_conversation({}))["id"]
            self.addAsyncCleanup(client.delete_conversation, conversation)
            params = {"conversation": conversation, "input": [USER], "metadata": {"live_delegation_id": "d1"}}
            stream = await client.create_response({"stream": True, **params})
            async with asyncio.timeout(5):
                events = [event async for event in stream]
        self.assertEqual(events[-1]["type"], "response.completed")
        (request,) = chat.requests
        self.assertEqual(request["messages"][0]["role"], "system")
        return request["messages"][0]["content"]

    async def test_a_request_with_no_instructions_gets_the_generic_prompt_and_the_ui_bot_the_cafe_prompt(self):
        live = await self.system_message(factory.delegation_core)
        self.assertTrue(live.startswith(catalog_prompt("backend")), live)
        self.assertNotIn(catalog_prompt(CHANNEL_PROMPT_KEY), live)
        self.assertNotIn("Bluebird", live)  # nothing of the cafe for a client that brings its own tools
        ui_bot = await self.system_message(ui_bot_core)
        self.assertTrue(ui_bot.startswith(role_instructions("thinker")), ui_bot)  # the cafe's prompt, then its channel


if __name__ == "__main__":
    unittest.main()
