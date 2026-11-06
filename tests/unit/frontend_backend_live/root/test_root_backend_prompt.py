# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""What the backend reads when a request sets no instructions: the Live core's generic prompt, the UI bot's cafe one."""

import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from examples.frontend_backend_live import factory, pipeline
from examples.frontend_backend_live.delegation_core.app import build_core
from examples.frontend_backend_live.delegation_core.backend.providers import base as providers
from examples.frontend_backend_live.delegation_core.im.input import compose
from examples.frontend_backend_live.prompts import load_prompts
from tests.unit.core_helpers import FakeChat, chunk
from tests.unit.frontend_backend_live.root.root_helpers import CATALOG, StubAdapter, clear_root_caches

USER = {"type": "message", "role": "user", "content": "What can you do for me?"}
# The example's prompts.yaml as written.
AGENT_PROMPTS = yaml.safe_load(Path(factory.__file__).with_name("prompts.yaml").read_text(encoding="utf-8"))
AGENT_PROMPTS = AGENT_PROMPTS["agent_prompts"]


def entry(group: str, name: str) -> str:
    """The text of ``agent_prompts.<group>.<name>`` as the file holds it, stripped as a backend prompt is used."""
    return AGENT_PROMPTS[group][name]["content"].strip()


def core_text(name: str) -> str:
    """The core's text ``agent_prompts.Core.<name>`` as the file holds it, used as written."""
    return AGENT_PROMPTS["Core"][name]["content"]


def ui_bot_core():
    """The UI bot's delegation core, as ``pipeline.py`` builds it."""
    state = pipeline.bot_root()
    prompts = load_prompts()
    instructions = factory.ui_bot_instructions(prompts, prompts.text("Channel", state.channel))
    endpoints = factory.endpoint_lookup(state.catalog, state.core.backend.slot)
    return build_core(
        state.core, endpoints=endpoints, defaults=factory.backend_defaults(state.core, endpoints, instructions)
    )


class DefaultPromptTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        clear_root_caches(self)
        pipeline.bot_root.cache_clear()
        self.addCleanup(pipeline.bot_root.cache_clear)
        self.enterContext(patch.object(factory.ExampleCatalog, "read", return_value=CATALOG))
        for root in (factory, pipeline):
            self.enterContext(patch.object(root, "frontend_adapters", return_value={"cascade": StubAdapter}))

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
        channel = entry("Channel", "cafe")
        # Then the core's notes, which the roots hand the core from the prompt file.
        notes = (core_text("texts.caller_notes.default"), core_text("texts.voice_notes.default"))
        live = await self.system_message(factory.delegation_core)
        self.assertEqual(live, compose(entry("Backend", "generic"), *notes))
        self.assertNotIn(channel, live)
        self.assertNotIn("Bluebird", live)  # nothing of the cafe for a client that brings its own tools
        ui_bot = await self.system_message(ui_bot_core)
        cafe = f"{entry('Backend', 'cafe')}\n\n{channel}"  # the cafe's prompt, then its channel
        self.assertEqual(ui_bot, compose(cafe, *notes, core_text("texts.result_formats.speak_display")))


if __name__ == "__main__":
    unittest.main()
