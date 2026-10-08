# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D107

"""The delegation coordinator on its own: no frontend, no Pipecat."""

import json
import unittest

from examples.frontend_backend_live.config_manager.schema import LiveConfig, RoleConfig
from examples.frontend_backend_live.delegation.coordinator import DelegationCoordinator
from examples.frontend_backend_live.delegation.tasks import CLIENT

CONFIG = LiveConfig(frontend=RoleConfig("p", "llm"), backend=RoleConfig("p", "thinker-llm"))


class RecordingSink:
    def __init__(self):
        self.results = []

    async def on_result(self, text, task, told, *, superseded):
        self.results.append((text, told, superseded))


class ClientModeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.announced: list[str] = []

        async def announce(delegation_id):
            self.announced.append(delegation_id)

        self.sink = RecordingSink()
        self.coordinator = DelegationCoordinator(
            None, CONFIG, backend_task=None, tools=None, sink=self.sink, mode=CLIENT, on_client_delegation=announce
        )

    async def test_a_delegation_is_announced_and_remembered(self):
        delegation_id = await self.coordinator.delegate("What teas do you have?")
        self.assertEqual(self.announced, [delegation_id])
        self.assertIn(delegation_id, self.coordinator.client_delegations)
        self.assertFalse(self.coordinator.backend_busy)

    async def test_the_delegate_sees_only_the_voice_turns_since_its_last_delegation(self):
        self.coordinator.remember_voice("assistant", "Welcome.")
        await self.coordinator.delegate("What teas do you have?")
        self.assertEqual(self.coordinator.pending_voice, [])
        self.coordinator.remember_voice("assistant", "Let me check.")
        payload = self.coordinator._delegate_input("Add one.")
        text = payload[0]["content"][0]["text"]
        context = json.loads(text.split("\n")[1])
        self.assertEqual(context, [{"role": "assistant", "content": "Let me check."}])
        self.assertTrue(text.endswith("Current request: Add one."))

    async def test_an_instruction_is_not_recorded_as_caller_speech(self):
        await self.coordinator.delegate("Be formal.", instruction=True)
        self.assertEqual(self.coordinator.pending_voice, [])
