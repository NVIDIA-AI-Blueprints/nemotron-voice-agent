# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Tests for atomic Generic Realtime Talker and Thinker prompt updates."""

# Test names describe the contract; separate public API docstrings add no value here.
# ruff: noqa: D101, D102

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from pipecat.processors.aggregators.llm_context import LLMContext

from examples.frontend_backend_agent.generic.realtime_prompts import GenericRealtimePromptCoordinator
from examples.frontend_backend_agent.src.tools import ToolSpec
from realtime.capabilities import reset_capability_cache_for_tests


async def _noop(_arguments, _context):
    return {"status": "success"}


def _client_tool(name: str = "find_order") -> dict:
    return {
        "type": "function",
        "name": name,
        "description": "Find an order by email.",
        "parameters": {
            "type": "object",
            "properties": {"email": {"type": "string"}},
            "required": ["email"],
            "additionalProperties": False,
        },
    }


class _Backend:
    def __init__(self) -> None:
        self.session_instruction_context = LLMContext([{"role": "system", "content": "Original policy."}])
        self.prepared: list[tuple[str, tuple[dict, ...]]] = []
        self.committed: list[object] = []

    @staticmethod
    def render_session_instructions(instructions: str) -> list[dict]:
        return [{"role": "system", "content": instructions}]

    def prepare_session_update(self, *, instructions, client_tools):
        receipt = SimpleNamespace(instructions=instructions, client_tools=tuple(client_tools))
        self.prepared.append((instructions, tuple(client_tools)))
        return receipt

    def snapshot_session_update(self):
        return list(self.session_instruction_context.get_messages())

    def commit_session_update(self, prepared) -> None:
        self.committed.append(prepared)
        self.session_instruction_context.set_messages(self.render_session_instructions(prepared.instructions))

    def restore_session_update(self, snapshot) -> None:
        self.session_instruction_context.set_messages(snapshot)


class GenericRealtimePromptCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        reset_capability_cache_for_tests()

    async def test_instruction_and_tool_updates_regenerate_with_talker_then_commit_together(self) -> None:
        backend = _Backend()
        generation_payloads: list[dict] = []

        async def generate(context, **_kwargs):
            payload = json.loads(context.messages[0]["content"])
            generation_payloads.append(payload)
            return {
                "domain": "Order support",
                "scope_in": ["customer order lookup"],
                "scope_out": [],
                "capabilities": [
                    {
                        "name": tool["name"],
                        "summary": "Checks current conditions."
                        if tool["name"] == "get_weather"
                        else "Retrieves a customer order.",
                    }
                    for tool in payload["tools"]
                ],
            }

        inference = AsyncMock(side_effect=generate)
        talker = SimpleNamespace(run_structured_inference=inference)
        weather = ToolSpec(name="get_weather", contract="Fetch current conditions.", params={}, run=_noop)
        coordinator = GenericRealtimePromptCoordinator(
            backend=backend,
            talker_llm=talker,
            server_specs=(weather,),
            trusted_tool_names=("call_backend", "cancel_backend", "get_weather"),
            static_talker_prompt="Trusted Talker policy.",
            initial_capability_digest="Initial model digest.",
            initial_instructions="Original policy.",
            initial_client_tools=(),
            capability_mode="model",
            render_talker_messages=lambda prompt: [{"role": "system", "content": prompt}],
            profile="generic-frontend-backend-agent",
        )

        prepared = await coordinator.prepare_session_update(
            instructions="Handle retail returns.",
            tools=(_client_tool(),),
            tool_choice="auto",
        )

        self.assertEqual(backend.session_instruction_context.messages[0]["content"], "Original policy.")
        self.assertEqual([tool["name"] for tool in generation_payloads[0]["tools"]], ["get_weather", "find_order"])
        self.assertIn("Retrieves a customer order.", prepared.talker_prompt)
        self.assertNotIn("find_order", prepared.talker_prompt)
        self.assertNotIn("Find an order by email.", prepared.talker_prompt)
        self.assertEqual(inference.await_args.kwargs["max_tokens"], 350)

        coordinator.commit_session_update(prepared)

        self.assertEqual(
            backend.session_instruction_context.messages,
            [{"role": "system", "content": "Handle retail returns."}],
        )
        self.assertEqual(backend.prepared[-1][1][0]["name"], "find_order")
        self.assertEqual(coordinator.current_talker_prompt_messages, list(prepared.talker_prompt_messages))

        await coordinator.prepare_session_update(
            instructions="Handle retail exchanges.",
            tools=(_client_tool(),),
            tool_choice="auto",
        )
        self.assertEqual(inference.await_count, 2)
        self.assertEqual(generation_payloads[-1]["instructions"], "Handle retail exchanges.")

    async def test_static_mode_skips_talker_inference_and_refreshes_tool_digest(self) -> None:
        backend = _Backend()
        inference = AsyncMock()
        coordinator = GenericRealtimePromptCoordinator(
            backend=backend,
            talker_llm=SimpleNamespace(run_structured_inference=inference),
            server_specs=(),
            trusted_tool_names=("call_backend", "cancel_backend"),
            static_talker_prompt="Trusted Talker policy.",
            initial_capability_digest="Initial static digest.",
            initial_instructions="Original policy.",
            initial_client_tools=(),
            capability_mode="static",
            render_talker_messages=lambda prompt: [{"role": "system", "content": prompt}],
            profile="generic-frontend-backend-agent",
        )

        prepared = await coordinator.prepare_session_update(
            instructions="New policy.",
            tools=(_client_tool(),),
            tool_choice="auto",
        )

        inference.assert_not_awaited()
        self.assertIn("- Find an order by email.", prepared.talker_prompt)
        self.assertNotIn("find_order", prepared.talker_prompt)
        coordinator.commit_session_update(prepared)
        self.assertEqual(
            backend.session_instruction_context.messages,
            [{"role": "system", "content": "New policy."}],
        )

    async def test_model_generation_failure_degrades_to_static_and_still_commits_atomically(self) -> None:
        backend = _Backend()
        talker = SimpleNamespace(run_structured_inference=AsyncMock(side_effect=TimeoutError))
        coordinator = GenericRealtimePromptCoordinator(
            backend=backend,
            talker_llm=talker,
            server_specs=(),
            trusted_tool_names=("call_backend", "cancel_backend"),
            static_talker_prompt="Trusted Talker policy.",
            initial_capability_digest="Initial model digest.",
            initial_instructions="Original policy.",
            initial_client_tools=(),
            capability_mode="model",
            render_talker_messages=lambda prompt: [{"role": "system", "content": prompt}],
            profile="generic-frontend-backend-agent",
        )
        prepared = await coordinator.prepare_session_update(
            instructions="New policy.",
            tools=(_client_tool(),),
            tool_choice="auto",
        )

        self.assertEqual(backend.committed, [])
        self.assertEqual(backend.session_instruction_context.messages[0]["content"], "Original policy.")
        self.assertIn("- Find an order by email.", prepared.talker_prompt)
        self.assertNotIn("find_order", prepared.talker_prompt)

        coordinator.commit_session_update(prepared)

        self.assertEqual(
            backend.session_instruction_context.messages,
            [{"role": "system", "content": "New policy."}],
        )
        self.assertEqual(coordinator.current_talker_prompt_messages, list(prepared.talker_prompt_messages))


if __name__ == "__main__":
    unittest.main()
