# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Rules 14 and 16: backend-only mode, and isolation between concurrent sessions."""

from __future__ import annotations

import asyncio
import unittest

from _fbv_voice_fakes import (
    PACKAGE_DIR,
    FakeChatClient,
    SessionHarness,
    delegate_response,
    pcmu_silence,
    pcmu_speech,
    tau2_session_update,
    tau2_update_with,
    text_response,
    tool_response,
)

from examples.frontend_backend_verdict.voice.agent.runner import AgentClients
from examples.frontend_backend_verdict.voice.config import load_voice_config

BACKEND_ONLY = load_voice_config(PACKAGE_DIR / "config" / "profiles" / "backend_only.yaml")


class BackendOnlyTests(unittest.IsolatedAsyncioTestCase):
    async def test_stateful_backend_full_contract_without_a_frontend(self) -> None:
        backend = FakeChatClient(
            [tool_response(("get_users", {}), ids=["call_1"]), text_response("You are Ann."), text_response("Bye now.")]
        )
        harness = SessionHarness(config=BACKEND_ONLY, clients=AgentClients(backend=backend, frontend=None))
        await harness.start(tau2_session_update())
        await harness.speak()
        await harness.wait_for("response.done")
        await harness.send(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": "call_1", "output": "{}"},
            }
        )
        await harness.send({"type": "response.create"})
        await harness.wait_for("response.done", 2)
        await harness.feed(pcmu_silence(2000))
        await harness.speak()
        await harness.wait_for("response.done", 3)
        runner = harness.runners[0]
        self.assertTrue(runner.backend_only)
        self.assertEqual(len(runner.state.frontend_history), 0)
        third_call = backend.calls[2]["messages"]
        self.assertIn("utterance 1", [m.content for m in third_call if m.role == "user"])  # stateful
        self.assertEqual(third_call[1].content, "Hi! How can I help you today?")  # seeded client greeting
        self.assertEqual(harness.filler_log.records, [])  # no frontend, no filler
        self.assertEqual(harness.of_type("error"), [])
        await harness.close()

    async def test_barge_in_repairs_the_backend_history(self) -> None:
        answer = "First part is here. Second part is much longer and nobody will hear it."
        backend = FakeChatClient([text_response(answer)])
        harness = SessionHarness(config=BACKEND_ONLY, clients=AgentClients(backend=backend))
        await harness.start(tau2_session_update())
        await harness.speak()
        await harness.wait_for("response.done")
        await harness.feed(pcmu_silence(220))
        await harness.feed(pcmu_speech(300))
        stored = harness.runners[0].state.backend_history.messages[-1].content
        self.assertTrue(stored.startswith("First part is here."))
        self.assertTrue(stored.endswith("[interrupted by the user]"))
        self.assertNotIn("nobody will hear it", stored)
        await harness.close()


class IsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_sessions_never_cross(self) -> None:
        tools_a = [
            {"type": "function", "name": "alpha_tool", "description": "Alpha.", "parameters": {"type": "object"}}
        ]
        tools_b = [{"type": "function", "name": "beta_tool", "description": "Beta.", "parameters": {"type": "object"}}]
        sessions = []
        for tools, policy, answer in (
            (tools_a, "Policy A only.", "Answer A."),
            (tools_b, "Policy B only.", "Answer B."),
        ):
            frontend = FakeChatClient([delegate_response(f"Request for {answer}")])
            backend = FakeChatClient([text_response(answer)])
            harness = SessionHarness(clients=AgentClients(backend=backend, frontend=frontend))
            await harness.start(tau2_update_with(tools, policy))
            sessions.append((harness, frontend, backend, tools, policy, answer))
        await asyncio.gather(*(harness.speak() for harness, *_ in sessions))
        for harness, frontend, backend, tools, policy, answer in sessions:
            await harness.wait_for("response.done")
            self.assertEqual([t["function"]["name"] for t in backend.calls[0]["tools"]], [tools[0]["name"]])
            system = backend.calls[0]["messages"][0].content
            self.assertIn(policy, system)
            other = "Policy B only." if policy == "Policy A only." else "Policy A only."
            self.assertNotIn(other, system)
            self.assertIn(tools[0]["name"], frontend.calls[0]["messages"][0].content)
            transcript = "".join(e["delta"] for e in harness.of_type("response.output_audio_transcript.delta"))
            self.assertEqual(transcript, answer)
            self.assertNotEqual(sessions[0][0].session.session_id, sessions[1][0].session.session_id)
        for harness, *_ in sessions:
            await harness.close()
