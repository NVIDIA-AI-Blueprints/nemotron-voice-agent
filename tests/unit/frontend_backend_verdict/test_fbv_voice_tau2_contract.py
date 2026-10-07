# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Rules 1, 6, 7, 8, 15, 21: the tau2 client contract, driven through a real session."""

from __future__ import annotations

import base64
import json
import unittest

from _fbv_voice_fakes import (
    FakeChatClient,
    SessionHarness,
    delegate_response,
    pcmu_silence,
    tau2_session_update,
    text_response,
    tool_response,
    voice_config,
)

from examples.frontend_backend_verdict.voice.agent.port import AgentReply, OutgoingCall
from examples.frontend_backend_verdict.voice.agent.runner import AgentClients
from examples.frontend_backend_verdict.voice.agent.scripted import ScriptedAgentPort


def output_event(call_id: str, output: str) -> dict:
    return {
        "type": "conversation.item.create",
        "item": {"type": "function_call_output", "call_id": call_id, "output": output},
    }


class HandshakeTests(unittest.IsolatedAsyncioTestCase):
    async def test_handshake_with_tau2_payload(self) -> None:
        for domain in ("mock", "airline", "retail", "telecom"):
            with self.subTest(domain=domain):
                harness = SessionHarness(clients=AgentClients(backend=FakeChatClient(), frontend=FakeChatClient()))
                await harness.start(tau2_session_update(domain))
                self.assertEqual(harness.events[0]["type"], "session.created")
                self.assertTrue(harness.events[0]["session"]["id"].startswith("sess_"))
                self.assertEqual(harness.events[1]["type"], "session.updated")
                self.assertEqual(harness.of_type("error"), [])
                await harness.close()

    async def test_no_greeting(self) -> None:
        harness = SessionHarness(agent_factory=lambda _: ScriptedAgentPort([{"say": "Hello there."}]))
        await harness.start(tau2_session_update())
        await harness.idle(1500)
        self.assertEqual(harness.of_type("response.created"), [])
        await harness.speak()
        await harness.wait_for("response.done")
        first_audio = harness.types().index("response.output_audio.delta")
        self.assertGreater(first_audio, harness.types().index("input_audio_buffer.speech_stopped"))
        await harness.close()


class FunctionCallTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.frontend = FakeChatClient()
        self.backend = FakeChatClient()
        self.harness = SessionHarness(clients=AgentClients(backend=self.backend, frontend=self.frontend))
        await self.harness.start(tau2_session_update("mock"))

    async def asyncTearDown(self) -> None:
        await self.harness.close()

    async def test_round_trip_preserves_call_id_and_passes_output_verbatim(self) -> None:
        self.frontend.queue(delegate_response("Look up user 7."))
        self.backend.queue(
            tool_response(("get_users", {}), ids=["call_provider_42"]), text_response("User seven is Ann.")
        )
        await self.harness.speak()
        done = await self.harness.wait_for("response.function_call_arguments.done")
        self.assertEqual((done["call_id"], done["name"], done["arguments"]), ("call_provider_42", "get_users", "{}"))
        await self.harness.wait_for("response.done")
        await self.harness.send(output_event("call_provider_42", "Error: user not found"))
        await self.harness.send({"type": "response.create"})
        await self.harness.wait_for("response.done", 2)
        tool_messages = [m for m in self.backend.calls[1]["messages"] if m.role == "tool"]
        self.assertEqual(tool_messages[0].content, "Error: user not found")
        transcript = "".join(e["delta"] for e in self.harness.of_type("response.output_audio_transcript.delta"))
        self.assertEqual(transcript, "User seven is Ann.")
        acks = [
            e for e in self.harness.of_type("conversation.item.added") if e["item"]["type"] == "function_call_output"
        ]
        self.assertEqual(len(acks), 1)
        group = self.harness.runners[0].state.frontend_history.messages[-4:]
        self.assertEqual([m.role for m in group], ["user", "assistant", "tool", "assistant"])

    async def test_parallel_calls_resume_only_after_all_outputs(self) -> None:
        self.frontend.queue(delegate_response("Do two things."))
        self.backend.queue(
            tool_response(("get_users", {}), ("create_task", {"title": "x"}), ids=["call_a", "call_b"]),
            text_response("Both done."),
        )
        await self.harness.speak()
        await self.harness.wait_for("response.done")
        calls = self.harness.of_type("response.function_call_arguments.done")
        self.assertEqual([c["call_id"] for c in calls], ["call_a", "call_b"])
        response_items = self.harness.of_type("response.done")[0]["response"]["output"]
        self.assertEqual([item["type"] for item in response_items], ["function_call", "function_call"])
        await self.harness.send(output_event("call_b", "B"))
        await self.harness.send({"type": "response.create"})
        await self.harness.settle()
        self.assertEqual(len(self.backend.calls), 1, "must not resume with an output missing")
        await self.harness.send(output_event("call_a", "A"))
        await self.harness.wait_for("response.done", 2)
        tool_messages = [m for m in self.backend.calls[1]["messages"] if m.role == "tool"]
        self.assertEqual([(m.tool_call_id, m.content) for m in tool_messages], [("call_a", "A"), ("call_b", "B")])

    async def test_bad_and_missing_outputs(self) -> None:
        self.frontend.queue(delegate_response("Look someone up."))
        self.backend.queue(tool_response(("get_users", {}), ids=["call_a"]), text_response("Sorry, that failed."))
        await self.harness.speak()
        await self.harness.wait_for("response.done")
        await self.harness.send(output_event("call_unknown", "x"))
        error = await self.harness.wait_for("error")
        self.assertEqual(error["error"]["param"], "item.call_id")
        # The session survives; a stalled client is unblocked by the result timeout.
        self.harness.session.turns._wait.timer.cancel()
        self.harness.session.turns._on_result_timeout(self.harness.session.turns._wait)
        await self.harness.wait_for("response.done", 2)
        tool_message = next(m for m in self.backend.calls[1]["messages"] if m.role == "tool")
        self.assertIn("not provided by the caller", tool_message.content)


class _CallsThenAnswer(ScriptedAgentPort):
    async def respond(self, text: str) -> AgentReply:
        self.inputs.append(text)
        return AgentReply(calls=(OutgoingCall(call_id="call_early", name="get_users", arguments="{}"),))

    async def resume(self, outputs: dict) -> AgentReply:
        self.outputs.append(dict(outputs))
        return AgentReply(text="Done now.")


class Tau2TimingTests(unittest.IsolatedAsyncioTestCase):
    async def test_early_response_create_is_queued_not_rejected(self) -> None:
        agent = _CallsThenAnswer([])
        harness = SessionHarness(agent_factory=lambda _: agent)
        await harness.start(tau2_session_update())
        turns = harness.session.turns
        original = turns._deliver

        def deliver_then_answer_immediately(turn, reply):
            original(turn, reply)
            # tau2 answers on its next tick, which can overtake our response.done.
            turns.on_function_output("call_early", "ok")
            turns.on_response_create()

        turns._deliver = deliver_then_answer_immediately
        await harness.speak()
        await harness.wait_for("response.done", 2)
        self.assertEqual(harness.of_type("error"), [])
        types = harness.types()
        first_done = types.index("response.done")
        second_created = [i for i, t in enumerate(types) if t == "response.created"][1]
        self.assertLess(first_done, second_created)
        self.assertEqual(agent.outputs, [{"call_early": "ok"}])
        await harness.close()

    async def test_truncate_is_clamped_not_rejected(self) -> None:
        harness = SessionHarness(agent_factory=lambda _: ScriptedAgentPort([{"say": "A short answer."}]))
        await harness.start(tau2_session_update())
        await harness.speak()
        await harness.wait_for("response.done")
        item_id = harness.of_type("response.output_audio.delta")[0]["item_id"]
        await harness.send(
            {"type": "conversation.item.truncate", "item_id": item_id, "content_index": 0, "audio_end_ms": 999999}
        )
        truncated = await harness.wait_for("conversation.item.truncated")
        span = harness.session.turns.progress.item(item_id)
        self.assertEqual(truncated["audio_end_ms"], int(span.end_ms - span.start_ms))
        self.assertEqual(harness.of_type("error"), [])
        await harness.close()

    async def test_pcmu_out_and_fresh_item_ids(self) -> None:
        harness = SessionHarness(agent_factory=lambda _: ScriptedAgentPort([{"say": "One. Two."}, {"say": "Three."}]))
        await harness.start(tau2_session_update())
        await harness.speak()
        await harness.wait_for("response.done")
        await harness.feed(pcmu_silence(3000))
        await harness.speak()
        await harness.wait_for("response.done", 2)
        items = {e["item_id"] for e in harness.of_type("response.output_audio.delta")}
        self.assertEqual(len(items), 2)
        audio = base64.b64decode(harness.of_type("response.output_audio.delta")[0]["delta"])
        self.assertEqual(len(audio), 800)  # 100 ms of 8 kHz mu-law per delta
        await harness.close()

    async def test_last_function_output_resume_mode(self) -> None:
        config = voice_config(protocol_resume_on="last_function_output")
        agent = _CallsThenAnswer([])
        harness = SessionHarness(config=config, agent_factory=lambda _: agent)
        await harness.start(tau2_session_update())
        await harness.speak()
        await harness.wait_for("response.done")
        await harness.send(output_event("call_early", json.dumps({"ok": True})))
        await harness.wait_for("response.done", 2)
        self.assertEqual(agent.outputs, [{"call_early": '{"ok": true}'}])
        await harness.close()
