# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""R1: resume after tool results when the caller's turn commits.

With automatic responses on, a caller turn committed while tool calls are out goes
into that wait's inbox. The wait resumes once every output is in, whichever came
last, and the backend's next request has the tool messages followed by one user
message with the caller's words. A later ``response.create`` that can only be the
client's request for that wait is consumed once; every other one follows the
existing rules.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

from _fbv_voice_fakes import (
    FakeChatClient,
    SessionHarness,
    delegate_response,
    normalize_trace,
    pcmu_silence,
    pcmu_speech,
    tau2_session_update,
    text_response,
    tool_response,
    voice_config,
)

from examples.frontend_backend_verdict.text.llm import to_wire_messages
from examples.frontend_backend_verdict.text.messages import CONTINUES_TURN
from examples.frontend_backend_verdict.voice.agent.runner import AgentClients
from examples.frontend_backend_verdict.voice.agent.sinks import EventLog
from examples.frontend_backend_verdict.voice.speech.stubs import StubRecognizer

#: SHA-256 of the normalized wire trace of a tool round trip with no caller speech during the
#: wait, produced by the turn manager before R1 (commit 4271cac). R1 must not change it.
NO_OVERLAP_TRACE_SHA256 = "3411ef9a037ca96ef1c9e42a942698e59b60b75146bf40e1b702d5fa4f9072d2"


async def until(condition: Callable[[], bool], timeout: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.005)


def output(call_id: str, value: str = "ok") -> dict[str, Any]:
    return {
        "type": "conversation.item.create",
        "item": {"type": "function_call_output", "call_id": call_id, "output": value},
    }


class ToolResumeTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.log_path = Path(self._tmp.name) / "events.jsonl"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def make(self, **harness: Any) -> tuple[SessionHarness, FakeChatClient, FakeChatClient]:
        frontend, backend = FakeChatClient(), FakeChatClient()
        session = SessionHarness(
            clients=AgentClients(backend=backend, frontend=frontend),
            event_log=EventLog(str(self.log_path)),
            **harness,
        )
        return session, frontend, backend

    def records(self, kind: str) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        return [record for record in map(json.loads, lines) if record["kind"] == kind]

    @staticmethod
    def spoken(harness: SessionHarness) -> str:
        return "".join(e["delta"] for e in harness.of_type("response.output_audio_transcript.delta"))

    async def tool_wait(self, harness: SessionHarness, *, update: dict[str, Any] | None = None) -> None:
        """Utterance 1 delegates and the backend's tool call goes out; the turn now waits on tools."""
        await harness.start(update or tau2_session_update())
        await harness.speak()
        await harness.wait_for("response.done")
        self.assertEqual(harness.session.turns.state, "AWAITING_TOOLS")

    def resumed_messages(self, backend: FakeChatClient, index: int = 1) -> list[Any]:
        return backend.calls[index]["messages"]


class ResumeTriggerTests(ToolResumeTestCase):
    async def test_outputs_complete_then_a_turn_commits(self) -> None:
        harness, frontend, backend = self.make()
        frontend.queue(delegate_response("Look up the user."))
        backend.queue(tool_response(("get_users", {}), ids=["call_a"]), text_response("Found them."))
        await self.tool_wait(harness)
        await harness.send(output("call_a"))
        self.assertEqual(harness.session.turns.state, "AWAITING_TOOLS")  # no response.create, nobody spoke

        await harness.speak()
        await harness.wait_for("response.done", 2)

        self.assertEqual(len(backend.calls), 2)
        tail = self.resumed_messages(backend)[-2:]
        self.assertEqual([(m.role, m.content) for m in tail], [("tool", "ok"), ("user", "utterance 2")])
        self.assertEqual(self.records("wait_resumed")[0]["trigger"], "caller_turn")
        self.assertEqual(self.records("wait_input")[0]["after_outputs"], ["call_a"])
        self.assertEqual(self.spoken(harness), "Found them.")
        await harness.close()

    async def test_turn_commits_then_the_last_output_arrives(self) -> None:
        harness, frontend, backend = self.make()
        frontend.queue(delegate_response("Look up the user."))
        backend.queue(tool_response(("get_users", {}), ids=["call_a"]), text_response("Found them."))
        await self.tool_wait(harness)
        await harness.speak()
        self.assertEqual(harness.session.turns.state, "AWAITING_TOOLS")
        self.assertEqual(len(backend.calls), 1)

        await harness.send(output("call_a"))
        await harness.wait_for("response.done", 2)

        self.assertEqual(len(backend.calls), 2)
        tail = self.resumed_messages(backend)[-2:]
        self.assertEqual([(m.role, m.content) for m in tail], [("tool", "ok"), ("user", "utterance 2")])
        # The "said while" marker: before any output had arrived. Metadata only.
        meta = tail[1].meta["tool_wait"]
        self.assertEqual(meta["inputs"][0]["after_outputs"], [])
        self.assertEqual(meta["outstanding"], ["call_a"])
        await harness.close()

    async def test_two_turns_before_the_last_output_form_one_user_message(self) -> None:
        harness, frontend, backend = self.make()
        frontend.queue(delegate_response("Look up the user."))
        backend.queue(tool_response(("get_users", {}), ids=["call_a"]), text_response("Found them."))
        await self.tool_wait(harness)
        await harness.speak()
        await harness.speak()

        await harness.send(output("call_a"))
        await harness.wait_for("response.done", 2)

        messages = self.resumed_messages(backend)
        self.assertEqual([m.role for m in messages[-2:]], ["tool", "user"])
        self.assertEqual(messages[-1].content, "utterance 2 utterance 3")
        seqs = [entry["seq"] for entry in messages[-1].meta["tool_wait"]["inputs"]]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(self.records("wait_resumed")), 1)
        await harness.close()

    async def test_result_timeout_resumes_with_available_outputs_and_the_words(self) -> None:
        harness, frontend, backend = self.make(config=voice_config(tools_result_timeout_s=0.2))
        frontend.queue(delegate_response("Look up both."))
        backend.queue(
            tool_response(("get_users", {}), ("get_orders", {}), ids=["call_a", "call_b"]),
            text_response("Here is what I found."),
        )
        await self.tool_wait(harness)
        await harness.send(output("call_a"))
        await harness.speak()
        self.assertEqual(harness.session.turns.state, "AWAITING_TOOLS")  # call_b is still missing

        await harness.wait_for("response.done", 2, timeout=3.0)

        self.assertEqual(self.records("tool_result_timeout")[0]["missing"], ["call_b"])
        messages = self.resumed_messages(backend)
        self.assertEqual([m.role for m in messages[-3:]], ["tool", "tool", "user"])
        self.assertEqual(messages[-3].content, "ok")
        self.assertEqual(messages[-1].content, "utterance 2")  # no loss
        await harness.close()

    async def test_backchannel_only_turn_resumes_and_is_passed_on(self) -> None:
        harness, frontend, backend = self.make(recognizer=StubRecognizer(["Look up my account.", "Okay."]))
        frontend.queue(delegate_response("Look up the user."))
        backend.queue(tool_response(("get_users", {}), ids=["call_a"]), text_response("Found them."))
        await self.tool_wait(harness)
        await harness.send(output("call_a"))
        await harness.speak()
        await harness.wait_for("response.done", 2)

        self.assertEqual(self.resumed_messages(backend)[-1].content, "Okay.")
        self.assertEqual(self.spoken(harness), "Found them.")
        await harness.close()

    async def test_turn_during_the_resumed_step_follows_todays_queue(self) -> None:
        harness, frontend, backend = self.make()
        frontend.queue(delegate_response("Look up the user."), delegate_response("Another question."))
        backend.queue(
            tool_response(("get_users", {}), ids=["call_a"]),
            text_response("Found them."),
            text_response("Second answer."),
        )
        gate = backend.gates.setdefault(1, asyncio.Event())
        await self.tool_wait(harness)
        await harness.speak()  # into the wait's inbox
        await harness.send(output("call_a"))
        self.assertEqual(harness.session.turns.state, "THINKING")

        await harness.speak()  # during the resumed step: queued as today, never cancels it
        self.assertEqual(self.records("input_queued")[0]["state"], "THINKING")
        self.assertEqual(self.records("input_queued")[0]["text"], "utterance 3")
        gate.set()
        await harness.wait_for("response.done", 3)

        self.assertEqual(frontend.calls[1]["messages"][-1].content, "utterance 3")
        self.assertIn("Second answer.", self.spoken(harness))
        self.assertFalse(self.records("thinking_cancelled"))
        await harness.close()


class LateResponseCreateTests(ToolResumeTestCase):
    async def inbox_resume(self) -> tuple[SessionHarness, FakeChatClient, FakeChatClient, asyncio.Event]:
        """Speech during the wait, then the output: the wait resumes and the resumed step is held."""
        harness, frontend, backend = self.make()
        frontend.queue(delegate_response("Look up the user."), delegate_response("Next."))
        backend.queue(
            tool_response(("get_users", {}), ids=["call_a"]),
            text_response("Found them."),
            text_response("Next answer."),
        )
        gate = backend.gates.setdefault(1, asyncio.Event())
        await self.tool_wait(harness)
        await harness.speak()
        await harness.send(output("call_a"))
        self.assertEqual(harness.session.turns.state, "THINKING")
        return harness, frontend, backend, gate

    def errors(self, harness: SessionHarness) -> list[str]:
        return [event["error"].get("code") for event in harness.of_type("error")]

    def rules(self) -> list[tuple[int, str]]:
        return [(record["wait_id"], record["rule"]) for record in self.records("response_create")]

    async def test_create_while_the_resumed_step_runs_is_consumed_once(self) -> None:
        harness, _, _, gate = await self.inbox_resume()
        await harness.send({"type": "response.create"})
        gate.set()
        await harness.wait_for("response.done", 2)
        await harness.idle(300)

        self.assertEqual(self.errors(harness), [])
        self.assertEqual(self.rules(), [(1, "consumed_after_inbox_resume")])
        self.assertEqual(len(harness.of_type("response.created")), 2)
        await harness.close()

    async def test_second_create_follows_todays_rules(self) -> None:
        harness, _, _, gate = await self.inbox_resume()
        await harness.send({"type": "response.create"})
        await harness.send({"type": "response.create"})
        gate.set()
        await harness.wait_for("response.done", 2)

        self.assertEqual(self.rules(), [(1, "consumed_after_inbox_resume")])
        self.assertEqual(self.errors(harness), ["conversation_already_has_active_response"])
        await harness.close()

    async def test_create_after_a_new_caller_turn_is_not_consumed(self) -> None:
        harness, _, _, gate = await self.inbox_resume()
        await harness.speak()  # a new caller turn commits while W's response runs
        await harness.send({"type": "response.create"})

        self.assertEqual(self.rules(), [])
        self.assertEqual(self.errors(harness), ["conversation_already_has_active_response"])
        gate.set()
        await harness.wait_for("response.done", 3)
        await harness.close()

    async def test_create_when_the_agent_is_idle_is_not_consumed(self) -> None:
        harness, _, _, gate = await self.inbox_resume()
        gate.set()
        await harness.wait_for("response.done", 2)
        await harness.idle(300)
        self.assertEqual(harness.session.turns.state, "IDLE")

        await harness.send({"type": "response.create"})

        self.assertEqual(self.rules(), [])
        # Today's rule when idle: start from pending inputs; there are none here.
        self.assertEqual(self.errors(harness), ["invalid_value"])
        await harness.close()

    async def test_create_after_the_next_waits_response_done_applies_to_that_wait(self) -> None:
        harness, frontend, backend = self.make()
        frontend.queue(delegate_response("Look up the user."))
        backend.queue(
            tool_response(("get_users", {}), ids=["call_a"]),
            tool_response(("get_orders", {}), ids=["call_b"]),
            text_response("Here are the orders."),
        )
        await self.tool_wait(harness)
        await harness.speak()
        await harness.send(output("call_a"))  # inbox resume of wait 1 -> a second tool call, wait 2
        await harness.wait_for("response.done", 2)
        self.assertEqual(harness.session.turns.state, "AWAITING_TOOLS")

        await harness.send({"type": "response.create"})  # after wait 2's response.done
        self.assertEqual(self.rules(), [(2, "resume_wait")])
        await harness.send(output("call_b"))
        await harness.wait_for("response.done", 3)

        self.assertEqual(self.errors(harness), [])
        self.assertEqual(self.spoken(harness), "Here are the orders.")
        await harness.close()


class ResumedMessageTests(ToolResumeTestCase):
    async def test_user_message_content_is_the_words_only(self) -> None:
        harness, frontend, backend = self.make()
        frontend.queue(delegate_response("Look up the user."))
        backend.queue(tool_response(("get_users", {}), ids=["call_a"]), text_response("Found them."))
        await self.tool_wait(harness)
        await harness.speak()
        await harness.send(output("call_a"))
        await harness.wait_for("response.done", 2)

        message = self.resumed_messages(backend)[-1]
        self.assertEqual(to_wire_messages([message]), [{"role": "user", "content": "utterance 2"}])
        self.assertTrue(message.meta[CONTINUES_TURN])
        self.assertEqual(message.meta["tool_wait"]["wait_id"], 1)
        logged = self.records("wait_input")
        self.assertEqual([(r["wait_id"], r["seq"], r["after_outputs"]) for r in logged], [(1, 1, [])])

        # The frontend history keeps the words after the delegation's result, inside the same turn.
        runner = harness.runners[0]
        frontend_messages = runner.state.frontend_history.messages
        self.assertEqual([m.role for m in frontend_messages[-3:]], ["tool", "user", "assistant"])
        self.assertEqual(frontend_messages[-2].content, "utterance 2")
        last_group = runner.state.frontend_history.groups()[-1]
        self.assertEqual([m.role for m in last_group], ["user", "assistant", "tool", "user", "assistant"])
        await harness.close()

    async def test_barge_in_after_an_inbox_resume_repairs_every_copy(self) -> None:
        answer = "Your booking is confirmed. The total is two hundred dollars. Anything else?"
        harness, frontend, backend = self.make()
        frontend.queue(delegate_response("Look up the user."))
        backend.queue(tool_response(("get_users", {}), ids=["call_a"]), text_response(answer))
        await self.tool_wait(harness)
        await harness.speak()
        await harness.send(output("call_a"))
        await harness.wait_for("response.done", 2)
        await harness.feed(pcmu_silence(300))
        await harness.feed(pcmu_speech(300))  # the caller talks over the answer
        await harness.wait_for("input_audio_buffer.speech_started", 3)

        self.assertFalse(self.records("history_repair_failed"))
        group = harness.runners[0].state.frontend_history.messages[-5:]
        self.assertEqual([m.role for m in group], ["user", "assistant", "tool", "user", "assistant"])
        self.assertEqual(group[2].content, group[4].content)
        self.assertTrue(group[4].content.endswith("[interrupted by the user]"))
        self.assertNotIn("Anything else", group[2].content)
        await harness.close()


class NoOverlapTests(ToolResumeTestCase):
    async def test_no_caller_speech_keeps_todays_event_stream(self) -> None:
        harness, frontend, backend = self.make()
        frontend.queue(delegate_response("Look up the user."))
        backend.queue(tool_response(("get_users", {}), ids=["call_a"]), text_response("Found them."))
        await self.tool_wait(harness)
        await harness.send(output("call_a"))
        await harness.send({"type": "response.create"})
        await harness.wait_for("response.done", 2)
        await harness.idle(300)
        await harness.close()

        trace = json.dumps(normalize_trace(harness.events), sort_keys=True).encode("utf-8")
        self.assertEqual(hashlib.sha256(trace).hexdigest(), NO_OVERLAP_TRACE_SHA256)
        self.assertFalse(self.records("wait_input"))
        self.assertFalse(self.records("wait_resumed"))


if __name__ == "__main__":
    unittest.main()
