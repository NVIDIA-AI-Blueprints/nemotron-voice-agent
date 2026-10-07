# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""barge_in.while_thinking: frontend_verdict (plan: voice-frontend-backend-agent-barge-in-frontend-verdict-plan.md)."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest import mock

import yaml
from _fbv_voice_fakes import (
    BASE_CONFIG,
    PACKAGE_DIR,
    FakeChatClient,
    SessionHarness,
    delegate_response,
    pcmu_silence,
    pcmu_speech,
    tau2_session_update,
    text_response,
    tool_response,
    voice_config,
)

from examples.frontend_backend_verdict.voice.agent.runner import AgentClients
from examples.frontend_backend_verdict.voice.agent.sinks import EventLog
from examples.frontend_backend_verdict.voice.config import FrontendVerdictConfig, VoiceConfig, load_voice_config
from examples.frontend_backend_verdict.voice.errors import VoiceConfigError
from examples.frontend_backend_verdict.voice.server import ServerOptions, _agent_factory, _AppState
from examples.frontend_backend_verdict.voice.speech.stubs import StubRecognizer, ToneSynthesizer

FILLER = "Let me check."


def verdict_config(**sections: Any) -> VoiceConfig:
    return voice_config(barge_in_while_thinking="frontend_verdict", **sections)


class FailingRecognizer(StubRecognizer):
    """A stub recognizer whose N-th utterance (1-based) fails to transcribe."""

    def __init__(self, fail: set[int], transcripts: list[str] | None = None) -> None:
        super().__init__(transcripts)
        self.fail = fail

    def open(self, **kwargs: Any):
        stream = super().open(**kwargs)
        if self.count in self.fail:

            async def finish() -> str:
                raise RuntimeError("ASR endpoint unavailable")

            stream.finish = finish
        return stream


async def until(condition: Callable[[], bool], timeout: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.005)


class VerdictTestCase(unittest.IsolatedAsyncioTestCase):
    redact = False

    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.log_path = Path(self._tmp.name) / "events.jsonl"

    def make(
        self, *, config: VoiceConfig | None = None, **harness: Any
    ) -> tuple[SessionHarness, FakeChatClient, FakeChatClient]:
        frontend, backend = FakeChatClient(), FakeChatClient()
        session = SessionHarness(
            config=config or verdict_config(),
            clients=AgentClients(backend=backend, frontend=frontend),
            event_log=EventLog(str(self.log_path), redact_content=self.redact),
            **harness,
        )
        return session, frontend, backend

    def records(self, kind: str | None = None) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines]
        return [r for r in records if kind is None or r["kind"] == kind]

    def has(self, kind: str) -> Callable[[], bool]:
        return lambda: bool(self.records(kind))

    @staticmethod
    def spoken(harness: SessionHarness) -> str:
        return "".join(e["delta"] for e in harness.of_type("response.output_audio_transcript.delta"))

    @staticmethod
    def users(harness: SessionHarness) -> list[str]:
        return [m.content for m in harness.runners[0].state.frontend_history.messages if m.role == "user"]

    async def start_running_turn(
        self, harness: SessionHarness, frontend: FakeChatClient, backend: FakeChatClient
    ) -> asyncio.Event:
        """Utterance 1 delegates; the backend call hangs until the returned gate is set."""
        gate = backend.gates.setdefault(0, asyncio.Event())
        await harness.start(tau2_session_update())
        await harness.speak()
        await until(lambda: len(backend.calls) == 1)
        self.assertIsNotNone(harness.runners[0].in_flight)
        return gate


class ContinueAndNewTests(VerdictTestCase):
    async def test_continue_keeps_the_running_turn(self) -> None:
        harness, frontend, backend = self.make()
        frontend.queue(delegate_response("Check order 1151."), delegate_response("Check order 1151.", task="continue"))
        backend.queue(text_response("Order 1151 has shipped."))
        gate = await self.start_running_turn(harness, frontend, backend)
        await harness.speak()
        await until(self.has("barge_in_verdict"))
        gate.set()
        await harness.wait_for("response.done")
        self.assertEqual(self.spoken(harness), "Order 1151 has shipped.")
        self.assertEqual((len(frontend.calls), len(backend.calls)), (2, 1))
        self.assertEqual(self.users(harness), ["utterance 1"])  # the acknowledgement left no trace
        probe = frontend.calls[1]["messages"]
        self.assertIn("REQUEST IN PROGRESS", probe[0].content)
        self.assertIn("Check order 1151.", probe[0].content)
        self.assertIn("not spoken aloud", probe[0].content)  # base config: filler log_only
        self.assertIn('The user has just said: "utterance 2"', probe[0].content)
        self.assertNotIn("the schema accepts no other fields", probe[0].content)
        self.assertEqual(probe[-1].content, "utterance 1 utterance 2")
        self.assertNotIn("REQUEST IN PROGRESS", frontend.calls[0]["messages"][0].content)
        (verdict,) = self.records("barge_in_verdict")
        self.assertEqual(
            (verdict["verdict"], verdict["reason"], verdict["task_state"]), ("continue", "model", "running")
        )
        self.assertEqual(verdict["running_query"], "Check order 1151.")
        self.assertFalse(self.records("thinking_cancelled"))
        self.assertNotIn("staged", self.records("agent_turn_done")[-1])
        await harness.close()

    async def test_new_cancels_and_carries_the_probe_out(self) -> None:
        harness, frontend, backend = self.make()
        frontend.queue(delegate_response("Check order 1151."), delegate_response("Check order 1152.", task="new"))
        backend.queue(text_response("Order 1152 is pending."))
        await self.start_running_turn(harness, frontend, backend)
        await harness.speak()
        await harness.wait_for("response.done")
        self.assertEqual(self.spoken(harness), "Order 1152 is pending.")
        self.assertEqual(len(frontend.calls), 2)  # no second frontend call for the new turn
        self.assertIn("Check order 1152.", backend.calls[1]["messages"][-1].content)
        self.assertEqual(self.users(harness), ["utterance 1 utterance 2"])  # same as cancel_and_merge
        (cancelled,) = self.records("thinking_cancelled")
        self.assertEqual(cancelled["reason"], "frontend_verdict_new")
        self.assertEqual([r["query"] for r in self.records("delegation")], ["Check order 1151.", "Check order 1152."])
        await harness.close()

    async def test_direct_answer_counts_as_new(self) -> None:
        harness, frontend, backend = self.make()
        frontend.queue(delegate_response("Check order 1151."), text_response("It is still running."))
        await self.start_running_turn(harness, frontend, backend)
        await harness.speak()
        await harness.wait_for("response.done")
        self.assertEqual(self.spoken(harness), "It is still running.")
        self.assertEqual(len(backend.calls), 1)
        verdict = self.records("barge_in_verdict")[0]
        self.assertEqual((verdict["verdict"], verdict["reason"]), ("new", "direct_answer"))
        await harness.close()

    async def test_invalid_task_counts_as_new(self) -> None:
        harness, frontend, backend = self.make()
        frontend.queue(delegate_response("Check order 1151."), delegate_response("Check order 1152.", task="maybe"))
        backend.queue(text_response("Order 1152 is pending."))
        await self.start_running_turn(harness, frontend, backend)
        await harness.speak()
        await harness.wait_for("response.done")
        verdict = self.records("barge_in_verdict")[0]
        self.assertEqual((verdict["verdict"], verdict["reason"], verdict["model_task"]), ("new", "task_invalid", ""))
        self.assertEqual(len(frontend.calls), 2)
        await harness.close()


class SameQueryGuardTests(VerdictTestCase):
    async def _verdict(self, probe_query: str, task: str, config: VoiceConfig | None = None) -> dict[str, Any]:
        harness, frontend, backend = self.make(config=config)
        frontend.queue(delegate_response("Check order 1151."), delegate_response(probe_query, task=task))
        backend.queue(text_response("Order 1151 has shipped."), text_response("Other answer."))
        gate = await self.start_running_turn(harness, frontend, backend)
        await harness.speak()
        await until(self.has("barge_in_verdict"))
        gate.set()
        await harness.wait_for("response.done")
        await harness.close()
        return self.records("barge_in_verdict")[0]

    async def test_identical_query_with_new_is_continue(self) -> None:
        verdict = await self._verdict("Check order 1151.", "new")
        self.assertEqual(
            (verdict["verdict"], verdict["reason"], verdict["model_task"]), ("continue", "same_query", "new")
        )
        self.assertFalse(self.records("thinking_cancelled"))

    async def test_case_and_punctuation_do_not_matter(self) -> None:
        verdict = await self._verdict("check ORDER 1151", "new")
        self.assertEqual((verdict["verdict"], verdict["reason"]), ("continue", "same_query"))

    async def test_a_changed_identifier_stays_new(self) -> None:
        verdict = await self._verdict("Check order 1152.", "new")
        self.assertEqual((verdict["verdict"], verdict["reason"]), ("new", "model"))

    async def test_identical_query_with_continue_keeps_reason_model(self) -> None:
        verdict = await self._verdict("Check order 1151.", "continue")
        self.assertEqual(
            (verdict["verdict"], verdict["reason"], verdict["model_task"]), ("continue", "model", "continue")
        )

    async def test_guard_off_lets_the_model_decide(self) -> None:
        config = verdict_config(barge_in_frontend_verdict=FrontendVerdictConfig(same_query_guard=False))
        verdict = await self._verdict("Check order 1151.", "new", config)
        self.assertEqual((verdict["verdict"], verdict["reason"]), ("new", "model"))

    async def test_session_6df167087c2d41e3990d_acknowledgements(self) -> None:
        """The first live sessions: the model said "new" with the identical query for both acknowledgements."""
        recognizer = StubRecognizer(
            [
                "Hello , can you please tell me the status of the order one, two, three one?",
                "Okay , please check.",
                "Okay .",
            ]
        )
        harness, frontend, backend = self.make(recognizer=recognizer)
        query = "The user is asking for the status of order 1231."
        frontend.queue(delegate_response(query), *(delegate_response(query, task="new") for _ in range(2)))
        backend.queue(text_response("I couldn't find an order with ID 1231."))
        gate = await self.start_running_turn(harness, frontend, backend)
        for count in (1, 2):
            await harness.speak()
            await until(lambda count=count: len(self.records("barge_in_verdict")) == count)
        gate.set()
        await harness.wait_for("response.done")
        self.assertEqual([r["reason"] for r in self.records("barge_in_verdict")], ["same_query", "same_query"])
        self.assertEqual((len(self.records("delegation")), len(backend.calls)), (1, 1))
        self.assertEqual(self.spoken(harness), "I couldn't find an order with ID 1231.")
        await harness.close()


class StagedTests(VerdictTestCase):
    async def _staged(self, probe_task: str) -> tuple[SessionHarness, FakeChatClient, FakeChatClient]:
        harness, frontend, backend = self.make()
        frontend.queue(delegate_response("Check order 1151."), delegate_response("Check order 1152.", task=probe_task))
        backend.queue(text_response("Order 1151 has shipped."), text_response("Order 1152 is pending."))
        probe_gate = frontend.gates.setdefault(1, asyncio.Event())
        gate = await self.start_running_turn(harness, frontend, backend)
        await harness.speak()
        await until(lambda: len(frontend.calls) == 2)
        gate.set()
        await until(self.has("turn_staged"))
        await harness.settle()
        self.assertEqual(harness.of_type("response.done"), [])  # nothing is spoken during the review
        self.assertEqual(harness.session.turns.state, "THINKING")
        probe_gate.set()
        await harness.wait_for("response.done")
        return harness, frontend, backend

    async def test_staged_then_continue_speaks_the_held_answer(self) -> None:
        harness, frontend, backend = await self._staged("continue")
        self.assertEqual(self.spoken(harness), "Order 1151 has shipped.")
        self.assertTrue(self.records("staged_committed"))
        self.assertEqual(self.records("barge_in_verdict")[0]["task_state"], "staged")
        self.assertEqual(self.users(harness), ["utterance 1"])
        self.assertTrue(self.records("agent_turn_done")[0]["staged"])
        await harness.close()

    async def test_staged_then_new_discards_it_and_proceeds_on_the_earlier_state(self) -> None:
        harness, frontend, backend = await self._staged("new")
        self.assertEqual(self.spoken(harness), "Order 1152 is pending.")
        self.assertTrue(self.records("staged_discarded"))
        self.assertFalse(self.records("barge_in_fallback"))  # the state identity check passed
        self.assertEqual(self.users(harness), ["utterance 1 utterance 2"])
        self.assertEqual(len(frontend.calls), 2)
        await harness.close()


class ToolCallsDuringReviewTests(VerdictTestCase):
    async def test_calls_go_out_and_new_words_are_queued(self) -> None:
        harness, frontend, backend = self.make()
        frontend.queue(
            delegate_response("Look up the user."),
            delegate_response("Second question.", task="new"),
            delegate_response("Second question."),
        )
        backend.queue(
            tool_response(("get_users", {}), ids=["call_a"]),
            text_response("Found them."),
            text_response("Second answer."),
        )
        probe_gate = frontend.gates.setdefault(1, asyncio.Event())
        gate = await self.start_running_turn(harness, frontend, backend)
        await harness.speak()
        await until(lambda: len(frontend.calls) == 2)
        gate.set()
        await harness.wait_for("response.done")
        self.assertEqual(harness.session.turns.state, "AWAITING_TOOLS")
        self.assertFalse(self.records("turn_staged"))  # tool calls are never staged
        probe_gate.set()
        await until(self.has("barge_in_verdict"))
        self.assertEqual(self.records("barge_in_verdict")[0]["task_state"], "tools_out")
        await harness.send(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": "call_a", "output": "ok"},
            }
        )
        await harness.send({"type": "response.create"})
        await harness.wait_for("response.done", 3)
        self.assertEqual(frontend.calls[2]["messages"][-1].content, "utterance 2")
        self.assertIn("Second answer.", self.spoken(harness))
        self.assertFalse(self.records("thinking_cancelled"))
        await harness.close()


class FallbackTests(VerdictTestCase):
    async def _fallback(self, harness, frontend, backend) -> None:
        frontend.queue(delegate_response("Check order 1151."), delegate_response("Merged request."))
        backend.queue(text_response("Merged answer."))
        await self.start_running_turn(harness, frontend, backend)
        await harness.speak()
        await harness.wait_for("response.done")
        self.assertEqual(self.spoken(harness), "Merged answer.")
        self.assertEqual(frontend.calls[2]["messages"][-1].content, "utterance 1 utterance 2")
        self.assertEqual(self.records("thinking_cancelled")[0]["reason"], "frontend_verdict_fallback")

    async def test_probe_timeout_falls_back_to_cancel_and_merge(self) -> None:
        config = verdict_config(barge_in_frontend_verdict=FrontendVerdictConfig(timeout_ms=50))
        harness, frontend, backend = self.make(config=config)
        frontend.gates[1] = asyncio.Event()  # the probe never answers
        await self._fallback(harness, frontend, backend)
        verdict = self.records("barge_in_verdict")[0]
        self.assertEqual((verdict["verdict"], verdict["reason"]), ("new_fallback", "timeout"))
        await harness.close()

    async def test_probe_error_falls_back_to_cancel_and_merge(self) -> None:
        harness, frontend, backend = self.make()
        frontend.failures[1] = RuntimeError("frontend down")
        await self._fallback(harness, frontend, backend)
        self.assertEqual(self.records("barge_in_verdict")[0]["reason"], "error")
        await harness.close()


class PendingUtteranceTests(VerdictTestCase):
    async def test_a_newer_transcript_supersedes_a_running_probe(self) -> None:
        harness, frontend, backend = self.make()
        frontend.queue(delegate_response("Check order 1151."), delegate_response("Check order 1151.", task="continue"))
        backend.queue(text_response("Order 1151 has shipped."))
        frontend.gates[1] = asyncio.Event()  # the first probe never answers
        gate = await self.start_running_turn(harness, frontend, backend)
        await harness.speak()
        await until(lambda: len(frontend.calls) == 2)
        await harness.speak()
        await until(self.has("barge_in_verdict"))
        gate.set()
        await harness.wait_for("response.done")
        self.assertEqual([r["reason"] for r in self.records("probe_discarded")], ["superseded"])
        self.assertEqual(len(self.records("barge_in_verdict")), 1)
        self.assertEqual(frontend.calls[2]["messages"][-1].content, "utterance 1 utterance 2 utterance 3")
        self.assertEqual(self.spoken(harness), "Order 1151 has shipped.")
        await harness.close()

    async def test_no_verdict_while_an_utterance_awaits_its_transcript(self) -> None:
        harness, frontend, backend = self.make()
        frontend.queue(
            delegate_response("Check order 1151."),
            delegate_response("Check order 1151.", task="continue"),
            delegate_response("Check order 1152.", task="new"),
        )
        backend.queue(text_response("Order 1152 is pending."))
        probe_gate = frontend.gates.setdefault(1, asyncio.Event())
        await self.start_running_turn(harness, frontend, backend)
        await harness.speak()
        await until(lambda: len(frontend.calls) == 2)
        await harness.feed(pcmu_speech(600))  # utterance 3 starts; its transcript is not in yet
        probe_gate.set()
        await until(lambda: not frontend.gates[1].is_set() or len(frontend.responses) == 1)
        await harness.settle()
        self.assertFalse(self.records("barge_in_verdict"))  # the ready "continue" is held (rule R3)
        await harness.feed(pcmu_silence(700))
        await harness.wait_for("response.done")
        self.assertEqual([r["reason"] for r in self.records("probe_discarded")], ["stale"])
        verdict = self.records("barge_in_verdict")[0]
        self.assertEqual(verdict["verdict"], "new")
        self.assertEqual(frontend.calls[2]["messages"][-1].content, "utterance 1 utterance 2 utterance 3")
        self.assertEqual(self.spoken(harness), "Order 1152 is pending.")
        await harness.close()

    async def test_a_held_verdict_applies_once_the_pending_utterance_fails(self) -> None:
        harness, frontend, backend = self.make(recognizer=FailingRecognizer({3}))
        frontend.queue(delegate_response("Check order 1151."), delegate_response("Check order 1151.", task="continue"))
        backend.queue(text_response("Order 1151 has shipped."))
        probe_gate = frontend.gates.setdefault(1, asyncio.Event())
        gate = await self.start_running_turn(harness, frontend, backend)
        await harness.speak()
        await until(lambda: len(frontend.calls) == 2)
        await harness.feed(pcmu_speech(600))
        probe_gate.set()
        await until(lambda: len(frontend.responses) == 0)
        await harness.settle()
        self.assertFalse(self.records("barge_in_verdict"))
        await harness.feed(pcmu_silence(700))  # utterance 3's transcription fails
        await until(self.has("barge_in_verdict"))
        verdict = self.records("barge_in_verdict")[0]
        self.assertEqual((verdict["verdict"], verdict["reason"]), ("continue", "model"))
        self.assertGreaterEqual(verdict["held_ms"], 0)
        gate.set()
        await harness.wait_for("response.done")
        self.assertEqual(self.spoken(harness), "Order 1151 has shipped.")
        await harness.close()


class ReviewClosingTests(VerdictTestCase):
    async def _closes_without_transcript(self, harness, frontend, backend, speak) -> None:
        frontend.queue(delegate_response("Check order 1151."))
        backend.queue(text_response("Order 1151 has shipped."))
        gate = await self.start_running_turn(harness, frontend, backend)
        await speak()
        await until(self.has("barge_in_closed"))
        self.assertEqual(self.records("barge_in_closed")[0]["reason"], "no_transcript")
        self.assertEqual(self.records("barge_in_verdict")[0]["verdict"], "continue")
        self.assertIsNone(harness.session.turns._turn.review)
        gate.set()
        await harness.wait_for("response.done")
        self.assertEqual(self.spoken(harness), "Order 1151 has shipped.")
        self.assertEqual(len(frontend.calls), 1)  # nothing to probe
        self.assertNotIn("staged", self.records("agent_turn_done")[-1])

    async def test_asr_failure(self) -> None:
        harness, frontend, backend = self.make(recognizer=FailingRecognizer({2}))
        await self._closes_without_transcript(harness, frontend, backend, harness.speak)
        self.assertEqual(self.records("barge_in_awaiting")[-1]["change"], "asr_failed")
        await harness.close()

    async def test_input_audio_cleared(self) -> None:
        harness, frontend, backend = self.make()

        async def speak_then_clear() -> None:
            await harness.feed(pcmu_speech(600))
            await harness.send({"type": "input_audio_buffer.clear"})

        await self._closes_without_transcript(harness, frontend, backend, speak_then_clear)
        self.assertEqual(self.records("barge_in_awaiting")[-1]["change"], "audio_cleared")
        await harness.close()

    async def test_empty_transcript(self) -> None:
        harness, frontend, backend = self.make(recognizer=StubRecognizer(["Check order 1151.", ""]))
        await self._closes_without_transcript(harness, frontend, backend, harness.speak)
        self.assertEqual(self.records("barge_in_awaiting")[-1]["change"], "empty_transcript")
        await harness.close()

    async def test_automatic_responses_turned_off_during_the_review(self) -> None:
        harness, frontend, backend = self.make()
        frontend.queue(delegate_response("Check order 1151."), delegate_response("Merged request."))
        backend.queue(text_response("Merged answer."))
        await self.start_running_turn(harness, frontend, backend)
        await harness.feed(pcmu_speech(600))
        update = tau2_session_update()
        update["session"]["audio"]["input"]["turn_detection"]["create_response"] = False
        await harness.send(update)
        await harness.feed(pcmu_silence(700))
        await until(self.has("barge_in_closed"))
        self.assertEqual(self.records("barge_in_closed")[0]["reason"], "auto_response_off")
        self.assertEqual(self.records("thinking_cancelled")[0]["reason"], "auto_response_off")
        await harness.send({"type": "response.create"})
        await harness.wait_for("response.done")
        self.assertEqual(frontend.calls[1]["messages"][-1].content, "utterance 1 utterance 2")
        self.assertEqual(self.spoken(harness), "Merged answer.")
        await harness.close()

    async def test_client_cancel(self) -> None:
        harness, frontend, backend = self.make()
        frontend.queue(delegate_response("Check order 1151."))
        frontend.gates[1] = asyncio.Event()
        await self.start_running_turn(harness, frontend, backend)
        await harness.speak()
        await until(lambda: len(frontend.calls) == 2)
        await harness.send({"type": "response.cancel"})
        self.assertEqual(self.records("barge_in_closed")[0]["reason"], "client_cancelled")
        self.assertEqual(self.records("thinking_cancelled")[0]["reason"], "client_cancelled")
        await harness.settle()
        self.assertEqual(harness.session.turns.state, "IDLE")
        self.assertIsNone(harness.runners[0]._staging_call)
        await harness.close()

    async def test_session_close(self) -> None:
        harness, frontend, backend = self.make()
        frontend.queue(delegate_response("Check order 1151."))
        frontend.gates[1] = asyncio.Event()
        await self.start_running_turn(harness, frontend, backend)
        await harness.speak()
        await until(lambda: len(frontend.calls) == 2)
        await harness.close()
        self.assertEqual(self.records("barge_in_closed")[0]["reason"], "session_closed")

    async def _failing_turn(self, harness: SessionHarness, frontend: FakeChatClient) -> asyncio.Event:
        """Utterance 1 delegates, then the agent step itself raises once the returned gate is set.

        (Backend LLM errors are turned into a final text by the text agent; this is a failure that escapes.)
        """
        gate = asyncio.Event()
        await harness.start(tau2_session_update())
        agent = harness.runners[0]._agent
        original = agent.continue_turn
        calls = []

        async def continue_turn(step):
            calls.append(step)
            if len(calls) == 1:
                await gate.wait()
                raise RuntimeError("agent step failed")
            return await original(step)

        agent.continue_turn = continue_turn
        await harness.speak()
        await until(lambda: harness.runners[0].in_flight is not None)
        return gate

    async def test_agent_failure_after_the_transcript(self) -> None:
        harness, frontend, backend = self.make()
        frontend.queue(delegate_response("Check order 1151."), delegate_response("Merged request."))
        backend.queue(text_response("Recovered answer."))
        frontend.gates[1] = asyncio.Event()  # the probe is still running when the turn fails
        gate = await self._failing_turn(harness, frontend)
        await harness.speak()
        await until(lambda: len(frontend.calls) == 2)
        gate.set()
        await harness.wait_for("error")
        await until(lambda: "Recovered answer." in self.spoken(harness))
        self.assertEqual(self.records("barge_in_closed")[0]["reason"], "agent_failed")
        self.assertEqual(frontend.calls[2]["messages"][-1].content, "utterance 1 utterance 2")
        self.assertNotIn("staged", self.records("agent_turn_done")[-1])  # the next turn is not staged
        self.assertIsNone(harness.runners[0]._staging_call)
        await harness.close()

    async def test_agent_failure_while_a_transcript_is_pending(self) -> None:
        harness, frontend, backend = self.make()
        frontend.queue(delegate_response("Check order 1151."), delegate_response("Merged request."))
        backend.queue(text_response("Recovered answer."))
        gate = await self._failing_turn(harness, frontend)
        await harness.feed(pcmu_speech(600))
        await until(self.has("barge_in_review"))
        gate.set()
        await harness.wait_for("error")
        await until(self.has("barge_in_closed"))
        await harness.feed(pcmu_silence(700))
        await until(lambda: "Recovered answer." in self.spoken(harness))
        self.assertEqual(self.records("barge_in_closed")[0]["reason"], "agent_failed")
        self.assertEqual(frontend.calls[1]["messages"][-1].content, "utterance 1 utterance 2")
        self.assertNotIn("staged", self.records("agent_turn_done")[-1])
        await harness.close()


class ReviewOpeningTests(VerdictTestCase):
    async def test_speech_before_the_delegation_keeps_cancel_and_merge(self) -> None:
        harness, frontend, backend = self.make()
        frontend.queue(delegate_response("Merged request."))
        backend.queue(text_response("Merged answer."))
        frontend.gates[0] = asyncio.Event()  # still in the initial frontend call
        await harness.start(tau2_session_update())
        await harness.speak()
        await until(lambda: len(frontend.calls) == 1)
        await harness.speak()
        await harness.wait_for("response.done")
        self.assertFalse(self.records("barge_in_review"))
        self.assertEqual(self.records("thinking_cancelled")[0]["reason"], "turn_detected")
        self.assertEqual(frontend.calls[1]["messages"][-1].content, "utterance 1 utterance 2")
        await harness.close()

    async def test_filler_is_not_spoken_once_a_review_opened(self) -> None:
        synth = ToneSynthesizer(sample_rate=16000, ms_per_char=10.0)
        config = verdict_config(filler_mode="speak", filler_speak_after_ms=150)
        harness, frontend, backend = self.make(config=config, synthesizer=synth)
        frontend.queue(delegate_response("Check order 1151.", FILLER), delegate_response("Check.", task="continue"))
        backend.queue(text_response("Order 1151 has shipped."))
        gate = await self.start_running_turn(harness, frontend, backend)
        await harness.speak()  # before speak_after_ms has passed
        await until(self.has("filler_skipped"))
        self.assertEqual(self.records("filler_skipped")[0]["reason"], "barge_in_review")
        gate.set()
        await harness.wait_for("response.done")
        self.assertNotIn(FILLER, synth.requests)
        await harness.close()

    async def test_playing_filler_is_stopped_and_never_repeated(self) -> None:
        synth = ToneSynthesizer(sample_rate=16000, ms_per_char=10.0)
        config = verdict_config(filler_mode="speak", filler_speak_after_ms=0)
        harness, frontend, backend = self.make(config=config, synthesizer=synth)
        frontend.queue(delegate_response("Check order 1151.", FILLER), delegate_response("Check.", task="continue"))
        backend.queue(text_response("Order 1151 has shipped."))
        gate = await self.start_running_turn(harness, frontend, backend)
        await harness.wait_for("response.output_audio.delta")
        await harness.speak()
        first = await harness.wait_for("response.done")
        self.assertEqual(first["response"]["status"], "cancelled")
        self.assertTrue(self.records("barge_in_review")[0]["filler_spoken"])
        probe_prompt = frontend.calls[1]["messages"][0].content
        self.assertIn('you told the user: "Let me check."', probe_prompt)
        self.assertNotIn("the user did not hear it", probe_prompt)
        gate.set()
        await harness.wait_for("response.done", 2)
        self.assertEqual(synth.requests.count(FILLER), 1)
        self.assertIn("Order 1151 has shipped.", self.spoken(harness))
        await harness.close()


class ReplayTests(VerdictTestCase):
    async def test_session_0717897ad9064de78ea7_acknowledgements(self) -> None:
        acknowledgements = [
            "Yes, please do that.",
            "Okay .",
            "You are checking now. Why are you telling me?",
            "To check",
            "You are checking only . Why are you saying again and again?",
        ]
        recognizer = StubRecognizer(["Can you please check on order one one five one", *acknowledgements])
        harness, frontend, backend = self.make(recognizer=recognizer)
        frontend.queue(
            delegate_response("The user wants to check the status of order 1151."),
            *(delegate_response("The user wants to check on order 1151.", task="continue") for _ in acknowledgements),
        )
        backend.queue(text_response("I couldn't find an order with ID 1151."))
        gate = await self.start_running_turn(harness, frontend, backend)
        for count in range(1, len(acknowledgements) + 1):
            await harness.speak()
            await until(lambda count=count: len(self.records("barge_in_verdict")) == count)
        gate.set()
        await harness.wait_for("response.done")
        self.assertEqual(len(self.records("delegation")), 1)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual({r["verdict"] for r in self.records("barge_in_verdict")}, {"continue"})
        self.assertFalse(self.records("thinking_cancelled"))
        self.assertEqual(self.spoken(harness), "I couldn't find an order with ID 1151.")
        await harness.close()


class RedactionTests(VerdictTestCase):
    redact = True

    async def test_new_content_fields_are_redacted(self) -> None:
        harness, frontend, backend = self.make()
        frontend.queue(delegate_response("Check order 1151."), delegate_response("Check order 1151.", task="continue"))
        backend.queue(text_response("Order 1151 has shipped."))
        gate = await self.start_running_turn(harness, frontend, backend)
        await harness.speak()
        await until(self.has("barge_in_verdict"))
        gate.set()
        await harness.wait_for("response.done")
        verdict = self.records("barge_in_verdict")[0]
        for key in ("utterance", "merged_text", "running_query", "probe_query"):
            self.assertNotIn(key, verdict)
        self.assertEqual((verdict["verdict"], verdict["reason"]), ("continue", "model"))
        self.assertIn("latency_ms", verdict)
        await harness.close()


class WiringAndConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _runner(self, config: VoiceConfig):
        clients = AgentClients(backend=FakeChatClient(), frontend=FakeChatClient())
        state = _AppState(config=config, options=ServerOptions(), clients=clients)
        return _agent_factory(state)("sess_test")

    def test_server_factory_passes_the_verdict_settings(self) -> None:
        runner = self._runner(verdict_config())
        self.assertIsNotNone(runner._barge_in)
        self.assertTrue(runner._barge_in.same_query_guard)
        self.assertEqual(runner._prompt_context["barge_in"]["note_key"], "frontend_task_in_progress")
        self.assertIsNone(self._runner(voice_config())._barge_in)
        self.assertNotIn("REQUEST IN PROGRESS", runner.rendered_prompts()["frontend"])

    def test_client_policy_with_template_syntax_is_verbatim(self) -> None:
        config = verdict_config()
        config = replace(
            config, instructions=replace(config.instructions, apply_to=("backend", "frontend"), placement="append")
        )
        runner = self._runner(config)
        policy = "Greet with {{ customer_name }}; {% raw %} and {# not a comment #} stay as written."
        runner.configure(tools=(), instructions=policy)
        prompts = runner.rendered_prompts()
        self.assertIn(policy, prompts["backend"])
        self.assertIn(policy, prompts["frontend"])

    def test_a_broken_template_fails_at_load_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = yaml.safe_load((PACKAGE_DIR / "config" / "prompts.voice.yaml").read_text(encoding="utf-8"))
            catalog["frontend_task_in_progress"]["content"] += "{{ in_progress.no_such_field }}"
            (root / "prompts.yaml").write_text(yaml.safe_dump(catalog), encoding="utf-8")
            body = (
                f"extends: {BASE_CONFIG}\nbarge_in: {{while_thinking: frontend_verdict}}\n"
                f"agent: {{overrides: {{prompts: {{path: {root / 'prompts.yaml'}}}}}}}\n"
            )
            (root / "broken.yaml").write_text(body, encoding="utf-8")
            with self.assertRaisesRegex(VoiceConfigError, r"frontend.*no_such_field.*broken\.yaml"):
                load_voice_config(root / "broken.yaml")

    def test_profiles(self) -> None:
        profiles = PACKAGE_DIR / "config" / "profiles"
        for name, filler in (
            ("browser_demo_frontend_verdict", "speak"),
            ("tau3_eval_frontend_verdict", "log_only"),
            ("tau3_eval_frontend_verdict_speak", "speak"),
        ):
            config = load_voice_config(profiles / f"{name}.yaml")
            self.assertEqual((config.barge_in.while_thinking, config.filler.mode), ("frontend_verdict", filler))
        self.assertEqual(load_voice_config(profiles / "tau3_eval.yaml").barge_in.while_thinking, "cancel_and_merge")

    def test_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def profile(name: str, body: str) -> Path:
                path = root / name
                path.write_text(body, encoding="utf-8")
                return path

            on = "barge_in: {while_thinking: frontend_verdict"
            load_voice_config(profile("ok.yaml", f"extends: {BASE_CONFIG}\n{on}}}\n"))
            with self.assertRaisesRegex(VoiceConfigError, r"note_key.*'nope'.*missing\.yaml"):
                load_voice_config(
                    profile("missing.yaml", f"extends: {BASE_CONFIG}\n{on}, frontend_verdict: {{note_key: nope}}}}\n")
                )
            with self.assertRaisesRegex(VoiceConfigError, r"timeout_ms.*>= 1"):
                load_voice_config(
                    profile("zero.yaml", f"extends: {BASE_CONFIG}\n{on}, frontend_verdict: {{timeout_ms: 0}}}}\n")
                )
            backend_only = PACKAGE_DIR / "config" / "profiles" / "backend_only.yaml"
            with self.assertRaisesRegex(VoiceConfigError, "backend_only"):
                load_voice_config(profile("bo.yaml", f"extends: {backend_only}\n{on}}}\n"))
            with self.assertRaisesRegex(VoiceConfigError, "while_thinking"):
                load_voice_config(profile("bad.yaml", f"extends: {BASE_CONFIG}\nbarge_in: {{while_thinking: maybe}}\n"))
