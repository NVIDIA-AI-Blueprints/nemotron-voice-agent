# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Paired backend conversation history in the voice layer: repair, barge-in, profiles, logging."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path
from typing import Any
from unittest import mock

from _fbv_voice_fakes import (
    PACKAGE_DIR,
    FakeChatClient,
    SessionHarness,
    base_config,
    delegate_response,
    pcmu_silence,
    pcmu_speech,
    tau2_session_update,
    tau2_update_with,
    text_response,
    voice_config,
)

from examples.frontend_backend_verdict.text.config import ConversationHistoryConfig
from examples.frontend_backend_verdict.text.history import History
from examples.frontend_backend_verdict.text.messages import Message, ToolCall
from examples.frontend_backend_verdict.text.session import SessionState
from examples.frontend_backend_verdict.voice.agent.history_repair import repair_interrupted_answer
from examples.frontend_backend_verdict.voice.agent.runner import AgentClients
from examples.frontend_backend_verdict.voice.agent.sinks import EventLog
from examples.frontend_backend_verdict.voice.config import load_voice_config
from examples.frontend_backend_verdict.voice.errors import HistoryRepairError

PROFILES = PACKAGE_DIR / "config" / "profiles"
FULL = "Your refund is approved. It arrives in five days."
HEARD = "Your refund is approved. [interrupted by the user]"
MARKER = "[interrupted by the user]"
ANSWER = "Your booking is confirmed. The total is two hundred dollars. Anything else?"


def _history_config(**history: Any):
    base = base_config()
    backend = replace(
        base.agent.backend, conversation_history=ConversationHistoryConfig(enabled=True, **history), stateful=True
    )
    return voice_config(agent=replace(base.agent, backend=backend))


def _paired_state(frontend_answer: str = FULL, backend_answer: str = FULL) -> SessionState:
    call = ToolCall(id="fc1", name="call_backend", arguments_json='{"query": "refund"}')
    frontend = History(
        (
            Message.user("refund please"),
            Message.assistant_tool_calls((call,)),
            Message.tool("fc1", frontend_answer),
            Message.assistant(frontend_answer),
        )
    )
    backend = History((Message.user("request: refund"), Message.assistant(backend_answer)))
    return SessionState(frontend_history=frontend, backend_history=backend)


class DualRepairTests(unittest.TestCase):
    def test_delegated_answer_is_repaired_in_both_histories(self) -> None:
        state = repair_interrupted_answer(
            _paired_state(), full_text=FULL, replacement=HEARD, frontend_enabled=True, backend_history=True
        )
        self.assertEqual(state.frontend_history.messages[-1].content, HEARD)
        self.assertEqual(state.frontend_history.messages[-2].content, HEARD)
        self.assertEqual(state.backend_history.messages[-1].content, HEARD)

    def test_without_the_flag_the_backend_history_is_untouched(self) -> None:
        state = repair_interrupted_answer(_paired_state(), full_text=FULL, replacement=HEARD, frontend_enabled=True)
        self.assertEqual(state.backend_history.messages[-1].content, FULL)

    def test_direct_answer_never_touches_the_backend(self) -> None:
        earlier = History((Message.user("request: earlier"), Message.assistant("earlier backend answer")))
        state = SessionState(
            frontend_history=History((Message.user("hi"), Message.assistant(FULL))), backend_history=earlier
        )
        repaired = repair_interrupted_answer(
            state, full_text=FULL, replacement=HEARD, frontend_enabled=True, backend_history=True
        )
        self.assertEqual(repaired.frontend_history.messages[-1].content, HEARD)
        self.assertEqual(repaired.backend_history, earlier)

    def test_backend_mismatch_leaves_both_histories_unchanged(self) -> None:
        state = _paired_state(backend_answer="something else")
        with self.assertRaises(HistoryRepairError):
            repair_interrupted_answer(
                state, full_text=FULL, replacement=HEARD, frontend_enabled=True, backend_history=True
            )
        self.assertEqual(state.frontend_history.messages[-1].content, FULL)
        self.assertEqual(state.backend_history.messages[-1].content, "something else")


class PairedHistorySessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_second_spoken_turn_reaches_the_backend_with_the_first(self) -> None:
        frontend = FakeChatClient([delegate_response("First request."), delegate_response("Second request.")])
        backend = FakeChatClient([text_response("First answer."), text_response("Second answer.")])
        harness = SessionHarness(config=_history_config(), clients=AgentClients(backend=backend, frontend=frontend))
        await harness.start(tau2_session_update())
        await harness.speak()
        await harness.wait_for("response.done")
        await harness.feed(pcmu_silence(2000))
        await harness.speak()
        await harness.wait_for("response.done", 2)
        first, second = backend.calls[0]["messages"], backend.calls[1]["messages"]
        # the seeded client greeting reaches the backend on the first delegation
        self.assertIn("Assistant: Hi! How can I help you today?", first[-1].content)
        self.assertIn("utterance 1", first[-1].content)
        self.assertEqual([m.role for m in second], ["system", "user", "assistant", "user"])
        self.assertEqual(second[2].content, "First answer.")
        self.assertIn("utterance 2", second[-1].content)
        self.assertIn("Second request.", second[-1].content)
        self.assertNotIn("Hi! How can I help you today?", second[-1].content)
        self.assertIn("speech-recognition transcript", second[-1].content)
        await harness.close()

    async def test_barge_in_repairs_the_backend_copy_too(self) -> None:
        frontend = FakeChatClient([delegate_response("Confirm the booking.")])
        backend = FakeChatClient([text_response(ANSWER)])
        harness = SessionHarness(config=_history_config(), clients=AgentClients(backend=backend, frontend=frontend))
        await harness.start(tau2_session_update())
        await harness.speak()
        await harness.wait_for("response.done")
        await harness.feed(pcmu_silence(300))
        await harness.feed(pcmu_speech(300))
        await harness.wait_for("input_audio_buffer.speech_started", 2)
        state = harness.runners[0].state
        stored = state.backend_history.messages[-1].content
        self.assertTrue(stored.startswith("Your booking is confirmed."))
        self.assertTrue(stored.endswith(MARKER))
        self.assertEqual(stored, state.frontend_history.messages[-1].content)
        await harness.close()

    async def test_cancel_while_thinking_leaves_the_backend_history_alone(self) -> None:
        frontend = FakeChatClient([delegate_response("First request."), delegate_response("Merged request.")])
        backend = FakeChatClient([text_response("Merged answer.")])
        backend.gates[0] = asyncio.Event()  # the first backend call hangs and is cancelled
        harness = SessionHarness(config=_history_config(), clients=AgentClients(backend=backend, frontend=frontend))
        await harness.start(tau2_session_update())
        await harness.speak()
        await asyncio.sleep(0.05)
        self.assertEqual(harness.session.turns.state, "THINKING")
        await harness.speak()
        await harness.wait_for("response.done")
        history = harness.runners[0].state.backend_history.messages
        self.assertEqual([m.role for m in history], ["user", "assistant"])
        self.assertIn("utterance 1 utterance 2", history[0].content)
        self.assertEqual(history[1].content, "Merged answer.")
        await harness.close()

    async def test_session_start_and_backend_context_record_the_arm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            frontend = FakeChatClient([delegate_response("Request.")])
            backend = FakeChatClient([text_response("Answer.")])
            harness = SessionHarness(
                config=_history_config(guidance_key=""),
                clients=AgentClients(backend=backend, frontend=frontend),
                event_log=EventLog(str(path)),
            )
            await harness.start(tau2_session_update())
            await harness.speak()
            await harness.wait_for("response.done")
            await harness.close()
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        start = next(record for record in records if record["kind"] == "session_start")
        self.assertEqual((start["backend_history"], start["backend_history_guidance"]), ("full", ""))
        context = next(record for record in records if record["kind"] == "backend_context")
        self.assertEqual((context["enabled"], context["include"], context["guidance"]), (True, "full", ""))

    async def test_rendered_prompts_end_with_the_history_note_after_the_cascade_addendum(self) -> None:
        backend = FakeChatClient([text_response("Answer.")])
        harness = SessionHarness(
            config=_history_config(), clients=AgentClients(backend=backend, frontend=FakeChatClient())
        )
        await harness.start(tau2_update_with([], "Policy text."))
        prompt = harness.runners[0].rendered_prompts()["backend"]
        cascade = prompt.index("Voice cascade notes.")
        note = prompt.index("Conversation so far. This is an ongoing conversation")
        guidance = prompt.index("Working with the conversation history:")
        self.assertLess(cascade, note)
        self.assertLess(note, guidance)
        await harness.close()


class ProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_history_profile_differs_from_tau3_eval_only_by_the_flag(self) -> None:
        base = load_voice_config(PROFILES / "tau3_eval.yaml")
        hist = load_voice_config(PROFILES / "tau3_eval_backend_history.yaml")
        for field in fields(base):
            if field.name not in ("agent", "source_files", "resolved_in_paths", "warnings"):
                self.assertEqual(getattr(base, field.name), getattr(hist, field.name), field.name)
        self.assertEqual(replace(base.agent, backend=None), replace(hist.agent, backend=None))
        differing = {
            field.name
            for field in fields(base.agent.backend)
            if getattr(base.agent.backend, field.name) != getattr(hist.agent.backend, field.name)
        }
        self.assertEqual(differing, {"conversation_history", "stateful"})
        self.assertEqual(hist.agent.backend.conversation_history.effective_include, "full")
        self.assertTrue(hist.agent.backend.stateful)

    def test_noguide_profile_differs_from_the_history_profile_only_by_guidance(self) -> None:
        hist = load_voice_config(PROFILES / "tau3_eval_backend_history.yaml")
        noguide = load_voice_config(PROFILES / "tau3_eval_backend_history_noguide.yaml")
        history = hist.agent.backend.conversation_history
        self.assertEqual(noguide.agent.backend.conversation_history, replace(history, guidance_key=""))
        self.assertEqual(
            replace(noguide.agent, backend=replace(noguide.agent.backend, conversation_history=history)), hist.agent
        )
        for field in fields(hist):
            if field.name not in ("agent", "source_files", "resolved_in_paths", "warnings"):
                self.assertEqual(getattr(hist, field.name), getattr(noguide, field.name), field.name)

    def test_eval_and_backend_only_profiles_ignore_the_environment_default(self) -> None:
        with mock.patch.dict(os.environ, {"FBA_BACKEND_HISTORY": "true"}):
            self.assertTrue(load_voice_config(PACKAGE_DIR / "config" / "voice_agent.yaml").agent.backend.stateful)
            for name in ("tau3_eval.yaml", "backend_only.yaml"):
                config = load_voice_config(PROFILES / name)
                self.assertFalse(config.agent.backend.conversation_history.enabled, name)
