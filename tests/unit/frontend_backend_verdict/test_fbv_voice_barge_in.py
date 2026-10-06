# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Rules 11, 11a, 12: barge-in cancels generation, repairs history, and respects tool calls."""

from __future__ import annotations

import asyncio
import unittest
from collections.abc import AsyncIterator

from _fbv_voice_fakes import (
    FakeChatClient,
    SessionHarness,
    delegate_response,
    pcmu_silence,
    pcmu_speech,
    tau2_session_update,
    text_response,
    tool_response,
)

from examples.frontend_backend_verdict.voice.agent.runner import AgentClients
from examples.frontend_backend_verdict.voice.audio.pcm import tone
from examples.frontend_backend_verdict.voice.speech.stubs import StubRecognizer, ToneSynthesizer

MARKER = " [interrupted by the user]"
ANSWER = "Your booking is confirmed. The total is two hundred dollars. Anything else?"


class BlockingSynthesizer(ToneSynthesizer):
    """Tone audio, but blocks forever on sentences containing ``block_on``."""

    def __init__(self, block_on: str) -> None:
        super().__init__(sample_rate=16000, ms_per_char=10.0)
        self.block_on = block_on
        self.started: list[str] = []

    async def synthesize(self, text: str, *, voice: str | None = None) -> AsyncIterator[bytes]:
        self.started.append(text)
        if self.block_on in text:
            try:
                await asyncio.Event().wait()
            except (asyncio.CancelledError, GeneratorExit):
                self.cancelled.append(text)
                raise
        for _ in range(3):
            yield tone(100, 16000)


def paired(harness_kwargs: dict | None = None, **clients) -> tuple[SessionHarness, FakeChatClient, FakeChatClient]:
    frontend, backend = FakeChatClient(), FakeChatClient()
    harness = SessionHarness(clients=AgentClients(backend=backend, frontend=frontend), **(harness_kwargs or {}))
    return harness, frontend, backend


class BargeInWhileSpeakingTests(unittest.IsolatedAsyncioTestCase):
    async def test_barge_in_repairs_both_paired_copies(self) -> None:
        harness, frontend, backend = paired()
        frontend.queue(delegate_response("Confirm the booking."))
        backend.queue(text_response(ANSWER))
        await harness.start(tau2_session_update())
        await harness.speak()
        await harness.wait_for("response.done")
        progress = harness.session.turns.progress
        self.assertGreater(progress.sent_ms, 0)
        # The client plays ~first sentence (260 ms of tone at 10 ms/char) plus a bit, then the user talks.
        await harness.feed(pcmu_silence(300))
        heard_before = progress.heard_ms
        await harness.feed(pcmu_speech(300))
        await harness.wait_for("input_audio_buffer.speech_started", 2)
        runner = harness.runners[0]
        group = runner.state.frontend_history.messages[-4:]
        self.assertEqual([m.role for m in group], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(group[2].content, group[3].content)
        self.assertTrue(group[3].content.startswith("Your booking is confirmed."))
        self.assertTrue(group[3].content.endswith(MARKER.strip()))
        self.assertNotIn("Anything else", group[3].content)
        self.assertNotIn("Anything else", group[2].content)
        self.assertGreater(heard_before, 0)
        await harness.close()

    async def test_barge_in_during_generation_cancels_the_response(self) -> None:
        synth = BlockingSynthesizer(block_on="Anything else")
        harness, frontend, backend = paired({"synthesizer": synth})
        frontend.queue(delegate_response("Confirm the booking."))
        backend.queue(text_response(ANSWER))
        await harness.start(tau2_session_update())
        await harness.speak()
        await harness.wait_for("response.output_audio.delta")
        await harness.feed(pcmu_silence(1000))  # sentences 1-2 fully heard; sentence 3 still synthesizing
        progress = harness.session.turns.progress
        self.assertEqual(progress.heard_ms, progress.sent_ms)
        self.assertTrue(progress.active)
        item_id = harness.of_type("response.output_audio.delta")[0]["item_id"]
        deltas_before = len(harness.of_type("response.output_audio.delta"))
        await harness.feed(pcmu_speech(300))
        done = await harness.wait_for("response.done")
        self.assertEqual(done["response"]["status"], "cancelled")
        self.assertEqual(done["response"]["status_details"]["reason"], "turn_detected")
        item_done = next(e for e in harness.of_type("response.output_item.done") if e["item"]["id"] == item_id)
        self.assertEqual(item_done["item"]["status"], "incomplete")
        self.assertIn(ANSWER.split(". ")[-1], synth.cancelled)
        types = harness.types()
        self.assertLess(types.index("input_audio_buffer.speech_started", 3), types.index("response.done"))
        await harness.feed(pcmu_silence(700))
        self.assertEqual(len(harness.of_type("response.output_audio.delta")), deltas_before)
        stored = harness.runners[0].state.frontend_history.messages[-1].content
        self.assertEqual(stored, "Your booking is confirmed. The total is two hundred dollars." + MARKER)
        await harness.close()

    async def test_cancels_generation_with_nothing_sent(self) -> None:
        synth = ToneSynthesizer(sample_rate=16000, gate=asyncio.Event())
        harness, frontend, backend = paired({"synthesizer": synth})
        frontend.queue(delegate_response("Confirm."))
        backend.queue(text_response(ANSWER))
        await harness.start(tau2_session_update())
        await harness.speak()
        await harness.wait_for("response.output_item.added")
        self.assertEqual(harness.session.turns.progress.sent_ms, 0)
        await harness.feed(pcmu_speech(300))
        done = await harness.wait_for("response.done")
        self.assertEqual(done["response"]["status"], "cancelled")
        self.assertEqual(harness.of_type("response.output_audio.delta"), [])
        stored = harness.runners[0].state.frontend_history.messages[-1].content
        self.assertEqual(stored, MARKER.strip())
        await harness.close()


class WhileThinkingTests(unittest.IsolatedAsyncioTestCase):
    async def test_speech_while_thinking_cancels_and_merges(self) -> None:
        harness, frontend, backend = paired()
        frontend.queue(delegate_response("First request."), delegate_response("Merged request."))
        backend.queue(text_response("Merged answer."))
        backend.gates[0] = asyncio.Event()  # the first backend call hangs and is cancelled before it returns
        await harness.start(tau2_session_update())
        await harness.speak()
        await asyncio.sleep(0.05)
        self.assertEqual(harness.session.turns.state, "THINKING")
        await harness.speak()
        await harness.wait_for("response.done")
        self.assertEqual(frontend.calls[1]["messages"][-1].content, "utterance 1 utterance 2")
        runner = harness.runners[0]
        users = [m.content for m in runner.state.frontend_history.messages if m.role == "user"]
        self.assertEqual(users, ["utterance 1 utterance 2"])  # the cancelled turn left no trace
        transcript = "".join(e["delta"] for e in harness.of_type("response.output_audio_transcript.delta"))
        self.assertEqual(transcript, "Merged answer.")
        await harness.close()

    async def test_speech_after_a_tool_call_left_is_queued_never_cancelled(self) -> None:
        harness, frontend, backend = paired()
        frontend.queue(delegate_response("Look up the user."), delegate_response("Second question."))
        backend.queue(
            tool_response(("get_users", {}), ids=["call_a"]),
            text_response("Found them."),
            text_response("Second answer."),
        )
        await harness.start(tau2_session_update())
        await harness.speak()
        await harness.wait_for("response.done")
        self.assertEqual(harness.session.turns.state, "AWAITING_TOOLS")
        await harness.speak()  # user talks while tools are outstanding
        self.assertEqual(harness.session.turns.state, "AWAITING_TOOLS")
        await harness.send(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": "call_a", "output": "ok"},
            }
        )
        await harness.send({"type": "response.create"})
        await harness.wait_for("response.done", 3)
        statuses = [e["response"]["status"] for e in harness.of_type("response.done")]
        self.assertEqual(statuses, ["completed", "completed", "completed"])
        self.assertEqual(frontend.calls[1]["messages"][-1].content, "utterance 2")
        await harness.close()

    async def test_noise_shorter_than_min_speech_never_interrupts(self) -> None:
        harness = SessionHarness(
            clients=AgentClients(
                backend=FakeChatClient([text_response(ANSWER)]), frontend=FakeChatClient([text_response(ANSWER)])
            ),
            recognizer=StubRecognizer(),
        )
        await harness.start(tau2_session_update())
        await harness.speak()
        await harness.wait_for("response.done")
        await harness.feed(pcmu_speech(60) + pcmu_silence(600))  # a click
        self.assertEqual(len(harness.of_type("input_audio_buffer.speech_started")), 1)
        await harness.close()
