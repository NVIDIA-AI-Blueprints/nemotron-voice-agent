# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""The tau2 client contract on the τ³ head, driven through a real session and the stub-gate runtime.

The stub gate is the no-GPU τ³ gate setup: rule-based frontend, an in-process fake
gateway whose backend calls the session's first tool once per run, stub ASR, tone TTS.
"""

from __future__ import annotations

import base64
import unittest

from _voice_fakes import SessionHarness, pcmu_silence, stub_gate_config, stub_runtime, tau2_session_update


def output_event(call_id: str, output: str) -> dict:
    return {
        "type": "conversation.item.create",
        "item": {"type": "function_call_output", "call_id": call_id, "output": output},
    }


def harness() -> SessionHarness:
    runtime = stub_runtime()
    return SessionHarness(config=stub_gate_config().voice, turn_manager_factory=runtime.turn_manager)


class HandshakeTests(unittest.IsolatedAsyncioTestCase):
    async def test_handshake_with_tau2_payload(self) -> None:
        for domain in ("mock", "airline", "retail", "telecom"):
            with self.subTest(domain=domain):
                session = harness()
                await session.start(tau2_session_update(domain))
                self.assertEqual(session.events[0]["type"], "session.created")
                self.assertTrue(session.events[0]["session"]["id"].startswith("sess_"))
                self.assertEqual(session.events[1]["type"], "session.updated")
                self.assertEqual(session.of_type("error"), [])
                await session.close()

    async def test_no_greeting(self) -> None:
        session = harness()
        await session.start(tau2_session_update())
        await session.idle(1500)
        self.assertEqual(session.of_type("response.created"), [])
        await session.speak(then_silence_ms=1200)
        await session.wait_for("response.output_audio.delta")
        first_audio = session.types().index("response.output_audio.delta")
        self.assertGreater(first_audio, session.types().index("input_audio_buffer.speech_stopped"))
        await session.close()


class FunctionCallTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.session = harness()
        await self.session.start(tau2_session_update("mock"))

    async def asyncTearDown(self) -> None:
        await self.session.close()

    async def test_round_trip_preserves_call_id_and_acknowledges_output_once(self) -> None:
        tools = [tool["name"] for tool in tau2_session_update("mock")["session"]["tools"]]
        await self.session.speak(then_silence_ms=1200)
        done = await self.session.wait_for("response.function_call_arguments.done")
        self.assertEqual(done["name"], tools[0])
        self.assertTrue(done["call_id"])
        await self.session.wait_for("response.done")
        await self.session.send(output_event(done["call_id"], '{"ok": true}'))
        await self.session.send({"type": "response.create"})
        await self.session.idle(3000)
        added = self.session.of_type("conversation.item.added")
        acks = [e for e in added if e["item"]["type"] == "function_call_output"]
        self.assertEqual(len(acks), 1)
        self.assertEqual(acks[0]["item"]["call_id"], done["call_id"])
        self.assertEqual(self.session.of_type("error"), [])

    async def test_unknown_call_id_is_a_non_fatal_error(self) -> None:
        await self.session.speak(then_silence_ms=1200)
        await self.session.wait_for("response.function_call_arguments.done")
        await self.session.send(output_event("call_unknown", "x"))
        error = await self.session.wait_for("error")
        self.assertEqual(error["error"]["param"], "item.call_id")


class Tau2TimingTests(unittest.IsolatedAsyncioTestCase):
    async def test_truncate_is_clamped_not_rejected(self) -> None:
        session = harness()
        await session.start(tau2_session_update())
        await session.speak(then_silence_ms=1200)
        await session.wait_for("response.output_audio.delta")
        await session.idle(500)
        item_id = session.of_type("response.output_audio.delta")[0]["item_id"]
        await session.send(
            {"type": "conversation.item.truncate", "item_id": item_id, "content_index": 0, "audio_end_ms": 999999}
        )
        truncated = await session.wait_for("conversation.item.truncated")
        span = session.session.turns.progress.item(item_id)
        self.assertEqual(truncated["audio_end_ms"], int(span.end_ms - span.start_ms))
        self.assertEqual(session.of_type("error"), [])
        await session.close()

    async def test_pcmu_out_in_100_ms_deltas(self) -> None:
        session = harness()
        await session.start(tau2_session_update())
        await session.speak(then_silence_ms=1200)
        await session.wait_for("response.output_audio.delta")
        audio = base64.b64decode(session.of_type("response.output_audio.delta")[0]["delta"])
        self.assertEqual(len(audio), 800)  # 100 ms of 8 kHz mu-law per delta
        await session.feed(pcmu_silence(200))
        await session.close()

    async def test_transcript_delta_precedes_its_audio(self) -> None:
        session = harness()
        await session.start(tau2_session_update())
        await session.speak(then_silence_ms=1200)
        await session.wait_for("response.output_audio.delta")
        await session.idle(3000)
        seen_text: set[str] = set()
        for event in session.events:
            if event["type"] == "response.output_audio_transcript.delta":
                seen_text.add(event["item_id"])
            elif event["type"] == "response.output_audio.delta":
                self.assertIn(event["item_id"], seen_text, "audio before its transcript")
        await session.close()


if __name__ == "__main__":
    unittest.main()
