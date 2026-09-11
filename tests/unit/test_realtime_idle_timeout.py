# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

from __future__ import annotations

import asyncio
import unittest

from realtime.controller import RealtimeSessionController
from realtime.idle_timeout import RealtimeServerVADIdleTimeout
from realtime.session import RealtimeSessionCapabilities


def _controller(
    *,
    idle_timeout_ms: int | None,
    input_transcription_model: str | None = None,
) -> RealtimeSessionController:
    controller = RealtimeSessionController(
        model="test-model",
        voice="test-voice",
        runtime_config={"pipeline_mode": "generic-assistant"},
        input_transcription_model=input_transcription_model,
        capabilities=RealtimeSessionCapabilities(voices=frozenset({"test-voice"})),
    )
    if idle_timeout_ms is not None:
        controller.apply_session_update(
            {
                "audio": {
                    "input": {
                        "turn_detection": {
                            "type": "server_vad",
                            "idle_timeout_ms": idle_timeout_ms,
                        }
                    }
                }
            }
        )
    return controller


def _response_done(*, output_type: str = "output_audio", status: str = "completed") -> dict:
    return {
        "type": "response.done",
        "response": {
            "id": "resp_test",
            "object": "realtime.response",
            "status": status,
            "output": [
                {
                    "id": "item_assistant",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": output_type}],
                }
            ],
        },
    }


class _IdleTimeoutHarness:
    def __init__(
        self,
        controller: RealtimeSessionController | None = None,
        *,
        sleep_gate: asyncio.Event | None = None,
        trigger_gate: asyncio.Event | None = None,
    ) -> None:
        self.sleeps: list[float] = []
        self.triggers: list[tuple[int, int, int, int, str]] = []
        self.errors: list[Exception] = []
        self.trigger_entered = asyncio.Event()

        async def sleep(delay: float) -> None:
            self.sleeps.append(delay)
            if sleep_gate is not None:
                await sleep_gate.wait()

        async def trigger(
            start_ms: int,
            end_ms: int,
            generation: int,
            timeout_ms: int,
            assistant_item_id: str,
        ) -> None:
            self.trigger_entered.set()
            if trigger_gate is not None:
                await trigger_gate.wait()
            self.triggers.append((start_ms, end_ms, generation, timeout_ms, assistant_item_id))

        async def report_error(exc: Exception) -> None:
            self.errors.append(exc)

        self.timeout = RealtimeServerVADIdleTimeout(
            controller=controller or _controller(idle_timeout_ms=5_000),
            trigger_idle_turn=trigger,
            report_error=report_error,
            sleep=sleep,
        )


class RealtimeServerVADIdleTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_controller_builds_native_empty_audio_turn_and_response(self) -> None:
        controller = _controller(
            idle_timeout_ms=5_000,
            input_transcription_model="test-asr",
        )

        turn = controller.start_idle_timeout_turn(
            audio_start_ms=125,
            audio_end_ms=175,
        )

        self.assertEqual(
            [event["type"] for event in turn.events],
            [
                "input_audio_buffer.timeout_triggered",
                "input_audio_buffer.committed",
                "conversation.item.added",
                "conversation.item.done",
                "conversation.item.input_audio_transcription.completed",
                "response.created",
            ],
        )
        timeout_event, committed_event, added_event, done_event, transcript_event, response_event = turn.events
        self.assertEqual(timeout_event["item_id"], turn.item_id)
        self.assertEqual(timeout_event["audio_start_ms"], 125)
        self.assertEqual(timeout_event["audio_end_ms"], 175)
        self.assertEqual(committed_event["item_id"], turn.item_id)
        self.assertEqual(added_event["item"]["id"], turn.item_id)
        self.assertEqual(done_event["item"]["id"], turn.item_id)
        self.assertIsNone(done_event["item"]["content"][0]["transcript"])
        self.assertEqual(controller.conversation.item(turn.item_id)["content"][0]["transcript"], "")
        self.assertEqual(transcript_event["item_id"], turn.item_id)
        self.assertEqual(transcript_event["transcript"], "")
        self.assertEqual(transcript_event["usage"], {"type": "duration", "seconds": 0.05})
        self.assertEqual(response_event["response"]["id"], turn.response_id)
        self.assertEqual(controller.active_response_id, turn.response_id)
        self.assertEqual(turn.context_message, {"role": "user", "content": ""})

    async def test_audio_response_arms_and_uses_truthful_input_audio_offsets(self) -> None:
        harness = _IdleTimeoutHarness()
        harness.timeout.record_input_audio(sample_count=2_400, sample_rate=24_000)
        harness.timeout.observe_published_events([_response_done()])
        harness.timeout.record_input_audio(sample_count=1_200, sample_rate=24_000)
        await asyncio.sleep(0)

        self.assertEqual(harness.sleeps, [5.0])
        self.assertEqual([(args[0], args[1], args[4]) for args in harness.triggers], [(100, 150, "item_assistant")])
        self.assertEqual(harness.errors, [])
        self.assertFalse(harness.timeout.armed)

    async def test_user_speech_cancels_a_pending_timeout(self) -> None:
        release_sleep = asyncio.Event()
        harness = _IdleTimeoutHarness(sleep_gate=release_sleep)
        harness.timeout.observe_published_events([_response_done()])
        await asyncio.sleep(0)
        self.assertTrue(harness.timeout.armed)

        harness.timeout.observe_published_events([{"type": "input_audio_buffer.speech_started"}])
        release_sleep.set()
        await asyncio.sleep(0)

        self.assertFalse(harness.timeout.armed)
        self.assertEqual(harness.triggers, [])
        self.assertEqual(harness.errors, [])

    async def test_preceding_assistant_mutation_cancels_pending_timeout(self) -> None:
        for event_type in ("conversation.item.deleted", "conversation.item.truncated"):
            with self.subTest(event_type=event_type):
                release_sleep = asyncio.Event()
                harness = _IdleTimeoutHarness(sleep_gate=release_sleep)
                harness.timeout.observe_published_events([_response_done()])
                await asyncio.sleep(0)
                self.assertTrue(harness.timeout.armed)

                harness.timeout.observe_published_events([{"type": event_type, "item_id": "item_assistant"}])
                release_sleep.set()
                await asyncio.sleep(0)

                self.assertFalse(harness.timeout.armed)
                self.assertEqual(harness.triggers, [])
                self.assertEqual(harness.errors, [])

    async def test_user_speech_cancels_after_trigger_hook_entry(self) -> None:
        release_trigger = asyncio.Event()
        harness = _IdleTimeoutHarness(trigger_gate=release_trigger)
        harness.timeout.observe_published_events([_response_done()])
        await harness.trigger_entered.wait()

        harness.timeout.observe_published_events([{"type": "input_audio_buffer.speech_started"}])
        release_trigger.set()
        await asyncio.sleep(0)

        self.assertFalse(harness.timeout.armed)
        self.assertEqual(harness.triggers, [])
        self.assertEqual(harness.errors, [])

    async def test_null_timeout_and_non_audio_responses_do_not_arm(self) -> None:
        for controller, event in (
            (_controller(idle_timeout_ms=None), _response_done()),
            (_controller(idle_timeout_ms=5_000), _response_done(output_type="output_text")),
            (_controller(idle_timeout_ms=5_000), _response_done(status="cancelled")),
        ):
            with self.subTest(event=event):
                harness = _IdleTimeoutHarness(controller)
                harness.timeout.observe_published_events([event])
                self.assertFalse(harness.timeout.armed)
                self.assertEqual(harness.triggers, [])
                self.assertEqual(harness.errors, [])

    async def test_close_cancels_and_permanently_disables_timer(self) -> None:
        release_sleep = asyncio.Event()
        harness = _IdleTimeoutHarness(sleep_gate=release_sleep)
        harness.timeout.observe_published_events([_response_done()])
        await asyncio.sleep(0)
        harness.timeout.close()
        harness.timeout.observe_published_events([_response_done()])
        release_sleep.set()
        await asyncio.sleep(0)

        self.assertFalse(harness.timeout.armed)
        self.assertEqual(harness.triggers, [])
        self.assertEqual(harness.errors, [])


if __name__ == "__main__":
    unittest.main()
