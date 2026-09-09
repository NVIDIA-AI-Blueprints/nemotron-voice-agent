# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from examples.omni_assistant.nvidia_omni_multimodal_service import NvidiaOmniInferenceResult
from examples.omni_assistant_subagents.subagents.transport.agent import OmniTransportAgent
from examples.omni_assistant_subagents.subagents.transport.webcam_controller import WebcamController
from examples.omni_assistant_subagents.subagents.webcam.agent import WebcamAgent, _steering_preamble
from webcam_frame_store import clear_session_webcam_frames, recent_webcam_frames, store_webcam_frame


def _controller(provider):
    return WebcamController(
        session_id="s",
        board=Mock(),
        request_job=AsyncMock(),
        queue_frame=AsyncMock(),
        conversation_provider=provider,
    )


class ConversationContextTests(unittest.TestCase):
    def test_returns_provider_output_stripped(self) -> None:
        controller = _controller(lambda: "  User: what is this?\nAssistant: a camera  ")
        self.assertEqual(controller._conversation_context(), "User: what is this?\nAssistant: a camera")

    def test_empty_when_no_provider(self) -> None:
        self.assertEqual(_controller(None)._conversation_context(), "")

    def test_empty_when_provider_raises(self) -> None:
        def boom() -> str:
            raise RuntimeError("no context")

        self.assertEqual(_controller(boom)._conversation_context(), "")


class VisualStatusTests(unittest.TestCase):
    def test_camera_off_when_disabled(self) -> None:
        controller = _controller(lambda: "")
        self.assertIn("OFF", controller.current_visual_status())

    def test_loading_when_on_without_observation(self) -> None:
        controller = _controller(lambda: "")
        controller._enabled = True
        self.assertIn("loading", controller.current_visual_status())

    def test_reports_latest_observation_when_live(self) -> None:
        controller = _controller(lambda: "")
        controller._enabled = True
        controller._board_state = "a GoPro and a small tripod"
        self.assertEqual(controller.current_visual_status(), "a GoPro and a small tripod")


class WebcamStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_uploaded_frame_starts_continuous_uploads(self) -> None:
        controller = _controller(lambda: "")
        controller._start_continuous_uploads = AsyncMock()

        controller._notify_frame_uploaded()
        await asyncio.sleep(0)

        self.assertTrue(controller._enabled)
        controller._start_continuous_uploads.assert_awaited_once()


class PreviousObservationTests(unittest.TestCase):
    def test_empty_when_camera_is_off_or_loading(self) -> None:
        controller = _controller(lambda: "")
        self.assertEqual(controller._previous_observation(), "")
        controller._board_state = "the camera just turned on; the live view is loading"
        self.assertEqual(controller._previous_observation(), "")

    def test_returns_last_published_scene_note(self) -> None:
        controller = _controller(lambda: "")
        controller._board_state = "The person holds up a white bottle with a purple label."
        self.assertEqual(
            controller._previous_observation(),
            "The person holds up a white bottle with a purple label.",
        )


class SteeringPreambleTests(unittest.TestCase):
    def test_defaults_to_newest_frames_without_conversation(self) -> None:
        block = _steering_preamble("")
        self.assertNotIn("RECENT USER TURNS", block)
        self.assertNotIn("LAST NOTE", block)
        self.assertIn("ONLY source of truth for the observation", block)
        self.assertIn("Do not name an object that left those newest frames", block)
        self.assertIn("never invent a phone, bottle", block)
        self.assertIn("final roughly 1.5 seconds", block)

    def test_empty_scene_does_not_invent_a_person(self) -> None:
        block = _steering_preamble("")
        self.assertIn("check whether a person is clearly visible", block)
        self.assertIn("describe only the actual room, furniture, or other surroundings", block)
        self.assertIn("never invent a person, face, pose, gaze, expression, or activity", block)

    def test_absent_things_are_never_narrated(self) -> None:
        block = _steering_preamble("")
        self.assertIn("never report anything as absent, missing, hidden, or not visible", block)
        self.assertIn("hands not visible", block)
        self.assertNotIn("Empty hands", block)

    def test_user_turns_are_included_and_grounded(self) -> None:
        block = _steering_preamble("User: what am I holding?")
        self.assertIn("RECENT USER TURNS", block)
        self.assertIn("what am I holding", block)
        self.assertIn("ONLY if it is still visible", block)
        self.assertNotIn("RECENT CONVERSATION", block)

    def test_previous_note_is_stale_unless_still_visible(self) -> None:
        block = _steering_preamble("", "The person holds up a white bottle with a purple label.")
        self.assertIn("LAST NOTE", block)
        self.assertIn("white bottle", block)
        self.assertIn("it is GONE", block)
        self.assertIn("drop it silently", block)
        self.assertIn("Never announce that it left", block)


class RecentConversationTests(unittest.TestCase):
    def test_omits_assistant_visual_claims_and_gesture_cues(self) -> None:
        cue = "(Visual cue, no audio: the user just waved and greeted you on camera.)"
        agent = object.__new__(OmniTransportAgent)
        agent._proactive_directives = {"proactive_greet": cue}
        agent._context = SimpleNamespace(
            get_messages=lambda: [
                {"role": "user", "content": "So can you tell me what I am holding right now?"},
                {"role": "assistant", "content": "You are holding a white bottle with a purple label."},
                {"role": "user", "content": cue},
            ]
        )
        text = OmniTransportAgent._recent_conversation(agent)
        self.assertIn("what I am holding", text)
        self.assertNotIn("bottle", text)
        self.assertNotIn("Visual cue", text)
        self.assertNotIn("Assistant:", text)


class WebcamOutputValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_malformed_worker_output_is_rejected(self) -> None:
        worker = object.__new__(WebcamAgent)
        worker._base_url = "http://localhost:8002/v1"
        worker._model_id = "test-model"
        worker._system_prompt = "Return JSON."
        worker._prompt = "Describe the video."
        worker._max_tokens = 128
        worker._temperature = 0.2
        worker._omni = AsyncMock()
        worker._omni.run_multimodal_inference.return_value = NvidiaOmniInferenceResult(text="not json")

        observation, visual_control, focus = await worker._describe(b"mp4", 2, 8.0)

        self.assertEqual(observation, "")
        self.assertEqual(visual_control["intent"], "none")
        self.assertEqual(focus, "")

    async def test_focus_is_parsed_from_worker_output(self) -> None:
        worker = object.__new__(WebcamAgent)
        worker._base_url = "http://localhost:8002/v1"
        worker._model_id = "test-model"
        worker._system_prompt = "Return JSON."
        worker._prompt = "Describe the video."
        worker._max_tokens = 128
        worker._temperature = 0.2
        worker._omni = AsyncMock()
        worker._omni.run_multimodal_inference.return_value = NvidiaOmniInferenceResult(
            text='{"observation":"holding a camera","focus":"camera","visual_control":{"intent":"none"}}'
        )

        observation, _, focus = await worker._describe(b"mp4", 2, 2.0)

        self.assertEqual(observation, "holding a camera")
        self.assertEqual(focus, "camera")

    async def test_previous_observation_is_steered_into_the_prompt(self) -> None:
        worker = object.__new__(WebcamAgent)
        worker._base_url = "http://localhost:8002/v1"
        worker._model_id = "test-model"
        worker._system_prompt = "Return JSON."
        worker._prompt = "Describe the video."
        worker._max_tokens = 128
        worker._temperature = 0.2
        worker._omni = AsyncMock()
        worker._omni.run_multimodal_inference.return_value = NvidiaOmniInferenceResult(
            text='{"observation":"empty hands","focus":"","visual_control":{"intent":"none"}}'
        )

        await worker._describe(
            b"mp4",
            2,
            2.0,
            previous_observation="The person holds up a white bottle with a purple label.",
        )

        context = worker._omni.run_multimodal_inference.await_args.args[0]
        user_text = context.messages[1]["content"][1]["text"]
        self.assertIn("LAST NOTE", user_text)
        self.assertIn("white bottle", user_text)
        self.assertIn("it is GONE", user_text)

    def test_non_finite_frame_window_does_not_raise(self) -> None:
        store_webcam_frame(
            session_id="non-finite-window",
            name="frame.jpg",
            content_type="image/jpeg",
            data=b"frame",
        )
        try:
            frames = recent_webcam_frames("non-finite-window", max_seconds=float("inf"))
            self.assertEqual(len(frames), 1)
        finally:
            clear_session_webcam_frames("non-finite-window")


if __name__ == "__main__":
    unittest.main()
