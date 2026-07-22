# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock

from examples.omni_assistant.nvidia_omni_multimodal_service import NvidiaOmniInferenceResult
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
        status = controller.current_visual_status()
        self.assertIn("currently see", status)
        self.assertIn("GoPro", status)


class WebcamStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_uploaded_frame_starts_continuous_uploads(self) -> None:
        controller = _controller(lambda: "")
        controller._start_continuous_uploads = AsyncMock()

        controller._notify_frame_uploaded()
        await asyncio.sleep(0)

        self.assertTrue(controller._enabled)
        controller._start_continuous_uploads.assert_awaited_once()


class SteeringPreambleTests(unittest.TestCase):
    def test_defaults_to_user_activity_priority_without_conversation(self) -> None:
        block = _steering_preamble("")
        self.assertNotIn("RECENT CONVERSATION", block)
        self.assertIn("actively holding, showing, or doing", block)
        self.assertIn("describe ALL the items", block)
        self.assertIn("ONLY what is genuinely visible", block)

    def test_conversation_is_included_and_grounded(self) -> None:
        block = _steering_preamble("User: what am I holding?\nAssistant: a camera")
        self.assertIn("RECENT CONVERSATION", block)
        self.assertIn("what am I holding", block)
        self.assertIn("actively holding, showing, or doing", block)
        self.assertIn("ONLY what is genuinely visible", block)


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

        observation, _, focus = await worker._describe(b"mp4", 2, 8.0)

        self.assertEqual(observation, "holding a camera")
        self.assertEqual(focus, "camera")

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
