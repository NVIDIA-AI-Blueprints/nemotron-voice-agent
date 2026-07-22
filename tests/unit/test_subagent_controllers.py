# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from pipecat.processors.aggregators.llm_context import LLMContext

from examples.omni_assistant_subagents.subagents.transport.media_analysis_controller import MediaAnalysisController
from examples.omni_assistant_subagents.subagents.transport.thinking_controller import ThinkingController


class MediaAnalysisBindingTests(unittest.IsolatedAsyncioTestCase):
    async def test_pending_prompt_is_not_dispatched_to_a_newer_attachment(self) -> None:
        board = Mock()
        board.is_available.return_value = True
        request_job = AsyncMock()
        controller = MediaAnalysisController(
            session_id="session",
            speaker_context=Mock(),
            board=board,
            request_job=request_job,
            queue_frame=AsyncMock(),
            followup_delay_secs=0,
        )
        first = SimpleNamespace(id="first", metadata=lambda: {"id": "first"})
        second = SimpleNamespace(id="second", metadata=lambda: {"id": "second"})

        with patch(
            "examples.omni_assistant_subagents.subagents.transport.media_analysis_controller.latest_user_attachment",
            return_value=first,
        ):
            await controller.queue_prompt(
                "Describe it",
                "Describe the upload",
                "new",
                "uploaded_attachment",
            )

        with patch(
            "examples.omni_assistant_subagents.subagents.transport.media_analysis_controller.latest_user_attachment",
            return_value=second,
        ):
            await controller.start_pending()

        request_job.assert_not_awaited()

    async def test_rerun_uses_full_analysis_mode(self) -> None:
        board = Mock()
        board.get_findings.return_value = "previous analysis"
        request_job = AsyncMock(return_value="task-1")
        controller = MediaAnalysisController(
            session_id="session",
            speaker_context=Mock(),
            board=board,
            request_job=request_job,
            queue_frame=AsyncMock(),
            followup_delay_secs=0,
        )
        attachment = SimpleNamespace(id="first", metadata=lambda: {"id": "first"})
        controller._analyzed_attachment_id = "first"

        with patch(
            "examples.omni_assistant_subagents.subagents.transport.media_analysis_controller.latest_user_attachment",
            return_value=attachment,
        ):
            await controller.queue_prompt(
                "Analyze it again",
                "Analyze the upload again",
                "rerun",
                "uploaded_attachment",
            )
            await controller.start_pending()

        payload = request_job.await_args.kwargs["payload"]
        self.assertEqual(payload["prior_analysis"], "")

    async def test_failed_capture_dispatch_removes_temporary_attachment(self) -> None:
        controller = MediaAnalysisController(
            session_id="session",
            speaker_context=Mock(),
            board=Mock(),
            request_job=AsyncMock(side_effect=RuntimeError("failed")),
            queue_frame=AsyncMock(),
            followup_delay_secs=0,
        )

        with patch(
            "examples.omni_assistant_subagents.subagents.transport.media_analysis_controller.remove_attachment"
        ) as remove:
            await controller.analyze_capture({"id": "capture-1"}, "Read the label")

        remove.assert_called_once_with("session", "capture-1")


class ThinkingInvalidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_active_thinker_response_is_ignored_after_interruption(self) -> None:
        queue_frame = AsyncMock()
        request_job = AsyncMock(return_value="task-1")
        controller = ThinkingController(
            context=LLMContext(messages=[]),
            request_job=request_job,
            queue_frame=queue_frame,
            followup_delay_secs=0,
        )
        controller.queue("Think about this", effort="medium")
        await controller.start_pending()
        queue_frame.reset_mock()

        controller.clear_pending()
        handled = await controller.handle_job_response(
            SimpleNamespace(job_id="task-1", response={"response": "stale answer"}, source="omni_thinker")
        )

        self.assertTrue(handled)
        queue_frame.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
