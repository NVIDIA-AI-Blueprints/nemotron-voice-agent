# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from pipecat.frames.frames import LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext

from examples.omni_assistant_subagents.subagents.transport.media_analysis_controller import MediaAnalysisController
from examples.omni_assistant_subagents.subagents.transport.subagent_state_board import SubagentStateBoard
from examples.omni_assistant_subagents.subagents.transport.thinking_controller import ThinkingController
from examples.shared.subagents import SubagentSpec


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

    def test_pending_board_names_the_uploaded_kind(self) -> None:
        board = Mock()
        controller = MediaAnalysisController(
            session_id="session",
            speaker_context=Mock(),
            board=board,
            request_job=AsyncMock(),
            queue_frame=AsyncMock(),
            followup_delay_secs=0,
        )

        for kind, expected in (("image", "an image file"), ("video", "a video file"), ("audio", "an audio file")):
            with patch(
                "examples.omni_assistant_subagents.subagents.transport.media_analysis_controller.latest_user_attachment",
                return_value=SimpleNamespace(id=kind, kind=kind),
            ):
                controller.mark_attachment_pending()
            self.assertIn(f"the user just uploaded {expected}", board.set_findings.call_args.args[1])

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


class StateBoardRenderingTests(unittest.TestCase):
    def _board(self) -> tuple[SubagentStateBoard, dict[str, str]]:
        pinned: dict[str, str] = {}
        registry = SimpleNamespace(
            specs=lambda: [
                SubagentSpec(
                    key="omni_webcam",
                    label="Webcam Vision",
                    capability="Describes the live webcam view.",
                    findings_label="what you currently see",
                    routing_rules="This is your live eyes.",
                )
            ]
        )
        context = SimpleNamespace(set_pinned_state=lambda prefix, text: pinned.__setitem__(prefix, text))
        return SubagentStateBoard(registry=registry, speaker_context=context), pinned

    def _text(self, pinned: dict[str, str]) -> str:
        return next(iter(pinned.values()))

    def test_no_subagent_is_ever_reported_as_switchable(self) -> None:
        board, pinned = self._board()

        board.set_findings("omni_webcam", "a GoPro on a tripod")

        self.assertNotIn("status:", self._text(pinned))

    def test_our_own_state_is_stated_as_fact(self) -> None:
        board, pinned = self._board()

        board.set_findings("omni_webcam", "the camera is OFF right now", trusted=True)

        text = self._text(pinned)
        self.assertIn("what you currently see: the camera is OFF right now", text)
        self.assertNotIn("what you currently see_untrusted_data_json", text)

    def test_subagent_output_stays_quoted_as_untrusted(self) -> None:
        board, pinned = self._board()

        board.set_findings("omni_webcam", "a GoPro on a tripod")

        self.assertIn('what you currently see_untrusted_data_json: "a GoPro on a tripod"', self._text(pinned))

    def test_appended_patch_carries_the_subagent_voice(self) -> None:
        board, pinned = self._board()

        board.set_findings("omni_webcam", "the camera is OFF right now", trusted=True)
        board.append_findings("omni_webcam", "a GoPro on a tripod")

        self.assertIn("what you currently see_untrusted_data_json", self._text(pinned))


class ThinkingInvalidationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.queue_frame = AsyncMock()
        self.cancel_job = AsyncMock()
        self.user_turn_open = False
        self.controller = ThinkingController(
            context=LLMContext(messages=[]),
            request_job=AsyncMock(return_value="task-1"),
            queue_frame=self.queue_frame,
            followup_delay_secs=0,
            cancel_job=self.cancel_job,
            user_turn_open=lambda: self.user_turn_open,
        )
        self.controller.queue("Think about this")
        await self.controller.start_pending()
        self.queue_frame.reset_mock()

    @staticmethod
    def _answer(text: str = "stale answer") -> SimpleNamespace:
        return SimpleNamespace(job_id="task-1", response={"response": text}, source="omni_thinker")

    def _spoken(self) -> list[str]:
        return [
            call.args[0].text for call in self.queue_frame.await_args_list if isinstance(call.args[0], LLMTextFrame)
        ]

    async def test_active_thinker_response_is_ignored_after_interruption(self) -> None:
        self.controller.clear_pending()
        handled = await self.controller.handle_job_response(self._answer())

        self.assertTrue(handled)
        self.queue_frame.assert_not_awaited()

    async def test_answer_is_spoken_when_the_user_does_not_move_on(self) -> None:
        await self.controller.handle_job_response(self._answer("five cents"))

        self.assertEqual(self._spoken(), ["five cents"])

    async def test_new_user_turn_cancels_the_thinker_job_and_drops_its_answer(self) -> None:
        await self.controller.cancel_active("new user turn")
        await self.controller.handle_job_response(self._answer())

        self.cancel_job.assert_awaited_once_with("task-1", reason="new user turn")
        self.assertEqual(self._spoken(), [])

    async def test_a_cancelled_thinker_card_reads_cancelled_not_failed(self) -> None:
        await self.controller.cancel_active("new user turn")

        update = self.queue_frame.await_args_list[-1].args[0].data
        self.assertEqual(update["status"], "cancelled")

    async def test_new_user_turn_during_dispatch_cancels_the_job_once_it_starts(self) -> None:
        dispatched = asyncio.Event()
        release = asyncio.Event()

        async def request_job(*_args, **_kwargs) -> str:
            dispatched.set()
            await release.wait()
            return "task-2"

        self.controller.clear_pending()
        self.controller._request_job = request_job
        self.controller.queue("Think about that")
        start = asyncio.create_task(self.controller.start_pending())
        await dispatched.wait()

        await self.controller.cancel_active("new user turn")
        release.set()
        await start
        await self.controller.handle_job_response(SimpleNamespace(job_id="task-2", response={"response": "stale"}))

        self.cancel_job.assert_awaited_once_with("task-2", reason="superseded during dispatch")
        self.assertEqual(self._spoken(), [])

    async def test_answer_arriving_during_a_user_turn_waits_for_that_turn(self) -> None:
        self.user_turn_open = True
        response_task = asyncio.create_task(self.controller.handle_job_response(self._answer()))
        await asyncio.sleep(0.2)
        self.assertEqual(self._spoken(), [])

        await self.controller.cancel_active("new user turn")
        self.user_turn_open = False
        await response_task

        self.assertEqual(self._spoken(), [])

    async def test_cancel_without_an_active_job_does_nothing(self) -> None:
        self.controller.clear_pending()

        await self.controller.cancel_active("new user turn")

        self.cancel_job.assert_not_awaited()
        self.queue_frame.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
