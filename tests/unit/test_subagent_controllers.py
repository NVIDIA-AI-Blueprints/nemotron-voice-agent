# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from pipecat.frames.frames import LLMConfigureOutputFrame, LLMFullResponseEndFrame, LLMTextFrame, MetricsFrame
from pipecat.metrics.metrics import LLMUsageMetricsData, ProcessingMetricsData, TTFBMetricsData
from pipecat.processors.aggregators.llm_context import LLMContext

from examples.omni_assistant_subagents.subagents.media_analyzer import MEDIA_ANALYZER_LLM_METRICS_PROCESSOR
from examples.omni_assistant_subagents.subagents.transport.agent import OmniTransportAgent
from examples.omni_assistant_subagents.subagents.transport.media_analysis_controller import MediaAnalysisController
from examples.omni_assistant_subagents.subagents.transport.subagent_state_board import SubagentStateBoard
from examples.omni_assistant_subagents.subagents.transport.thinking_controller import ThinkingController
from examples.shared.frames import LLMProviderCompletionReasonFrame
from examples.shared.subagents import SubagentSpec
from realtime.frames import RealtimeOwnedLLMFullResponseStartFrame


class MediaAnalysisBindingTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _controller(*, board=None, request_job=None, realtime_mode: bool = False):
        emit_response = AsyncMock(return_value=True)
        controller = MediaAnalysisController(
            session_id="session",
            speaker_context=Mock(),
            board=board if board is not None else Mock(),
            request_job=request_job if request_job is not None else AsyncMock(),
            queue_frame=AsyncMock(),
            emit_response=emit_response,
            followup_delay_secs=0,
            realtime_mode=realtime_mode,
        )
        return controller, emit_response

    async def _handle_response(self, response: dict, *, realtime_mode: bool):
        controller, emit_response = self._controller(realtime_mode=realtime_mode)
        message = SimpleNamespace(job_id="task-1", source="omni_media_analyzer", response=response)
        attachment = SimpleNamespace(id="attachment-1")
        with patch(
            "examples.omni_assistant_subagents.subagents.transport.media_analysis_controller.latest_user_attachment",
            return_value=attachment,
        ):
            handled = await controller.handle_job_response(message)
        return handled, emit_response

    async def test_pending_prompt_is_not_dispatched_to_a_newer_attachment(self) -> None:
        board = Mock()
        board.is_available.return_value = True
        request_job = AsyncMock()
        controller, _ = self._controller(board=board, request_job=request_job)
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

    async def test_followup_queues_real_llm_metrics_inside_its_response_lifecycle(self) -> None:
        handled, emit_response = await self._handle_response(
            {
                "tts": "The image contains a green square.",
                "analysis": "A green square appears on a white background.",
                "attachment": {"id": "attachment-1"},
                "finish_reason": "stop",
                "llm_metrics": {
                    "processor": MEDIA_ANALYZER_LLM_METRICS_PROCESSOR,
                    "model": "test-model",
                    "ttfb_seconds": 0.25,
                    "processing_seconds": 1.5,
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 7,
                        "total_tokens": 27,
                        "reasoning_tokens": 3,
                    },
                },
            },
            realtime_mode=True,
        )

        self.assertTrue(handled)
        frames = emit_response.await_args.args[0]
        self.assertIsInstance(frames[0], MetricsFrame)
        self.assertIsInstance(frames[1], LLMTextFrame)
        self.assertIsInstance(frames[2], LLMProviderCompletionReasonFrame)
        metrics = frames[0].data
        self.assertEqual(
            [type(metric) for metric in metrics], [TTFBMetricsData, ProcessingMetricsData, LLMUsageMetricsData]
        )
        self.assertEqual(metrics[0].value, 0.25)
        self.assertEqual(metrics[1].value, 1.5)
        self.assertEqual(metrics[2].value.prompt_tokens, 20)
        self.assertEqual(metrics[2].value.completion_tokens, 7)
        self.assertEqual(metrics[2].value.total_tokens, 27)
        self.assertEqual(metrics[2].value.reasoning_tokens, 3)

    async def test_followup_without_provider_usage_does_not_create_metrics(self) -> None:
        handled, emit_response = await self._handle_response(
            {
                "tts": "The analyzer request failed.",
                "attachment": {"id": "attachment-1"},
                "finish_reason": "stop",
            },
            realtime_mode=True,
        )

        self.assertTrue(handled)
        frames = emit_response.await_args.args[0]
        self.assertFalse(any(isinstance(frame, MetricsFrame) for frame in frames))

    async def test_rtvi_followup_keeps_existing_payload_and_frame_contract(self) -> None:
        handled, emit_response = await self._handle_response(
            {
                "tts": "The image contains a green square.",
                "analysis": "A green square appears on a white background.",
                "attachment": {"id": "attachment-1"},
                "llm_metrics": {
                    "processor": MEDIA_ANALYZER_LLM_METRICS_PROCESSOR,
                    "ttfb_seconds": 0.25,
                },
            },
            realtime_mode=False,
        )

        self.assertTrue(handled)
        frames = emit_response.await_args.args[0]
        self.assertEqual(len(frames), 1)
        self.assertIsInstance(frames[0], LLMTextFrame)

    async def test_realtime_followup_rejects_a_missing_provider_terminal(self) -> None:
        handled, emit_response = await self._handle_response(
            {
                "tts": "The image contains a green square.",
                "attachment": {"id": "attachment-1"},
            },
            realtime_mode=True,
        )

        self.assertFalse(handled)
        emit_response.assert_not_awaited()

    async def test_rerun_uses_full_analysis_mode(self) -> None:
        board = Mock()
        board.get_findings.return_value = "previous analysis"
        request_job = AsyncMock(return_value="task-1")
        controller, _ = self._controller(board=board, request_job=request_job)
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
        controller, _ = self._controller(
            request_job=AsyncMock(side_effect=RuntimeError("failed")),
        )

        with patch(
            "examples.omni_assistant_subagents.subagents.transport.media_analysis_controller.remove_attachment"
        ) as remove:
            await controller.analyze_capture({"id": "capture-1"}, "Read the label")

        remove.assert_called_once_with("session", "capture-1")


class DeferredSubagentResponseTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _emitter(hook):
        queued_batches = AsyncMock()
        queued_frame = AsyncMock()
        emitter = SimpleNamespace(
            _subagent_response_lock=asyncio.Lock(),
            _realtime_deferred_response_snapshot_hook=hook,
            _context=LLMContext([{"role": "user", "content": "latest turn"}]),
            queue_frames=queued_batches,
            queue_frame=queued_frame,
        )
        return emitter, queued_batches, queued_frame

    async def test_realtime_emitter_uses_snapshot_id_and_applies_text_output_mode(self) -> None:
        setup = LLMConfigureOutputFrame(skip_tts=True)

        async def activate_owned(on_started) -> str:
            await on_started("resp_background")
            return "resp_background"

        activate = AsyncMock(side_effect=activate_owned)
        abort = Mock()
        hook = AsyncMock(return_value=(LLMContext([]), (setup,), activate, abort))
        emitter, queued_batches, queued_frame = self._emitter(hook)

        emitted = await OmniTransportAgent._emit_subagent_response(
            emitter,
            (LLMTextFrame(text="background result"),),
        )

        self.assertTrue(emitted)
        hook.assert_awaited_once_with(emitter._context)
        self.assertEqual(queued_batches.await_args_list[0].args[0], (setup,))
        response_frames = queued_batches.await_args_list[1].args[0]
        self.assertIsInstance(response_frames[0], RealtimeOwnedLLMFullResponseStartFrame)
        self.assertEqual(response_frames[0].response_id, "resp_background")
        self.assertEqual(response_frames[1].text, "background result")
        self.assertIsInstance(response_frames[2], LLMFullResponseEndFrame)
        self.assertTrue(response_frames[0].skip_tts)
        self.assertTrue(response_frames[1].skip_tts)
        self.assertTrue(response_frames[2].skip_tts)
        queued_frame.assert_not_awaited()
        abort.assert_not_called()

    async def test_realtime_emitter_aborts_when_activation_declines_the_response(self) -> None:
        setup = LLMConfigureOutputFrame(skip_tts=True)
        activate = AsyncMock(return_value=None)
        abort = Mock()
        hook = AsyncMock(return_value=(LLMContext([]), (setup,), activate, abort))
        emitter, queued_batches, _ = self._emitter(hook)

        emitted = await OmniTransportAgent._emit_subagent_response(
            emitter,
            (LLMTextFrame(text="background result"),),
        )

        self.assertFalse(emitted)
        self.assertEqual(queued_batches.await_args_list[0].args[0], (setup,))
        self.assertEqual(queued_batches.await_count, 1)
        abort.assert_called_once_with()

    async def test_realtime_emitter_aborts_and_propagates_activation_failure(self) -> None:
        activate = AsyncMock(side_effect=RuntimeError("activation failed"))
        abort = Mock()
        hook = AsyncMock(return_value=(LLMContext([]), (), activate, abort))
        emitter, _, _ = self._emitter(hook)

        with self.assertRaisesRegex(RuntimeError, "activation failed"):
            await OmniTransportAgent._emit_subagent_response(
                emitter,
                (LLMTextFrame(text="background result"),),
            )

        abort.assert_called_once_with()

    async def test_rtvi_emitter_preserves_the_existing_spoken_response_lifecycle(self) -> None:
        emitter, queued_batches, _ = self._emitter(None)

        emitted = await OmniTransportAgent._emit_subagent_response(
            emitter,
            (LLMTextFrame(text="background result"),),
        )

        self.assertTrue(emitted)
        response_frames = queued_batches.await_args.args[0]
        self.assertEqual(len(response_frames), 3)
        self.assertFalse(response_frames[0].skip_tts)
        self.assertFalse(response_frames[1].skip_tts)
        self.assertFalse(response_frames[2].skip_tts)


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
    async def _handle_followup(self, response: dict, *, realtime_mode: bool):
        queue_frame = AsyncMock()
        emit_response = AsyncMock(return_value=True)
        controller = ThinkingController(
            context=LLMContext(messages=[]),
            request_job=AsyncMock(),
            queue_frame=queue_frame,
            emit_response=emit_response,
            followup_delay_secs=0,
            realtime_mode=realtime_mode,
        )
        controller._active_task_id = "task-1"
        handled = await controller.handle_job_response(
            SimpleNamespace(job_id="task-1", response=response, source="omni_thinker")
        )
        return handled, queue_frame, emit_response

    async def test_rtvi_thinker_followup_does_not_require_provider_metadata(self) -> None:
        handled, _, emit_response = await self._handle_followup(
            {"response": "The answer is 42."},
            realtime_mode=False,
        )

        self.assertTrue(handled)
        frames = emit_response.await_args.args[0]
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].text, "The answer is 42.")

    async def test_realtime_thinker_followup_requires_provider_terminal(self) -> None:
        handled, _, emit_response = await self._handle_followup(
            {"response": "The answer is 42."},
            realtime_mode=True,
        )

        self.assertTrue(handled)
        emit_response.assert_not_awaited()

    async def test_realtime_thinker_followup_carries_the_provider_terminal(self) -> None:
        handled, queue_frame, emit_response = await self._handle_followup(
            {
                "response": "The answer is 42.",
                "reasoning": "Computed it.",
                "finish_reason": "stop",
            },
            realtime_mode=True,
        )

        self.assertTrue(handled)
        frames = emit_response.await_args.args[0]
        self.assertEqual([type(frame) for frame in frames], [LLMTextFrame, LLMProviderCompletionReasonFrame])
        self.assertEqual(frames[0].text, "The answer is 42.")
        self.assertEqual(frames[1].finish_reason, "stop")
        queue_frame.assert_awaited_once()

    async def test_active_thinker_response_is_ignored_after_interruption(self) -> None:
        queue_frame = AsyncMock()
        request_job = AsyncMock(return_value="task-1")
        controller = ThinkingController(
            context=LLMContext(messages=[]),
            request_job=request_job,
            queue_frame=queue_frame,
            emit_response=AsyncMock(return_value=True),
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
