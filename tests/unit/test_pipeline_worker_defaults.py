# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Tests for shared pipeline worker lifecycle defaults."""

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock

from pipecat.bus.messages import BusCancelMessage
from pipecat.frames.frames import ErrorFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import ProcessorUnusablePolicy

from examples.shared.pipeline_utils import VoiceAgentPipelineWorker


class VoiceAgentPipelineWorkerTests(unittest.IsolatedAsyncioTestCase):
    """Verify shared unusable-processor behavior."""

    def test_unusable_processors_end_gracefully_by_default(self) -> None:
        """Use END unless a caller explicitly overrides the project default."""
        worker = VoiceAgentPipelineWorker(Pipeline([]), enable_rtvi=False)

        self.assertIs(worker._processor_unusable_policy, ProcessorUnusablePolicy.END)

    async def test_multi_worker_error_cancels_runner_once(self) -> None:
        """Broadcast one runner cancellation when a required processor is unusable."""
        worker = VoiceAgentPipelineWorker(
            Pipeline([]),
            cancel_runner_on_unusable_processor=True,
            enable_rtvi=False,
        )
        bus = Mock()
        bus.send = AsyncMock()
        worker._bus = bus
        processor = Mock(is_usable=False)
        frame = ErrorFrame(error="service unavailable", processor=processor)

        await worker._call_event_handler("on_pipeline_error", frame)
        await asyncio.sleep(0)
        await worker._call_event_handler("on_pipeline_error", frame)
        await asyncio.sleep(0)

        bus.send.assert_awaited_once()
        message = bus.send.await_args.args[0]
        self.assertIsInstance(message, BusCancelMessage)
        self.assertEqual(message.source, worker.name)
        self.assertIn("can no longer do its job", message.reason)
