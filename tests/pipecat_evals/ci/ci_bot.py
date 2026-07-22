# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Deterministic Pipecat Eval bot for CI.

This bot intentionally avoids ASR, TTS, LLM, and NVIDIA service dependencies.
It exercises the eval transport and RTVI event path by turning incoming
``send-text`` messages into the same Pipecat frames emitted by production
pipelines.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipecat.frames.frames import (
    Frame,
    LLMConfigureOutputFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.workers.runner import WorkerRunner

from examples.shared.pipeline_utils import create_transport


class CIEvalResponder(FrameProcessor):
    """Convert eval text turns into deterministic RTVI-observable bot replies."""

    def __init__(self) -> None:
        """Initialize the responder turn counter."""
        super().__init__()
        self._turn_count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Emit CI eval response frames for incoming user text turns."""
        await super().process_frame(frame, direction)
        if direction is not FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMMessagesAppendFrame):
            user_text = _latest_user_text(frame.messages)
            await self._respond_to_user_text(user_text or "(empty message)")
            return

        # The eval harness toggles skip-TTS around text-mode turns. The CI bot
        # has no TTS stage, so these configuration frames do not need to flow on.
        if isinstance(frame, LLMConfigureOutputFrame):
            return

        await self.push_frame(frame, direction)

    async def _respond_to_user_text(self, user_text: str) -> None:
        self._turn_count += 1
        await self.push_frame(UserStartedSpeakingFrame())
        await self.push_frame(
            TranscriptionFrame(
                text=user_text,
                user_id="ci-user",
                timestamp=datetime.now(UTC).isoformat(),
                finalized=True,
            )
        )
        await self.push_frame(UserStoppedSpeakingFrame())
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame(text=f"CI eval bot turn {self._turn_count}: {user_text}"))
        await self.push_frame(LLMFullResponseEndFrame())


def _latest_user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [str(part.get("text", "")) for part in content if isinstance(part, dict)]
            return " ".join(part for part in parts if part).strip()
    return ""


async def bot(runner_args: RunnerArguments) -> None:
    """Run a service-free bot compatible with Pipecat Eval suites."""
    transport = create_transport(runner_args)
    worker = PipelineWorker(
        Pipeline([transport.input(), CIEvalResponder(), transport.output()]),
        params=PipelineParams(enable_metrics=False, enable_usage_metrics=False),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
