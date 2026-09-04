# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Tests for Webex BYOVA output-drained completion detection."""

import unittest
from unittest.mock import AsyncMock, MagicMock

from pipecat.frames.frames import BotStoppedSpeakingFrame, TTSStoppedFrame
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_output import BaseOutputTransport

from examples.webex_byova.output_completion import WebexBYOVAOutputCompletionObserver


class WebexBYOVAOutputCompletionObserverTests(unittest.IsolatedAsyncioTestCase):
    """Ensure completion follows the output transport's drained FIFO edge."""

    async def test_only_output_transport_tts_stop_completes_response(self) -> None:
        """Ignore sentence stops and the pre-drain TTS stop."""
        on_output_drained = AsyncMock()
        observer = WebexBYOVAOutputCompletionObserver(on_output_drained=on_output_drained)
        destination = MagicMock()

        async def observe(frame, source, direction=FrameDirection.DOWNSTREAM) -> None:
            await observer.on_push_frame(
                FramePushed(
                    source=source,
                    destination=destination,
                    frame=frame,
                    direction=direction,
                    timestamp=0,
                )
            )

        await observe(BotStoppedSpeakingFrame(), MagicMock())
        await observe(BotStoppedSpeakingFrame(), MagicMock())
        tts_stopped = TTSStoppedFrame()
        await observe(tts_stopped, MagicMock())
        await observe(tts_stopped, MagicMock(spec=BaseOutputTransport), FrameDirection.UPSTREAM)
        on_output_drained.assert_not_awaited()

        await observe(tts_stopped, MagicMock(spec=BaseOutputTransport))

        on_output_drained.assert_awaited_once_with()


if __name__ == "__main__":
    unittest.main()
