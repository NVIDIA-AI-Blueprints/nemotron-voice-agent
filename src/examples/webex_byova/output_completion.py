# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Observe authoritative completion of Webex BYOVA bot audio output."""

from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger
from pipecat.frames.frames import TTSStoppedFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_output import BaseOutputTransport


class WebexBYOVAOutputCompletionObserver(BaseObserver):
    """Notify after the output transport has drained a TTS response."""

    def __init__(self, *, on_output_drained: Callable[[], Awaitable[None]], **kwargs: Any) -> None:
        """Create an observer bound to an asynchronous completion callback."""
        super().__init__(**kwargs)
        self._on_output_drained = on_output_drained

    async def on_push_frame(self, data: FramePushed) -> None:
        """Handle only the output transport's downstream TTS stop edge."""
        if (
            data.direction == FrameDirection.DOWNSTREAM
            and isinstance(data.frame, TTSStoppedFrame)
            and isinstance(data.source, BaseOutputTransport)
        ):
            try:
                await self._on_output_drained()
            except Exception:
                logger.exception("Failed to publish Webex BYOVA output completion")
