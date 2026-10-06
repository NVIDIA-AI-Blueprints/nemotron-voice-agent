# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Pipecat processor that runs one prototype :class:`RealtimeSession`.

The prototype's WebSocket handler (``voice/server.py`` ``realtime``) builds a
``RealtimeSession`` and calls ``session.run(receive)`` with a socket reader and a
``CallbackTransport(websocket.send_text)``. This processor does the same inside a
Pipecat pipeline: ``receive`` reads the client messages that the transport delivered
as :class:`PrototypeClientMessageFrame` (in arrival order), and the session's writer
sends each server event downstream as an urgent transport message. The session's
behaviour, event order included, is the prototype's.
"""

from __future__ import annotations

import asyncio
import contextlib

from loguru import logger
from pipecat.frames.frames import CancelFrame, EndFrame, Frame, OutputTransportMessageUrgentFrame, StartFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from examples.frontend_backend_verdict.bridge.runtime import VerdictRuntime
from examples.frontend_backend_verdict.bridge.wire import PrototypeClientMessageFrame
from examples.frontend_backend_verdict.voice.engine.session import RealtimeSession


class _PipecatWireTransport:
    """The prototype ``WireTransport``: one server event per urgent transport message."""

    def __init__(self, processor: PrototypeSessionProcessor) -> None:
        self._processor = processor

    async def send_text(self, text: str) -> None:
        """Send one serialized server event (called by the session's single FIFO writer)."""
        await self._processor.push_frame(OutputTransportMessageUrgentFrame(message=text))


class PrototypeSessionProcessor(FrameProcessor):
    """Serve one Realtime connection with the prototype session."""

    def __init__(self, *, runtime: VerdictRuntime, model: str, show_silent_filler: bool, **kwargs) -> None:
        """Hold the shared runtime; the session starts with the pipeline."""
        super().__init__(**kwargs)
        self._runtime = runtime
        self._model = model
        self._show_silent_filler = show_silent_filler
        self._inbox: asyncio.Queue[str | bytes | None] = asyncio.Queue()
        self._session: RealtimeSession | None = None
        self._session_task: asyncio.Task[None] | None = None

    @property
    def session(self) -> RealtimeSession | None:
        """The running prototype session (``None`` before the pipeline starts)."""
        return self._session

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Start the session, feed it client messages, and stop it with the pipeline."""
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            self._start_session()
            return
        if isinstance(frame, PrototypeClientMessageFrame):
            self._inbox.put_nowait(frame.message)
            return
        if isinstance(frame, (EndFrame, CancelFrame)):
            await self.client_disconnected()
        await self.push_frame(frame, direction)

    def _start_session(self) -> None:
        if self._session is not None:
            return
        state = self._runtime.state
        self._session = RealtimeSession(
            config=self._runtime.config,
            transport=_PipecatWireTransport(self),
            services=state.services,
            agent_factory=self._runtime.agent_factory,
            routing_sink=state.routing_sink,
            filler_log=state.filler_log,
            model=self._model,
            show_silent_filler=self._show_silent_filler,
        )
        self._session_task = asyncio.create_task(self._run_session(), name=f"fbv-session-{self._session.session_id}")

    async def _run_session(self) -> None:
        assert self._session is not None  # noqa: S101 - set by _start_session
        try:
            await self._session.run(self._inbox.get)
        except Exception:  # noqa: BLE001 - log and drop this connection only (prototype server rule)
            logger.exception(f"[{self._session.session_id}] session crashed")

    async def client_disconnected(self) -> None:
        """End the session as the prototype does when the socket closes (``receive`` returns ``None``)."""
        task = self._session_task
        if task is None:
            return
        self._inbox.put_nowait(None)
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
