# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebRTC connection for live sessions, built on Pipecat's SmallWebRTC."""

from __future__ import annotations

import asyncio

from loguru import logger
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection

DRAIN_TIMEOUT_SECONDS = 1.0


class LiveWebRTCConnection(SmallWebRTCConnection):
    """A SmallWebRTC connection whose lifetime belongs to the live session.

    Pipecat's own ``disconnect`` sends a Pipecat-specific "peer left" message and closes the peer connection at
    once. A live-session client must see only live events, and it must receive the final ``session.closed`` event
    before the channel goes away. So the pipeline's disconnect is a no-op, and the session closes the connection
    with :meth:`close_after_drain` once its last event is queued. A client that closes only its data channel ends the
    session too: the connection reports ``data-channel-closed``.
    """

    def __init__(self, *args, **kwargs):
        """Create the connection, which also reports its data channel's close."""
        super().__init__(*args, **kwargs)
        self._register_event_handler("data-channel-closed")
        self._transport_attached = False  # a pipeline transport registered its "connected" handler

    def _setup_listeners(self):
        """Pipecat's listeners, and one more on the data channel: its close ends the session."""
        super()._setup_listeners()

        @self._pc.on("datachannel")
        def on_datachannel(channel):
            @channel.on("close")
            async def on_close():
                if channel is self._data_channel:  # not a channel a renegotiation replaced
                    await self._call_event_handler("data-channel-closed")

    def add_event_handler(self, event_name: str, handler) -> None:
        """Add a handler; a ``connected`` handler means a pipeline transport now runs on this connection."""
        super().add_event_handler(event_name, handler)
        self._transport_attached = self._transport_attached or event_name == "connected"

    def is_connected(self) -> bool:
        """Report whether the peer is connected, by Pipecat's own check once a pipeline transport runs on it.

        A live-session client may send commands as soon as the channel opens, and the session handles them without
        the pipeline, so until a transport exists the peer state decides and they are dispatched as they arrive. From
        then on Pipecat's check decides, so the transport's ``connect()`` always runs its setup and the session cannot
        go deaf; commands that arrive before it are flushed when it connects. (The keepalive ping Pipecat's own
        clients send does not apply here.)
        """
        if self._transport_attached:
            return super().is_connected()
        return self._pc.connectionState == "connected"

    async def disconnect(self) -> None:
        """Ignore the pipeline's disconnect; the session decides when the connection ends."""

    async def close_after_drain(self) -> None:
        """Let the data channel flush its buffered events, then close the peer connection."""
        channel = self._data_channel
        if channel is not None and channel.readyState == "open":
            try:
                async with asyncio.timeout(DRAIN_TIMEOUT_SECONDS):
                    while channel.bufferedAmount:
                        await asyncio.sleep(0.01)
                    # One more tick lets the SCTP stack deliver what was just written.
                    await asyncio.sleep(0.05)
            except TimeoutError:
                logger.warning("Data channel did not drain before close")
        await self._close()
