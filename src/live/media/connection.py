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
    with :meth:`close_after_drain` once its last event is queued.
    """

    def is_connected(self) -> bool:
        """Report the peer connection state, without waiting for the pipeline's transport to call ``connect``.

        Pipecat queues incoming data-channel messages until its transport has started. A live-session client may
        send commands as soon as the channel opens, and the session handles them without the pipeline, so they
        are dispatched as they arrive. (The keepalive ping Pipecat's own clients send does not apply here.)
        """
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
