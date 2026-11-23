# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A live session's WebRTC peer: Pipecat's SmallWebRTC connection, which the session closes."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from loguru import logger
from pipecat.transports.smallwebrtc.connection import SIGNALLING_TYPE, IceServer, SmallWebRTCConnection

if TYPE_CHECKING:
    from collections.abc import Callable

    from aiortc import RTCDataChannel

DRAIN_TIMEOUT_SECONDS = 1.0
CLIENT_EVENT = "client-event"
# The members the gateway and Pipecat's transport use: any other is missing, so a new way to end the peer fails loudly.
CONNECTION_MEMBERS = frozenset(
    {
        "initialize",
        "get_answer",
        "connect",
        "is_connected",
        "pc_id",
        "send_app_message",
        "audio_input_track",
        "video_input_track",
        "screen_video_input_track",
        "replace_audio_track",
        "replace_video_track",
    }
)


def client_event(message: str | bytes) -> dict | None:
    """Return a data-channel message's client event by Pipecat's filter: ``None`` for a ping, signalling or no type."""
    if isinstance(message, str) and message.startswith("ping"):
        return None
    try:
        event = json.loads(message)
        if event["type"] == SIGNALLING_TYPE and event.get("message"):
            return None
    except Exception:
        return None
    return event


class LivePeer:
    """Pipecat's SmallWebRTC connection, passed through member by member; the pipeline's end leaves it open."""

    def __init__(self, connection: SmallWebRTCConnection):
        """Wrap ``connection`` before its offer is answered, and follow the client's data channel."""
        self.connection = connection
        self.channel: RTCDataChannel | None = None
        self.closing = False
        self.handlers: list[Callable[..., Any]] = []
        # Pipecat's connection holds client messages until its transport connects; the session needs them earlier.
        connection.pc.on("datachannel", self._follow)

    @classmethod
    def from_ice_servers(cls, ice_servers: list[dict]) -> LivePeer:
        """Return a peer on the gateway's ICE servers, each a ``{urls, username, credential}`` object."""
        return cls(
            SmallWebRTCConnection(
                ice_servers=[
                    IceServer(urls=server["urls"], username=server.get("username"), credential=server.get("credential"))
                    for server in ice_servers
                ]
            )
        )

    def __getattr__(self, name: str) -> Any:
        """Return a member of Pipecat's connection that :data:`CONNECTION_MEMBERS` lists; any other is missing."""
        if name not in CONNECTION_MEMBERS or "connection" not in vars(self):
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self.connection, name)

    def event_handler(self, event_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Return a decorator adding a handler of ``event_name``: the peer's :data:`CLIENT_EVENT`, or Pipecat's."""
        if event_name != CLIENT_EVENT:
            return self.connection.event_handler(event_name)

        def add(handler: Callable[..., Any]) -> Callable[..., Any]:
            self.handlers.append(handler)
            return handler

        return add

    def _follow(self, channel: RTCDataChannel) -> None:
        """Follow a client data channel: hand on its events; its close ends the peer unless a newer channel opened."""
        self.channel = channel

        @channel.on("message")
        async def on_message(message):
            if (event := client_event(message)) is None:
                return
            for handler in self.handlers:
                try:
                    await handler(self, event)
                except Exception:
                    logger.exception(f"A {CLIENT_EVENT} handler failed")

        @channel.on("close")
        async def on_close():
            if channel is self.channel and not self.closing:
                await self.connection.disconnect()

    async def disconnect(self) -> None:
        """Leave the peer open when the pipeline ends: the session closes it once ``session.closed`` has gone out."""

    async def close_after_drain(self) -> None:
        """Flush the data channel, then close it and the peer connection.

        The channel closes first, so the "peer left" message of Pipecat's ``disconnect`` never reaches the client.
        """
        self.closing = True
        channel = self.channel
        if channel is not None and channel.readyState == "open":
            try:
                async with asyncio.timeout(DRAIN_TIMEOUT_SECONDS):
                    while channel.bufferedAmount:
                        await asyncio.sleep(0.01)
                    # One more tick lets the SCTP stack deliver what was just written.
                    await asyncio.sleep(0.05)
            except TimeoutError:
                logger.warning("Data channel did not drain before close")
            channel.close()
        await self.connection.disconnect()
