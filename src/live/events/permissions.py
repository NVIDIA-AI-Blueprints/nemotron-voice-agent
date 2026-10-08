# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which events may cross a WebRTC data channel, from ``client.data_channel`` in the session config."""

from __future__ import annotations

from live.protocol import SessionConfig


def server_event_allowed(config: SessionConfig, transport: str, message: dict, source: str) -> bool:
    """Apply ``allowed_server_events`` to a WebRTC primary connection; everything else sees every event."""
    if source == "sideband" or transport != "webrtc" or config.client is None:
        return True
    allow = config.client.data_channel.allowed_server_events
    if allow == "all":
        return True
    for selector in allow:
        if isinstance(selector, str) and selector == message["type"]:
            return True
        if isinstance(selector, dict) and selector.get("type") == message["type"]:
            nested = selector.get("response_event")
            if nested is None or nested == message.get("event", {}).get("type"):
                return True
    return False


def client_event_allowed(config: SessionConfig, transport: str, kind: str, source: str) -> bool:
    """Apply ``allowed_client_events`` to a WebRTC primary connection."""
    if source != "primary" or transport != "webrtc" or not config.client:
        return True
    allow = config.client.data_channel.allowed_client_events
    return allow == "all" or kind in allow
