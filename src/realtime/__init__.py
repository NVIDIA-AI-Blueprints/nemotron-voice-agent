# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""OpenAI Realtime–shaped WebSocket protocol (v1)."""

from realtime.gateway import handle_realtime_websocket
from realtime.session import DEFAULT_PIPELINE_MODE, RealtimeSession, map_session_update_to_flat_config
from realtime.transport import (
    create_realtime_transport,
    realtime_lifecycle_observer,
    shutdown_realtime_transport,
)

__all__ = [
    "DEFAULT_PIPELINE_MODE",
    "RealtimeSession",
    "create_realtime_transport",
    "handle_realtime_websocket",
    "map_session_update_to_flat_config",
    "realtime_lifecycle_observer",
    "shutdown_realtime_transport",
]
