# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""OpenAI Realtime-compatible WebSocket protocol."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "RealtimeSessionController": "realtime.controller",
    "RealtimeModelRoute": "realtime.gateway",
    "create_realtime_transport": "realtime.transport",
    "handle_realtime_websocket": "realtime.gateway",
    "realtime_controller": "realtime.transport",
    "realtime_lifecycle_observer": "realtime.transport",
    "shutdown_realtime_transport": "realtime.transport",
}

__all__ = [
    "RealtimeSessionController",
    "RealtimeModelRoute",
    "create_realtime_transport",
    "handle_realtime_websocket",
    "realtime_controller",
    "realtime_lifecycle_observer",
    "shutdown_realtime_transport",
]


def __getattr__(name: str) -> Any:
    """Load public protocol components only when callers request them."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the package's stable public surface without importing it."""
    return sorted((*globals(), *__all__))
