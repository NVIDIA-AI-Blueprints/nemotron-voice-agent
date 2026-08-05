# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103, D107

"""Shared helpers for Realtime gateway unit tests."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any


class FakeWebSocket:
    """Minimal WebSocket stand-in for gateway unit tests."""

    def __init__(self, client_messages: list[str]) -> None:
        """Queue outbound client frames; record server sends."""
        self._messages = list(client_messages)
        self.sent: list[dict[str, Any]] = []
        self.accepted = False
        self.closed = False
        self.close_code: int | None = None
        self.close_reason: str | None = None
        self.headers: dict[str, str] = {}
        self.query_params: dict[str, str] = {}
        self.state = SimpleNamespace()
        self.subprotocol: str | None = None

    async def accept(self, subprotocol: str | None = None) -> None:
        self.accepted = True
        self.subprotocol = subprotocol

    async def send_text(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def receive_text(self) -> str:
        from fastapi import WebSocketDisconnect

        if not self._messages:
            raise WebSocketDisconnect()
        return self._messages.pop(0)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = True
        self.close_code = code
        self.close_reason = reason
