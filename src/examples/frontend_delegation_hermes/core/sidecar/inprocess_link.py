# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``InProcessBackendLink``: the gateway and a fake worker inside the voice server's process.

Used by the voice side's offline tests and by ``--stub-backend``. Messages go through
the same validation and the same :class:`SessionRuntime` / :class:`BackendController`
as over the WebSocket; voice→gateway messages are handled strictly in order.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from examples.frontend_delegation_hermes.core.backend import protocol as proto
from examples.frontend_delegation_hermes.core.sidecar.events import GatewayEventLog
from examples.frontend_delegation_hermes.core.sidecar.gateway_config import (
    GatewayConfig,
    default_fake_gateway_config,
    load_gateway_config,
)
from examples.frontend_delegation_hermes.core.sidecar.session_runtime import SessionRuntime

__all__ = ["InProcessBackendLink", "default_fake_gateway_config", "load_gateway_config"]

logger = logging.getLogger(__name__)

OnMessage = Callable[[dict[str, Any]], None]


class InProcessBackendLink:
    """A ``BackendLink`` whose gateway runs on the caller's loop."""

    def __init__(self, gateway_config: GatewayConfig | None = None, *, event_log: GatewayEventLog | None = None):
        """Defaults to :func:`default_fake_gateway_config`.

        A config with ``workers.agent_kind: fake`` (for example ``gateway.fake.yaml``) always
        runs its fake worker in this process: this link never spawns processes for fakes.
        """
        config = gateway_config or default_fake_gateway_config()
        if config.workers.agent_kind == "fake" and config.workers.mode != "in_process_fake":
            config = replace(config, workers=replace(config.workers, mode="in_process_fake"))
        self._config = config
        self._event_log = event_log or GatewayEventLog(None)
        self._runtime: SessionRuntime | None = None
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._on_message: OnMessage = lambda _msg: None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False

    @property
    def event_log(self) -> GatewayEventLog:
        """The gateway-side event records (in memory by default)."""
        return self._event_log

    async def open(self, on_message: OnMessage) -> None:
        """Start the in-process gateway session."""
        self._on_message = on_message
        self._loop = asyncio.get_running_loop()
        self._runtime = SessionRuntime(self._config, self._deliver, event_log=self._event_log)
        self._task = asyncio.create_task(self._serve(), name="inprocess-gateway")

    def _deliver(self, message: dict[str, Any]) -> None:
        if not self._closed and self._loop is not None:
            self._loop.call_soon(self._on_message, message)

    def send(self, message: dict[str, Any]) -> None:
        """Validate and enqueue one voice→gateway message."""
        if self._closed:
            raise RuntimeError("backend link is closed")
        fields = {k: v for k, v in message.items() if k not in ("v", "type")}
        self._queue.put_nowait(proto.message(proto.VOICE_TO_GATEWAY, message["type"], **fields))

    async def _serve(self) -> None:
        while True:
            msg = await self._queue.get()
            if msg is None or self._runtime is None:
                return
            try:
                await self._runtime.handle(msg)
            except Exception:  # noqa: BLE001 - one bad message must not end the session
                logger.exception("in-process gateway failed on %s", msg.get("type"))

    async def close(self) -> None:
        """Close the session and its worker."""
        if self._closed:
            return
        self._closed = True
        self._queue.put_nowait(None)
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(self._task, timeout=2.0)
        if self._runtime is not None:
            await self._runtime.close()
