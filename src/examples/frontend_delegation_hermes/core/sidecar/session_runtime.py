# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One voice connection in the gateway: routes messages to a :class:`BackendController`.

The runtime validates voice→gateway messages, wraps controller output into
gateway→voice protocol messages for the connection's single writer, and builds the
controller with a worker from the :class:`WorkerPool`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from examples.frontend_delegation_hermes.core.backend import protocol as proto
from examples.frontend_delegation_hermes.core.backend.controller import BackendController, ControllerSettings
from examples.frontend_delegation_hermes.core.sidecar.events import GatewayEventLog
from examples.frontend_delegation_hermes.core.sidecar.gateway_config import GatewayConfig
from examples.frontend_delegation_hermes.core.sidecar.templates import BackendTemplates
from examples.frontend_delegation_hermes.core.sidecar.worker_pool import WorkerPool

logger = logging.getLogger(__name__)

Send = Callable[[dict[str, Any]], None]


def controller_settings(config: GatewayConfig) -> ControllerSettings:
    """The controller's view of the gateway config."""
    h = config.hermes
    return ControllerSettings(
        run_hard_deadline_s=h.run_hard_deadline_s,
        unwind_timeout_s=h.unwind_timeout_s,
        steer_timeout_s=h.steer_timeout_s,
        status_timeout_s=h.status_timeout_s,
        configure_timeout_s=config.workers.configure_timeout_s,
        respawn=config.recovery.respawn,
        max_respawns=config.recovery.max_respawns_per_session,
        worker_hermes={
            **h.worker_settings(),
            "fake_default_tool": config.workers.fake_default_tool,
            "backend_trace_dir": config.workers.log_dir if h.backend_trace else "",
        },
        domains=config.domains,
    )


class SessionRuntime:
    """The gateway side of one voice session."""

    def __init__(
        self,
        config: GatewayConfig,
        send: Send,
        *,
        pool: WorkerPool | None = None,
        templates: BackendTemplates | None = None,
        event_log: GatewayEventLog | None = None,
    ) -> None:
        """``send`` takes a validated gateway→voice message dict (enqueue only)."""
        self._config = config
        self._send = send
        self._templates = templates or BackendTemplates(
            config.hermes.prompts_path, prompt_features=config.prompt_features
        )
        self._pool = pool or WorkerPool(config, soul=self._templates.render("backend_soul"))
        self._event_log = event_log
        self.controller: BackendController | None = None
        self.session_id: str | None = None

    async def handle(self, msg: dict[str, Any]) -> None:
        """One validated voice→gateway message."""
        kind = msg["type"]
        if kind == "session.open":
            if self.controller is not None:
                self._error("already_open", "session.open was already received", fatal=False)
                return
            self.session_id = str(msg["session_id"])
            self.controller = BackendController(
                session_id=self.session_id,
                settings=controller_settings(self._config),
                templates=self._templates,
                emit=self._emit,
                worker_factory=lambda: self._pool.acquire(self.session_id or "session"),
                log=self._log,
            )
            self._log("session_open", config=self._config.summary(), settings=msg.get("settings"))
            await self.controller.open(dict(msg.get("settings") or {}))
            return
        controller = self.controller
        if controller is None:
            self._error("not_open", f"{kind} before session.open", fatal=False)
            return
        if kind == "session.configure":
            await controller.configure(list(msg["tools"]), str(msg["instructions"]))
        elif kind == "history.append":
            controller.append(list(msg["entries"]))
        elif kind == "delegate":
            request = msg["request"] if msg["request"] in proto.REQUESTS else proto.REQUEST_TASK
            await controller.delegate(msg["turn_id"], request, list(msg["run_input"]))
        elif kind == "tool.result":
            controller.tool_result(
                str(msg["call_id"]), str(msg["output"]), msg.get("epoch"), local=bool(msg.get("local"))
            )
        elif kind == "session.close":
            await self.close()

    async def close(self) -> None:
        """Close the controller (stops the worker); idempotent."""
        if self.controller is not None:
            await self.controller.close()
            self._log("session_close")

    def _emit(self, data: dict[str, Any]) -> None:
        fields = dict(data)
        kind = fields.pop("type")
        self._send(proto.message(proto.GATEWAY_TO_VOICE, kind, **fields))

    def _error(self, code: str, message: str, *, fatal: bool) -> None:
        self._emit({"type": "error", "code": code, "message": message, "fatal": fatal})

    def _log(self, event: str, /, **data: Any) -> None:
        if self._event_log is not None:
            self._event_log.write(event, self.session_id, data)
