# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D100, D101, D102, D103, D107

"""Shared fakes for the backend-half tests (controller, context, gateway)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from examples.frontend_delegation_hermes.core.backend.controller import BackendController, ControllerSettings
from examples.frontend_delegation_hermes.core.sidecar.gateway_config import CONFIG_DIR
from examples.frontend_delegation_hermes.core.sidecar.templates import BackendTemplates

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKFLOW_CSV = REPO_ROOT / "src" / "examples" / "frontend_delegation_hermes" / "fixtures" / "workflow.csv"

TOOL = {
    "type": "function",
    "name": "get_order",
    "description": "Look up an order.",
    "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
}


def templates() -> BackendTemplates:
    """The shipped backend catalog."""
    return BackendTemplates(CONFIG_DIR / "prompts" / "customer_service" / "backend" / "hermes.yaml")


class RecordingWorker:
    """A scriptable ``WorkerPort``: records commands, auto-replies to configure/steer/status on the loop."""

    instances: list[RecordingWorker] = []

    def __init__(self, *, steer_via: str = "steer", steer_accepted: bool = True, configure_error: str | None = None):
        """Replies are delivered with ``loop.call_soon`` like a real worker's."""
        self.sent: list[dict[str, Any]] = []
        self.pid = 1000 + len(RecordingWorker.instances)
        self.steer_via = steer_via
        self.steer_accepted = steer_accepted
        self.configure_error = configure_error
        self._alive = False
        self.on_message: Any = None
        self.on_exit: Any = None
        self.stopped = False
        self.killed: list[str] = []
        self.start_ms = 1
        RecordingWorker.instances.append(self)

    @property
    def alive(self) -> bool:
        """Started and not killed/stopped."""
        return self._alive

    def set_listener(self, on_message: Any, on_exit: Any) -> None:
        """Bind callbacks."""
        self.on_message, self.on_exit = on_message, on_exit

    async def start(self) -> dict[str, Any]:
        """Ready at once."""
        self._alive = True
        return {"type": "ready", "pid": self.pid, "python": "test"}

    def send(self, message: dict[str, Any]) -> None:
        """Record; auto-reply to request/response commands."""
        if not self._alive:
            raise RuntimeError("worker is gone")
        self.sent.append(message)
        loop = asyncio.get_running_loop()
        kind = message["type"]
        if kind == "configure":
            reply = (
                {"type": "error", "phase": "construct", "message": self.configure_error}
                if self.configure_error
                else {"type": "configured", "tools": [t["name"] for t in message["tools"]], "built": True}
            )
            loop.call_soon(self.on_message, reply)
        elif kind == "steer":
            loop.call_soon(
                self.on_message,
                {
                    "type": "steer_result",
                    "epoch": message["epoch"],
                    "accepted": self.steer_accepted,
                    "via": self.steer_via,
                },
            )
        elif kind == "status":
            loop.call_soon(
                self.on_message,
                {
                    "type": "status_result",
                    "epoch": message["epoch"],
                    "summary": {"current_tool": "get_order", "api_call_count": 2},
                },
            )

    def reply(self, message: dict[str, Any]) -> None:
        """Deliver a worker message now."""
        self.on_message(message)

    def outcome(self, epoch: int, status: str = "ok", **fields: Any) -> None:
        """Deliver a ``run_outcome``."""
        self.on_message({"type": "run_outcome", "epoch": epoch, "status": status, **fields})

    async def stop(self) -> str:
        """Graceful stop."""
        self.stopped = True
        self._alive = False
        return "closed"

    async def kill(self, reason: str) -> None:
        """Kill."""
        self.killed.append(reason)
        self._alive = False

    def die(self, reason: str = "exited: -9") -> None:
        """Simulate an unexpected exit."""
        self._alive = False
        self.on_exit(reason)

    def rss_mb(self) -> float | None:
        """Unknown."""
        return None

    def of_type(self, kind: str) -> list[dict[str, Any]]:
        """Recorded commands of one type."""
        return [m for m in self.sent if m["type"] == kind]


class Harness:
    """A controller over :class:`RecordingWorker` s, with the emitted messages."""

    def __init__(self, **settings: Any) -> None:
        """Build (call :meth:`start` inside the test's loop)."""
        RecordingWorker.instances.clear()
        self.emitted: list[dict[str, Any]] = []
        self.logs: list[tuple[str, dict[str, Any]]] = []
        self.worker_kwargs: dict[str, Any] = settings.pop("worker_kwargs", {})
        self.controller = BackendController(
            session_id="s1",
            settings=ControllerSettings(
                **{"steer_timeout_s": 0.5, "status_timeout_s": 0.5, "configure_timeout_s": 1.0, **settings}
            ),
            templates=templates(),
            emit=self.emitted.append,
            worker_factory=lambda: RecordingWorker(**self.worker_kwargs),
            log=lambda event, **data: self.logs.append((event, data)),
        )

    @property
    def worker(self) -> RecordingWorker:
        """The newest worker."""
        return RecordingWorker.instances[-1]

    async def start(self, *, delay_s: float = 0.0, update_after_start: str = "error") -> None:
        """Open and configure."""
        await self.controller.open(
            {
                "simulated_delay": {"seconds": delay_s, "where": "per_delegation"},
                "steer_mode": "auto",
                "update_after_start": update_after_start,
            }
        )
        await self.controller.configure([TOOL], "POLICY")

    async def delegate(self, turn_id: int, text: str, *, seq: int | None = None, request: str = "task") -> None:
        """One delegation."""
        await self.controller.delegate(
            turn_id, request, [{"seq": seq if seq is not None else turn_id * 10, "text": text}]
        )

    async def settle(self) -> None:
        """Let scheduled settle tasks run."""
        for _ in range(5):
            await asyncio.sleep(0)

    def of_type(self, kind: str) -> list[dict[str, Any]]:
        """Emitted messages of one type."""
        return [m for m in self.emitted if m["type"] == kind]


def history_after(user: str, answer: str, prior: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Hermes-shaped messages of one finished turn."""
    return [*(prior or []), {"role": "user", "content": user}, {"role": "assistant", "content": answer}]
