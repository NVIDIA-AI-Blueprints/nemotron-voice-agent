# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command handling of one worker, independent of the transport (plan sections 4.2 and 9.2).

The core owns one agent host, the keyed tool futures and the current epoch. It runs
``configure`` and ``run_conversation`` on **one** executor thread and serves
``steer`` / ``redirect`` / ``status`` / ``tool.result`` / ``interrupt`` / ``close`` on
the loop while a run is in progress. It keeps no conversation state between runs:
every ``run`` carries the settled history.

Stdlib only; the agent comes from the backend's host (``core/backends/<kind>/host.py``).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import os
import sys
from collections.abc import Callable
from typing import Any

from examples.frontend_delegation_hermes.core.backends.base import AgentHost
from examples.frontend_delegation_hermes.core.worker.tool_futures import ToolFutures

logger = logging.getLogger(__name__)

#: Emits one worker→gateway message (``{"type": ..., **fields}``); must be thread-safe.
Emit = Callable[[dict[str, Any]], None]
ActivityEmit = Callable[[str, str, str], None]


HostFactory = Callable[[ToolFutures, ActivityEmit], AgentHost]


def usage_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """Per-run token usage from two cumulative snapshots."""
    return {key: max(0, int(after.get(key, 0)) - int(before.get(key, 0))) for key in after}


class WorkerCore:
    """Serves the gateway's commands for one session."""

    def __init__(
        self,
        *,
        session_id: str,
        host_factory: HostFactory,
        emit: Emit,
        on_exit: Callable[[], None] | None = None,
        tool_timeout_s: float = 120.0,
    ) -> None:
        """``emit`` is thread-safe; ``on_exit`` runs after ``close`` finished."""
        self.session_id = session_id
        self._emit = emit
        self._on_exit = on_exit
        self.futures = ToolFutures(
            session_id=session_id, emit=emit, timeout_s=tool_timeout_s, is_interrupted=self._host_interrupted
        )
        self.host: AgentHost = host_factory(self.futures, self._activity)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="fdh-run")
        self._run_task: asyncio.Task[None] | None = None
        self._run_epoch: int | None = None
        self._unwind_timeout_s = 10.0
        self.exited = False

    # -- dispatch -----------------------------------------------------------------------

    async def handle(self, msg: dict[str, Any]) -> None:
        """Handle one gateway→worker message."""
        kind = msg["type"]
        if kind == "configure":
            await self._configure(msg)
        elif kind == "run":
            self._start_run(msg)
        elif kind in ("steer", "redirect"):
            self._steer(msg, force_redirect=kind == "redirect")
        elif kind == "status":
            self._status(msg)
        elif kind == "tool.result":
            outcome = self.futures.resolve(str(msg["call_id"]), str(msg["output"]), msg.get("epoch"))
            if outcome != "ok":
                logger.info("tool.result %s ignored: %s", msg["call_id"], outcome)
        elif kind == "interrupt":
            self._interrupt(hard=bool(msg.get("hard", True)))
        elif kind == "close":
            await self.close()

    # -- commands ---------------------------------------------------------------------------

    async def _configure(self, msg: dict[str, Any]) -> None:
        settings = dict(msg.get("backend") or {})
        self._unwind_timeout_s = float(settings.get("unwind_timeout_s", self._unwind_timeout_s))
        self.futures.timeout_s = float(settings.get("tool_result_timeout_s", self.futures.timeout_s))
        loop = asyncio.get_running_loop()
        try:
            names = await loop.run_in_executor(
                self._executor,
                lambda: self.host.configure(
                    tools=list(msg["tools"]), instructions=str(msg["instructions"]), settings=settings
                ),
            )
        except Exception as exc:  # noqa: BLE001 - reported to the gateway
            logger.exception("configure failed")
            self._emit({"type": "error", "phase": "construct", "message": f"{type(exc).__name__}: {exc}"})
            return
        self._emit({"type": "configured", "tools": names, "built": self.host.built, "version": self.host.version()})

    def _start_run(self, msg: dict[str, Any]) -> None:
        epoch = int(msg["epoch"])
        if self._run_task is not None and not self._run_task.done():
            self._emit({"type": "run_outcome", "epoch": epoch, "status": "error", "error": "a run is already active"})
            return
        self._run_epoch = epoch
        self.futures.current_epoch = epoch
        self.futures.clear_interrupt()
        self._run_task = asyncio.create_task(self._execute(msg), name=f"run-{epoch}")

    async def _execute(self, msg: dict[str, Any]) -> None:
        epoch = int(msg["epoch"])
        loop = asyncio.get_running_loop()
        try:
            outcome = await loop.run_in_executor(self._executor, self._run_blocking, msg)
        except Exception as exc:  # noqa: BLE001 - never lose an outcome
            outcome = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        self._emit({"type": "run_outcome", "epoch": epoch, **outcome})

    def _run_blocking(self, msg: dict[str, Any]) -> dict[str, Any]:
        agent = self.host.agent()
        agent.clear_interrupt()
        before = agent.usage_snapshot()
        try:
            result = agent.run_conversation(
                str(msg["user_message"]), list(msg["conversation_history"]), str(msg["task_id"])
            )
        except Exception as exc:  # noqa: BLE001 - an agent exception is an ``error`` outcome
            logger.exception("run_conversation raised")
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "usage_delta": usage_delta(before, agent.usage_snapshot()),
            }
        if result.get("interrupted"):
            status = "interrupted"
        elif result.get("failed"):
            status = "failed"
        else:
            status = "ok"
        outcome: dict[str, Any] = {
            "status": status,
            "final_response": result.get("final_response") or "",
            "messages": result.get("messages"),
            "turn_exit_reason": result.get("turn_exit_reason"),
            "usage_delta": usage_delta(before, agent.usage_snapshot()),
        }
        if result.get("pending_steer"):
            outcome["pending_steer"] = result["pending_steer"]
        if result.get("error"):
            outcome["error"] = str(result["error"])
        return outcome

    def _running(self, epoch: Any) -> bool:
        return (
            self._run_task is not None
            and not self._run_task.done()
            and self._run_epoch is not None
            and epoch == self._run_epoch
        )

    def _steer(self, msg: dict[str, Any], *, force_redirect: bool) -> None:
        epoch, text = msg["epoch"], str(msg["text"])
        if not self._running(epoch) or not self.host.built:
            self._emit({"type": "steer_result", "epoch": epoch, "accepted": False, "via": "none"})
            return
        agent = self.host.agent()
        mode = "redirect" if force_redirect else str(msg.get("mode") or "auto")
        if mode in ("auto", "redirect") and agent.model_request_active() and agent.redirect(text):
            self._emit({"type": "steer_result", "epoch": epoch, "accepted": True, "via": "redirect"})
            return
        accepted = agent.steer(text)
        self._emit({"type": "steer_result", "epoch": epoch, "accepted": bool(accepted), "via": "steer"})

    def _status(self, msg: dict[str, Any]) -> None:
        summary: dict[str, Any] = {}
        if self.host.built:
            with contextlib.suppress(Exception):
                summary = dict(self.host.agent().get_activity_summary())
        summary["outstanding"] = self.futures.outstanding()
        summary["running"] = self._running(msg["epoch"])
        self._emit({"type": "status_result", "epoch": msg["epoch"], "summary": summary})

    def _interrupt(self, *, hard: bool) -> None:
        self.futures.interrupt_all()
        if self.host.built:
            with contextlib.suppress(Exception):
                self.host.agent().hard_interrupt("fdh: interrupted by the gateway" if hard else "fdh: interrupted")

    async def close(self, reason: str = "close") -> None:
        """Resolve futures, interrupt, join the run (bounded), close the agent, report ``closed``."""
        if self.exited:
            return
        self.futures.close_all()
        if self.host.built:
            with contextlib.suppress(Exception):
                self.host.agent().hard_interrupt(f"fdh: {reason}")
        unwound = True
        if self._run_task is not None and not self._run_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._run_task), timeout=self._unwind_timeout_s)
            except TimeoutError:
                unwound = False
        if unwound:
            with contextlib.suppress(Exception):
                self.host.close()
        else:
            logger.warning("run thread did not unwind within %ss; agent.close() skipped", self._unwind_timeout_s)
        self.exited = True
        with contextlib.suppress(Exception):
            self._emit({"type": "closed", "unwound": unwound, "reason": reason})
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self._on_exit is not None:
            self._on_exit()

    # -- helpers ----------------------------------------------------------------------------

    def _activity(self, event: str, name: str, preview: str) -> None:
        with contextlib.suppress(Exception):
            self._emit(
                {
                    "type": "activity",
                    "epoch": self.futures.current_epoch,
                    "event": event,
                    "name": name,
                    "preview": preview[:200],
                }
            )

    def _host_interrupted(self) -> bool:
        host = getattr(self, "host", None)
        return bool(host is not None and host.built and host.is_interrupted())


def ready_message(host: AgentHost | None = None) -> dict[str, Any]:
    """The ``ready`` message a worker sends once connected."""
    return {
        "type": "ready",
        "pid": os.getpid(),
        "python": sys.version.split()[0],
        "agent_version": host.version() if host is not None else "",
    }
