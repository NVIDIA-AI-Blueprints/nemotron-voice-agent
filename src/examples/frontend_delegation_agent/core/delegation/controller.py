# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-session backend controller, run by the gateway (plan section 5).

The controller owns the conversation state of one Realtime session: the backend
state (NO_SESSION / IDLE / WORKING), run epochs, the backend's settled history, the
context queue and the tool-call routing table. The backend worker is a disposable
agent host behind :class:`WorkerPort`; when it dies it is replaced from this state.

Every command starts with ``async with self._lock: self._settle_locked()``, so a
delegation is always classified against a settled state (section 5.2). Worker
messages only record outcomes and schedule a settle.

Stdlib only: templates and the worker are injected.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from examples.frontend_delegation_agent.core.delegation import protocol as proto
from examples.frontend_delegation_agent.core.delegation.context_queue import ContextEntry, ContextQueue
from examples.frontend_delegation_agent.core.prompt_features import render_backend_system, sha256_text

logger = logging.getLogger(__name__)

Emit = Callable[[dict[str, Any]], None]
Log = Callable[..., None]

#: Outcomes whose returned messages are committed (section 5.3).
_COMMIT_STATUSES = frozenset({"ok", "failed", "interrupted"})


class Templates(Protocol):
    """Backend prompt rendering (``sidecar.templates.BackendTemplates``)."""

    def render(self, key: str, **variables: Any) -> str:
        """Render one template."""
        ...


class WorkerPort(Protocol):
    """The worker handle the controller drives (``sidecar.worker_pool``)."""

    pid: int | None

    @property
    def alive(self) -> bool:
        """Whether the worker takes commands."""
        ...

    def set_listener(self, on_message: Callable[[dict[str, Any]], None], on_exit: Callable[[str], None]) -> None:
        """Bind callbacks."""
        ...

    async def start(self) -> dict[str, Any]:
        """Start and return ``ready``."""
        ...

    def send(self, message: dict[str, Any]) -> None:
        """Send one gateway→worker message."""
        ...

    async def stop(self) -> str:
        """Graceful stop with escalation."""
        ...

    async def kill(self, reason: str) -> None:
        """Kill now."""
        ...

    def rss_mb(self) -> float | None:
        """Resident memory."""
        ...


class WorkerGoneError(RuntimeError):
    """The worker died while a reply was awaited."""


@dataclass(frozen=True, slots=True)
class ControllerSettings:
    """What the controller needs from the gateway config (kept stdlib-only)."""

    run_hard_deadline_s: float = 330.0
    unwind_timeout_s: float = 10.0
    steer_timeout_s: float = 3.0
    status_timeout_s: float = 2.0
    configure_timeout_s: float = 30.0
    respawn: bool = True
    max_respawns: int = 2
    #: The ``backend`` block of the worker ``configure`` message (``GatewayConfig.worker_settings`` and more).
    worker_backend: dict[str, Any] = field(default_factory=dict)
    #: ``backend.kind`` of the gateway config, reported in the session fingerprint.
    backend_kind: str = ""
    #: ``(domain, tools_any)`` in detection order (gateway.yaml ``domains``).
    domains: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass(slots=True)
class ToolRecord:
    """One routed tool call."""

    call_id: str
    name: str
    arguments: str
    executed: bool = False
    open: bool = True
    local: bool = False
    result_preview: str = ""


@dataclass(slots=True)
class SteerRecord:
    """One steer/redirect accepted by the worker."""

    carrier: str
    text: str
    via: str
    turn_id: Any


@dataclass(slots=True)
class RunHandle:
    """One backend run (section 5.2)."""

    epoch: int
    run_id: str
    kind: str
    turn_ids: list[Any]
    words: list[str]
    started_at: float
    outcome: asyncio.Future[dict[str, Any]]
    preset_message: str | None = None
    dispatched: bool = False
    settled: bool = False
    delay_task: asyncio.Task[None] | None = None
    watchdog: asyncio.Task[None] | None = None
    steers: list[SteerRecord] = field(default_factory=list)
    tools: dict[str, ToolRecord] = field(default_factory=dict)
    current_tool: str | None = None
    steer_counter: int = 0


class BackendController:
    """The backend state machine of one Realtime session."""

    def __init__(
        self,
        *,
        session_id: str,
        settings: ControllerSettings,
        templates: Templates,
        emit: Emit,
        worker_factory: Callable[[], WorkerPort],
        log: Log | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Nothing starts until :meth:`open`."""
        self.session_id = session_id
        self._settings = settings
        self._templates = templates
        self._emit_raw = emit
        self._worker_factory = worker_factory
        self._log_fn = log
        self._clock = clock
        self._lock = asyncio.Lock()
        self.state = proto.NO_SESSION
        self.context = ContextQueue()
        self.history: list[dict[str, Any]] = []
        self._epoch = 0
        self._run: RunHandle | None = None
        self._worker: WorkerPort | None = None
        self._waiters: list[tuple[str, Any, asyncio.Future[dict[str, Any]]]] = []
        self._tools: list[dict[str, Any]] = []
        self._instructions = ""
        self._configured = False
        self._agent_built = False
        self._runs_started = 0
        self._committed_any = False
        self._respawns = 0
        self._backend_failed = False
        self._closing = False
        self._session_settings: dict[str, Any] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._closed_event = asyncio.Event()

    # -- voice → gateway ------------------------------------------------------------------

    async def open(self, settings: dict[str, Any]) -> None:
        """``session.open``: remember settings and start the worker early (hides startup latency)."""
        self._session_settings = dict(settings or {})
        async with self._lock:
            try:
                ready = await self._spawn_locked()
            except Exception as exc:  # noqa: BLE001 - reported to the voice side
                self._emit("error", code="worker_start_timeout", message=str(exc), fatal=True)
                return
            self._emit("session.ready", worker_pid=ready.get("pid"), python=ready.get("python"))

    async def configure(self, tools: list[dict[str, Any]], instructions: str) -> None:
        """``session.configure`` (transactional on the voice side; section 4.3)."""
        async with self._lock:
            self._settle_locked()
            names = [str(t.get("name")) for t in tools]
            if self._configured and (self._agent_built or self._runs_started > 0):
                if tools == self._tools and instructions == self._instructions:
                    self._emit("session.configured", applied=False, tools=self._tool_names(), **self._fingerprint())
                    return
                if self._session_settings.get("update_after_start", "error") == "ignore":
                    self._log("session_update_ignored", tools=names)
                    self._emit("session.configured", applied=False, tools=self._tool_names(), **self._fingerprint())
                else:
                    self._emit(
                        "error",
                        code="tools_locked",
                        message="tools and instructions cannot change once the backend agent exists",
                        fatal=False,
                    )
                return
            if self._worker is None or not self._worker.alive:
                self._emit(
                    "error", code="worker_start_timeout", message="the backend worker is not running", fatal=False
                )
                return
            self._tools, self._instructions = list(tools), str(instructions)
            error = await self._configure_worker_locked()
            if error is not None:
                self._emit("error", code="construct_failed", message=error, fatal=False)
                return
            fingerprint = self._fingerprint()
            self._log("backend_fingerprint", **fingerprint)
            self._emit(
                "session.configured", applied=True, tools=self._tool_names(), built=self._agent_built, **fingerprint
            )

    def _fingerprint(self) -> dict[str, Any]:
        """What this session's Hermes actually runs.

        Optional ``session.configured`` fields: the voice server logs them in ``backend_configured``.
        """
        fingerprint = {
            "backend_kind": self._settings.backend_kind,
            "backend_features": dict(getattr(self._templates, "features", {}) or {}),
            "backend_catalog_sha256": str(getattr(self._templates, "catalog_sha256", "") or ""),
            "backend_soul_sha256": sha256_text(self._templates.render("backend_soul")),
            "backend_system_sha256": sha256_text(self._system_prompt()),
            "backend_domain": self._domain(),
        }
        harness = self._settings.worker_backend.get("harness")
        if isinstance(harness, dict):  # the LLM backend's own prompt text (llm.yaml harness_* keys)
            fingerprint["backend_harness_sha256"] = sha256_text(json.dumps(harness, sort_keys=True, ensure_ascii=False))
        return fingerprint

    def _domain(self) -> str:
        """The session's domain: the first configured domain with one of its tools in the tool list ("" = none)."""
        names = {str(t.get("name")) for t in self._tools}
        for domain, tools_any in self._settings.domains:
            if names.intersection(tools_any):
                return domain
        return ""

    def _system_prompt(self) -> str:
        """The rendered ``backend_system``: the one render input of both the worker and the fingerprint."""
        note_of = getattr(self._templates, "domain_note", None)
        note = note_of(self._domain()) if callable(note_of) else ""
        return render_backend_system(self._templates, self._instructions, note)

    def append(self, entries: list[dict[str, Any]]) -> None:
        """``history.append``: context entries (deduplicated by ``seq``)."""
        during_run = self.state == proto.WORKING
        for data in entries:
            entry = ContextEntry.from_wire(data, during_run=during_run)
            if not self.context.add(entry):
                self._log("context_duplicate", seq=entry.seq)

    async def delegate(self, turn_id: Any, request: str, run_input: list[dict[str, Any]]) -> None:
        """``delegate``: start, continue, steer or report status (section 3)."""
        entries = [
            ContextEntry(seq=int(item["seq"]), kind="user", text=str(item.get("text") or ""), origin="user")
            for item in run_input
        ]
        async with self._lock:
            self._settle_locked()
            if self._closing:
                return
            if self._backend_failed or self._worker is None or not self._worker.alive or not self._configured:
                self._unavailable_locked(turn_id, entries)
                return
            run = self._run
            if self.state == proto.WORKING and run is not None:
                if request == proto.REQUEST_STATUS:
                    await self._status_locked(run, turn_id, entries)
                else:
                    await self._steer_locked(run, turn_id, entries)
                return
            kind = "start" if self.state == proto.NO_SESSION else "continue"
            self._start_run_locked([turn_id], entries, kind=kind, request=request)

    def tool_result(self, call_id: str, output: str, epoch: Any, *, local: bool) -> None:
        """``tool.result`` from the voice side: forward to the worker unless stale."""
        run = self._run
        record = run.tools.get(call_id) if run is not None else None
        if run is None or record is None or run.settled or (epoch is not None and epoch != run.epoch):
            self._log("tool_result_ignored", call_id=call_id, reason="stale_epoch")
            return
        if not record.open:
            self._log("tool_result_ignored", call_id=call_id, reason="late")
            return
        record.open, record.local, record.executed = False, local, not local
        record.result_preview = output[:160]
        if self._worker is not None and self._worker.alive:
            with contextlib.suppress(Exception):
                self._worker.send({"type": "tool.result", "call_id": call_id, "output": output, "epoch": run.epoch})

    async def close(self) -> None:
        """Stop the run and the worker; nothing is reported to Hermes any more."""
        async with self._lock:
            if self._closing:
                return
            self._closing = True
            self.state = proto.CLOSING
            run = self._run
            if run is not None and not run.outcome.done():
                run.outcome.set_result({"status": "closed"})
            self._settle_locked()
            worker, self._worker = self._worker, None
        if worker is not None:
            with contextlib.suppress(Exception):
                how = await worker.stop()
                self._log("worker_stopped", pid=worker.pid, how=how)
        self._fail_waiters(WorkerGoneError("session closed"))
        for task in list(self._tasks):
            if not task.done():
                task.cancel()
        self.state = proto.CLOSED

    # -- worker → gateway ----------------------------------------------------------------------

    def on_worker_message(self, msg: dict[str, Any]) -> None:
        """One worker message (on the loop)."""
        kind = msg["type"]
        if kind in ("configured", "steer_result", "status_result", "error"):
            self._resolve_waiter(kind, msg)
            if kind == "error" and msg.get("phase") != "construct":
                self._log("worker_error", **{k: v for k, v in msg.items() if k != "v"})
            return
        if kind == "tool.call":
            self._on_tool_call(msg)
        elif kind == "tool.cancel":
            self._on_tool_cancel(msg)
        elif kind == "activity":
            self._on_activity(msg)
        elif kind == "event":
            self._log(str(msg["event"]), **{k: v for k, v in msg.items() if k not in ("v", "type", "event")})
        elif kind == "run_outcome":
            self._on_outcome(msg)
        elif kind == "closed":
            self._log("worker_closed", unwound=msg.get("unwound"))

    def on_worker_exit(self, reason: str) -> None:
        """The worker went away (crash, kill, socket loss or a requested stop)."""
        worker = self._worker
        if self._closing or worker is None:
            return
        self._log("worker_died", reason=reason, pid=worker.pid)
        self._emit("worker", event="died", pid=worker.pid, reason=reason)
        self._fail_waiters(WorkerGoneError(reason))
        run = self._run
        if run is not None and not run.outcome.done():
            run.outcome.set_result({"status": "worker_died", "error": reason})
        self._spawn_task(self._recover(reason))

    # -- runs -------------------------------------------------------------------------------

    def _start_run_locked(
        self,
        turn_ids: list[Any],
        entries: list[ContextEntry],
        *,
        kind: str,
        request: str = proto.REQUEST_TASK,
        preset_message: str | None = None,
    ) -> RunHandle:
        self._epoch += 1
        epoch = self._epoch
        run = RunHandle(
            epoch=epoch,
            run_id=f"{self.session_id}-run{epoch}",
            kind=kind,
            turn_ids=list(turn_ids),
            words=[e.text for e in entries],
            started_at=self._clock(),
            outcome=asyncio.get_running_loop().create_future(),
            preset_message=preset_message,
        )
        self.context.add_run_input(entries, epoch, f"{epoch}:start")
        self._run = run
        self._runs_started += 1
        self.state = proto.WORKING
        self._emit("state", state=self.state, epoch=epoch, run_id=run.run_id)
        for turn_id in turn_ids[:1]:
            self._emit("action", turn_id=turn_id, kind=kind, request=request, run_id=run.run_id, epoch=epoch)
        self._log("backend_run_started", run_id=run.run_id, epoch=epoch, turn_ids=turn_ids, kind=kind)
        delay = self._delay_s() if kind in ("start", "continue") else 0.0
        if delay > 0:
            run.delay_task = self._spawn_task(self._delayed_dispatch(run, delay))
        else:
            self._dispatch_locked(run)
        return run

    def _delay_s(self) -> float:
        delay = self._session_settings.get("simulated_delay") or {}
        if str(delay.get("where", "per_delegation")) != "per_delegation":
            return 0.0
        try:
            return max(0.0, float(delay.get("seconds", 0) or 0))
        except (TypeError, ValueError):
            return 0.0

    async def _delayed_dispatch(self, run: RunHandle, delay: float) -> None:
        await asyncio.sleep(delay)
        async with self._lock:
            if self._run is run and not run.settled and not run.dispatched:
                self._dispatch_locked(run)

    def _dispatch_locked(self, run: RunHandle) -> None:
        if run.preset_message is not None:
            user_message = run.preset_message
        else:
            block = self._render_block(self.context.take(run.epoch, f"{run.epoch}:start"))
            user_message = self._templates.render(
                "backend_delegation_message", block=block, text="\n".join(w for w in run.words if w)
            )
        run.dispatched = True
        try:
            assert self._worker is not None  # noqa: S101 - checked by the caller
            self._worker.send(
                {
                    "type": "run",
                    "epoch": run.epoch,
                    "task_id": f"{self.session_id}:{run.epoch}",
                    "user_message": user_message,
                    "conversation_history": self.history,
                }
            )
        except Exception as exc:  # noqa: BLE001 - the exit notice settles the run
            logger.warning("run dispatch failed: %s", exc)
            if not run.outcome.done():
                run.outcome.set_result({"status": "worker_died", "error": str(exc)})
                self._spawn_task(self._settle())
            return
        self._log("backend_run_dispatched", run_id=run.run_id, epoch=run.epoch, chars=len(user_message))
        run.watchdog = self._spawn_task(self._watchdog(run))

    async def _steer_locked(self, run: RunHandle, turn_id: Any, entries: list[ContextEntry]) -> None:
        if not run.dispatched:
            self.context.add_run_input(entries, run.epoch, f"{run.epoch}:start")
            run.words.extend(e.text for e in entries)
            run.turn_ids.append(turn_id)
            self._emit(
                "action",
                turn_id=turn_id,
                kind="queued_in_delay",
                request=proto.REQUEST_TASK,
                run_id=run.run_id,
                epoch=run.epoch,
            )
            self._log("backend_run_input", run_id=run.run_id, turn_id=turn_id, kind="queued_in_delay")
            return
        run.steer_counter += 1
        carrier = f"{run.epoch}:steer{run.steer_counter}"
        self.context.add_run_input(entries, run.epoch, carrier)
        block = self._render_block(self.context.take(run.epoch, carrier))
        text = self._templates.render("backend_steer", block=block, text="\n".join(e.text for e in entries if e.text))
        mode = str(self._session_settings.get("steer_mode") or "auto")
        reply: dict[str, Any] | None = None
        try:
            assert self._worker is not None  # noqa: S101
            self._worker.send({"type": "steer", "epoch": run.epoch, "text": text, "mode": mode})
            reply = await self._wait_reply("steer_result", run.epoch, self._settings.steer_timeout_s)
        except (WorkerGoneError, TimeoutError, Exception) as exc:  # noqa: BLE001
            self._log("steer_failed", run_id=run.run_id, error=str(exc))
        if reply is not None and reply.get("accepted"):
            via = str(reply.get("via") or "steer")
            run.steers.append(SteerRecord(carrier, text, via, turn_id))
            run.turn_ids.append(turn_id)
            self._emit(
                "action", turn_id=turn_id, kind=via, request=proto.REQUEST_TASK, run_id=run.run_id, epoch=run.epoch
            )
            self._log("backend_run_input", run_id=run.run_id, turn_id=turn_id, kind=via)
            return
        # Not accepted: the run ended in the worker (its outcome is on the way) or the worker is gone.
        words = self.context.requeue_carrier(run.epoch, carrier)
        with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(run.outcome), timeout=self._settings.steer_timeout_s)
        self._settle_locked()
        if self._run is None and self._worker is not None and self._worker.alive and not self._closing:
            kind = "start" if self.state == proto.NO_SESSION else "continue"
            self._start_run_locked([turn_id], words, kind=kind)
            return
        for entry in words:  # still running or no worker: the words wait as context
            entry.kind, entry.during_run = "user", True
            self.context.add(entry)
        self._emit(
            "action", turn_id=turn_id, kind="queued", request=proto.REQUEST_TASK, run_id=run.run_id, epoch=run.epoch
        )

    async def _status_locked(self, run: RunHandle, turn_id: Any, entries: list[ContextEntry]) -> None:
        for entry in entries:  # the question itself reaches Hermes later as context
            entry.kind, entry.during_run = "user", True
            self.context.add(entry)
        worker_summary: dict[str, Any] = {}
        if run.dispatched and self._worker is not None and self._worker.alive:
            try:
                self._worker.send({"type": "status", "epoch": run.epoch})
                reply = await self._wait_reply("status_result", run.epoch, self._settings.status_timeout_s)
                worker_summary = dict(reply.get("summary") or {})
            except (WorkerGoneError, TimeoutError, Exception) as exc:  # noqa: BLE001
                self._log("status_failed", run_id=run.run_id, error=str(exc))
        summary = {
            "task": " / ".join(w for w in run.words if w),
            "elapsed_s": round(self._clock() - run.started_at, 1),
            "tools_done": [
                {"name": r.name, "args_preview": r.arguments[:120], "result_preview": r.result_preview}
                for r in run.tools.values()
                if not r.open
            ],
            "outstanding": [r.name for r in run.tools.values() if r.open],
            "current_tool": worker_summary.get("current_tool") or run.current_tool,
            "api_call_count": worker_summary.get("api_call_count"),
            "started": run.dispatched,
            "steers": len(run.steers),
        }
        self._emit(
            "action", turn_id=turn_id, kind="status", request=proto.REQUEST_STATUS, run_id=run.run_id, epoch=run.epoch
        )
        self._emit("status", turn_id=turn_id, summary=summary)
        self._log("backend_status", run_id=run.run_id, turn_id=turn_id)

    # -- settling -------------------------------------------------------------------------------

    async def _settle(self) -> None:
        async with self._lock:
            self._settle_locked()

    def _settle_locked(self) -> None:
        """Apply a finished run's outcome exactly once (idempotent)."""
        run = self._run
        if run is None or run.settled or not run.outcome.done():
            return
        run.settled = True
        for task in (run.watchdog, run.delay_task):
            if task is not None and not task.done() and task is not asyncio.current_task():
                task.cancel()
        outcome = run.outcome.result()
        status = str(outcome.get("status") or "error")
        messages = outcome.get("messages")
        commit = status in _COMMIT_STATUSES and isinstance(messages, list)
        pending_text = str(outcome.get("pending_steer") or "") if commit else ""
        pending = (
            [s for s in run.steers if s.via == "steer" and not _delivered(s.text, messages)] if pending_text else []
        )
        next_epoch = self._epoch + 1
        if commit:
            if pending:
                self.context.move(run.epoch, next_epoch, [s.carrier for s in pending])
            self.history = list(messages)
            self._committed_any = True
            upto = self.context.commit(run.epoch)
            self._emit("history.ack", committed_upto=upto)
        else:
            notes = self._side_effect_notes(run)
            self.context.requeue(run.epoch, notes)
        for record in run.tools.values():
            if record.open:
                record.open = False
                self._emit("tool.cancel", call_id=record.call_id, reason=status)
        usage = _usage(outcome.get("usage_delta"))
        self._answer(run, status, str(outcome.get("final_response") or ""), usage)
        self._emit(
            "backend_run_done",
            run_id=run.run_id,
            epoch=run.epoch,
            status=status,
            history="committed" if commit else "rolled_back",
            turn_ids=run.turn_ids,
            usage=usage,
            tool_call_ids=list(run.tools),
            error=outcome.get("error"),
        )
        self._log(
            "backend_run_done",
            run_id=run.run_id,
            epoch=run.epoch,
            status=status,
            history="committed" if commit else "rolled_back",
            turn_ids=run.turn_ids,
            usage_delta=usage,
            tool_call_ids=list(run.tools),
            error=outcome.get("error"),
        )
        self._run = None
        if pending and not self._closing and self._worker is not None and self._worker.alive:
            self._start_run_locked([s.turn_id for s in pending], [], kind="pending_steer", preset_message=pending_text)
            return
        if self._closing:
            return
        self.state = proto.IDLE if self._committed_any else proto.NO_SESSION
        self._emit("state", state=self.state, epoch=run.epoch, run_id=None)

    def _answer(self, run: RunHandle, status: str, final: str, usage: dict[str, int]) -> None:
        if status == "closed" or self._closing:
            return
        answer_id = f"{run.run_id}-answer"
        if status == "ok" and final.strip():
            self._emit(
                "answer",
                answer_id=answer_id,
                run_id=run.run_id,
                epoch=run.epoch,
                kind="hermes",  # every backend's answer; the voice side cleans "hermes" answers (turn_manager)
                text=final,
                usage=usage,
                turn_ids=run.turn_ids,
            )
            return
        reason = "empty_response" if status == "ok" else status
        self._emit(
            "answer",
            answer_id=answer_id,
            run_id=run.run_id,
            epoch=run.epoch,
            kind="apology",
            text=self._templates.render("backend_error_spoken"),
            reason=reason,
            usage=usage,
            turn_ids=run.turn_ids,
        )

    def _side_effect_notes(self, run: RunHandle) -> list[ContextEntry]:
        executed = [r for r in run.tools.values() if r.executed]
        if not executed:
            return []
        calls = [{"name": r.name, "arguments": r.arguments, "result": r.result_preview} for r in executed]
        return [ContextEntry(seq=None, kind="side_effect_note", text="", origin="derived", extra={"calls": calls})]

    # -- watchdog and recovery ---------------------------------------------------------------------

    async def _watchdog(self, run: RunHandle) -> None:
        await asyncio.sleep(self._settings.run_hard_deadline_s)
        if run.outcome.done():
            return
        self._log("run_deadline", run_id=run.run_id, epoch=run.epoch)
        worker = self._worker
        if worker is not None and worker.alive:
            with contextlib.suppress(Exception):
                worker.send({"type": "interrupt", "hard": True})
        try:
            await asyncio.wait_for(asyncio.shield(run.outcome), timeout=self._settings.unwind_timeout_s)
        except TimeoutError:
            if not run.outcome.done():
                run.outcome.set_result({"status": "deadline", "error": "the run did not unwind; worker killed"})
            if worker is not None:
                self._log("worker_kill", pid=worker.pid, reason="deadline")
                await worker.kill("deadline")
        await self._settle()

    async def _recover(self, reason: str) -> None:
        async with self._lock:
            self._settle_locked()
            if self._closing:
                return
            old, self._worker = self._worker, None
            if old is not None:
                with contextlib.suppress(Exception):
                    await old.kill(f"replaced after: {reason}")
            if not self._settings.respawn or self._respawns >= self._settings.max_respawns:
                self._backend_failed = True
                self._log("backend_failed", reason=reason, respawns=self._respawns)
                return
            self._respawns += 1
            try:
                ready = await self._spawn_locked()
            except Exception as exc:  # noqa: BLE001
                self._backend_failed = True
                self._log("respawn_failed", error=str(exc))
                return
            if self._configured:
                error = await self._configure_worker_locked()
                if error is not None:
                    self._backend_failed = True
                    self._log("respawn_configure_failed", error=error)
                    return
            self._emit("worker", event="respawned", pid=ready.get("pid"), reason=reason)

    async def _spawn_locked(self) -> dict[str, Any]:
        worker = self._worker_factory()
        worker.set_listener(self.on_worker_message, self.on_worker_exit)
        started = self._clock()
        ready = await worker.start()
        self._worker = worker
        start_ms = getattr(worker, "start_ms", None) or int((self._clock() - started) * 1000)
        self._emit("worker", event="ready", pid=ready.get("pid"), start_ms=start_ms, rss_mb=worker.rss_mb())
        self._log("worker_ready", pid=ready.get("pid"), start_ms=start_ms, rss_mb=worker.rss_mb())
        return ready

    async def _configure_worker_locked(self) -> str | None:
        assert self._worker is not None  # noqa: S101
        tools = []
        for t in self._tools:
            tool = {
                "name": t.get("name"),
                "description": t.get("description") or "",
                "parameters": t.get("parameters") or {},
            }
            if isinstance(t.get("annotations"), dict):  # MCP hints (readOnlyHint); never sent to the model
                tool["annotations"] = dict(t["annotations"])
            tools.append(tool)
        system = self._system_prompt()
        backend = dict(self._settings.worker_backend)
        if self._settings.backend_kind == "llm":
            # The client's own policy (not the rendered backend prompt) and the session domain, for the
            # policy follow-up rules (core/backends/llm/follow_up.py).
            backend["client_instructions"] = self._instructions
            backend["session_domain"] = self._domain()
        try:
            self._worker.send({"type": "configure", "tools": tools, "instructions": system, "backend": backend})
            reply = await self._wait_reply(("configured", "error"), None, self._settings.configure_timeout_s)
        except (WorkerGoneError, TimeoutError, Exception) as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"
        if reply["type"] == "error":
            return str(reply.get("message"))
        self._configured = True
        self._agent_built = bool(reply.get("built"))
        return None

    def _unavailable_locked(self, turn_id: Any, entries: list[ContextEntry]) -> None:
        for entry in entries:
            self.context.add(entry)
        self._emit(
            "action", turn_id=turn_id, kind="unavailable", request=proto.REQUEST_TASK, run_id=None, epoch=self._epoch
        )
        self._emit(
            "answer",
            answer_id=f"{self.session_id}-unavailable-{turn_id}",
            run_id=None,
            epoch=self._epoch,
            kind="apology",
            text=self._templates.render("backend_unavailable"),
            reason="backend_unavailable",
            usage=_usage(None),
            turn_ids=[turn_id],
        )

    # -- worker message handlers ----------------------------------------------------------------------

    def _on_tool_call(self, msg: dict[str, Any]) -> None:
        run = self._run
        if run is None or run.settled or msg.get("epoch") != run.epoch:
            self._log("tool_call_ignored", call_id=msg.get("call_id"), reason="stale_epoch")
            return
        record = ToolRecord(call_id=str(msg["call_id"]), name=str(msg["name"]), arguments=str(msg["arguments"]))
        run.tools[record.call_id] = record
        self._emit(
            "tool.call",
            call_id=record.call_id,
            epoch=run.epoch,
            run_id=run.run_id,
            name=record.name,
            arguments=record.arguments,
        )

    def _on_tool_cancel(self, msg: dict[str, Any]) -> None:
        run = self._run
        record = run.tools.get(str(msg["call_id"])) if run is not None else None
        if record is not None and record.open:
            record.open = False
            self._emit("tool.cancel", call_id=record.call_id, reason=str(msg.get("reason") or "cancelled"))

    def _on_activity(self, msg: dict[str, Any]) -> None:
        run = self._run
        if run is None or msg.get("epoch") != run.epoch:
            return
        run.current_tool = str(msg["name"]) if msg.get("event") == "tool_started" else None
        self._emit("activity", epoch=run.epoch, event=msg["event"], name=msg["name"], preview=msg.get("preview", ""))

    def _on_outcome(self, msg: dict[str, Any]) -> None:
        run = self._run
        if run is None or msg.get("epoch") != run.epoch:
            self._log("outcome_ignored", epoch=msg.get("epoch"), reason="stale_epoch")
            return
        if not run.outcome.done():
            run.outcome.set_result({k: v for k, v in msg.items() if k not in ("v", "type")})
        self._spawn_task(self._settle())

    # -- replies ----------------------------------------------------------------------------------------

    async def _wait_reply(self, kinds: str | tuple[str, ...], epoch: Any, timeout: float) -> dict[str, Any]:
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        entry = (kinds if isinstance(kinds, tuple) else (kinds,), epoch, future)
        self._waiters.append(entry)  # type: ignore[arg-type]
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            with contextlib.suppress(ValueError):
                self._waiters.remove(entry)  # type: ignore[arg-type]

    def _resolve_waiter(self, kind: str, msg: dict[str, Any]) -> None:
        for kinds, epoch, future in list(self._waiters):
            if kind in kinds and (epoch is None or msg.get("epoch") == epoch) and not future.done():
                future.set_result(msg)
                return

    def _fail_waiters(self, exc: BaseException) -> None:
        for _kinds, _epoch, future in list(self._waiters):
            if not future.done():
                future.set_exception(exc)

    # -- helpers ------------------------------------------------------------------------------------------

    def _render_block(self, entries: list[ContextEntry]) -> str:
        lines = [line for line in (self._render_entry(e) for e in entries if not e.run_input) if line]
        if not lines:
            return ""
        return self._templates.render("context_block", lines=lines)

    def _render_entry(self, entry: ContextEntry) -> str:
        r = self._templates.render
        partial = entry.outcome == "partial"
        spoken = entry.heard_text if partial else entry.text
        if entry.kind == "frontend_speech":
            return r("context_frontend", text=spoken, partial=partial)
        if entry.kind == "status_speech":
            return r("context_status", text=spoken, partial=partial)
        if entry.kind == "controller_speech":
            return r("context_controller", text=spoken, partial=partial)
        if entry.kind == "delivery_note":
            return r("context_delivery_note", outcome=entry.outcome, heard_text=entry.heard_text)
        if entry.kind == "requeued_request":
            return r("context_requeued_request", text=entry.text)
        if entry.kind == "side_effect_note":
            return r("context_side_effect", calls=entry.extra.get("calls", []))
        return r("context_user", text=entry.text, during_run=entry.during_run)

    def _tool_names(self) -> list[str]:
        return sorted(str(t.get("name")) for t in self._tools)

    def _emit(self, type_: str, **fields: Any) -> None:
        try:
            self._emit_raw({"type": type_, **fields})
        except Exception as exc:  # noqa: BLE001 - the voice link may be gone during close
            logger.debug("emit %s failed: %s", type_, exc)

    def _log(self, event: str, /, **data: Any) -> None:
        if self._log_fn is not None:
            with contextlib.suppress(Exception):
                self._log_fn(event, **data)

    def _spawn_task(self, coro: Any) -> asyncio.Task[Any]:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task


def _delivered(text: str, messages: Any) -> bool:
    """Whether a steer's text reached the backend's history (undelivered ones come back as ``pending_steer``)."""
    if not isinstance(messages, list):
        return False
    needle = text.strip()[:200]
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str) and needle and needle in content:
            return True
        if isinstance(content, list) and needle and needle in json.dumps(content, ensure_ascii=False):
            return True
    return False


def _usage(data: Any) -> dict[str, int]:
    data = data if isinstance(data, dict) else {}
    return {
        "input_tokens": int(data.get("input_tokens", 0) or 0),
        "output_tokens": int(data.get("output_tokens", 0) or 0),
        "reasoning_tokens": int(data.get("reasoning_tokens", 0) or 0),
    }
