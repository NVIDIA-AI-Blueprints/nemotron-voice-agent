# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One IM session's harness backend: its conductor, its worker runs and its runtime processes."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from loguru import logger

from examples.frontend_backend_live.delegation_core.backend.conductors import SteerOpenWork
from examples.frontend_backend_live.delegation_core.backend.port import (
    BackendRun,
    ControlContext,
    ControlRun,
    ControlTools,
    TaskContext,
    ToolSource,
)

from .bridge import Door
from .conductor import HarnessConductor
from .control import BorrowedControl
from .protocol import NativeCleanup, Role
from .render import Renderer
from .runtime import RuntimeOpener, RuntimeProcess
from .sessions import SessionPool
from .worker import WorkerRun

if TYPE_CHECKING:
    from examples.frontend_backend_live.delegation_core.backend.llm.adapter import LlmAdapter

    from .sessions import KeptSession
    from .source import HarnessSource

_DRAINS: set[asyncio.Task] = set()


class HarnessAdapter:
    """The backend of one IM session on the harness (``port.BackendAdapter``)."""

    def __init__(
        self,
        source: HarnessSource,
        *,
        tools: Sequence[ToolSource],
        on_drained: Callable[[HarnessAdapter], None] | None = None,
    ):
        """Serve one IM session from ``source``, with its server tool sources; nothing starts before conductor()."""
        self.source, self.settings = source, source.settings
        self.max_running = self.settings.max_running
        self.tools = tuple(tools)
        self.vocabulary: ControlTools = source.vocabulary
        self.bridge = source.bridge
        self.retirements = source.retirements
        self.renderer = Renderer(self.settings.prompts)
        self.opener = RuntimeOpener(self.settings)
        self.conductor_role = Role("conductor", "c")
        self.workers = Role("workers", "w")
        self.closed = False
        self._started = False
        self._conductor: HarnessConductor | None = None
        self._restricted: LlmAdapter | None = None
        self._shared: dict[str, list[asyncio.Task]] = {}
        self._runs: set[WorkerRun] = set()
        self._doors: set[Door] = set()
        self._draining: list[RuntimeProcess] = []
        self._closing: asyncio.Task | None = None
        self._drain: asyncio.Task | None = None
        self._on_drained = on_drained
        self._close_errors: list[BaseException] = []
        self.pool = SessionPool(self.max_running)

    def can_run_without_attachment(self) -> bool:
        """Whether every native tool effect is covered by the IM's attachment fence."""
        return self.settings.agent.capabilities.tool_mediation == "bridge_only"

    def usage_identity(self, task: TaskContext) -> dict:
        """Identify the fixed native model, which may differ from the requested model."""
        if task.business_access != "full" and not self.can_run_without_attachment():
            return self._restricted_adapter().usage_identity(task)
        return {
            "backend": "harness",
            "protocol": self.settings.protocol_name,
            "model": self.source.endpoint().model,
            "model_source": "configured_endpoint",
        }

    def conductor(self) -> HarnessConductor | SteerOpenWork:
        """Return the session's conductor, and start the runtime processes in the background.

        At ``max_running`` 1 it is the steer rule, and no conductor process starts.
        """
        if not self._started:
            self._started = True
            if self.max_running > 1:
                self._conductor = HarnessConductor(self)
                self._conductor.warm()
            self._warm(self.workers, first_session=True)
        return self._conductor or SteerOpenWork()

    def start(self, task: TaskContext) -> BackendRun:
        """Start native work or route a restricted task through the Core-controlled tool loop."""
        if task.business_access != "full" and not self.can_run_without_attachment():
            return self._restricted_adapter().start(task)
        return self._start_native(task)

    def _start_native(self, task: TaskContext) -> WorkerRun:
        """Create a native worker and pool idle sessions; neither opens until its events are read."""
        model, served = task.request().get("model"), self.source.endpoint().model
        if model not in (None, served):
            logger.info(f"harness.other_model task={task.task_id} model={model!r}: the agent runs its own model")
        self.pool.open()
        run = WorkerRun(task, self)
        self._runs.add(run)
        return run

    def start_control(self, task: ControlContext, *, parent: BackendRun | None = None) -> ControlRun:
        """Interpret input under Core-only gates while keeping the original task and its effects."""
        if isinstance(parent, WorkerRun) and (parent.host is not self or parent.task is not task.original_context):
            raise ValueError("A control foreground requires this adapter's original worker")
        if not self.can_run_without_attachment():
            prepare = parent.prepare_restricted_control if isinstance(parent, WorkerRun) else None
            return self._restricted_adapter().start_control(task, parent=parent, prepare=prepare)
        if parent is None:
            parent = self._start_native(task.original_context)
        if not isinstance(parent, WorkerRun):
            raise ValueError("A native control foreground requires this adapter's original worker")
        return BorrowedControl(parent, task)

    def _restricted_adapter(self) -> LlmAdapter:
        """Reuse one provider pool for this session's restricted invocations."""
        if self.closed:
            raise RuntimeError("The harness session is closing")
        if self._restricted is None:
            self._restricted = self.source.new_restricted(self.tools)
        return self._restricted

    def release_idle(self) -> None:
        """Retire every idle session, and pool none until a task starts: the caller left, so no task takes one."""
        self._retire_idle(self.pool.close(), "the pool closed")

    async def close(self) -> None:
        """Fence and logically close once; native physical cleanup remains owned for final source shutdown."""
        if self._closing is None:
            self.closed = True
            self._closing = asyncio.create_task(self._close(), name="harness-adapter-close")
            self._closing.add_done_callback(lambda done: done.cancelled() or done.exception())
        await asyncio.shield(self._closing)

    async def wait_closed(self) -> None:
        """Join only this adapter's physical cleanup, without cancelling its logical close or native owners."""
        results = await asyncio.gather(self.close(), return_exceptions=True)
        if self._drain is not None:
            results.extend(await asyncio.gather(asyncio.shield(self._drain), return_exceptions=True))
        for result in [*self._close_errors, *results]:
            if isinstance(result, BaseException):
                raise result

    async def _close(self) -> None:
        if self._conductor is not None:
            try:
                await self._conductor.close()
            except BaseException as error:
                self._close_errors.append(error)
        released = await asyncio.gather(*(run.release() for run in self._runs), return_exceptions=True)
        self._close_errors.extend(result for result in released if isinstance(result, BaseException))
        self._runs.clear()
        if self._restricted is not None:
            try:
                await self._restricted.close()
            except BaseException as error:
                self._close_errors.append(error)
        self._retire_idle(self.pool.drain(), "the call's end")
        opened, self._shared = [task for pool in self._shared.values() for task in pool], {}
        for task in opened:
            if not task.done():
                task.cancel()
        results = await asyncio.gather(*opened, return_exceptions=True)
        processes = [result for result in results if not isinstance(result, BaseException)]
        stopped = await asyncio.gather(*(process.close() for process in processes), return_exceptions=True)
        self._close_errors.extend(result for result in stopped if isinstance(result, BaseException))
        processes += self._draining
        self._draining = []
        for door in self._doors:
            self.bridge.close(door, "")
        doors, self._doors = self._doors, set()

        async def forget() -> None:
            # Release the bridge state before waiting for native workspace cleanup.
            calls = {call for door in doors for call in door.pending_calls}
            await asyncio.gather(*(asyncio.shield(call) for call in calls), return_exceptions=True)
            for door in doors:
                self.bridge.forget(door)
            unique = list(dict.fromkeys(processes))
            results = await asyncio.gather(*(process.wait_closed() for process in unique), return_exceptions=True)
            receipts = {
                process: result if isinstance(result, NativeCleanup) else process.cleanup_receipt
                for process, result in zip(unique, results, strict=True)
            }
            for process, receipt in receipts.items():
                if isinstance(receipt, NativeCleanup) and receipt.confirmed and receipt.scope == "runtime":
                    self.source.retirements.native_retired(process)
            results.extend(await asyncio.gather(self.opener.wait_closed(), return_exceptions=True))
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            if self._on_drained is not None and not self._close_errors:
                self._on_drained(self)

        self._drain = asyncio.create_task(forget(), name="harness-door-cleanup")
        self._drain.add_done_callback(_logged)
        _DRAINS.add(self._drain)
        self._drain.add_done_callback(_DRAINS.discard)

    def own_door[T: Door](self, door: T) -> T:
        """Keep a session's tool door until this adapter closes."""
        self._doors.add(door)
        return door

    async def shared(self, role: Role) -> RuntimeProcess:
        """Return a process of ``role`` with a turn free, its place taken for the caller's session.

        The place is taken in the same step that finds it free, so two sessions never take the last one.
        """
        while True:
            if self.closed:
                raise RuntimeError("The harness session is closing")
            pool = self._shared.setdefault(role.name, [])
            lost = [opening for opening in pool if opening.done() and _lost(opening)]
            if lost:
                for opening in lost:
                    pool.remove(opening)
                alive = [opening.result() for opening in lost if not opening.cancelled() and not opening.exception()]
                await asyncio.gather(*(process.close() for process in alive), return_exceptions=True)
                self._draining.extend(alive)
                # Look again from the start: the session may have closed meanwhile.
                continue
            free = next((each.result() for each in pool if each.done() and not self._full(each.result())), None)
            if free is not None:
                free.places += 1
                return free
            pending = next((opening for opening in pool if not opening.done()), None) or self._warm(role)
            process = await asyncio.shield(pending)
            if not process.client.alive:
                raise RuntimeError(f"The {role.name} runtime process ended as it started")
            # Look again: a session that waited for the same process may have taken its last place.

    def take_place(self, runtime: RuntimeProcess) -> bool:
        """Take a place in ``runtime`` for the session kept there, while it has a turn free and its client is alive."""
        if self._full(runtime) or not runtime.client.alive:
            return False
        runtime.places += 1
        return True

    def candidates(self, task: TaskContext) -> list[KeptSession]:
        """Return the idle sessions of ``task``'s thread in the order it tries them; a new thread has none."""
        return self.pool.candidates(task)

    def take_kept(self, kept: KeptSession) -> None:
        """Take ``kept`` out of the pool, so two tasks never share it."""
        self.pool.take(kept)

    def keep(self, kept: KeptSession) -> KeptSession | None:
        """Pool ``kept``; return the session idle longest when more than ``max_running`` are idle, to retire.

        While the pool is closed, it returns ``kept`` itself.
        """
        return self.pool.keep(kept)

    def _retire_idle(self, idle: list[KeptSession], why: str) -> None:
        """Retire the idle sessions ``idle``, each logged with ``why``."""
        for kept in idle:
            closed = kept.retire(self.bridge)
            logger.debug(
                f"harness.session_retired server={kept.door.name} session={kept.session.id} why={why} closed={closed}"
            )

    def _full(self, process: RuntimeProcess) -> bool:
        limit = self.settings.turns_per_process
        return limit is not None and process.places >= limit

    def _warm(self, role: Role, *, first_session: bool = False) -> asyncio.Task:
        """Start opening one more process of ``role`` in the background; a failure is only logged."""
        opening = asyncio.ensure_future(self._open(role, first_session))
        opening.add_done_callback(_logged)
        self._shared.setdefault(role.name, []).append(opening)
        return opening

    async def _open(self, role: Role, first_session: bool = False) -> RuntimeProcess:
        await self.bridge.start()
        process = await self.opener.open(role)
        if first_session:
            try:
                await self._first_session(process, role)
            except BaseException:
                self._draining.append(process)
                try:
                    await process.close()
                except BaseException as error:
                    self._close_errors.append(error)
                    raise
                raise
        return process

    async def _first_session(self, process: RuntimeProcess, role: Role) -> None:
        """Open one session that serves nothing and retire it, so no task pays the process's cold first session.

        An agent that does not close the session keeps its MCP server and pings it, and a refused ping makes the agent
        reconnect and then pause that server: so its door serves nothing until the adapter closes.
        """
        name = process.server_name(role.prefix)
        door = self.own_door(Door(name))
        server = self.bridge.open(door)
        session = None
        try:
            folder = process.session_folder(name)
            self.settings.agent.prepare(folder, "")
            session = await process.client.new_session(folder, server, asyncio.Queue(), role=role.name)
            async with asyncio.timeout(self.settings.ready_timeout_seconds):
                await door.listed.wait()
            logger.debug(f"harness.first_session_opened role={role.name} server={name}")
        except Exception as exc:
            # The warm-up only saves time: the first task opens its own session either way.
            logger.warning(f"harness.first_session_failed role={role.name} {type(exc).__name__}: {exc}")
        finally:
            if session is not None:
                session.retire()
            self.bridge.close(door, "")


def _lost(opening: asyncio.Future) -> bool:
    return opening.cancelled() or opening.exception() is not None or not opening.result().client.alive


def _logged(task: asyncio.Future) -> None:
    if not task.cancelled() and task.exception() is not None:
        logger.warning(f"harness.runtime_open_failed {type(task.exception()).__name__}: {task.exception()}")
