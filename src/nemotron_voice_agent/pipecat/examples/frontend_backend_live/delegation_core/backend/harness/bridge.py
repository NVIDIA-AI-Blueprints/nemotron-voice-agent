# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tool bridge: the IM's tools for the agent, one MCP server over streamable HTTP with a door per session."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import secrets
import socket
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlsplit

from loguru import logger

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import (
    ControlTools,
    Decision,
    DecisionRejected,
    Outcome,
    TaskContext,
    ToolNotExecuted,
    ToolSource,
)
from nemotron_voice_agent.pipecat.shared.ids import uid

from .protocol import (
    AdmissionReady,
    AppCall,
    CallRunning,
    CallSettled,
    HeldCall,
    Mailbox,
    McpServer,
    RefusedReply,
    refused_reply,
)
from .settings import Bridge

UNKNOWN_LABEL = "harness"
MCP_PATH = "/mcp"
OPERATION_SCOPE = "nvidia.voice_agent/native_operation"
REFUSAL_META = "nvidia.voice_agent/refusal"
NATIVE_OPERATION: contextvars.ContextVar[str | None] = contextvars.ContextVar("native_bridge_operation", default=None)
type ConductorSubmission = Callable[[Decision | DecisionRejected], Awaitable[str]]


@dataclass(frozen=True)
class ForegroundBinding:
    """A reversible logical dispatch binding; retained original handlers keep their own immutable owners."""

    task: TaskContext
    owners: Mapping[str, ToolSource]
    context: contextvars.Context
    pause: object
    retained: frozenset[str]


def mcp_tool(tool: Mapping[str, Any]) -> dict:
    """Convert a Responses function tool to an MCP tool: its name, description and JSON schema."""
    return {
        "name": tool["name"],
        "description": tool.get("description") or tool["name"],
        "input_schema": tool.get("parameters") or {"type": "object", "properties": {}},
    }


class Door:
    """One session's door on the bridge: its server name, its bearer token and its toolset; a bare door serves none."""

    role = "warmup"

    def __init__(self, name: str):
        """Open a door named ``name``, with a fresh token; its calls run in a copy of the context it is opened in."""
        self.name = name
        self.token = secrets.token_urlsafe(24)
        # Set by the agent's tools/list: a failed MCP connect is silent at the ACP level.
        self.listed = asyncio.Event()
        self.retired = False
        self.retired_output = ""
        self.context = contextvars.copy_context()
        self.pending_calls: set[asyncio.Task] = set()

    def tools(self) -> list[dict]:
        """Return the tools it serves, as MCP tools: none."""
        return []

    async def call(self, name: str, arguments: dict) -> str | RefusedReply:
        """Serve one call and return its plain-text result; a bare door answers empty."""
        return ""

    def capture_call(self) -> Callable[[str, dict], Awaitable[str | RefusedReply]]:
        """Capture first HTTP admission's handler; specialized doors also freeze their mutable logical binding."""
        return self.call

    def failed(self, name: str, error: str) -> str:
        """Return the answer to a call that raised; ``error`` is the exception's type and message."""
        return error

    def retire(self, text: str) -> None:
        """Serve nothing more; held calls are answered with ``text``."""
        self.retired = True
        self.retired_output = text


@dataclass(frozen=True)
class Operation:
    """Bind an HTTP request to the handler and context present when it arrived."""

    id: str
    door: Door
    call: Callable[[str, dict], Awaitable[str | RefusedReply]]
    context: contextvars.Context

    @classmethod
    def capture(cls, door: Door) -> Operation:
        """Capture the door's immutable foreground binding."""
        return cls(uid("native_operation"), door, door.capture_call(), door.context.copy())


class ConductorDoor(Door):
    """Expose the conductor's control tools and report invalid submissions to its active turn."""

    role = "conductor"

    def __init__(self, name: str, vocabulary: ControlTools):
        """Create the door with no active placement."""
        super().__init__(name)
        self.vocabulary = vocabulary
        self.submission: ConductorSubmission | None = None

    def bind(self, submission: ConductorSubmission) -> None:
        """Bind submissions and their execution context to the current decision round."""
        self.submission = submission
        self.context = contextvars.copy_context()

    def tools(self) -> list[dict]:
        """Return the control tools."""
        return [mcp_tool(tool) for tool in self.vocabulary.conductor_tools()]

    async def call(self, name: str, arguments: dict) -> str:
        """Submit a decision or a validation failure and return the turn's receipt."""
        return await self._decide(name, arguments, self.submission)

    def capture_call(self) -> Callable[[str, dict], Awaitable[str]]:
        """Keep a delayed first invocation from applying through a later conductor placement binding."""
        submission = self.submission

        async def captured(name: str, arguments: dict) -> str:
            if self.submission is not submission:
                return self.vocabulary.idle
            return await self._decide(name, arguments, submission)

        return captured

    async def _decide(self, name: str, arguments: dict, submission: ConductorSubmission | None) -> str:
        try:
            decision = self.vocabulary.decision(name, json.dumps(arguments))
        except ValueError as exc:
            logger.warning(f"harness.conductor_call_malformed name={name} error={exc}")
            if submission is not None:
                return await submission(DecisionRejected(str(exc)))
            return json.dumps({"error": str(exc)})
        if submission is None:
            return self.vocabulary.idle
        return await submission(decision)

    def failed(self, name: str, error: str) -> str:
        """Return the vocabulary's ``failed`` answer."""
        return self.vocabulary.failed.format(name=name, error=error)


class WorkerDoor(Door):
    """A worker's door: the app's functions, the tool sources' and the IM's work tools, for one task."""

    role = "workers"

    def __init__(
        self,
        name: str,
        task: TaskContext,
        sources: Sequence[ToolSource],
        mailbox: Mailbox,
        *,
        tool_timeout_seconds: float,
        invocation_active: Callable[[], bool] = lambda: True,
    ):
        """Serve ``task``'s toolset, reporting each call to its run through ``mailbox``."""
        super().__init__(name)
        self.task, self.mailbox, self.limit = task, mailbox, tool_timeout_seconds
        self.invocation: object | None = None
        self.invocation_active = invocation_active
        self._call_invocations: dict[str, object | None] = {}
        self._call_tasks: dict[str, asyncio.Task] = {}
        self.app_functions, self.owners, self.served = self._compose(task, sources)
        self.held: list[HeldCall] = []
        self.admitting: set[asyncio.Task] = set()
        self.app_admissions: dict[str, asyncio.Task] = {}
        self._admission_changed = asyncio.Event()
        self._admission_stopped = False
        self._admission_pause: object | None = None
        self._admission_output = ""
        self._placement_started: float | None = None
        self._placement_seconds = 0.0
        self._borrowed_binding: ForegroundBinding | None = None
        self._activity_callbacks: dict[object, Callable[[], None]] = {}

    def _compose(
        self, task: TaskContext, sources: Sequence[ToolSource], *, warn: bool = True
    ) -> tuple[set[str], dict[str, ToolSource], list[dict]]:
        """Build the toolset: the request's functions, each tool source's, then the work tools', each name once."""
        app_functions: set[str] = set()
        owners: dict[str, ToolSource] = {}
        served: list[dict] = []
        for tool in task.request().get("tools") or []:
            if tool.get("type") == "function" and tool.get("name") and tool["name"] not in app_functions:
                app_functions.add(tool["name"])
                served.append(mcp_tool(tool))
        for source in (*sources, task.tools()):
            for tool in source.tools():
                name = tool.get("name")
                if not name or name in app_functions or name in owners:
                    if warn:
                        logger.warning(f"harness.tool_collision door={self.name} name={name}: offered once only")
                    continue
                owners[name] = source
                served.append(mcp_tool(tool))
        return app_functions, owners, served

    def serves(self, task: TaskContext, sources: Sequence[ToolSource]) -> bool:
        """Return whether it can serve ``task`` as it is: the same toolset, no call held, not retired (no change)."""
        return (
            not (self.pending_calls or self.held or self.admitting or self.app_admissions or not self.admission_open)
            and self._compose(task, sources, warn=False)[2] == self.served
        )

    def rebind(self, task: TaskContext, sources: Sequence[ToolSource]) -> bool:
        """Serve ``task`` under the same name and token if its toolset is the one served; else return ``False``.

        The agent keeps the tools it listed; a door that holds a call or is retired stays as it was.
        """
        app_functions, owners, served = self._compose(task, sources)
        if (
            self.pending_calls
            or self.held
            or self.admitting
            or self.app_admissions
            or not self.admission_open
            or served != self.served
        ):
            return False
        self.task, self.app_functions, self.owners = task, app_functions, owners
        self.context = contextvars.copy_context()
        return True

    def tools(self) -> list[dict]:
        """Return the task's toolset."""
        return list(self.served)

    def capture_call(self) -> Callable[[str, dict], Awaitable[str | RefusedReply]]:
        """Bind original Task/invocation before SDK dispatch; a later rebind cannot grant it new authority."""
        task, invocation, execution = self.task, self.invocation, self.task.execution_ref
        active_on_arrival = self.invocation_active()
        functions, owners = frozenset(self.app_functions), dict(self.owners)

        async def captured(name: str, arguments: dict) -> str | RefusedReply:
            if (
                not active_on_arrival
                or self.task is not task
                or self.invocation is not invocation
                or self.task.execution_ref != execution
            ):
                return task.texts.not_run_output
            return await self._call_bound(name, arguments, task, invocation, functions, owners)

        return captured

    async def call(self, name: str, arguments: dict) -> str | RefusedReply:
        """Hold an app call for the run; run any other through ``route()``, ``tool_started`` and ``tool_outcome``."""
        return await self._call_bound(
            name,
            arguments,
            self.task,
            self.invocation,
            frozenset(self.app_functions),
            dict(self.owners),
        )

    async def _call_bound(
        self,
        name: str,
        arguments: dict,
        task: TaskContext,
        invocation: object | None,
        functions: frozenset[str],
        owners: Mapping[str, ToolSource],
    ) -> str | RefusedReply:
        if not self.admission_open:
            return self._admission_output
        call = {
            "type": "function_call",
            "id": uid("fc"),
            "call_id": NATIVE_OPERATION.get() or uid("call"),
            "name": name,
            "arguments": json.dumps(arguments),
            "status": "completed",
        }
        self._call_invocations[call["call_id"]] = invocation
        self._call_tasks[call["call_id"]] = asyncio.current_task()
        try:
            return await self._serve(call, task, functions, owners)
        finally:
            self._call_invocations.pop(call["call_id"], None)
            self._call_tasks.pop(call["call_id"], None)
            for token, callback in tuple(self._activity_callbacks.items()):
                if self._activity_callbacks.get(token) is callback:
                    with contextlib.suppress(Exception):
                        callback()

    def calls_pending_for(self, invocation: object) -> bool:
        """Only this exact foreground's handlers; older retained effects keep their original owner."""
        return any(owner is invocation for owner in self._call_invocations.values())

    def managed_calls(self, retained_app_calls: frozenset[str]) -> tuple[asyncio.Task, ...]:
        """Exclude only genuine already-forwarded application callbacks, never managed backend tool activity."""
        return tuple(
            task
            for identity, task in self._call_tasks.items()
            if identity not in retained_app_calls and not task.done()
        )

    def on_call_settled(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Subscribe to bridge call settlement and return an unsubscribe callback."""
        token = object()
        self._activity_callbacks[token] = callback
        return lambda: self._activity_callbacks.pop(token, None)

    async def _serve(
        self, call: dict, task: TaskContext, functions: frozenset[str], owners: Mapping[str, ToolSource]
    ) -> str | RefusedReply:
        """Serve a call with the task and invocation it had before any admission wait."""
        name = call["name"]
        if not self.call_current(call["call_id"]):
            return task.texts.not_run_output
        if task.business_access == "core_only" and name not in {tool["name"] for tool in task.tools().tools()}:
            task.dispatch(call)
            self.mailbox.put_nowait(CallRunning(call, refused=True))
            return refused_reply(task, call["call_id"])
        if name in functions:
            return await self._hold(call)
        if not await self.wait_for_admission(call, task, live=lambda: self.call_current(call["call_id"])):
            return self._admission_output if not self.admission_open else task.texts.not_run_output
        if task.dispatch(call) == "refused":
            reply = refused_reply(task, call["call_id"])
            if isinstance(reply, RefusedReply) and reply.details.input_scope == "global":
                return await self._hold(call)
            self.mailbox.put_nowait(CallRunning(call, refused=True))
            return reply
        self.mailbox.put_nowait(CallRunning(call))
        source = owners.get(name)
        task.tool_started(call, source.label if source is not None else UNKNOWN_LABEL)
        output, outcome, reply = await self._run(source, call, task)
        task.tool_outcome(call["call_id"], output, outcome)
        self.mailbox.put_nowait(CallSettled(call["call_id"]))
        return reply

    def placement_pending(self, call: dict) -> bool:
        """Check whether the task's tool call must wait for routing or placement."""
        return self.task.placement_pending(call)

    def call_current(self, call_id: str) -> bool:
        """Whether a not-yet-admitted call still belongs to this door's current native invocation."""
        return self.invocation_active() and self._call_invocations.get(call_id) is self.invocation

    @property
    def admission_open(self) -> bool:
        """Whether this door still accepts first admission; stopping need not wait for native retirement."""
        return not (self.retired or self._admission_stopped or self._admission_pause is not None)

    @property
    def placement_waiting(self) -> bool:
        """Whether calls currently wait for global placement before their first admission."""
        return bool(self.admitting)

    def placement_elapsed(self) -> float:
        """The union of placement-wait intervals; concurrent calls do not multiply the paused reading time."""
        running = 0.0 if self._placement_started is None else time.monotonic() - self._placement_started
        return self._placement_seconds + running

    def admission_changed(self) -> None:
        """Wake pre-admission waiters to recheck task ownership or stopping, without cancelling native work."""
        changed, self._admission_changed = self._admission_changed, asyncio.Event()
        changed.set()

    def stop_admissions(self, text: str) -> None:
        """Fence unadmitted calls and release their waits; already running calls retain their outcome path."""
        self._admission_stopped, self._admission_output = True, text
        self._admission_pause = None
        self.admission_changed()

    def pause_admissions(self, text: str) -> object | None:
        """Own one reversible pause; stopping or retirement permanently invalidates its token."""
        if not self.admission_open:
            return None
        token = self._admission_pause = object()
        self._admission_output = text
        self.admission_changed()
        return token

    def resume_admissions(self, token: object, task: TaskContext, sources: Sequence[ToolSource]) -> bool:
        """Resume only the exact pause and unchanged task/tool ceiling, never a stopped or retired door."""
        if (
            self._admission_pause is not token
            or self._admission_stopped
            or self.retired
            or self.task is not task
            or self.held
            or self.admitting
            or self.app_admissions
            or self._compose(task, sources, warn=False)[2] != self.served
        ):
            return False
        self._admission_pause = None
        self.admission_changed()
        return True

    def borrow_foreground(self, task: TaskContext, pause: object, retained: frozenset[str]) -> ForegroundBinding:
        """Use the same advertised native tools under a narrower Core context without retiring real old calls."""
        if (
            self.retired
            or self._admission_stopped
            or self._admission_pause is not pause
            or self._borrowed_binding is not None
            or self.admitting
            or self.app_admissions
            or (task.work_id, task.task_id, task.conversation_ref)
            != (self.task.work_id, self.task.task_id, self.task.conversation_ref)
            or task.business_access != "core_only"
            or any(held.call_id not in retained for held in self.held if not held.answer.done())
        ):
            raise RuntimeError("The original native foreground cannot safely lend its Core dispatch binding")
        binding = ForegroundBinding(self.task, dict(self.owners), self.context.copy(), pause, retained)
        self._borrowed_binding = binding
        self.task = task
        core = task.tools()
        self.owners = {**self.owners, **{tool["name"]: core for tool in core.tools()}}
        self.context = contextvars.copy_context()
        self.invocation = object()
        self._admission_pause = None
        self.admission_changed()
        return binding

    def resume_borrowed(self, binding: ForegroundBinding, pause: object, task: TaskContext) -> bool:
        """Reprompt a settled control foreground without treating its original app callback as a new call."""
        if (
            self._borrowed_binding is not binding
            or self._admission_pause is not pause
            or self._admission_stopped
            or self.retired
            or self.task is not task
            or self.admitting
            or self.app_admissions
            or any(held.call_id not in binding.retained for held in self.held if not held.answer.done())
        ):
            return False
        self._admission_pause = None
        self.admission_changed()
        return True

    def return_foreground(self, binding: ForegroundBinding) -> bool:
        """Restore the original owner with business admission still fenced until its explicit resume boundary."""
        if self._borrowed_binding is not binding:
            return False
        self._borrowed_binding = None
        self.task, self.owners, self.context = binding.task, dict(binding.owners), binding.context.copy()
        self.invocation = object()
        if not (self.retired or self._admission_stopped):
            self._admission_pause = binding.pause
        self.admission_changed()
        return not (self.retired or self._admission_stopped)

    async def wait_for_admission(
        self, call: dict, task: TaskContext, *, live: Callable[[], bool] | None = None
    ) -> bool:
        """Wait only before first routing, racing lifecycle changes and returning no permission of its own."""
        current = asyncio.current_task()
        waiting = False
        try:
            while self.admission_open and self.task is task and (live is None or live()):
                changed = self._admission_changed
                if not self.placement_pending(call):
                    return True
                if not waiting:
                    waiting = True
                    if not self.admitting:
                        self._placement_started = time.monotonic()
                    self.admitting.add(current)
                    self.mailbox.put_nowait(AdmissionReady())
                placed = asyncio.create_task(task.wait_for_placement())
                notified = asyncio.create_task(changed.wait())
                try:
                    done, _ = await asyncio.wait((placed, notified), return_when=asyncio.FIRST_COMPLETED)
                    for completed in done:
                        completed.result()
                finally:
                    for future in (placed, notified):
                        if not future.done():
                            future.cancel()
                    await asyncio.gather(placed, notified, return_exceptions=True)
            return False
        finally:
            if waiting:
                self.admitting.discard(current)
                if not self.admitting:
                    self._placement_seconds += time.monotonic() - self._placement_started
                    self._placement_started = None
                self.mailbox.put_nowait(AdmissionReady())

    def defer_admission(self, held: HeldCall) -> None:
        """Keep a queued app call at its first route while the reader continues handling native events and stops."""
        if held.call_id in self.app_admissions:
            return
        task = self.task

        async def ready() -> None:
            try:
                await self.wait_for_admission(
                    held.call, task, live=lambda: not held.answer.done() and self.call_current(held.call_id)
                )
            except Exception as exc:
                if not held.answer.done():
                    held.answer.set_exception(exc)
            finally:
                self.app_admissions.pop(held.call_id, None)
                self.mailbox.put_nowait(AdmissionReady())

        waiter = asyncio.create_task(ready(), name="harness-admission")
        self.app_admissions[held.call_id] = waiter
        held.answer.add_done_callback(lambda _: self.admission_changed())

    async def drain_admissions(self) -> None:
        """Join only pre-admission waits after retirement; no native cancellation or effect retry occurs here."""
        current = asyncio.current_task()
        waiting = (self.admitting | set(self.app_admissions.values())) - {current}
        if waiting:
            await asyncio.gather(*waiting, return_exceptions=True)

    async def _hold(self, call: dict) -> str | RefusedReply:
        """Hold an app call or global-placement refusal until its worker reaches the existing Core boundary."""
        held = HeldCall(call, asyncio.get_running_loop().create_future())
        self.held.append(held)
        self.mailbox.put_nowait(AppCall(held))
        try:
            return await held.answer
        finally:
            self.held.remove(held)

    async def _run(self, source: ToolSource | None, call: dict, task: TaskContext) -> tuple[str, Outcome, str]:
        """Run a call on its source; return the output, its outcome and the agent's plain-text answer."""
        name, call_id, texts = call["name"], call["call_id"], task.texts
        if source is None:
            offered = ", ".join(tool["name"] for tool in self.served) or texts.none
            output = json.dumps({"error": texts.unknown_tool_output.format(name=name, offered=offered)})
            return output, "failed", output
        try:
            async with asyncio.timeout(self.limit):
                output = await source.execute(call_id, name, call["arguments"])
            return output, "done", output
        except TimeoutError:
            # It may have run: the IM and the bridge answer it as owed.
            return f"{name} did not answer within {self.limit:g} s", "unclear", texts.owed_output
        except ToolNotExecuted as exc:
            error = f"{type(exc).__name__}: {exc}"
            output = json.dumps({"error": texts.failed_output.format(name=name, error=error)})
            return output, "failed", output
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            output = json.dumps({"error": texts.failed_output.format(name=name, error=error)})
            return output, "unclear", texts.owed_output

    def failed(self, name: str, error: str) -> str:
        """Return the task's ``texts.failed_output`` for a call that raised."""
        return self.task.texts.failed_output.format(name=name, error=error)

    def retire(self, text: str) -> None:
        """Serve nothing more; the runtime's held app calls end with ``text``."""
        super().retire(text)
        self.stop_admissions(text)
        for held in list(self.held):
            held.reply(text)


class _QuietServer:
    """A uvicorn server that leaves the process's signal handlers alone (the app's own server owns them)."""

    def __init__(self, app: Any):
        import uvicorn

        class Server(uvicorn.Server):
            @contextlib.contextmanager
            def capture_signals(self):
                """Take no signal: uvicorn would install handlers for the whole process."""
                yield

        self.server = Server(uvicorn.Config(app, log_level="warning", lifespan="on", access_log=False))

    async def serve(self, sock: socket.socket) -> None:
        await self.server.serve(sockets=[sock])

    @property
    def started(self) -> bool:
        return bool(self.server.started)

    def stop(self) -> None:
        self.server.should_exit = True


async def _answer(send: Callable, status: int, body: bytes, *headers: tuple[bytes, bytes]) -> None:
    fields = [(b"content-type", b"text/plain"), (b"content-length", str(len(body)).encode()), *headers]
    await send({"type": "http.response.start", "status": status, "headers": fields})
    await send({"type": "http.response.body", "body": body})


class McpBridge:
    """One MCP server over streamable HTTP for a core's harness sessions: a door per session, by bearer token."""

    def __init__(self, settings: Bridge, *, start_timeout_seconds: float = 30.0):
        """Listen where ``settings`` say, once :meth:`start` runs."""
        self.settings, self.start_timeout = settings, start_timeout_seconds
        self.doors: dict[str, Door] = {}
        self.base_url: str | None = None
        self._server: _QuietServer | None = None
        self._serving: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    @property
    def url(self) -> str:
        """The MCP endpoint's URL as the runtime reaches it."""
        return f"{self.base_url}{MCP_PATH}"

    async def start(self) -> None:
        """Start listening, once; later calls return at once."""
        async with self._lock:
            if self._serving is not None and not self._serving.done():
                return
            # Imported here: the packages of the harness dependency group load only with the harness.
            from mcp.server.lowlevel import Server

            server = Server("delegation-core-bridge", on_list_tools=self._list_tools, on_call_tool=self._call_tool)
            app = server.streamable_http_app(
                streamable_http_path=MCP_PATH, host=self.settings.host, stateless_http=True, json_response=True
            )
            sock = socket.socket(socket.AF_INET6 if ":" in self.settings.host else socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((self.settings.host, self.settings.port))
            except BaseException:
                sock.close()
                raise
            self.base_url = self.settings.url(sock.getsockname()[1])
            self._server = _QuietServer(self._gate(app))
            self._serving = asyncio.create_task(
                self._server.serve(sock), name="harness-bridge", context=contextvars.Context()
            )
            async with asyncio.timeout(self.start_timeout):
                while not self._server.started:
                    if self._serving.done():
                        sock.close()
                        self._serving.result()
                        raise RuntimeError("The bridge's server stopped before it started")
                    await asyncio.sleep(0.02)
            logger.info(f"harness.bridge_listening url={self.url}")

    def open(self, door: Door) -> McpServer:
        """Serve ``door`` from now on; return the MCP server entry the runtime is given."""
        self.doors[door.token] = door
        return McpServer(door.name, self.url, door.token)

    def close(self, door: Door, text: str) -> None:
        """Refuse tools while retaining authenticated keepalives until the owning runtime finishes."""
        door.retire(text)

    async def forget(self, door: Door) -> None:
        """Revoke a retired door's token and settle any admitted handlers."""
        self.doors.pop(door.token, None)
        await asyncio.gather(*(asyncio.shield(call) for call in door.pending_calls), return_exceptions=True)

    async def aclose(self) -> None:
        """Close every door and stop listening."""
        calls = {call for door in self.doors.values() for call in door.pending_calls}
        for door in list(self.doors.values()):
            self.close(door, "")
        self.doors.clear()
        if self._server is not None:
            self._server.stop()
        if self._serving is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._serving), 10)
            except Exception:
                self._serving.cancel()
                await asyncio.gather(self._serving, return_exceptions=True)
        self._serving = None
        if calls:
            await asyncio.gather(*(asyncio.shield(call) for call in calls), return_exceptions=True)

    def _gate(self, app: Any) -> Callable:
        """Authenticate the HTTP request and capture its handler before SDK dispatch."""
        endpoint = urlsplit(self.base_url or self.settings.url(self.settings.port))
        default_port = ":443" if endpoint.scheme == "https" else ":80"
        authority = endpoint.netloc.rpartition("@")[2].lower().removesuffix(default_port)
        origin = f"{endpoint.scheme}://{authority}".encode()

        async def gated(scope: dict, receive: Callable, send: Callable) -> None:
            if scope.get("type") == "http":
                raw_headers = scope.get("headers") or []
                for header in (b"authorization", b"mcp-session-id", b"mcp-protocol-version"):
                    if sum(name == header for name, _ in raw_headers) > 1:
                        await _answer(send, 400, b"duplicate MCP authentication or session header")
                        return
                headers = dict(raw_headers)
                door = self._door_of(headers.get(b"authorization", b""))
                if door is None:
                    await _answer(send, 401, b"unknown or retired session")
                    return
                if scope.get("path") != MCP_PATH:
                    await _answer(send, 404, b"not found")
                    return
                origins = [value for name, value in raw_headers if name == b"origin"]
                if origins and origins != [origin]:
                    await _answer(send, 403, b"invalid origin")
                    return
                if scope.get("method") != "POST":
                    await _answer(send, 405, b"", (b"allow", b"POST"))
                    return
                if headers.get(b"mcp-session-id"):
                    await _answer(send, 404, b"unknown MCP session")
                    return
                await app({**scope, OPERATION_SCOPE: Operation.capture(door)}, receive, send)
                return
            await app(scope, receive, send)

        return gated

    def _door_of(self, authorization: bytes | str | None) -> Door | None:
        """Return the owned door named by a bearer token, including a retired keepalive-only door."""
        value = authorization or ""
        text = value.decode("latin-1") if isinstance(value, bytes) else str(value)
        scheme, _, token = text.partition(" ")
        return self.doors.get(token.strip()) if scheme.lower() == "bearer" else None

    def _door(self, ctx: Any) -> Door:
        door = self._door_of(ctx.request.headers.get("authorization") if ctx.request is not None else None)
        if door is None:
            raise PermissionError("unknown or retired session")
        return door

    async def _list_tools(self, ctx: Any, params: Any) -> Any:
        import mcp_types as types

        door = self._door(ctx)
        if not door.retired:
            door.listed.set()
        tools = [
            types.Tool(name=tool["name"], description=tool["description"], input_schema=tool["input_schema"])
            for tool in (() if door.retired else door.tools())
        ]
        catalog = {
            "door": door.name,
            "role": door.role,
            "names": [tool.name for tool in tools],
            "retired": door.retired,
        }
        door.context.run(logger.debug, f"harness.tools {catalog}")
        return types.ListToolsResult(tools=tools)

    async def _call_tool(self, ctx: Any, params: Any) -> Any:
        import mcp_types as types

        door = self._door(ctx)
        arguments = dict(params.arguments or {})
        operation = ctx.request.scope.get(OPERATION_SCOPE) if ctx.request is not None else None
        if not isinstance(operation, Operation) or operation.door is not door:
            raise PermissionError("A native tool call requires authenticated immutable HTTP operation admission")

        async def serve() -> str | RefusedReply:
            if door.retired:
                return door.retired_output
            try:
                return await operation.call(params.name, arguments)
            except Exception as exc:
                logger.exception(f"harness.bridge_call_failed door={door.name} name={params.name}")
                return door.failed(params.name, f"{type(exc).__name__}: {exc}")

        context = operation.context.copy()
        context.run(NATIVE_OPERATION.set, operation.id)
        call = asyncio.Task(
            serve(), loop=asyncio.get_running_loop(), name="harness-call", context=context, eager_start=True
        )
        door.pending_calls.add(call)
        call.add_done_callback(door.pending_calls.discard)
        call.add_done_callback(lambda done: done.cancelled() or done.exception())
        reply = await asyncio.shield(call)
        text, meta = reply, None
        if isinstance(reply, RefusedReply):
            text = reply.text
            meta = {
                REFUSAL_META: {
                    "version": 1,
                    "door": door.name,
                    **asdict(reply.details),
                }
            }
        # Never an error result: an agent may pause a server whose calls failed a few times in a row.
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)], is_error=False, meta=meta)
