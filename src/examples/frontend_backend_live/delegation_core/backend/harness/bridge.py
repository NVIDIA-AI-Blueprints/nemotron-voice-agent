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
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from loguru import logger

from examples.frontend_backend_live.delegation_core.backend.port import ControlTools, Outcome, TaskContext, ToolSource
from examples.shared.ids import uid

from .protocol import AppCall, CallRunning, CallSettled, HeldCall, Mailbox, McpServer
from .settings import Bridge

UNKNOWN_LABEL = "harness"
MCP_PATH = "/mcp"


def mcp_tool(tool: Mapping[str, Any]) -> dict:
    """Convert a Responses function tool to an MCP tool: its name, description and JSON schema."""
    return {
        "name": tool["name"],
        "description": tool.get("description") or tool["name"],
        "input_schema": tool.get("parameters") or {"type": "object", "properties": {}},
    }


class Door:
    """One session's door on the bridge: its server name, its bearer token and its toolset; a bare door serves none."""

    def __init__(self, name: str):
        """Open a door named ``name``, with a fresh token; its calls run in a copy of the context it is opened in."""
        self.name = name
        self.token = secrets.token_urlsafe(24)
        # Set by the agent's tools/list: a failed MCP connect is silent at the ACP level.
        self.listed = asyncio.Event()
        self.retired = False
        self.context = contextvars.copy_context()

    def tools(self) -> list[dict]:
        """Return the tools it serves, as MCP tools: none."""
        return []

    async def call(self, name: str, arguments: dict) -> str:
        """Serve one call and return its plain-text result; a bare door answers empty."""
        return ""

    def failed(self, name: str, error: str) -> str:
        """Return the answer to a call that raised; ``error`` is the exception's type and message."""
        return error

    def retire(self, text: str) -> None:
        """Serve nothing more; held calls are answered with ``text``."""
        self.retired = True


class ConductorDoor(Door):
    """The conductor's door: the four control tools; each call is a decision the IM applies."""

    def __init__(self, name: str, vocabulary: ControlTools):
        """Serve ``vocabulary``'s control tools; before any ``decide()`` binds a ``place``, answer its idle outcome."""
        super().__init__(name)
        self.vocabulary = vocabulary
        self.place: Callable[[Any], Awaitable[str]] | None = None

    def bind(self, place: Callable[[Any], Awaitable[str]]) -> None:
        """Apply the next calls through ``place`` (the running ``decide()``'s; after it ends the IM refuses).

        The calls run in a copy of the context it is bound in, so a conductor kept for a later call serves that call.
        """
        self.place = place
        self.context = contextvars.copy_context()

    def tools(self) -> list[dict]:
        """Return the four control tools."""
        return [mcp_tool(tool) for tool in self.vocabulary.conductor_tools()]

    async def call(self, name: str, arguments: dict) -> str:
        """Turn the call into a decision and answer with the IM's outcome line; a malformed call with its error."""
        try:
            decision = self.vocabulary.decision(name, json.dumps(arguments))
        except ValueError as exc:
            logger.warning(f"harness.conductor_call_malformed name={name} error={exc}")
            return json.dumps({"error": str(exc)})
        if self.place is None:
            return self.vocabulary.idle
        return await self.place(decision)

    def failed(self, name: str, error: str) -> str:
        """Return the vocabulary's ``failed`` answer."""
        return self.vocabulary.failed.format(name=name, error=error)


class WorkerDoor(Door):
    """A worker's door: the app's functions, the tool sources' and the IM's work tools, for one task."""

    def __init__(
        self,
        name: str,
        task: TaskContext,
        sources: Sequence[ToolSource],
        mailbox: Mailbox,
        *,
        tool_timeout_seconds: float,
    ):
        """Serve ``task``'s toolset, reporting each call to its run through ``mailbox``."""
        super().__init__(name)
        self.task, self.mailbox, self.limit = task, mailbox, tool_timeout_seconds
        self.app_functions, self.owners, self.served = self._compose(task, sources)
        self.held: list[HeldCall] = []

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
        return not (self.held or self.retired) and self._compose(task, sources, warn=False)[2] == self.served

    def rebind(self, task: TaskContext, sources: Sequence[ToolSource]) -> bool:
        """Serve ``task`` under the same name and token if its toolset is the one served; else return ``False``.

        The agent keeps the tools it listed; a door that holds a call or is retired stays as it was.
        """
        app_functions, owners, served = self._compose(task, sources)
        if self.held or self.retired or served != self.served:
            return False
        self.task, self.app_functions, self.owners = task, app_functions, owners
        return True

    def tools(self) -> list[dict]:
        """Return the task's toolset."""
        return list(self.served)

    async def call(self, name: str, arguments: dict) -> str:
        """Hold an app call for the run; run any other through ``route()``, ``tool_started`` and ``tool_outcome``."""
        call = {
            "type": "function_call",
            "id": uid("fc"),
            "call_id": uid("call"),
            "name": name,
            "arguments": json.dumps(arguments),
            "status": "completed",
        }
        if name in self.app_functions:
            held = HeldCall(call, asyncio.get_running_loop().create_future())
            self.held.append(held)
            self.mailbox.put_nowait(AppCall(held))
            try:
                return await held.answer
            finally:
                self.held.remove(held)
        if self.task.route(call) == "refused":
            self.mailbox.put_nowait(CallRunning(call, refused=True))
            return self.task.refusal(call["call_id"])
        self.mailbox.put_nowait(CallRunning(call))
        source = self.owners.get(name)
        self.task.tool_started(call, source.label if source is not None else UNKNOWN_LABEL)
        output, outcome, reply = await self._run(source, call)
        self.task.tool_outcome(call["call_id"], output, outcome)
        self.mailbox.put_nowait(CallSettled(call["call_id"]))
        return reply

    async def _run(self, source: ToolSource | None, call: dict) -> tuple[str, Outcome, str]:
        """Run a call on its source; return the output, its outcome and the agent's plain-text answer."""
        name, call_id, texts = call["name"], call["call_id"], self.task.texts
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
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            output = json.dumps({"error": texts.failed_output.format(name=name, error=error)})
            return output, "failed", output

    def failed(self, name: str, error: str) -> str:
        """Return the task's ``texts.failed_output`` for a call that raised."""
        return self.task.texts.failed_output.format(name=name, error=error)

    def retire(self, text: str) -> None:
        """Serve nothing more; the runtime's held app calls end with ``text``."""
        super().retire(text)
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
                streamable_http_path=MCP_PATH, host=self.settings.host, stateless_http=True
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
            # In an empty context: the server serves every session's door, not only the one whose code started it.
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
        """Serve ``door`` no more (its token is refused from now on); its held calls end with ``text``."""
        self.doors.pop(door.token, None)
        door.retire(text)

    async def aclose(self) -> None:
        """Close every door and stop listening."""
        for door in list(self.doors.values()):
            self.close(door, "")
        if self._server is not None:
            self._server.stop()
        if self._serving is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._serving), 10)
            except Exception:
                self._serving.cancel()
                await asyncio.gather(self._serving, return_exceptions=True)
        self._serving = None

    def _gate(self, app: Any) -> Callable:
        """Wrap ``app`` in a gate: 401 without a live door's bearer token, 405 at once for any method but POST.

        Stateless HTTP opens no event stream, and MCP lets a server answer GET with 405; an agent may probe the URL with
        HEAD and GET and wait on each, so a stream left open would hold every session's start.
        """

        async def gated(scope: dict, receive: Callable, send: Callable) -> None:
            if scope.get("type") == "http":
                headers = dict(scope.get("headers") or [])
                if self._door_of(headers.get(b"authorization", b"")) is None:
                    await _answer(send, 401, b"unknown or retired session")
                    return
                if scope.get("method") != "POST":
                    await _answer(send, 405, b"", (b"allow", b"POST"))
                    return
            await app(scope, receive, send)

        return gated

    def _door_of(self, authorization: bytes | str | None) -> Door | None:
        """Return the live door that an ``Authorization: Bearer <token>`` value names, if any."""
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
        door.listed.set()
        tools = [
            types.Tool(name=tool["name"], description=tool["description"], input_schema=tool["input_schema"])
            for tool in door.tools()
        ]
        return types.ListToolsResult(tools=tools)

    async def _call_tool(self, ctx: Any, params: Any) -> Any:
        import mcp_types as types

        door = self._door(ctx)

        async def serve() -> str:
            try:
                return await door.call(params.name, dict(params.arguments or {}))
            except Exception as exc:
                logger.exception(f"harness.bridge_call_failed door={door.name} name={params.name}")
                return door.failed(params.name, f"{type(exc).__name__}: {exc}")

        # In a fresh copy of the door's context, not the server's: what the call logs and starts is its session's.
        # Started at once: the call reaches its door before a close already queued for this loop turn retires it.
        loop = asyncio.get_running_loop()
        call = asyncio.Task(serve(), loop=loop, name="harness-call", context=door.context.copy(), eager_start=True)
        text = await call
        # Never an error result: an agent may pause a server whose calls failed a few times in a row.
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)], is_error=False)
