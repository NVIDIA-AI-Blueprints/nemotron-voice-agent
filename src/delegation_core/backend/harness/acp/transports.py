# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How the ACP client reaches one runtime process (``backend.harness.transport``): JSON-RPC lines over asyncio streams.

``stdio`` starts ``command`` in its work folder and its own process group, with an environment built only from
``pass_env``, the agent's needs and ``env`` (the agent may read every variable it gets). It drains stderr into the log,
since a full pipe blocks the agent, and its close sends end-of-file on stdin, then terminates and kills the group.
``websocket`` connects to the ``url`` the operator serves: one connection per runtime process, one text frame per
message. Either sets ``closed`` when the process exits or the connection closes, so the client ends every outstanding
prompt; an end it did not ask for is a warning with the exit code and last stderr lines, or the close code and reason.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from loguru import logger

from ..protocol import RuntimeLaunch

# An agent's single message can be large (a tool result in an update): the stream's line limit.
STREAM_LIMIT = 50 * 1024 * 1024
# How much of one stderr line goes into the log, and how many of its last lines an unexpected exit's warning carries.
STDERR_LINE_CHARS = 500
STDERR_TAIL_LINES = 10
# How long that warning waits for the stderr the agent wrote as it ended to be read.
STDERR_DRAIN_SECONDS = 0.5
# How often a started process is looked at for its exit, and its group for a process it left running.
EXIT_POLL_SECONDS = 0.1


class Transport(Protocol):
    """How the client reaches one runtime process: newline-delimited JSON-RPC over a pair of asyncio streams."""

    closed: asyncio.Event  # set once the process exited or the connection closed

    @classmethod
    def check(cls, command: Sequence[str], url: str | None) -> None:
        """Refuse settings it cannot run with (``ValueError``), when the core is built."""

    @classmethod
    def for_launch(cls, launch: RuntimeLaunch, *, stop_grace_seconds: float) -> Transport:
        """A transport for one runtime process; it connects nothing until :meth:`open`."""

    async def open(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Start or reach the agent; return the stream to read from and the one to write to."""

    async def close(self) -> None:
        """End the process (stdin closed, then terminate, then kill) or the connection; safe to call twice."""


class StdioTransport:
    """``stdio``: start the agent as a child process and speak over its stdin and stdout."""

    def __init__(self, launch: RuntimeLaunch, *, stop_grace_seconds: float):
        """Start ``launch.command`` when opened; give it ``stop_grace_seconds`` to exit at each close step."""
        self.launch, self.grace = launch, stop_grace_seconds
        self.closed = asyncio.Event()
        self.process: asyncio.subprocess.Process | None = None
        self._tasks: list[asyncio.Task] = []
        self._stderr: deque[str] = deque(maxlen=STDERR_TAIL_LINES)  # its last lines, for an unexpected exit's warning
        self._closing = False  # close() was called: the exit that follows is the one asked for

    @classmethod
    def check(cls, command: Sequence[str], url: str | None) -> None:
        """A command to start is needed, and no URL: one would go unused while the command starts another agent.

        Raises:
            ValueError: ``command`` is empty, or ``url`` is set.
        """
        if not command:
            raise ValueError("backend.harness.command must name the agent to start (transport stdio)")
        if url is not None:
            raise ValueError("backend.harness.url is for transport websocket: set transport: websocket, or url: null")

    @classmethod
    def for_launch(cls, launch: RuntimeLaunch, *, stop_grace_seconds: float) -> StdioTransport:
        """A transport for one runtime process."""
        return cls(launch, stop_grace_seconds=stop_grace_seconds)

    async def open(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Start the process in its work folder; return its stdout and its stdin."""
        self.process = await asyncio.create_subprocess_exec(
            *self.launch.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=dict(self.launch.env),
            cwd=str(self.launch.folder),
            limit=STREAM_LIMIT,
            start_new_session=True,
        )
        name = os.path.basename(self.launch.command[0])
        reading = asyncio.create_task(self._log_stderr(name), name="harness-stderr")
        self._tasks = [reading, asyncio.create_task(self._watch(reading), name="harness-exit")]
        return self.process.stdout, self.process.stdin

    async def _log_stderr(self, name: str) -> None:
        """Read the agent's stderr to its end, one log line each (debug: it may be chatty); keep its last lines."""
        stream = self.process.stderr
        with contextlib.suppress(Exception):
            while line := await stream.readline():
                text = line.decode("utf-8", "replace").rstrip()[:STDERR_LINE_CHARS]
                self._stderr.append(text)
                logger.debug(f"harness.stderr [{name} {self.process.pid}] {text}")

    async def _watch(self, reading: asyncio.Task) -> None:
        """Set ``closed`` once the process exits; an exit not asked for is a warning, with its code and last stderr.

        The exit is looked for rather than awaited: ``Process.wait`` returns only once the pipes are closed, and a
        process the agent left running may hold them open.
        """
        process = self.process
        try:
            while process.returncode is None:
                await asyncio.sleep(EXIT_POLL_SECONDS)
            if self._closing:
                logger.debug(f"harness.runtime_exited pid={process.pid} code={process.returncode}")
                return
            try:
                await asyncio.wait([reading], timeout=STDERR_DRAIN_SECONDS)
            finally:  # also when a close cuts the wait short: the exit came first
                tail = " | ".join(self._stderr) or "(none)"
                logger.warning(f"harness.runtime_exited pid={process.pid} code={process.returncode} stderr: {tail}")
        finally:
            self.closed.set()

    async def close(self) -> None:
        """End-of-file on stdin, then terminate the process group, then kill it, each after the grace.

        Each step waits for the agent and every process of its group to end. The group is ended even once the agent
        has exited: a process it left there may still run, and is terminated at once.
        """
        process = self.process
        if process is None:
            self.closed.set()
            return
        self._closing = True
        with contextlib.suppress(Exception):
            process.stdin.close()
        steps = (None, signal.SIGTERM, signal.SIGKILL)
        if process.returncode is not None:  # it has exited: what it left in its group gets no end-of-file step
            steps = steps[1:] if _group_runs(process.pid) else ()  # nothing left: its group id may be another's later
        for sig in steps:
            if sig is not None:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(process.pid, sig)
            if await self._ended():
                break
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self.closed.set()

    async def _ended(self) -> bool:
        """Whether the process and every process of its group end within the grace."""
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(self.grace):
                while self.process.returncode is None or _group_runs(self.process.pid):
                    await asyncio.sleep(EXIT_POLL_SECONDS)
                return True
        return False


def _group_runs(group: int) -> bool:
    """Whether a live process of the process group ``group`` remains (one this side may not signal counts).

    A zombie does not count: it has ended, though it stays in the group until its parent reaps it, and the app
    container's PID 1 (``uv run``) never reaps an orphan. ``/proc`` tells it apart; without it (macOS), it counts.
    """
    try:
        os.killpg(group, 0)  # a zombie answers too
    except ProcessLookupError:
        return False
    except PermissionError:  # one this side may not signal: /proc still shows whether it runs
        pass
    if not os.path.isdir("/proc/self"):  # no /proc (macOS)
        return True
    return any(_runs_in(group, pid) for pid in os.listdir("/proc") if pid.isdigit())


def _runs_in(group: int, pid: str) -> bool:
    """Whether the process ``pid`` is of the process group ``group`` and no zombie, as ``/proc/<pid>/stat`` says."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as stat:
            state, _, pgid = stat.read().rsplit(b")", 1)[1].split()[:3]  # fields 3 to 5, past a name that may hold ")"
    except OSError:  # it ended meanwhile
        return False
    return int(pgid) == group and state not in (b"Z", b"X")


class _FrameWriter(asyncio.Transport):
    """A write transport that sends each complete line written to it as one WebSocket text frame, in order."""

    def __init__(self, socket: Any):
        """Send each complete line written to it over ``socket``, in order."""
        super().__init__()
        self.socket = socket
        self.pending = b""
        self.queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.sender = asyncio.create_task(self._send(), name="harness-ws-send")
        self.closing = False

    def write(self, data: bytes) -> None:
        """Queue every complete line of ``data`` as a frame."""
        self.pending += bytes(data)
        *lines, self.pending = self.pending.split(b"\n")
        for line in lines:
            if line.strip():
                self.queue.put_nowait(line)

    async def _send(self) -> None:
        while (line := await self.queue.get()) is not None:
            await self.socket.send(line.decode("utf-8"))

    def is_closing(self) -> bool:
        """Whether the writer was closed."""
        return self.closing

    def close(self) -> None:
        """Stop sending once what was queued is sent."""
        self.closing = True
        self.queue.put_nowait(None)

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        """No extra information."""
        return default


class WebSocketTransport:
    """``websocket``: connect to the URL the operator serves the agent on (one connection, one runtime process).

    Its close only closes the connection, and it sees the process end only when the connection closes. So the
    operator's server has two duties: it ends the process when the connection closes, and closes the connection when
    the process exits.
    """

    def __init__(self, launch: RuntimeLaunch, *, stop_grace_seconds: float):
        """Connect to ``launch.url`` when opened."""
        self.launch, self.grace = launch, stop_grace_seconds
        self.closed = asyncio.Event()
        self.socket: Any = None
        self._writer: _FrameWriter | None = None
        self._reading: asyncio.Task | None = None
        self._closing = False  # close() was called: the close that follows is the one asked for

    @classmethod
    def check(cls, command: Sequence[str], url: str | None) -> None:
        """A WebSocket URL is needed.

        Raises:
            ValueError: ``url`` is missing or not a WebSocket URL.
        """
        if not url or not url.startswith(("ws://", "wss://")):
            raise ValueError("backend.harness.url must be the agent's ws:// or wss:// URL (transport websocket)")

    @classmethod
    def for_launch(cls, launch: RuntimeLaunch, *, stop_grace_seconds: float) -> WebSocketTransport:
        """A transport for one runtime process behind ``launch.url``."""
        return cls(launch, stop_grace_seconds=stop_grace_seconds)

    async def open(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Connect; return a reader fed with each frame as a line, and a writer that sends each line as a frame."""
        import websockets  # the harness's own package: loaded only when a core uses this transport

        self.socket = await websockets.connect(self.launch.url, max_size=None, open_timeout=None)
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader(limit=STREAM_LIMIT)
        protocol = asyncio.StreamReaderProtocol(reader)
        self._writer = _FrameWriter(self.socket)
        protocol.connection_made(self._writer)
        writer = asyncio.StreamWriter(self._writer, protocol, reader, loop)
        self._reading = asyncio.create_task(self._read(reader), name="harness-ws-read")
        return reader, writer

    async def _read(self, reader: asyncio.StreamReader) -> None:
        """Feed each frame to ``reader`` as a line; a close not asked for is a warning with its code and reason."""
        error = ""
        try:
            async for frame in self.socket:
                text = frame if isinstance(frame, str) else frame.decode("utf-8")
                reader.feed_data(text.rstrip("\n").encode("utf-8") + b"\n")
        except Exception as exc:  # a closed connection ends the stream like an exited process
            error = f" {type(exc).__name__}: {exc}"
        finally:
            ended = f"code={self.socket.close_code} reason={self.socket.close_reason!r}{error}"
            if self._closing:
                logger.debug(f"harness.connection_closed {ended}")
            else:
                logger.warning(f"harness.connection_closed {ended}")
            reader.feed_eof()
            self.closed.set()

    async def close(self) -> None:
        """Close the connection (the operator's side ends its runtime process)."""
        self._closing = True
        if self._writer is not None:
            self._writer.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._writer.sender, self.grace)
        if self.socket is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.socket.close(), self.grace)
        if self._reading is not None:
            self._reading.cancel()
            await asyncio.gather(self._reading, return_exceptions=True)
        self.closed.set()


# The transports, by the name backend.harness.transport gives.
TRANSPORTS: Mapping[str, type[Transport]] = {"stdio": StdioTransport, "websocket": WebSocketTransport}
