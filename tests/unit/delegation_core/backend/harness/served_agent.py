# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""An operator's agent served over WebSocket, for the harness's ``websocket`` transport in tests.

Each connection starts one ``command`` process; each text frame is one JSON-RPC line on its stdin, and each stdout line
goes back as one frame. A closed connection ends the process's stdin, and the end of its output closes the connection,
so the core sees the runtime lost.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import os
import signal
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
TEXT, CLOSE, PING, PONG, CONTINUATION = 0x1, 0x8, 0x9, 0xA, 0x0
LIMIT = 64 * 1024 * 1024


def frame(opcode: int, payload: bytes) -> bytes:
    """One unmasked server frame."""
    length = len(payload)
    if length < 126:
        header = struct.pack(">BB", 0x80 | opcode, length)
    elif length < 1 << 16:
        header = struct.pack(">BBH", 0x80 | opcode, 126, length)
    else:
        header = struct.pack(">BBQ", 0x80 | opcode, 127, length)
    return header + payload


async def read_frame(reader: asyncio.StreamReader) -> tuple[bool, int, bytes]:
    """One client frame: whether it ends its message, its opcode, its unmasked payload."""
    first, second = await reader.readexactly(2)
    length = second & 0x7F
    if length == 126:
        (length,) = struct.unpack(">H", await reader.readexactly(2))
    elif length == 127:
        (length,) = struct.unpack(">Q", await reader.readexactly(8))
    mask = await reader.readexactly(4) if second & 0x80 else b"\0\0\0\0"
    data = await reader.readexactly(length)
    return bool(first & 0x80), first & 0x0F, bytes(byte ^ mask[index % 4] for index, byte in enumerate(data))


class ServedAgent:
    """Serves ``command`` on ``ws://127.0.0.1:<port>/acp``: one process per connection, in ``folder``."""

    def __init__(self, command: Sequence[str], env: Mapping[str, str], folder: Path):
        self.command, self.env, self.folder = tuple(command), dict(env), folder
        self.server: asyncio.base_events.Server | None = None
        self.processes: list[asyncio.subprocess.Process] = []
        self.connections = 0

    async def start(self) -> str:
        self.server = await asyncio.start_server(self._serve, "127.0.0.1", 0, limit=LIMIT)
        return f"ws://127.0.0.1:{self.server.sockets[0].getsockname()[1]}/acp"

    async def stop(self) -> None:
        if self.server is not None:
            self.server.close()
        for process in self.processes:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = (await reader.readuntil(b"\r\n\r\n")).decode("latin-1").split("\r\n")
        headers = {name.strip().lower(): value.strip() for name, _, value in (line.partition(":") for line in head[1:])}
        accept = base64.b64encode(hashlib.sha1((headers["sec-websocket-key"] + GUID).encode()).digest()).decode()
        writer.write(
            "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n".encode()
        )
        self.connections += 1
        process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=self.env,
            cwd=str(self.folder),
            limit=LIMIT,
            start_new_session=True,
        )
        self.processes.append(process)
        lock = asyncio.Lock()

        async def send(opcode: int, payload: bytes) -> None:
            async with lock:
                writer.write(frame(opcode, payload))
                await writer.drain()

        async def to_process() -> None:
            message = b""
            with contextlib.suppress(asyncio.IncompleteReadError, ConnectionError):
                while True:
                    final, opcode, payload = await read_frame(reader)
                    if opcode == PING:
                        await send(PONG, payload)
                    elif opcode == CLOSE:
                        break
                    elif opcode in (TEXT, CONTINUATION):
                        message += payload
                        if final:
                            process.stdin.write(message.rstrip(b"\n") + b"\n")
                            await process.stdin.drain()
                            message = b""
            with contextlib.suppress(Exception):
                process.stdin.close()

        async def from_process() -> None:
            with contextlib.suppress(ConnectionError):
                while line := await process.stdout.readline():
                    await send(TEXT, line.rstrip(b"\n"))
                await send(CLOSE, struct.pack(">H", 1000))

        inbound = asyncio.ensure_future(to_process())
        try:
            await from_process()
        finally:
            inbound.cancel()
            await asyncio.gather(inbound, return_exceptions=True)
            with contextlib.suppress(Exception):
                process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), 5)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
            writer.close()
