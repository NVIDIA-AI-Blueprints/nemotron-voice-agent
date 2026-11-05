# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""A runtime process's work folder, environment, opening and close, and the ACP client's two transports."""

import asyncio
import base64
import contextlib
import ctypes
import fcntl
import hashlib
import os
import signal
import struct
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

from examples.frontend_backend_live import delegation_core
from examples.frontend_backend_live.delegation_core.backend.harness.acp.transports import (
    StdioTransport,
    WebSocketTransport,
)
from examples.frontend_backend_live.delegation_core.backend.harness.protocol import Role, RuntimeLaunch
from examples.frontend_backend_live.delegation_core.backend.harness.runtime import RuntimeOpener, RuntimeProcess
from examples.frontend_backend_live.delegation_core.backend.harness.settings import HarnessSettings
from examples.frontend_backend_live.delegation_core.backend.harness.workdir import MARKER, WorkFolder
from tests.unit.delegation_core.backend.harness.doubles import AGENT_BLOCKS, PROMPTS, FakeClient, scratch

# A process that makes a work folder, says where, holds it until its stdin ends, then crashes (it never removes it).
HOLDER = (
    "import os, sys\n"
    "from pathlib import Path\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "from examples.frontend_backend_live.delegation_core.backend.harness.workdir import WorkFolder\n"
    "print(WorkFolder.make(Path(sys.argv[2])).path, flush=True)\n"
    "sys.stdin.read()\n"
    "os._exit(9)\n"
)
SOURCES = Path(delegation_core.__file__).parents[3]  # src
ECHO = (
    "import sys, os\n"
    "sys.stdout.write(os.getcwd() + ' ' + os.environ.get('MARK', '-') + ' ' + os.environ.get('HOME', '-') + '\\n')\n"
    "sys.stdout.flush()\n"
    "for line in sys.stdin:\n"
    "    sys.stdout.write(line)\n"
    "    sys.stdout.flush()\n"
)
SHELL = {"PATH": os.environ["PATH"]}
# Linux's prctl options (<linux/prctl.h>) for a child subreaper: the process that adopts its descendants' orphans.
PR_SET_CHILD_SUBREAPER, PR_GET_CHILD_SUBREAPER = 36, 37


@contextlib.contextmanager
def warnings_logged():
    """The warnings the transports log inside the block (their logger is patched)."""
    lines: list[str] = []
    with mock.patch("examples.frontend_backend_live.delegation_core.backend.harness.acp.transports.logger") as log:
        log.warning.side_effect = lines.append
        yield lines


def stdio(command, env=None, grace=2.0):
    """A stdio transport that starts ``command`` in a scratch folder with ``env`` only."""
    launch = RuntimeLaunch(tuple(command), env or {}, None, scratch("proc-"))
    return StdioTransport.for_launch(launch, stop_grace_seconds=grace)


class WorkFolderTests(unittest.TestCase):
    def setUp(self):
        self.root = scratch("harness-work-")

    def test_a_work_folder_is_marked_and_the_sweep_takes_only_the_marked_folders_no_process_holds(self):
        live = WorkFolder.make(self.root)
        self.assertEqual(sorted(path.name for path in live.path.iterdir()), [MARKER, "sessions"])
        self.assertEqual((live.path / MARKER).read_text().strip(), str(os.getpid()))  # a note: the lock says it lives
        self.assertEqual(live.session_folder("w1"), live.path / "sessions" / "w1")
        stranger = self.root / "not-ours"
        stranger.mkdir()
        WorkFolder(stranger).remove()  # only a marked folder is removed
        # A crashed run's leftover names a process id that runs: in a restarted container, the server usually gets its
        # predecessor's id; here, this process's own.
        leftover = self.root / "work-crashed"
        (leftover / "sessions").mkdir(parents=True)
        (leftover / MARKER).write_text(f"{os.getpid()}\n")
        self.assertEqual(WorkFolder.sweep(self.root), 1)
        self.assertEqual((live.path.exists(), leftover.exists(), stranger.exists()), (True, False, True))
        with mock.patch.object(fcntl, "flock"):  # as on NFS, where a process may take its own lock again
            self.assertEqual(WorkFolder.sweep(self.root), 0)  # never its own process's folder
        live.remove()
        self.assertFalse(live.path.exists())

    def test_another_processs_folder_is_kept_whatever_id_it_names_until_that_process_ends(self):
        command = [sys.executable, "-I", "-c", HOLDER, str(SOURCES), str(self.root)]
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as holder:
            try:
                folder = Path(holder.stdout.readline().strip())
                # A server in another container (another PID namespace) names an id that does not run here.
                (folder / MARKER).write_text("999999999\n")
                self.assertEqual((WorkFolder.sweep(self.root), folder.exists()), (0, True))
                holder.stdin.close()
                self.assertEqual(holder.wait(10), 9)  # it crashed: the system released its lock
            finally:
                holder.kill()
        self.assertEqual((WorkFolder.sweep(self.root), folder.exists()), (1, False))

    def test_the_process_environment_is_the_allowlist_the_agents_needs_and_env_only(self):
        block = {"prompts": PROMPTS, "agents": AGENT_BLOCKS, "command": ["hermes", "acp"]}
        block |= {"env": {"HERMES_HOME": "/srv/hermes"}, "pass_env": ["PATH", "UNSET_NAME"]}
        with mock.patch.dict(os.environ, {"HARNESS_TEST_SECRET": "must-not-pass"}):
            plain = RuntimeOpener(HarnessSettings.parse(block)).environment()
            hermes = RuntimeOpener(HarnessSettings.parse({**block, "agent": "hermes"})).environment()
            operator = {**block, "agent": "hermes", "env": {"HERMES_ACP_SKIP_CONFIGURED_MCP": "0"}}
            overridden = RuntimeOpener(HarnessSettings.parse(operator)).environment()
        self.assertEqual(plain, {"PATH": os.environ["PATH"], "HERMES_HOME": "/srv/hermes"})
        self.assertEqual(hermes, {**plain, "HERMES_ACP_SKIP_CONFIGURED_MCP": "1"})  # what Hermes needs
        self.assertEqual(overridden["HERMES_ACP_SKIP_CONFIGURED_MCP"], "0")  # the configured env has the last word


class RuntimeProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_runtime_that_cannot_start_or_be_reached_in_time_leaves_no_work_folder(self):
        held = []  # the connections a server that never answers the handshake keeps
        server = await asyncio.start_server(lambda reader, writer: held.append(writer), "127.0.0.1", 0)
        self.addAsyncCleanup(server.wait_closed)
        self.addCleanup(server.close)
        url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/acp"
        served = {"transport": "websocket", "url": url, "start_timeout_seconds": 0.5, "stop_grace_seconds": 5}
        for block, error in (({"command": ["/nonexistent/runtime"]}, FileNotFoundError), (served, TimeoutError)):
            root = scratch("harness-work-")
            opener = RuntimeOpener(HarnessSettings.parse({"prompts": PROMPTS, "work_root": str(root), **block}))
            started = time.monotonic()
            with self.subTest(error.__name__), self.assertRaises(error):
                await opener.open(Role("workers", "w"))
            self.assertLess(time.monotonic() - started, 2)  # bounded by the start timeout, not the stop grace
            self.assertEqual(list(root.iterdir()), [])
        for writer in held:
            writer.close()

    async def test_closing_a_runtime_process_ends_its_client_then_deletes_its_folder_even_when_the_close_fails(self):
        class Stubborn(FakeClient):
            async def close(self) -> None:
                raise RuntimeError("the agent would not stop")

        root = scratch("harness-work-")
        client, work = FakeClient(), WorkFolder.make(root)
        await RuntimeProcess(client, work).close()
        self.assertEqual((client.alive, work.path.exists()), (False, False))
        with self.assertRaisesRegex(RuntimeError, "would not stop"):
            await RuntimeProcess(Stubborn(), WorkFolder.make(root)).close()
        self.assertEqual(list(root.iterdir()), [])


@contextlib.contextmanager
def orphans_adopted():
    """Inside the block this process adopts the orphans of its descendants; it yields whether Linux's prctl let it.

    It never reaps them, like the app container's PID 1 (``uv run``, no init): one that ends stays a zombie.
    """
    prctl = ctypes.CDLL(None, use_errno=True).prctl
    before = ctypes.c_int()
    if prctl(PR_GET_CHILD_SUBREAPER, ctypes.byref(before), 0, 0, 0) or prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0):
        yield False
        return
    try:
        yield True
    finally:
        prctl(PR_SET_CHILD_SUBREAPER, before.value, 0, 0, 0)


def group_states(group: int) -> dict[int, str]:
    """Each process of the process group ``group`` with its state, as Linux's /proc shows it (``Z``: a zombie)."""
    states = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            with contextlib.suppress(OSError):  # it ended meanwhile
                state, _, pgid = (entry / "stat").read_text().rsplit(")", 1)[1].split()[:3]  # after its name
                if int(pgid) == group:
                    states[int(entry.name)] = state
    return states


def reap(group: int) -> None:
    """Kill what is left of the process group ``group``, and reap those of its processes this process adopted."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(group, signal.SIGKILL)
    for pid in group_states(group):
        with contextlib.suppress(ChildProcessError):
            os.waitpid(pid, 0)


class StdioTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_it_starts_the_command_in_its_folder_with_only_the_given_environment_and_closes_it(self):
        transport = stdio((sys.executable, "-I", "-c", ECHO), {"MARK": "given"})
        with warnings_logged() as warnings:
            reader, writer = await transport.open()
            started = (await asyncio.wait_for(reader.readline(), 5)).decode().split()
            self.assertEqual(started, [str(transport.launch.folder), "given", "-"])  # its folder; only its env
            writer.write(b'{"jsonrpc": "2.0"}\n')
            await writer.drain()
            self.assertEqual(await asyncio.wait_for(reader.readline(), 5), b'{"jsonrpc": "2.0"}\n')
            await transport.close()
        self.assertEqual((transport.closed.is_set(), transport.process.returncode), (True, 0))  # stdin closed: it ended
        self.assertEqual(warnings, [])  # the exit it asked for
        stubborn = stdio((sys.executable, "-I", "-c", "import time\nwhile True:\n    time.sleep(1)\n"), grace=0.3)
        await stubborn.open()
        await stubborn.close()  # it ignores the end of its input: terminated
        self.assertNotEqual(stubborn.process.returncode, 0)

    async def test_an_exit_it_did_not_ask_for_is_a_warning_with_the_exit_code_and_the_last_stderr_lines(self):
        transport = stdio(("sh", "-c", "echo starting >&2; echo 'No usable credentials' >&2; exit 3"), SHELL)
        with warnings_logged() as warnings:
            await transport.open()
            await asyncio.wait_for(transport.closed.wait(), 5)
            await transport.close()
        [warning] = warnings
        self.assertIn(f"pid={transport.process.pid} code=3", warning)
        self.assertIn("starting | No usable credentials", warning)

    async def test_its_close_ends_its_group_while_a_process_it_left_holds_its_pipes(self):
        transport = stdio(("sh", "-c", "sleep 30 & sleep 0.5; exit 1"), SHELL, grace=1)  # the child keeps its pipes
        await transport.open()
        group = transport.process.pid
        self.addCleanup(reap, group)
        await asyncio.wait_for(transport.closed.wait(), 2)  # its exit is seen
        await transport.close()
        with self.assertRaises(ProcessLookupError):
            os.killpg(group, 0)  # the process it left was ended with it

    @unittest.skipUnless(sys.platform == "linux", "a child subreaper and /proc are Linux's")
    async def test_its_close_does_not_wait_out_the_grace_for_a_zombie_left_in_its_group(self):
        # This test adopts the process the agent leaves and never reaps it, as the app container's PID 1 does: once
        # that process ends, it stays in the agent's group as a zombie.
        if not self.enterContext(orphans_adopted()):
            self.skipTest("this process may not be a child subreaper here")
        transport = stdio(("sh", "-c", "read x; sleep 0.2 </dev/null >/dev/null 2>&1 & exit 0"), SHELL)
        await transport.open()
        self.addCleanup(reap, transport.process.pid)
        began = time.monotonic()
        await transport.close()  # at end-of-file it leaves a sleep and exits
        self.assertLess(time.monotonic() - began, 1)  # under half a grace step: no close step waits for the zombie
        self.assertEqual(set(group_states(transport.process.pid).values()), {"Z"})  # nothing runs; the zombie stays


GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


async def handshake(reader, writer):
    """Accept one WebSocket connection (RFC 6455) with the standard library only: its request's head."""
    head = (await reader.readuntil(b"\r\n\r\n")).decode()
    key = next(
        line.split(":", 1)[1].strip() for line in head.split("\r\n") if line.lower().startswith("sec-websocket-key")
    )
    accept = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
    upgrade = "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
    writer.write(f"{upgrade}Sec-WebSocket-Accept: {accept}\r\n\r\n".encode())
    return head


async def frame(reader) -> tuple[int, bytes]:
    """One frame from the client (masked): its opcode and payload."""
    first, second = await reader.readexactly(2)
    length = second & 0x7F
    if length == 126:
        (length,) = struct.unpack(">H", await reader.readexactly(2))
    mask = await reader.readexactly(4)
    return first & 0x0F, bytes(b ^ mask[i % 4] for i, b in enumerate(await reader.readexactly(length)))


class WebSocketTransportTests(unittest.IsolatedAsyncioTestCase):
    async def serve(self, handler, env=None):
        """A transport to a one-connection server run by ``handler(reader, writer)``."""
        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        self.addAsyncCleanup(server.wait_closed)
        self.addCleanup(server.close)
        url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/acp"
        return WebSocketTransport.for_launch(
            RuntimeLaunch((), env or {}, url, Path("/work/work-1")), stop_grace_seconds=2
        )

    async def test_each_line_is_one_frame_and_each_frame_one_line_on_the_url_as_served(self):
        heads = []

        async def echo(reader, writer):
            heads.append(await handshake(reader, writer))
            while (message := await frame(reader))[0] != 0x8:  # until the close
                writer.write(bytes([0x81, len(message[1])]) + message[1])
            writer.write(b"\x88\x00")
            writer.close()

        transport = await self.serve(echo, {"KEY": "never sent"})
        reader, writer = await transport.open()
        self.assertEqual(heads[0].split("\r\n")[0], "GET /acp HTTP/1.1")  # no query of ours
        self.assertNotIn("never sent", heads[0])  # nothing of this side's environment
        writer.write(b'{"a": 1}\n{"b"')
        writer.write(b": 2}\n")
        await writer.drain()
        for line in (b'{"a": 1}\n', b'{"b": 2}\n'):
            self.assertEqual(await asyncio.wait_for(reader.readline(), 5), line)
        with warnings_logged() as warnings:
            await transport.close()
        self.assertEqual((transport.closed.is_set(), warnings), (True, []))  # the close it asked for

    async def test_a_connection_the_agents_side_closes_ends_the_stream_with_a_warning_of_its_code_and_reason(self):
        async def crash(reader, writer):  # the agent ends at once: a close with 1011 and a reason
            await handshake(reader, writer)
            reason = struct.pack(">H", 1011) + b"agent crashed"
            writer.write(bytes([0x88, len(reason)]) + reason)
            with contextlib.suppress(asyncio.IncompleteReadError, ConnectionError):
                await frame(reader)  # the client's close
            writer.close()

        transport = await self.serve(crash)
        with warnings_logged() as warnings:
            reader, _ = await transport.open()
            await asyncio.wait_for(transport.closed.wait(), 5)
        self.assertEqual(await asyncio.wait_for(reader.read(), 5), b"")  # the stream ended, as at an exit
        [warning] = warnings
        self.assertIn("code=1011 reason='agent crashed'", warning)
        await transport.close()

    def test_it_needs_a_websocket_url_and_stdio_a_command(self):
        for url in (None, "", "http://agent:8790/acp"):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, "ws:// or wss:// URL"):
                WebSocketTransport.check(("hermes", "acp"), url)
        WebSocketTransport.check((), "wss://agent.example/acp")
        with self.assertRaisesRegex(ValueError, "must name the agent to start"):
            StdioTransport.check((), "ws://agent:8790/acp")


if __name__ == "__main__":
    unittest.main()
