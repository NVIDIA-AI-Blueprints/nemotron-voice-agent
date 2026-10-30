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

import delegation_core
from delegation_core.backend.harness.acp.transports import StdioTransport, WebSocketTransport
from delegation_core.backend.harness.protocol import Role, RuntimeLaunch
from delegation_core.backend.harness.runtime import RuntimeOpener, RuntimeProcess
from delegation_core.backend.harness.settings import HarnessSettings
from delegation_core.backend.harness.workdir import MARKER, WorkFolder
from tests.unit.delegation_core.backend.harness.doubles import AGENT_BLOCKS, PROMPTS, FakeClient, scratch

# A process that makes a work folder, says where, holds it until its stdin ends, then crashes (it never removes it).
HOLDER = (
    "import os, sys\n"
    "from pathlib import Path\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "from delegation_core.backend.harness.workdir import WorkFolder\n"
    "print(WorkFolder.make(Path(sys.argv[2])).path, flush=True)\n"
    "sys.stdin.read()\n"
    "os._exit(9)\n"
)
SOURCES = Path(delegation_core.__file__).parents[1]


@contextlib.contextmanager
def warnings_logged():
    """The warnings the transports log inside the block (their logger is patched)."""
    lines: list[str] = []
    with mock.patch("delegation_core.backend.harness.acp.transports.logger") as log:
        log.warning.side_effect = lines.append
        yield lines


class WorkFolderTests(unittest.TestCase):
    def setUp(self):
        self.root = scratch("harness-work-")

    def test_a_work_folder_is_marked_and_holds_the_sessions_folders(self):
        work = WorkFolder.make(self.root)
        self.assertEqual(work.path.parent, self.root)
        self.assertEqual(sorted(path.name for path in work.path.iterdir()), [MARKER, "sessions"])
        self.assertEqual((work.path / MARKER).read_text().strip(), str(os.getpid()))  # a note: the lock says it lives
        folder = work.session_folder("w1")
        self.assertEqual(folder, work.path / "sessions" / "w1")
        self.assertEqual(list(folder.iterdir()), [])  # the agent class readies it (Agent.prepare)
        work.remove()
        self.assertFalse(work.path.exists())

    def test_only_a_marked_folder_is_removed_and_the_sweep_takes_only_the_folders_no_process_holds(self):
        stranger = self.root / "not-ours"
        stranger.mkdir()
        WorkFolder(stranger).remove()
        self.assertTrue(stranger.exists())
        live = WorkFolder.make(self.root)
        # A crashed run's leftover names a process id that runs: in a restarted container, the server usually gets
        # its predecessor's id; here, this process's own.
        leftover = self.root / "work-crashed"
        (leftover / "sessions").mkdir(parents=True)
        (leftover / MARKER).write_text(f"{os.getpid()}\n")
        self.assertEqual(WorkFolder.sweep(self.root), 1)
        self.assertEqual((live.path.exists(), leftover.exists(), stranger.exists()), (True, False, True))
        live.remove()

    def test_another_processs_folder_is_kept_whatever_id_it_names_until_that_process_ends(self):
        command = [sys.executable, "-I", "-c", HOLDER, str(SOURCES), str(self.root)]
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as holder:
            try:
                folder = Path(holder.stdout.readline().strip())
                # A server in another container (another PID namespace) names an id that does not run here.
                (folder / MARKER).write_text("999999999\n")
                self.assertEqual(WorkFolder.sweep(self.root), 0)
                self.assertTrue(folder.exists())
                holder.stdin.close()
                self.assertEqual(holder.wait(10), 9)  # it crashed: the system released its lock
            finally:
                holder.kill()
        self.assertEqual(WorkFolder.sweep(self.root), 1)
        self.assertFalse(folder.exists())

    def test_the_sweep_never_takes_its_own_processs_folder_even_where_a_lock_never_refuses_its_owner(self):
        live = WorkFolder.make(self.root)
        with mock.patch.object(fcntl, "flock"):  # as on NFS, where a process may take its own lock again
            self.assertEqual(WorkFolder.sweep(self.root), 0)
        self.assertTrue(live.path.exists())
        live.remove()

    def test_the_process_environment_is_the_allowlist_the_agents_needs_and_env_only(self):
        block = {
            "prompts": PROMPTS,
            "agents": AGENT_BLOCKS,
            "command": ["hermes", "acp"],
            "env": {"HERMES_HOME": "/srv/hermes"},
            "pass_env": ["PATH", "UNSET_NAME"],
        }
        os.environ["HARNESS_TEST_SECRET"] = "must-not-pass"
        try:
            plain = RuntimeOpener(HarnessSettings.parse(block)).environment()
            hermes = RuntimeOpener(HarnessSettings.parse({**block, "agent": "hermes"})).environment()
            overridden = {**block, "agent": "hermes", "env": {"HERMES_ACP_SKIP_CONFIGURED_MCP": "0"}}
            operator = RuntimeOpener(HarnessSettings.parse(overridden)).environment()
        finally:
            del os.environ["HARNESS_TEST_SECRET"]
        self.assertEqual(plain, {"PATH": os.environ["PATH"], "HERMES_HOME": "/srv/hermes"})
        self.assertEqual(hermes, {**plain, "HERMES_ACP_SKIP_CONFIGURED_MCP": "1"})  # what Hermes needs
        self.assertEqual(operator["HERMES_ACP_SKIP_CONFIGURED_MCP"], "0")  # the configured env has the last word


class _StubbornClient(FakeClient):
    async def close(self) -> None:
        raise RuntimeError("the agent would not stop")


class RuntimeProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_runtime_that_cannot_start_leaves_no_work_folder(self):
        root = scratch("harness-work-")
        block = {"prompts": PROMPTS, "command": ["/nonexistent/runtime"], "work_root": str(root)}
        opener = RuntimeOpener(HarnessSettings.parse(block))
        with self.assertRaises(FileNotFoundError):
            await opener.open(Role("workers", "w"))
        self.assertEqual(list(root.iterdir()), [])

    async def test_closing_a_runtime_process_ends_its_client_then_deletes_its_folder_even_when_the_close_fails(self):
        root = scratch("harness-work-")
        client, work = FakeClient(), WorkFolder.make(root)
        await RuntimeProcess(client, work).close()
        self.assertEqual((client.alive, work.path.exists()), (False, False))
        stubborn = WorkFolder.make(root)
        with self.assertRaisesRegex(RuntimeError, "would not stop"):
            await RuntimeProcess(_StubbornClient(), stubborn).close()
        self.assertEqual(list(root.iterdir()), [])


ECHO = (
    "import sys, os\n"
    "sys.stdout.write(os.getcwd() + ' ' + os.environ.get('MARK', '-') + ' ' + os.environ.get('HOME', '-') + '\\n')\n"
    "sys.stdout.flush()\n"
    "for line in sys.stdin:\n"
    "    sys.stdout.write(line)\n"
    "    sys.stdout.flush()\n"
)
# Linux's prctl options (<linux/prctl.h>) for a child subreaper: the process that adopts its descendants' orphans.
PR_SET_CHILD_SUBREAPER, PR_GET_CHILD_SUBREAPER = 36, 37


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
    async def test_it_starts_the_command_in_its_work_folder_with_only_the_given_environment_and_closes_it(self):
        folder = scratch("harness-proc-")
        launch = RuntimeLaunch((sys.executable, "-I", "-c", ECHO), {"MARK": "given"}, None, folder)
        transport = StdioTransport.for_launch(launch, stop_grace_seconds=2)
        with warnings_logged() as warnings:
            reader, writer = await transport.open()
            started = (await asyncio.wait_for(reader.readline(), 5)).decode().split()
            self.assertEqual(started, [str(folder), "given", "-"])  # its folder; its environment and nothing else
            writer.write(b'{"jsonrpc": "2.0"}\n')
            await writer.drain()
            self.assertEqual(await asyncio.wait_for(reader.readline(), 5), b'{"jsonrpc": "2.0"}\n')
            await transport.close()
        self.assertTrue(transport.closed.is_set())
        self.assertEqual(transport.process.returncode, 0)  # stdin closed: it ended on its own
        self.assertEqual(warnings, [])  # the exit it asked for

    async def test_a_runtime_that_ignores_end_of_file_is_terminated(self):
        folder = scratch("harness-proc-")
        stubborn = "import time\nwhile True:\n    time.sleep(1)\n"
        transport = StdioTransport.for_launch(
            RuntimeLaunch((sys.executable, "-I", "-c", stubborn), {}, None, folder), stop_grace_seconds=0.3
        )
        await transport.open()
        await transport.close()
        self.assertTrue(transport.closed.is_set())
        self.assertNotEqual(transport.process.returncode, 0)

    async def test_an_exit_it_did_not_ask_for_is_a_warning_with_the_exit_code_and_the_last_stderr_lines(self):
        crash = "echo starting >&2; echo 'No usable credentials found for the provider' >&2; exit 3"
        launch = RuntimeLaunch(("sh", "-c", crash), {"PATH": os.environ["PATH"]}, None, scratch("harness-proc-"))
        transport = StdioTransport.for_launch(launch, stop_grace_seconds=2)
        with warnings_logged() as warnings:
            await transport.open()
            await asyncio.wait_for(transport.closed.wait(), 5)
            await transport.close()
        self.assertEqual(len(warnings), 1, warnings)
        self.assertIn(f"pid={transport.process.pid} code=3", warnings[0])
        self.assertIn("starting | No usable credentials found for the provider", warnings[0])

    async def test_its_exit_is_seen_and_its_close_ends_its_group_while_a_process_it_left_holds_its_pipes(self):
        left = "sleep 30 & sleep 0.5; exit 1"  # the child keeps the agent's stdout and stderr open
        launch = RuntimeLaunch(("sh", "-c", left), {"PATH": os.environ["PATH"]}, None, scratch("harness-proc-"))
        transport = StdioTransport.for_launch(launch, stop_grace_seconds=1)
        await transport.open()
        group = transport.process.pid
        try:
            await asyncio.wait_for(transport.closed.wait(), 2)
            await transport.close()
            with self.assertRaises(ProcessLookupError):
                os.killpg(group, 0)  # the process it left was ended with it
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(group, signal.SIGKILL)

    @unittest.skipUnless(sys.platform == "linux", "a child subreaper and /proc are Linux's")
    async def test_its_close_does_not_wait_out_the_grace_for_a_zombie_left_in_its_group(self):
        # This test adopts the process the agent leaves and never reaps it, as the app container's PID 1 does: once
        # that process ends, it stays in the agent's group as a zombie.
        if not self.enterContext(orphans_adopted()):
            self.skipTest("this process may not be a child subreaper here")
        left = "read x; sleep 0.2 </dev/null >/dev/null 2>&1 & exit 0"  # at end-of-file it leaves a sleep and exits
        launch = RuntimeLaunch(("sh", "-c", left), {"PATH": os.environ["PATH"]}, None, scratch("harness-proc-"))
        transport = StdioTransport.for_launch(launch, stop_grace_seconds=2)
        await transport.open()
        group = transport.process.pid
        self.addCleanup(reap, group)
        began = time.monotonic()
        await transport.close()
        self.assertLess(time.monotonic() - began, 1)  # under half a grace step: no close step waits for the zombie
        self.assertEqual(set(group_states(group).values()), {"Z"})  # nothing of its group runs; the zombie stays


GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


REQUESTS: list[str] = []


async def _handshake(reader, writer):
    """Accept one WebSocket connection (RFC 6455), the standard library only."""
    head = await reader.readuntil(b"\r\n\r\n")
    REQUESTS.append(head.decode())
    key = next(
        line.split(":", 1)[1].strip()
        for line in head.decode().split("\r\n")
        if line.lower().startswith("sec-websocket-key")
    )
    accept = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
    writer.write(
        (
            "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        ).encode()
    )


async def _frame(reader) -> tuple[int, bytes]:
    """One frame from the client (masked): its opcode and payload."""
    first, second = await reader.readexactly(2)
    length = second & 0x7F
    if length == 126:
        (length,) = struct.unpack(">H", await reader.readexactly(2))
    elif length == 127:
        (length,) = struct.unpack(">Q", await reader.readexactly(8))
    mask = await reader.readexactly(4)
    return first & 0x0F, bytes(b ^ mask[i % 4] for i, b in enumerate(await reader.readexactly(length)))


async def _echo(reader, writer):
    """A one-connection WebSocket echo server (text frames)."""
    await _handshake(reader, writer)
    while True:
        opcode, payload = await _frame(reader)
        if opcode == 0x8:  # close
            writer.write(b"\x88\x00")
            await writer.drain()
            writer.close()
            return
        header = (
            bytes([0x81, len(payload)]) if len(payload) < 126 else bytes([0x81, 126]) + struct.pack(">H", len(payload))
        )
        writer.write(header + payload)
        await writer.drain()


async def _closes(reader, writer):
    """A server whose agent ends at once: it closes the connection with 1011 and a reason, then answers the close."""
    await _handshake(reader, writer)
    reason = struct.pack(">H", 1011) + b"agent crashed"
    writer.write(bytes([0x88, len(reason)]) + reason)
    await writer.drain()
    with contextlib.suppress(asyncio.IncompleteReadError, ConnectionError):
        await _frame(reader)  # the client's close
    writer.close()


class WebSocketTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_line_is_one_frame_and_each_frame_one_line(self):
        server = await asyncio.start_server(_echo, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        launch = RuntimeLaunch((), {"KEY": "never sent"}, f"ws://127.0.0.1:{port}/acp", Path("/work/work-1"))
        transport = WebSocketTransport.for_launch(launch, stop_grace_seconds=2)
        reader, writer = await transport.open()
        # The URL as the operator serves it: no query of ours, and nothing of this side's environment.
        self.assertEqual(REQUESTS[-1].split("\r\n")[0], "GET /acp HTTP/1.1")
        self.assertNotIn("never sent", REQUESTS[-1])
        writer.write(b'{"a": 1}\n{"b"')
        writer.write(b": 2}\n")
        await writer.drain()
        self.assertEqual(await asyncio.wait_for(reader.readline(), 5), b'{"a": 1}\n')
        self.assertEqual(await asyncio.wait_for(reader.readline(), 5), b'{"b": 2}\n')
        with warnings_logged() as warnings:
            await transport.close()
        self.assertTrue(transport.closed.is_set())
        self.assertEqual(warnings, [])  # the close it asked for
        server.close()
        await server.wait_closed()

    async def test_a_connection_the_agents_side_closes_ends_the_stream_with_a_warning_of_its_code_and_reason(self):
        server = await asyncio.start_server(_closes, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        launch = RuntimeLaunch((), {}, f"ws://127.0.0.1:{port}/acp", Path("/work/work-1"))
        transport = WebSocketTransport.for_launch(launch, stop_grace_seconds=2)
        with warnings_logged() as warnings:
            reader, _ = await transport.open()
            await asyncio.wait_for(transport.closed.wait(), 5)
        self.assertEqual(await asyncio.wait_for(reader.read(), 5), b"")  # the stream ended, as at an exit
        self.assertEqual(len(warnings), 1, warnings)
        self.assertIn("code=1011 reason='agent crashed'", warnings[0])
        await transport.close()
        server.close()
        await server.wait_closed()

    def test_it_needs_a_websocket_url_and_stdio_a_command(self):
        for url in (None, "", "http://agent:8790/acp"):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, "ws:// or wss:// URL"):
                WebSocketTransport.check(("hermes", "acp"), url)
        WebSocketTransport.check((), "wss://agent.example/acp")
        with self.assertRaisesRegex(ValueError, "must name the agent to start"):
            StdioTransport.check((), "ws://agent:8790/acp")


class ConnectTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_reaching_a_served_agent_is_bounded_by_the_start_timeout_not_the_stop_grace(self):
        from delegation_core.backend.harness.protocol import Role

        held = []  # the connections a server that never answers the handshake keeps

        async def hold(reader, writer):
            held.append(writer)

        server = await asyncio.start_server(hold, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        block = {
            "prompts": PROMPTS,
            "transport": "websocket",
            "url": f"ws://127.0.0.1:{port}/acp",
            "start_timeout_seconds": 0.5,
            "stop_grace_seconds": 5,
            "work_root": str(scratch("harness-work-")),
        }
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            with self.assertRaisesRegex(TimeoutError, r"start_timeout_seconds \(0.5 s\)"):
                await RuntimeOpener(HarnessSettings.parse(block)).open(Role("workers", "w"))
            self.assertLess(loop.time() - started, 2)
        finally:
            for writer in held:
                writer.close()
            server.close()
            await server.wait_closed()


if __name__ == "__main__":
    unittest.main()
