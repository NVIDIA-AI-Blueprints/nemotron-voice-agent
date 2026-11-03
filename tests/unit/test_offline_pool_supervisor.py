# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D102, D107

"""Real loopback processes prove placement and cleanup independently of models."""

import asyncio
import json
import os
import socket as socketlib
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
import uvicorn
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from examples.generic.offline import pool_supervisor as supervisor_module
from examples.generic.offline.pool_proxy import ExclusiveWorkerPool, create_pool_proxy
from examples.generic.offline.pool_supervisor import PoolStartupError, WorkerSupervisor, _OwnedWorker
from examples.generic.offline.worker import TOKEN_HEADER


class PoolConfigurationTests(unittest.TestCase):
    """Reject resource and watchdog settings before spawning a child."""

    def test_invalid_pool_size_and_watchdogs(self):
        for count in (True, 0, -1, 65, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                WorkerSupervisor(count)
        for timeout in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                WorkerSupervisor(startup_timeout_s=timeout)


class _ControlledProcess:
    """An owned subprocess handle with explicit TERM/reap outcomes, without a PID signal."""

    def __init__(self, pid, *, ignore_term=False):
        self.pid = pid
        self.returncode = None
        self.ignore_term = ignore_term
        self.wait_error = False
        self.terminate_calls = 0
        self.kill_calls = 0
        self.terminated = asyncio.Event()
        self.waiting = asyncio.Event()
        self.exited = asyncio.Event()

    def terminate(self):
        self.terminate_calls += 1
        self.terminated.set()
        if not self.ignore_term:
            self.returncode = 0
            self.exited.set()

    def kill(self):
        self.kill_calls += 1
        self.returncode = -9
        self.exited.set()

    async def wait(self):
        self.waiting.set()
        if self.wait_error:
            raise RuntimeError("controlled wait failure")
        await self.exited.wait()
        return self.returncode


class SupervisorFailureTests(unittest.IsolatedAsyncioTestCase):
    """Partial startup and cancelled callers must not abandon owned children."""

    def listener(self):
        listener = Mock()
        listener.fileno.return_value = 91
        listener.getsockname.return_value = ("127.0.0.1", 40123)
        return listener

    async def test_partial_spawn_failure_closes_listeners_and_reaps_only_owned_process(self):
        process = _ControlledProcess(12345)
        failure = OSError("controlled second spawn failure")
        listeners = [self.listener(), self.listener()]
        supervisor = WorkerSupervisor(3, stop_timeout_s=0.01)
        with (
            patch.object(supervisor_module.socket, "socket", side_effect=listeners),
            patch.object(
                supervisor_module.asyncio, "create_subprocess_exec", AsyncMock(side_effect=[process, failure])
            ),
            self.assertRaises(OSError) as caught,
        ):
            await supervisor.start()
        self.assertIs(caught.exception, failure)
        self.assertEqual(supervisor.owned_pids, (process.pid,))
        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.returncode, 0)
        self.assertTrue(supervisor._closed)
        for listener in listeners:
            listener.close.assert_called_once_with()
        with self.assertRaises(PoolStartupError):
            await supervisor.start()
        await supervisor.aclose()
        self.assertEqual(process.terminate_calls, 1)

    async def test_cancelled_readiness_stops_every_successfully_spawned_child(self):
        processes = [_ControlledProcess(12345), _ControlledProcess(12346)]
        supervisor = WorkerSupervisor(2, stop_timeout_s=0.01)
        entered = asyncio.Event()
        ready_calls = []

        async def ready(client, worker, deadline):
            ready_calls.append(worker.process.pid)
            if len(ready_calls) == 2:
                entered.set()
            await asyncio.Event().wait()

        with (
            patch.object(supervisor_module.socket, "socket", side_effect=[self.listener(), self.listener()]),
            patch.object(supervisor_module.asyncio, "create_subprocess_exec", AsyncMock(side_effect=processes)),
            patch.object(supervisor, "_ready", ready),
        ):
            task = asyncio.create_task(supervisor.start())
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
        self.assertEqual(supervisor.owned_pids, (12345, 12346))
        self.assertTrue(all(process.returncode == 0 for process in processes))
        self.assertTrue(supervisor._closed)

    async def test_cancelled_shutdown_joins_kill_and_reap_before_reraising(self):
        process = _ControlledProcess(12345, ignore_term=True)
        supervisor = WorkerSupervisor(1, stop_timeout_s=0.01)
        supervisor._workers = [_OwnedWorker("worker-01", "http://127.0.0.1:40123", process)]
        task = asyncio.create_task(supervisor.aclose())
        await asyncio.wait_for(process.waiting.wait(), 1)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 1)
        self.assertEqual(process.returncode, -9)
        self.assertTrue(supervisor._shutdown_task.done())
        self.assertTrue(supervisor._closed)
        await supervisor.aclose()
        self.assertEqual(process.kill_calls, 1)

    async def test_startup_failure_retains_primary_error_when_cleanup_also_fails(self):
        process = _ControlledProcess(12345, ignore_term=True)
        process.wait_error = True
        failure = OSError("controlled second spawn failure")
        supervisor = WorkerSupervisor(2, stop_timeout_s=0.01)
        with (
            patch.object(supervisor_module.socket, "socket", side_effect=[self.listener(), self.listener()]),
            patch.object(
                supervisor_module.asyncio, "create_subprocess_exec", AsyncMock(side_effect=[process, failure])
            ),
            self.assertRaises(OSError) as caught,
        ):
            await supervisor.start()
        self.assertIs(caught.exception, failure)
        self.assertIsInstance(failure.__cause__, PoolStartupError)
        self.assertIn("offline_pool_owned_backend_shutdown_failed", failure.__notes__)
        self.assertTrue(supervisor._closed)
        self.assertIsNone(process.returncode)
        process.wait_error = False
        await supervisor.aclose()
        self.assertEqual(process.returncode, -9)

    async def test_new_cancellation_during_startup_cleanup_remains_cancellation_after_reap(self):
        process = _ControlledProcess(12345, ignore_term=True)
        failure = OSError("controlled second spawn failure")
        supervisor = WorkerSupervisor(2, stop_timeout_s=0.01)
        with (
            patch.object(supervisor_module.socket, "socket", side_effect=[self.listener(), self.listener()]),
            patch.object(
                supervisor_module.asyncio, "create_subprocess_exec", AsyncMock(side_effect=[process, failure])
            ),
        ):
            task = asyncio.create_task(supervisor.start())
            await asyncio.wait_for(process.waiting.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError) as caught:
                await asyncio.wait_for(task, 1)
        self.assertIs(caught.exception.__cause__, failure)
        self.assertIn("offline_pool_startup_failed:OSError", caught.exception.__notes__)
        self.assertEqual(process.returncode, -9)
        self.assertTrue(supervisor._closed)


@unittest.skipUnless(os.name == "posix", "Inherited loopback sockets require POSIX")
class ManagedPoolProcessTests(unittest.IsolatedAsyncioTestCase):
    """Use ten actual private listeners and one public proxy, with no providers."""

    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.release = Path(self.temporary.name)
        repo = Path(__file__).resolve().parents[2]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join((str(repo / "src"), str(repo / "tests/unit")))
        environment["OFFLINE_POOL_FIXTURE_RELEASE_DIR"] = str(self.release)
        self.supervisor = WorkerSupervisor(
            10,
            worker_factory="offline_pool_fixture_app:create_app",
            environment=environment,
            startup_timeout_s=30,
            stop_timeout_s=0.5,
        )
        self.addAsyncCleanup(self.supervisor.aclose)
        self.backends = await self.supervisor.start()
        self.pool = ExclusiveWorkerPool(self.backends, token=self.supervisor.token, drain_timeout_s=5)
        self.addAsyncCleanup(self.pool.aclose)
        initial = await self.pool.initialize()
        self.assertEqual(initial["idle"], 10)
        self.listener = socketlib.socket(socketlib.AF_INET, socketlib.SOCK_STREAM)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(socketlib.SOMAXCONN)
        self.addCleanup(self.listener.close)
        port = self.listener.getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}/api/ws/offline"
        self.server = uvicorn.Server(
            uvicorn.Config(
                create_pool_proxy(self.pool),
                lifespan="off",
                access_log=False,
                log_config=None,
                log_level="critical",
                ws_max_size=8 * 1024 * 1024,
                ws_max_queue=1,
            )
        )
        self.server_task = asyncio.create_task(self.server.serve(sockets=[self.listener]))

        async def close_front():
            self.server.should_exit = True
            await asyncio.wait_for(self.server_task, 5)

        self.addAsyncCleanup(close_front)
        async with asyncio.timeout(5):
            while not self.server.started:
                if self.server_task.done():
                    self.server_task.result()
                await asyncio.sleep(0.01)

    async def connection(self):
        # websockets15 is the locked project dependency; proxy=None prevents
        # ambient corporate proxy configuration from changing this local test.
        socket = await connect(self.url, proxy=None, max_queue=1, close_timeout=1)
        self.addAsyncCleanup(socket.close)
        pid = json.loads(await asyncio.wait_for(socket.recv(), 3))["pid"]
        return socket, pid

    async def release_and_wait(self, pids):
        for pid in pids:
            (self.release / str(pid)).touch()
        async with asyncio.timeout(5):
            while (await self.pool.status())["idle"] != 10:
                await asyncio.sleep(0.02)

    async def test_six_streams_have_six_distinct_processes_and_exact_wire_frames(self):
        sessions = await asyncio.gather(*(self.connection() for _ in range(6)))
        pids = [pid for _, pid in sessions]
        self.assertEqual(len(set(pids)), 6)
        self.assertTrue(set(pids) <= set(self.supervisor.owned_pids))
        self.assertEqual((await self.pool.status())["leased"], 6)
        binary = b"NVAI\x00\xff\x01\x02" * 20
        text = '{"type":"opaque","sequence":1}'
        for socket, _ in sessions:
            await socket.send(binary)
            self.assertEqual(await socket.recv(), binary)
            await socket.send(text)
            self.assertEqual(await socket.recv(), text)
        await asyncio.gather(*(socket.close() for socket, _ in sessions))
        async with asyncio.timeout(3):
            while (await self.pool.status())["draining"] != 6:
                await asyncio.sleep(0.01)
        # A completed frontend close has not made any backend reusable.
        self.assertEqual((await self.pool.status())["idle"], 4)
        async with httpx.AsyncClient(trust_env=False) as client:
            for backend in self.backends:
                if backend.pid in pids:
                    response = await client.get(
                        backend.base_url + "/__offline_pool/status", headers={TOKEN_HEADER: self.supervisor.token}
                    )
                    self.assertTrue(response.json()["busy"])
        await self.release_and_wait(pids)
        final = await self.pool.status()
        self.assertEqual(final["quarantined"], 0)
        self.assertTrue(all(worker["max_reserved_streams"] <= 1 for worker in final["workers"]))

    async def test_eleventh_fully_active_stream_is_rejected_without_reusing_a_process(self):
        sessions = await asyncio.gather(*(self.connection() for _ in range(10)))
        self.assertEqual(len({pid for _, pid in sessions}), 10)
        started = time.monotonic()
        async with connect(self.url, proxy=None) as extra:
            with self.assertRaises(ConnectionClosed) as closed:
                await extra.recv()
            self.assertEqual(closed.exception.rcvd.code, 1013)
        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual((await self.pool.status())["leased"], 10)
        for _, pid in sessions:
            (self.release / str(pid)).touch()
        await asyncio.gather(*(socket.close() for socket, _ in sessions))
        await self.release_and_wait([pid for _, pid in sessions])

    async def test_normal_stop_reaps_all_owned_children_and_closes_private_listeners(self):
        await self.pool.aclose()
        await self.supervisor.aclose()
        self.assertTrue(all(worker.process.returncode is not None for worker in self.supervisor._workers))
        async with httpx.AsyncClient(trust_env=False, timeout=0.3) as client:
            for backend in self.backends:
                with self.assertRaises(httpx.HTTPError):
                    await client.get(backend.base_url + "/__offline_pool/status")
