# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The private worker owns one native ASGI handler, not a synthetic provider EOF."""

import asyncio
import errno
import json
import os
import signal
import socket
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from examples.generic.offline import worker

TOKEN = "fixture-pool-token-" + "a" * 32
LEASE = "a" * 32
OTHER_LEASE = "b" * 32


def scope(kind="websocket", *, lease=LEASE, path=None, authenticated=True, extra=()):
    """Construct only controlled ASGI headers and source-independent routing."""
    headers = [(b"authorization", b"Bearer native-client-auth")]
    if authenticated:
        headers.append((worker.TOKEN_HEADER.encode(), TOKEN.encode()))
    if lease is not None:
        headers.append((worker.LEASE_HEADER.encode(), lease.encode()))
    headers.extend(extra)
    return {
        "type": kind,
        "path": path or (worker.OFFLINE_PATH if kind == "websocket" else worker.STATUS_PATH),
        "method": "GET",
        "headers": headers,
    }


class OfflineWorkerTests(unittest.IsolatedAsyncioTestCase):
    """An accepted connection retains its reservation through native cleanup."""

    async def status(self, app, **changes):
        """Read the real authenticated status response without a network socket."""
        sent = []

        async def send(value):
            sent.append(value)

        await app(scope("http") | changes, Mock(), send)
        self.assertEqual(sent[0]["status"], 200)
        return json.loads(sent[1]["body"])

    async def test_status_is_authenticated_and_contains_only_lifecycle_scalars(self):
        """Private tokens, native authorization and configuration never enter status."""
        delegate = Mock()
        app = worker.OfflineWorkerApp(delegate, token=TOKEN)
        state = await self.status(app)
        self.assertEqual(
            state,
            {"pid": os.getpid(), "busy": False, "active_lease": None, "last_completed_lease": None, "outcome": "idle"},
        )
        delegate.assert_not_called()
        self.assertNotIn(TOKEN, json.dumps(state))
        self.assertNotIn("native-client-auth", json.dumps(state))

    async def test_missing_bad_duplicate_auth_and_invalid_leases_never_delegate(self):
        """The worker cannot become a second unauthenticated pipeline endpoint."""
        delegate = Mock()
        app = worker.OfflineWorkerApp(delegate, token=TOKEN)
        cases = [
            scope(authenticated=False),
            scope(extra=[(worker.TOKEN_HEADER.encode(), b"wrong")]),
            scope(lease=None),
            scope(lease="not-random-hex"),
            scope(extra=[(worker.LEASE_HEADER.encode(), OTHER_LEASE.encode())]),
            scope(path="/api/ws"),
            scope(path="/v1/realtime"),
        ]
        for request in cases:
            with self.subTest(headers=len(request["headers"]), path=request["path"]):
                sent = []

                async def send(value, sent=sent):
                    sent.append(value)

                await app(request, Mock(), send)
                self.assertEqual(sent, [{"type": "websocket.close", "code": 1008}])
                delegate.assert_not_called()
        sent = []

        async def send(value):
            sent.append(value)

        await app(scope("http", authenticated=False), Mock(), send)
        self.assertEqual(sent[0]["status"], 401)
        self.assertNotIn(TOKEN.encode(), sent[1]["body"])
        self.assertFalse((await self.status(app))["busy"])

    async def test_busy_is_retained_after_disconnect_until_actual_handler_cleanup_returns(self):
        """Socket close alone cannot make a worker eligible for another lease."""
        connected = asyncio.Event()
        cleanup_entered = asyncio.Event()
        cleanup_release = asyncio.Event()
        inbound = asyncio.Queue()
        calls = []

        async def native(request, receive, send):
            calls.append(request)
            self.assertEqual(await receive(), {"type": "websocket.connect"})
            await send({"type": "websocket.accept"})
            connected.set()
            try:
                await receive()
            finally:
                cleanup_entered.set()
                await cleanup_release.wait()

        app = worker.OfflineWorkerApp(native, token=TOKEN)
        sent = []

        async def send(value):
            sent.append(value)

        inbound.put_nowait({"type": "websocket.connect"})
        first = asyncio.create_task(app(scope(), inbound.get, send))
        try:
            await connected.wait()
            inbound.put_nowait({"type": "websocket.disconnect", "code": 1000})
            await cleanup_entered.wait()
            status = await self.status(app)
            self.assertTrue(status["busy"])
            self.assertEqual(status["active_lease"], LEASE)
            self.assertIsNone(status["last_completed_lease"])
            self.assertEqual(status["outcome"], "running")
            second_inbound = asyncio.Queue()
            second_inbound.put_nowait({"type": "websocket.connect"})
            second_sent = []

            async def second_send(value):
                second_sent.append(value)

            await app(scope(lease=OTHER_LEASE), second_inbound.get, second_send)
            self.assertEqual(second_sent, [{"type": "websocket.accept"}, {"type": "websocket.close", "code": 1013}])
            self.assertEqual(len(calls), 1)
            cleanup_release.set()
            await asyncio.wait_for(first, 1)
            self.assertEqual(
                await self.status(app),
                {
                    "pid": os.getpid(),
                    "busy": False,
                    "active_lease": None,
                    "last_completed_lease": LEASE,
                    "outcome": "returned",
                },
            )
            self.assertEqual(calls[0]["headers"], [(b"authorization", b"Bearer native-client-auth")])
        finally:
            cleanup_release.set()
            if not first.done():
                first.cancel()
            await asyncio.gather(first, return_exceptions=True)

    async def test_cancelled_handler_does_not_publish_idle_until_its_finally_finishes(self):
        """Native cancellation is completed evidence only after awaited cleanup."""
        entered = asyncio.Event()
        cleanup_entered = asyncio.Event()
        release = asyncio.Event()

        async def native(request, receive, send):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_entered.set()
                await release.wait()

        app = worker.OfflineWorkerApp(native, token=TOKEN)
        task = asyncio.create_task(app(scope(), Mock(), Mock()))
        await entered.wait()
        task.cancel()
        await cleanup_entered.wait()
        self.assertTrue((await self.status(app))["busy"])
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual((await self.status(app))["outcome"], "cancelled")
        self.assertEqual((await self.status(app))["last_completed_lease"], LEASE)

    async def test_native_error_is_unchanged_and_status_never_contains_its_body(self):
        """The wrapper records only a terminal label, not provider error content."""
        error = RuntimeError("private-provider-body")

        async def native(request, receive, send):
            raise error

        app = worker.OfflineWorkerApp(native, token=TOKEN)
        with self.assertRaises(RuntimeError) as caught:
            await app(scope(), Mock(), Mock())
        self.assertIs(caught.exception, error)
        state = await self.status(app)
        self.assertFalse(state["busy"])
        self.assertEqual(state["outcome"], "error")
        self.assertEqual(state["last_completed_lease"], LEASE)
        self.assertNotIn("private-provider-body", json.dumps(state))

    async def test_trusted_metadata_and_lifespan_delegate_without_taking_a_stream_lease(self):
        """Control requests preserve native auth while pool credentials stay private."""
        calls = []

        async def native(request, receive, send):
            calls.append(request)

        app = worker.OfflineWorkerApp(native, token=TOKEN)
        await app(scope("http", path="/api/deployment"), Mock(), Mock())
        await app({"type": "lifespan"}, Mock(), Mock())
        self.assertEqual(calls[0]["headers"], [(b"authorization", b"Bearer native-client-auth")])
        self.assertEqual(calls[1], {"type": "lifespan"})
        self.assertEqual((await self.status(app))["outcome"], "idle")


class WorkerEntryTests(unittest.TestCase):
    """A child serves only its inherited private listener and watches its parent."""

    def test_factory_requires_token_before_importing_native_configuration(self):
        """Capability/auth failures cannot start any native provider or factory."""
        for token in ("", "short", "bad\n" + "a" * 40):
            with (
                self.subTest(size=len(token)),
                patch.dict(os.environ, {worker.TOKEN_ENV: token}),
                patch.object(worker.importlib, "import_module") as load,
            ):
                with self.assertRaises(ValueError):
                    worker.build_worker_app()
                load.assert_not_called()
        native = Mock()
        factory = Mock(return_value=native)
        with (
            patch.dict(os.environ, {worker.TOKEN_ENV: TOKEN}),
            patch.object(worker.importlib, "import_module", return_value=SimpleNamespace(factory=factory)),
        ):
            app = worker.build_worker_app("fixture:factory")
        self.assertIs(app._app, native)
        factory.assert_called_once_with()

    def test_factory_rejects_invalid_callable_contract(self):
        """A trusted import still has to return a callable ASGI application."""
        for factory in (None, Mock(return_value=None)):
            with (
                self.subTest(callable=callable(factory)),
                patch.dict(os.environ, {worker.TOKEN_ENV: TOKEN}),
                patch.object(worker.importlib, "import_module", return_value=SimpleNamespace(factory=factory)),
                self.assertRaises(ValueError),
            ):
                worker.build_worker_app("fixture:factory")

    def test_parent_watch_signals_only_its_own_pid_after_reparenting(self):
        """Original-parent loss cannot authorize a signal to another process."""
        stop = Mock(spec=threading.Event)
        stop.wait.side_effect = [False, False]
        with (
            patch.object(worker.os, "getppid", side_effect=[123, 1]),
            patch.object(worker.os, "getpid", return_value=456),
            patch.object(worker.os, "kill") as kill,
        ):
            worker._watch_parent(123, stop)
        kill.assert_called_once_with(456, signal.SIGKILL)
        stopped = Mock(spec=threading.Event)
        stopped.wait.return_value = True
        with patch.object(worker.os, "kill") as kill:
            worker._watch_parent(123, stopped)
        kill.assert_not_called()

    def test_parent_watch_uses_self_exit_when_sigkill_is_unavailable(self):
        """The portable fallback cannot invoke an unrelated process identifier."""
        stop = Mock(spec=threading.Event)
        stop.wait.return_value = False
        with (
            patch.object(worker.os, "getppid", return_value=1),
            patch.object(worker, "signal", SimpleNamespace()),
            patch.object(worker.os, "_exit") as exit_process,
            patch.object(worker.os, "kill") as kill,
        ):
            worker._watch_parent(123, stop)
        exit_process.assert_called_once_with(1)
        kill.assert_not_called()

    def test_inherited_listener_rejects_unbound_nonloopback_or_non_tcp_socket(self):
        """A caller cannot accidentally expose an independent worker listener."""
        for changed in (
            {"family": socket.AF_UNIX},
            {"type": socket.SOCK_DGRAM},
            {"address": ("0.0.0.0", 123)},
            {"address": ("127.0.0.1", 0)},
        ):
            with self.subTest(changed=changed):
                candidate = Mock(
                    family=changed.get("family", socket.AF_INET), type=changed.get("type", socket.SOCK_STREAM)
                )
                candidate.getsockname.return_value = changed.get("address", ("127.0.0.1", 123))
                with (
                    patch.object(worker.os, "dup", return_value=91),
                    patch.object(worker.socket, "socket", return_value=candidate),
                    self.assertRaises(ValueError),
                ):
                    worker._inherited_listener(90)
                candidate.close.assert_called_once()
                candidate.listen.assert_not_called()
        with self.assertRaises(ValueError):
            worker._inherited_listener(1)

    def test_inherited_bound_listener_ensures_listening_without_nonportable_socket_option(self):
        """A valid bound private FD is accepted only after listen succeeds."""
        candidate = Mock(family=socket.AF_INET, type=socket.SOCK_STREAM)
        candidate.getsockname.return_value = ("127.0.0.1", 123)
        candidate.getsockopt.side_effect = OSError(errno.ENOPROTOOPT, "unavailable socket option")
        with (
            patch.object(worker.os, "dup", return_value=91),
            patch.object(worker.socket, "socket", return_value=candidate),
        ):
            self.assertIs(worker._inherited_listener(90), candidate)
        candidate.listen.assert_called_once_with(socket.SOMAXCONN)
        candidate.getsockopt.assert_not_called()
        candidate.close.assert_not_called()

    def test_connected_inherited_tcp_socket_cannot_become_a_listener(self):
        """A failed listen keeps the original error and closes only the duplicate."""
        candidate = Mock(family=socket.AF_INET, type=socket.SOCK_STREAM)
        candidate.getsockname.return_value = ("127.0.0.1", 123)
        error = OSError(errno.EINVAL, "connected socket")
        candidate.listen.side_effect = error
        with (
            patch.object(worker.os, "dup", return_value=91),
            patch.object(worker.socket, "socket", return_value=candidate),
            self.assertRaises(OSError) as caught,
        ):
            worker._inherited_listener(90)
        self.assertIs(caught.exception, error)
        candidate.close.assert_called_once_with()

    def test_inherited_listener_closes_only_owned_duplicate_on_constructor_error(self):
        """Invalid inherited descriptors retain the supervisor's ownership."""
        error = OSError("not a socket")
        with (
            patch.object(worker.os, "dup", return_value=91),
            patch.object(worker.socket, "socket", side_effect=error),
            patch.object(worker.os, "close") as close,
            self.assertRaises(OSError) as caught,
        ):
            worker._inherited_listener(90)
        self.assertIs(caught.exception, error)
        close.assert_called_once_with(91)

    def test_entry_uses_one_uvicorn_worker_and_stops_watcher_on_normal_return(self):
        """Existing native configuration is instantiated only after the parent guard."""
        listener = Mock()
        listener.getsockname.return_value = ("127.0.0.1", 12345)
        native = Mock()
        with (
            patch.object(worker.os, "getppid", return_value=123),
            patch.object(worker, "_inherited_listener", return_value=listener),
            patch.object(worker, "build_worker_app", return_value=native) as factory,
            patch.object(worker.threading, "Thread") as thread,
            patch("uvicorn.Config") as config,
            patch("uvicorn.Server") as server,
            patch.dict(os.environ, {"UVICORN_WORKERS": "10"}),
        ):
            self.assertEqual(worker.main(["--fd", "90", "--parent-pid", "123", "--factory", "fixture:factory"]), 0)
            factory.assert_called_once_with("fixture:factory")
            self.assertEqual(os.environ["UVICORN_WORKERS"], "1")
            self.assertEqual(config.call_args.kwargs["workers"], 1)
            self.assertEqual(config.call_args.kwargs["ws_max_size"], 8 * 1024 * 1024)
            self.assertEqual(config.call_args.kwargs["ws_max_queue"], 1)
            server.return_value.run.assert_called_once_with(sockets=[listener])
            thread.return_value.start.assert_called_once_with()
            thread.return_value.join.assert_called_once_with(timeout=1)
        listener.close.assert_called_once()
