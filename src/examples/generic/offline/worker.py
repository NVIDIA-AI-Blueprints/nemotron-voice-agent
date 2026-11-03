# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private single-stream worker for the optional offline process pool.

The wrapper owns admission and handler-return evidence. It does not change the
native pipeline, interpret audio, or attest that every provider reached EOF.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import importlib
import json
import os
import re
import signal
import socket
import threading
import time
from contextlib import suppress

TOKEN_ENV = "NVA_OFFLINE_POOL_TOKEN"
TOKEN_HEADER = "x-nva-offline-pool-token"
LEASE_HEADER = "x-nva-offline-pool-lease"
STATUS_PATH = "/__offline_pool/status"
OFFLINE_PATH = "/api/ws/offline"
_LEASE = re.compile(rb"[0-9a-f]{32}\Z")
_TOKEN = re.compile(r"[A-Za-z0-9_-]{32,512}\Z")


def _token_bytes(token: str) -> bytes:
    if not isinstance(token, str) or _TOKEN.fullmatch(token) is None:
        raise ValueError("Offline worker requires a strong private pool token")
    return token.encode("ascii")


class OfflineWorkerApp:
    """Authenticate private traffic and admit exactly one offline ASGI handler."""

    def __init__(self, app, *, token: str, telemetry=None):
        """Retain the existing native app and an operator-owned private token."""
        self._app = app
        self._token = _token_bytes(token)
        self._lock = threading.Lock()
        self._active_lease: str | None = None
        self._last_completed_lease: str | None = None
        self._outcome = "idle"
        self._telemetry = telemetry

    def status(self) -> dict:
        """Expose scalar handler lifecycle evidence, never provider completion."""
        with self._lock:
            return {
                "pid": os.getpid(),
                "busy": self._active_lease is not None,
                "active_lease": self._active_lease,
                "last_completed_lease": self._last_completed_lease,
                "outcome": self._outcome,
            }

    @staticmethod
    def _headers(scope, name):
        return [value for key, value in scope.get("headers", ()) if key.lower() == name]

    def _authenticated(self, scope):
        values = self._headers(scope, TOKEN_HEADER.encode("ascii"))
        return len(values) == 1 and isinstance(values[0], bytes) and hmac.compare_digest(values[0], self._token)

    @staticmethod
    def _native_scope(scope):
        private = {TOKEN_HEADER.encode("ascii"), LEASE_HEADER.encode("ascii")}
        return {
            **scope,
            "headers": [(key, value) for key, value in scope.get("headers", ()) if key.lower() not in private],
        }

    @staticmethod
    async def _json(send, status, value):
        body = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        """Delegate authenticated native work while retaining its lease through cleanup."""
        kind = scope.get("type")
        if kind == "lifespan":
            try:
                return await self._app(scope, receive, send)
            finally:
                if self._telemetry is not None:
                    await self._telemetry.aclose()
        if kind not in {"http", "websocket"}:
            raise ValueError("Unsupported offline worker ASGI scope")
        if not self._authenticated(scope):
            if kind == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            else:
                await self._json(send, 401, {"error": "unauthorized"})
            return
        if kind == "http":
            if scope.get("path") == STATUS_PATH:
                if scope.get("method") == "GET":
                    await self._json(send, 200, self.status())
                else:
                    await self._json(send, 405, {"error": "method_not_allowed"})
                return
            return await self._app(self._native_scope(scope), receive, send)
        if scope.get("path") != OFFLINE_PATH:
            await send({"type": "websocket.close", "code": 1008})
            return
        leases = self._headers(scope, LEASE_HEADER.encode("ascii"))
        if len(leases) != 1 or not isinstance(leases[0], bytes) or _LEASE.fullmatch(leases[0]) is None:
            await send({"type": "websocket.close", "code": 1008})
            return
        lease = leases[0].decode("ascii")
        with self._lock:
            busy = self._active_lease is not None
            if not busy:
                self._active_lease = lease
                self._outcome = "running"
        if busy:
            # A trusted internal connection receives an observable WS retry
            # code. It never enters the native app or acquires this worker.
            message = await receive()
            if message.get("type") == "websocket.connect":
                await send({"type": "websocket.accept"})
                await send({"type": "websocket.close", "code": 1013})
            return
        outcome = "error"
        from .pool_telemetry import lease_digest

        holder = {
            "lease_sha256": lease_digest(lease),
            "handler_started_monotonic_ns": time.monotonic_ns(),
            "server_metrics_session_id": None,
        }
        native_scope = self._native_scope(scope)
        native_scope["nva.offline.pool"] = holder
        try:
            await self._app(native_scope, receive, send)
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        else:
            outcome = "returned"
        finally:
            exited = time.monotonic_ns()
            with self._lock:
                self._last_completed_lease = lease
                self._outcome = outcome
                self._active_lease = None
                released = time.monotonic_ns()
            if self._telemetry is not None:
                # The native app has returned. The optional writer schedules
                # bounded I/O without interpreting a socket or provider EOF.
                with suppress(Exception):
                    self._telemetry.submit(
                        "worker",
                        holder["lease_sha256"],
                        worker_pid=os.getpid(),
                        handler_started_monotonic_ns=holder["handler_started_monotonic_ns"],
                        native_app_exit_monotonic_ns=exited,
                        lease_released_monotonic_ns=released,
                        diagnostics_persistence_started_monotonic_ns=holder.get(
                            "diagnostics_persistence_started_monotonic_ns"
                        ),
                        diagnostics_persistence_completed_monotonic_ns=holder.get(
                            "diagnostics_persistence_completed_monotonic_ns"
                        ),
                        server_metrics_session_id=holder["server_metrics_session_id"],
                        outcome=outcome,
                    )


def build_worker_app(factory_path: str = "server:app_factory") -> OfflineWorkerApp:
    """Load the trusted native factory without changing provider initialization."""
    token = os.environ.get(TOKEN_ENV, "")
    _token_bytes(token)
    module, separator, attribute = factory_path.partition(":")
    if not separator or not module or not attribute or "." in attribute:
        raise ValueError("Offline worker factory must be module:callable")
    factory = getattr(importlib.import_module(module), attribute)
    if not callable(factory):
        raise ValueError("Offline worker factory must be callable")
    app = factory()
    if not callable(app):
        raise ValueError("Offline worker factory must return an ASGI app")
    from .pool_telemetry import PoolTelemetry

    return OfflineWorkerApp(app, token=token, telemetry=PoolTelemetry.from_environment())


def _watch_parent(parent_pid: int, stop: threading.Event) -> None:
    """Terminate only this process if its original supervisor has disappeared."""
    while not stop.wait(0.25):
        if os.getppid() != parent_pid:
            if hasattr(signal, "SIGKILL"):
                os.kill(os.getpid(), signal.SIGKILL)
            else:
                os._exit(1)
            return


def _inherited_listener(descriptor: int) -> socket.socket:
    """Duplicate a prebound private IPv4 TCP socket and ensure it is listening."""
    if descriptor < 3:
        raise ValueError("Offline worker requires a private inherited socket")
    duplicate = os.dup(descriptor)
    try:
        listener = socket.socket(fileno=duplicate)
    except BaseException:
        with suppress(OSError):
            os.close(duplicate)
        raise
    try:
        address = listener.getsockname()
        if (
            listener.family != socket.AF_INET
            or listener.type != socket.SOCK_STREAM
            or not isinstance(address, tuple)
            or len(address) != 2
            or address[0] != "127.0.0.1"
            or address[1] <= 0
        ):
            raise ValueError("Offline worker socket must be prebound to IPv4 loopback")
        # SO_ACCEPTCONN inspection is unavailable on some supported Darwin
        # runtimes. listen() is idempotent for listeners, preserves the bound
        # endpoint, and rejects a connected socket without trusting its state.
        listener.listen(socket.SOMAXCONN)
        return listener
    except BaseException:
        listener.close()
        raise


def main(argv=None):
    """Serve one guarded worker using a supervisor-owned inherited socket."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--factory", default="server:app_factory")
    parser.add_argument("--host", choices=("127.0.0.1",), default="127.0.0.1")
    args = parser.parse_args(argv)
    if args.parent_pid <= 1 or os.getppid() != args.parent_pid:
        parser.error("Offline worker supervisor identity is unavailable")
    stop = threading.Event()
    watcher = threading.Thread(target=_watch_parent, args=(args.parent_pid, stop), daemon=True)
    watcher.start()
    listener = None
    try:
        listener = _inherited_listener(args.fd)
        os.environ["UVICORN_WORKERS"] = "1"
        app = build_worker_app(args.factory)
        import uvicorn

        config = uvicorn.Config(
            app,
            host=args.host,
            port=listener.getsockname()[1],
            workers=1,
            access_log=False,
            ws_max_size=8 * 1024 * 1024,
            ws_max_queue=1,
        )
        uvicorn.Server(config).run(sockets=[listener])
        return 0
    finally:
        stop.set()
        watcher.join(timeout=1)
        if listener is not None:
            listener.close()


if __name__ == "__main__":
    raise SystemExit(main())
