# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lease one private offline worker per stream without interpreting its payloads."""

import asyncio
import contextlib
import math
import re
import time
import uuid
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, Response
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

TOKEN_HEADER = "X-NVA-Offline-Pool-Token"
LEASE_HEADER = "X-NVA-Offline-Pool-Lease"
_MAX_MESSAGE_BYTES = 8 * 1024 * 1024
_TERMINAL = frozenset({"returned", "cancelled", "error"})
_PRIVATE_HEADERS = frozenset({TOKEN_HEADER.lower(), LEASE_HEADER.lower()})
_HOP_HEADERS = frozenset(
    {
        "host",
        "connection",
        "upgrade",
        "transfer-encoding",
        "content-length",
        "sec-websocket-key",
        "sec-websocket-version",
        "sec-websocket-extensions",
        "sec-websocket-protocol",
    }
)


@dataclass(frozen=True)
class BackendDescriptor:
    """A supervisor-attested private address and immutable process identity."""

    worker_id: str
    base_url: str
    pid: int

    def __post_init__(self):
        """Reject ambiguous descriptors before any child can be selected."""
        parsed = urlsplit(self.base_url)
        if (
            not isinstance(self.worker_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", self.worker_id)
            or type(self.pid) is not int
            or self.pid <= 1
            or parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
            or parsed.port is None
            or not 1 <= parsed.port <= 65535
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Invalid private offline worker descriptor")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    @property
    def websocket_url(self):
        """Use the same private listener for its sole public-data route."""
        parsed = urlsplit(self.base_url)
        return urlunsplit(("ws", parsed.netloc, "/api/ws/offline", "", ""))


@dataclass(frozen=True)
class WorkerLease:
    """An opaque reservation that is never exposed by public pool status."""

    backend: BackendDescriptor
    lease_id: str


@dataclass
class _Slot:
    backend: BackendDescriptor
    state: str = "UNVERIFIED"
    lease_id: str | None = None
    max_reserved_streams: int = 0
    total_leases: int = 0
    quarantine_reason: str | None = None


class _StreamEnded(Exception):
    def __init__(self, code=1000):
        self.code = code


def _headers(incoming, token, lease_id=None):
    forwarded = {name: value for name, value in incoming.items() if name.lower() not in _HOP_HEADERS | _PRIVATE_HEADERS}
    forwarded[TOKEN_HEADER] = token
    if lease_id is not None:
        forwarded[LEASE_HEADER] = lease_id
    return forwarded


def _seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Pool watchdogs must be finite positive seconds")
    return float(value)


class ExclusiveWorkerPool:
    """Reserve a worker through authenticated handler completion, including drain."""

    def __init__(
        self,
        backends,
        *,
        token,
        http_client=None,
        connect=connect,
        drain_timeout_s=60.0,
        poll_interval_s=0.05,
        admission_timeout_s=1.0,
        http_timeout_s=5.0,
        connect_timeout_s=10.0,
        close_timeout_s=5.0,
        max_message_bytes=_MAX_MESSAGE_BYTES,
    ):
        """Keep secrets private and bound every queue and operational wait."""
        backends = tuple(backends)
        if not backends or any(not isinstance(item, BackendDescriptor) for item in backends):
            raise ValueError("A nonempty descriptor pool is required")
        for key in ("worker_id", "base_url", "pid"):
            if len({getattr(item, key) for item in backends}) != len(backends):
                raise ValueError("Worker descriptors must have unique identities and listeners")
        if not isinstance(token, str) or not token or any(char in token for char in "\r\n"):
            raise ValueError("A trusted private worker token is required")
        if type(max_message_bytes) is not int or max_message_bytes <= 0:
            raise ValueError("Proxy message bound must be a positive integer")
        self._slots = {item.worker_id: _Slot(item) for item in backends}
        self._token = token
        self._http_timeout = _seconds(http_timeout_s)
        self._http = http_client or httpx.AsyncClient(timeout=self._http_timeout, trust_env=False)
        self._owns_http = http_client is None
        self._connect = connect
        self._drain_timeout = _seconds(drain_timeout_s)
        self._poll_interval = _seconds(poll_interval_s)
        self._admission_timeout = _seconds(admission_timeout_s)
        self._connect_timeout = _seconds(connect_timeout_s)
        self._close_timeout = _seconds(close_timeout_s)
        self.max_message_bytes = max_message_bytes
        self._condition = asyncio.Condition()
        self._drainers = set()
        self._pending_admissions = 0
        self._peak_reserved = 0
        self._closed = False
        self._cursor = 0
        from .pool_telemetry import PoolTelemetry

        self._telemetry = PoolTelemetry.from_environment()
        self._profiles = {}

    async def _worker_status(self, backend, *, timeout=None):
        response = await asyncio.wait_for(
            self._http.get(backend.base_url + "/__offline_pool/status", headers={TOKEN_HEADER: self._token}),
            timeout=timeout or self._http_timeout,
        )
        if response.status_code != 200:
            raise ValueError("Worker status unavailable")
        data = response.json()
        if (
            not isinstance(data, dict)
            or not {"pid", "busy", "active_lease", "last_completed_lease", "outcome"} <= data.keys()
            or type(data.get("pid")) is not int
            or data["pid"] != backend.pid
            or type(data.get("busy")) is not bool
            or data.get("outcome") not in {"idle", "running", *_TERMINAL}
            or any(
                value is not None and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value))
                for value in (data.get("active_lease"), data.get("last_completed_lease"))
            )
        ):
            raise ValueError("Worker status identity or schema mismatch")
        return data

    async def initialize(self):
        """Verify fresh owned idle children without reserving stream capacity."""

        async def verify(slot):
            try:
                data = await self._worker_status(slot.backend)
                valid = (
                    not data["busy"]
                    and data.get("active_lease") is None
                    and data.get("last_completed_lease") is None
                    and data["outcome"] == "idle"
                )
            except Exception:
                valid = False
            async with self._condition:
                if slot.state == "UNVERIFIED":
                    slot.state = "IDLE" if valid and not self._closed else "QUARANTINED"
                    if slot.state == "QUARANTINED":
                        slot.quarantine_reason = "startup_identity_or_idle_unverified"
                    self._condition.notify_all()

        await asyncio.gather(*(verify(slot) for slot in self._slots.values()))
        return await self.status()

    def _reserve_idle(self, admission_started_ns):
        slots = tuple(self._slots.values())
        for offset in range(len(slots)):
            index = (self._cursor + offset) % len(slots)
            slot = slots[index]
            if slot.state == "IDLE":
                slot.state, slot.lease_id = "LEASED", uuid.uuid4().hex
                slot.total_leases += 1
                slot.max_reserved_streams = 1
                self._cursor = (index + 1) % len(slots)
                reserved = sum(item.lease_id is not None for item in slots)
                self._peak_reserved = max(self._peak_reserved, reserved)
                if self._telemetry is not None:
                    self._profiles[slot.lease_id] = {
                        "worker_id": slot.backend.worker_id,
                        "worker_pid": slot.backend.pid,
                        "admission_started_monotonic_ns": admission_started_ns,
                        "admission_acquired_monotonic_ns": time.monotonic_ns(),
                        "status_polls": 0,
                    }
                return WorkerLease(slot.backend, slot.lease_id)
        return None

    async def acquire(self):
        """Fail immediately if active-full; wait at most 1 s only for known drain."""
        admission_started_ns = time.monotonic_ns()
        async with self._condition:
            if self._closed:
                return None
            lease = self._reserve_idle(admission_started_ns)
            if lease is not None:
                return lease
            if not any(slot.state == "DRAINING" for slot in self._slots.values()) or self._pending_admissions >= len(
                self._slots
            ):
                return None
            self._pending_admissions += 1
            try:
                async with asyncio.timeout(self._admission_timeout):
                    while not self._closed:
                        await self._condition.wait()
                        lease = self._reserve_idle(admission_started_ns)
                        if lease is not None:
                            return lease
                        if not any(slot.state == "DRAINING" for slot in self._slots.values()):
                            return None
            except TimeoutError:
                return None
            finally:
                self._pending_admissions -= 1
            return None

    async def open_backend(self, lease, incoming_headers):
        """Open exactly the reserved backend; never replay on another worker."""
        async with self._condition:
            slot = self._slots.get(lease.backend.worker_id)
            if (
                self._closed
                or slot is None
                or slot.backend != lease.backend
                or slot.state != "LEASED"
                or slot.lease_id != lease.lease_id
            ):
                raise RuntimeError("Offline worker lease is no longer active")
            snapshot = (slot.state, slot.lease_id, slot.total_leases)
        self._profile(lease, "identity_check_started_monotonic_ns")
        try:
            data = await self._worker_status(lease.backend)
        except Exception:
            await self._quarantine_unchanged_snapshot(slot, snapshot, "admission_identity_or_transport_unverified")
            raise RuntimeError("Offline worker admission identity is unverified") from None
        finally:
            self._profile(lease, "identity_check_finished_monotonic_ns")
        async with self._condition:
            if self._closed or (slot.state, slot.lease_id, slot.total_leases) != snapshot:
                raise RuntimeError("Offline worker lease changed during admission")
            if data["busy"] or data.get("active_lease") is not None or data["outcome"] not in {"idle", *_TERMINAL}:
                slot.state, slot.quarantine_reason = "QUARANTINED", "admission_worker_not_idle"
                self._condition.notify_all()
                raise RuntimeError("Offline worker is not idle before admission")
        self._profile(lease, "backend_connect_started_monotonic_ns")
        try:
            return await self._connect(
                lease.backend.websocket_url,
                additional_headers=_headers(incoming_headers, self._token, lease.lease_id),
                open_timeout=self._connect_timeout,
                close_timeout=self._close_timeout,
                max_size=self.max_message_bytes,
                max_queue=1,
                write_limit=65536,
            )
        finally:
            self._profile(lease, "backend_connect_finished_monotonic_ns")

    def _profile(self, lease, name):
        profile = self._profiles.get(lease.lease_id)
        if profile is not None:
            profile[name] = time.monotonic_ns()

    async def _quarantine(self, slot, reason):
        async with self._condition:
            slot.state, slot.quarantine_reason = "QUARANTINED", reason
            self._condition.notify_all()

    async def _quarantine_unchanged_snapshot(self, slot, snapshot, reason):
        async with self._condition:
            if (slot.state, slot.lease_id, slot.total_leases) == snapshot:
                slot.state, slot.quarantine_reason = "QUARANTINED", reason
                self._condition.notify_all()
                return True
            return False

    async def _quarantine_unchanged_idle(self, slot, snapshot, reason):
        if snapshot[0] != "IDLE":
            return False
        return await self._quarantine_unchanged_snapshot(slot, snapshot, reason)

    async def finish(self, lease, backend_socket=None):
        """Begin tracked background drain; socket close alone never frees a slot."""
        self._profile(lease, "relay_finished_monotonic_ns")
        close_after_shutdown = False
        async with self._condition:
            slot = self._slots[lease.backend.worker_id]
            if slot.lease_id != lease.lease_id or slot.state not in {"LEASED", "QUARANTINED"}:
                return
            if self._closed:
                close_after_shutdown = True
            else:
                if slot.state == "LEASED":
                    slot.state = "DRAINING"
                task = asyncio.create_task(self._drain(slot, lease, backend_socket))
                self._drainers.add(task)
                task.add_done_callback(self._drainers.discard)
            self._condition.notify_all()
        if close_after_shutdown and backend_socket is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(backend_socket.close(), timeout=self._close_timeout)

    async def _drain(self, slot, lease, backend_socket):
        deadline = asyncio.get_running_loop().time() + self._drain_timeout
        try:
            if backend_socket is not None:
                self._profile(lease, "backend_close_started_monotonic_ns")
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(backend_socket.close(), timeout=self._close_timeout)
                self._profile(lease, "backend_close_finished_monotonic_ns")
            if slot.state == "QUARANTINED":
                return
            while not self._closed:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    await self._quarantine(slot, "handler_drain_timeout")
                    return
                try:
                    profile = self._profiles.get(lease.lease_id)
                    if profile is not None:
                        profile["status_polls"] += 1
                    data = await self._worker_status(slot.backend, timeout=min(remaining, self._http_timeout))
                except Exception:
                    await self._quarantine(slot, "worker_status_or_identity_unverified")
                    return
                if data["busy"]:
                    if data.get("active_lease") != lease.lease_id or data["outcome"] != "running":
                        await self._quarantine(slot, "foreign_or_unknown_active_lease")
                        return
                elif (
                    data.get("active_lease") is not None
                    or data.get("last_completed_lease") != lease.lease_id
                    or data["outcome"] not in _TERMINAL
                ):
                    await self._quarantine(slot, "handler_return_not_attested")
                    return
                else:
                    self._profile(lease, "handler_return_verified_monotonic_ns")
                    async with self._condition:
                        if slot.state == "DRAINING" and slot.lease_id == lease.lease_id and not self._closed:
                            slot.state, slot.lease_id = "IDLE", None
                            self._condition.notify_all()
                    return
                await asyncio.sleep(min(self._poll_interval, remaining))
        except asyncio.CancelledError:
            await self._quarantine(slot, "pool_shutdown_during_drain")
            raise
        except Exception:
            await self._quarantine(slot, "handler_drain_unverified")
        finally:
            profile = self._profiles.pop(lease.lease_id, None)
            if profile is not None and self._telemetry is not None:
                from .pool_telemetry import lease_digest

                with contextlib.suppress(Exception):
                    self._telemetry.submit(
                        "proxy",
                        lease_digest(lease.lease_id),
                        **profile,
                        profile_finished_monotonic_ns=time.monotonic_ns(),
                        outcome=slot.state,
                        reusable=slot.state == "IDLE",
                    )

    async def stateless_get(self, path, incoming_headers=None):
        """Delegate only native preflight GETs without acquiring a stream lease."""
        if path not in {"/health", "/api/deployment"}:
            raise ValueError("Unsupported public offline preflight path")
        async with self._condition:
            if self._closed:
                return None
            candidates = sorted(
                (slot for slot in self._slots.values() if slot.state in {"IDLE", "LEASED", "DRAINING"}),
                key=lambda slot: slot.state != "IDLE",
            )
        for slot in candidates:
            async with self._condition:
                if self._closed or slot.state not in {"IDLE", "LEASED", "DRAINING"}:
                    continue
                snapshot = (slot.state, slot.lease_id, slot.total_leases)
            try:
                data = await self._worker_status(slot.backend)
                if (data["busy"] or data.get("active_lease") is not None) and await self._quarantine_unchanged_idle(
                    slot, snapshot, "unexpected_busy_worker"
                ):
                    continue
                response = await asyncio.wait_for(
                    self._http.get(slot.backend.base_url + path, headers=_headers(incoming_headers or {}, self._token)),
                    timeout=self._http_timeout,
                )
                if len(response.content) > self.max_message_bytes:
                    continue
                return response
            except Exception:
                # An active slot remains reserved; readiness cannot return it
                # to IDLE or use this stateless request as handler-return proof.
                await self._quarantine_unchanged_idle(slot, snapshot, "preflight_identity_or_transport_unverified")
        return None

    async def status(self):
        """Return safe reservation scalars, never tokens, URLs, leases or payloads."""
        async with self._condition:
            counts = {
                state.lower(): sum(slot.state == state for slot in self._slots.values())
                for state in ("UNVERIFIED", "IDLE", "LEASED", "DRAINING", "QUARANTINED")
            }
            return {
                "closed": self._closed,
                "capacity": len(self._slots),
                **counts,
                "pending_admissions": self._pending_admissions,
                "held_leases": sum(slot.lease_id is not None for slot in self._slots.values()),
                "peak_reserved_streams": self._peak_reserved,
                "workers": [
                    {
                        "worker_id": slot.backend.worker_id,
                        "pid": slot.backend.pid,
                        "state": slot.state,
                        "max_reserved_streams": slot.max_reserved_streams,
                        "total_leases": slot.total_leases,
                        "quarantine_reason": slot.quarantine_reason,
                    }
                    for slot in self._slots.values()
                ],
            }

    async def aclose(self):
        """Stop owned drain tasks before the supervisor terminates its children."""
        async with self._condition:
            self._closed = True
            for slot in self._slots.values():
                if slot.state in {"LEASED", "DRAINING"}:
                    slot.state, slot.quarantine_reason = "QUARANTINED", "pool_shutdown_during_drain"
            tasks = tuple(self._drainers)
            self._condition.notify_all()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._owns_http:
            await self._http.aclose()
        if self._telemetry is not None:
            await self._telemetry.aclose()


def _close_code(code):
    return (
        code
        if type(code) is int
        and (code in {1000, 1001, 1002, 1003, 1007, 1008, 1009, 1011, 1012, 1013, 1014} or 3000 <= code <= 4999)
        else 1011
    )


async def _relay(websocket, backend, max_bytes, observation=None):
    async def forward(operation, direction):
        if observation is None:
            return await operation
        started = time.monotonic_ns()
        name = direction + "_send_"
        observation[name + "attempts"] = observation.get(name + "attempts", 0) + 1
        try:
            return await operation
        except BaseException:
            observation[name + "failures"] = observation.get(name + "failures", 0) + 1
            raise
        finally:
            elapsed = time.monotonic_ns() - started
            observation[name + "wall_ns_sum"] = observation.get(name + "wall_ns_sum", 0) + elapsed
            observation[name + "wall_ns_max"] = max(observation.get(name + "wall_ns_max", 0), elapsed)

    async def upstream():
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                raise _StreamEnded(message.get("code", 1000))
            if message.get("type") != "websocket.receive":
                raise _StreamEnded(1002)
            if message.get("text") is not None and message.get("bytes") is None:
                value = message["text"]
                size = len(value.encode("utf-8"))
            elif message.get("bytes") is not None and message.get("text") is None:
                value, size = message["bytes"], len(message["bytes"])
            else:
                raise _StreamEnded(1002)
            if size > max_bytes:
                raise _StreamEnded(1009)
            await forward(backend.send(value), "upstream")

    async def downstream():
        while True:
            try:
                value = await backend.recv()
            except ConnectionClosed as error:
                raise _StreamEnded(error.rcvd.code if error.rcvd is not None else 1011) from None
            if isinstance(value, str):
                if len(value.encode("utf-8")) > max_bytes:
                    raise _StreamEnded(1009)
                await forward(websocket.send_text(value), "downstream")
            elif isinstance(value, bytes):
                if len(value) > max_bytes:
                    raise _StreamEnded(1009)
                await forward(websocket.send_bytes(value), "downstream")
            else:
                raise _StreamEnded(1002)

    tasks = [asyncio.create_task(upstream()), asyncio.create_task(downstream())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def create_pool_proxy(pool=None, *, lifespan=None):
    """Expose only offline WebSocket and the native client's stateless GETs."""
    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    if pool is not None:
        app.state.worker_pool = pool

    @app.get("/health")
    async def health(request: Request):
        selected = getattr(app.state, "worker_pool", None)
        if selected is None:
            return JSONResponse({"status": "unavailable"}, status_code=503)
        response = await selected.stateless_get("/health", request.headers)
        return JSONResponse(
            {
                "status": "ok" if response is not None and response.status_code == 200 else "unavailable",
                "worker_pool": await selected.status(),
            },
            status_code=200 if response is not None and response.status_code == 200 else 503,
        )

    @app.get("/api/deployment")
    async def deployment(request: Request):
        selected = getattr(app.state, "worker_pool", None)
        response = None if selected is None else await selected.stateless_get("/api/deployment", request.headers)
        if response is None:
            return JSONResponse({"error": "offline_worker_unavailable"}, status_code=503)
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
        )

    @app.websocket("/api/ws/offline")
    async def offline(websocket: WebSocket):
        if websocket.scope.get("query_string"):
            await websocket.accept()
            await websocket.close(code=1008)
            return
        selected = getattr(app.state, "worker_pool", None)
        lease = None if selected is None else await selected.acquire()
        if lease is None:
            await websocket.accept()
            await websocket.close(code=1013)
            return
        backend = None
        code = 1000
        try:
            await websocket.accept()
            backend = await selected.open_backend(lease, websocket.headers)
            await _relay(
                websocket,
                backend,
                selected.max_message_bytes,
                getattr(selected, "_profiles", {}).get(getattr(lease, "lease_id", None)),
            )
        except _StreamEnded as error:
            code = _close_code(error.code)
        except asyncio.CancelledError:
            code = 1012
            raise
        except Exception:
            code = 1011
        finally:
            await selected.finish(lease, backend)
            with contextlib.suppress(Exception):
                await websocket.close(code=code)

    return app
