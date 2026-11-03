# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D107
"""Model-free admission, transparent frames and authenticated handler drain."""

import asyncio
import json
import time
import unittest
from urllib.parse import urlsplit

import httpx

from examples.generic.offline.pool_proxy import (
    LEASE_HEADER,
    TOKEN_HEADER,
    BackendDescriptor,
    ExclusiveWorkerPool,
    create_pool_proxy,
)

TOKEN = "trusted-test-token"


class BackendSocket:
    def __init__(self, http, backend, lease, close_delay=0):
        self.http, self.backend, self.lease = http, backend, lease
        self.received = asyncio.Queue()
        self.sent = []
        self.send_entered = asyncio.Event()
        self.allow_send = asyncio.Event()
        self.allow_send.set()
        self.close_delay = close_delay
        self.closed = False

    async def send(self, value):
        self.send_entered.set()
        await self.allow_send.wait()
        self.sent.append(value)

    async def recv(self):
        value = await self.received.get()
        if isinstance(value, BaseException):
            raise value
        return value

    async def close(self):
        if self.close_delay:
            await asyncio.sleep(self.close_delay)
        self.closed = True
        self.http.complete(self.backend, self.lease)


class HTTP:
    def __init__(self, backends):
        self.backends = {urlsplit(b.base_url).port: b for b in backends}
        self.states = {
            b.worker_id: {
                "pid": b.pid,
                "busy": False,
                "active_lease": None,
                "last_completed_lease": None,
                "outcome": "idle",
            }
            for b in backends
        }
        self.calls = []
        self.errors = set()
        self.maximum_busy = {b.worker_id: 0 for b in backends}

    async def get(self, url, *, headers):
        parsed = urlsplit(url)
        backend = self.backends[parsed.port]
        self.calls.append((backend.worker_id, parsed.path, dict(headers)))
        assert headers[TOKEN_HEADER] == TOKEN
        if backend.worker_id in self.errors:
            raise OSError("Synthetic unavailable worker")
        await asyncio.sleep(0)
        if parsed.path == "/__offline_pool/status":
            return httpx.Response(200, json=self.states[backend.worker_id])
        if parsed.path == "/api/deployment":
            return httpx.Response(
                200, content=b'{"active":{"key":"generic-assistant"}}', headers={"content-type": "application/json"}
            )
        return httpx.Response(200, json={"status": "ok"})

    def start(self, backend, lease):
        current = self.states[backend.worker_id]
        assert current["busy"] is False
        current.update(busy=True, active_lease=lease, outcome="running")
        self.maximum_busy[backend.worker_id] = 1

    def complete(self, backend, lease):
        self.states[backend.worker_id].update(
            busy=False, active_lease=None, last_completed_lease=lease, outcome="returned"
        )


class Connector:
    def __init__(self, http, *, close_delay=0):
        self.http, self.close_delay = http, close_delay
        self.calls = []
        self.sockets = []

    async def __call__(self, url, **kwargs):
        backend = self.http.backends[urlsplit(url).port]
        headers = kwargs["additional_headers"]
        assert headers[TOKEN_HEADER] == TOKEN
        lease = headers[LEASE_HEADER]
        self.http.start(backend, lease)
        socket = BackendSocket(self.http, backend, lease, self.close_delay)
        self.calls.append((url, kwargs))
        self.sockets.append(socket)
        return socket


class Stream:
    def __init__(self, app, *, path="/api/ws/offline", headers=(), query_string=b""):
        self.input = asyncio.Queue()
        self.output = asyncio.Queue()
        self.input.put_nowait({"type": "websocket.connect"})
        self.receive_calls = 0
        self.task = asyncio.create_task(
            app(
                {
                    "type": "websocket",
                    "asgi": {"version": "3.0"},
                    "scheme": "ws",
                    "path": path,
                    "raw_path": path.encode(),
                    "query_string": query_string,
                    "headers": headers,
                    "subprotocols": [],
                    "server": ("proxy", 80),
                    "client": ("client", 1),
                },
                self.receive,
                self.output.put,
            )
        )

    async def receive(self):
        self.receive_calls += 1
        return await self.input.get()

    def text(self, value):
        self.input.put_nowait({"type": "websocket.receive", "text": value})

    def binary(self, value):
        self.input.put_nowait({"type": "websocket.receive", "bytes": value})

    def disconnect(self):
        self.input.put_nowait({"type": "websocket.disconnect", "code": 1000})

    async def next(self):
        return await asyncio.wait_for(self.output.get(), 2)

    async def stop(self):
        if not self.task.done():
            self.disconnect()
        await asyncio.wait_for(self.task, 2)


class PoolProxyTests(unittest.IsolatedAsyncioTestCase):
    async def pool(self, count=2, *, close_delay=0, **kwargs):
        backends = [BackendDescriptor(f"w{i}", f"http://127.0.0.1:{19000 + i}", 10000 + i) for i in range(count)]
        http = HTTP(backends)
        connector = Connector(http, close_delay=close_delay)
        pool = ExclusiveWorkerPool(
            backends, token=TOKEN, http_client=http, connect=connector, poll_interval_s=0.005, **kwargs
        )
        await pool.initialize()
        self.addAsyncCleanup(pool.aclose)
        return pool, http, connector

    async def until(self, condition):
        async with asyncio.timeout(2):
            while not await condition():
                await asyncio.sleep(0.001)

    async def test_ten_reservations_are_distinct_and_eleventh_active_request_fails_fast(self):
        pool, _, _ = await self.pool(10)
        leases = await asyncio.gather(*(pool.acquire() for _ in range(10)))
        self.assertEqual(len({x.backend.pid for x in leases}), 10)
        before = time.monotonic()
        self.assertIsNone(await pool.acquire())
        self.assertLess(time.monotonic() - before, 0.05)
        stream = Stream(create_pool_proxy(pool))
        self.assertEqual((await stream.next())["type"], "websocket.accept")
        self.assertEqual((await stream.next())["code"], 1013)
        await stream.stop()
        status = await pool.status()
        self.assertEqual(status["leased"], 10)
        self.assertTrue(all(x["max_reserved_streams"] == 1 for x in status["workers"]))

    async def test_socket_close_does_not_release_until_exact_handler_return(self):
        pool, http, _ = await self.pool(1, admission_timeout_s=0.02)
        lease = await pool.acquire()
        http.start(lease.backend, lease.lease_id)
        await pool.finish(lease)
        self.assertEqual((await pool.status())["draining"], 1)
        self.assertIsNone(await pool.acquire())
        http.complete(lease.backend, lease.lease_id)

        async def idle():
            return (await pool.status())["idle"] == 1

        await self.until(idle)
        second = await pool.acquire()
        self.assertEqual(second.backend.pid, lease.backend.pid)
        self.assertNotEqual(second.lease_id, lease.lease_id)
        self.assertEqual((await pool.status())["workers"][0]["max_reserved_streams"], 1)

    async def test_six_closing_probes_and_six_connects_wait_for_proven_quick_drains(self):
        pool, http, connector = await self.pool(10, close_delay=0.025)
        app = create_pool_proxy(pool)
        probes = [Stream(app) for _ in range(6)]
        for stream in probes:
            self.addAsyncCleanup(stream.stop)
        await asyncio.gather(*(x.next() for x in probes))

        async def six():
            return len(connector.sockets) == 6

        await self.until(six)
        for stream in probes:
            stream.disconnect()
        await asyncio.gather(*(x.task for x in probes))
        self.assertEqual((await pool.status())["draining"], 6)
        actual = [Stream(app) for _ in range(6)]
        for stream in actual:
            self.addAsyncCleanup(stream.stop)
        accepted = await asyncio.gather(*(x.next() for x in actual))
        self.assertTrue(all(x["type"] == "websocket.accept" for x in accepted))

        async def twelve():
            return len(connector.sockets) == 12

        await self.until(twelve)
        self.assertEqual((await pool.status())["leased"], 6)
        self.assertTrue(all(value <= 1 for value in http.maximum_busy.values()))
        await asyncio.gather(*(x.stop() for x in actual))

    async def test_long_drain_bounds_pending_admission_then_quarantines(self):
        pool, http, _ = await self.pool(1, admission_timeout_s=0.02, drain_timeout_s=0.07)
        lease = await pool.acquire()
        http.start(lease.backend, lease.lease_id)
        await pool.finish(lease)
        first = asyncio.create_task(pool.acquire())
        await asyncio.sleep(0)
        before = time.monotonic()
        self.assertIsNone(await pool.acquire())
        self.assertLess(time.monotonic() - before, 0.02)
        self.assertIsNone(await first)

        async def quarantined():
            return (await pool.status())["quarantined"] == 1

        await self.until(quarantined)
        self.assertIsNone(await pool.acquire())
        self.assertEqual((await pool.status())["workers"][0]["quarantine_reason"], "handler_drain_timeout")

    async def test_wrong_pid_lease_or_status_error_never_reuses_worker(self):
        for case in ("pid", "lease", "last_lease", "unavailable"):
            pool, http, _ = await self.pool(1)
            lease = await pool.acquire()
            http.start(lease.backend, lease.lease_id)
            current = http.states["w0"]
            if case == "pid":
                current["pid"] += 1
            elif case == "lease":
                current["active_lease"] = "f" * 32
            elif case == "last_lease":
                http.complete(lease.backend, "e" * 32)
            else:
                http.errors.add("w0")
            await pool.finish(lease)

            async def bad(pool=pool):
                return (await pool.status())["quarantined"] == 1

            await self.until(bad)
            with self.subTest(case=case):
                self.assertIsNone(await pool.acquire())

    async def test_bidirectional_frames_order_authorization_and_backpressure(self):
        pool, http, connector = await self.pool(1)
        stream = Stream(
            create_pool_proxy(pool),
            headers=[
                (b"authorization", b"Bearer native-user"),
                (b"x-nva-offline-pool-token", b"spoof"),
                (b"x-nva-offline-pool-lease", b"spoof"),
                (b"sec-websocket-key", b"ignored"),
            ],
        )
        self.assertEqual((await stream.next())["type"], "websocket.accept")

        async def opened():
            return len(connector.sockets) == 1

        await self.until(opened)
        backend = connector.sockets[0]
        headers = connector.calls[0][1]["additional_headers"]
        self.assertEqual(headers["authorization"], "Bearer native-user")
        self.assertEqual(headers[TOKEN_HEADER], TOKEN)
        self.assertRegex(headers[LEASE_HEADER], r"^[0-9a-f]{32}$")
        self.assertNotIn("sec-websocket-key", headers)
        self.assertEqual(connector.calls[0][1]["max_queue"], 1)
        self.assertEqual(connector.calls[0][1]["max_size"], 8 * 1024 * 1024)
        backend.allow_send.clear()
        stream.text("opaque🙂{not json")
        stream.binary(b"\0\xff")
        await asyncio.wait_for(backend.send_entered.wait(), 1)
        await asyncio.sleep(0)
        self.assertEqual(stream.receive_calls, 2)  # handshake + one blocked frame
        backend.allow_send.set()

        async def sent():
            return len(backend.sent) == 2

        await self.until(sent)
        self.assertEqual(backend.sent, ["opaque🙂{not json", b"\0\xff"])
        backend.received.put_nowait("first")
        backend.received.put_nowait(b"\x05\0")
        backend.received.put_nowait("last")
        messages = [await stream.next() for _ in range(3)]
        self.assertEqual(
            messages,
            [
                {"type": "websocket.send", "text": "first"},
                {"type": "websocket.send", "bytes": b"\x05\0"},
                {"type": "websocket.send", "text": "last"},
            ],
        )
        await stream.stop()
        safe = json.dumps(await pool.status())
        self.assertNotIn(TOKEN, safe)
        self.assertNotIn(headers[LEASE_HEADER], safe)
        self.assertNotIn("native-user", safe)

    async def test_oversize_frame_is_rejected_without_forwarding(self):
        pool, _, connector = await self.pool(1, max_message_bytes=8)
        stream = Stream(create_pool_proxy(pool))
        await stream.next()
        stream.binary(b"x" * 9)
        self.assertEqual((await stream.next())["code"], 1009)
        await stream.stop()
        self.assertEqual(connector.sockets[0].sent, [])

    async def test_public_http_is_allowlisted_and_does_not_borrow_slots(self):
        pool, http, _ = await self.pool(1)
        lease = await pool.acquire()
        http.start(lease.backend, lease.lease_id)
        app = create_pool_proxy(pool)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
            response = await client.get("/api/deployment", headers={"Authorization": "Bearer native-user"})
            self.assertEqual(response.content, b'{"active":{"key":"generic-assistant"}}')
            response = await client.get("/health")
            self.assertEqual(response.status_code, 200)
            before = len(http.calls)
            for path in ("/api/session-config", "/__offline_pool/status", "/api/ws", "/docs", "/openapi.json"):
                self.assertEqual((await client.get(path)).status_code, 404)
            self.assertEqual((await client.post("/api/deployment")).status_code, 405)
            self.assertEqual((await client.post("/api/session-config")).status_code, 404)
            self.assertEqual(len(http.calls), before)
        self.assertEqual((await pool.status())["leased"], 1)
        forwarded = [headers for _, path, headers in http.calls if path == "/api/deployment"]
        self.assertEqual(forwarded[0]["authorization"], "Bearer native-user")

    async def test_shutdown_cancels_drainers_and_wakes_pending_admission(self):
        pool, http, _ = await self.pool(1)
        lease = await pool.acquire()
        http.start(lease.backend, lease.lease_id)
        await pool.finish(lease)
        waiting = asyncio.create_task(pool.acquire())
        await asyncio.sleep(0)
        await pool.aclose()
        self.assertIsNone(await waiting)
        self.assertIsNone(await pool.acquire())
        self.assertEqual((await pool.status())["pending_admissions"], 0)
        self.assertEqual((await pool.status())["quarantined"], 1)

    async def test_shutdown_before_drainer_first_runs_keeps_slot_unusable(self):
        pool, http, _ = await self.pool(1)
        lease = await pool.acquire()
        http.start(lease.backend, lease.lease_id)
        await pool.finish(lease)
        # No scheduling yield occurs before closing the pool.
        await pool.aclose()
        status = await pool.status()
        self.assertEqual(status["quarantined"], 1)
        self.assertEqual(status["held_leases"], 1)
        self.assertIsNone(await pool.acquire())

    async def test_failed_backend_connect_has_no_midstream_failover(self):
        pool, _, connector = await self.pool(2)
        calls = []

        async def failed(url, **kwargs):
            calls.append(url)
            raise OSError("synthetic private failure")

        pool._connect = failed
        stream = Stream(create_pool_proxy(pool))
        self.assertEqual((await stream.next())["type"], "websocket.accept")
        self.assertEqual((await stream.next())["code"], 1011)
        await stream.stop()

        async def quarantined():
            return (await pool.status())["quarantined"] == 1

        await self.until(quarantined)
        self.assertEqual(len(calls), 1)
        self.assertFalse(connector.calls)
        self.assertEqual((await pool.status())["idle"], 1)

    async def test_front_close_does_not_wait_for_backend_cleanup(self):
        pool, _, connector = await self.pool(1, close_delay=10)
        stream = Stream(create_pool_proxy(pool))
        await stream.next()

        async def opened():
            return bool(connector.sockets)

        await self.until(opened)
        started = time.monotonic()
        await stream.stop()
        self.assertLess(time.monotonic() - started, 0.2)
        self.assertEqual((await pool.status())["draining"], 1)
        self.assertFalse(connector.sockets[0].closed)

    async def test_missing_status_fields_are_not_an_idle_attestation(self):
        descriptors = [BackendDescriptor("w0", "http://127.0.0.1:19000", 10000)]
        http = HTTP(descriptors)
        http.states["w0"].pop("last_completed_lease")
        pool = ExclusiveWorkerPool(descriptors, token=TOKEN, http_client=http)
        self.addAsyncCleanup(pool.aclose)
        status = await pool.initialize()
        self.assertEqual(status["quarantined"], 1)
        self.assertIsNone(await pool.acquire())

    async def test_stale_stateless_busy_snapshot_cannot_quarantine_a_completed_drain(self):
        pool, http, _ = await self.pool(1)
        lease = await pool.acquire()
        http.start(lease.backend, lease.lease_id)
        entered, release = asyncio.Event(), asyncio.Event()
        original = http.get
        first = True

        async def delayed(url, *, headers):
            nonlocal first
            if first and url.endswith("/__offline_pool/status"):
                first = False
                stale = dict(http.states["w0"])
                entered.set()
                await release.wait()
                return httpx.Response(200, json=stale)
            return await original(url, headers=headers)

        http.get = delayed
        request = asyncio.create_task(pool.stateless_get("/api/deployment"))
        await asyncio.wait_for(entered.wait(), 1)
        http.complete(lease.backend, lease.lease_id)
        await pool.finish(lease)

        async def idle():
            return (await pool.status())["idle"] == 1

        await self.until(idle)
        release.set()
        self.assertEqual((await request).status_code, 200)
        self.assertEqual((await pool.status())["idle"], 1)
        self.assertEqual((await pool.status())["quarantined"], 0)

    async def test_query_session_configuration_is_rejected_before_leasing(self):
        pool, _, connector = await self.pool(1)
        stream = Stream(create_pool_proxy(pool), query_string=b"session_id=old-local-config")
        self.assertEqual((await stream.next())["type"], "websocket.accept")
        self.assertEqual((await stream.next())["code"], 1008)
        await stream.stop()
        self.assertFalse(connector.calls)
        status = await pool.status()
        self.assertEqual(status["idle"], 1)
        self.assertEqual(status["workers"][0]["total_leases"], 0)

    async def test_fresh_pid_or_busy_mismatch_blocks_backend_connect_without_replay(self):
        for case in ("pid", "busy"):
            pool, http, connector = await self.pool(2)
            lease = await pool.acquire()
            if case == "pid":
                http.states[lease.backend.worker_id]["pid"] += 1
            else:
                http.start(lease.backend, "e" * 32)
            with self.subTest(case=case), self.assertRaises(RuntimeError):
                await pool.open_backend(lease, {"Authorization": "Bearer native-user"})
            self.assertFalse(connector.calls)
            await pool.finish(lease)
            await asyncio.sleep(0)
            status = await pool.status()
            self.assertEqual(status["quarantined"], 1)
            self.assertEqual(status["idle"], 1)
            self.assertEqual(status["held_leases"], 1)

    async def test_shutdown_during_fresh_identity_check_cannot_open_backend(self):
        pool, http, connector = await self.pool(1)
        lease = await pool.acquire()
        original = http.get
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed(url, *, headers):
            entered.set()
            await release.wait()
            return await original(url, headers=headers)

        http.get = delayed
        opening = asyncio.create_task(pool.open_backend(lease, {}))
        await asyncio.wait_for(entered.wait(), 1)
        await pool.aclose()
        release.set()
        with self.assertRaises(RuntimeError):
            await opening
        self.assertFalse(connector.calls)
        self.assertEqual((await pool.status())["quarantined"], 1)


if __name__ == "__main__":
    unittest.main()
