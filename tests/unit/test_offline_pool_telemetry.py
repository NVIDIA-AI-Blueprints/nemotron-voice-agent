# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D102

"""Bounded optional observations preserve wire and handler lifecycle truth."""

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from examples.generic.offline.pool_proxy import BackendDescriptor, ExclusiveWorkerPool
from examples.generic.offline.pool_telemetry import PoolTelemetry, lease_digest, valid_document
from examples.generic.offline.worker import LEASE_HEADER, TOKEN_HEADER, OfflineWorkerApp


class TelemetryTests(unittest.IsolatedAsyncioTestCase):
    """Persistence is optional, bounded, private and independent of payloads."""

    async def test_worker_sidecar_joins_trusted_scope_without_persisting_token_or_pcm(self):
        with tempfile.TemporaryDirectory() as folder:
            observer = PoolTelemetry(Path(folder))
            lease = "a" * 32
            token = "s" * 40
            seen = []

            async def native(scope, receive, send):
                holder = scope["nva.offline.pool"]
                holder["server_metrics_session_id"] = "b" * 32
                seen.append(scope)

            app = OfflineWorkerApp(native, token=token, telemetry=observer)
            await app(
                {
                    "type": "websocket",
                    "path": "/api/ws/offline",
                    "headers": [
                        (TOKEN_HEADER.encode(), token.encode()),
                        (LEASE_HEADER.encode(), lease.encode()),
                    ],
                },
                AsyncMock(),
                AsyncMock(),
            )
            await observer.aclose()
            path = Path(folder) / f"{lease_digest(lease)}.worker.json"
            raw = path.read_text()
            document = json.loads(raw)
            self.assertTrue(valid_document(document))
            self.assertEqual(document["server_metrics_session_id"], "b" * 32)
            self.assertEqual(document["worker_pid"], os.getpid())
            self.assertLessEqual(document["handler_started_monotonic_ns"], document["native_app_exit_monotonic_ns"])
            self.assertLessEqual(document["native_app_exit_monotonic_ns"], document["lease_released_monotonic_ns"])
            self.assertNotIn(token, raw)
            self.assertNotIn(lease, raw)
            self.assertEqual(seen[0]["headers"], [])
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertFalse(app.status()["busy"])

    async def test_capacity_drops_observation_without_waiting_for_disk(self):
        with tempfile.TemporaryDirectory() as folder:
            observer = PoolTelemetry(Path(folder), max_pending=1)
            blocked = asyncio.Event()

            async def persist(_):
                await blocked.wait()

            with patch.object(observer, "_persist", persist):
                observer.submit("worker", "a" * 64, outcome="returned")
                observer.submit("worker", "b" * 64, outcome="returned")
                self.assertEqual(observer.status()["dropped"], 1)
                self.assertEqual(len(observer.tasks), 1)
                blocked.set()
                await observer.aclose()

    async def test_private_fields_are_rejected_and_io_failure_is_observational(self):
        with tempfile.TemporaryDirectory() as folder:
            observer = PoolTelemetry(Path(folder))
            observer.submit("worker", "a" * 64, request_messages=["private"])
            self.assertEqual(observer.status()["rejected"], 1)
            self.assertEqual(list(Path(folder).iterdir()), [])
            with patch.object(observer, "_write", side_effect=OSError("controlled")):
                observer.submit("worker", "b" * 64, outcome="returned")
                await observer.aclose()
            self.assertEqual(observer.status()["rejected"], 2)

    async def test_proxy_observations_end_after_exact_handler_release(self):
        with tempfile.TemporaryDirectory() as folder:
            observer = PoolTelemetry(Path(folder))
            state = {"pid": 12345, "busy": False, "active_lease": None, "last_completed_lease": None, "outcome": "idle"}

            def request(_):
                return httpx.Response(200, json=dict(state))

            async def close():
                state.update(busy=False, active_lease=None, last_completed_lease=lease.lease_id, outcome="returned")

            socket = AsyncMock()
            socket.close.side_effect = close
            client = httpx.AsyncClient(transport=httpx.MockTransport(request))
            pool = ExclusiveWorkerPool(
                [BackendDescriptor("worker-01", "http://127.0.0.1:12345", 12345)],
                token="s" * 40,
                http_client=client,
                connect=AsyncMock(return_value=socket),
            )
            pool._telemetry = observer
            try:
                await pool.initialize()
                lease = await pool.acquire()
                await pool.open_backend(lease, {})
                state.update(busy=True, active_lease=lease.lease_id, outcome="running")
                await pool.finish(lease, socket)
                async with asyncio.timeout(1):
                    while (await pool.status())["idle"] != 1:
                        await asyncio.sleep(0)
                await pool.aclose()
                document = json.loads((Path(folder) / f"{lease_digest(lease.lease_id)}.proxy.json").read_text())
                self.assertTrue(valid_document(document))
                self.assertTrue(document["reusable"])
                self.assertGreaterEqual(
                    document["handler_return_verified_monotonic_ns"], document["backend_close_finished_monotonic_ns"]
                )
                self.assertGreaterEqual(
                    document["profile_finished_monotonic_ns"], document["handler_return_verified_monotonic_ns"]
                )
            finally:
                await pool.aclose()
                await client.aclose()

    async def test_observer_exception_does_not_replace_native_handler_outcome(self):
        async def native(scope, receive, send):
            raise ValueError("native failure")

        observer = AsyncMock()
        observer.submit = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("observer failure"))
        app = OfflineWorkerApp(native, token="s" * 40, telemetry=observer)
        with self.assertRaisesRegex(ValueError, "native failure"):
            await app(
                {
                    "type": "websocket",
                    "path": "/api/ws/offline",
                    "headers": [(TOKEN_HEADER.encode(), b"s" * 40), (LEASE_HEADER.encode(), b"a" * 32)],
                },
                AsyncMock(),
                AsyncMock(),
            )
        self.assertEqual(app.status()["outcome"], "error")
