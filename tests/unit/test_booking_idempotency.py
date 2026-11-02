# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from examples.frontend_backend_agent.airline.backend import HTTPBookingBackend
from examples.frontend_backend_agent.airline.database.api import BookingAPI, IdempotencyConflict
from examples.frontend_backend_agent.airline.database.db import apply_schema, connect
from examples.frontend_backend_agent.airline.database.server import app

_REQUEST = {
    "passenger": "Ada Lovelace",
    "origin": "JFK",
    "destination": "LAX",
    "flight_number": "AA124",
}
_FLIGHT = {
    "flight_id": "AA124",
    "origin_airport": "JFK",
    "dest_airport": "LAX",
    "origin_city": "New York",
    "dest_city": "Los Angeles",
    "date": "2026-11-10",
    "departure_time": "2026-11-10T10:00:00",
}


class BookingIdempotencyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "bookings.db"
        self.enterContext(patch("examples.frontend_backend_agent.airline.database.db.db_path", return_value=self.path))
        with closing(connect()) as conn, conn:
            apply_schema(conn)
            conn.execute(
                "INSERT INTO flights (flight_number, origin, destination, departure, arrival, cabin) "
                "VALUES ('AA124', 'JFK', 'LAX', '2026-11-10T10:00:00', '2026-11-10T16:00:00', 'economy')"
            )

    def create(self, **kwargs):
        with closing(connect()) as conn:
            return BookingAPI(conn).create_booking(**{**_REQUEST, **kwargs})

    def assert_counts(self, count):
        with closing(connect()) as conn:
            for table in ("pnrs", "ancillaries", "confirmation_codes", "activity_log"):
                self.assertEqual(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], count, table)

    def test_replay_survives_new_connections_and_schema_reapplication(self):
        first = self.create(idempotency_key="retry")
        with closing(connect()) as conn:
            apply_schema(conn)
        self.assertEqual(first, self.create(idempotency_key="retry"))
        self.assert_counts(1)

    def test_different_keys_allow_intentional_duplicate_bookings(self):
        first = self.create(idempotency_key="first")
        second = self.create(idempotency_key="second")
        self.assertNotEqual(first["pnr"], second["pnr"])
        self.assert_counts(2)

    def test_without_key_preserves_existing_create_semantics(self):
        self.assertNotEqual(self.create()["pnr"], self.create()["pnr"])
        self.assert_counts(2)

    def test_key_reuse_with_changed_parameters_conflicts_without_writes(self):
        first = self.create(idempotency_key="retry")
        for changes in (
            {"passenger": "Grace Hopper"},
            {"flight_number": "AA999"},
            {"departure": "2026-11-11T10:00:00"},
            {"seat": "14A"},
            {"meal": "vegetarian"},
            {"cabin": "business"},
            {"fare_basis": "refundable"},
            {"session_id": "another-session"},
        ):
            with self.subTest(changes=changes), self.assertRaises(IdempotencyConflict):
                self.create(idempotency_key="retry", **changes)
        self.assertEqual(first, self.create(idempotency_key="retry"))
        self.assert_counts(1)

    def test_concurrent_requests_on_independent_connections_commit_once(self):
        barrier = threading.Barrier(8)

        def create():
            barrier.wait(timeout=5)
            return self.create(idempotency_key="concurrent")

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: create(), range(8)))
        self.assertTrue(all(result == results[0] for result in results))
        self.assert_counts(1)

    def test_failure_rolls_back_booking_and_key_together(self):
        with (
            patch.object(BookingAPI, "_log_activity", side_effect=RuntimeError("lost transaction")),
            self.assertRaises(RuntimeError),
        ):
            self.create(idempotency_key="retry")
        self.assert_counts(0)
        with closing(connect()) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM booking_requests").fetchone()[0], 0)
        self.assertIsNotNone(self.create(idempotency_key="retry"))
        self.assert_counts(1)

    def test_missing_flight_does_not_reserve_key(self):
        self.assertIsNone(self.create(idempotency_key="retry", flight_number="AA999"))
        self.assertIsNotNone(self.create(idempotency_key="retry"))
        self.assert_counts(1)

    def test_replay_does_not_reactivate_a_cancelled_booking(self):
        first = self.create(idempotency_key="retry")
        with closing(connect()) as conn, conn:
            conn.execute("UPDATE pnrs SET status = 'cancelled' WHERE pnr = ?", (first["pnr"],))
        self.assertEqual(first, self.create(idempotency_key="retry"))
        with closing(connect()) as conn:
            self.assertEqual(BookingAPI(conn).get_pnr(first["pnr"])["booking_status"], "cancelled")
        self.assert_counts(1)

    def booking_client(self):
        client = TestClient(app)
        self.addCleanup(client.close)
        return client

    def test_http_replay_conflict_and_key_validation(self):
        client = self.booking_client()
        headers = {"Idempotency-Key": "http-retry"}
        first = client.post("/pnrs", json=_REQUEST, headers=headers)
        second = client.post("/pnrs", json=_REQUEST, headers=headers)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json(), second.json())
        conflict = client.post("/pnrs", json={**_REQUEST, "seat": "14A"}, headers=headers)
        self.assertEqual(conflict.status_code, 409)
        for key in ("", "x" * 129):
            self.assertEqual(client.post("/pnrs", json=_REQUEST, headers={"Idempotency-Key": key}).status_code, 422)
        self.assert_counts(1)

    async def test_http_backend_retry_after_server_commit_and_lost_response(self):
        client = self.booking_client()
        committed = []

        def handle(request):
            response = client.post(
                "/pnrs",
                content=request.content,
                headers={"Idempotency-Key": request.headers["Idempotency-Key"], "Content-Type": "application/json"},
            )
            self.assertEqual(response.status_code, 200)
            committed.append(response.json())
            if len(committed) == 1:
                raise httpx.ReadTimeout("response lost after server commit", request=request)
            return httpx.Response(response.status_code, json=response.json())

        factory = partial(httpx.AsyncClient, transport=httpx.MockTransport(handle))
        with patch("examples.frontend_backend_agent.airline.backend.httpx.AsyncClient", factory):
            backend = HTTPBookingBackend("http://booking.test")
            kwargs = {"passenger_name": "Ada", "flight": _FLIGHT, "idempotency_key": "lost-response"}
            with self.assertRaises(httpx.ReadTimeout):
                await backend.create_booking(**kwargs)
            result = await backend.create_booking(**kwargs)
        self.assertEqual(committed[0], committed[1])
        self.assertEqual(result["pnr"], committed[0]["pnr"])
        self.assertEqual(result["confirmation_code"], committed[0]["confirmation_code"])
        self.assert_counts(1)


class HTTPBookingBackendIdempotencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_backend_forwards_stable_key_and_request_body(self):
        requests = []

        def handle(request):
            requests.append(request)
            return httpx.Response(200, json={"pnr": "ABC123", "confirmation_code": "BOOK1234"})

        factory = partial(httpx.AsyncClient, transport=httpx.MockTransport(handle))
        with patch("examples.frontend_backend_agent.airline.backend.httpx.AsyncClient", factory):
            backend = HTTPBookingBackend("http://booking.test")
            for _ in range(2):
                result = await backend.create_booking(passenger_name="Ada", flight=_FLIGHT, idempotency_key="same-key")
                self.assertEqual(result["pnr"], "ABC123")
        self.assertEqual([request.headers["Idempotency-Key"] for request in requests], ["same-key", "same-key"])
        self.assertEqual(requests[0].content, requests[1].content)
