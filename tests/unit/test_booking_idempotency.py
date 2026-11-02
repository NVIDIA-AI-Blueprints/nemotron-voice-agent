# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102

import sqlite3
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from examples.frontend_backend_agent.airline.backend import HTTPBookingBackend
from examples.frontend_backend_agent.airline.database.api import BookingAPI, IdempotencyConflict
from examples.frontend_backend_agent.airline.database.db import apply_schema
from examples.frontend_backend_agent.airline.database.server import app, get_api

_REQUEST = {
    "passenger": "Ada Lovelace",
    "origin": "JFK",
    "destination": "LAX",
    "flight_number": "AA124",
}


class BookingIdempotencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "bookings.db"
        with self.connect() as conn:
            apply_schema(conn)
            conn.execute(
                "INSERT INTO flights (flight_number, origin, destination, departure, arrival, cabin) "
                "VALUES ('AA124', 'JFK', 'LAX', '2026-11-10T10:00:00', '2026-11-10T16:00:00', 'economy')"
            )
        conn.close()

    def connect(self):
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def create(self, **kwargs):
        conn = self.connect()
        try:
            return BookingAPI(conn).create_booking(**{**_REQUEST, **kwargs})
        finally:
            conn.close()

    def assert_counts(self, count):
        conn = self.connect()
        try:
            for table in ("pnrs", "ancillaries", "confirmation_codes", "activity_log"):
                self.assertEqual(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], count, table)
        finally:
            conn.close()

    def test_replay_survives_new_connections_and_schema_reapplication(self):
        first = self.create(idempotency_key="retry")
        conn = self.connect()
        try:
            apply_schema(conn)
        finally:
            conn.close()
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
        conn = self.connect()
        try:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM booking_requests").fetchone()[0], 0)
        finally:
            conn.close()
        self.assertIsNotNone(self.create(idempotency_key="retry"))
        self.assert_counts(1)

    def test_missing_flight_does_not_reserve_key(self):
        self.assertIsNone(self.create(idempotency_key="retry", flight_number="AA999"))
        self.assertIsNotNone(self.create(idempotency_key="retry"))
        self.assert_counts(1)

    def test_replay_does_not_reactivate_a_cancelled_booking(self):
        first = self.create(idempotency_key="retry")
        conn = self.connect()
        try:
            with conn:
                conn.execute("UPDATE pnrs SET status = 'cancelled' WHERE pnr = ?", (first["pnr"],))
        finally:
            conn.close()
        self.assertEqual(first, self.create(idempotency_key="retry"))
        conn = self.connect()
        try:
            self.assertEqual(BookingAPI(conn).get_pnr(first["pnr"])["booking_status"], "cancelled")
        finally:
            conn.close()
        self.assert_counts(1)

    def test_http_replay_conflict_and_key_validation(self):
        def api_override():
            conn = self.connect()
            try:
                yield BookingAPI(conn)
            finally:
                conn.close()

        app.dependency_overrides[get_api] = api_override
        self.addCleanup(app.dependency_overrides.pop, get_api)
        client = TestClient(app)
        self.addCleanup(client.close)
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


class HTTPBookingBackendIdempotencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_backend_forwards_key_and_replays_the_server_receipt(self):
        requests = []

        def handle(request):
            requests.append(request)
            return httpx.Response(200, json={"pnr": "ABC123", "confirmation_code": "BOOK1234"})

        factory = partial(httpx.AsyncClient, transport=httpx.MockTransport(handle))
        flight = {
            "flight_id": "AA124",
            "origin_airport": "JFK",
            "dest_airport": "LAX",
            "origin_city": "New York",
            "dest_city": "Los Angeles",
            "date": "2026-11-10",
            "departure_time": "2026-11-10T10:00:00",
        }
        with patch("examples.frontend_backend_agent.airline.backend.httpx.AsyncClient", factory):
            backend = HTTPBookingBackend("http://booking.test")
            for _ in range(2):
                result = await backend.create_booking(passenger_name="Ada", flight=flight, idempotency_key="same-key")
                self.assertEqual(result["pnr"], "ABC123")
        self.assertEqual([request.headers["Idempotency-Key"] for request in requests], ["same-key", "same-key"])
        self.assertEqual(requests[0].content, requests[1].content)
