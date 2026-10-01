# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Booking backend clients for the independent Frontend/Backend Agent example."""

from __future__ import annotations

from typing import Any, Protocol

import httpx

from examples.frontend_backend_agent.airline.transform import (
    server_booking_to_record,
    server_flight_to_option,
    server_pnr_to_record,
    sort_flights,
    unique_flights,
)
from examples.frontend_backend_agent.src.delegation import current_run_id


class BookingBackend(Protocol):
    """Backend interface used by the Thinker internal tools."""

    async def search_flights(
        self,
        *,
        origin: str,
        destination: str,
        travel_date: str,
        sorting: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return matching flight options."""

    async def create_booking(
        self,
        *,
        passenger_name: str | None,
        flight: dict[str, Any],
        seat_pref: str | None = None,
        meal_pref: str | None = None,
    ) -> dict[str, Any] | None:
        """Create a booking for ``flight``."""

    async def get_pnr(self, pnr_code: str) -> dict[str, Any] | None:
        """Return a booking record by PNR."""


class HTTPBookingBackend:
    """HTTP client for the shared booking-server sidecar."""

    def __init__(self, base_url: str, *, timeout: float = 10.0) -> None:
        """Create a backend client for ``base_url``."""
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def search_flights(
        self,
        *,
        origin: str,
        destination: str,
        travel_date: str,
        sorting: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return flights from the booking server."""
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            response = await client.get(
                "/flights",
                params={"origin": origin, "destination": destination, "date": travel_date},
            )
        response.raise_for_status()
        rows = response.json()
        flights = [server_flight_to_option(row, fallback_date=travel_date) for row in rows if isinstance(row, dict)]
        return sort_flights(unique_flights(flights), sorting)[:5]

    async def create_booking(
        self,
        *,
        passenger_name: str | None,
        flight: dict[str, Any],
        seat_pref: str | None = None,
        meal_pref: str | None = None,
    ) -> dict[str, Any] | None:
        """Create a PNR through the booking server."""
        payload = {
            "passenger": passenger_name or "Guest",
            "origin": flight["origin_airport"],
            "destination": flight["dest_airport"],
            "flight_number": flight["flight_id"],
            "departure": flight.get("departure_time"),
            "seat": seat_pref,
            "meal": meal_pref,
        }
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            response = await client.post("/pnrs", json={key: value for key, value in payload.items() if value})
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return server_booking_to_record(
            response.json(),
            flight=flight,
            passenger_name=passenger_name,
            seat_pref=seat_pref,
            meal_pref=meal_pref,
        )

    async def get_pnr(self, pnr_code: str) -> dict[str, Any] | None:
        """Look up PNR status through the booking server."""
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            response = await client.get(f"/pnrs/{pnr_code.strip().upper()}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return server_pnr_to_record(response.json())


class RecordingBookingBackend:
    """Record the booking write where it happens; reads pass straight through.

    ``create_booking`` is the airline domain's only side effect. A ``booking``
    plan that only asks for confirmation never reaches it, so it is never
    recorded as a write.
    """

    def __init__(self, backend: BookingBackend, ledger: Any) -> None:
        """Wrap ``backend`` and record writes in the session's delegation ledger."""
        self._backend = backend
        self._ledger = ledger

    async def search_flights(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Return flights from the wrapped backend."""
        return await self._backend.search_flights(**kwargs)

    async def get_pnr(self, pnr_code: str) -> dict[str, Any] | None:
        """Return a PNR record from the wrapped backend."""
        return await self._backend.get_pnr(pnr_code)

    async def create_booking(self, **kwargs: Any) -> dict[str, Any] | None:
        """Create a booking, recording it as started, then confirmed or unconfirmed."""
        flight = kwargs.get("flight") if isinstance(kwargs.get("flight"), dict) else {}
        arguments = {
            "passenger_name": kwargs.get("passenger_name"),
            "flight_id": flight.get("flight_id"),
            "date": flight.get("date"),
            "seat_pref": kwargs.get("seat_pref"),
            "meal_pref": kwargs.get("meal_pref"),
        }
        record = self._ledger.write_started(current_run_id(), "create_booking", arguments)
        try:
            result = await self._backend.create_booking(**kwargs)
        except BaseException:
            # A failed or cancelled HTTP request may still have created the PNR.
            self._ledger.write_unconfirmed(record)
            raise
        self._ledger.write_confirmed(record, "success" if result is not None else "not_found")
        return result
