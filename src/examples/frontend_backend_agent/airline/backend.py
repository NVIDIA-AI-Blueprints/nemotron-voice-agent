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
