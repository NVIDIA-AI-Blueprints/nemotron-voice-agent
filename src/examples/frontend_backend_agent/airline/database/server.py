# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI layer over :class:`examples.frontend_backend_agent.airline.database.api.BookingAPI`.

Run directly::

    python -m examples.frontend_backend_agent.airline.database.server

or via ``uvicorn examples.frontend_backend_agent.airline.database.server:app --port 8001``.

Endpoint surface (one method per :class:`BookingAPI` call):

- ``GET  /health``                             — liveness
- ``GET  /pnrs/{pnr}``                         — booking lookup (fuzzy-tolerant)
- ``GET  /pnrs/{pnr}/activity``                — full audit log
- ``GET  /pnrs/{pnr}/ancillaries-diff``        — preview seat/bag/meal carry-over
- ``GET  /flights?origin=&destination=``       — route alternatives
- ``GET  /flights/{flight_number}/pnrs``       — bookings on a flight
- ``GET  /flights/{flight_number}/status``     — live flight status
- ``POST /pnrs/{pnr}/rebook``                  — commit a flight swap
- ``POST /pnrs/{pnr}/cancel``                  — cancel + refund/credit
- ``POST /pnrs/{pnr}/standby``                 — list for an earlier flight

Concurrency: each request gets its own SQLite connection via the
:func:`get_api` dependency so writers don't contend on a shared
connection.  Pipeline-side session correlation is carried through the
``X-Session-Id`` header and stamped on every activity_log row.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from loguru import logger

from examples.frontend_backend_agent.airline.database.api import BookingAPI
from examples.frontend_backend_agent.airline.database.db import connect, init_db
from examples.frontend_backend_agent.airline.database.schemas import (
    ActivityEvent,
    AncillariesDiff,
    Booking,
    BookingCreateRequest,
    CancelRequest,
    Flight,
    MutationResponse,
    RebookRequest,
    RebookResponse,
    StandbyRequest,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Apply schema + seed once per process."""
    init_db()
    logger.info("booking-server ready")
    yield


app = FastAPI(title="Airline Booking Server", version="1.0.0", lifespan=lifespan)


def get_api() -> Iterator[BookingAPI]:
    """Per-request API with its own short-lived connection.

    WAL journal + busy-timeout let concurrent sessions mutate without
    tripping the default writer-exclusivity lock.  The connection
    closes automatically when the request handler returns.
    """
    conn = connect()
    try:
        yield BookingAPI(conn)
    finally:
        conn.close()


ApiDependency = Annotated[BookingAPI, Depends(get_api)]


# ----------------------------------------------------------------------
# Health
# ----------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    """Liveness probe — returns ``{"status": "ok"}`` once the app is up."""
    return {"status": "ok"}


# ----------------------------------------------------------------------
# Reads
# ----------------------------------------------------------------------


@app.get("/pnrs/{pnr}", response_model=Booking)
def get_pnr(pnr: str, api: ApiDependency) -> Booking:
    """Look up a single booking by PNR."""
    booking = api.get_pnr(pnr)
    if booking is None:
        raise HTTPException(status_code=404, detail=f"PNR {pnr!r} not found")
    return booking


@app.get("/pnrs/{pnr}/activity", response_model=list[ActivityEvent])
def get_activity(pnr: str, api: ApiDependency) -> list[dict]:
    """Return the audit log for ``pnr`` (newest first)."""
    return api.get_activity(pnr)


@app.get("/pnrs/{pnr}/ancillaries-diff", response_model=AncillariesDiff)
def ancillaries_diff(
    pnr: str,
    new_flight_number: str,
    api: ApiDependency,
) -> AncillariesDiff:
    """Diff seat / meal between the current and proposed flight."""
    diff = api.ancillaries_diff(pnr, new_flight_number)
    if diff is None:
        raise HTTPException(status_code=404, detail=f"PNR {pnr!r} not found")
    return diff


@app.get("/flights", response_model=list[Flight])
def list_alternatives(
    origin: str,
    destination: str,
    api: ApiDependency,
    date: str | None = Query(None, description="Optional YYYY-MM-DD date to materialize the route schedule onto."),
) -> list[Flight]:
    """List scheduled flights between ``origin`` and ``destination``."""
    return api.list_alternatives(origin, destination, travel_date=date)


@app.get("/routes")
def list_routes(
    api: ApiDependency,
    origin: str | None = Query(None, description="Filter: airports reachable from this origin"),
    destination: str | None = Query(None, description="Filter: airports that fly into this destination"),
) -> list[dict]:
    """List airports served from ``origin`` or that feed ``destination``."""
    try:
        return api.list_routes(origin=origin, destination=destination)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/flights/{flight_number}/pnrs")
def find_by_flight(flight_number: str, api: ApiDependency) -> list[dict]:
    """Return all bookings currently sitting on ``flight_number``."""
    return api.find_by_flight(flight_number)


@app.get("/flights/{flight_number}/status")
def get_flight_status(flight_number: str, api: ApiDependency) -> dict:
    """Return live flight status (independent of whether any PNR is booked on it)."""
    result = api.get_flight_status(flight_number)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Flight {flight_number!r} not scheduled")
    return result


# ----------------------------------------------------------------------
# Mutations — session_id comes through the ``X-Session-Id`` header
# ----------------------------------------------------------------------


SessionHeader = Header(default=None, alias="X-Session-Id")


@app.get("/price")
def price_quote(
    api: ApiDependency,
    origin: str = Query(..., description="Origin IATA code"),
    destination: str = Query(..., description="Destination IATA code"),
    cabin: str = Query("economy", description="Cabin class (economy / premium_economy / business / first)"),
) -> dict:
    """Return a fare estimate for a prospective booking (no DB write)."""
    if not api.route_exists(origin, destination):
        raise HTTPException(status_code=404, detail=f"No scheduled route from {origin!r} to {destination!r}")
    price = api.price_for(origin, destination, cabin)
    return {
        "origin": origin.upper(),
        "destination": destination.upper(),
        "cabin": cabin,
        "price": price,
        "currency": "USD",
    }


@app.post("/pnrs")
def create_booking(
    body: BookingCreateRequest,
    api: ApiDependency,
    x_session_id: str | None = SessionHeader,
) -> dict:
    """Mint a brand-new PNR for the caller and return the booking + code."""
    result = api.create_booking(
        passenger=body.passenger,
        origin=body.origin,
        destination=body.destination,
        flight_number=body.flight_number,
        departure=body.departure,
        seat=body.seat,
        meal=body.meal,
        cabin=body.cabin,
        session_id=x_session_id,
    )
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Could not create booking on {body.flight_number!r} for "
                f"{body.origin!r}->{body.destination!r}: flight not scheduled"
            ),
        )
    return result


@app.post("/pnrs/{pnr}/rebook", response_model=RebookResponse)
def rebook(
    pnr: str,
    body: RebookRequest,
    api: ApiDependency,
    x_session_id: str | None = SessionHeader,
) -> RebookResponse:
    """Swap ``pnr`` onto a different flight (and optionally update seat / meal)."""
    result = api.commit_rebook(
        pnr,
        body.new_flight_number,
        session_id=x_session_id,
        seat=body.seat,
        meal=body.meal,
        departure=body.departure,
    )
    if result is None:
        raise HTTPException(status_code=404, detail=f"PNR {pnr!r} or flight {body.new_flight_number!r} not found")
    return {
        "confirmation_code": result["confirmation_code"],
        "pnr": result["booking"]["pnr"],
        "booking": result["booking"],
        "from_flight": result["from_flight"],
        "to_flight": result["to_flight"],
    }


@app.post("/pnrs/{pnr}/cancel", response_model=MutationResponse)
def cancel(
    pnr: str,
    body: CancelRequest,
    api: ApiDependency,
    x_session_id: str | None = SessionHeader,
) -> MutationResponse:
    """Cancel ``pnr`` under the given policy and return the confirmation code."""
    result = api.cancel_booking(pnr, body.kind, body.policy_ref, session_id=x_session_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"PNR {pnr!r} not found")
    return {"confirmation_code": result["confirmation_code"], "pnr": result["pnr"]}


@app.post("/pnrs/{pnr}/standby", response_model=MutationResponse)
def standby(
    pnr: str,
    body: StandbyRequest,
    api: ApiDependency,
    x_session_id: str | None = SessionHeader,
) -> MutationResponse:
    """Add ``pnr`` to the standby list for ``body.flight_number``."""
    result = api.list_standby(pnr, body.flight_number, session_id=x_session_id, departure=body.departure)
    if result is None:
        raise HTTPException(status_code=404, detail=f"PNR {pnr!r} or flight {body.flight_number!r} not found")
    return {"confirmation_code": result["confirmation_code"], "pnr": result["pnr"]}


if __name__ == "__main__":
    import os

    import uvicorn

    host = os.environ.get("BOOKING_HOST", "0.0.0.0")
    port = int(os.environ.get("BOOKING_PORT", "8001"))
    uvicorn.run("examples.frontend_backend_agent.airline.database.server:app", host=host, port=port, log_level="info")
