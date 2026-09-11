-- SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- Booking-server SQLite schema.
-- Applied on startup via ``db.apply_schema``.  Safe to re-run.

PRAGMA foreign_keys = ON;

-- Shared flight catalog.  Scheduled flights + operational status live here.
CREATE TABLE IF NOT EXISTS flights (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    flight_number  TEXT    NOT NULL,
    origin         TEXT    NOT NULL,
    destination    TEXT    NOT NULL,
    departure      TEXT    NOT NULL,              -- ISO 8601
    arrival        TEXT    NOT NULL,
    cabin          TEXT    NOT NULL,
    status         TEXT    NOT NULL DEFAULT 'scheduled',
    delay_minutes  INTEGER NOT NULL DEFAULT 0,
    UNIQUE(flight_number, departure)
);
CREATE INDEX IF NOT EXISTS idx_flights_route ON flights(origin, destination, departure);

-- Bookings.  A PNR points at the flight it is currently booked on.
-- Rebook mutations replace ``flight_id`` and update ``updated_at``.
CREATE TABLE IF NOT EXISTS pnrs (
    pnr         TEXT    PRIMARY KEY,
    passenger   TEXT    NOT NULL,
    flight_id   INTEGER NOT NULL REFERENCES flights(id),
    fare_basis  TEXT    NOT NULL,
    elite_tier  TEXT    NOT NULL DEFAULT 'none',
    status      TEXT    NOT NULL DEFAULT 'active',   -- active | cancelled
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

-- Ancillaries per PNR.  Split from ``pnrs`` so rebook doesn't have to
-- touch seat/bag/meal columns; we just leave the row pointing at the
-- PNR and let the caller decide whether to carry forward.
CREATE TABLE IF NOT EXISTS ancillaries (
    pnr        TEXT    PRIMARY KEY REFERENCES pnrs(pnr) ON DELETE CASCADE,
    seat       TEXT,
    bag_count  INTEGER NOT NULL DEFAULT 0,
    meal       TEXT
);

-- Full audit log of every mutation on a PNR.  Rebook captures both
-- from_* and to_* columns so you can reconstruct the route chain.
-- Cancel / standby fill only the relevant subset.
CREATE TABLE IF NOT EXISTS activity_log (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    pnr                TEXT    NOT NULL,
    action             TEXT    NOT NULL,  -- REBOOK | CANCEL | STANDBY | BOOK
    session_id         TEXT,              -- pipeline stream_id from X-Session-Id
    from_flight        TEXT,
    from_origin        TEXT,
    from_destination   TEXT,
    from_departure     TEXT,
    to_flight          TEXT,
    to_origin          TEXT,
    to_destination     TEXT,
    to_departure       TEXT,
    outcome            TEXT,
    confirmation_code  TEXT,
    policy_ref         TEXT,
    amounts_json       TEXT,
    notes              TEXT,
    created_at         TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_activity_pnr_time ON activity_log(pnr, created_at);
CREATE INDEX IF NOT EXISTS idx_activity_session ON activity_log(session_id, created_at);

-- Confirmation-code uniqueness ledger.  Every mutation that produces a
-- code inserts here; DB-level UNIQUE catches collisions so the caller
-- can retry the random-generation loop.
CREATE TABLE IF NOT EXISTS confirmation_codes (
    code       TEXT PRIMARY KEY,
    pnr        TEXT NOT NULL,
    action     TEXT NOT NULL,
    issued_at  TEXT NOT NULL
);
