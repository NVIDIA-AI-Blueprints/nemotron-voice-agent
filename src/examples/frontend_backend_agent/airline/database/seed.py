# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed data for a fresh booking database.

The fixture rows themselves live in JSONL files under ``seed_data/``
(``flights.jsonl`` for the alternatives catalog, ``pnrs.jsonl`` for the
PNR fixtures); this module only owns the load + insert logic.
Idempotent only in the sense that ``db.init_db`` calls this at most
once per database lifecycle (gated on empty ``pnrs``); the individual
INSERTs themselves rely on UNIQUE constraints and will raise on repeat.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

_SEED_DATA_DIR = Path(__file__).parent / "seed_data"
_FLIGHTS_PATH = _SEED_DATA_DIR / "flights.jsonl"
_PNRS_PATH = _SEED_DATA_DIR / "pnrs.jsonl"


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts. Skips blank lines."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def seed_all(conn: sqlite3.Connection) -> None:
    """Insert all fixture rows into an empty database (single transaction).

    Loads two JSONL catalogs:
      * ``seed_data/flights.jsonl`` — flight alternatives (~40 curated
        entries aligned with PNR fixtures plus ~6.6K synthesised
        schedules over real OpenFlights route topology between 36 US
        hubs, 2026-04-22 → 2026-05-10).
      * ``seed_data/pnrs.jsonl`` — 11 PNR fixtures, each tuned to one
        scenario class (delay, weather cancel, misconnect, diversion,
        airline cancel, basic-economy, international, red-eye, etc.).
        Each PNR row is also inserted into the ``flights`` table (as a
        distinct row, no overlap with ``flights.jsonl``), so the final
        seeded flight count is ``len(flights.jsonl) + len(pnrs.jsonl)``
        (currently 6,690 + 11 = 6,701).
    """
    flights = _load_jsonl(_FLIGHTS_PATH)
    pnrs = _load_jsonl(_PNRS_PATH)
    now = _now()
    with conn:
        for row in flights:
            conn.execute(
                "INSERT OR IGNORE INTO flights "
                "(flight_number, origin, destination, departure, arrival, cabin) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    row["flight_number"],
                    row["origin"],
                    row["destination"],
                    row["departure"],
                    row["arrival"],
                    row["cabin"],
                ),
            )
        for pnr_row in pnrs:
            conn.execute(
                "INSERT OR IGNORE INTO flights "
                "(flight_number, origin, destination, departure, arrival, cabin, status, delay_minutes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    pnr_row["flight_number"],
                    pnr_row["origin"],
                    pnr_row["destination"],
                    pnr_row["departure"],
                    pnr_row["arrival"],
                    pnr_row["cabin"],
                    pnr_row["status"],
                    pnr_row["delay_minutes"],
                ),
            )

        flight_key_to_id: dict[tuple[str, str], int] = {
            (row["flight_number"], row["departure"]): row["id"]
            for row in conn.execute("SELECT id, flight_number, departure FROM flights")
        }

        for pnr_row in pnrs:
            flight_id = flight_key_to_id[(pnr_row["flight_number"], pnr_row["departure"])]
            conn.execute(
                "INSERT INTO pnrs (pnr, passenger, flight_id, fare_basis, elite_tier, "
                "                  status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 'active', ?, ?)",
                (
                    pnr_row["pnr"],
                    pnr_row["passenger"],
                    flight_id,
                    pnr_row["fare_basis"],
                    pnr_row["elite_tier"],
                    now,
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO ancillaries (pnr, seat, bag_count, meal) VALUES (?, ?, ?, ?)",
                (pnr_row["pnr"], pnr_row["seat"], pnr_row["bag_count"], pnr_row["meal"]),
            )
