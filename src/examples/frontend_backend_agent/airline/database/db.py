# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQLite connection helpers for the booking server.

Single-file database (path from ``BOOKING_DB_PATH`` env var, default
``./data/bookings.db``).  Schema is applied idempotently on every
startup; seed runs only when the ``pnrs`` table is empty, so restarts
preserve prior activity / rebook / cancellation history.

Concurrency model: one connection per request (opened in FastAPI's
``Depends(get_api)``, closed when the request returns).  WAL journal
mode lets reads proceed concurrently with a writer; a 5-second busy
timeout absorbs short contention when two mutations land together.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from loguru import logger

DEFAULT_DB_PATH = "./data/bookings.db"
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"
_BUSY_TIMEOUT_MS = 5000


def db_path() -> Path:
    """Resolve the configured database path (creates parent dir if missing)."""
    path = Path(os.environ.get("BOOKING_DB_PATH", DEFAULT_DB_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def connect() -> sqlite3.Connection:
    """Open a per-request SQLite connection.

    - WAL journal so concurrent readers don't block on a writer.
    - ``busy_timeout`` lets a short writer-writer collision retry
      instead of raising ``OperationalError`` immediately.
    - ``check_same_thread=False`` for FastAPI's threadpool.
    - Deferred isolation (sqlite3 default) so ``with conn:`` wraps a
      multi-statement mutation as a single transaction.
    """
    conn = sqlite3.connect(str(db_path()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
    return conn


def apply_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist. Safe to re-run on every startup."""
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))


def is_empty(conn: sqlite3.Connection) -> bool:
    """True when no PNRs are stored — signal to run seed."""
    row = conn.execute("SELECT COUNT(*) AS n FROM pnrs").fetchone()
    return row["n"] == 0


def init_db() -> None:
    """Apply schema + seed-if-empty. Called once at server startup."""
    conn = connect()
    try:
        apply_schema(conn)
        if is_empty(conn):
            from examples.frontend_backend_agent.airline.database.seed import seed_all

            logger.info(f"booking db empty at {db_path()}, seeding fixtures")
            seed_all(conn)
        else:
            logger.info(f"booking db already populated at {db_path()}, skipping seed")
    finally:
        conn.close()
