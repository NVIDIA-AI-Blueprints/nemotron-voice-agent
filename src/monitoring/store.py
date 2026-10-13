# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session store (relational rows) and artifact store (binary blobs).

Both are synchronous and meant to be called from worker threads
(``asyncio.to_thread``), never directly from the pipeline event loop.

- ``SessionStore`` uses SQLAlchemy Core, so moving from SQLite to Postgres is a
  ``MONITORING_DB_URL`` change.
- ``ArtifactStore`` is a small protocol; ``LocalArtifactStore`` is the only
  implementation for now. An object-storage implementation only needs the same
  four methods (``local_path`` may stage to a temp dir and upload on ``commit``).
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import Engine, create_engine, delete, event, func, inspect, select, text, update
from sqlalchemy.dialects import postgresql, sqlite

from monitoring import schema
from monitoring.config import MonitoringConfig

# Tables written through ``write_batch`` and how conflicts are resolved.
_UPSERT_TABLES = {
    "turns": ("session_id", "idx"),
    "turn_metrics": ("session_id", "idx"),
    "tool_calls": ("session_id", "call_id"),
}
_IGNORE_CONFLICT_TABLES = {"media"}


class ArtifactStore(Protocol):
    """Blob storage addressed by slash-separated keys."""

    def put(self, key: str, data: bytes) -> None:
        """Store ``data`` under ``key`` (overwrite)."""

    def get(self, key: str) -> bytes:
        """Return the bytes stored under ``key``."""

    def exists(self, key: str) -> bool:
        """Return whether ``key`` exists."""

    def local_path(self, key: str) -> Path:
        """Return a local path to stream-write ``key``; call ``commit`` when done."""

    def commit(self, key: str) -> None:
        """Publish a file written through ``local_path``."""


class LocalArtifactStore:
    """Artifact store backed by a local directory."""

    def __init__(self, root: Path):
        """Use ``root`` as the store directory."""
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError(f"Artifact key escapes the store root: {key}")
        return path

    def put(self, key: str, data: bytes) -> None:
        """Store ``data`` under ``key`` (overwrite)."""
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)

    def get(self, key: str) -> bytes:
        """Return the bytes stored under ``key``."""
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        """Return whether ``key`` exists."""
        return self._path(key).exists()

    def read_path(self, key: str) -> Path:
        """Return the local file path of ``key`` for serving (no directories created)."""
        return self._path(key)

    def local_path(self, key: str) -> Path:
        """Return a local path to stream-write ``key``; call ``commit`` when done."""
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def commit(self, key: str) -> None:
        """Publish a file written through ``local_path`` (no-op locally)."""
        return None


def _configure_sqlite(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()


class SessionStore:
    """Relational store for sessions, turns, timelines, metrics, annotations and jobs."""

    def __init__(self, db_url: str):
        """Connect to ``db_url`` (tables are created by ``create_schema``)."""
        if db_url.startswith("sqlite:///"):
            Path(db_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(db_url, future=True)
        if self.engine.dialect.name == "sqlite":
            _configure_sqlite(self.engine)
        self._insert = postgresql.insert if self.engine.dialect.name == "postgresql" else sqlite.insert

    # ------------------------------------------------------------------ schema
    def create_schema(self) -> None:
        """Create missing tables and record the schema version."""
        schema.metadata.create_all(self.engine)
        self._add_missing_columns()
        with self.engine.begin() as conn:
            current = conn.execute(select(func.max(schema.schema_version.c.version))).scalar()
            # Schema changes so far only add tables (``create_all``) and nullable
            # columns (``_add_missing_columns``); neither needs a data migration.
            # Rows derived from older recordings are filled by
            # ``python -m monitoring.turn_metrics``.
            if current is None or current < schema.SCHEMA_VERSION:
                conn.execute(schema.schema_version.insert().values(version=schema.SCHEMA_VERSION))

    def _add_missing_columns(self) -> None:
        """Add columns declared in the schema but missing from an older database.

        Idempotent, and safe when two processes (server and dreamer) start together:
        a column the other process added first is skipped.
        """
        inspector = inspect(self.engine)
        for table in schema.metadata.sorted_tables:
            existing = {column["name"] for column in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing or column.primary_key or not column.nullable:
                    continue
                column_type = column.type.compile(dialect=self.engine.dialect)
                try:
                    with self.engine.begin() as conn:
                        conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {column_type}'))
                except Exception:
                    if column.name not in {c["name"] for c in inspect(self.engine).get_columns(table.name)}:
                        raise

    # ---------------------------------------------------------------- sessions
    def create_session(self, row: dict[str, Any]) -> None:
        """Insert or replace a session row."""
        with self.engine.begin() as conn:
            stmt = self._insert(schema.sessions).values(**row)
            conn.execute(stmt.on_conflict_do_update(index_elements=["id"], set_=row))

    def touch_session(self, session_id: str, ts: float) -> None:
        """Update the session heartbeat."""
        with self.engine.begin() as conn:
            conn.execute(update(schema.sessions).where(schema.sessions.c.id == session_id).values(last_seen_at=ts))

    def end_session(self, session_id: str, ts: float, reason: str) -> None:
        """Mark a session ended (first call wins)."""
        with self.engine.begin() as conn:
            conn.execute(
                update(schema.sessions)
                .where(schema.sessions.c.id == session_id, schema.sessions.c.ended_at.is_(None))
                .values(ended_at=ts, last_seen_at=ts, end_reason=reason)
            )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Return a session row, or None."""
        with self.engine.connect() as conn:
            row = conn.execute(select(schema.sessions).where(schema.sessions.c.id == session_id)).mappings().first()
        return dict(row) if row else None

    def open_session_count(self, *, now: float, stale_after_secs: float) -> int:
        """Count sessions still live (not ended and seen recently)."""
        with self.engine.connect() as conn:
            return conn.execute(
                select(func.count())
                .select_from(schema.sessions)
                .where(
                    schema.sessions.c.ended_at.is_(None),
                    schema.sessions.c.last_seen_at >= now - stale_after_secs,
                )
            ).scalar_one()

    def last_session_activity(self) -> float | None:
        """Return the most recent heartbeat/end timestamp across all sessions."""
        with self.engine.connect() as conn:
            return conn.execute(select(func.max(schema.sessions.c.last_seen_at))).scalar()

    def close_orphaned_sessions(self, *, now: float, stale_after_secs: float) -> list[str]:
        """End sessions whose recorder died without closing them (process crash)."""
        with self.engine.begin() as conn:
            stale = (
                conn.execute(
                    select(schema.sessions.c.id).where(
                        schema.sessions.c.ended_at.is_(None),
                        schema.sessions.c.last_seen_at < now - stale_after_secs,
                    )
                )
                .scalars()
                .all()
            )
            if stale:
                conn.execute(
                    update(schema.sessions)
                    .where(schema.sessions.c.id.in_(stale))
                    .values(ended_at=schema.sessions.c.last_seen_at, end_reason="orphaned")
                )
        return list(stale)

    # ------------------------------------------------------------ batch writes
    def write_batch(self, rows: Sequence[tuple[str, dict[str, Any]]]) -> None:
        """Insert ``(table_name, row)`` pairs in one transaction."""
        if not rows:
            return
        with self.engine.begin() as conn:
            for table_name, row in rows:
                table = schema.metadata.tables[table_name]
                stmt = self._insert(table).values(**row)
                if table_name in _UPSERT_TABLES:
                    keys = _UPSERT_TABLES[table_name]
                    changes = {k: v for k, v in row.items() if k not in keys}
                    stmt = stmt.on_conflict_do_update(index_elements=list(keys), set_=changes)
                elif table_name in _IGNORE_CONFLICT_TABLES:
                    stmt = stmt.on_conflict_do_nothing()
                conn.execute(stmt)

    # ------------------------------------------------------------------ reads
    def rows(self, table_name: str, session_id: str, **filters: Any) -> list[dict[str, Any]]:
        """Return rows of ``table_name`` for a session, optionally filtered by equality."""
        table = schema.metadata.tables[table_name]
        stmt = select(table).where(table.c.session_id == session_id)
        for column, value in filters.items():
            stmt = stmt.where(table.c[column] == value)
        order = [c for c in ("idx", "turn_idx", "ts", "started_at", "id") if c in table.c]
        if order:
            stmt = stmt.order_by(*(table.c[c] for c in order))
        with self.engine.connect() as conn:
            return [dict(r) for r in conn.execute(stmt).mappings()]

    def ended_session_ids(self) -> list[str]:
        """Return ids of ended sessions, oldest first."""
        with self.engine.connect() as conn:
            return list(
                conn.execute(
                    select(schema.sessions.c.id)
                    .where(schema.sessions.c.ended_at.is_not(None))
                    .order_by(schema.sessions.c.started_at)
                ).scalars()
            )

    # ---------------------------------------------------------- derived metrics
    def write_derived(
        self,
        session_id: str,
        *,
        turn_rows: Sequence[dict[str, Any]],
        call_rows: Sequence[dict[str, Any]],
        turn_updates: dict[int, dict[str, Any]],
        replace: bool,
    ) -> None:
        """Store derived turn metrics and tool calls and patch the turn rows.

        ``replace`` drops the session's previous derived rows first (whole-session
        recompute); otherwise rows are upserted (one finished turn).
        """
        with self.engine.begin() as conn:
            if replace:
                for table in (schema.turn_metrics, schema.tool_calls):
                    conn.execute(delete(table).where(table.c.session_id == session_id))
            for table, keys, rows in (
                (schema.turn_metrics, ("session_id", "idx"), turn_rows),
                (schema.tool_calls, ("session_id", "call_id"), call_rows),
            ):
                for row in rows:
                    changes = {k: v for k, v in row.items() if k not in keys}
                    stmt = self._insert(table).values(**row)
                    conn.execute(stmt.on_conflict_do_update(index_elements=list(keys), set_=changes))
            for idx, values in turn_updates.items():
                if values:
                    conn.execute(
                        update(schema.turns)
                        .where(schema.turns.c.session_id == session_id, schema.turns.c.idx == idx)
                        .values(**values)
                    )

    def add_system_sample(self, row: dict[str, Any], *, min_gap_secs: float = 0.0) -> bool:
        """Insert one host-wide system sample.

        With ``min_gap_secs`` the sample is skipped when a newer one is already
        stored, so several server workers sampling the same host do not duplicate
        rows. Returns whether the row was inserted.
        """
        t = schema.system_samples
        with self.engine.begin() as conn:
            if min_gap_secs > 0:
                latest = conn.execute(select(func.max(t.c.ts))).scalar()
                if latest is not None and latest > row["ts"] - min_gap_secs:
                    return False
            conn.execute(t.insert().values(**row))
        return True

    def system_samples_between(self, start: float, end: float) -> list[dict[str, Any]]:
        """Return the system samples with ``start <= ts <= end``, oldest first."""
        t = schema.system_samples
        stmt = select(t).where(t.c.ts >= start, t.c.ts <= end).order_by(t.c.ts)
        with self.engine.connect() as conn:
            return [dict(r) for r in conn.execute(stmt).mappings()]

    def all_session_ids(self) -> list[str]:
        """Return every session id, oldest first."""
        with self.engine.connect() as conn:
            return list(conn.execute(select(schema.sessions.c.id).order_by(schema.sessions.c.started_at)).scalars())

    # ------------------------------------------------------------ annotations
    def add_annotations(self, rows: Iterable[dict[str, Any]]) -> None:
        """Insert annotation rows (``created_at`` is filled in)."""
        now = time.time()
        payload = [{"created_at": now, **row} for row in rows]
        if not payload:
            return
        with self.engine.begin() as conn:
            conn.execute(schema.annotations.insert(), payload)

    def replace_annotations(self, session_id: str, kinds: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
        """Atomically replace a session's annotations of ``kinds`` (derived data) with ``rows``."""
        now = time.time()
        payload = [{"created_at": now, **row} for row in rows]
        with self.engine.begin() as conn:
            conn.execute(
                schema.annotations.delete().where(
                    schema.annotations.c.session_id == session_id, schema.annotations.c.kind.in_(list(kinds))
                )
            )
            if payload:
                conn.execute(schema.annotations.insert(), payload)

    def annotations_for(self, session_id: str, *, kind: str | None = None) -> list[dict[str, Any]]:
        """Return a session's annotations, optionally of one ``kind``."""
        stmt = select(schema.annotations).where(schema.annotations.c.session_id == session_id)
        if kind:
            stmt = stmt.where(schema.annotations.c.kind == kind)
        with self.engine.connect() as conn:
            return [dict(r) for r in conn.execute(stmt.order_by(schema.annotations.c.id)).mappings()]

    # ---------------------------------------------------------------------- kv
    def get_kv(self, key: str) -> dict[str, Any] | None:
        """Return ``{"value", "updated_at"}`` for ``key``, or None."""
        with self.engine.connect() as conn:
            row = conn.execute(select(schema.kv).where(schema.kv.c.key == key)).mappings().first()
        return {"value": row["value"], "updated_at": row["updated_at"]} if row else None

    def set_kv(self, key: str, value: Any) -> None:
        """Upsert ``key``."""
        now = time.time()
        stmt = self._insert(schema.kv).values(key=key, value=value, updated_at=now)
        with self.engine.begin() as conn:
            conn.execute(stmt.on_conflict_do_update(index_elements=["key"], set_={"value": value, "updated_at": now}))

    # -------------------------------------------------------------------- jobs
    def enqueue_job(self, kind: str, target: str, params: dict[str, Any] | None = None) -> bool:
        """Queue a job; returns False when the same (kind, target) already exists."""
        stmt = (
            self._insert(schema.jobs)
            .values(kind=kind, target=target, params=params or {}, status="pending", created_at=time.time())
            .on_conflict_do_nothing()
        )
        with self.engine.begin() as conn:
            return conn.execute(stmt).rowcount > 0

    def reset_job(self, kind: str, target: str, params: dict[str, Any] | None = None) -> None:
        """Queue a job from scratch, replacing any previous run of it."""
        values = {
            "status": "pending",
            "params": params or {},
            "progress": {},
            "attempts": 0,
            "error": None,
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
        }
        stmt = self._insert(schema.jobs).values(kind=kind, target=target, **values)
        with self.engine.begin() as conn:
            conn.execute(stmt.on_conflict_do_update(index_elements=["kind", "target"], set_=values))

    def claim_next_job(self, kinds: Sequence[str], *, max_attempts: int) -> dict[str, Any] | None:
        """Atomically move the oldest pending job of ``kinds`` to ``running``."""
        with self.engine.begin() as conn:
            row = (
                conn.execute(
                    select(schema.jobs)
                    .where(
                        schema.jobs.c.status == "pending",
                        schema.jobs.c.kind.in_(list(kinds)),
                        schema.jobs.c.attempts < max_attempts,
                    )
                    .order_by(schema.jobs.c.created_at)
                    .limit(1)
                )
                .mappings()
                .first()
            )
            if row is None:
                return None
            claimed = conn.execute(
                update(schema.jobs)
                .where(schema.jobs.c.id == row["id"], schema.jobs.c.status == "pending")
                .values(status="running", attempts=row["attempts"] + 1, started_at=time.time())
            ).rowcount
            if not claimed:
                return None
            return {**dict(row), "status": "running", "attempts": row["attempts"] + 1}

    def update_job(self, job_id: int, **values: Any) -> None:
        """Update columns of a job row."""
        with self.engine.begin() as conn:
            conn.execute(update(schema.jobs).where(schema.jobs.c.id == job_id).values(**values))

    def cancel_job(self, job_id: int) -> bool:
        """Cancel a pending job (running jobs are stopped by pausing the dreamer instead)."""
        with self.engine.begin() as conn:
            return (
                conn.execute(
                    update(schema.jobs)
                    .where(schema.jobs.c.id == job_id, schema.jobs.c.status == "pending")
                    .values(status="cancelled", finished_at=time.time())
                ).rowcount
                > 0
            )

    def list_jobs(
        self, *, status: str | None = None, kind: str | None = None, limit: int = 100, offset: int = 0
    ) -> tuple[int, list[dict[str, Any]]]:
        """Return ``(total, jobs)`` newest first, optionally filtered."""
        conditions = []
        if status:
            conditions.append(schema.jobs.c.status == status)
        if kind:
            conditions.append(schema.jobs.c.kind == kind)
        with self.engine.connect() as conn:
            total = conn.execute(select(func.count()).select_from(schema.jobs).where(*conditions)).scalar_one()
            rows = conn.execute(
                select(schema.jobs)
                .where(*conditions)
                .order_by(schema.jobs.c.created_at.desc(), schema.jobs.c.id.desc())
                .limit(limit)
                .offset(offset)
            ).mappings()
            return total, [dict(r) for r in rows]

    def requeue_running_jobs(self) -> int:
        """Return jobs left ``running`` by a dead runner to ``pending``."""
        with self.engine.begin() as conn:
            return conn.execute(
                update(schema.jobs).where(schema.jobs.c.status == "running").values(status="pending")
            ).rowcount

    def jobs(self, *, status: str | None = None) -> list[dict[str, Any]]:
        """Return job rows, optionally filtered by status."""
        stmt = select(schema.jobs).order_by(schema.jobs.c.id)
        if status:
            stmt = stmt.where(schema.jobs.c.status == status)
        with self.engine.connect() as conn:
            return [dict(r) for r in conn.execute(stmt).mappings()]


def open_stores(config: MonitoringConfig) -> tuple[SessionStore, LocalArtifactStore]:
    """Create (and migrate) the configured session and artifact stores."""
    store = SessionStore(config.db_url)
    store.create_schema()
    artifacts = LocalArtifactStore(config.artifacts_dir)
    return store, artifacts
