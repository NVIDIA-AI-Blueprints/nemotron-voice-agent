# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Transactional single-host SQLite state for VoiceClaw recovery evidence."""

from __future__ import annotations

import errno
import json
import os
import sqlite3
import stat
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from voiceclaw.domain.models import (
    AgentQueryKind,
    AgentQueryProjection,
    AgentQueryState,
    BackendOperation,
    CommandRecord,
    CommandState,
    DisplayPayload,
    DisplayState,
    PresentationPriority,
    PresentationRecord,
    ResultState,
    SessionBinding,
    SessionControlSnapshot,
    SpeechRoute,
    SpeechState,
    WorkProjection,
    WorkState,
    utc_now,
)
from voiceclaw.ports.state import StaleSessionControlError

_SCHEMA_VERSION = 6
_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_UNSETTLED_COMMAND_STATES = (
    CommandState.STAGED.value,
    CommandState.DISPATCHING.value,
    CommandState.INCONCLUSIVE.value,
    CommandState.RECONCILING.value,
)


def _reject_posix_acl(path_or_descriptor: str | int, *, label: str, default: bool = False) -> None:
    """Reject ACLs that could grant access beyond the checked mode bits."""
    getxattr = getattr(os, "getxattr", None)
    if getxattr is None:
        return
    attribute = "system.posix_acl_default" if default else "system.posix_acl_access"
    try:
        value = getxattr(path_or_descriptor, attribute)
    except OSError as error:
        no_acl_errors = {
            errno.ENODATA,
            errno.ENOTSUP,
            getattr(errno, "ENOATTR", errno.ENODATA),
        }
        if error.errno not in no_acl_errors:
            raise PermissionError(f"{label} POSIX ACL could not be inspected") from error
    else:
        if value:
            raise PermissionError(f"{label} must not have a POSIX ACL")


def _prepare_private_state_directory(path: Path) -> None:
    """Create a private state directory or reject an unsafe existing one."""
    parent = path.parent
    created = False
    try:
        parent.mkdir(parents=True, mode=_PRIVATE_DIRECTORY_MODE)
        created = True
    except FileExistsError:
        pass
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(parent, flags)
    except OSError as error:
        raise PermissionError("VoiceClaw state directory must be a real directory, not a symbolic link") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise PermissionError("VoiceClaw state directory must be a directory")
        if metadata.st_uid != os.geteuid():
            raise PermissionError("VoiceClaw state directory must be owned by the VoiceClaw process identity")
        if created:
            os.fchmod(descriptor, _PRIVATE_DIRECTORY_MODE)
            metadata = os.fstat(descriptor)
        mode = stat.S_IMODE(metadata.st_mode)
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise PermissionError("VoiceClaw state directory must have owner-only permissions (0700 or stricter)")
        required_owner_access = stat.S_IWUSR | stat.S_IXUSR
        if mode & required_owner_access != required_owner_access:
            raise PermissionError("VoiceClaw state directory must be writable and searchable by its owner")
        _reject_posix_acl(descriptor, label="VoiceClaw state directory")
        _reject_posix_acl(descriptor, label="VoiceClaw state directory", default=True)
    finally:
        os.close(descriptor)


def _secure_state_file(path: Path, *, create: bool) -> None:
    """Open one state artifact without following links and enforce mode 0600."""
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    if create:
        flags |= os.O_CREAT
    try:
        descriptor = os.open(path, flags, _PRIVATE_FILE_MODE)
    except FileNotFoundError:
        if create:
            raise
        return
    except OSError as error:
        raise PermissionError(f"VoiceClaw state artifact could not be opened securely: {path.name}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise PermissionError(f"VoiceClaw state artifact must be one regular, unlinked file: {path.name}")
        if metadata.st_uid != os.geteuid():
            raise PermissionError(f"VoiceClaw state artifact must be owned by the VoiceClaw process: {path.name}")
        _reject_posix_acl(descriptor, label=f"VoiceClaw state artifact {path.name}")
        os.fchmod(descriptor, _PRIVATE_FILE_MODE)
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != _PRIVATE_FILE_MODE:
            raise PermissionError(f"VoiceClaw state artifact could not be restricted to mode 0600: {path.name}")
    finally:
        os.close(descriptor)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC).isoformat()


def _datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("stored timestamp is not timezone-aware")
    return parsed


class SqliteStateStore:
    """Store only VoiceClaw mappings, cursors, outbox, and delivery state."""

    def __init__(self, path: str | Path) -> None:
        """Open or create a local SQLite database."""
        self.path = str(path)
        if self.path != ":memory:":
            state_path = Path(self.path)
            _prepare_private_state_directory(state_path)
            _secure_state_file(state_path, create=True)
        self._lock = RLock()
        self._connection = sqlite3.connect(self.path, timeout=5.0, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        try:
            self._initialize()
            if self.path != ":memory:":
                state_path = Path(self.path)
                for suffix in ("", "-wal", "-shm", "-journal"):
                    _secure_state_file(Path(f"{state_path}{suffix}"), create=suffix == "")
        except BaseException:
            self._connection.close()
            raise

    def __enter__(self) -> SqliteStateStore:
        """Return the open store for context-manager use."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the store on context-manager exit."""
        self.close()

    def close(self) -> None:
        """Close the embedded database connection."""
        with self._lock:
            self._connection.close()

    def save_session(self, binding: SessionBinding) -> None:
        """Insert or update a local session/backend attachment mapping."""
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO session_bindings (
                    session_id, conversation_id, backend_profile, attachment_id, backend_session_id,
                    last_applied_sequence, last_presented_sequence, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    conversation_id = excluded.conversation_id,
                    backend_profile = excluded.backend_profile,
                    attachment_id = excluded.attachment_id,
                    backend_session_id = excluded.backend_session_id,
                    last_applied_sequence = CASE
                        WHEN session_bindings.attachment_id IS excluded.attachment_id
                        THEN MAX(session_bindings.last_applied_sequence, excluded.last_applied_sequence)
                        ELSE excluded.last_applied_sequence
                    END,
                    last_presented_sequence = CASE
                        WHEN session_bindings.attachment_id IS excluded.attachment_id
                        THEN MAX(session_bindings.last_presented_sequence, excluded.last_presented_sequence)
                        ELSE excluded.last_presented_sequence
                    END,
                    control_revision = session_bindings.control_revision + 1,
                    updated_at = excluded.updated_at
                """,
                (
                    binding.session_id,
                    binding.conversation_id,
                    binding.backend_profile,
                    binding.attachment_id,
                    binding.backend_session_id,
                    binding.last_applied_sequence,
                    binding.last_presented_sequence,
                    _timestamp(binding.created_at),
                    _timestamp(binding.updated_at),
                ),
            )

    def get_session(self, session_id: str) -> SessionBinding | None:
        """Load a session mapping."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM session_bindings WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return self._session_from_row(row) if row is not None else None

    def discard_ephemeral_session(self, session_id: str) -> None:
        """Delete only a response-only mapping with no durable child evidence."""
        with self._lock, self._connection:
            row = self._connection.execute(
                """
                SELECT
                    attachment_id,
                    backend_session_id,
                    EXISTS(SELECT 1 FROM work_projections WHERE session_id = ?) AS has_work,
                    EXISTS(SELECT 1 FROM agent_query_projections WHERE session_id = ?) AS has_queries,
                    EXISTS(SELECT 1 FROM presentations WHERE session_id = ?) AS has_presentations,
                    EXISTS(SELECT 1 FROM commands WHERE session_id = ?) AS has_commands
                FROM session_bindings
                WHERE session_id = ?
                """,
                (session_id, session_id, session_id, session_id, session_id),
            ).fetchone()
            if row is None:
                return
            if (
                row["attachment_id"] is not None
                or row["backend_session_id"] is not None
                or bool(row["has_work"])
                or bool(row["has_queries"])
                or bool(row["has_presentations"])
                or bool(row["has_commands"])
            ):
                raise ValueError("cannot discard a session with durable backend or Work evidence")
            self._connection.execute(
                "DELETE FROM session_bindings WHERE session_id = ?",
                (session_id,),
            )

    def advance_cursor(self, session_id: str, sequence: int) -> int:
        """Advance and return a monotonically increasing replay cursor."""
        if sequence < 0:
            raise ValueError("sequence must be non-negative")
        now = _timestamp(utc_now())
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT last_applied_sequence FROM session_bindings WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise KeyError(session_id)
            current = int(row["last_applied_sequence"])
            if sequence > current:
                self._connection.execute(
                    """
                    UPDATE session_bindings
                    SET last_applied_sequence = ?,
                        control_revision = control_revision + 1,
                        updated_at = ?
                    WHERE session_id = ?
                    """,
                    (sequence, now, session_id),
                )
                return sequence
            return current

    def advance_presented_cursor(self, session_id: str, sequence: int) -> int:
        """Persist an evidence-derived presented prefix without passing applied state.

        This storage primitive does not decide whether content was presented.
        Only a delivery-receipt aggregator that proved a contiguous prefix may
        call it; the Interaction Coordinator deliberately exposes no arbitrary
        integer setter.
        """
        if sequence < 0:
            raise ValueError("sequence must be non-negative")
        now = _timestamp(utc_now())
        with self._lock, self._connection:
            row = self._connection.execute(
                """
                SELECT last_applied_sequence, last_presented_sequence
                FROM session_bindings WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                raise KeyError(session_id)
            applied = int(row["last_applied_sequence"])
            current = int(row["last_presented_sequence"])
            if sequence > applied:
                raise ValueError("presented cursor cannot advance beyond the applied-event cursor")
            if sequence > current:
                self._connection.execute(
                    """
                    UPDATE session_bindings
                    SET last_presented_sequence = ?, updated_at = ?
                    WHERE session_id = ?
                    """,
                    (sequence, now, session_id),
                )
                return sequence
            return current

    def save_work_projection(self, projection: WorkProjection) -> bool:
        """Persist a newer session-scoped projection without rewriting terminal Work."""
        with self._lock, self._connection:
            attachment_row = self._connection.execute(
                "SELECT session_id FROM work_projections WHERE attachment_id = ? AND work_id = ?",
                (projection.attachment_id, projection.work_id),
            ).fetchone()
            if attachment_row is not None and attachment_row["session_id"] != projection.session_id:
                raise ValueError("Work projection cannot move between voice sessions")

            rows = self._connection.execute(
                """
                SELECT state, sequence, summary, raw_state, agent_target
                FROM work_projections
                WHERE session_id = ? AND work_id = ?
                ORDER BY sequence DESC, updated_at DESC
                """,
                (projection.session_id, projection.work_id),
            ).fetchall()
            if rows:
                if projection.agent_target != rows[0]["agent_target"]:
                    raise ValueError("Work agent_target is immutable")
                newest_sequence = int(rows[0]["sequence"])
                newest_rows = [row for row in rows if int(row["sequence"]) == newest_sequence]
                newest_payload = self._work_payload_from_row(newest_rows[0])
                if any(self._work_payload_from_row(row) != newest_payload for row in newest_rows[1:]):
                    raise ValueError("conflicting Work projections share the newest backend sequence")

                incoming_payload = self._work_payload_from_projection(projection)
                if projection.sequence < newest_sequence:
                    return False
                if projection.sequence == newest_sequence:
                    if incoming_payload != newest_payload:
                        raise ValueError("conflicting Work projections share the newest backend sequence")
                    return False

                newest_state = WorkState(rows[0]["state"])
                if newest_state.terminal:
                    if incoming_payload != newest_payload:
                        raise ValueError("terminal Work outcome is immutable")
                    return False
            self._connection.execute(
                """
                INSERT INTO work_projections (
                    attachment_id, work_id, session_id, state, sequence, summary,
                    raw_state, agent_target, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(attachment_id, work_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    state = excluded.state,
                    sequence = excluded.sequence,
                    summary = excluded.summary,
                    raw_state = excluded.raw_state,
                    agent_target = excluded.agent_target,
                    updated_at = excluded.updated_at
                """,
                (
                    projection.attachment_id,
                    projection.work_id,
                    projection.session_id,
                    projection.state.value,
                    projection.sequence,
                    projection.summary,
                    projection.raw_state,
                    projection.agent_target,
                    _timestamp(projection.updated_at),
                ),
            )
            self._increment_control_revision(projection.session_id)
            return True

    def get_work_projection(self, attachment_id: str, work_id: str) -> WorkProjection | None:
        """Load one Work projection."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM work_projections WHERE attachment_id = ? AND work_id = ?",
                (attachment_id, work_id),
            ).fetchone()
        if row is None:
            return None
        return self._work_from_row(row)

    def get_session_work_projection(self, session_id: str, work_id: str) -> WorkProjection | None:
        """Load the newest attachment projection for session-scoped backend Work."""
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM work_projections
                WHERE session_id = ? AND work_id = ?
                ORDER BY sequence DESC, updated_at DESC
                """,
                (session_id, work_id),
            ).fetchall()
        if not rows:
            return None
        newest_sequence = int(rows[0]["sequence"])
        newest_rows = [row for row in rows if int(row["sequence"]) == newest_sequence]
        newest_payload = self._work_payload_from_row(newest_rows[0])
        if any(self._work_payload_from_row(row) != newest_payload for row in newest_rows[1:]):
            raise ValueError("conflicting Work projections share the newest backend sequence")
        return self._work_from_row(rows[0])

    def get_session_control_snapshot(self, session_id: str) -> SessionControlSnapshot:
        """Atomically capture the applied Work/query prefix and query claims."""
        with self._lock:
            # RLock alone protects only users of this store instance. An
            # explicit read transaction gives every SELECT below one SQLite
            # snapshot when another process or connection is writing.
            self._connection.execute("BEGIN DEFERRED")
            try:
                binding = self._connection.execute(
                    """
                    SELECT last_applied_sequence, backend_session_id, attachment_id, control_revision
                    FROM session_bindings
                    WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchone()
                if binding is None:
                    raise KeyError(session_id)
                applied_sequence = int(binding["last_applied_sequence"])
                control_revision = int(binding["control_revision"])
                work_rows = self._connection.execute(
                    """
                    SELECT * FROM work_projections
                    WHERE session_id = ? AND sequence <= ?
                    ORDER BY work_id ASC, sequence DESC, updated_at DESC
                    """,
                    (session_id, applied_sequence),
                ).fetchall()
                query_rows = self._connection.execute(
                    """
                    SELECT * FROM agent_query_projections
                    WHERE session_id = ? AND sequence <= ?
                    ORDER BY query_id ASC, sequence DESC, updated_at DESC
                    """,
                    (session_id, applied_sequence),
                ).fetchall()
                command_rows: list[sqlite3.Row]
                backend_session_id = binding["backend_session_id"]
                if backend_session_id is not None:
                    command_rows = self._connection.execute(
                        """
                        SELECT operation, state, work_id, payload_json FROM commands
                        WHERE session_id = ? AND backend_session_id = ?
                          AND state != ?
                        ORDER BY created_at ASC, command_id ASC
                        """,
                        (
                            session_id,
                            backend_session_id,
                            CommandState.REJECTED.value,
                        ),
                    ).fetchall()
                else:
                    command_rows = self._connection.execute(
                        """
                        SELECT operation, state, work_id, payload_json FROM commands
                        WHERE session_id = ? AND attachment_id = ?
                          AND backend_session_id IS NULL AND state != ?
                        ORDER BY created_at ASC, command_id ASC
                        """,
                        (
                            session_id,
                            binding["attachment_id"],
                            CommandState.REJECTED.value,
                        ),
                    ).fetchall()
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

        projections: list[WorkProjection] = []
        offset = 0
        while offset < len(work_rows):
            work_id = work_rows[offset]["work_id"]
            matching_rows: list[sqlite3.Row] = []
            while offset < len(work_rows) and work_rows[offset]["work_id"] == work_id:
                matching_rows.append(work_rows[offset])
                offset += 1
            newest_sequence = int(matching_rows[0]["sequence"])
            newest_rows = [row for row in matching_rows if int(row["sequence"]) == newest_sequence]
            newest_payload = self._work_payload_from_row(newest_rows[0])
            if any(self._work_payload_from_row(row) != newest_payload for row in newest_rows[1:]):
                raise ValueError("conflicting Work projections share the newest backend sequence")
            projections.append(self._work_from_row(matching_rows[0]))

        pending_queries: list[AgentQueryProjection] = []
        offset = 0
        while offset < len(query_rows):
            query_id = query_rows[offset]["query_id"]
            matching_rows = []
            while offset < len(query_rows) and query_rows[offset]["query_id"] == query_id:
                matching_rows.append(query_rows[offset])
                offset += 1
            self._validate_agent_query_tie(matching_rows)
            query = self._agent_query_from_row(matching_rows[0])
            if query.state is AgentQueryState.PENDING:
                pending_queries.append(query)

        projected_work_ids = {projection.work_id for projection in projections}
        claimed_query_ids: set[str] = set()
        reserved_work_ids: list[str] = []
        anonymous_capacity_reservations = 0
        cancel_claimed_work_ids: set[str] = set()
        command_recovery_pending = False
        for row in command_rows:
            operation = BackendOperation(row["operation"])
            state = CommandState(row["state"])
            payload = json.loads(row["payload_json"])
            command_recovery_pending = command_recovery_pending or state in {
                CommandState.STAGED,
                CommandState.DISPATCHING,
                CommandState.INCONCLUSIVE,
                CommandState.RECONCILING,
            }
            if operation in {BackendOperation.ANSWER_QUERY, BackendOperation.RESPOND_PERMISSION}:
                query_id = payload.get("query_id")
                if isinstance(query_id, str):
                    claimed_query_ids.add(query_id)
            elif operation is BackendOperation.SUBMIT:
                work_id = row["work_id"]
                if isinstance(work_id, str):
                    if work_id not in projected_work_ids and work_id not in reserved_work_ids:
                        reserved_work_ids.append(work_id)
                else:
                    anonymous_capacity_reservations += 1
            elif operation is BackendOperation.CANCEL:
                work_id = row["work_id"] or payload.get("work_id")
                if isinstance(work_id, str):
                    cancel_claimed_work_ids.add(work_id)
        return SessionControlSnapshot(
            session_id=session_id,
            applied_sequence=applied_sequence,
            control_revision=control_revision,
            works=tuple(projections),
            pending_queries=tuple(pending_queries),
            claimed_query_ids=frozenset(claimed_query_ids),
            reserved_work_ids=tuple(reserved_work_ids),
            anonymous_capacity_reservations=anonymous_capacity_reservations,
            cancel_claimed_work_ids=frozenset(cancel_claimed_work_ids),
            command_recovery_pending=command_recovery_pending,
        )

    @staticmethod
    def _work_from_row(row: sqlite3.Row) -> WorkProjection:
        """Materialize one normalized Work projection from SQLite."""
        return WorkProjection(
            work_id=row["work_id"],
            session_id=row["session_id"],
            attachment_id=row["attachment_id"],
            state=WorkState(row["state"]),
            sequence=int(row["sequence"]),
            summary=row["summary"],
            raw_state=row["raw_state"],
            agent_target=row["agent_target"],
            updated_at=_datetime(row["updated_at"]),
        )

    @staticmethod
    def _work_payload_from_row(row: sqlite3.Row) -> tuple[str, str, str | None, str | None]:
        """Return the replay-significant payload of a stored Work projection."""
        return (row["state"], row["summary"], row["raw_state"], row["agent_target"])

    @staticmethod
    def _work_payload_from_projection(projection: WorkProjection) -> tuple[str, str, str | None, str | None]:
        """Return the replay-significant payload of an incoming Work projection."""
        return (projection.state.value, projection.summary, projection.raw_state, projection.agent_target)

    def save_agent_query_projection(self, projection: AgentQueryProjection) -> bool:
        """Persist backend query state without claiming that a response resolved it."""
        with self._lock, self._connection:
            attachment_row = self._connection.execute(
                """
                SELECT session_id FROM agent_query_projections
                WHERE attachment_id = ? AND query_id = ?
                """,
                (projection.attachment_id, projection.query_id),
            ).fetchone()
            if attachment_row is not None and attachment_row["session_id"] != projection.session_id:
                raise ValueError("AgentQuery projection cannot move between voice sessions")

            rows = self._connection.execute(
                """
                SELECT * FROM agent_query_projections
                WHERE session_id = ? AND query_id = ?
                ORDER BY sequence DESC, updated_at DESC
                """,
                (projection.session_id, projection.query_id),
            ).fetchall()
            if rows:
                newest_sequence = int(rows[0]["sequence"])
                newest_rows = [row for row in rows if int(row["sequence"]) == newest_sequence]
                newest_payload = self._agent_query_payload_from_row(newest_rows[0])
                if any(self._agent_query_payload_from_row(row) != newest_payload for row in newest_rows[1:]):
                    raise ValueError("conflicting AgentQuery projections share the newest backend sequence")

                incoming_payload = self._agent_query_payload_from_projection(projection)
                if projection.sequence < newest_sequence:
                    return False
                if projection.sequence == newest_sequence:
                    if incoming_payload != newest_payload:
                        raise ValueError("conflicting AgentQuery projections share the newest backend sequence")
                    return False

                newest = self._agent_query_from_row(rows[0])
                if projection.work_id != newest.work_id or projection.kind is not newest.kind:
                    raise ValueError("AgentQuery identity cannot move between Work or query kinds")
                if newest.state.terminal:
                    if incoming_payload != newest_payload:
                        raise ValueError("terminal AgentQuery outcome is immutable")
                    return False

            self._connection.execute(
                """
                INSERT INTO agent_query_projections (
                    attachment_id, query_id, session_id, work_id, kind, state,
                    blocking, sequence, prompt, raw_state, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(attachment_id, query_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    work_id = excluded.work_id,
                    kind = excluded.kind,
                    state = excluded.state,
                    blocking = excluded.blocking,
                    sequence = excluded.sequence,
                    prompt = excluded.prompt,
                    raw_state = excluded.raw_state,
                    updated_at = excluded.updated_at
                """,
                (
                    projection.attachment_id,
                    projection.query_id,
                    projection.session_id,
                    projection.work_id,
                    projection.kind.value,
                    projection.state.value,
                    int(projection.blocking),
                    projection.sequence,
                    projection.prompt,
                    projection.raw_state,
                    _timestamp(projection.updated_at),
                ),
            )
            self._increment_control_revision(projection.session_id)
            return True

    def get_session_agent_query_projection(
        self,
        session_id: str,
        query_id: str,
    ) -> AgentQueryProjection | None:
        """Load the newest session-scoped projection for one backend query."""
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM agent_query_projections
                WHERE session_id = ? AND query_id = ?
                ORDER BY sequence DESC, updated_at DESC
                """,
                (session_id, query_id),
            ).fetchall()
        if not rows:
            return None
        self._validate_agent_query_tie(rows)
        return self._agent_query_from_row(rows[0])

    def pending_agent_queries(self, session_id: str) -> tuple[AgentQueryProjection, ...]:
        """Load only the latest pending state of each session-scoped query."""
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM agent_query_projections
                WHERE session_id = ?
                ORDER BY query_id ASC, sequence DESC, updated_at DESC
                """,
                (session_id,),
            ).fetchall()

        pending: list[AgentQueryProjection] = []
        offset = 0
        while offset < len(rows):
            query_id = rows[offset]["query_id"]
            query_rows: list[sqlite3.Row] = []
            while offset < len(rows) and rows[offset]["query_id"] == query_id:
                query_rows.append(rows[offset])
                offset += 1
            self._validate_agent_query_tie(query_rows)
            projection = self._agent_query_from_row(query_rows[0])
            if projection.state is AgentQueryState.PENDING:
                pending.append(projection)
        return tuple(pending)

    @classmethod
    def _validate_agent_query_tie(cls, rows: list[sqlite3.Row]) -> None:
        """Reject two different payloads claiming the same newest query sequence."""
        newest_sequence = int(rows[0]["sequence"])
        newest_rows = [row for row in rows if int(row["sequence"]) == newest_sequence]
        newest_payload = cls._agent_query_payload_from_row(newest_rows[0])
        if any(cls._agent_query_payload_from_row(row) != newest_payload for row in newest_rows[1:]):
            raise ValueError("conflicting AgentQuery projections share the newest backend sequence")

    @staticmethod
    def _agent_query_payload_from_row(row: sqlite3.Row) -> tuple[str, str, str, bool, str, str | None]:
        """Return replay-significant stored AgentQuery content."""
        return (
            row["work_id"],
            row["kind"],
            row["state"],
            bool(row["blocking"]),
            row["prompt"],
            row["raw_state"],
        )

    @staticmethod
    def _agent_query_payload_from_projection(
        projection: AgentQueryProjection,
    ) -> tuple[str, str, str, bool, str, str | None]:
        """Return replay-significant incoming AgentQuery content."""
        return (
            projection.work_id,
            projection.kind.value,
            projection.state.value,
            projection.blocking,
            projection.prompt,
            projection.raw_state,
        )

    @staticmethod
    def _agent_query_from_row(row: sqlite3.Row) -> AgentQueryProjection:
        """Restore one persisted AgentQuery projection."""
        return AgentQueryProjection(
            query_id=row["query_id"],
            work_id=row["work_id"],
            session_id=row["session_id"],
            attachment_id=row["attachment_id"],
            kind=AgentQueryKind(row["kind"]),
            state=AgentQueryState(row["state"]),
            blocking=bool(row["blocking"]),
            sequence=int(row["sequence"]),
            prompt=row["prompt"],
            raw_state=row["raw_state"],
            updated_at=_datetime(row["updated_at"]),
        )

    def save_presentation(self, record: PresentationRecord) -> bool:
        """Persist a presentation idempotently across replayed backend events."""
        display_json = json.dumps(
            {
                "kind": record.display.kind,
                "title": record.display.title,
                "body": record.display.body,
                "data": dict(record.display.data),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        values = (
            record.presentation_id,
            record.session_id,
            record.attachment_id,
            record.work_id,
            record.result_id,
            record.sequence,
            display_json,
            record.speech_text,
            record.speech_route.value,
            int(record.priority),
            record.result_state.value,
            record.display_state.value,
            record.speech_state.value,
            _timestamp(record.created_at),
            _timestamp(record.updated_at),
        )
        with self._lock, self._connection:
            try:
                self._connection.execute(
                    """
                    INSERT INTO presentations (
                        presentation_id, session_id, attachment_id, work_id, result_id, sequence,
                        display_json, speech_text, speech_route, priority,
                        result_state, display_state, speech_state, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
            except sqlite3.IntegrityError:
                rows = self._connection.execute(
                    """
                    SELECT * FROM presentations
                    WHERE presentation_id = ? OR (
                        session_id = ? AND work_id = ? AND result_id = ? AND sequence = ?
                    )
                    """,
                    (
                        record.presentation_id,
                        record.session_id,
                        record.work_id,
                        record.result_id,
                        record.sequence,
                    ),
                ).fetchall()
                if len(rows) == 1 and self._presentation_from_row(rows[0]).has_same_immutable_evidence(record):
                    return False
                if rows:
                    raise ValueError("presentation identity conflicts with persisted result evidence") from None
                raise
            return True

    def update_presentation(self, record: PresentationRecord) -> None:
        """Persist independent result, display, and speech lifecycle changes."""
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE presentations SET
                    result_state = ?, display_state = ?, speech_state = ?, updated_at = ?
                WHERE presentation_id = ? AND session_id = ?
                """,
                (
                    record.result_state.value,
                    record.display_state.value,
                    record.speech_state.value,
                    _timestamp(record.updated_at),
                    record.presentation_id,
                    record.session_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(record.presentation_id)

    def get_presentation(self, presentation_id: str) -> PresentationRecord | None:
        """Load one persisted presentation."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM presentations WHERE presentation_id = ?",
                (presentation_id,),
            ).fetchone()
        return self._presentation_from_row(row) if row is not None else None

    def get_result_presentation(
        self,
        session_id: str,
        work_id: str,
        result_id: str,
        sequence: int,
    ) -> PresentationRecord | None:
        """Load one presentation by its attachment-independent result identity."""
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM presentations
                WHERE session_id = ? AND work_id = ? AND result_id = ? AND sequence = ?
                """,
                (session_id, work_id, result_id, sequence),
            ).fetchone()
        return self._presentation_from_row(row) if row is not None else None

    def pending_presentations(self, session_id: str) -> tuple[PresentationRecord, ...]:
        """Load pending or locally leased channel deliveries in scheduler order."""
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM presentations
                WHERE session_id = ? AND (
                    display_state = ? OR speech_state IN (?, ?, ?, ?)
                )
                ORDER BY priority ASC, sequence ASC, created_at ASC
                """,
                (
                    session_id,
                    DisplayState.READY.value,
                    SpeechState.QUEUED.value,
                    SpeechState.DEFERRED.value,
                    SpeechState.CLAIMED.value,
                    SpeechState.SPEAKING.value,
                ),
            ).fetchall()
        return tuple(self._presentation_from_row(row) for row in rows)

    def save_command(
        self,
        command: CommandRecord,
        *,
        expected_control_revision: int | None = None,
    ) -> bool:
        """Persist an outbox record and atomically reserve its AgentQuery."""
        if expected_control_revision is not None and (
            isinstance(expected_control_revision, bool) or expected_control_revision < 0
        ):
            raise ValueError("expected_control_revision must be a non-negative integer")
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                session = self._connection.execute(
                    "SELECT control_revision FROM session_bindings WHERE session_id = ?",
                    (command.session_id,),
                ).fetchone()
                if session is None:
                    raise KeyError(command.session_id)
                if (
                    expected_control_revision is not None
                    and int(session["control_revision"]) != expected_control_revision
                ):
                    raise StaleSessionControlError("session control state changed after tool projection")
                claim_identity = self._command_query_claim_identity(command)
                if claim_identity is not None:
                    backend_session_id, query_id = claim_identity
                    claimed = self._agent_query_command_claim_locked(backend_session_id, query_id)
                    if claimed is not None and claimed.command_id != command.command_id:
                        self._connection.rollback()
                        return False
                cursor = self._connection.execute(
                    """
                INSERT OR IGNORE INTO commands (
                    command_id, session_id, attachment_id, backend_session_id,
                    commit_id, work_id, reason_code,
                    operation, capability_revision, payload_json,
                    state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        command.command_id,
                        command.session_id,
                        command.attachment_id,
                        command.backend_session_id,
                        command.commit_id,
                        command.work_id,
                        command.reason_code,
                        command.operation.value,
                        command.capability_revision,
                        json.dumps(dict(command.payload), separators=(",", ":"), sort_keys=True),
                        command.state.value,
                        _timestamp(command.created_at),
                        _timestamp(command.updated_at),
                    ),
                )
                if cursor.rowcount == 1:
                    self._increment_control_revision(command.session_id)
                self._connection.commit()
                return cursor.rowcount == 1
            except Exception:
                self._connection.rollback()
                raise

    def update_command(self, command: CommandRecord) -> None:
        """Update the local admission outcome without changing Work state."""
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE commands SET work_id = ?, reason_code = ?, state = ?, updated_at = ?
                WHERE command_id = ? AND session_id = ?
                """,
                (
                    command.work_id,
                    command.reason_code,
                    command.state.value,
                    _timestamp(command.updated_at),
                    command.command_id,
                    command.session_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(command.command_id)
            self._increment_control_revision(command.session_id)

    def get_command(self, command_id: str) -> CommandRecord | None:
        """Load one command/outbox record by its idempotency identity."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM commands WHERE command_id = ?",
                (command_id,),
            ).fetchone()
        return self._command_from_row(row) if row is not None else None

    def get_agent_query_command_claim(
        self,
        backend_session_id: str,
        query_id: str,
    ) -> CommandRecord | None:
        """Load VoiceClaw's one non-rejected response claim for a query."""
        if not isinstance(backend_session_id, str) or not backend_session_id.strip():
            raise ValueError("backend_session_id is required for an AgentQuery command claim")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("query_id is required for an AgentQuery command claim")
        with self._lock:
            return self._agent_query_command_claim_locked(backend_session_id, query_id)

    @staticmethod
    def _command_query_claim_identity(command: CommandRecord) -> tuple[str, str] | None:
        """Return the durable query namespace claimed by one response command."""
        if command.operation not in {
            BackendOperation.ANSWER_QUERY,
            BackendOperation.RESPOND_PERMISSION,
        }:
            return None
        if command.backend_session_id is None:
            raise ValueError("AgentQuery response commands require a durable backend session identity")
        query_id = command.payload.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("AgentQuery response commands require query_id")
        return command.backend_session_id, query_id

    def _agent_query_command_claim_locked(
        self,
        backend_session_id: str,
        query_id: str,
    ) -> CommandRecord | None:
        """Find a query claim while the caller holds the connection lock."""
        rows = self._connection.execute(
            """
            SELECT * FROM commands
            WHERE backend_session_id = ?
              AND operation IN (?, ?)
              AND state != ?
            ORDER BY created_at ASC, command_id ASC
            """,
            (
                backend_session_id,
                BackendOperation.ANSWER_QUERY.value,
                BackendOperation.RESPOND_PERMISSION.value,
                CommandState.REJECTED.value,
            ),
        ).fetchall()
        claims = [
            self._command_from_row(row) for row in rows if json.loads(row["payload_json"]).get("query_id") == query_id
        ]
        if len(claims) > 1:
            raise ValueError("multiple VoiceClaw commands claim the same backend AgentQuery")
        return claims[0] if claims else None

    def unsettled_commands(self, session_id: str) -> tuple[CommandRecord, ...]:
        """Load commands whose acceptance outcome needs reconciliation."""
        placeholders = ",".join("?" for _ in _UNSETTLED_COMMAND_STATES)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT * FROM commands
                WHERE session_id = ? AND state IN ({placeholders})
                ORDER BY created_at ASC
                """,  # noqa: S608 - placeholders are generated locally, never from input
                (session_id, *_UNSETTLED_COMMAND_STATES),
            ).fetchall()
        return tuple(self._command_from_row(row) for row in rows)

    def _initialize(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 5000")
            if self.path != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
                self._connection.execute("PRAGMA synchronous = FULL")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                );

                CREATE TABLE IF NOT EXISTS session_bindings (
                    session_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    backend_profile TEXT NOT NULL,
                    attachment_id TEXT,
                    backend_session_id TEXT,
                    last_applied_sequence INTEGER NOT NULL CHECK(last_applied_sequence >= 0),
                    last_presented_sequence INTEGER NOT NULL CHECK(last_presented_sequence >= 0),
                    control_revision INTEGER NOT NULL DEFAULT 0 CHECK(control_revision >= 0),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    CHECK(last_presented_sequence <= last_applied_sequence)
                );

                CREATE TABLE IF NOT EXISTS work_projections (
                    attachment_id TEXT NOT NULL,
                    work_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    sequence INTEGER NOT NULL CHECK(sequence >= 0),
                    summary TEXT NOT NULL,
                    raw_state TEXT,
                    agent_target TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(attachment_id, work_id),
                    FOREIGN KEY(session_id) REFERENCES session_bindings(session_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS agent_query_projections (
                    attachment_id TEXT NOT NULL,
                    query_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    work_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    state TEXT NOT NULL,
                    blocking INTEGER NOT NULL CHECK(blocking IN (0, 1)),
                    sequence INTEGER NOT NULL CHECK(sequence >= 1),
                    prompt TEXT NOT NULL,
                    raw_state TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(attachment_id, query_id),
                    FOREIGN KEY(session_id) REFERENCES session_bindings(session_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS presentations (
                    presentation_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    attachment_id TEXT NOT NULL,
                    work_id TEXT NOT NULL,
                    result_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL CHECK(sequence >= 0),
                    display_json TEXT NOT NULL,
                    speech_text TEXT,
                    speech_route TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    result_state TEXT NOT NULL,
                    display_state TEXT NOT NULL,
                    speech_state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES session_bindings(session_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS commands (
                    command_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    attachment_id TEXT NOT NULL,
                    backend_session_id TEXT,
                    commit_id TEXT NOT NULL,
                    work_id TEXT,
                    reason_code TEXT,
                    operation TEXT NOT NULL,
                    capability_revision TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES session_bindings(session_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS presentations_pending_idx
                    ON presentations(session_id, display_state, speech_state, priority, sequence);
                CREATE INDEX IF NOT EXISTS commands_unsettled_idx
                    ON commands(session_id, state, created_at);
                CREATE INDEX IF NOT EXISTS agent_queries_session_state_idx
                    ON agent_query_projections(session_id, state, query_id, sequence);
                CREATE INDEX IF NOT EXISTS work_projections_session_sequence_idx
                    ON work_projections(session_id, work_id, sequence);
                """
            )
            existing = self._connection.execute("SELECT version FROM schema_version").fetchone()
            if existing is None:
                self._connection.execute("INSERT INTO schema_version(version) VALUES (?)", (_SCHEMA_VERSION,))
            else:
                version = int(existing["version"])
                if version == 1:
                    self._migrate_v1_to_v2()
                    version = 2
                if version == 2:
                    self._migrate_v2_to_v3()
                    version = 3
                if version == 3:
                    self._migrate_v3_to_v4()
                    version = 4
                if version == 4:
                    self._migrate_v4_to_v5()
                    version = 5
                if version == 5:
                    self._migrate_v5_to_v6()
                    version = 6
                if version != _SCHEMA_VERSION:
                    raise RuntimeError(f"unsupported VoiceClaw state schema version: {existing['version']}")
            self._connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS presentations_result_identity_idx
                ON presentations(session_id, work_id, result_id, sequence)
                """
            )
            self._connection.execute(
                "UPDATE presentations SET speech_state = ? WHERE speech_state IN (?, ?)",
                (SpeechState.DEFERRED.value, SpeechState.CLAIMED.value, SpeechState.SPEAKING.value),
            )
            self._connection.commit()

    def _migrate_v4_to_v5(self) -> None:
        """Persist optional backend-authored specialized-agent ownership."""
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            columns = {
                row["name"] for row in self._connection.execute("PRAGMA table_info(work_projections)").fetchall()
            }
            if "agent_target" not in columns:
                self._connection.execute("ALTER TABLE work_projections ADD COLUMN agent_target TEXT")
            cursor = self._connection.execute(
                "UPDATE schema_version SET version = ? WHERE version = 4",
                (5,),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("VoiceClaw state schema changed during migration")
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def _migrate_v5_to_v6(self) -> None:
        """Add the optimistic local admission revision atomically."""
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            columns = {
                row["name"] for row in self._connection.execute("PRAGMA table_info(session_bindings)").fetchall()
            }
            if "control_revision" not in columns:
                self._connection.execute(
                    """
                    ALTER TABLE session_bindings
                    ADD COLUMN control_revision INTEGER NOT NULL DEFAULT 0
                    """
                )
            cursor = self._connection.execute(
                "UPDATE schema_version SET version = ? WHERE version = 5",
                (6,),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("VoiceClaw state schema changed during migration")
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def _increment_control_revision(self, session_id: str) -> None:
        """Invalidate stale tool projections inside the caller's transaction."""
        cursor = self._connection.execute(
            """
            UPDATE session_bindings
            SET control_revision = control_revision + 1
            WHERE session_id = ?
            """,
            (session_id,),
        )
        if cursor.rowcount != 1:
            raise KeyError(session_id)

    def _migrate_v1_to_v2(self) -> None:
        """Add durable command correlation and receipt evidence atomically."""
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            columns = {row["name"] for row in self._connection.execute("PRAGMA table_info(commands)").fetchall()}
            if "backend_session_id" not in columns:
                self._connection.execute("ALTER TABLE commands ADD COLUMN backend_session_id TEXT")
            if "reason_code" not in columns:
                self._connection.execute("ALTER TABLE commands ADD COLUMN reason_code TEXT")
            cursor = self._connection.execute(
                "UPDATE schema_version SET version = ? WHERE version = 1",
                (2,),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("VoiceClaw state schema changed during migration")
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def _migrate_v2_to_v3(self) -> None:
        """Add the backend-authoritative AgentQuery projection atomically."""
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_query_projections (
                    attachment_id TEXT NOT NULL,
                    query_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    work_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    state TEXT NOT NULL,
                    blocking INTEGER NOT NULL CHECK(blocking IN (0, 1)),
                    sequence INTEGER NOT NULL CHECK(sequence >= 1),
                    prompt TEXT NOT NULL,
                    raw_state TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(attachment_id, query_id),
                    FOREIGN KEY(session_id) REFERENCES session_bindings(session_id) ON DELETE CASCADE
                )
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS agent_queries_session_state_idx
                ON agent_query_projections(session_id, state, query_id, sequence)
                """
            )
            cursor = self._connection.execute(
                "UPDATE schema_version SET version = ? WHERE version = 2",
                (3,),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("VoiceClaw state schema changed during migration")
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def _migrate_v3_to_v4(self) -> None:
        """Collapse identical attachment-era replay rows transactionally."""
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            archive_exists = self._connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'presentations_v3_archive'"
            ).fetchone()
            if archive_exists is not None:
                raise RuntimeError("VoiceClaw presentation migration archive already exists")
            self._connection.execute("ALTER TABLE presentations RENAME TO presentations_v3_archive")
            self._connection.execute(
                """
                CREATE TABLE presentations (
                    presentation_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    attachment_id TEXT NOT NULL,
                    work_id TEXT NOT NULL,
                    result_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL CHECK(sequence >= 0),
                    display_json TEXT NOT NULL,
                    speech_text TEXT,
                    speech_route TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    result_state TEXT NOT NULL,
                    display_state TEXT NOT NULL,
                    speech_state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES session_bindings(session_id) ON DELETE CASCADE
                )
                """
            )

            grouped: dict[tuple[str, str, str, int], list[PresentationRecord]] = {}
            for row in self._connection.execute("SELECT * FROM presentations_v3_archive").fetchall():
                record = self._presentation_from_row(row)
                grouped.setdefault(record.immutable_identity, []).append(record)

            for records in grouped.values():
                survivor = min(records, key=lambda item: (item.created_at, item.presentation_id))
                immutable_evidence = self._migration_presentation_evidence(survivor)
                if any(self._migration_presentation_evidence(record) != immutable_evidence for record in records):
                    raise RuntimeError("cannot migrate conflicting immutable evidence for one backend result identity")
                result_state, display_state, speech_state = self._merge_presentation_lifecycle(records)
                self._connection.execute(
                    """
                    INSERT INTO presentations (
                        presentation_id, session_id, attachment_id, work_id, result_id, sequence,
                        display_json, speech_text, speech_route, priority,
                        result_state, display_state, speech_state, created_at, updated_at
                    )
                    SELECT
                        presentation_id, session_id, attachment_id, work_id, result_id, sequence,
                        display_json, speech_text, speech_route, priority,
                        ?, ?, ?, ?, ?
                    FROM presentations_v3_archive WHERE presentation_id = ?
                    """,
                    (
                        result_state.value,
                        display_state.value,
                        speech_state.value,
                        _timestamp(min(record.created_at for record in records)),
                        _timestamp(max(record.updated_at for record in records)),
                        survivor.presentation_id,
                    ),
                )
            self._connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS presentations_result_identity_idx
                ON presentations(session_id, work_id, result_id, sequence)
                """
            )
            # The archive is only a transactional source table. Keeping it
            # would retain duplicate conversational content indefinitely and
            # leave the old pending-index name attached to stale rows.
            self._connection.execute("DROP TABLE presentations_v3_archive")
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS presentations_pending_idx
                ON presentations(session_id, display_state, speech_state, priority, sequence)
                """
            )
            cursor = self._connection.execute(
                "UPDATE schema_version SET version = ? WHERE version = 3",
                (4,),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("VoiceClaw state schema changed during migration")
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    @staticmethod
    def _migration_presentation_evidence(record: PresentationRecord) -> tuple[object, ...]:
        """Return content that cannot change when the same result is replayed.

        Version 3 allowed both the attachment and local presentation identity
        to change after reconnect, so neither participates in this migration
        comparison. The canonical result key already includes session, Work,
        result, and sequence.
        """
        return (
            record.display.kind,
            record.display.title,
            record.display.body,
            dict(record.display.data),
            record.speech_text,
            record.speech_route,
            record.priority,
        )

    @staticmethod
    def _merge_presentation_lifecycle(
        records: list[PresentationRecord],
    ) -> tuple[ResultState, DisplayState, SpeechState]:
        """Merge replay delivery evidence without re-announcing delivered content."""
        display_states = {record.display_state for record in records}
        if DisplayState.DELIVERED in display_states:
            display_state = DisplayState.DELIVERED
        elif DisplayState.FAILED in display_states:
            display_state = DisplayState.FAILED
        elif DisplayState.READY in display_states:
            display_state = DisplayState.READY
        else:
            display_state = DisplayState.NOT_REQUESTED

        speech_states = {record.speech_state for record in records}
        if SpeechState.HEARD in speech_states:
            speech_state = SpeechState.HEARD
        elif SpeechState.INTERRUPTED in speech_states:
            speech_state = SpeechState.INTERRUPTED
        elif SpeechState.EXPIRED in speech_states:
            speech_state = SpeechState.EXPIRED
        elif SpeechState.FAILED in speech_states:
            speech_state = SpeechState.FAILED
        elif speech_states.intersection(
            {SpeechState.QUEUED, SpeechState.DEFERRED, SpeechState.CLAIMED, SpeechState.SPEAKING}
        ):
            speech_state = SpeechState.DEFERRED
        else:
            speech_state = SpeechState.NOT_REQUESTED

        result_states = {record.result_state for record in records}
        if ResultState.ACKNOWLEDGED in result_states or (
            display_state is DisplayState.DELIVERED and speech_state in {SpeechState.HEARD, SpeechState.NOT_REQUESTED}
        ):
            result_state = ResultState.ACKNOWLEDGED
        elif ResultState.EXPIRED in result_states:
            result_state = ResultState.EXPIRED
        elif ResultState.AVAILABLE in result_states:
            result_state = ResultState.AVAILABLE
        elif ResultState.NORMALIZED in result_states:
            result_state = ResultState.NORMALIZED
        else:
            result_state = ResultState.RECEIVED
        return result_state, display_state, speech_state

    @staticmethod
    def _session_from_row(row: sqlite3.Row) -> SessionBinding:
        return SessionBinding(
            session_id=row["session_id"],
            conversation_id=row["conversation_id"],
            backend_profile=row["backend_profile"],
            attachment_id=row["attachment_id"],
            backend_session_id=row["backend_session_id"],
            last_applied_sequence=int(row["last_applied_sequence"]),
            last_presented_sequence=int(row["last_presented_sequence"]),
            created_at=_datetime(row["created_at"]),
            updated_at=_datetime(row["updated_at"]),
        )

    @staticmethod
    def _presentation_from_row(row: sqlite3.Row) -> PresentationRecord:
        display: dict[str, Any] = json.loads(row["display_json"])
        return PresentationRecord(
            presentation_id=row["presentation_id"],
            session_id=row["session_id"],
            attachment_id=row["attachment_id"],
            work_id=row["work_id"],
            result_id=row["result_id"],
            sequence=int(row["sequence"]),
            display=DisplayPayload(
                kind=display["kind"],
                title=display["title"],
                body=display["body"],
                data=display["data"],
            ),
            speech_text=row["speech_text"],
            speech_route=SpeechRoute(row["speech_route"]),
            priority=PresentationPriority(int(row["priority"])),
            result_state=ResultState(row["result_state"]),
            display_state=DisplayState(row["display_state"]),
            speech_state=SpeechState(row["speech_state"]),
            created_at=_datetime(row["created_at"]),
            updated_at=_datetime(row["updated_at"]),
        )

    @staticmethod
    def _command_from_row(row: sqlite3.Row) -> CommandRecord:
        return CommandRecord(
            command_id=row["command_id"],
            session_id=row["session_id"],
            attachment_id=row["attachment_id"],
            backend_session_id=row["backend_session_id"],
            commit_id=row["commit_id"],
            work_id=row["work_id"],
            reason_code=row["reason_code"],
            operation=BackendOperation(row["operation"]),
            capability_revision=row["capability_revision"],
            payload=json.loads(row["payload_json"]),
            state=CommandState(row["state"]),
            created_at=_datetime(row["created_at"]),
            updated_at=_datetime(row["updated_at"]),
        )
