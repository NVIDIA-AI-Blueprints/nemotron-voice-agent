# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

import os
import sqlite3
import stat
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from voiceclaw.adapters.state import SqliteStateStore
from voiceclaw.domain.models import (
    AgentQueryKind,
    AgentQueryProjection,
    AgentQueryState,
    BackendOperation,
    CommandRecord,
    CommandState,
    DisplayPayload,
    DisplayState,
    PresentationRecord,
    ResultState,
    SessionBinding,
    SpeechState,
    WorkProjection,
    WorkResult,
    WorkState,
    utc_now,
)
from voiceclaw.ports.state import StaleSessionControlError


def _binding() -> SessionBinding:
    return SessionBinding(
        session_id="session-a",
        conversation_id="conversation-a",
        backend_profile="default",
        attachment_id="attachment-a",
    )


def _presentation() -> PresentationRecord:
    return PresentationRecord.from_result(
        WorkResult(
            presentation_id="presentation-a",
            session_id="session-a",
            attachment_id="attachment-a",
            work_id="work-a",
            result_id="result-a",
            sequence=4,
            display=DisplayPayload(kind="work.result", title="Ready", data={"answer": 4}),
            speech_text="The result is ready.",
        )
    )


def test_state_directory_and_live_sqlite_artifacts_are_owner_only_with_permissive_umask(tmp_path) -> None:
    state_path = tmp_path / "private-state" / "state.db"
    previous_umask = os.umask(0)
    try:
        with SqliteStateStore(state_path):
            assert stat.S_IMODE(state_path.parent.stat().st_mode) == 0o700
            for artifact in (state_path, Path(f"{state_path}-wal"), Path(f"{state_path}-shm")):
                assert artifact.is_file()
                assert stat.S_IMODE(artifact.stat().st_mode) == 0o600
    finally:
        os.umask(previous_umask)


def test_existing_database_is_hardened_before_sqlite_opens_it(tmp_path) -> None:
    state_path = tmp_path / "state.db"
    state_path.touch(mode=0o644)
    state_path.chmod(0o644)

    with SqliteStateStore(state_path):
        assert stat.S_IMODE(state_path.stat().st_mode) == 0o600


def test_state_store_rejects_a_shared_state_directory(tmp_path) -> None:
    state_directory = tmp_path / "shared-state"
    state_directory.mkdir(mode=0o750)

    with pytest.raises(PermissionError, match="owner-only permissions"):
        SqliteStateStore(state_directory / "state.db")


def test_state_store_rejects_a_symbolic_link_database(tmp_path) -> None:
    target = tmp_path / "target.db"
    target.touch(mode=0o600)
    link = tmp_path / "state.db"
    link.symlink_to(target)

    with pytest.raises(PermissionError, match="opened securely"):
        SqliteStateStore(link)


def test_session_cursors_survive_reopen_and_never_regress(tmp_path) -> None:
    path = tmp_path / "state.db"
    with SqliteStateStore(path) as store:
        store.save_session(_binding())
        assert store.advance_cursor("session-a", 7) == 7
        assert store.advance_cursor("session-a", 5) == 7
        assert store.advance_presented_cursor("session-a", 4) == 4
        assert store.advance_presented_cursor("session-a", 3) == 4
        with pytest.raises(ValueError, match="beyond the applied"):
            store.advance_presented_cursor("session-a", 8)

    with SqliteStateStore(path) as reopened:
        binding = reopened.get_session("session-a")
        assert binding is not None
        assert binding.last_applied_sequence == 7
        assert binding.last_presented_sequence == 4


def test_new_attachment_starts_a_new_event_cursor_epoch(tmp_path) -> None:
    with SqliteStateStore(tmp_path / "state.db") as store:
        store.save_session(_binding())
        store.advance_cursor("session-a", 7)
        store.save_session(
            replace(
                _binding(),
                attachment_id="attachment-b",
                backend_session_id="backend-session-b",
                last_applied_sequence=0,
                last_presented_sequence=0,
            )
        )

        binding = store.get_session("session-a")
        assert binding is not None
        assert binding.attachment_id == "attachment-b"
        assert binding.backend_session_id == "backend-session-b"
        assert binding.last_applied_sequence == 0
        assert binding.last_presented_sequence == 0


def test_only_response_only_session_mappings_can_be_discarded(tmp_path) -> None:
    with SqliteStateStore(tmp_path / "state.db") as store:
        store.save_session(
            SessionBinding(
                session_id="ephemeral-session",
                conversation_id="ephemeral-conversation",
                backend_profile="response-only",
            )
        )
        store.discard_ephemeral_session("ephemeral-session")
        assert store.get_session("ephemeral-session") is None

        store.save_session(_binding())
        with pytest.raises(ValueError, match="durable backend or Work evidence"):
            store.discard_ephemeral_session("session-a")
        assert store.get_session("session-a") is not None


def test_presentation_is_idempotent_and_channel_states_persist(tmp_path) -> None:
    path = tmp_path / "state.db"
    with SqliteStateStore(path) as store:
        store.save_session(_binding())
        presentation = _presentation()
        assert store.save_presentation(presentation) is True
        assert store.save_presentation(presentation) is False
        speaking = replace(presentation, speech_state=SpeechState.SPEAKING, updated_at=utc_now())
        store.update_presentation(speaking)

    with SqliteStateStore(path) as reopened:
        restored = reopened.get_presentation("presentation-a")
        assert restored is not None
        assert restored.speech_state is SpeechState.DEFERRED
        assert [item.presentation_id for item in reopened.pending_presentations("session-a")] == ["presentation-a"]


def test_presentation_replay_across_attachment_epochs_keeps_first_evidence(tmp_path) -> None:
    with SqliteStateStore(tmp_path / "state.db") as store:
        store.save_session(_binding())
        presentation = _presentation()
        assert store.save_presentation(presentation) is True

        store.save_session(replace(_binding(), attachment_id="attachment-b"))
        replay = replace(
            presentation,
            attachment_id="attachment-b",
            sequence=presentation.sequence,
            updated_at=utc_now(),
        )

        assert store.save_presentation(replay) is False
        assert store.get_presentation("presentation-a") == presentation
        assert store.get_result_presentation("session-a", "work-a", "result-a", 4) == presentation

        with pytest.raises(ValueError, match="presentation identity conflicts"):
            store.save_presentation(
                replace(
                    replay,
                    display=DisplayPayload(kind="work.result", title="Conflicting replay"),
                )
            )


def test_presentation_conflicts_are_not_treated_as_idempotent_replay(tmp_path) -> None:
    with SqliteStateStore(tmp_path / "state.db") as store:
        store.save_session(_binding())
        presentation = _presentation()
        assert store.save_presentation(presentation) is True

        with pytest.raises(ValueError, match="presentation identity conflicts"):
            store.save_presentation(
                replace(
                    presentation,
                    display=DisplayPayload(kind="work.result", title="Conflicting payload"),
                )
            )
        assert store.save_presentation(replace(presentation, presentation_id="presentation-replayed")) is False

        assert store.get_presentation("presentation-a") == presentation
        assert store.get_presentation("presentation-replayed") is None


def test_terminal_work_projection_is_immutable(tmp_path) -> None:
    with SqliteStateStore(tmp_path / "state.db") as store:
        store.save_session(_binding())
        succeeded = WorkProjection(
            work_id="work-a",
            session_id="session-a",
            attachment_id="attachment-a",
            state=WorkState.SUCCEEDED,
            sequence=10,
        )
        assert store.save_work_projection(succeeded) is True
        assert store.save_work_projection(replace(succeeded, sequence=10)) is False
        with pytest.raises(ValueError, match="terminal Work outcome"):
            store.save_work_projection(
                replace(succeeded, state=WorkState.CANCELLED, sequence=11),
            )

        assert store.save_work_projection(replace(succeeded, state=WorkState.CANCELLED, sequence=9)) is False


def test_backend_work_ids_are_scoped_by_voice_session(tmp_path) -> None:
    with SqliteStateStore(tmp_path / "state.db") as store:
        store.save_session(_binding())
        store.save_session(
            SessionBinding(
                session_id="session-b",
                conversation_id="conversation-b",
                backend_profile="default",
                attachment_id="attachment-b",
            )
        )
        first = WorkProjection(
            work_id="1",
            session_id="session-a",
            attachment_id="attachment-a",
            state=WorkState.RUNNING,
            sequence=1,
        )
        second = WorkProjection(
            work_id="1",
            session_id="session-b",
            attachment_id="attachment-b",
            state=WorkState.SUCCEEDED,
            sequence=1,
        )

        assert store.save_work_projection(first) is True
        assert store.save_work_projection(second) is True
        assert store.get_work_projection("attachment-a", "1") == first
        assert store.get_work_projection("attachment-b", "1") == second
        assert store.get_session_work_projection("session-a", "1") == first
        assert store.get_session_work_projection("session-b", "1") == second
        assert store.get_session_work_projection("session-a", "missing") is None


def test_identical_replay_tie_across_attachments_is_accepted(tmp_path) -> None:
    with SqliteStateStore(tmp_path / "state.db") as store:
        store.save_session(_binding())
        first = WorkProjection(
            work_id="backend-work-a",
            session_id="session-a",
            attachment_id="attachment-a",
            state=WorkState.RUNNING,
            sequence=7,
            summary="Still working",
            raw_state="executing",
        )
        assert store.save_work_projection(first) is True

        store.save_session(replace(_binding(), attachment_id="attachment-b"))
        replay = replace(first, attachment_id="attachment-b", updated_at=utc_now())
        assert store.save_work_projection(replay) is False
        assert store.get_session_work_projection("session-a", "backend-work-a") == first


def test_conflicting_replay_tie_across_attachments_is_rejected(tmp_path) -> None:
    with SqliteStateStore(tmp_path / "state.db") as store:
        store.save_session(_binding())
        first = WorkProjection(
            work_id="backend-work-a",
            session_id="session-a",
            attachment_id="attachment-a",
            state=WorkState.RUNNING,
            sequence=7,
            summary="Still working",
        )
        assert store.save_work_projection(first) is True

        store.save_session(replace(_binding(), attachment_id="attachment-b"))
        with pytest.raises(ValueError, match="conflicting Work projections"):
            store.save_work_projection(
                replace(
                    first,
                    attachment_id="attachment-b",
                    summary="A different event at the same sequence",
                )
            )


def test_terminal_work_outcome_is_immutable_across_new_attachment(tmp_path) -> None:
    with SqliteStateStore(tmp_path / "state.db") as store:
        store.save_session(_binding())
        succeeded = WorkProjection(
            work_id="backend-work-a",
            session_id="session-a",
            attachment_id="attachment-a",
            state=WorkState.SUCCEEDED,
            sequence=10,
            summary="Finished",
            raw_state="completed",
        )
        assert store.save_work_projection(succeeded) is True

        store.save_session(replace(_binding(), attachment_id="attachment-b"))
        assert (
            store.save_work_projection(
                replace(succeeded, attachment_id="attachment-b", sequence=11, updated_at=utc_now())
            )
            is False
        )
        with pytest.raises(ValueError, match="terminal Work outcome"):
            store.save_work_projection(
                replace(
                    succeeded,
                    attachment_id="attachment-b",
                    state=WorkState.CANCELLED,
                    sequence=11,
                    summary="Cancelled",
                    raw_state="cancelled",
                    updated_at=utc_now(),
                )
            )

        assert store.get_session_work_projection("session-a", "backend-work-a") == succeeded


def test_session_projection_reader_accepts_identical_ties_and_rejects_conflicts(tmp_path) -> None:
    with SqliteStateStore(tmp_path / "state.db") as store:
        store.save_session(_binding())
        projection = WorkProjection(
            work_id="backend-work-a",
            session_id="session-a",
            attachment_id="attachment-a",
            state=WorkState.RUNNING,
            sequence=4,
            summary="Working",
            raw_state="executing",
        )
        assert store.save_work_projection(projection) is True

        # Insert replay rows directly to exercise recovery from databases written by older builds.
        store._connection.execute(  # noqa: SLF001
            """
            INSERT INTO work_projections (
                attachment_id, work_id, session_id, state, sequence, summary, raw_state, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "attachment-b",
                projection.work_id,
                projection.session_id,
                projection.state.value,
                projection.sequence,
                projection.summary,
                projection.raw_state,
                projection.updated_at.isoformat(),
            ),
        )
        assert store.get_session_work_projection("session-a", "backend-work-a") is not None

        store._connection.execute(  # noqa: SLF001
            "UPDATE work_projections SET summary = ? WHERE attachment_id = ? AND work_id = ?",
            ("Conflicting replay", "attachment-b", projection.work_id),
        )
        with pytest.raises(ValueError, match="conflicting Work projections"):
            store.get_session_work_projection("session-a", "backend-work-a")


def test_agent_query_projection_is_session_scoped_and_backend_authoritative(tmp_path) -> None:
    with SqliteStateStore(tmp_path / "state.db") as store:
        store.save_session(_binding())
        store.save_session(
            SessionBinding(
                session_id="session-b",
                conversation_id="conversation-b",
                backend_profile="default",
                attachment_id="attachment-b",
            )
        )
        pending = AgentQueryProjection(
            query_id="query-a",
            work_id="work-a",
            session_id="session-a",
            attachment_id="attachment-a",
            kind=AgentQueryKind.INFORMATION,
            state=AgentQueryState.PENDING,
            blocking=True,
            sequence=1,
            prompt="Which destination?",
        )
        other_session = replace(
            pending,
            work_id="work-b",
            session_id="session-b",
            attachment_id="attachment-b",
        )

        assert store.save_agent_query_projection(pending) is True
        assert store.save_agent_query_projection(other_session) is True
        assert store.get_session_agent_query_projection("session-a", "query-a") == pending
        assert store.get_session_agent_query_projection("session-b", "query-a") == other_session
        assert store.pending_agent_queries("session-a") == (pending,)

        forwarded = replace(
            pending,
            state=AgentQueryState.RESPONSE_FORWARDED,
            sequence=2,
            updated_at=utc_now(),
        )
        assert store.save_agent_query_projection(forwarded) is True
        assert forwarded.state.terminal is False
        assert store.pending_agent_queries("session-a") == ()

        resolved = replace(
            forwarded,
            state=AgentQueryState.RESOLVED,
            sequence=3,
            updated_at=utc_now(),
        )
        assert store.save_agent_query_projection(resolved) is True
        with pytest.raises(ValueError, match="terminal AgentQuery outcome"):
            store.save_agent_query_projection(
                replace(
                    resolved,
                    state=AgentQueryState.PENDING,
                    sequence=4,
                    updated_at=utc_now(),
                )
            )


def test_agent_query_replay_conflicts_fail_closed_across_attachments(tmp_path) -> None:
    with SqliteStateStore(tmp_path / "state.db") as store:
        store.save_session(_binding())
        projection = AgentQueryProjection(
            query_id="query-a",
            work_id="work-a",
            session_id="session-a",
            attachment_id="attachment-a",
            kind=AgentQueryKind.PERMISSION,
            state=AgentQueryState.PENDING,
            blocking=True,
            sequence=7,
            prompt="Allow it?",
        )
        assert store.save_agent_query_projection(projection) is True
        store.save_session(replace(_binding(), attachment_id="attachment-b"))

        assert (
            store.save_agent_query_projection(replace(projection, attachment_id="attachment-b", updated_at=utc_now()))
            is False
        )
        with pytest.raises(ValueError, match="conflicting AgentQuery projections"):
            store.save_agent_query_projection(
                replace(
                    projection,
                    attachment_id="attachment-b",
                    prompt="A conflicting prompt",
                    updated_at=utc_now(),
                )
            )


def test_command_outbox_keeps_idempotency_identity_until_settled(tmp_path) -> None:
    with SqliteStateStore(tmp_path / "state.db") as store:
        store.save_session(_binding())
        command = CommandRecord(
            command_id="command-a",
            session_id="session-a",
            attachment_id="attachment-a",
            backend_session_id="backend-session-a",
            commit_id="commit-a",
            operation=BackendOperation.SUBMIT,
            capability_revision="cap-v1",
            payload={"request": "Do the work"},
        )
        assert store.save_command(command) is True
        assert store.save_command(command) is False
        assert [item.command_id for item in store.unsettled_commands("session-a")] == ["command-a"]
        restored = store.get_command("command-a")
        assert restored is not None
        assert restored.backend_session_id == "backend-session-a"

        accepted = replace(
            command,
            work_id="work-a",
            state=CommandState.ACCEPTED,
            reason_code="accepted_after_reconcile",
            updated_at=utc_now(),
        )
        store.update_command(accepted)
        assert store.unsettled_commands("session-a") == ()
        restored = store.get_command("command-a")
        assert restored is not None
        assert restored.reason_code == "accepted_after_reconcile"


def test_v1_database_migrates_to_v6_without_losing_outbox(tmp_path) -> None:
    path = tmp_path / "state.db"
    timestamp = utc_now().isoformat()
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
            INSERT INTO schema_version(version) VALUES (1);

            CREATE TABLE session_bindings (
                session_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                backend_profile TEXT NOT NULL,
                attachment_id TEXT,
                backend_session_id TEXT,
                last_applied_sequence INTEGER NOT NULL CHECK(last_applied_sequence >= 0),
                last_presented_sequence INTEGER NOT NULL CHECK(last_presented_sequence >= 0),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                CHECK(last_presented_sequence <= last_applied_sequence)
            );

            CREATE TABLE commands (
                command_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                attachment_id TEXT NOT NULL,
                commit_id TEXT NOT NULL,
                work_id TEXT,
                operation TEXT NOT NULL,
                capability_revision TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                state TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES session_bindings(session_id) ON DELETE CASCADE
            );
            """
        )
        connection.execute(
            """
            INSERT INTO session_bindings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "session-a",
                "conversation-a",
                "default",
                "attachment-old",
                "backend-session-a",
                0,
                0,
                timestamp,
                timestamp,
            ),
        )
        connection.execute(
            """
            INSERT INTO commands VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "command-a",
                "session-a",
                "attachment-old",
                "commit-a",
                None,
                BackendOperation.SUBMIT.value,
                "cap-v1",
                '{"instruction":"prepare the requested work"}',
                CommandState.INCONCLUSIVE.value,
                timestamp,
                timestamp,
            ),
        )
        connection.commit()
    finally:
        connection.close()

    with SqliteStateStore(path) as migrated:
        restored = migrated.get_command("command-a")
        assert restored is not None
        assert restored.attachment_id == "attachment-old"
        assert restored.backend_session_id is None
        assert restored.reason_code is None

    verification = sqlite3.connect(path)
    try:
        assert verification.execute("SELECT version FROM schema_version").fetchone() == (6,)
        columns = {row[1] for row in verification.execute("PRAGMA table_info(session_bindings)").fetchall()}
        assert "control_revision" in columns
        columns = {row[1] for row in verification.execute("PRAGMA table_info(commands)")}
        assert {"backend_session_id", "reason_code"} <= columns
        query_columns = {row[1] for row in verification.execute("PRAGMA table_info(agent_query_projections)")}
        assert {"query_id", "work_id", "kind", "state", "blocking", "sequence"} <= query_columns
    finally:
        verification.close()


def test_v2_database_migrates_agent_query_projection_schema(tmp_path) -> None:
    path = tmp_path / "state.db"
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
            INSERT INTO schema_version(version) VALUES (2);
            """
        )
        connection.commit()
    finally:
        connection.close()

    with SqliteStateStore(path):
        pass

    verification = sqlite3.connect(path)
    try:
        assert verification.execute("SELECT version FROM schema_version").fetchone() == (6,)
        assert verification.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'agent_query_projections'"
        ).fetchone() == ("agent_query_projections",)
    finally:
        verification.close()


def test_v3_database_migrates_to_attachment_independent_result_identity(tmp_path) -> None:
    path = tmp_path / "state.db"
    timestamp = utc_now().isoformat()
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
            INSERT INTO schema_version(version) VALUES (3);

            CREATE TABLE session_bindings (
                session_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                backend_profile TEXT NOT NULL,
                attachment_id TEXT,
                backend_session_id TEXT,
                last_applied_sequence INTEGER NOT NULL CHECK(last_applied_sequence >= 0),
                last_presented_sequence INTEGER NOT NULL CHECK(last_presented_sequence >= 0),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                CHECK(last_presented_sequence <= last_applied_sequence)
            );

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
                UNIQUE(session_id, attachment_id, work_id, result_id, sequence),
                FOREIGN KEY(session_id) REFERENCES session_bindings(session_id) ON DELETE CASCADE
            );
            """
        )
        connection.execute(
            "INSERT INTO session_bindings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "session-a",
                "conversation-a",
                "default",
                "attachment-a",
                None,
                0,
                0,
                timestamp,
                timestamp,
            ),
        )
        connection.execute(
            """
            INSERT INTO presentations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "presentation-a",
                "session-a",
                "attachment-a",
                "work-a",
                "result-a",
                4,
                '{"body":"","data":{"answer":4},"kind":"work.result","title":"Ready"}',
                "The result is ready.",
                "frontend_model",
                30,
                "available",
                "ready",
                "queued",
                timestamp,
                timestamp,
            ),
        )
        connection.commit()
    finally:
        connection.close()

    with SqliteStateStore(path) as migrated:
        presentation = migrated.get_result_presentation("session-a", "work-a", "result-a", 4)
        assert presentation is not None
        assert presentation.presentation_id == "presentation-a"
        assert (
            migrated.save_presentation(
                replace(
                    presentation,
                    attachment_id="attachment-b",
                    sequence=presentation.sequence,
                    updated_at=utc_now(),
                )
            )
            is False
        )

    verification = sqlite3.connect(path)
    try:
        assert verification.execute("SELECT version FROM schema_version").fetchone() == (6,)
        indexes = {row[1] for row in verification.execute("PRAGMA index_list(presentations)")}
        assert "presentations_result_identity_idx" in indexes
    finally:
        verification.close()


def test_v3_migration_collapses_identical_attachment_replays_without_redelivery(tmp_path) -> None:
    path = tmp_path / "state.db"
    with SqliteStateStore(path) as store:
        store.save_session(_binding())
        assert store.save_presentation(_presentation()) is True

    connection = sqlite3.connect(path)
    try:
        connection.execute("DROP INDEX presentations_result_identity_idx")
        connection.execute("UPDATE schema_version SET version = 3")
        connection.execute(
            """
            INSERT INTO presentations (
                presentation_id, session_id, attachment_id, work_id, result_id, sequence,
                display_json, speech_text, speech_route, priority,
                result_state, display_state, speech_state, created_at, updated_at
            )
            SELECT
                ?, session_id, ?, work_id, result_id, sequence,
                display_json, speech_text, speech_route, priority,
                ?, ?, ?, ?, updated_at
            FROM presentations WHERE presentation_id = ?
            """,
            (
                "presentation-replay",
                "attachment-b",
                ResultState.ACKNOWLEDGED.value,
                DisplayState.DELIVERED.value,
                SpeechState.HEARD.value,
                "2000-01-01T00:00:00+00:00",
                "presentation-a",
            ),
        )
        connection.commit()
    finally:
        connection.close()

    with SqliteStateStore(path) as migrated:
        record = migrated.get_result_presentation("session-a", "work-a", "result-a", 4)
        assert record is not None
        assert record.presentation_id == "presentation-replay"
        assert record.result_state is ResultState.ACKNOWLEDGED
        assert record.display_state is DisplayState.DELIVERED
        assert record.speech_state is SpeechState.HEARD
        assert migrated.pending_presentations("session-a") == ()

    verification = sqlite3.connect(path)
    try:
        assert verification.execute("SELECT COUNT(*) FROM presentations").fetchone() == (1,)
        assert (
            verification.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'presentations_v3_archive'"
            ).fetchone()
            is None
        )
        indexes = {row[1] for row in verification.execute("PRAGMA index_list(presentations)")}
        assert {"presentations_pending_idx", "presentations_result_identity_idx"} <= indexes
    finally:
        verification.close()


def test_v3_migration_rejects_conflicting_attachment_replay_atomically(tmp_path) -> None:
    path = tmp_path / "state.db"
    with SqliteStateStore(path) as store:
        store.save_session(_binding())
        assert store.save_presentation(_presentation()) is True

    connection = sqlite3.connect(path)
    try:
        connection.execute("DROP INDEX presentations_result_identity_idx")
        connection.execute("UPDATE schema_version SET version = 3")
        connection.execute(
            """
            INSERT INTO presentations (
                presentation_id, session_id, attachment_id, work_id, result_id, sequence,
                display_json, speech_text, speech_route, priority,
                result_state, display_state, speech_state, created_at, updated_at
            )
            SELECT
                ?, session_id, ?, work_id, result_id, sequence,
                ?, speech_text, speech_route, priority,
                result_state, display_state, speech_state, ?, updated_at
            FROM presentations WHERE presentation_id = ?
            """,
            (
                "presentation-conflict",
                "attachment-b",
                '{"body":"different","data":{},"kind":"work.result","title":"Conflict"}',
                "2000-01-01T00:00:00+00:00",
                "presentation-a",
            ),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="conflicting immutable evidence"):
        SqliteStateStore(path)

    verification = sqlite3.connect(path)
    try:
        assert verification.execute("SELECT version FROM schema_version").fetchone() == (3,)
        assert verification.execute("SELECT COUNT(*) FROM presentations").fetchone() == (2,)
        assert (
            verification.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'presentations_v3_archive'"
            ).fetchone()
            is None
        )
    finally:
        verification.close()


def test_control_snapshot_withholds_ahead_of_cursor_rows_and_captures_local_claims(tmp_path) -> None:
    with SqliteStateStore(tmp_path / "state.db") as store:
        store.save_session(
            replace(
                _binding(),
                backend_session_id="backend-session-a",
            )
        )
        assert store.save_work_projection(
            WorkProjection(
                work_id="work-a",
                session_id="session-a",
                attachment_id="attachment-a",
                state=WorkState.WAITING_INPUT,
                sequence=1,
            )
        )

        before_commit = store.get_session_control_snapshot("session-a")
        assert before_commit.applied_sequence == 0
        assert before_commit.works == ()

        store.advance_cursor("session-a", 1)
        assert store.save_agent_query_projection(
            AgentQueryProjection(
                query_id="query-a",
                work_id="work-a",
                session_id="session-a",
                attachment_id="attachment-a",
                kind=AgentQueryKind.INFORMATION,
                state=AgentQueryState.PENDING,
                blocking=True,
                sequence=2,
            )
        )
        query_ahead = store.get_session_control_snapshot("session-a")
        assert [work.work_id for work in query_ahead.works] == ["work-a"]
        assert query_ahead.pending_queries == ()

        store.advance_cursor("session-a", 2)
        assert store.save_command(
            CommandRecord(
                command_id="command-a",
                session_id="session-a",
                attachment_id="attachment-a",
                backend_session_id="backend-session-a",
                commit_id="commit-a",
                operation=BackendOperation.ANSWER_QUERY,
                capability_revision="cap-v1",
                payload={"query_id": "query-a", "response": "Yes."},
                state=CommandState.ACCEPTED,
            )
        )

        captured = store.get_session_control_snapshot("session-a")
        assert captured.applied_sequence == 2
        assert [query.query_id for query in captured.pending_queries] == ["query-a"]
        assert captured.claimed_query_ids == frozenset({"query-a"})


def test_control_snapshot_tracks_receipt_backed_capacity_without_inventing_work_state(tmp_path) -> None:
    with SqliteStateStore(tmp_path / "state.db") as store:
        store.save_session(
            replace(
                _binding(),
                backend_session_id="backend-session-a",
            )
        )
        assert store.save_command(
            CommandRecord(
                command_id="command-submit",
                session_id="session-a",
                attachment_id="attachment-a",
                backend_session_id="backend-session-a",
                commit_id="commit-submit",
                operation=BackendOperation.SUBMIT,
                capability_revision="cap-v1",
                payload={"instruction": "Do the work."},
                state=CommandState.ACCEPTED,
                work_id="work-receipt",
            )
        )

        snapshot = store.get_session_control_snapshot("session-a")
        assert snapshot.works == ()
        assert snapshot.reserved_work_ids == ("work-receipt",)
        assert snapshot.anonymous_capacity_reservations == 0


def test_control_snapshot_is_consistent_across_sqlite_connections(tmp_path) -> None:
    path = tmp_path / "state.db"
    with SqliteStateStore(path) as reader, SqliteStateStore(path) as writer:
        reader.save_session(replace(_binding(), backend_session_id="backend-session-a"))
        snapshot_reached_second_read = threading.Event()
        writer_finished = threading.Event()
        writer_errors: list[BaseException] = []
        paused = False

        def pause_after_binding_read(statement: str) -> None:
            nonlocal paused
            if not paused and "SELECT * FROM work_projections" in statement:
                paused = True
                snapshot_reached_second_read.set()
                if not writer_finished.wait(5):
                    raise RuntimeError("concurrent SQLite writer did not finish")

        def write_command() -> None:
            try:
                if not snapshot_reached_second_read.wait(5):
                    raise RuntimeError("snapshot did not reach its second read")
                assert writer.save_command(
                    CommandRecord(
                        command_id="command-concurrent",
                        session_id="session-a",
                        attachment_id="attachment-a",
                        backend_session_id="backend-session-a",
                        commit_id="commit-concurrent",
                        operation=BackendOperation.SUBMIT,
                        capability_revision="cap-v1",
                        payload={"instruction": "Do the work."},
                        state=CommandState.ACCEPTED,
                        work_id="work-concurrent",
                    )
                )
            except BaseException as error:
                writer_errors.append(error)
            finally:
                writer_finished.set()

        reader._connection.set_trace_callback(pause_after_binding_read)  # noqa: SLF001
        worker = threading.Thread(target=write_command)
        worker.start()
        try:
            captured = reader.get_session_control_snapshot("session-a")
        finally:
            reader._connection.set_trace_callback(None)  # noqa: SLF001
            worker.join(timeout=5)

        assert not worker.is_alive()
        assert writer_errors == []
        assert captured.reserved_work_ids == ()
        assert reader.get_session_control_snapshot("session-a").reserved_work_ids == ("work-concurrent",)


def test_command_admission_rejects_a_stale_cross_process_control_snapshot(tmp_path) -> None:
    path = tmp_path / "state.db"
    with SqliteStateStore(path) as first, SqliteStateStore(path) as second:
        first.save_session(replace(_binding(), backend_session_id="backend-session-a"))
        captured = first.get_session_control_snapshot("session-a")
        assert second.save_command(
            CommandRecord(
                command_id="command-winner",
                session_id="session-a",
                attachment_id="attachment-a",
                backend_session_id="backend-session-a",
                commit_id="commit-winner",
                operation=BackendOperation.SUBMIT,
                capability_revision="cap-v1",
                payload={"instruction": "First request."},
            ),
            expected_control_revision=captured.control_revision,
        )

        with pytest.raises(StaleSessionControlError, match="changed after tool projection"):
            first.save_command(
                CommandRecord(
                    command_id="command-loser",
                    session_id="session-a",
                    attachment_id="attachment-a",
                    backend_session_id="backend-session-a",
                    commit_id="commit-loser",
                    operation=BackendOperation.SUBMIT,
                    capability_revision="cap-v1",
                    payload={"instruction": "Conflicting request."},
                ),
                expected_control_revision=captured.control_revision,
            )

        assert first.get_command("command-loser") is None


def test_backend_authored_specialized_target_persists_and_cannot_change(tmp_path) -> None:
    with SqliteStateStore(tmp_path / "state.db") as store:
        store.save_session(_binding())
        first = WorkProjection(
            work_id="work-a",
            session_id="session-a",
            attachment_id="attachment-a",
            state=WorkState.RUNNING,
            sequence=1,
            agent_target="finance",
        )
        assert store.save_work_projection(first)
        store.advance_cursor("session-a", 1)

        with pytest.raises(ValueError, match="agent_target is immutable"):
            store.save_work_projection(
                replace(
                    first,
                    state=WorkState.SUCCEEDED,
                    sequence=2,
                    agent_target="procurement",
                )
            )

        snapshot = store.get_session_control_snapshot("session-a")
        assert snapshot.works[0].agent_target == "finance"
