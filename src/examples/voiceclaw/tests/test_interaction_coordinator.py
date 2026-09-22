# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103, D107

import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest

from voiceclaw.adapters.state import SqliteStateStore
from voiceclaw.application.interaction import (
    CommandRecoveryPendingError,
    InteractionCoordinator,
    InvalidToolArgumentsError,
    UnsupportedOperationError,
)
from voiceclaw.domain.capabilities import CapabilityToolRegistry
from voiceclaw.domain.models import (
    AgentQueryKind,
    AgentQueryProjection,
    AgentQueryState,
    BackendCapabilities,
    BackendEvent,
    BackendEventKind,
    BackendOperation,
    CommandRecord,
    CommandState,
    Durability,
    EventDelivery,
    SessionBinding,
    WorkProjection,
    WorkState,
    WorkStateChanged,
)
from voiceclaw.interaction_profiles import load_interaction_profile_catalog
from voiceclaw.ports.interaction import (
    AttachRequest,
    BackendAdmission,
    BackendAttachment,
    BackendCommandReceipt,
    DetachRequest,
    ReconcileRequest,
    WorkCommand,
)

_INTERACTION_PROFILES = load_interaction_profile_catalog()


def _profile_tools(profile: str) -> CapabilityToolRegistry:
    """Build a registry from one packaged interaction preset."""
    return CapabilityToolRegistry(profile=_INTERACTION_PROFILES.resolve(profile))


class RecordingBackend:
    def __init__(self) -> None:
        self.attach_requests: list[AttachRequest] = []
        self.commands: list[WorkCommand] = []
        self.reconciliations: list[ReconcileRequest] = []
        self.detach_requests: list[DetachRequest] = []
        self.attachment = BackendAttachment(
            attachment_id="attachment-a",
            backend_session_id=None,
            capabilities=BackendCapabilities(
                backend_kind="stateless_model",
                target_label="frontier model",
                revision="cap-v1",
                operations=frozenset({BackendOperation.SUBMIT}),
                durability=Durability.NONE,
                event_delivery=EventDelivery.RESPONSE_ONLY,
            ),
        )

    async def attach(self, request: AttachRequest) -> BackendAttachment:
        self.attach_requests.append(request)
        return self.attachment

    async def execute(self, command: WorkCommand) -> BackendCommandReceipt:
        self.commands.append(command)
        return BackendCommandReceipt(
            command_id=command.command_id,
            admission=BackendAdmission.ACCEPTED,
            work_id="work-a",
        )

    async def reconcile(self, request: ReconcileRequest) -> BackendCommandReceipt:
        self.reconciliations.append(request)
        return BackendCommandReceipt(
            command_id=request.command.command_id,
            admission=BackendAdmission.ACCEPTED,
            work_id="work-a",
        )

    async def events(self, attachment_id: str, *, after_sequence: int | None) -> AsyncIterator[BackendEvent]:
        if False:
            yield BackendEvent(
                event_id="unused",
                session_id="unused",
                attachment_id=attachment_id,
                sequence=after_sequence or 1,
                kind=BackendEventKind.WORK_STATE_CHANGED,
                work_id="unused",
                payload=WorkStateChanged(state=WorkState.RUNNING),
            )

    async def detach(self, request: DetachRequest) -> None:
        self.detach_requests.append(request)


def _save_work(store: SqliteStateStore, projection: WorkProjection) -> WorkProjection:
    """Persist one test Work fact as part of the applied event prefix."""
    binding = store.get_session(projection.session_id)
    assert binding is not None
    applied = replace(projection, sequence=max(projection.sequence, binding.last_applied_sequence + 1))
    assert store.save_work_projection(applied) is True
    store.advance_cursor(applied.session_id, applied.sequence)
    return applied


def _save_query(store: SqliteStateStore, projection: AgentQueryProjection) -> AgentQueryProjection:
    """Persist a correlated test query and advance the applied event prefix."""
    work = store.get_session_work_projection(projection.session_id, projection.work_id)
    if work is None:
        _save_work(
            store,
            WorkProjection(
                work_id=projection.work_id,
                session_id=projection.session_id,
                attachment_id=projection.attachment_id,
                state=(
                    WorkState.WAITING_PERMISSION
                    if projection.kind is AgentQueryKind.PERMISSION
                    else WorkState.WAITING_INPUT
                ),
                sequence=1,
            ),
        )
    binding = store.get_session(projection.session_id)
    assert binding is not None
    applied = replace(projection, sequence=max(projection.sequence, binding.last_applied_sequence + 1))
    assert store.save_agent_query_projection(applied) is True
    store.advance_cursor(applied.session_id, applied.sequence)
    return applied


def test_reconnect_replays_from_presented_cursor_without_discarding_applied_projection(tmp_path) -> None:
    async def scenario() -> None:
        backend = RecordingBackend()
        backend.attachment = BackendAttachment(
            attachment_id="attachment-new",
            backend_session_id="backend-session-a",
            capabilities=backend.attachment.capabilities,
        )
        with SqliteStateStore(tmp_path / "state.db") as store:
            store.save_session(
                SessionBinding(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                    attachment_id="attachment-old",
                    backend_session_id="backend-session-a",
                    last_applied_sequence=12,
                    last_presented_sequence=9,
                )
            )
            coordinator = InteractionCoordinator(backend=backend, state_store=store)

            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )

            assert backend.attach_requests == [
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                    resume_backend_session_id="backend-session-a",
                    after_sequence=9,
                )
            ]
            binding = store.get_session("session-a")
            assert binding is not None
            assert binding.attachment_id == "attachment-new"
            assert binding.last_applied_sequence == 12
            assert binding.last_presented_sequence == 9

    asyncio.run(scenario())


def test_reconnect_rejects_a_different_backend_session_namespace(tmp_path) -> None:
    async def scenario() -> None:
        backend = RecordingBackend()
        backend.attachment = BackendAttachment(
            attachment_id="attachment-new",
            backend_session_id="backend-session-b",
            capabilities=backend.attachment.capabilities,
        )
        with SqliteStateStore(tmp_path / "state.db") as store:
            original = SessionBinding(
                session_id="session-a",
                conversation_id="conversation-a",
                backend_profile="default",
                attachment_id="attachment-old",
                backend_session_id="backend-session-a",
                last_applied_sequence=12,
                last_presented_sequence=9,
            )
            store.save_session(original)
            coordinator = InteractionCoordinator(backend=backend, state_store=store)

            with pytest.raises(ValueError, match="different session"):
                await coordinator.attach(
                    AttachRequest(
                        session_id="session-a",
                        conversation_id="conversation-a",
                        backend_profile="default",
                    )
                )

            assert store.get_session("session-a") == original
            assert backend.detach_requests == [
                DetachRequest(
                    attachment_id="attachment-new",
                    last_presented_sequence=0,
                    reason="resume_session_mismatch",
                )
            ]

    asyncio.run(scenario())


def test_reconnect_requires_a_fresh_attachment_identity(tmp_path) -> None:
    async def scenario() -> None:
        backend = RecordingBackend()
        backend.attachment = BackendAttachment(
            attachment_id="attachment-old",
            backend_session_id="backend-session-a",
            capabilities=backend.attachment.capabilities,
        )
        with SqliteStateStore(tmp_path / "state.db") as store:
            original = SessionBinding(
                session_id="session-a",
                conversation_id="conversation-a",
                backend_profile="default",
                attachment_id="attachment-old",
                backend_session_id="backend-session-a",
                last_applied_sequence=12,
                last_presented_sequence=9,
            )
            store.save_session(original)
            coordinator = InteractionCoordinator(backend=backend, state_store=store)

            with pytest.raises(ValueError, match="new attachment"):
                await coordinator.attach(
                    AttachRequest(
                        session_id="session-a",
                        conversation_id="conversation-a",
                        backend_profile="default",
                    )
                )

            assert store.get_session("session-a") == original
            assert backend.detach_requests == []

    asyncio.run(scenario())


def test_reconnect_rejects_cursor_evidence_without_backend_session_identity(tmp_path) -> None:
    async def scenario() -> None:
        backend = RecordingBackend()
        with SqliteStateStore(tmp_path / "state.db") as store:
            store.save_session(
                SessionBinding(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                    attachment_id="attachment-old",
                    last_applied_sequence=2,
                    last_presented_sequence=1,
                )
            )
            coordinator = InteractionCoordinator(backend=backend, state_store=store)

            with pytest.raises(ValueError, match="missing its backend session identity"):
                await coordinator.attach(
                    AttachRequest(
                        session_id="session-a",
                        conversation_id="conversation-a",
                        backend_profile="default",
                    )
                )

            assert backend.attach_requests == []

    asyncio.run(scenario())


def test_attach_rejects_a_cursor_without_local_presentation_evidence(tmp_path) -> None:
    async def scenario() -> None:
        backend = RecordingBackend()
        with SqliteStateStore(tmp_path / "state.db") as store:
            coordinator = InteractionCoordinator(backend=backend, state_store=store)

            with pytest.raises(ValueError, match="local recovery evidence"):
                await coordinator.attach(
                    AttachRequest(
                        session_id="session-a",
                        conversation_id="conversation-a",
                        backend_profile="default",
                        after_sequence=7,
                    )
                )

            assert backend.attach_requests == []

    asyncio.run(scenario())


def test_attach_rejects_a_backend_session_id_without_local_recovery_evidence(tmp_path) -> None:
    async def scenario() -> None:
        backend = RecordingBackend()
        with SqliteStateStore(tmp_path / "state.db") as store:
            coordinator = InteractionCoordinator(backend=backend, state_store=store)

            with pytest.raises(ValueError, match="local recovery evidence"):
                await coordinator.attach(
                    AttachRequest(
                        session_id="session-a",
                        conversation_id="conversation-a",
                        backend_profile="default",
                        resume_backend_session_id="backend-session-private",
                    )
                )

            assert backend.attach_requests == []

    asyncio.run(scenario())


def test_model_authored_goal_is_dispatched_while_source_turn_stays_out_of_backend_payload(tmp_path) -> None:
    async def scenario() -> None:
        backend = RecordingBackend()
        with SqliteStateStore(tmp_path / "state.db") as store:
            coordinator = InteractionCoordinator(backend=backend, state_store=store)
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )

            assert [tool.name for tool in coordinator.tools_for_session("session-a")] == ["work.delegate"]
            first = await coordinator.execute_tool(
                session_id="session-a",
                commit_id="commit-a",
                command_id="command-a",
                tool_name="work.delegate",
                arguments={
                    "goal": (
                        "Compare the plain BST and AVL implementations using the same 5,000-value "
                        "random and sorted datasets requested earlier."
                    )
                },
                finalized_user_text="Now compare both with the same dataset.",
            )
            retry = await coordinator.execute_tool(
                session_id="session-a",
                commit_id="commit-a",
                command_id="command-a",
                tool_name="work.delegate",
                arguments={
                    "goal": (
                        "Compare the plain BST and AVL implementations using the same 5,000-value "
                        "random and sorted datasets requested earlier."
                    )
                },
                finalized_user_text="Now compare both with the same dataset.",
            )

            assert first.state is CommandState.ACCEPTED
            assert retry.state is CommandState.ACCEPTED
            assert len(backend.commands) == 1
            dispatched = backend.commands[0]
            assert dispatched.attachment_id == "attachment-a"
            assert dispatched.capability_revision == "cap-v1"
            assert dict(dispatched.payload) == {
                "instruction": (
                    "Compare the plain BST and AVL implementations using the same 5,000-value "
                    "random and sorted datasets requested earlier."
                )
            }
            assert "goal" not in dispatched.payload
            assert "source_turn" not in dispatched.payload
            assert store.get_work_projection("attachment-a", "work-a") is None

    asyncio.run(scenario())


def test_inconclusive_dispatch_is_reconciled_without_redispatch(tmp_path) -> None:
    class InitiallyUnknownBackend(RecordingBackend):
        async def execute(self, command: WorkCommand) -> BackendCommandReceipt:
            self.commands.append(command)
            raise TimeoutError

    async def scenario() -> None:
        backend = InitiallyUnknownBackend()
        with SqliteStateStore(tmp_path / "state.db") as store:
            coordinator = InteractionCoordinator(backend=backend, state_store=store)
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )
            first = await coordinator.execute_tool(
                session_id="session-a",
                commit_id="commit-a",
                command_id="command-a",
                tool_name="work.delegate",
                arguments={"goal": "Prepare the requested work."},
                finalized_user_text="prepare the requested work",
            )
            recovered = await coordinator.execute_tool(
                session_id="session-a",
                commit_id="commit-a",
                command_id="command-a",
                tool_name="work.delegate",
                arguments={"goal": "Prepare the requested work."},
                finalized_user_text="prepare the requested work",
            )

            assert first.state is CommandState.INCONCLUSIVE
            assert recovered.state is CommandState.ACCEPTED
            assert len(backend.commands) == 1
            assert [request.command.command_id for request in backend.reconciliations] == ["command-a"]

    asyncio.run(scenario())


def test_unsettled_command_reconciles_through_fresh_attachment_without_rewriting_origin(tmp_path) -> None:
    class InitiallyUnknownBackend(RecordingBackend):
        async def execute(self, command: WorkCommand) -> BackendCommandReceipt:
            self.commands.append(command)
            raise TimeoutError

    async def scenario() -> None:
        backend = InitiallyUnknownBackend()
        backend.attachment = BackendAttachment(
            attachment_id="attachment-old",
            backend_session_id="backend-session-a",
            capabilities=backend.attachment.capabilities,
        )
        with SqliteStateStore(tmp_path / "state.db") as store:
            coordinator = InteractionCoordinator(backend=backend, state_store=store)
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )
            first = await coordinator.execute_tool(
                session_id="session-a",
                commit_id="commit-a",
                command_id="command-a",
                tool_name="work.delegate",
                arguments={"goal": "Prepare the requested work."},
                finalized_user_text="prepare the requested work",
            )
            assert first.state is CommandState.INCONCLUSIVE

            backend.attachment = BackendAttachment(
                attachment_id="attachment-new",
                backend_session_id="backend-session-a",
                capabilities=backend.attachment.capabilities,
            )
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )
            automatically_recovered = store.get_command("command-a")
            assert automatically_recovered is not None
            assert automatically_recovered.state is CommandState.ACCEPTED
            recovered = await coordinator.execute_tool(
                session_id="session-a",
                commit_id="commit-a",
                command_id="command-a",
                tool_name="work.delegate",
                arguments={"goal": "Prepare the requested work."},
                finalized_user_text="prepare the requested work",
            )

            assert recovered.state is CommandState.ACCEPTED
            assert len(backend.commands) == 1
            assert len(backend.reconciliations) == 1
            reconciliation = backend.reconciliations[0]
            assert reconciliation.command.attachment_id == "attachment-old"
            assert reconciliation.command.capability_revision == "cap-v1"
            assert dict(reconciliation.command.payload) == {"instruction": "Prepare the requested work."}
            assert reconciliation.current_attachment_id == "attachment-new"
            assert reconciliation.current_backend_session_id == "backend-session-a"
            stored = store.get_command("command-a")
            assert stored is not None
            assert stored.attachment_id == "attachment-old"
            assert stored.backend_session_id == "backend-session-a"

    asyncio.run(scenario())


def test_new_command_is_blocked_while_recovered_admission_remains_inconclusive(tmp_path) -> None:
    class StillUnknownBackend(RecordingBackend):
        async def reconcile(self, request: ReconcileRequest) -> BackendCommandReceipt:
            self.reconciliations.append(request)
            return BackendCommandReceipt(
                command_id=request.command.command_id,
                admission=BackendAdmission.INCONCLUSIVE,
                reason_code="receipt_unavailable",
            )

    async def scenario() -> None:
        backend = StillUnknownBackend()
        backend.attachment = BackendAttachment(
            attachment_id="attachment-new",
            backend_session_id="backend-session-a",
            capabilities=backend.attachment.capabilities,
        )
        with SqliteStateStore(tmp_path / "state.db") as store:
            store.save_session(
                SessionBinding(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                    attachment_id="attachment-old",
                    backend_session_id="backend-session-a",
                )
            )
            store.save_command(
                CommandRecord(
                    command_id="command-old",
                    session_id="session-a",
                    attachment_id="attachment-old",
                    backend_session_id="backend-session-a",
                    commit_id="commit-old",
                    operation=BackendOperation.SUBMIT,
                    capability_revision="cap-v1",
                    payload={"instruction": "prepare the requested work"},
                    state=CommandState.INCONCLUSIVE,
                )
            )
            coordinator = InteractionCoordinator(backend=backend, state_store=store)
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )

            with pytest.raises(CommandRecoveryPendingError, match="requires backend reconciliation"):
                await coordinator.execute_tool(
                    session_id="session-a",
                    commit_id="commit-new",
                    command_id="command-new",
                    tool_name="work.delegate",
                    arguments={"goal": "Do not duplicate the earlier request."},
                    finalized_user_text="do not duplicate the earlier request",
                )

            assert [request.command.command_id for request in backend.reconciliations] == ["command-old"]
            assert backend.commands == []
            assert store.get_command("command-new") is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("stored_backend_session_id", "current_backend_session_id", "message"),
    [
        ("backend-session-a", "backend-session-b", "different backend session"),
        (None, None, "without a backend session identity"),
    ],
)
def test_unsettled_command_cannot_cross_unrelated_attachment_namespaces(
    tmp_path,
    stored_backend_session_id,
    current_backend_session_id,
    message,
) -> None:
    async def scenario() -> None:
        backend = RecordingBackend()
        backend.attachment = BackendAttachment(
            attachment_id="attachment-new",
            backend_session_id=current_backend_session_id,
            capabilities=backend.attachment.capabilities,
        )
        with SqliteStateStore(tmp_path / "state.db") as store:
            store.save_session(
                SessionBinding(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                    attachment_id="attachment-before-current",
                    backend_session_id=current_backend_session_id,
                )
            )
            store.save_command(
                CommandRecord(
                    command_id="command-a",
                    session_id="session-a",
                    attachment_id="attachment-origin",
                    backend_session_id=stored_backend_session_id,
                    commit_id="commit-a",
                    operation=BackendOperation.SUBMIT,
                    capability_revision="cap-old",
                    payload={"instruction": "prepare the requested work"},
                    state=CommandState.INCONCLUSIVE,
                )
            )
            coordinator = InteractionCoordinator(backend=backend, state_store=store)
            with pytest.raises(ValueError, match=message):
                await coordinator.attach(
                    AttachRequest(
                        session_id="session-a",
                        conversation_id="conversation-a",
                        backend_profile="default",
                    )
                )

            assert backend.commands == []
            assert backend.reconciliations == []
            stored = store.get_command("command-a")
            assert stored is not None
            assert stored.state is CommandState.INCONCLUSIVE
            assert stored.attachment_id == "attachment-origin"

    asyncio.run(scenario())


def test_rejected_reason_survives_retry_and_process_restart(tmp_path) -> None:
    class RejectingBackend(RecordingBackend):
        async def execute(self, command: WorkCommand) -> BackendCommandReceipt:
            self.commands.append(command)
            return BackendCommandReceipt(
                command_id=command.command_id,
                admission=BackendAdmission.REJECTED,
                reason_code="policy_denied",
            )

    async def scenario() -> None:
        path = tmp_path / "state.db"
        backend = RejectingBackend()
        with SqliteStateStore(path) as store:
            coordinator = InteractionCoordinator(backend=backend, state_store=store)
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )
            first = await coordinator.execute_tool(
                session_id="session-a",
                commit_id="commit-a",
                command_id="command-a",
                tool_name="work.delegate",
                arguments={"goal": "Prepare the requested work."},
                finalized_user_text="prepare the requested work",
            )
            assert first.reason_code == "policy_denied"

        backend.attachment = BackendAttachment(
            attachment_id="attachment-new",
            backend_session_id=None,
            capabilities=backend.attachment.capabilities,
        )
        with SqliteStateStore(path) as reopened:
            coordinator = InteractionCoordinator(backend=backend, state_store=reopened)
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )
            retry = await coordinator.execute_tool(
                session_id="session-a",
                commit_id="commit-a",
                command_id="command-a",
                tool_name="work.delegate",
                arguments={"goal": "Prepare the requested work."},
                finalized_user_text="prepare the requested work",
            )

            assert retry.state is CommandState.REJECTED
            assert retry.reason_code == "policy_denied"
            assert len(backend.commands) == 1
            stored = reopened.get_command("command-a")
            assert stored is not None
            assert stored.reason_code == "policy_denied"

    asyncio.run(scenario())


def test_invalid_rejection_reason_is_not_persisted_as_a_terminal_outcome(tmp_path) -> None:
    class InvalidReasonBackend(RecordingBackend):
        async def execute(self, command: WorkCommand) -> BackendCommandReceipt:
            self.commands.append(command)
            return BackendCommandReceipt(
                command_id=command.command_id,
                admission=BackendAdmission.REJECTED,
                reason_code="x" * 129,
            )

    async def scenario() -> None:
        backend = InvalidReasonBackend()
        with SqliteStateStore(tmp_path / "state.db") as store:
            coordinator = InteractionCoordinator(backend=backend, state_store=store)
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )

            with pytest.raises(ValueError, match="invalid admission evidence"):
                await coordinator.execute_tool(
                    session_id="session-a",
                    commit_id="commit-a",
                    command_id="command-a",
                    tool_name="work.delegate",
                    arguments={"goal": "Prepare the requested work."},
                    finalized_user_text="prepare the requested work",
                )

            stored = store.get_command("command-a")
            assert stored is not None
            assert stored.state is CommandState.INCONCLUSIVE
            assert stored.reason_code is None

    asyncio.run(scenario())


def test_accepted_submit_requires_backend_work_identity(tmp_path) -> None:
    class MissingWorkIdBackend(RecordingBackend):
        async def execute(self, command: WorkCommand) -> BackendCommandReceipt:
            self.commands.append(command)
            return BackendCommandReceipt(
                command_id=command.command_id,
                admission=BackendAdmission.ACCEPTED,
            )

    async def scenario() -> None:
        backend = MissingWorkIdBackend()
        with SqliteStateStore(tmp_path / "state.db") as store:
            coordinator = InteractionCoordinator(backend=backend, state_store=store)
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )
            with pytest.raises(ValueError, match="must include work_id"):
                await coordinator.execute_tool(
                    session_id="session-a",
                    commit_id="commit-a",
                    command_id="command-a",
                    tool_name="work.delegate",
                    arguments={"goal": "Prepare the requested work."},
                    finalized_user_text="prepare the requested work",
                )

            stored = store.get_command("command-a")
            assert stored is not None
            assert stored.state is CommandState.INCONCLUSIVE

    asyncio.run(scenario())


def test_unsupported_tool_is_rejected_before_backend_dispatch(tmp_path) -> None:
    async def scenario() -> None:
        backend = RecordingBackend()
        with SqliteStateStore(tmp_path / "state.db") as store:
            coordinator = InteractionCoordinator(backend=backend, state_store=store)
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )
            with pytest.raises(UnsupportedOperationError):
                await coordinator.execute_tool(
                    session_id="session-a",
                    commit_id="commit-a",
                    command_id="command-a",
                    tool_name="work.cancel",
                    arguments={"work_id": "work-a"},
                    finalized_user_text=None,
                )
            assert backend.commands == []

    asyncio.run(scenario())


def test_command_identity_cannot_be_reused_for_different_arguments(tmp_path) -> None:
    async def scenario() -> None:
        backend = RecordingBackend()
        backend.attachment = BackendAttachment(
            attachment_id="attachment-a",
            backend_session_id=None,
            capabilities=BackendCapabilities(
                backend_kind="stateless_model",
                target_label="frontier model",
                revision="cap-v1",
                operations=frozenset({BackendOperation.CANCEL}),
            ),
        )
        with SqliteStateStore(tmp_path / "state.db") as store:
            coordinator = InteractionCoordinator(backend=backend, state_store=store)
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )
            for suffix, sequence in (("a", 1), ("b", 2)):
                _save_work(
                    store,
                    WorkProjection(
                        work_id=f"work-{suffix}",
                        session_id="session-a",
                        attachment_id="attachment-a",
                        state=WorkState.RUNNING,
                        sequence=sequence,
                    ),
                )
            await coordinator.execute_tool(
                session_id="session-a",
                commit_id="commit-a",
                command_id="command-a",
                tool_name="work.cancel",
                arguments={"work_id": "work-a"},
                finalized_user_text=None,
            )

            with pytest.raises(ValueError, match="different request"):
                await coordinator.execute_tool(
                    session_id="session-a",
                    commit_id="commit-a",
                    command_id="command-a",
                    tool_name="work.cancel",
                    arguments={"work_id": "work-b"},
                    finalized_user_text=None,
                )
            assert len(backend.commands) == 1

    asyncio.run(scenario())


def test_targeted_command_rejects_a_receipt_for_different_backend_work(tmp_path) -> None:
    class MismatchedReceiptBackend(RecordingBackend):
        async def execute(self, command: WorkCommand) -> BackendCommandReceipt:
            self.commands.append(command)
            return BackendCommandReceipt(
                command_id=command.command_id,
                admission=BackendAdmission.ACCEPTED,
                work_id="work-b",
            )

    async def scenario() -> None:
        backend = MismatchedReceiptBackend()
        backend.attachment = BackendAttachment(
            attachment_id="attachment-a",
            backend_session_id="backend-session-a",
            capabilities=BackendCapabilities(
                backend_kind="durable_backend",
                target_label="agent",
                revision="cap-v1",
                operations=frozenset({BackendOperation.CANCEL}),
            ),
        )
        with SqliteStateStore(tmp_path / "state.db") as store:
            coordinator = InteractionCoordinator(backend=backend, state_store=store)
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )
            _save_work(
                store,
                WorkProjection(
                    work_id="work-a",
                    session_id="session-a",
                    attachment_id="attachment-a",
                    state=WorkState.RUNNING,
                    sequence=1,
                ),
            )

            with pytest.raises(ValueError, match="does not match the targeted Work"):
                await coordinator.execute_tool(
                    session_id="session-a",
                    commit_id="commit-a",
                    command_id="command-a",
                    tool_name="work.cancel",
                    arguments={"work_id": "work-a"},
                    finalized_user_text=None,
                )

            stored = store.get_command("command-a")
            assert stored is not None
            assert stored.state is CommandState.INCONCLUSIVE
            assert stored.work_id == "work-a"

    asyncio.run(scenario())


def test_targeted_work_id_must_belong_to_the_current_voice_session(tmp_path) -> None:
    async def scenario() -> None:
        backend = RecordingBackend()
        backend.attachment = BackendAttachment(
            attachment_id="attachment-a",
            backend_session_id="backend-session-a",
            capabilities=BackendCapabilities(
                backend_kind="durable_backend",
                target_label="agent",
                revision="cap-v1",
                operations=frozenset({BackendOperation.CANCEL}),
            ),
        )
        with SqliteStateStore(tmp_path / "state.db") as store:
            coordinator = InteractionCoordinator(backend=backend, state_store=store)
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )
            store.save_session(
                SessionBinding(
                    session_id="session-b",
                    conversation_id="conversation-b",
                    backend_profile="default",
                    attachment_id="attachment-b",
                    backend_session_id="backend-session-b",
                    last_applied_sequence=1,
                )
            )
            _save_work(
                store,
                WorkProjection(
                    work_id="work-private",
                    session_id="session-b",
                    attachment_id="attachment-b",
                    state=WorkState.RUNNING,
                    sequence=1,
                ),
            )

            with pytest.raises(UnsupportedOperationError, match="work.cancel"):
                await coordinator.execute_tool(
                    session_id="session-a",
                    commit_id="commit-a",
                    command_id="command-a",
                    tool_name="work.cancel",
                    arguments={"work_id": "work-private"},
                    finalized_user_text=None,
                )

            assert backend.commands == []
            assert store.get_command("command-a") is None

    asyncio.run(scenario())


def test_detach_reports_the_local_fully_presented_cursor(tmp_path) -> None:
    async def scenario() -> None:
        backend = RecordingBackend()
        backend.attachment = BackendAttachment(
            attachment_id="attachment-a",
            backend_session_id="backend-session-a",
            capabilities=backend.attachment.capabilities,
        )
        with SqliteStateStore(tmp_path / "state.db") as store:
            coordinator = InteractionCoordinator(backend=backend, state_store=store)
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )
            store.advance_cursor("session-a", 5)
            # Simulate a future trusted presentation-receipt aggregator. The
            # Interaction Coordinator intentionally has no arbitrary cursor setter.
            store.advance_presented_cursor("session-a", 3)

            await coordinator.detach("session-a", "client_disconnected")

            assert backend.detach_requests == [
                DetachRequest(
                    attachment_id="attachment-a",
                    last_presented_sequence=3,
                    reason="client_disconnected",
                )
            ]
            with pytest.raises(LookupError):
                coordinator.tools_for_session("session-a")

    asyncio.run(scenario())


def test_invalid_tool_arguments_are_rejected_before_persistence_or_dispatch(tmp_path) -> None:
    async def scenario() -> None:
        backend = RecordingBackend()
        with SqliteStateStore(tmp_path / "state.db") as store:
            coordinator = InteractionCoordinator(backend=backend, state_store=store)
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )
            with pytest.raises(InvalidToolArgumentsError, match="unknown arguments"):
                await coordinator.execute_tool(
                    session_id="session-a",
                    commit_id="commit-a",
                    command_id="command-a",
                    tool_name="work.delegate",
                    arguments={"goal": "Prepare the requested work.", "backend_url": "http://attacker"},
                    finalized_user_text="prepare the requested work",
                )
            assert backend.commands == []
            assert store.get_command("command-a") is None

    asyncio.run(scenario())


def test_delegate_requires_model_authored_goal(tmp_path) -> None:
    async def scenario() -> None:
        backend = RecordingBackend()
        with SqliteStateStore(tmp_path / "state.db") as store:
            coordinator = InteractionCoordinator(backend=backend, state_store=store)
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )
            with pytest.raises(InvalidToolArgumentsError, match="missing arguments for work.delegate: goal"):
                await coordinator.execute_tool(
                    session_id="session-a",
                    commit_id="commit-a",
                    command_id="command-a",
                    tool_name="work.delegate",
                    arguments={},
                    finalized_user_text="Now use the same dataset.",
                )
            assert backend.commands == []
            assert store.get_command("command-a") is None

    asyncio.run(scenario())


def test_delegate_requires_server_supplied_finalized_turn(tmp_path) -> None:
    async def scenario() -> None:
        backend = RecordingBackend()
        with SqliteStateStore(tmp_path / "state.db") as store:
            coordinator = InteractionCoordinator(backend=backend, state_store=store)
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )
            with pytest.raises(InvalidToolArgumentsError, match="finalized user turn"):
                await coordinator.execute_tool(
                    session_id="session-a",
                    commit_id="commit-a",
                    command_id="command-a",
                    tool_name="work.delegate",
                    arguments={"goal": "Prepare the requested work."},
                    finalized_user_text=None,
                )
            assert backend.commands == []
            assert store.get_command("command-a") is None

    asyncio.run(scenario())


def test_answer_agent_injects_exact_response_outside_model_arguments(tmp_path) -> None:
    async def scenario() -> None:
        backend = RecordingBackend()
        backend.attachment = BackendAttachment(
            attachment_id="attachment-a",
            backend_session_id="conductor-a",
            capabilities=BackendCapabilities(
                backend_kind="conductor",
                target_label="conductor",
                revision="cap-v1",
                operations=frozenset({BackendOperation.ANSWER_QUERY}),
                sessionful=True,
            ),
        )
        with SqliteStateStore(tmp_path / "state.db") as store:
            coordinator = InteractionCoordinator(
                backend=backend,
                state_store=store,
                tools=_profile_tools("conductor"),
            )
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )
            _save_query(
                store,
                AgentQueryProjection(
                    query_id="query-a",
                    work_id="work-a",
                    session_id="session-a",
                    attachment_id="attachment-a",
                    kind=AgentQueryKind.INFORMATION,
                    state=AgentQueryState.PENDING,
                    blocking=True,
                    sequence=1,
                    prompt="May I use the existing credentials?",
                ),
            )
            await coordinator.execute_tool(
                session_id="session-a",
                commit_id="commit-a",
                command_id="command-a",
                tool_name="work.answer_agent",
                arguments={"query_id": "query-a"},
                finalized_user_text="Yes, use the existing credentials.",
            )

            assert dict(backend.commands[0].payload) == {
                "query_id": "query-a",
                "response": "Yes, use the existing credentials.",
            }

    asyncio.run(scenario())


def test_single_stateful_answer_agent_routes_the_projected_query_id(tmp_path) -> None:
    async def scenario() -> None:
        backend = RecordingBackend()
        backend.attachment = BackendAttachment(
            attachment_id="attachment-a",
            backend_session_id="booking-a",
            capabilities=BackendCapabilities(
                backend_kind="stateful_agent",
                target_label="booking agent",
                revision="cap-v1",
                operations=frozenset({BackendOperation.ANSWER_QUERY}),
                sessionful=True,
            ),
        )
        with SqliteStateStore(tmp_path / "state.db") as store:
            coordinator = InteractionCoordinator(
                backend=backend,
                state_store=store,
                tools=_profile_tools("single_stateful"),
            )
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )

            assert coordinator.tools_for_session("session-a") == ()
            _save_query(
                store,
                AgentQueryProjection(
                    query_id="query-a",
                    work_id="work-a",
                    session_id="session-a",
                    attachment_id="attachment-a",
                    kind=AgentQueryKind.INFORMATION,
                    state=AgentQueryState.PENDING,
                    blocking=True,
                    sequence=1,
                    prompt="Should I book that flight?",
                ),
            )
            tools = coordinator.tools_for_session("session-a")
            assert [tool.name for tool in tools] == ["work.answer_agent"]
            assert tools[0].input_schema["properties"]["query_id"]["enum"] == ["query-a"]
            await coordinator.execute_tool(
                session_id="session-a",
                commit_id="commit-a",
                command_id="command-a",
                tool_name="work.answer_agent",
                arguments={"query_id": "query-a"},
                finalized_user_text="Yes, book that flight.",
            )

            assert dict(backend.commands[0].payload) == {
                "query_id": "query-a",
                "response": "Yes, book that flight.",
            }
            assert backend.commands[0].operation is BackendOperation.ANSWER_QUERY

    asyncio.run(scenario())


def test_answer_agent_rejects_unknown_or_closed_query_before_dispatch(tmp_path) -> None:
    async def scenario() -> None:
        backend = RecordingBackend()
        backend.attachment = BackendAttachment(
            attachment_id="attachment-a",
            backend_session_id="conductor-a",
            capabilities=BackendCapabilities(
                backend_kind="conductor",
                target_label="conductor",
                revision="cap-v1",
                operations=frozenset({BackendOperation.ANSWER_QUERY}),
                sessionful=True,
            ),
        )
        with SqliteStateStore(tmp_path / "state.db") as store:
            coordinator = InteractionCoordinator(
                backend=backend,
                state_store=store,
                tools=_profile_tools("conductor"),
            )
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )

            with pytest.raises(UnsupportedOperationError, match="work.answer_agent"):
                await coordinator.execute_tool(
                    session_id="session-a",
                    commit_id="commit-a",
                    command_id="command-a",
                    tool_name="work.answer_agent",
                    arguments={"query_id": "query-a"},
                    finalized_user_text="Yes.",
                )

            store.save_session(
                SessionBinding(
                    session_id="session-b",
                    conversation_id="conversation-b",
                    backend_profile="default",
                    attachment_id="attachment-b",
                    backend_session_id="conductor-b",
                )
            )
            _save_query(
                store,
                AgentQueryProjection(
                    query_id="query-private",
                    work_id="work-private",
                    session_id="session-b",
                    attachment_id="attachment-b",
                    kind=AgentQueryKind.INFORMATION,
                    state=AgentQueryState.PENDING,
                    blocking=True,
                    sequence=1,
                ),
            )
            with pytest.raises(UnsupportedOperationError, match="work.answer_agent"):
                await coordinator.execute_tool(
                    session_id="session-a",
                    commit_id="commit-private",
                    command_id="command-private",
                    tool_name="work.answer_agent",
                    arguments={"query_id": "query-private"},
                    finalized_user_text="Yes.",
                )

            _save_query(
                store,
                AgentQueryProjection(
                    query_id="query-a",
                    work_id="work-a",
                    session_id="session-a",
                    attachment_id="attachment-a",
                    kind=AgentQueryKind.INFORMATION,
                    state=AgentQueryState.RESOLVED,
                    blocking=True,
                    sequence=1,
                ),
            )
            with pytest.raises(UnsupportedOperationError, match="work.answer_agent"):
                await coordinator.execute_tool(
                    session_id="session-a",
                    commit_id="commit-b",
                    command_id="command-b",
                    tool_name="work.answer_agent",
                    arguments={"query_id": "query-a"},
                    finalized_user_text="Yes.",
                )

            assert backend.commands == []
            assert store.get_command("command-a") is None
            assert store.get_command("command-private") is None
            assert store.get_command("command-b") is None

    asyncio.run(scenario())


def test_answer_agent_uses_query_kind_instead_of_capability_order(tmp_path) -> None:
    async def scenario() -> None:
        backend = RecordingBackend()
        backend.attachment = BackendAttachment(
            attachment_id="attachment-a",
            backend_session_id="conductor-a",
            capabilities=BackendCapabilities(
                backend_kind="conductor",
                target_label="conductor",
                revision="cap-v1",
                operations=frozenset({BackendOperation.ANSWER_QUERY, BackendOperation.RESPOND_PERMISSION}),
                sessionful=True,
            ),
        )
        with SqliteStateStore(tmp_path / "state.db") as store:
            coordinator = InteractionCoordinator(
                backend=backend,
                state_store=store,
                tools=_profile_tools("conductor"),
            )
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )
            _save_query(
                store,
                AgentQueryProjection(
                    query_id="permission-a",
                    work_id="work-a",
                    session_id="session-a",
                    attachment_id="attachment-a",
                    kind=AgentQueryKind.PERMISSION,
                    state=AgentQueryState.PENDING,
                    blocking=True,
                    sequence=1,
                    prompt="Allow this external action?",
                ),
            )

            await coordinator.execute_tool(
                session_id="session-a",
                commit_id="commit-a",
                command_id="command-a",
                tool_name="work.answer_agent",
                arguments={"query_id": "permission-a"},
                finalized_user_text="Allow it once.",
            )

            assert backend.commands[0].operation is BackendOperation.RESPOND_PERMISSION
            assert dict(backend.commands[0].payload) == {
                "query_id": "permission-a",
                "response": "Allow it once.",
            }

    asyncio.run(scenario())


def test_single_stateful_query_response_requires_an_explicit_pending_query_id(tmp_path) -> None:
    async def scenario() -> None:
        backend = RecordingBackend()
        backend.attachment = BackendAttachment(
            attachment_id="attachment-a",
            backend_session_id="booking-a",
            capabilities=BackendCapabilities(
                backend_kind="stateful_agent",
                target_label="booking agent",
                revision="cap-v1",
                operations=frozenset({BackendOperation.ANSWER_QUERY}),
                sessionful=True,
            ),
        )
        with SqliteStateStore(tmp_path / "state.db") as store:
            coordinator = InteractionCoordinator(
                backend=backend,
                state_store=store,
                tools=_profile_tools("single_stateful"),
            )
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )

            with pytest.raises(UnsupportedOperationError, match="work.answer_agent"):
                await coordinator.execute_tool(
                    session_id="session-a",
                    commit_id="commit-a",
                    command_id="command-a",
                    tool_name="work.answer_agent",
                    arguments={"query_id": "query-a"},
                    finalized_user_text="Yes.",
                )

            for sequence, query_id in enumerate(("query-a", "query-b"), start=1):
                _save_query(
                    store,
                    AgentQueryProjection(
                        query_id=query_id,
                        work_id="work-a",
                        session_id="session-a",
                        attachment_id="attachment-a",
                        kind=AgentQueryKind.INFORMATION,
                        state=AgentQueryState.PENDING,
                        blocking=True,
                        sequence=sequence,
                    ),
                )
            tools = coordinator.tools_for_session("session-a")
            assert [tool.name for tool in tools] == ["work.answer_agent"]
            assert tools[0].input_schema["properties"]["query_id"]["enum"] == ["query-a", "query-b"]
            with pytest.raises(InvalidToolArgumentsError, match="missing arguments.*query_id"):
                await coordinator.execute_tool(
                    session_id="session-a",
                    commit_id="commit-b",
                    command_id="command-b",
                    tool_name="work.answer_agent",
                    arguments={},
                    finalized_user_text="Yes.",
                )
            outcome = await coordinator.execute_tool(
                session_id="session-a",
                commit_id="commit-c",
                command_id="command-c",
                tool_name="work.answer_agent",
                arguments={"query_id": "query-b"},
                finalized_user_text="Use the second option.",
            )

            assert outcome.state is CommandState.ACCEPTED
            assert len(backend.commands) == 1
            assert dict(backend.commands[0].payload) == {
                "query_id": "query-b",
                "response": "Use the second option.",
            }

    asyncio.run(scenario())


def test_agent_query_response_has_one_outbox_claim_while_projection_lags(tmp_path) -> None:
    async def scenario() -> None:
        backend = RecordingBackend()
        backend.attachment = BackendAttachment(
            attachment_id="attachment-a",
            backend_session_id="conductor-a",
            capabilities=BackendCapabilities(
                backend_kind="conductor",
                target_label="conductor",
                revision="cap-v1",
                operations=frozenset({BackendOperation.ANSWER_QUERY}),
                sessionful=True,
            ),
        )
        with SqliteStateStore(tmp_path / "state.db") as store:
            coordinator = InteractionCoordinator(
                backend=backend,
                state_store=store,
                tools=_profile_tools("conductor"),
            )
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )
            _save_query(
                store,
                AgentQueryProjection(
                    query_id="query-a",
                    work_id="work-a",
                    session_id="session-a",
                    attachment_id="attachment-a",
                    kind=AgentQueryKind.INFORMATION,
                    state=AgentQueryState.PENDING,
                    blocking=True,
                    sequence=1,
                ),
            )

            await coordinator.execute_tool(
                session_id="session-a",
                commit_id="commit-a",
                command_id="command-a",
                tool_name="work.answer_agent",
                arguments={"query_id": "query-a"},
                finalized_user_text="Use the first option.",
            )
            with pytest.raises(InvalidToolArgumentsError, match="already has a VoiceClaw response command"):
                await coordinator.execute_tool(
                    session_id="session-a",
                    commit_id="commit-b",
                    command_id="command-b",
                    tool_name="work.answer_agent",
                    arguments={"query_id": "query-a"},
                    finalized_user_text="Use the second option.",
                )

            assert [command.command_id for command in backend.commands] == ["command-a"]
            assert store.get_command("command-b") is None
            claim = store.get_agent_query_command_claim("conductor-a", "query-a")
            assert claim is not None
            assert claim.command_id == "command-a"

    asyncio.run(scenario())


def test_rejected_agent_query_response_releases_outbox_claim(tmp_path) -> None:
    class RejectOnceBackend(RecordingBackend):
        async def execute(self, command: WorkCommand) -> BackendCommandReceipt:
            self.commands.append(command)
            return BackendCommandReceipt(
                command_id=command.command_id,
                admission=(BackendAdmission.REJECTED if len(self.commands) == 1 else BackendAdmission.ACCEPTED),
            )

    async def scenario() -> None:
        backend = RejectOnceBackend()
        backend.attachment = BackendAttachment(
            attachment_id="attachment-a",
            backend_session_id="conductor-a",
            capabilities=BackendCapabilities(
                backend_kind="conductor",
                target_label="conductor",
                revision="cap-v1",
                operations=frozenset({BackendOperation.ANSWER_QUERY}),
                sessionful=True,
            ),
        )
        with SqliteStateStore(tmp_path / "state.db") as store:
            coordinator = InteractionCoordinator(
                backend=backend,
                state_store=store,
                tools=_profile_tools("conductor"),
            )
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )
            _save_query(
                store,
                AgentQueryProjection(
                    query_id="query-a",
                    work_id="work-a",
                    session_id="session-a",
                    attachment_id="attachment-a",
                    kind=AgentQueryKind.INFORMATION,
                    state=AgentQueryState.PENDING,
                    blocking=True,
                    sequence=1,
                ),
            )

            rejected = await coordinator.execute_tool(
                session_id="session-a",
                commit_id="commit-a",
                command_id="command-a",
                tool_name="work.answer_agent",
                arguments={"query_id": "query-a"},
                finalized_user_text="An incomplete answer.",
            )
            accepted = await coordinator.execute_tool(
                session_id="session-a",
                commit_id="commit-b",
                command_id="command-b",
                tool_name="work.answer_agent",
                arguments={"query_id": "query-a"},
                finalized_user_text="The corrected answer.",
            )

            assert rejected.state is CommandState.REJECTED
            assert accepted.state is CommandState.ACCEPTED
            assert [command.command_id for command in backend.commands] == ["command-a", "command-b"]

    asyncio.run(scenario())


def test_same_query_command_reconciles_after_projection_is_no_longer_pending(tmp_path) -> None:
    class InitiallyInconclusiveBackend(RecordingBackend):
        async def execute(self, command: WorkCommand) -> BackendCommandReceipt:
            self.commands.append(command)
            return BackendCommandReceipt(
                command_id=command.command_id,
                admission=BackendAdmission.INCONCLUSIVE,
            )

    async def scenario() -> None:
        backend = InitiallyInconclusiveBackend()
        backend.attachment = BackendAttachment(
            attachment_id="attachment-a",
            backend_session_id="conductor-a",
            capabilities=BackendCapabilities(
                backend_kind="stateful_agent",
                target_label="stateful agent",
                revision="cap-v1",
                operations=frozenset({BackendOperation.ANSWER_QUERY}),
                sessionful=True,
            ),
        )
        with SqliteStateStore(tmp_path / "state.db") as store:
            coordinator = InteractionCoordinator(
                backend=backend,
                state_store=store,
                tools=_profile_tools("single_stateful"),
            )
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
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
            )
            _save_query(store, pending)

            first = await coordinator.execute_tool(
                session_id="session-a",
                commit_id="commit-a",
                command_id="command-a",
                tool_name="work.answer_agent",
                arguments={"query_id": "query-a"},
                finalized_user_text="Use the saved address.",
            )
            _save_query(
                store,
                AgentQueryProjection(
                    query_id=pending.query_id,
                    work_id=pending.work_id,
                    session_id=pending.session_id,
                    attachment_id=pending.attachment_id,
                    kind=pending.kind,
                    state=AgentQueryState.RESPONSE_FORWARDED,
                    blocking=pending.blocking,
                    sequence=2,
                ),
            )
            retried = await coordinator.execute_tool(
                session_id="session-a",
                commit_id="commit-a",
                command_id="command-a",
                tool_name="work.answer_agent",
                arguments={"query_id": "query-a"},
                finalized_user_text="Use the saved address.",
            )

            assert first.state is CommandState.INCONCLUSIVE
            assert retried.state is CommandState.ACCEPTED
            assert len(backend.commands) == 1
            assert len(backend.reconciliations) == 1

    asyncio.run(scenario())


def test_single_stateful_nonblocking_query_keeps_steering_and_answer_routes_independent(tmp_path) -> None:
    async def scenario() -> None:
        backend = RecordingBackend()
        backend.attachment = BackendAttachment(
            attachment_id="attachment-a",
            backend_session_id="stateful-a",
            capabilities=BackendCapabilities(
                backend_kind="stateful_agent",
                target_label="stateful agent",
                revision="cap-v1",
                operations=frozenset({BackendOperation.STEER, BackendOperation.ANSWER_QUERY}),
                sessionful=True,
            ),
        )
        with SqliteStateStore(tmp_path / "state.db") as store:
            coordinator = InteractionCoordinator(
                backend=backend,
                state_store=store,
                tools=_profile_tools("single_stateful"),
            )
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )
            _save_query(
                store,
                AgentQueryProjection(
                    query_id="query-a",
                    work_id="work-existing",
                    session_id="session-a",
                    attachment_id="attachment-a",
                    kind=AgentQueryKind.INFORMATION,
                    state=AgentQueryState.PENDING,
                    blocking=False,
                    sequence=1,
                ),
            )

            assert [tool.name for tool in coordinator.tools_for_session("session-a")] == [
                "work.answer_agent",
                "work.delegate",
            ]
            steered = await coordinator.execute_tool(
                session_id="session-a",
                commit_id="commit-steer",
                command_id="command-steer",
                tool_name="work.delegate",
                arguments={"goal": "Continue the active request with the new constraint."},
                finalized_user_text="Continue it with the new constraint.",
            )
            answered = await coordinator.execute_tool(
                session_id="session-a",
                commit_id="commit-answer",
                command_id="command-answer",
                tool_name="work.answer_agent",
                arguments={"query_id": "query-a"},
                finalized_user_text="Use the saved address.",
            )

            assert steered.state is CommandState.ACCEPTED
            assert answered.state is CommandState.ACCEPTED
            assert [command.operation for command in backend.commands] == [
                BackendOperation.STEER,
                BackendOperation.ANSWER_QUERY,
            ]
            assert dict(backend.commands[1].payload) == {
                "query_id": "query-a",
                "response": "Use the saved address.",
            }

    asyncio.run(scenario())


def test_read_only_status_remains_admissible_while_an_earlier_command_needs_recovery(tmp_path) -> None:
    async def scenario() -> None:
        backend = RecordingBackend()
        backend.attachment = BackendAttachment(
            attachment_id="attachment-a",
            backend_session_id="conductor-a",
            capabilities=BackendCapabilities(
                backend_kind="conductor",
                target_label="conductor",
                revision="cap-v1",
                operations=frozenset({BackendOperation.STATUS}),
                durability=Durability.BACKEND,
                sessionful=True,
            ),
        )
        with SqliteStateStore(tmp_path / "state.db") as store:
            coordinator = InteractionCoordinator(
                backend=backend,
                state_store=store,
                tools=_profile_tools("conductor"),
            )
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )
            assert store.save_command(
                CommandRecord(
                    command_id="command-unsettled",
                    session_id="session-a",
                    attachment_id="attachment-a",
                    backend_session_id="conductor-a",
                    commit_id="commit-unsettled",
                    operation=BackendOperation.SUBMIT,
                    capability_revision="cap-old",
                    payload={"instruction": "Earlier work."},
                    state=CommandState.INCONCLUSIVE,
                )
            )

            assert [tool.name for tool in coordinator.tools_for_session("session-a")] == ["work.status"]
            outcome = await coordinator.execute_tool(
                session_id="session-a",
                commit_id="commit-status",
                command_id="command-status",
                tool_name="work.status",
                arguments={},
                finalized_user_text=None,
            )

            assert outcome.state is CommandState.ACCEPTED
            assert backend.commands[-1].operation is BackendOperation.STATUS

    asyncio.run(scenario())


def test_retry_uses_the_persisted_command_shape_after_capabilities_change(tmp_path) -> None:
    async def scenario() -> None:
        backend = RecordingBackend()
        backend.attachment = BackendAttachment(
            attachment_id="attachment-a",
            backend_session_id="backend-session-a",
            capabilities=BackendCapabilities(
                backend_kind="stateless_model",
                target_label="replacement target",
                revision="cap-v2",
                operations=frozenset({BackendOperation.SUBMIT}),
            ),
        )
        with SqliteStateStore(tmp_path / "state.db") as store:
            store.save_session(
                SessionBinding(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                    attachment_id="attachment-a",
                    backend_session_id=None,
                )
            )
            assert store.save_command(
                CommandRecord(
                    command_id="command-a",
                    session_id="session-a",
                    attachment_id="attachment-a",
                    backend_session_id="backend-session-a",
                    commit_id="commit-a",
                    operation=BackendOperation.SUBMIT,
                    capability_revision="cap-v1",
                    payload={
                        "agent": "finance",
                        "instruction": "Prepare the finance report using the current quarter.",
                    },
                    state=CommandState.ACCEPTED,
                    work_id="work-a",
                )
            )
            coordinator = InteractionCoordinator(backend=backend, state_store=store)
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )

            outcome = await coordinator.execute_tool(
                session_id="session-a",
                commit_id="commit-a",
                command_id="command-a",
                tool_name="work.delegate",
                arguments={"agent": "finance", "goal": "Prepare the finance report using the current quarter."},
                finalized_user_text="Prepare the report.",
            )

            assert outcome.state is CommandState.ACCEPTED
            assert backend.commands == []

    asyncio.run(scenario())


def test_single_stateful_explicit_query_id_disambiguates_multiple_queries(tmp_path) -> None:
    async def scenario() -> None:
        backend = RecordingBackend()
        backend.attachment = BackendAttachment(
            attachment_id="attachment-a",
            backend_session_id="stateful-a",
            capabilities=BackendCapabilities(
                backend_kind="stateful_agent",
                target_label="stateful agent",
                revision="cap-v1",
                operations=frozenset({BackendOperation.SUBMIT, BackendOperation.ANSWER_QUERY}),
                sessionful=True,
            ),
        )
        with SqliteStateStore(tmp_path / "state.db") as store:
            coordinator = InteractionCoordinator(
                backend=backend,
                state_store=store,
                tools=_profile_tools("single_stateful"),
            )
            await coordinator.attach(
                AttachRequest(
                    session_id="session-a",
                    conversation_id="conversation-a",
                    backend_profile="default",
                )
            )
            for sequence, query_id in enumerate(("query-a", "query-b"), start=1):
                _save_query(
                    store,
                    AgentQueryProjection(
                        query_id=query_id,
                        work_id="work-a",
                        session_id="session-a",
                        attachment_id="attachment-a",
                        kind=AgentQueryKind.INFORMATION,
                        state=AgentQueryState.PENDING,
                        blocking=False,
                        sequence=sequence,
                    ),
                )

            outcome = await coordinator.execute_tool(
                session_id="session-a",
                commit_id="commit-a",
                command_id="command-a",
                tool_name="work.answer_agent",
                arguments={"query_id": "query-b"},
                finalized_user_text="Use the second answer.",
            )

            assert outcome.state is CommandState.ACCEPTED
            assert len(backend.commands) == 1
            assert dict(backend.commands[0].payload) == {
                "query_id": "query-b",
                "response": "Use the second answer.",
            }

    asyncio.run(scenario())


def test_agent_query_claim_insert_is_atomic_across_sqlite_connections(tmp_path) -> None:
    database = tmp_path / "state.db"
    with SqliteStateStore(database) as first_store, SqliteStateStore(database) as second_store:
        first_store.save_session(
            SessionBinding(
                session_id="session-a",
                conversation_id="conversation-a",
                backend_profile="default",
                attachment_id="attachment-a",
                backend_session_id="backend-session-a",
            )
        )
        commands = (
            CommandRecord(
                command_id="command-a",
                session_id="session-a",
                attachment_id="attachment-a",
                backend_session_id="backend-session-a",
                commit_id="commit-a",
                operation=BackendOperation.ANSWER_QUERY,
                capability_revision="cap-v1",
                payload={"query_id": "query-a", "response": "First"},
            ),
            CommandRecord(
                command_id="command-b",
                session_id="session-a",
                attachment_id="attachment-a",
                backend_session_id="backend-session-a",
                commit_id="commit-b",
                operation=BackendOperation.ANSWER_QUERY,
                capability_revision="cap-v1",
                payload={"query_id": "query-a", "response": "Second"},
            ),
        )
        barrier = Barrier(2)

        def save(store: SqliteStateStore, command: CommandRecord) -> bool:
            barrier.wait()
            return store.save_command(command)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = (
                executor.submit(save, first_store, commands[0]),
                executor.submit(save, second_store, commands[1]),
            )
            saved = [future.result() for future in futures]

        assert sorted(saved) == [False, True]
        claim = first_store.get_agent_query_command_claim("backend-session-a", "query-a")
        assert claim is not None
        assert claim.command_id == commands[saved.index(True)].command_id
