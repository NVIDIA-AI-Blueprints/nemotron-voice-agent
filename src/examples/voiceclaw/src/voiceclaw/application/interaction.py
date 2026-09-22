# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Backend-neutral attachment and capability-gated command admission."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Any

from voiceclaw.domain.capabilities import (
    CapabilityToolRegistry,
    PendingToolQuery,
    SemanticTool,
    ToolProjectionState,
    ToolWork,
)
from voiceclaw.domain.models import (
    BackendOperation,
    CommandRecord,
    CommandState,
    SessionBinding,
    utc_now,
)
from voiceclaw.interaction_profiles import MAX_DELEGATED_GOAL_BYTES
from voiceclaw.ports.interaction import (
    AgentInteractionPort,
    AttachRequest,
    BackendAdmission,
    BackendAttachment,
    BackendCommandReceipt,
    DetachRequest,
    ReconcileRequest,
    WorkCommand,
)
from voiceclaw.ports.state import StaleSessionControlError, StateStore


class SessionNotAttachedError(LookupError):
    """Raised when a command has no live backend attachment."""


class UnsupportedOperationError(ValueError):
    """Raised when a model requests a capability absent from the attachment."""


class InvalidToolArgumentsError(ValueError):
    """Raised when untrusted model arguments violate the canonical schema."""


class CommandRecoveryPendingError(RuntimeError):
    """Raised when an earlier command has no authoritative admission outcome."""


class SessionControlChangedError(RuntimeError):
    """Raised when authoritative/local control facts changed during admission."""


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    """VoiceClaw's local outbox view of backend command admission.

    ``command_id`` belongs to VoiceClaw.  ``work_id``, when present, is the
    normalized identifier issued by the authoritative backend.
    """

    command_id: str
    state: CommandState
    work_id: str | None = None
    reason_code: str | None = None


class InteractionCoordinator:
    """Attach sessions and execute model-selected commands on the server."""

    def __init__(
        self,
        *,
        backend: AgentInteractionPort,
        state_store: StateStore,
        tools: CapabilityToolRegistry | None = None,
    ) -> None:
        """Bind a backend adapter without exposing it to client/model inputs."""
        self._backend = backend
        self._state_store = state_store
        self._tools = tools or CapabilityToolRegistry()
        self._attachments: dict[str, BackendAttachment] = {}
        self._command_locks: dict[str, asyncio.Lock] = {}

    _QUERY_RESPONSE_OPERATIONS = frozenset({BackendOperation.ANSWER_QUERY, BackendOperation.RESPOND_PERMISSION})
    _TOOLS_BY_OPERATION = {
        BackendOperation.SUBMIT: frozenset({"work.delegate"}),
        BackendOperation.STEER: frozenset({"work.delegate"}),
        BackendOperation.ANSWER_QUERY: frozenset({"work.answer_agent"}),
        BackendOperation.RESPOND_PERMISSION: frozenset({"work.answer_agent"}),
        BackendOperation.CANCEL: frozenset({"work.cancel"}),
        BackendOperation.STATUS: frozenset({"work.status"}),
    }

    async def attach(self, request: AttachRequest) -> BackendAttachment:
        """Attach one voice session from VoiceClaw-owned recovery evidence.

        ``after_sequence`` is the highest backend event that VoiceClaw fully
        presented.  It is not the applied-event cursor and callers cannot move
        it independently of the local presentation ledger.  On reconnect, the
        backend replays after that presented cursor while VoiceClaw keeps its
        newer materialized projection and pending delivery records.
        """
        existing = self._state_store.get_session(request.session_id)
        attach_request = self._recovery_request(request, existing)
        attachment = await self._backend.attach(attach_request)

        presented_sequence = attach_request.after_sequence or 0
        applied_sequence = presented_sequence
        created_at = utc_now()
        if existing is not None:
            created_at = existing.created_at
            expected_backend_session = attach_request.resume_backend_session_id
            if expected_backend_session is not None and attachment.backend_session_id != expected_backend_session:
                if attachment.attachment_id != existing.attachment_id:
                    with suppress(Exception):
                        await self._backend.detach(
                            DetachRequest(
                                attachment_id=attachment.attachment_id,
                                last_presented_sequence=0,
                                reason="resume_session_mismatch",
                            )
                        )
                raise ValueError("backend attached a different session than the requested resume target")
            if (
                expected_backend_session is not None
                and existing.attachment_id is not None
                and attachment.attachment_id == existing.attachment_id
            ):
                raise ValueError("backend resume must create a new attachment")
            if expected_backend_session is not None:
                applied_sequence = max(existing.last_applied_sequence, presented_sequence)

        unsettled = self._state_store.unsettled_commands(request.session_id)
        try:
            self._validate_recovery_namespace(unsettled, attachment)
        except Exception:
            if existing is None or attachment.attachment_id != existing.attachment_id:
                with suppress(Exception):
                    await self._backend.detach(
                        DetachRequest(
                            attachment_id=attachment.attachment_id,
                            last_presented_sequence=presented_sequence,
                            reason="command_recovery_namespace_mismatch",
                        )
                    )
            raise

        binding = SessionBinding(
            session_id=request.session_id,
            conversation_id=request.conversation_id,
            backend_profile=request.backend_profile,
            attachment_id=attachment.attachment_id,
            backend_session_id=attachment.backend_session_id,
            last_applied_sequence=applied_sequence,
            last_presented_sequence=presented_sequence,
            created_at=created_at,
            updated_at=utc_now(),
        )
        self._attachments[request.session_id] = attachment
        self._state_store.save_session(binding)
        try:
            await self._reconcile_unsettled(request.session_id, unsettled, attachment)
        except Exception:
            self._attachments.pop(request.session_id, None)
            with suppress(Exception):
                await self._backend.detach(
                    DetachRequest(
                        attachment_id=attachment.attachment_id,
                        last_presented_sequence=presented_sequence,
                        reason="command_recovery_failed",
                    )
                )
            if existing is not None:
                self._state_store.save_session(existing)
            raise
        return attachment

    @staticmethod
    def _validate_recovery_namespace(
        commands: tuple[CommandRecord, ...],
        attachment: BackendAttachment,
    ) -> None:
        """Fail closed before binding an attachment to unrelated outbox state."""
        for command in commands:
            if command.backend_session_id != attachment.backend_session_id:
                raise ValueError("unsettled command belongs to a different backend session")
            if command.attachment_id != attachment.attachment_id and command.backend_session_id is None:
                raise ValueError("unsettled command cannot cross attachments without a backend session identity")

    async def _reconcile_unsettled(
        self,
        session_id: str,
        commands: tuple[CommandRecord, ...],
        attachment: BackendAttachment,
    ) -> None:
        """Reconcile every persisted ambiguous command before admitting new work.

        A reconnecting browser cannot be expected to reproduce a prior
        ``command_id``.  Recovery is therefore an attachment responsibility,
        not a side effect of a later model retry.
        """
        for candidate in commands:
            lock = self._command_locks.setdefault(candidate.command_id, asyncio.Lock())
            async with lock:
                current = self._state_store.get_command(candidate.command_id)
                if current is None or current.session_id != session_id:
                    raise ValueError("unsettled command disappeared during attachment recovery")
                if current.state not in {
                    CommandState.STAGED,
                    CommandState.DISPATCHING,
                    CommandState.INCONCLUSIVE,
                    CommandState.RECONCILING,
                }:
                    continue
                await self._resume(current, attachment)

    @staticmethod
    def _recovery_request(
        request: AttachRequest,
        existing: SessionBinding | None,
    ) -> AttachRequest:
        """Bind reconnect input to the local mapping and presentation ledger."""
        if existing is None:
            if request.resume_backend_session_id is not None or request.after_sequence not in {None, 0}:
                raise ValueError("cannot resume a backend session without local recovery evidence")
            return request
        if existing.conversation_id != request.conversation_id:
            raise ValueError("session_id is already bound to a different conversation")
        if existing.backend_profile != request.backend_profile:
            raise ValueError("session_id is already bound to a different backend profile")
        if request.after_sequence is not None and request.after_sequence != existing.last_presented_sequence:
            raise ValueError("after_sequence does not match the local presented cursor")
        if existing.last_applied_sequence > 0 and existing.backend_session_id is None:
            raise ValueError("local replay evidence is missing its backend session identity")
        if (
            request.resume_backend_session_id is not None
            and request.resume_backend_session_id != existing.backend_session_id
        ):
            raise ValueError("resume_backend_session_id does not match the local session mapping")
        return replace(
            request,
            resume_backend_session_id=request.resume_backend_session_id or existing.backend_session_id,
            after_sequence=existing.last_presented_sequence,
        )

    async def detach(self, session_id: str, reason: str) -> None:
        """End only this attachment and report VoiceClaw's presented cursor."""
        attachment = self._attachment(session_id)
        binding = self._state_store.get_session(session_id)
        if binding is None:
            raise SessionNotAttachedError(session_id)
        normalized_reason = reason.strip()
        if not normalized_reason or "\x00" in normalized_reason or len(normalized_reason) > 128:
            raise ValueError("detach reason is invalid")
        await self._backend.detach(
            DetachRequest(
                attachment_id=attachment.attachment_id,
                last_presented_sequence=binding.last_presented_sequence,
                reason=normalized_reason,
            )
        )
        self._attachments.pop(session_id, None)

    def tools_for_session(self, session_id: str) -> tuple[SemanticTool, ...]:
        """Project tools authorized by attachment and applicable to current Work."""
        attachment = self._attachment(session_id)
        return self._tools.project(
            attachment.capabilities,
            state=self._tool_projection_state(session_id),
        )

    async def execute_tool(
        self,
        *,
        session_id: str,
        commit_id: str,
        command_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        finalized_user_text: str | None,
        resolved_operation: BackendOperation | None = None,
    ) -> CommandOutcome:
        """Validate, persist, and safely dispatch or reconcile one command."""
        lock = self._command_locks.setdefault(command_id, asyncio.Lock())
        async with lock:
            return await self._execute_tool_locked(
                session_id=session_id,
                commit_id=commit_id,
                command_id=command_id,
                tool_name=tool_name,
                arguments=arguments,
                finalized_user_text=finalized_user_text,
                resolved_operation=resolved_operation,
            )

    async def _execute_tool_locked(
        self,
        *,
        session_id: str,
        commit_id: str,
        command_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        finalized_user_text: str | None,
        resolved_operation: BackendOperation | None,
    ) -> CommandOutcome:
        attachment = self._attachment(session_id)
        existing = self._state_store.get_command(command_id)
        if existing is not None:
            retry_payload = self._retry_command_payload(
                existing=existing,
                tool_name=tool_name,
                arguments=arguments,
                finalized_user_text=finalized_user_text,
                resolved_operation=resolved_operation,
            )
            self._validate_retry_identity(
                existing,
                session_id=session_id,
                commit_id=commit_id,
                operation=existing.operation,
                arguments=retry_payload,
            )
            return await self._resume(existing, attachment)

        projection_state = self._tool_projection_state(session_id)
        if projection_state.command_recovery_pending and tool_name != "work.status":
            raise CommandRecoveryPendingError("an earlier command still requires backend reconciliation")
        if tool_name == "work.answer_agent":
            query_id = arguments.get("query_id")
            if any(query.query_id == query_id and query.claimed for query in projection_state.pending_queries):
                raise InvalidToolArgumentsError("AgentQuery already has a VoiceClaw response command")
        try:
            validated_arguments = self._tools.validate_arguments(
                tool_name,
                arguments,
                capabilities=attachment.capabilities,
                state=projection_state,
                resolved_operation=resolved_operation,
            )
        except KeyError as error:
            raise UnsupportedOperationError(tool_name) from error
        except ValueError as error:
            raise InvalidToolArgumentsError(str(error)) from error

        query_operation, validated_arguments = self._bind_agent_query(
            tool_name=tool_name,
            arguments=validated_arguments,
            state=projection_state,
            resolved_operation=resolved_operation,
        )
        selected_operation = query_operation or resolved_operation
        try:
            operation = self._tools.operation_for_tool(
                tool_name,
                attachment.capabilities,
                state=projection_state,
                arguments=validated_arguments,
                resolved_operation=selected_operation,
            )
        except KeyError as error:
            raise UnsupportedOperationError(tool_name) from error
        except ValueError as error:
            raise UnsupportedOperationError(str(error)) from error
        if operation in self._QUERY_RESPONSE_OPERATIONS and query_operation is None:
            raise InvalidToolArgumentsError(f"{tool_name} has no pending AgentQuery for this session")
        if query_operation is not None and attachment.backend_session_id is None:
            raise InvalidToolArgumentsError("AgentQuery responses require a durable backend session identity")
        self._validate_target_binding(
            tool_name=tool_name,
            arguments=validated_arguments,
            state=projection_state,
        )
        command_payload = self._command_payload(
            tool_name,
            validated_arguments,
            operation=operation,
            finalized_user_text=finalized_user_text,
        )

        # A retry under a new command ID could duplicate backend Work while an
        # earlier admission is still inconclusive. Attachment recovery already
        # attempted reconciliation; fail closed until the backend resolves the
        # original immutable envelope.
        if operation is not BackendOperation.STATUS and self._state_store.unsettled_commands(session_id):
            raise CommandRecoveryPendingError("an earlier command still requires backend reconciliation")

        if not attachment.capabilities.supports(operation):
            raise UnsupportedOperationError(
                f"{tool_name} is not authorized by capability revision {attachment.capabilities.revision}"
            )

        staged = CommandRecord(
            command_id=command_id,
            session_id=session_id,
            attachment_id=attachment.attachment_id,
            work_id=(command_payload.get("work_id") if isinstance(command_payload.get("work_id"), str) else None),
            commit_id=commit_id,
            operation=operation,
            capability_revision=attachment.capabilities.revision,
            payload=command_payload,
            backend_session_id=attachment.backend_session_id,
        )
        try:
            inserted = self._state_store.save_command(
                staged,
                expected_control_revision=projection_state.control_revision,
            )
        except StaleSessionControlError as error:
            raced = self._state_store.get_command(command_id)
            if raced is None:
                raise SessionControlChangedError(
                    "session state changed after this tool was projected; refresh and retry"
                ) from error
            self._validate_retry_identity(
                raced,
                session_id=session_id,
                commit_id=commit_id,
                operation=operation,
                arguments=command_payload,
            )
            return await self._resume(raced, attachment)
        if not inserted:
            raced = self._state_store.get_command(command_id)
            if raced is not None:
                self._validate_retry_identity(
                    raced,
                    session_id=session_id,
                    commit_id=commit_id,
                    operation=operation,
                    arguments=command_payload,
                )
                return await self._resume(raced, attachment)
            if query_operation is not None:
                query_id = command_payload.get("query_id")
                if isinstance(query_id, str) and attachment.backend_session_id is not None:
                    claimed = self._state_store.get_agent_query_command_claim(
                        attachment.backend_session_id,
                        query_id,
                    )
                    if claimed is not None:
                        raise InvalidToolArgumentsError("AgentQuery already has a VoiceClaw response command")
            raise RuntimeError("command insert lost without a stored record")
        return await self._dispatch(staged, attachment=attachment, reconcile=False)

    def _tool_projection_state(self, session_id: str) -> ToolProjectionState:
        """Build the tool snapshot only from backend-authored projections."""
        snapshot = self._state_store.get_session_control_snapshot(session_id)
        return ToolProjectionState(
            control_revision=snapshot.control_revision,
            applied_sequence=snapshot.applied_sequence,
            works=tuple(
                ToolWork(
                    work_id=projection.work_id,
                    state=projection.state,
                    agent_target=projection.agent_target,
                )
                for projection in snapshot.works
            ),
            pending_queries=tuple(
                PendingToolQuery(
                    query_id=query.query_id,
                    work_id=query.work_id,
                    kind=query.kind,
                    blocking=query.blocking,
                    claimed=query.query_id in snapshot.claimed_query_ids,
                )
                for query in snapshot.pending_queries
            ),
            reserved_work_ids=snapshot.reserved_work_ids,
            anonymous_capacity_reservations=snapshot.anonymous_capacity_reservations,
            cancel_claimed_work_ids=snapshot.cancel_claimed_work_ids,
            command_recovery_pending=snapshot.command_recovery_pending,
        )

    def _retry_command_payload(
        self,
        *,
        existing: CommandRecord,
        tool_name: str,
        arguments: Mapping[str, Any],
        finalized_user_text: str | None,
        resolved_operation: BackendOperation | None,
    ) -> Mapping[str, Any]:
        """Rebuild one immutable retry without consulting mutable projections."""
        if resolved_operation is not None and resolved_operation is not existing.operation:
            raise ValueError("command identity is already bound to a different request")
        compatible_tools = self._TOOLS_BY_OPERATION.get(existing.operation, frozenset())
        if tool_name not in compatible_tools:
            raise ValueError("command identity is already bound to a different tool operation")
        retry_arguments = self._normalize_retry_arguments(tool_name, arguments)
        return self._command_payload(
            tool_name,
            retry_arguments,
            operation=existing.operation,
            finalized_user_text=finalized_user_text,
        )

    @staticmethod
    def _normalize_retry_arguments(tool_name: str, arguments: Mapping[str, Any]) -> dict[str, str]:
        """Normalize immutable retry data without consulting changed capabilities.

        Current model-visible Work schemas contain only bounded strings. The
        persisted command payload remains the authority for the exact historic
        shape; identity comparison rejects missing, added, or changed fields.
        """
        if not isinstance(arguments, Mapping):
            raise InvalidToolArgumentsError(f"arguments for {tool_name} must be an object")
        normalized: dict[str, str] = {}
        for name, value in arguments.items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise InvalidToolArgumentsError(f"arguments for {tool_name} must contain string values")
            clean = value.strip()
            try:
                clean_bytes = len(clean.encode("utf-8"))
            except UnicodeEncodeError:
                clean_bytes = 64 * 1024 + 1
            if not clean or "\x00" in clean or clean_bytes > 64 * 1024:
                raise InvalidToolArgumentsError(f"argument {name} for {tool_name} is invalid")
            normalized[name] = clean
        return normalized

    @staticmethod
    def _command_payload(
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        operation: BackendOperation,
        finalized_user_text: str | None,
    ) -> dict[str, Any]:
        """Bind model-authored goals and exact query replies to the backend port.

        The finalized turn proves which user turn caused this command and is
        retained by the frontend conversation ledger.  It is deliberately not
        reused as a submit/steer instruction: the frontend model must formulate
        the validated, standalone ``goal`` visible in the tool schema.  Pending
        AgentQuery and permission responses remain exact user replies because
        paraphrasing those answers could change their meaning.
        """
        payload = dict(arguments)
        if tool_name not in {"work.delegate", "work.answer_agent"}:
            return payload
        if not isinstance(finalized_user_text, str):
            raise InvalidToolArgumentsError(f"{tool_name} requires a finalized user turn")
        normalized = finalized_user_text.strip()
        try:
            source_turn_bytes = len(normalized.encode("utf-8"))
        except UnicodeEncodeError:
            source_turn_bytes = 64 * 1024 + 1
        if not normalized or "\x00" in normalized or source_turn_bytes > 64 * 1024:
            raise InvalidToolArgumentsError(f"{tool_name} requires a valid finalized user turn")
        is_query_response = operation in {
            BackendOperation.ANSWER_QUERY,
            BackendOperation.RESPOND_PERMISSION,
        }
        if is_query_response:
            # Backend query responses use the exact user turn and authoritative query identity.
            payload.pop("goal", None)
            payload["response"] = normalized
            return payload

        if tool_name != "work.delegate":
            raise InvalidToolArgumentsError(f"{tool_name} cannot issue {operation.value}")
        goal = payload.pop("goal", None)
        if not isinstance(goal, str):
            raise InvalidToolArgumentsError("work.delegate requires a model-authored goal")
        normalized_goal = goal.strip()
        try:
            goal_bytes = len(normalized_goal.encode("utf-8"))
        except UnicodeEncodeError:
            goal_bytes = MAX_DELEGATED_GOAL_BYTES + 1
        if not normalized_goal or "\x00" in normalized_goal or goal_bytes > MAX_DELEGATED_GOAL_BYTES:
            raise InvalidToolArgumentsError("work.delegate requires a valid model-authored goal")
        payload["instruction"] = normalized_goal
        return payload

    def _bind_agent_query(
        self,
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
        state: ToolProjectionState,
        resolved_operation: BackendOperation | None,
    ) -> tuple[BackendOperation | None, Mapping[str, Any]]:
        """Bind query responses from the same captured backend state snapshot."""
        query: PendingToolQuery | None = None
        if tool_name == "work.answer_agent":
            query_id = arguments.get("query_id")
            if not isinstance(query_id, str):
                raise InvalidToolArgumentsError("work.answer_agent requires a pending query_id")
            query = next(
                (
                    candidate
                    for candidate in state.pending_queries
                    if candidate.query_id == query_id and not candidate.claimed
                ),
                None,
            )
            if query is None:
                raise InvalidToolArgumentsError(
                    "work.answer_agent references no answerable AgentQuery in this snapshot"
                )
        if query is None:
            return None, arguments

        operation = query.operation
        if resolved_operation is not None and resolved_operation is not operation:
            raise InvalidToolArgumentsError("requested operation conflicts with the pending AgentQuery kind")
        bound_arguments = dict(arguments)
        bound_arguments["query_id"] = query.query_id
        return operation, bound_arguments

    @staticmethod
    def _validate_target_binding(
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
        state: ToolProjectionState,
    ) -> None:
        """Reject model-supplied Work IDs outside the captured projection."""
        work_id = arguments.get("work_id")
        if work_id is None:
            return
        if not isinstance(work_id, str) or work_id not in state.known_work_ids:
            raise InvalidToolArgumentsError(f"{tool_name} references unknown Work for this session")

    async def _resume(self, record: CommandRecord, attachment: BackendAttachment) -> CommandOutcome:
        if record.state not in {
            CommandState.STAGED,
            CommandState.DISPATCHING,
            CommandState.INCONCLUSIVE,
            CommandState.RECONCILING,
        }:
            return self._outcome(record)

        same_attachment = record.attachment_id == attachment.attachment_id
        if record.backend_session_id != attachment.backend_session_id:
            raise ValueError("unsettled command belongs to a different backend session")
        if not same_attachment and record.backend_session_id is None:
            raise ValueError("unsettled command cannot cross attachments without a backend session identity")
        if record.state is CommandState.STAGED:
            return await self._dispatch(record, attachment=attachment, reconcile=not same_attachment)
        return await self._dispatch(record, attachment=attachment, reconcile=True)

    async def _dispatch(
        self,
        record: CommandRecord,
        *,
        attachment: BackendAttachment,
        reconcile: bool,
    ) -> CommandOutcome:
        pending_state = CommandState.RECONCILING if reconcile else CommandState.DISPATCHING
        pending = replace(record, state=pending_state, updated_at=utc_now())
        self._state_store.update_command(pending)
        command = self._work_command(pending)
        try:
            if reconcile:
                receipt = await self._backend.reconcile(
                    ReconcileRequest(
                        command=command,
                        current_attachment_id=attachment.attachment_id,
                        current_backend_session_id=attachment.backend_session_id,
                        current_capability_revision=attachment.capabilities.revision,
                    )
                )
            else:
                receipt = await self._backend.execute(command)
        except Exception:
            inconclusive = replace(pending, state=CommandState.INCONCLUSIVE, updated_at=utc_now())
            self._state_store.update_command(inconclusive)
            return self._outcome(inconclusive)
        return self._accept_receipt(pending, receipt)

    def _accept_receipt(self, pending: CommandRecord, receipt: BackendCommandReceipt) -> CommandOutcome:
        if receipt.command_id != pending.command_id:
            self._mark_inconclusive(pending)
            raise ValueError("backend receipt command_id does not match the dispatched command")
        if (
            receipt.admission is BackendAdmission.ACCEPTED
            and pending.operation is BackendOperation.SUBMIT
            and not receipt.work_id
        ):
            self._mark_inconclusive(pending)
            raise ValueError("an accepted work.submit receipt must include work_id")
        if pending.work_id is not None and receipt.work_id is not None and receipt.work_id != pending.work_id:
            self._mark_inconclusive(pending)
            raise ValueError("backend receipt work_id does not match the targeted Work")

        state = {
            BackendAdmission.ACCEPTED: CommandState.ACCEPTED,
            BackendAdmission.REJECTED: CommandState.REJECTED,
            BackendAdmission.INCONCLUSIVE: CommandState.INCONCLUSIVE,
        }[receipt.admission]
        try:
            admitted = replace(
                pending,
                state=state,
                work_id=receipt.work_id or pending.work_id,
                reason_code=receipt.reason_code,
                updated_at=utc_now(),
            )
        except (TypeError, ValueError) as error:
            self._mark_inconclusive(pending)
            raise ValueError("backend receipt contains invalid admission evidence") from error
        self._state_store.update_command(admitted)
        return CommandOutcome(
            command_id=admitted.command_id,
            state=admitted.state,
            work_id=admitted.work_id,
            reason_code=receipt.reason_code,
        )

    def _mark_inconclusive(self, record: CommandRecord) -> None:
        self._state_store.update_command(replace(record, state=CommandState.INCONCLUSIVE, updated_at=utc_now()))

    @staticmethod
    def _validate_retry_identity(
        existing: CommandRecord,
        *,
        session_id: str,
        commit_id: str,
        operation: BackendOperation,
        arguments: Mapping[str, Any],
    ) -> None:
        if (
            existing.session_id != session_id
            or existing.commit_id != commit_id
            or existing.operation is not operation
            or dict(existing.payload) != dict(arguments)
        ):
            raise ValueError("command identity is already bound to a different request")

    @staticmethod
    def _work_command(record: CommandRecord) -> WorkCommand:
        return WorkCommand(
            command_id=record.command_id,
            session_id=record.session_id,
            attachment_id=record.attachment_id,
            commit_id=record.commit_id,
            operation=record.operation,
            capability_revision=record.capability_revision,
            payload=record.payload,
        )

    @staticmethod
    def _outcome(record: CommandRecord) -> CommandOutcome:
        return CommandOutcome(
            command_id=record.command_id,
            state=record.state,
            work_id=record.work_id,
            reason_code=record.reason_code,
        )

    def _attachment(self, session_id: str) -> BackendAttachment:
        try:
            return self._attachments[session_id]
        except KeyError as error:
            raise SessionNotAttachedError(session_id) from error
