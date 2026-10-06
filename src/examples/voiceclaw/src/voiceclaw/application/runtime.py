# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Application Interaction Manager for realtime sessions and response-only adapters."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from voiceclaw.application.context import (
    ConversationTurn,
    FrontendContextProjector,
    FrontendContextSnapshot,
    LocalRequestContext,
)
from voiceclaw.application.routing import ModelSelectedTurnRoutingPolicy, TurnRoutingPolicy
from voiceclaw.domain.capabilities import CapabilityToolRegistry, SemanticTool
from voiceclaw.domain.models import (
    Durability,
    EventDelivery,
    FrontendActivity,
    SessionBinding,
)
from voiceclaw.domain.response_only import (
    RUNTIME_PROJECTION_SCHEMA,
    ResponseOnlyContextContinuity,
    ResponseOnlyRequestState,
    ResponseOnlyResultEventKind,
    ResponseOnlyResultState,
    ResponseOnlySpeechSource,
    ResponseOnlyTerminalOutcome,
    ResponseOnlyUpdateKind,
)
from voiceclaw.interaction_profiles import (
    INTERACTION_PROFILE_SCHEMA,
    InteractionProfile,
    load_interaction_profile_catalog,
)
from voiceclaw.model_contracts import ModelContractCatalog, load_model_contract_catalog
from voiceclaw.ports.runtime import (
    BackendTargetSnapshot,
    FrontendContextPurpose,
    FrontendContextRequest,
    FrontendConversationDeliveryState,
    FrontendConversationTurn,
    FrontendPlaybackReceipt,
    FrontendPlaybackReceiptState,
    FrontendResponse,
    FrontendResponsePurpose,
    FrontendSpeechDeliveryOutcome,
    InteractionUpdate,
    SessionSnapshot,
    TurnDirective,
    TurnDirectiveKind,
)
from voiceclaw.ports.state import StateStore
from voiceclaw.ports.turns import (
    MAX_COMMITTED_TURN_GOAL_BYTES,
    CommittedTurnBackend,
    CommittedTurnCompleted,
    CommittedTurnDisplayDelta,
    CommittedTurnError,
    CommittedTurnRequest,
    CommittedTurnResult,
    EphemeralCommittedTurnPort,
)

_RECENT_CONVERSATION_LIMIT = 8
_RECENT_CONVERSATION_TEXT_LIMIT = 512
_PLAYBACK_RECEIPT_LIMIT = 8


class _ResultPresentationState(StrEnum):
    """VoiceClaw-local speech presentation state for one available result."""

    NOT_REQUESTED = "not_requested"
    READY = "ready"
    AWAITING_PLAYBACK_RECEIPT = "awaiting_playback_receipt"
    TERMINAL = "terminal"


def _request_summary(text: str, *, character_limit: int) -> str:
    """Capture one stable, bounded description of a local request."""
    compact = " ".join(text.split()).replace("\x00", "")
    if len(compact) <= character_limit:
        return compact
    clipped = compact[: character_limit + 1]
    boundary = clipped.rfind(" ")
    return clipped[:boundary].rstrip() if boundary > 0 else compact[:character_limit]


def _result_frontend_response(
    result: CommittedTurnResult,
    *,
    local_request_id: str,
) -> FrontendResponse | None:
    """Route only backend-authorized presentation material, never Markdown."""
    if result.speak_text is None:
        return None
    return FrontendResponse(
        purpose=FrontendResponsePurpose.RESULT_DELIVERY,
        local_request_id=local_request_id,
        payload_text=result.speak_text,
    )


def _speech_source(result: CommittedTurnResult) -> ResponseOnlySpeechSource:
    """Describe who authorized presentation material without copying its content."""
    if result.speak_text is not None:
        return ResponseOnlySpeechSource.BACKEND_AUTHORED
    return ResponseOnlySpeechSource.NONE


@dataclass(frozen=True, slots=True)
class _LatestResult:
    """Body-free metadata retained for the next frontend response boundary."""

    state: ResponseOnlyResultState = ResponseOnlyResultState.NONE
    display_available: bool = False
    speech_source: ResponseOnlySpeechSource = ResponseOnlySpeechSource.NONE
    presentation_state: _ResultPresentationState = _ResultPresentationState.NOT_REQUESTED
    delivery_state: FrontendConversationDeliveryState | None = None
    presentation_id: str | None = None
    turn_id: str | None = None


@dataclass(frozen=True, slots=True)
class _LocalRequestAdmission:
    """Immutable evidence captured before backend dispatch can change state."""

    request_summary: str
    operation: str
    projection_revision: int
    state: ResponseOnlyRequestState = ResponseOnlyRequestState.LOCALLY_QUEUED
    backend_acceptance: str = "unknown"
    durability: Durability = Durability.NONE


@dataclass(slots=True)
class _LocalRequestProjection:
    """Bounded state for one VoiceClaw-local, explicitly non-durable request."""

    local_request_id: str
    request_summary: str = "Request details unavailable"
    operation: str = "work.delegate"
    phase: ResponseOnlyRequestState = ResponseOnlyRequestState.LOCALLY_QUEUED
    terminal_outcome: ResponseOnlyTerminalOutcome = ResponseOnlyTerminalOutcome.NONE
    failure_code: str | None = None
    result: _LatestResult = field(default_factory=_LatestResult)
    admission: _LocalRequestAdmission | None = None


@dataclass(slots=True)
class _LocalRequestGuardState:
    """Track whether an admitted request crossed the backend effects boundary."""

    invocation_started: bool = False


@dataclass(slots=True)
class _SessionState:
    session_id: str
    conversation_id: str
    activity: FrontendActivity = field(default_factory=lambda: FrontendActivity(connected=True))
    operation: str = "idle"
    phase: ResponseOnlyRequestState = ResponseOnlyRequestState.IDLE
    backend: CommittedTurnBackend | None = None
    frontend_tools: tuple[SemanticTool, ...] = ()
    last_correlation: dict[str, str] = field(default_factory=dict)
    latest_terminal_outcome: ResponseOnlyTerminalOutcome = ResponseOnlyTerminalOutcome.NONE
    latest_failure_code: str | None = None
    latest_result: _LatestResult = field(default_factory=_LatestResult)
    latest_local_request_id: str | None = None
    local_requests: dict[str, _LocalRequestProjection] = field(default_factory=dict)
    recent_conversation: dict[str, FrontendConversationTurn] = field(default_factory=dict)
    playback_receipts: dict[str, FrontendPlaybackReceipt] = field(default_factory=dict)
    heard_through_presentation_id: str | None = None
    heard_through_turn_id: str | None = None
    latest_playback_receipt: FrontendPlaybackReceipt | None = None
    context_revision: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class RealtimeInteractionManager:
    """Coordinate frontend state without pretending response-only APIs own Work.

    Durable ``AgentInteractionPort`` Work continues through
    ``InteractionCoordinator`` and ``BackendEventCoordinator``. The response-only
    committed-turn surface handled here is intentionally
    marked non-durable and never receive Work IDs, replay cursors, or receipts.
    """

    def __init__(
        self,
        *,
        backend_profile: str,
        state_store: StateStore,
        committed_turns: EphemeralCommittedTurnPort | None,
        turn_routing_policy: TurnRoutingPolicy | None = None,
        request_summary_character_limit: int = 512,
        retained_request_limit: int = 8,
        model_contracts: ModelContractCatalog | None = None,
        interaction_profile: InteractionProfile | None = None,
        interaction_profile_schema: str = INTERACTION_PROFILE_SCHEMA,
        interaction_profile_hash: str | None = None,
    ) -> None:
        """Bind application ports while keeping protocol and provider objects out."""
        self._backend_profile = backend_profile
        self._state_store = state_store
        self._committed_turns = committed_turns
        self._turn_routing_policy = turn_routing_policy or ModelSelectedTurnRoutingPolicy()
        if request_summary_character_limit < 1 or retained_request_limit < 1:
            raise ValueError("interaction projection limits must be positive")
        self._request_summary_character_limit = request_summary_character_limit
        self._retained_request_limit = retained_request_limit
        self._model_contracts = model_contracts or load_model_contract_catalog()
        if interaction_profile is None:
            catalog = load_interaction_profile_catalog()
            interaction_profile = catalog.resolve("stateless")
            interaction_profile_schema = catalog.schema_version
            interaction_profile_hash = interaction_profile.resolved_digest
        self._interaction_profile = interaction_profile
        self._interaction_profile_schema = interaction_profile_schema
        self._interaction_profile_hash = interaction_profile_hash
        self._tool_registry = CapabilityToolRegistry(profile=interaction_profile)
        self._sessions: dict[str, _SessionState] = {}
        # The response-only profile admits one active backend exchange
        # process-wide. Keep that compatibility constraint outside the frontend
        # protocol and each individual session lock.
        self._backend_admission_lock = asyncio.Lock()

    async def open_session(self, session_id: str, conversation_id: str) -> SessionSnapshot:
        """Verify the configured target, then persist the local session mapping."""
        backend: CommittedTurnBackend | None = None
        if self._committed_turns is not None:
            try:
                backend = await self._committed_turns.inspect()
            except Exception:
                # The voice frontend must remain usable when the optional
                # server-side agent target is temporarily unavailable.
                backend = None
        frontend_tools: tuple[SemanticTool, ...] = ()
        if backend is not None:
            frontend_tools = self._tool_registry.project(backend.capabilities)
        state = _SessionState(
            session_id=session_id,
            conversation_id=conversation_id,
            backend=backend,
            frontend_tools=frontend_tools,
        )
        self._sessions[session_id] = state
        self._state_store.save_session(
            SessionBinding(
                session_id=session_id,
                conversation_id=conversation_id,
                backend_profile=self._backend_profile,
            )
        )
        return SessionSnapshot(
            session_id=session_id,
            conversation_id=conversation_id,
            backend_profile=self._backend_profile,
            recoverable_inflight=False,
            projection=self.projection(session_id),
            backend_label=backend.capabilities.target_label if backend is not None else "Backend unavailable",
            backend_mode=backend.capabilities.backend_kind if backend is not None else "disabled",
            target_binding_verified=backend is not None,
            target_ref=backend.target_ref if backend is not None else "not-attached",
            target_availability=backend.target_availability if backend is not None else None,
            capabilities=(
                tuple(sorted(operation.value for operation in backend.capabilities.operations))
                if backend is not None
                else ()
            ),
            frontend_tools=frontend_tools,
            durability=backend.capabilities.durability if backend is not None else Durability.NONE,
            event_delivery=(
                backend.capabilities.event_delivery if backend is not None else EventDelivery.RESPONSE_ONLY
            ),
            max_parallel_work=backend.capabilities.max_parallel_work if backend is not None else 1,
            context_continuity=(
                backend.context_continuity if backend is not None else ResponseOnlyContextContinuity.NONE
            ),
            capability_source=backend.capability_evidence.source if backend is not None else None,
            capability_source_id=backend.capability_evidence.source_id if backend is not None else None,
            capability_revision=backend.capability_evidence.revision if backend is not None else None,
            capability_hash=backend.capability_evidence.digest if backend is not None else None,
            model_contract_schema=self._model_contracts.schema_version,
            model_contract_profile=self._model_contracts.profile,
            model_contract_hash=self._model_contracts.digest,
            interaction_profile_schema=self._interaction_profile_schema,
            interaction_profile_name=self._interaction_profile.name,
            interaction_profile_hash=self._interaction_profile_hash,
        )

    def projection(self, session_id: str) -> str:
        """Return a bounded, capability-gated snapshot for one response boundary."""
        state = self._session(session_id)
        target_binding_state = (
            "verified"
            if state.backend is not None
            else ("unavailable" if self._committed_turns is not None else "disabled")
        )
        target_availability = state.backend.target_availability if state.backend is not None else None
        attachment_state = "not_persistent" if state.backend is not None else target_binding_state
        typed_capabilities = state.backend.capabilities if state.backend is not None else None
        capability_evidence = state.backend.capability_evidence if state.backend is not None else None
        backend_capabilities = (
            sorted(operation.value for operation in typed_capabilities.operations)
            if typed_capabilities is not None
            else []
        )
        frontend_tools = [tool.name for tool in state.frontend_tools]
        active_request = any(request.phase.active for request in state.local_requests.values())
        latest_request = (
            state.local_requests.get(state.latest_local_request_id)
            if state.latest_local_request_id is not None
            else None
        )
        local_requests = [
            {
                "local_request_id": request.local_request_id,
                "request_summary": request.request_summary,
                "identity_authority": "voiceclaw_local",
                "operation": request.operation,
                "request_state": request.phase,
                "request_active": request.phase.active,
                "durability": "none",
                "durable_work_id_issued": False,
                "terminal_outcome": request.terminal_outcome,
                "failure_code": request.failure_code,
                "result": {
                    "state": request.result.state,
                    "display_available": request.result.display_available,
                    "speech_source": request.result.speech_source,
                },
            }
            for request in state.local_requests.values()
        ]
        active_local_requests = [request for request in local_requests if request["request_active"]]
        ready_results = [
            {
                "local_request_id": request.local_request_id,
                "identity_authority": "voiceclaw_local",
                "state": request.result.state,
                "display_available": request.result.display_available,
                "speech_source": request.result.speech_source,
                "delivery_state": self._result_delivery_state(request.result),
            }
            for request in state.local_requests.values()
            if self._result_available(request.result)
        ]
        recent_conversation = []
        for turn in state.recent_conversation.values():
            receipt = state.playback_receipts.get(turn.turn_id)
            recent_conversation.append(
                {
                    "turn_id": turn.turn_id,
                    "role": turn.role,
                    "text": turn.text,
                    "text_truncated": turn.text_truncated,
                    "delivery_state": turn.delivery_state,
                    "presentation_id": turn.presentation_id,
                    "local_request_id": turn.local_request_id,
                    "heard_through_ms": None if receipt is None else receipt.heard_through_ms,
                    "audio_end_ms": None if receipt is None else receipt.audio_end_ms,
                }
            )
        latest_receipt = state.latest_playback_receipt
        authoritative_receipt = (
            latest_receipt is not None and latest_receipt.state is not FrontendPlaybackReceiptState.FAILED
        )
        payload = {
            "schema": RUNTIME_PROJECTION_SCHEMA,
            "session_id": state.session_id,
            "backend_profile": self._backend_profile,
            "backend_mode": typed_capabilities.backend_kind if typed_capabilities is not None else "disabled",
            "target_binding_verified": state.backend is not None,
            "backend_label": (
                typed_capabilities.target_label if typed_capabilities is not None else "Backend unavailable"
            ),
            "capabilities": backend_capabilities,
            "frontend_tools": frontend_tools,
            "durability": typed_capabilities.durability if typed_capabilities is not None else Durability.NONE,
            "recoverable_inflight": False,
            "operation": state.operation,
            "phase": state.phase,
            "connection": {
                "frontend_state": "connected" if state.activity.connected else "disconnected",
                "target_binding_state": target_binding_state,
                "target_state": target_availability,
                "attachment_state": attachment_state,
                "attachment_mode": "terminal_invocation" if self._committed_turns is not None else "none",
                "reconnect_mode": "fresh_frontend_session",
                "backend_context_continuity": (
                    state.backend.context_continuity
                    if state.backend is not None
                    else ResponseOnlyContextContinuity.NONE
                ),
                "resume_supported": False,
                "recoverable_inflight": False,
            },
            "contract": {
                "capability_source": capability_evidence.source if capability_evidence is not None else None,
                "capability_source_id": capability_evidence.source_id if capability_evidence is not None else None,
                "capability_revision": capability_evidence.revision if capability_evidence is not None else None,
                "capability_hash": capability_evidence.digest if capability_evidence is not None else None,
                "backend_capabilities": backend_capabilities,
                "operations": backend_capabilities,
                "frontend_tools": frontend_tools,
                "durability": typed_capabilities.durability if typed_capabilities is not None else Durability.NONE,
                "event_delivery": (
                    typed_capabilities.event_delivery if typed_capabilities is not None else EventDelivery.RESPONSE_ONLY
                ),
                "interaction_profile_schema": self._interaction_profile_schema,
                "interaction_profile_name": self._interaction_profile.name,
                "interaction_profile_hash": self._interaction_profile_hash,
                "session_scope": self._interaction_profile.session_scope,
                "work_cardinality": self._interaction_profile.work_cardinality,
                "sessionful": typed_capabilities.sessionful if typed_capabilities is not None else False,
                "supports_parallel_work": (
                    typed_capabilities.supports_parallel_work if typed_capabilities is not None else False
                ),
                "max_parallel_requests": (
                    typed_capabilities.max_parallel_work if typed_capabilities is not None else 1
                ),
                "model_contract_schema": self._model_contracts.schema_version,
                "model_contract_profile": self._model_contracts.profile,
                "model_contract_hash": self._model_contracts.digest,
            },
            "execution": {
                "local_request_id": state.latest_local_request_id,
                "request_summary": latest_request.request_summary if latest_request is not None else None,
                "identity_authority": "voiceclaw_local" if state.latest_local_request_id is not None else "none",
                "operation": state.operation,
                "request_state": state.phase,
                "request_active": (latest_request is not None and latest_request.phase.active),
                "any_request_active": active_request,
                "durable_work_id_issued": False,
                "latest_terminal_outcome": state.latest_terminal_outcome,
                "latest_failure_code": state.latest_failure_code,
            },
            "local_requests": local_requests,
            # The response-only adapter has no authoritative Work or AgentQuery
            # surface. Keep durable categories explicit and empty instead of
            # promoting VoiceClaw-local request evidence into backend state.
            "active_work": [],
            "active_local_requests": active_local_requests,
            "pending_interactions": [],
            "ready_results": ready_results,
            "recent_conversation": recent_conversation,
            "heard_through_presentation_id": state.heard_through_presentation_id,
            "heard_through_turn_id": state.heard_through_turn_id,
            "latest_result": {
                "state": state.latest_result.state,
                "display_available": state.latest_result.display_available,
                "display_body_in_context": False,
                "speech_source": state.latest_result.speech_source,
                "provider_identifiers_in_context": False,
            },
            "delivery": {
                "speech_policy": "approved_payload_only",
                "speech_handoff_state": "not_observed" if latest_receipt is None else "client_reported",
                "playback_state": "unknown" if latest_receipt is None else latest_receipt.state,
                "heard_state": "unknown" if latest_receipt is None else latest_receipt.state,
                "authoritative_client_receipt": authoritative_receipt,
                "presentation_id": None if latest_receipt is None else latest_receipt.presentation_id,
                "turn_id": None if latest_receipt is None else latest_receipt.turn_id,
                "local_request_id": None if latest_receipt is None else latest_receipt.local_request_id,
                "heard_through_ms": None if latest_receipt is None else latest_receipt.heard_through_ms,
                "audio_end_ms": None if latest_receipt is None else latest_receipt.audio_end_ms,
            },
            "activity": {
                "connected": state.activity.connected,
                "input": state.activity.input.value,
                "model": state.activity.model.value,
                "output": state.activity.output.value,
            },
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    def frontend_context(
        self,
        session_id: str,
        request: FrontendContextRequest,
        *,
        maximum_characters: int,
    ) -> str:
        """Return only the application state needed for one model purpose."""
        if not isinstance(request, FrontendContextRequest):
            raise TypeError("request must be FrontendContextRequest")
        state = self._session(session_id)
        target: LocalRequestContext | None = None
        projection_revision = state.context_revision
        if request.local_request_id is not None:
            local_request = state.local_requests.get(request.local_request_id)
            if local_request is None:
                raise LookupError("frontend context references an unknown local request")
            if request.purpose is FrontendContextPurpose.DELEGATION_ACK:
                if local_request.admission is None:
                    raise LookupError("local request has no acknowledgement admission snapshot")
                target = self._admission_context(local_request.admission)
                projection_revision = local_request.admission.projection_revision
            else:
                target = self._local_request_context(local_request)

        active_requests = tuple(
            self._local_request_context(local_request)
            for local_request in state.local_requests.values()
            if local_request.phase.active
        )
        ready_results = tuple(
            self._local_request_context(local_request)
            for local_request in state.local_requests.values()
            if self._result_ready_for_speech(local_request.result)
        )
        recent_conversation = tuple(
            self._conversation_context(state, turn) for turn in state.recent_conversation.values()
        )
        capabilities = (
            tuple(sorted(operation.value for operation in state.backend.capabilities.operations))
            if state.backend is not None
            else ()
        )
        snapshot = FrontendContextSnapshot(
            purpose=request.purpose,
            projection_revision=projection_revision,
            target_request=target,
            capabilities=capabilities,
            active_local_requests=active_requests,
            ready_results=ready_results,
            recent_conversation=recent_conversation,
            heard_through_presentation_id=state.heard_through_presentation_id,
            heard_through_turn_id=state.heard_through_turn_id,
            voice_activity=state.activity,
        )
        return FrontendContextProjector(maximum_characters=maximum_characters).project(snapshot)

    @staticmethod
    def _admission_context(admission: _LocalRequestAdmission) -> LocalRequestContext:
        """Convert immutable local-admission evidence without reading later state."""
        return LocalRequestContext(
            summary=admission.request_summary,
            operation=admission.operation,
            state=admission.state,
            backend_acceptance=admission.backend_acceptance,
            durability=admission.durability,
        )

    @classmethod
    def _local_request_context(cls, request: _LocalRequestProjection) -> LocalRequestContext:
        """Normalize mutable local state without promoting it to backend Work."""
        return LocalRequestContext(
            summary=request.request_summary,
            operation=request.operation,
            state=request.phase,
            terminal_outcome=request.terminal_outcome,
            failure_code=request.failure_code,
            result_state=request.result.state,
            display_available=request.result.display_available,
            speech_source=request.result.speech_source,
            delivery_state=cls._result_delivery_state(request.result),
        )

    @staticmethod
    def _conversation_context(state: _SessionState, turn: FrontendConversationTurn) -> ConversationTurn:
        """Copy only bounded public conversation and playback evidence."""
        receipt = state.playback_receipts.get(turn.turn_id)
        return ConversationTurn(
            turn_id=turn.turn_id,
            role=turn.role,
            text=turn.text,
            delivery_state=turn.delivery_state,
            presentation_id=turn.presentation_id,
            local_request_id=turn.local_request_id,
            heard_through_ms=None if receipt is None else receipt.heard_through_ms,
            audio_end_ms=None if receipt is None else receipt.audio_end_ms,
        )

    @staticmethod
    def _result_ready_for_speech(result: _LatestResult) -> bool:
        """Return whether semantic speech material still awaits presentation."""
        return (
            result.state is ResponseOnlyResultState.AVAILABLE
            and result.speech_source is ResponseOnlySpeechSource.BACKEND_AUTHORED
            and result.presentation_state is _ResultPresentationState.READY
            and result.delivery_state is None
        )

    @staticmethod
    def _result_available(result: _LatestResult) -> bool:
        """Return whether normalized result evidence remains locally available."""
        return result.state is ResponseOnlyResultState.AVAILABLE

    def record_conversation_turn(self, session_id: str, turn: FrontendConversationTurn) -> None:
        """Retain one bounded public turn without creating backend Work evidence."""
        if not isinstance(turn, FrontendConversationTurn):
            raise TypeError("turn must be FrontendConversationTurn")
        state = self._session(session_id)
        bounded = turn
        if len(turn.text) > _RECENT_CONVERSATION_TEXT_LIMIT:
            bounded = replace(
                turn,
                text=f"{turn.text[: _RECENT_CONVERSATION_TEXT_LIMIT - 1]}…",
                text_truncated=True,
            )
        receipt = state.playback_receipts.get(bounded.turn_id)
        if receipt is not None:
            if bounded.role != "assistant":
                raise ValueError("a playback receipt cannot correlate to a user conversation turn")
            if bounded.presentation_id is not None and receipt.presentation_id != bounded.presentation_id:
                raise ValueError("conversation turn presentation_id does not match its playback receipt")
            if (
                bounded.local_request_id is not None
                and receipt.local_request_id is not None
                and receipt.local_request_id != bounded.local_request_id
            ):
                raise ValueError("conversation turn local_request_id does not match its playback receipt")
            bounded = replace(
                bounded,
                delivery_state=FrontendConversationDeliveryState(receipt.state.value),
                presentation_id=receipt.presentation_id,
                local_request_id=receipt.local_request_id or bounded.local_request_id,
            )
        if bounded.local_request_id is not None and bounded.local_request_id not in state.local_requests:
            raise ValueError("conversation turn references an unknown local request")
        existing = state.recent_conversation.get(bounded.turn_id)
        if existing is not None:
            if existing == bounded:
                return
            raise ValueError("conversation turn identity was reused with different evidence")
        state.recent_conversation[bounded.turn_id] = bounded
        self._update_result_delivery_from_turn(state, bounded)
        while len(state.recent_conversation) > _RECENT_CONVERSATION_LIMIT:
            oldest_turn_id = next(iter(state.recent_conversation))
            state.recent_conversation.pop(oldest_turn_id)
            state.playback_receipts.pop(oldest_turn_id, None)
        state.context_revision += 1

    def record_playback_receipt(self, session_id: str, receipt: FrontendPlaybackReceipt) -> None:
        """Apply an idempotent receipt without inventing missing transcript text."""
        if not isinstance(receipt, FrontendPlaybackReceipt):
            raise TypeError("receipt must be FrontendPlaybackReceipt")
        state = self._session(session_id)
        turn = state.recent_conversation.get(receipt.turn_id)
        if turn is not None and turn.role != "assistant":
            raise ValueError("playback receipts require an assistant conversation turn")
        if turn is not None and turn.presentation_id is not None and receipt.presentation_id != turn.presentation_id:
            raise ValueError("playback receipt presentation_id does not match the conversation turn")
        if (
            turn is not None
            and receipt.local_request_id is not None
            and turn.local_request_id is not None
            and receipt.local_request_id != turn.local_request_id
        ):
            raise ValueError("playback receipt local_request_id does not match the conversation turn")
        correlated_local_request_id = receipt.local_request_id or (None if turn is None else turn.local_request_id)
        if correlated_local_request_id is not None and correlated_local_request_id not in state.local_requests:
            raise ValueError("playback receipt references an unknown local request")
        existing = state.playback_receipts.get(receipt.turn_id)
        if existing is not None:
            if existing == receipt:
                return
            raise ValueError("conversation turn already has a different terminal playback receipt")

        delivery_state = FrontendConversationDeliveryState(receipt.state.value)
        updated_turn: FrontendConversationTurn | None = None
        if turn is not None:
            updated_turn = replace(
                turn,
                delivery_state=delivery_state,
                presentation_id=receipt.presentation_id or turn.presentation_id,
                local_request_id=receipt.local_request_id or turn.local_request_id,
            )
            state.recent_conversation[receipt.turn_id] = updated_turn
        state.playback_receipts[receipt.turn_id] = receipt
        while len(state.playback_receipts) > _PLAYBACK_RECEIPT_LIMIT:
            oldest_turn_id = next(iter(state.playback_receipts))
            state.playback_receipts.pop(oldest_turn_id)
        state.latest_playback_receipt = receipt
        if delivery_state in {
            FrontendConversationDeliveryState.HEARD,
            FrontendConversationDeliveryState.INTERRUPTED,
        }:
            state.heard_through_turn_id = receipt.turn_id
            state.heard_through_presentation_id = (
                receipt.presentation_id if updated_turn is None else updated_turn.presentation_id
            )
        if updated_turn is not None:
            self._update_result_delivery_from_turn(state, updated_turn)
        else:
            self._update_result_delivery_from_receipt(state, receipt)
        state.context_revision += 1

    def record_speech_delivery_outcome(
        self,
        session_id: str,
        outcome: FrontendSpeechDeliveryOutcome,
    ) -> None:
        """Apply a terminal no-audio outcome without fabricating client evidence."""
        if not isinstance(outcome, FrontendSpeechDeliveryOutcome):
            raise TypeError("outcome must be FrontendSpeechDeliveryOutcome")
        state = self._session(session_id)
        request = state.local_requests.get(outcome.local_request_id)
        if request is None:
            raise ValueError("speech delivery outcome references an unknown local request")
        if request.result.state is not ResponseOnlyResultState.AVAILABLE:
            raise ValueError("speech delivery outcome requires an available result")
        delivery_state = FrontendConversationDeliveryState(outcome.state.value)
        if request.result.delivery_state is not None:
            if (
                request.result.delivery_state is delivery_state
                and request.result.presentation_id == outcome.presentation_id
            ):
                return
            raise ValueError("result already has a different terminal speech outcome")
        updated_result = replace(
            request.result,
            presentation_state=_ResultPresentationState.TERMINAL,
            delivery_state=delivery_state,
            presentation_id=outcome.presentation_id,
        )
        request.result = updated_result
        if state.latest_local_request_id == outcome.local_request_id:
            state.latest_result = updated_result
        state.context_revision += 1

    @staticmethod
    def _result_delivery_state(result: _LatestResult) -> str:
        """Return a truthful local speech-delivery state for an available result."""
        if result.delivery_state is not None:
            return result.delivery_state.value
        return result.presentation_state.value

    @staticmethod
    def _update_result_delivery_from_turn(state: _SessionState, turn: FrontendConversationTurn) -> None:
        """Correlate public delivery evidence without promoting it to backend state."""
        local_request_id = turn.local_request_id
        if local_request_id is None:
            return
        request = state.local_requests.get(local_request_id)
        if request is None:
            raise ValueError("conversation turn references an unknown local request")
        if request.result.state is not ResponseOnlyResultState.AVAILABLE:
            return
        if turn.delivery_state is FrontendConversationDeliveryState.DELIVERED:
            updated_result = replace(
                request.result,
                presentation_state=_ResultPresentationState.AWAITING_PLAYBACK_RECEIPT,
                presentation_id=turn.presentation_id,
                turn_id=turn.turn_id,
            )
        else:
            updated_result = replace(
                request.result,
                presentation_state=_ResultPresentationState.TERMINAL,
                delivery_state=turn.delivery_state,
                presentation_id=turn.presentation_id,
                turn_id=turn.turn_id,
            )
        request.result = updated_result
        if state.latest_local_request_id == local_request_id:
            state.latest_result = updated_result

    @staticmethod
    def _update_result_delivery_from_receipt(state: _SessionState, receipt: FrontendPlaybackReceipt) -> None:
        """Correlate receipt-only delivery evidence without synthesizing a turn."""
        local_request_id = receipt.local_request_id
        if local_request_id is None:
            return
        request = state.local_requests.get(local_request_id)
        if request is None:
            raise ValueError("playback receipt references an unknown local request")
        if request.result.state is not ResponseOnlyResultState.AVAILABLE:
            return
        updated_result = replace(
            request.result,
            presentation_state=_ResultPresentationState.TERMINAL,
            delivery_state=FrontendConversationDeliveryState(receipt.state.value),
            presentation_id=receipt.presentation_id,
            turn_id=receipt.turn_id,
        )
        request.result = updated_result
        if state.latest_local_request_id == local_request_id:
            state.latest_result = updated_result

    def route_finalized_turn(self, session_id: str, text: str) -> TurnDirective:
        """Route only against the current session's advertised semantic tools."""
        state = self._session(session_id)
        if not isinstance(text, str) or not text.strip():
            return TurnDirective.reject(reason_code="empty_finalized_turn")
        directive = self._turn_routing_policy.decide(text.strip())
        if directive.kind is not TurnDirectiveKind.TOOL:
            return directive
        advertised = {tool.name for tool in state.frontend_tools}
        if directive.logical_tool not in advertised:
            return TurnDirective.reject(reason_code="operation_unavailable")
        return directive

    async def update_activity(self, session_id: str, activity: FrontendActivity) -> None:
        """Feed the same independent activity axes used by the safe speech queue."""
        state = self._session(session_id)
        if state.activity == activity:
            return
        state.activity = activity
        state.context_revision += 1

    async def execute_tool(
        self,
        *,
        session_id: str,
        commit_id: str,
        call_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        finalized_user_text: str | None,
    ) -> AsyncIterator[InteractionUpdate]:
        """Execute one advertised Work tool without exposing the backend adapter."""
        state = self._session(session_id)
        advertised = {tool.name for tool in state.frontend_tools}
        if tool_name not in advertised:
            yield self._failed_turn(state, call_id, commit_id, "capability_unsupported")
            return
        try:
            validated_arguments = self._tool_registry.validate_arguments(tool_name, arguments)
        except (KeyError, ValueError):
            yield self._failed_turn(state, call_id, commit_id, "invalid_tool_arguments")
            return
        if tool_name != "work.delegate":
            yield self._failed_turn(state, call_id, commit_id, "capability_unsupported")
            return
        if not isinstance(finalized_user_text, str) or not finalized_user_text.strip():
            yield self._failed_turn(state, call_id, commit_id, "missing_finalized_user_turn")
            return
        try:
            source_turn_bytes = len(finalized_user_text.encode("utf-8"))
        except UnicodeEncodeError:
            source_turn_bytes = MAX_COMMITTED_TURN_GOAL_BYTES + 1
        if "\x00" in finalized_user_text or source_turn_bytes > MAX_COMMITTED_TURN_GOAL_BYTES:
            yield self._failed_turn(state, call_id, commit_id, "invalid_tool_arguments")
            return
        delegated_goal = validated_arguments.get("goal")
        if not isinstance(delegated_goal, str):
            yield self._failed_turn(state, call_id, commit_id, "invalid_tool_arguments")
            return
        # Retain the source turn locally; send only the validated standalone goal to the backend.
        request_summary = _request_summary(
            delegated_goal,
            character_limit=self._request_summary_character_limit,
        )
        if self._committed_turns is None:
            yield self._failed_turn(
                state,
                call_id,
                commit_id,
                "backend_not_configured",
                request_summary=request_summary,
            )
            return
        if state.lock.locked():
            yield self._failed_turn(
                state,
                call_id,
                commit_id,
                "session_busy",
                request_summary=request_summary,
            )
            return
        guard = _LocalRequestGuardState()
        async with state.lock, self._guard_local_request(state, call_id, commit_id, guard):
            state.operation = "work.delegate"
            # Publish a prompt local receipt before awaiting the shared backend
            # lane.  Only the holder of that lane may claim ``dispatching``.
            state.phase = ResponseOnlyRequestState.LOCALLY_QUEUED
            state.latest_terminal_outcome = ResponseOnlyTerminalOutcome.NONE
            state.latest_failure_code = None
            state.latest_result = _LatestResult()
            self._touch_local_request(
                state,
                commit_id,
                phase=ResponseOnlyRequestState.LOCALLY_QUEUED,
                request_summary=request_summary,
                terminal_outcome=ResponseOnlyTerminalOutcome.NONE,
                failure_code=None,
                result=_LatestResult(),
            )
            state.last_correlation = self._backend_correlation(
                state,
                call_id=call_id,
                commit_id=commit_id,
            )
            yield InteractionUpdate(
                kind=ResponseOnlyUpdateKind.BACKEND_TURN,
                phase=state.phase,
                title="Request queued locally",
                text="VoiceClaw locally queued this non-durable request for backend dispatch.",
                correlation=state.last_correlation,
                request_summary=request_summary,
                tool_output={
                    "status": ResponseOnlyRequestState.LOCALLY_QUEUED,
                    "request_state": ResponseOnlyRequestState.LOCALLY_QUEUED,
                    "local_request_id": commit_id,
                    "identity_authority": "voiceclaw_local",
                    "evidence": "voiceclaw_local",
                    "backend_acceptance": "unknown",
                    "durability": "none",
                },
                frontend_response=FrontendResponse(
                    purpose=FrontendResponsePurpose.DELEGATION_ACK,
                    payload_text=None,
                    local_request_id=commit_id,
                ),
            )
            async with self._backend_admission_lock:
                state.phase = ResponseOnlyRequestState.DISPATCHING
                self._touch_local_request(state, commit_id, phase=ResponseOnlyRequestState.DISPATCHING)
                yield InteractionUpdate(
                    kind=ResponseOnlyUpdateKind.BACKEND_TURN,
                    phase=ResponseOnlyRequestState.DISPATCHING,
                    title="Dispatching request",
                    text="VoiceClaw is dispatching the queued request to the configured response-only backend.",
                    correlation=state.last_correlation,
                )
                # This bridge provides no separate durable acceptance or
                # running receipt.  Expose only that VoiceClaw is waiting for
                # the terminal exchange instead of inventing backend state.
                state.phase = ResponseOnlyRequestState.WAITING_FOR_RESPONSE
                self._touch_local_request(state, commit_id, phase=ResponseOnlyRequestState.WAITING_FOR_RESPONSE)
                yield InteractionUpdate(
                    kind=ResponseOnlyUpdateKind.BACKEND_TURN,
                    phase=ResponseOnlyRequestState.WAITING_FOR_RESPONSE,
                    title="Waiting for agent response",
                    text=(
                        "VoiceClaw is waiting for the configured target; this terminal invocation does not "
                        "supply a separate durable acceptance or running receipt."
                    ),
                    correlation=state.last_correlation,
                )
                try:
                    result: CommittedTurnResult | None = None
                    display_fragments: list[str] = []
                    expected_display_sequence = 0
                    guard.invocation_started = True
                    async for event in self._committed_turns.stream_turn(
                        CommittedTurnRequest(
                            runtime_conversation_id=state.conversation_id,
                            commit_id=commit_id,
                            text=delegated_goal,
                        )
                    ):
                        if isinstance(event, CommittedTurnDisplayDelta):
                            if event.sequence != expected_display_sequence:
                                raise CommittedTurnError("invocation_result_unusable")
                            expected_display_sequence += 1
                            display_fragments.append(event.delta)
                            yield InteractionUpdate(
                                kind=ResponseOnlyUpdateKind.RESULT_DISPLAY,
                                phase=ResponseOnlyResultEventKind.DISPLAY_DELTA,
                                title="Response",
                                text=event.delta,
                                correlation=self._backend_correlation(
                                    state,
                                    call_id=call_id,
                                    commit_id=commit_id,
                                    backend_session_id=event.backend_session_id,
                                    turn_id=event.turn_id,
                                    response_id=event.response_id,
                                ),
                            )
                            continue
                        if not isinstance(event, CommittedTurnCompleted) or result is not None:
                            raise CommittedTurnError("invocation_result_unusable")
                        result = event.result
                    if result is None or "".join(display_fragments) != result.display_text:
                        raise CommittedTurnError("invocation_result_unusable")
                except CommittedTurnError as error:
                    update = self._failed_turn(state, call_id, commit_id, error.code, tool_output=False)
                    await self._refresh_backend_contract()
                    update = self._with_frontend_contract(update, state)
                    yield update
                    return
                except Exception:
                    update = self._failed_turn(
                        state,
                        call_id,
                        commit_id,
                        "runtime_interrupted",
                        tool_output=False,
                    )
                    await self._refresh_backend_contract()
                    update = self._with_frontend_contract(update, state)
                    yield update
                    return
            if not result.display_text.strip():
                update = self._failed_turn(
                    state,
                    call_id,
                    commit_id,
                    "invocation_result_unusable",
                    tool_output=False,
                )
                await self._refresh_backend_contract()
                update = self._with_frontend_contract(update, state)
                yield update
                return
            speech_source = _speech_source(result)
            latest_result = _LatestResult(
                state=ResponseOnlyResultState.AVAILABLE,
                display_available=True,
                speech_source=speech_source,
                presentation_state=(
                    _ResultPresentationState.READY
                    if speech_source is ResponseOnlySpeechSource.BACKEND_AUTHORED
                    else _ResultPresentationState.NOT_REQUESTED
                ),
            )
            frontend_response = _result_frontend_response(
                result,
                local_request_id=commit_id,
            )
            result_correlation = self._backend_correlation(
                state,
                call_id=call_id,
                commit_id=commit_id,
                backend_session_id=result.backend_session_id,
                turn_id=result.turn_id,
                response_id=result.response_id,
            )
            yield InteractionUpdate(
                kind=ResponseOnlyUpdateKind.RESULT_DISPLAY,
                phase=ResponseOnlyResultEventKind.COMPLETED,
                title="Response",
                text=result.display_text,
                correlation=result_correlation,
            )
            terminal_update = InteractionUpdate(
                kind=ResponseOnlyUpdateKind.BACKEND_TURN,
                phase=ResponseOnlyRequestState.SUCCEEDED,
                title="Response received",
                text="",
                correlation={
                    **result_correlation,
                    "speech_source": speech_source.value,
                },
                frontend_response=frontend_response,
            )
            # Validate every terminal artifact before changing the authoritative
            # local projection to succeeded. If normalization fails, the guard
            # below closes the admitted request as runtime_interrupted.
            state.phase = ResponseOnlyRequestState.SUCCEEDED
            state.latest_terminal_outcome = ResponseOnlyTerminalOutcome.SUCCEEDED
            state.latest_failure_code = None
            state.latest_result = latest_result
            self._touch_local_request(
                state,
                commit_id,
                phase=ResponseOnlyRequestState.SUCCEEDED,
                terminal_outcome=ResponseOnlyTerminalOutcome.SUCCEEDED,
                failure_code=None,
                result=latest_result,
            )
            state.last_correlation = terminal_update.correlation
            await self._refresh_backend_contract()
            terminal_update = self._with_frontend_contract(terminal_update, state)
            yield terminal_update

    async def _refresh_backend_contract(self) -> None:
        """Refresh model-visible operations after a terminal backend exchange."""
        backend: CommittedTurnBackend | None = None
        if self._committed_turns is not None:
            try:
                backend = await self._committed_turns.inspect()
            except Exception:
                backend = None
        frontend_tools = () if backend is None else self._tool_registry.project(backend.capabilities)
        for session in self._sessions.values():
            session.backend = backend
            session.frontend_tools = frontend_tools
            session.context_revision += 1

    @staticmethod
    def _target_snapshot(state: _SessionState) -> BackendTargetSnapshot:
        """Build one atomic target replacement for protocol adapters."""
        backend = state.backend
        if backend is None:
            return BackendTargetSnapshot(
                backend_label="Backend unavailable",
                backend_mode="disabled",
                target_binding_verified=False,
                target_ref="not-attached",
                target_availability=None,
            )
        capabilities = backend.capabilities
        evidence = backend.capability_evidence
        return BackendTargetSnapshot(
            backend_label=capabilities.target_label,
            backend_mode=capabilities.backend_kind,
            target_binding_verified=True,
            target_ref=backend.target_ref,
            target_availability=backend.target_availability,
            capabilities=tuple(sorted(operation.value for operation in capabilities.operations)),
            frontend_tools=state.frontend_tools,
            durability=capabilities.durability,
            event_delivery=capabilities.event_delivery,
            max_parallel_work=capabilities.max_parallel_work,
            context_continuity=backend.context_continuity,
            capability_source=evidence.source,
            capability_source_id=evidence.source_id,
            capability_revision=evidence.revision,
            capability_hash=evidence.digest,
        )

    def _with_frontend_contract(self, update: InteractionUpdate, state: _SessionState) -> InteractionUpdate:
        """Attach one authoritative replacement target snapshot."""
        target = self._target_snapshot(state)
        return replace(update, frontend_tools=target.frontend_tools, backend_target=target)

    @asynccontextmanager
    async def _guard_local_request(
        self,
        state: _SessionState,
        call_id: str,
        commit_id: str,
        guard: _LocalRequestGuardState,
    ) -> AsyncIterator[None]:
        """Close any admitted request whose update stream exits nonterminally."""
        try:
            yield
        finally:
            request = state.local_requests.get(commit_id)
            if request is not None and request.phase.active:
                failure_code = "runtime_interrupted" if guard.invocation_started else "local_request_cancelled"
                self._failed_turn(
                    state,
                    call_id,
                    commit_id,
                    failure_code,
                    tool_output=False,
                )
                if guard.invocation_started:
                    await self._refresh_backend_contract()

    async def close_session(self, session_id: str, reason: str) -> None:
        """Release only the frontend attachment; never reinterpret it as Work cancellation."""
        state = self._session(session_id)
        state.activity = replace(state.activity, connected=False)
        state.operation = "disconnected"
        del reason
        state.phase = ResponseOnlyRequestState.FAILED
        try:
            self._state_store.discard_ephemeral_session(session_id)
        finally:
            self._sessions.pop(session_id, None)

    def _failed_turn(
        self,
        state: _SessionState,
        call_id: str,
        commit_id: str,
        code: str,
        *,
        tool_output: bool = True,
        request_summary: str | None = None,
    ) -> InteractionUpdate:
        uncertain = code in {
            "invocation_outcome_unknown",
            "invocation_result_unusable",
            "runtime_interrupted",
            "target_context_reused",
        }
        terminal_phase = ResponseOnlyRequestState.OUTCOME_UNKNOWN if uncertain else ResponseOnlyRequestState.FAILED
        terminal_outcome = (
            ResponseOnlyTerminalOutcome.OUTCOME_UNKNOWN if uncertain else ResponseOnlyTerminalOutcome.FAILED
        )
        state.operation = "work.delegate"
        state.phase = terminal_phase
        state.latest_terminal_outcome = terminal_outcome
        state.latest_failure_code = code
        state.latest_result = _LatestResult()
        self._touch_local_request(
            state,
            commit_id,
            phase=terminal_phase,
            request_summary=request_summary,
            terminal_outcome=terminal_outcome,
            failure_code=code,
            result=_LatestResult(),
        )
        state.last_correlation = self._backend_correlation(
            state,
            call_id=call_id,
            commit_id=commit_id,
            error_code=code,
        )
        failure_copy = self._model_contracts.failure_copy(code)
        return InteractionUpdate(
            kind=ResponseOnlyUpdateKind.BACKEND_TURN,
            phase=terminal_phase,
            title=failure_copy.title,
            text=failure_copy.display,
            correlation=state.last_correlation,
            request_summary=request_summary,
            tool_output={"status": terminal_phase, "error": code} if tool_output else None,
            frontend_response=FrontendResponse(
                purpose=FrontendResponsePurpose.FAILURE_DELIVERY,
                payload_text=failure_copy.speech,
                local_request_id=commit_id,
            ),
        )

    def _touch_local_request(
        self,
        state: _SessionState,
        local_request_id: str,
        *,
        phase: ResponseOnlyRequestState,
        request_summary: str | None = None,
        terminal_outcome: ResponseOnlyTerminalOutcome | None = None,
        failure_code: str | None = None,
        result: _LatestResult | None = None,
    ) -> _LocalRequestProjection:
        """Update one bounded local request record and make it the freshest projection."""
        request = state.local_requests.pop(local_request_id, None)
        if request is None:
            request = _LocalRequestProjection(
                local_request_id=local_request_id,
                request_summary=request_summary or "Request details unavailable",
            )
        request.phase = phase
        if terminal_outcome is not None:
            request.terminal_outcome = terminal_outcome
        request.failure_code = failure_code
        if result is not None:
            request.result = result
        state.context_revision += 1
        if request.admission is None and phase is ResponseOnlyRequestState.LOCALLY_QUEUED:
            request.admission = _LocalRequestAdmission(
                request_summary=request.request_summary,
                operation=request.operation,
                projection_revision=state.context_revision,
            )
        state.local_requests[local_request_id] = request
        state.latest_local_request_id = local_request_id
        while len(state.local_requests) > self._retained_request_limit:
            oldest_request_id = next(iter(state.local_requests))
            state.local_requests.pop(oldest_request_id)
        return request

    @staticmethod
    def _backend_correlation(state: _SessionState, **values: str) -> dict[str, str]:
        """Keep request and target identities visible without copying user content."""
        correlation = dict(values)
        commit_id = correlation.get("commit_id")
        if commit_id is not None:
            correlation["local_request_id"] = commit_id
            correlation["identity_authority"] = "voiceclaw_local"
        if state.backend is not None:
            correlation.update(
                {
                    "backend_name": state.backend.capabilities.target_label,
                    "backend_mode": state.backend.capabilities.backend_kind,
                    "target_ref": state.backend.target_ref,
                }
            )
        return correlation

    def _session(self, session_id: str) -> _SessionState:
        try:
            return self._sessions[session_id]
        except KeyError as error:
            raise LookupError(f"unknown realtime session: {session_id}") from error


__all__ = ["RealtimeInteractionManager"]
