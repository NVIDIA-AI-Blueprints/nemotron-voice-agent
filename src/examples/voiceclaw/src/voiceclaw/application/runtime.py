# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Application Interaction Manager for realtime sessions and response-only adapters."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field, replace
from typing import Any
from uuid import uuid4

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
    ResponseOnlyResultEnvelope,
    ResponseOnlyResultEventKind,
    ResponseOnlyResultState,
    ResponseOnlySpeechSource,
    ResponseOnlyTerminalOutcome,
    ResponseOnlyUpdateKind,
)
from voiceclaw.domain.volatile import (
    VolatileCancelDisposition,
    VolatileCapacityError,
    VolatileDeliveryOutcome,
    VolatilePlaybackState,
    VolatileRequestNotFoundError,
    VolatileSpeechClaim,
    VolatileSpeechClaimOwner,
    VolatileSpeechState,
    VolatileTransitionError,
    VolatileWorkLedger,
    VolatileWorkSnapshot,
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
    CommittedTurnDispatchMarker,
    CommittedTurnDisplayDelta,
    CommittedTurnError,
    CommittedTurnEvent,
    CommittedTurnRequest,
    CommittedTurnResult,
    EphemeralCommittedTurnPort,
)

_RECENT_CONVERSATION_LIMIT = 8
_RECENT_CONVERSATION_TEXT_LIMIT = 512
_PLAYBACK_RECEIPT_LIMIT = 8
_LOCAL_CONTROL_TEMPLATES = {
    "work.status": "status_delivery",
    "work.cancel": "control_delivery",
}


def _request_summary(text: str, *, character_limit: int) -> str:
    """Capture one stable, bounded description of a local request."""
    compact = " ".join(text.split()).replace("\x00", "")
    if len(compact) <= character_limit:
        return compact
    clipped = compact[: character_limit + 1]
    boundary = clipped.rfind(" ")
    return clipped[:boundary].rstrip() if boundary > 0 else compact[:character_limit]


def _request_state(request: VolatileWorkSnapshot) -> ResponseOnlyRequestState:
    """Translate the canonical volatile lifecycle into the public response-only enum."""
    return ResponseOnlyRequestState(request.state.value)


def _terminal_outcome(request: VolatileWorkSnapshot) -> ResponseOnlyTerminalOutcome:
    """Project a terminal outcome without inventing one for active requests."""
    if request.state.active:
        return ResponseOnlyTerminalOutcome.NONE
    return ResponseOnlyTerminalOutcome(request.state.value)


def _result_state(request: VolatileWorkSnapshot) -> ResponseOnlyResultState:
    return ResponseOnlyResultState.AVAILABLE if request.result is not None else ResponseOnlyResultState.NONE


def _speech_source(request: VolatileWorkSnapshot) -> ResponseOnlySpeechSource:
    if request.result is not None and request.result.envelope.speech is not None:
        return ResponseOnlySpeechSource.BACKEND_AUTHORED
    return ResponseOnlySpeechSource.NONE


def _delivery_state(request: VolatileWorkSnapshot) -> str:
    if request.result is None:
        return VolatileSpeechState.NOT_REQUESTED.value
    if request.result.speech_state is VolatileSpeechState.CLAIMED:
        return VolatileSpeechState.READY.value
    return request.result.speech_state.value


@dataclass(frozen=True, slots=True)
class _LocalRequestAdmission:
    """Immutable evidence captured before backend dispatch can change state."""

    request: VolatileWorkSnapshot
    projection_revision: int


@dataclass(slots=True)
class _LocalRequestController:
    """Process-local cancellation fence for one volatile request."""

    cancel_requested: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass(slots=True)
class _SessionState:
    session_id: str
    conversation_id: str
    activity: FrontendActivity = field(default_factory=lambda: FrontendActivity(connected=True))
    backend: CommittedTurnBackend | None = None
    frontend_tools: tuple[SemanticTool, ...] = ()
    admissions: dict[str, _LocalRequestAdmission] = field(default_factory=dict)
    request_controllers: dict[str, _LocalRequestController] = field(default_factory=dict)
    speech_claims: dict[str, VolatileSpeechClaim] = field(default_factory=dict)
    recent_conversation: dict[str, FrontendConversationTurn] = field(default_factory=dict)
    playback_receipts: dict[str, FrontendPlaybackReceipt] = field(default_factory=dict)
    heard_through_presentation_id: str | None = None
    heard_through_turn_id: str | None = None
    latest_playback_receipt: FrontendPlaybackReceipt | None = None
    context_revision: int = 0
    closed: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class RealtimeInteractionManager:
    """Coordinate frontend state without pretending response-only APIs own Work.

    Durable ``AgentInteractionPort`` Work continues through
    ``InteractionCoordinator`` and ``BackendEventCoordinator``. The response-only
    committed-turn surface handled here receives only process-local volatile
    identities; it never promotes them to backend Work IDs or recovery evidence.
    """

    def __init__(
        self,
        *,
        backend_profile: str,
        state_store: StateStore,
        committed_turns: EphemeralCommittedTurnPort | None,
        turn_routing_policy: TurnRoutingPolicy | None = None,
        delegate_original_text: bool = False,
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
        self._delegate_original_text = delegate_original_text
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
        self._ledger = VolatileWorkLedger(max_requests_per_session=retained_request_limit)
        # The response-only profile admits one active backend exchange
        # process-wide. Keep that compatibility constraint outside the frontend
        # protocol and each individual session lock.
        self._backend_admission_lock = asyncio.Lock()

    def _requests(self, state: _SessionState) -> tuple[VolatileWorkSnapshot, ...]:
        """Return the canonical ordered request snapshots for one live session."""
        try:
            return self._ledger.session(state.session_id).requests
        except VolatileRequestNotFoundError:
            return ()

    def _request(self, state: _SessionState, local_request_id: str) -> VolatileWorkSnapshot | None:
        try:
            return self._ledger.get(state.session_id, local_request_id)
        except VolatileRequestNotFoundError:
            return None

    def _aggregate_request(self, state: _SessionState) -> VolatileWorkSnapshot | None:
        """Prefer the newest active request, otherwise the newest terminal request."""
        requests = self._requests(state)
        for request in reversed(requests):
            if request.state.active:
                return request
        return requests[-1] if requests else None

    def _latest_result_request(self, state: _SessionState) -> VolatileWorkSnapshot | None:
        for request in reversed(self._requests(state)):
            if request.result is not None:
                return request
        return None

    def _project_frontend_tools(self, state: _SessionState) -> tuple[SemanticTool, ...]:
        """Compose backend-authorized submission with VoiceClaw-local controls."""
        projected: dict[str, SemanticTool] = {}
        if state.backend is not None:
            projected.update(
                (tool.name, tool)
                for tool in self._tool_registry.project(state.backend.capabilities)
                if self._frontend_tool_supported(tool.name)
            )
        requests = self._requests(state)
        if requests and self._frontend_tool_supported("work.status"):
            projected["work.status"] = self._tool_registry.definition("work.status")
        if self._frontend_tool_supported("work.cancel") and any(
            request.state.active and not request.effects_started for request in requests
        ):
            projected["work.cancel"] = self._tool_registry.definition("work.cancel")
        return tuple(projected[name] for name in sorted(projected))

    def _frontend_tool_supported(self, tool_name: str) -> bool:
        """Require both configured prose and delivery policy for local controls."""
        if tool_name not in self._interaction_profile.tools:
            return False
        template = _LOCAL_CONTROL_TEMPLATES.get(tool_name)
        return template is None or template in self._model_contracts.instruction_templates

    def _refresh_session_tools(self, state: _SessionState) -> tuple[SemanticTool, ...] | None:
        """Replace a session tool snapshot only when its applicable controls changed."""
        tools = self._project_frontend_tools(state)
        if tools == state.frontend_tools:
            return None
        state.frontend_tools = tools
        state.context_revision += 1
        return tools

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
        if session_id in self._sessions:
            raise ValueError("realtime session_id is already open")
        state = _SessionState(
            session_id=session_id,
            conversation_id=conversation_id,
            backend=backend,
        )
        state.frontend_tools = self._project_frontend_tools(state)
        frontend_tools = state.frontend_tools
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
            local_control_tools=tuple(
                tool.name for tool in frontend_tools if tool.name in {"work.status", "work.cancel"}
            ),
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
        local_controls = [name for name in frontend_tools if name in {"work.status", "work.cancel"}]
        requests = self._requests(state)
        active_request = any(request.state.active for request in requests)
        latest_request = self._aggregate_request(state)
        latest_result_request = self._latest_result_request(state)
        operation = latest_request.operation if latest_request is not None else "idle"
        phase = _request_state(latest_request) if latest_request is not None else ResponseOnlyRequestState.IDLE
        local_requests = [
            {
                "local_request_id": request.local_request_id,
                "local_work_id": request.local_work_id,
                "attempt_id": request.attempt_id,
                "request_summary": request.summary,
                "identity_authority": "voiceclaw_volatile",
                "operation": request.operation,
                "request_state": _request_state(request),
                "request_active": request.state.active,
                "durability": "none",
                "durable_work_id_issued": False,
                "terminal_outcome": _terminal_outcome(request),
                "failure_code": request.failure_code,
                "result": {
                    "state": _result_state(request),
                    "result_id": None if request.result is None else request.result.result_id,
                    "delivery_id": None if request.result is None else request.result.delivery_id,
                    "display_available": request.result is not None,
                    "speech_source": _speech_source(request),
                },
            }
            for request in requests
        ]
        active_local_requests = [request for request in local_requests if request["request_active"]]
        ready_results = [
            {
                "local_request_id": request.local_request_id,
                "local_work_id": request.local_work_id,
                "attempt_id": request.attempt_id,
                "result_id": request.result.result_id,
                "delivery_id": request.result.delivery_id,
                "identity_authority": "voiceclaw_volatile",
                "state": ResponseOnlyResultState.AVAILABLE,
                "display_available": True,
                "speech_source": _speech_source(request),
                "delivery_state": _delivery_state(request),
            }
            for request in requests
            if request.result is not None
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
            "local_controls": local_controls,
            "durability": typed_capabilities.durability if typed_capabilities is not None else Durability.NONE,
            "recoverable_inflight": False,
            "operation": operation,
            "phase": phase,
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
                "local_request_id": None if latest_request is None else latest_request.local_request_id,
                "local_work_id": latest_request.local_work_id if latest_request is not None else None,
                "attempt_id": latest_request.attempt_id if latest_request is not None else None,
                "request_summary": latest_request.summary if latest_request is not None else None,
                "identity_authority": "voiceclaw_volatile" if latest_request is not None else "none",
                "operation": operation,
                "request_state": phase,
                "request_active": (latest_request is not None and latest_request.state.active),
                "any_request_active": active_request,
                "durable_work_id_issued": False,
                "latest_terminal_outcome": (
                    ResponseOnlyTerminalOutcome.NONE if latest_request is None else _terminal_outcome(latest_request)
                ),
                "latest_failure_code": None if latest_request is None else latest_request.failure_code,
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
                "state": (
                    ResponseOnlyResultState.NONE if latest_result_request is None else ResponseOnlyResultState.AVAILABLE
                ),
                "display_available": latest_result_request is not None,
                "display_body_in_context": False,
                "speech_source": (
                    ResponseOnlySpeechSource.NONE
                    if latest_result_request is None
                    else _speech_source(latest_result_request)
                ),
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
            local_request = self._request(state, request.local_request_id)
            if local_request is None:
                raise LookupError("frontend context references an unknown local request")
            if request.purpose is FrontendContextPurpose.DELEGATION_ACK:
                admission = state.admissions.get(request.local_request_id)
                if admission is None:
                    raise LookupError("local request has no acknowledgement admission snapshot")
                target = self._admission_context(admission)
                projection_revision = admission.projection_revision
            else:
                target = self._local_request_context(local_request)

        active_requests = tuple(
            self._local_request_context(local_request)
            for local_request in self._requests(state)
            if local_request.state.active
        )
        ready_results = tuple(
            self._local_request_context(local_request)
            for local_request in self._requests(state)
            if local_request.result is not None
            and local_request.result.speech_state
            in {
                VolatileSpeechState.READY,
                VolatileSpeechState.CLAIMED,
                VolatileSpeechState.SPEAKING,
            }
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
        request = admission.request
        return LocalRequestContext(
            summary=request.summary,
            operation=request.operation,
            state=_request_state(request),
            backend_acceptance="unknown",
            durability=Durability.NONE,
            local_work_id=request.local_work_id,
            attempt_id=request.attempt_id,
        )

    @staticmethod
    def _local_request_context(request: VolatileWorkSnapshot) -> LocalRequestContext:
        """Normalize mutable local state without promoting it to backend Work."""
        return LocalRequestContext(
            summary=request.summary,
            operation=request.operation,
            state=_request_state(request),
            terminal_outcome=_terminal_outcome(request),
            failure_code=request.failure_code,
            result_state=_result_state(request),
            display_available=request.result is not None,
            speech_source=_speech_source(request),
            delivery_state=_delivery_state(request),
            local_work_id=request.local_work_id,
            attempt_id=request.attempt_id,
            result_id=None if request.result is None else request.result.result_id,
            delivery_id=None if request.result is None else request.result.delivery_id,
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
        if bounded.local_request_id is not None and self._request(state, bounded.local_request_id) is None:
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
        if correlated_local_request_id is not None and self._request(state, correlated_local_request_id) is None:
            raise ValueError("playback receipt references an unknown local request")
        correlated_receipt = (
            receipt
            if receipt.local_request_id == correlated_local_request_id
            else replace(receipt, local_request_id=correlated_local_request_id)
        )
        existing = state.playback_receipts.get(receipt.turn_id)
        if existing is not None:
            if existing == correlated_receipt:
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
        state.playback_receipts[receipt.turn_id] = correlated_receipt
        while len(state.playback_receipts) > _PLAYBACK_RECEIPT_LIMIT:
            oldest_turn_id = next(iter(state.playback_receipts))
            state.playback_receipts.pop(oldest_turn_id)
        state.latest_playback_receipt = correlated_receipt
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
        self._update_result_delivery_from_receipt(state, correlated_receipt)
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
        request = self._request(state, outcome.local_request_id)
        if request is None:
            raise ValueError("speech delivery outcome references an unknown local request")
        if request.result is None:
            raise ValueError("speech delivery outcome requires an available result")
        claim = state.speech_claims.get(outcome.local_request_id)
        if claim is None:
            raise ValueError("speech delivery outcome has no active result delivery claim")
        if outcome.presentation_id != claim.presentation_id:
            raise ValueError("speech delivery outcome references a different result presentation")
        try:
            self._ledger.record_delivery_outcome(
                claim,
                outcome=VolatileDeliveryOutcome(outcome.state.value),
            )
        except VolatileTransitionError as error:
            raise ValueError(str(error)) from error
        state.speech_claims.pop(outcome.local_request_id, None)
        state.context_revision += 1

    def _update_result_delivery_from_turn(self, state: _SessionState, turn: FrontendConversationTurn) -> None:
        """Correlate public delivery evidence without promoting it to backend state."""
        local_request_id = turn.local_request_id
        if local_request_id is None:
            return
        request = self._request(state, local_request_id)
        if request is None:
            raise ValueError("conversation turn references an unknown local request")
        if request.result is None:
            return
        claim = state.speech_claims.get(local_request_id)
        if claim is None:
            return
        if turn.presentation_id != claim.presentation_id:
            return
        if turn.delivery_state is FrontendConversationDeliveryState.DELIVERED:
            try:
                self._ledger.mark_speech_delivered(claim, turn_id=turn.turn_id)
            except VolatileTransitionError as error:
                raise ValueError(str(error)) from error
            return
        outcome = {
            FrontendConversationDeliveryState.SKIPPED: VolatileDeliveryOutcome.SKIPPED,
            FrontendConversationDeliveryState.CANCELLED: VolatileDeliveryOutcome.CANCELLED,
            FrontendConversationDeliveryState.FAILED: VolatileDeliveryOutcome.FAILED,
        }.get(turn.delivery_state)
        if outcome is not None:
            try:
                self._ledger.record_delivery_outcome(claim, outcome=outcome)
            except VolatileTransitionError as error:
                raise ValueError(str(error)) from error
            state.speech_claims.pop(local_request_id, None)

    def _update_result_delivery_from_receipt(self, state: _SessionState, receipt: FrontendPlaybackReceipt) -> None:
        """Correlate receipt-only delivery evidence without synthesizing a turn."""
        local_request_id = receipt.local_request_id
        if local_request_id is None:
            return
        request = self._request(state, local_request_id)
        if request is None:
            raise ValueError("playback receipt references an unknown local request")
        if request.result is None:
            return
        claim = state.speech_claims.get(local_request_id)
        if claim is None:
            return
        if receipt.presentation_id != claim.presentation_id:
            return
        if request.result.speech_state in {VolatileSpeechState.CLAIMED, VolatileSpeechState.SPEAKING}:
            try:
                self._ledger.mark_speech_delivered(claim, turn_id=receipt.turn_id)
            except VolatileTransitionError as error:
                raise ValueError(str(error)) from error
        heard_through_ms = receipt.heard_through_ms
        audio_end_ms = receipt.audio_end_ms
        if heard_through_ms is None or audio_end_ms is None:
            if receipt.state is FrontendPlaybackReceiptState.FAILED:
                heard_through_ms = heard_through_ms or 0
                audio_end_ms = audio_end_ms or heard_through_ms
            else:
                raise ValueError("heard and interrupted playback receipts require audio boundaries")
        try:
            self._ledger.record_playback(
                claim,
                state=VolatilePlaybackState(receipt.state.value),
                heard_through_ms=heard_through_ms,
                audio_end_ms=audio_end_ms,
            )
        except (VolatileTransitionError, ValueError) as error:
            raise ValueError(str(error)) from error
        state.speech_claims.pop(local_request_id, None)

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
        """Execute one currently advertised semantic operation."""
        state = self._session(session_id)
        if tool_name not in {tool.name for tool in state.frontend_tools}:
            yield self._control_failure(state, call_id, commit_id, "capability_unsupported")
            return
        try:
            validated_arguments = self._tool_registry.validate_arguments(tool_name, arguments)
        except (KeyError, ValueError):
            if tool_name == "work.delegate":
                yield self._failed_turn(state, call_id, commit_id, "invalid_tool_arguments")
            else:
                yield self._control_failure(state, call_id, commit_id, "invalid_tool_arguments")
            return
        if tool_name == "work.delegate":
            delegate = self._execute_delegate(
                session_id=session_id,
                commit_id=commit_id,
                call_id=call_id,
                arguments=validated_arguments,
                finalized_user_text=finalized_user_text,
            )
            try:
                async for update in delegate:
                    yield update
            finally:
                await delegate.aclose()
            return
        if tool_name == "work.status":
            yield await self._execute_local_status(
                state=state,
                operation_id=commit_id,
                call_id=call_id,
                arguments=validated_arguments,
            )
            return
        if tool_name == "work.cancel":
            yield await self._execute_local_cancel(
                state=state,
                operation_id=commit_id,
                call_id=call_id,
                arguments=validated_arguments,
            )
            return
        yield self._control_failure(state, call_id, commit_id, "capability_unsupported")

    async def _execute_delegate(
        self,
        *,
        session_id: str,
        commit_id: str,
        call_id: str,
        arguments: Mapping[str, Any],
        finalized_user_text: str | None,
    ) -> AsyncIterator[InteractionUpdate]:
        """Admit and dispatch one volatile request without exposing its adapter."""
        state = self._session(session_id)
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
        delegated_goal = arguments.get("goal")
        if not isinstance(delegated_goal, str):
            yield self._failed_turn(state, call_id, commit_id, "invalid_tool_arguments")
            return
        if self._delegate_original_text:
            # A voice relay must preserve references for the selected agent's
            # verified live context, rather than trust a model's guessed rewrite.
            delegated_goal = finalized_user_text.strip()
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
        controller = _LocalRequestController()
        async with self._guard_local_request(state, commit_id):
            admission_failure: InteractionUpdate | None = None
            async with state.lock:
                try:
                    request = self._ledger.admit(
                        state.session_id,
                        commit_id,
                        summary=request_summary,
                    )
                except (VolatileCapacityError, VolatileTransitionError):
                    admission_failure = self._control_failure(
                        state,
                        call_id,
                        commit_id,
                        "session_busy",
                    )
                else:
                    state.context_revision += 1
                    state.admissions[commit_id] = _LocalRequestAdmission(
                        request=request,
                        projection_revision=state.context_revision,
                    )
                    self._prune_session_records(state)
                    state.request_controllers[commit_id] = controller
                    queued_correlation = self._backend_correlation(
                        state,
                        call_id=call_id,
                        commit_id=commit_id,
                    )
                    changed_tools = self._refresh_session_tools(state)
                    changed_target = self._target_snapshot(state) if changed_tools is not None else None
            if admission_failure is not None:
                yield admission_failure
                return
            yield InteractionUpdate(
                kind=ResponseOnlyUpdateKind.BACKEND_TURN,
                phase=ResponseOnlyRequestState.LOCALLY_QUEUED,
                title="Request queued locally",
                text="VoiceClaw locally queued this non-durable request for backend dispatch.",
                correlation=queued_correlation,
                request_summary=request_summary,
                tool_output={
                    "status": ResponseOnlyRequestState.LOCALLY_QUEUED,
                    "request_state": ResponseOnlyRequestState.LOCALLY_QUEUED,
                    "local_request_id": commit_id,
                    "work_id": request.local_work_id,
                    "identity_authority": "voiceclaw_volatile",
                    "evidence": "voiceclaw_volatile",
                    "backend_acceptance": "unknown",
                    "durability": "none",
                },
                frontend_response=FrontendResponse(
                    purpose=FrontendResponsePurpose.DELEGATION_ACK,
                    payload_text=None,
                    local_request_id=commit_id,
                ),
                frontend_tools=changed_tools,
                backend_target=changed_target,
            )
            lane_acquired = await self._acquire_backend_lane_or_cancel(controller)
            if not lane_acquired:
                return
            try:
                try:
                    inspected_backend = await self._committed_turns.inspect()
                except Exception:
                    inspected_backend = None
                async with state.lock:
                    state.backend = inspected_backend
                    current = self._request(state, commit_id)
                    if current is None or not current.state.active:
                        return
                    if state.closed or controller.cancel_requested.is_set():
                        disposition = self._ledger.cancel_before_effects(state.session_id, commit_id)
                        if disposition is VolatileCancelDisposition.CANCELLED:
                            state.context_revision += 1
                            self._refresh_session_tools(state)
                        return
                    delegate_available = inspected_backend is not None and any(
                        tool.name == "work.delegate"
                        for tool in self._tool_registry.project(inspected_backend.capabilities)
                    )
                    if not delegate_available:
                        update = self._failed_turn(
                            state,
                            call_id,
                            commit_id,
                            "backend_unavailable",
                            tool_output=False,
                        )
                        update = self._with_frontend_contract(update, state)
                        dispatching = None
                    else:
                        request = self._ledger.begin_dispatch(state.session_id, commit_id)
                        state.context_revision += 1
                        changed_tools = self._refresh_session_tools(state)
                        changed_target = self._target_snapshot(state) if changed_tools is not None else None
                        dispatching = InteractionUpdate(
                            kind=ResponseOnlyUpdateKind.BACKEND_TURN,
                            phase=ResponseOnlyRequestState.DISPATCHING,
                            title="Dispatching request",
                            text="VoiceClaw is dispatching the queued request to the configured response-only backend.",
                            correlation=self._backend_correlation(
                                state,
                                call_id=call_id,
                                commit_id=commit_id,
                            ),
                            frontend_tools=changed_tools,
                            backend_target=changed_target,
                        )
                        update = None
                if update is not None:
                    if not state.closed:
                        yield update
                    return
                assert dispatching is not None
                yield dispatching
                loop = asyncio.get_running_loop()
                dispatch_observed = asyncio.Event()

                def mark_backend_effects() -> bool:
                    try:
                        self._ledger.begin_invocation(state.session_id, commit_id)
                    except (VolatileRequestNotFoundError, VolatileTransitionError):
                        return False
                    with suppress(RuntimeError):
                        loop.call_soon_threadsafe(dispatch_observed.set)
                    return True

                dispatch_marker = CommittedTurnDispatchMarker(mark_backend_effects)
                request = CommittedTurnRequest(
                    runtime_conversation_id=state.conversation_id,
                    commit_id=commit_id,
                    text=delegated_goal,
                    dispatch_marker=dispatch_marker,
                )
                first_pull = asyncio.create_task(self._pull_first_committed_turn_event(request))
                dispatch_waiter = asyncio.create_task(dispatch_observed.wait())
                cancellation_waiter = asyncio.create_task(controller.cancel_requested.wait())
                stream: AsyncIterator[CommittedTurnEvent] | None = None
                try:
                    done, _ = await asyncio.wait(
                        {first_pull, dispatch_waiter, cancellation_waiter},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if (
                        cancellation_waiter in done
                        and controller.cancel_requested.is_set()
                        and not dispatch_marker.marked
                    ):
                        return
                    if first_pull in done and not dispatch_marker.marked:
                        started_without_marker = await first_pull
                        if started_without_marker is not None:
                            self._ledger.begin_invocation(state.session_id, commit_id)
                        raise CommittedTurnError("invocation_result_unusable")
                    await dispatch_waiter
                    try:
                        waiting_backend = await self._committed_turns.inspect()
                    except Exception:
                        waiting_backend = None
                    async with state.lock:
                        state.backend = waiting_backend
                        current = self._request(state, commit_id)
                        if current is None or not current.effects_started:
                            raise CommittedTurnError("runtime_interrupted")
                        state.context_revision += 1
                        waiting_correlation = self._backend_correlation(
                            state,
                            call_id=call_id,
                            commit_id=commit_id,
                        )
                        changed_tools = self._refresh_session_tools(state)
                        waiting_target = self._target_snapshot(state)
                    yield InteractionUpdate(
                        kind=ResponseOnlyUpdateKind.BACKEND_TURN,
                        phase=ResponseOnlyRequestState.WAITING_FOR_RESPONSE,
                        title="Waiting for agent response",
                        text=(
                            "VoiceClaw is waiting for the configured target; this terminal invocation does not "
                            "supply a separate durable acceptance or running receipt."
                        ),
                        correlation=waiting_correlation,
                        frontend_tools=changed_tools,
                        backend_target=waiting_target,
                    )

                    result_id = f"result_vc_{uuid4().hex}"
                    delivery_id = f"delivery_vc_{uuid4().hex}"
                    presentation_id = f"presentation_vc_{uuid4().hex}"
                    result: CommittedTurnResult | None = None
                    display_fragments: list[str] = []
                    expected_display_sequence = 0
                    started = await first_pull
                    stream, event = started
                    while True:
                        if isinstance(event, CommittedTurnDisplayDelta):
                            if event.sequence != expected_display_sequence:
                                raise CommittedTurnError("invocation_result_unusable")
                            expected_display_sequence += 1
                            display_fragments.append(event.delta)
                            if state.closed:
                                raise CommittedTurnError("runtime_interrupted")
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
                                    result_id=result_id,
                                    delivery_id=delivery_id,
                                ),
                            )
                        elif not isinstance(event, CommittedTurnCompleted) or result is not None:
                            raise CommittedTurnError("invocation_result_unusable")
                        else:
                            result = event.result
                        try:
                            event = await anext(stream)
                        except StopAsyncIteration:
                            break
                    if result is None or "".join(display_fragments) != result.display_text:
                        raise CommittedTurnError("invocation_result_unusable")
                except CommittedTurnError as error:
                    async with state.lock:
                        current = self._request(state, commit_id)
                        update = (
                            None
                            if current is not None
                            and current.state.value == ResponseOnlyRequestState.CANCELLED.value
                            and error.code == "invocation_cancelled"
                            else self._failed_turn(state, call_id, commit_id, error.code, tool_output=False)
                        )
                    if update is None:
                        return
                    await self._refresh_backend_contract()
                    update = self._with_frontend_contract(update, state)
                    if not state.closed:
                        yield update
                    return
                except Exception:
                    async with state.lock:
                        update = self._failed_turn(
                            state,
                            call_id,
                            commit_id,
                            "runtime_interrupted",
                            tool_output=False,
                        )
                    await self._refresh_backend_contract()
                    update = self._with_frontend_contract(update, state)
                    if not state.closed:
                        yield update
                    return
                finally:
                    if not cancellation_waiter.done():
                        cancellation_waiter.cancel()
                    await asyncio.gather(cancellation_waiter, return_exceptions=True)
                    if not dispatch_waiter.done():
                        dispatch_waiter.cancel()
                    await asyncio.gather(dispatch_waiter, return_exceptions=True)
                    if not first_pull.done():
                        first_pull.cancel()
                    await asyncio.gather(first_pull, return_exceptions=True)
                    if stream is None and not first_pull.cancelled() and first_pull.exception() is None:
                        started = first_pull.result()
                        stream = started[0]
                    if stream is not None:
                        with suppress(Exception):
                            await stream.aclose()
                if not result.display_text.strip():
                    async with state.lock:
                        update = self._failed_turn(
                            state,
                            call_id,
                            commit_id,
                            "invocation_result_unusable",
                            tool_output=False,
                        )
                    await self._refresh_backend_contract()
                    update = self._with_frontend_contract(update, state)
                    if not state.closed:
                        yield update
                    return
                async with state.lock:
                    if state.closed:
                        update = self._failed_turn(
                            state,
                            call_id,
                            commit_id,
                            "runtime_interrupted",
                            tool_output=False,
                        )
                        completed = None
                    else:
                        completed = self._ledger.complete_with_result(
                            state.session_id,
                            commit_id,
                            result_id=result_id,
                            delivery_id=delivery_id,
                            presentation_id=presentation_id,
                            envelope=ResponseOnlyResultEnvelope(
                                display=result.display_text,
                                speech=result.speak_text,
                            ),
                        )
                        state.context_revision += 1
                        update = None
            finally:
                self._backend_admission_lock.release()
            if update is not None:
                await self._refresh_backend_contract()
                update = self._with_frontend_contract(update, state)
                if not state.closed:
                    yield update
                return
            assert completed is not None
            result_correlation = self._backend_correlation(
                state,
                call_id=call_id,
                commit_id=commit_id,
                backend_session_id=result.backend_session_id,
                turn_id=result.turn_id,
                response_id=result.response_id,
                result_id=completed.result.result_id,
                delivery_id=completed.result.delivery_id,
            )
            yield InteractionUpdate(
                kind=ResponseOnlyUpdateKind.RESULT_DISPLAY,
                phase=ResponseOnlyResultEventKind.COMPLETED,
                title="Response",
                text=result.display_text,
                correlation=result_correlation,
            )
            with suppress(VolatileTransitionError):
                self._ledger.mark_display_delivered(state.session_id, commit_id)
            async with state.lock:
                claim = self._ledger.claim_speech(
                    state.session_id,
                    commit_id,
                    owner=VolatileSpeechClaimOwner.AUTOMATIC,
                )
                if claim is not None:
                    state.speech_claims[commit_id] = claim
                    state.context_revision += 1
                frontend_response = (
                    None
                    if claim is None
                    else FrontendResponse(
                        purpose=FrontendResponsePurpose.RESULT_DELIVERY,
                        local_request_id=commit_id,
                        payload_text=claim.speech_text,
                        presentation_id=claim.presentation_id,
                    )
                )
                completed = self._ledger.get(state.session_id, commit_id)
            terminal_update = InteractionUpdate(
                kind=ResponseOnlyUpdateKind.BACKEND_TURN,
                phase=ResponseOnlyRequestState.SUCCEEDED,
                title="Response received",
                text="",
                correlation={
                    **result_correlation,
                    "speech_source": _speech_source(completed).value,
                },
                frontend_response=frontend_response,
            )
            await self._refresh_backend_contract()
            terminal_update = self._with_frontend_contract(terminal_update, state)
            if not state.closed:
                yield terminal_update

    async def _pull_first_committed_turn_event(
        self,
        request: CommittedTurnRequest,
    ) -> tuple[AsyncIterator[CommittedTurnEvent], CommittedTurnEvent]:
        """Enter the generic adapter while it owns the exact effects marker."""
        stream: AsyncIterator[CommittedTurnEvent] | None = None
        try:
            port = self._committed_turns
            if port is None:
                raise CommittedTurnError("backend_not_configured")
            stream = port.stream_turn(request)
            first = await anext(stream)
        except StopAsyncIteration as error:
            assert stream is not None
            with suppress(Exception):
                await stream.aclose()
            raise CommittedTurnError("invocation_result_unusable") from error
        except BaseException:
            if stream is not None:
                with suppress(Exception):
                    await stream.aclose()
            raise
        assert stream is not None
        return stream, first

    async def _acquire_backend_lane_or_cancel(self, controller: _LocalRequestController) -> bool:
        """Wait for the response-only lane without hiding a queued cancellation."""
        acquire = asyncio.create_task(self._backend_admission_lock.acquire())
        cancellation = asyncio.create_task(controller.cancel_requested.wait())
        try:
            done, _ = await asyncio.wait({acquire, cancellation}, return_when=asyncio.FIRST_COMPLETED)
            if cancellation in done and controller.cancel_requested.is_set():
                if acquire.done() and not acquire.cancelled() and acquire.exception() is None and acquire.result():
                    self._backend_admission_lock.release()
                else:
                    acquire.cancel()
                    await asyncio.gather(acquire, return_exceptions=True)
                return False
            cancellation.cancel()
            await asyncio.gather(cancellation, return_exceptions=True)
            return bool(await acquire)
        except BaseException:
            if not acquire.done():
                acquire.cancel()
            await asyncio.gather(acquire, return_exceptions=True)
            if (
                acquire.done()
                and not acquire.cancelled()
                and acquire.exception() is None
                and acquire.result()
                and self._backend_admission_lock.locked()
            ):
                self._backend_admission_lock.release()
            raise
        finally:
            if not cancellation.done():
                cancellation.cancel()
            await asyncio.gather(cancellation, return_exceptions=True)

    def _resolve_local_request(
        self,
        state: _SessionState,
        arguments: Mapping[str, Any],
        *,
        cancellable_only: bool = False,
    ) -> VolatileWorkSnapshot | None:
        """Resolve one request only inside its owning realtime session."""
        work_id = arguments.get("work_id")
        requests = self._requests(state)
        if isinstance(work_id, str):
            return next(
                (
                    request
                    for request in requests
                    if request.local_request_id == work_id or request.local_work_id == work_id
                ),
                None,
            )
        if cancellable_only:
            candidates = [request for request in requests if request.state.active and not request.effects_started]
            return candidates[0] if len(candidates) == 1 else None
        return self._aggregate_request(state)

    async def _execute_local_status(
        self,
        *,
        state: _SessionState,
        operation_id: str,
        call_id: str,
        arguments: Mapping[str, Any],
    ) -> InteractionUpdate:
        """Read process-local lifecycle evidence without invoking Fabric."""
        async with state.lock:
            request = self._resolve_local_request(state, arguments)
            if request is None:
                return self._control_failure(state, call_id, operation_id, "local_request_not_found")
            correlation = self._backend_correlation(
                state,
                call_id=call_id,
                operation_id=operation_id,
                local_request_id=request.local_request_id,
            )
            output = {
                "status": request.state.value,
                "request_state": request.state.value,
                "local_request_id": request.local_request_id,
                "work_id": request.local_work_id,
                "identity_authority": "voiceclaw_volatile",
                "terminal_outcome": _terminal_outcome(request).value,
                "result_state": _result_state(request).value,
            }
            if request.failure_code is not None:
                output["failure_code"] = request.failure_code
            claim_owner = None
            if request.result is not None and request.result.speech_state is VolatileSpeechState.READY:
                claim_owner = VolatileSpeechClaimOwner.STATUS
            elif request.result is not None and request.result.speech_state is VolatileSpeechState.INTERRUPTED:
                claim_owner = VolatileSpeechClaimOwner.RECAP
            claim = (
                None
                if claim_owner is None
                else self._ledger.claim_speech(state.session_id, request.local_request_id, owner=claim_owner)
            )
            if claim is not None:
                state.speech_claims[request.local_request_id] = claim
                state.context_revision += 1
                frontend_response = FrontendResponse(
                    purpose=FrontendResponsePurpose.RESULT_DELIVERY,
                    payload_text=claim.speech_text,
                    local_request_id=request.local_request_id,
                    presentation_id=claim.presentation_id,
                )
            else:
                frontend_response = FrontendResponse(
                    purpose=FrontendResponsePurpose.STATUS_DELIVERY,
                    payload_text=None,
                    local_request_id=request.local_request_id,
                )
            return InteractionUpdate(
                kind=ResponseOnlyUpdateKind.BACKEND_TURN,
                phase=_request_state(request),
                title="Request status",
                text="VoiceClaw read the current process-local request state.",
                correlation=correlation,
                request_summary=request.summary,
                tool_output=output,
                frontend_response=frontend_response,
            )

    async def _execute_local_cancel(
        self,
        *,
        state: _SessionState,
        operation_id: str,
        call_id: str,
        arguments: Mapping[str, Any],
    ) -> InteractionUpdate:
        """Cancel only a queued request that has not crossed the effects fence."""
        async with state.lock:
            request = self._resolve_local_request(state, arguments, cancellable_only=True)
            if request is None:
                return self._control_failure(state, call_id, operation_id, "local_request_not_found")
            controller = state.request_controllers.get(request.local_request_id)
            if controller is None:
                correlation = self._backend_correlation(
                    state,
                    call_id=call_id,
                    operation_id=operation_id,
                    local_request_id=request.local_request_id,
                    error_code="backend_cancellation_unsupported",
                )
                return InteractionUpdate(
                    kind=ResponseOnlyUpdateKind.BACKEND_TURN,
                    phase=_request_state(request),
                    title="Cancellation unavailable",
                    text="The request has already reached the backend effects boundary.",
                    correlation=correlation,
                    request_summary=request.summary,
                    tool_output={
                        "status": "rejected",
                        "error": "backend_cancellation_unsupported",
                        "local_request_id": request.local_request_id,
                        "work_id": request.local_work_id,
                    },
                    frontend_response=FrontendResponse(
                        purpose=FrontendResponsePurpose.CONTROL_DELIVERY,
                        payload_text=None,
                        local_request_id=request.local_request_id,
                    ),
                )
            disposition = self._ledger.cancel_before_effects(state.session_id, request.local_request_id)
            if disposition is not VolatileCancelDisposition.CANCELLED:
                return self._control_failure(
                    state,
                    call_id,
                    operation_id,
                    "backend_cancellation_unsupported",
                    local_request_id=request.local_request_id,
                )
            state.context_revision += 1
            controller.cancel_requested.set()
            changed_tools = self._refresh_session_tools(state)
            changed_target = self._target_snapshot(state) if changed_tools is not None else None
            cancelled = self._cancelled_request_update(
                state,
                call_id,
                request.local_request_id,
                changed_tools=changed_tools,
                backend_target=changed_target,
            )
            return replace(
                cancelled,
                correlation={
                    **cancelled.correlation,
                    "operation_id": operation_id,
                },
                tool_output={
                    "status": ResponseOnlyRequestState.CANCELLED.value,
                    "request_state": ResponseOnlyRequestState.CANCELLED.value,
                    "local_request_id": request.local_request_id,
                    "work_id": request.local_work_id,
                    "identity_authority": "voiceclaw_volatile",
                },
                frontend_response=FrontendResponse(
                    purpose=FrontendResponsePurpose.CONTROL_DELIVERY,
                    payload_text=None,
                    local_request_id=request.local_request_id,
                ),
            )

    def _cancelled_request_update(
        self,
        state: _SessionState,
        call_id: str,
        local_request_id: str,
        *,
        changed_tools: tuple[SemanticTool, ...] | None = None,
        backend_target: BackendTargetSnapshot | None = None,
    ) -> InteractionUpdate:
        """Project an already committed pre-effect cancellation exactly once."""
        request = self._ledger.get(state.session_id, local_request_id)
        correlation = self._backend_correlation(
            state,
            call_id=call_id,
            commit_id=local_request_id,
        )
        return InteractionUpdate(
            kind=ResponseOnlyUpdateKind.BACKEND_TURN,
            phase=ResponseOnlyRequestState.CANCELLED,
            title="Request cancelled",
            text="VoiceClaw cancelled the request before backend invocation.",
            correlation=correlation,
            request_summary=request.summary,
            frontend_tools=changed_tools,
            backend_target=backend_target,
        )

    async def _refresh_backend_contract(self) -> None:
        """Refresh model-visible operations after a terminal backend exchange."""
        backend: CommittedTurnBackend | None = None
        if self._committed_turns is not None:
            try:
                backend = await self._committed_turns.inspect()
            except Exception:
                backend = None
        for session in self._sessions.values():
            session.backend = backend
            self._refresh_session_tools(session)

    def _target_snapshot(self, state: _SessionState) -> BackendTargetSnapshot:
        """Build one atomic target replacement for protocol adapters."""
        backend = state.backend
        if backend is None:
            local_controls = tuple(
                tool.name for tool in state.frontend_tools if tool.name in {"work.status", "work.cancel"}
            )
            return BackendTargetSnapshot(
                backend_label="Backend unavailable",
                backend_mode="disabled",
                target_binding_verified=False,
                target_ref="not-attached",
                target_availability=None,
                frontend_tools=state.frontend_tools,
                local_control_tools=local_controls,
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
            local_control_tools=tuple(
                tool.name for tool in state.frontend_tools if tool.name in {"work.status", "work.cancel"}
            ),
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
        commit_id: str,
    ) -> AsyncIterator[None]:
        """Close any admitted request whose update stream exits nonterminally."""
        try:
            yield
        finally:
            refresh_backend = False
            discard_closed = False
            async with state.lock:
                request = self._request(state, commit_id)
                if request is not None and request.state.active:
                    if request.effects_started:
                        self._ledger.fail(
                            state.session_id,
                            commit_id,
                            failure_code="runtime_interrupted",
                            outcome_unknown=True,
                        )
                        refresh_backend = True
                    else:
                        controller = state.request_controllers.get(commit_id)
                        if controller is not None:
                            controller.cancel_requested.set()
                        self._ledger.cancel_before_effects(state.session_id, commit_id)
                    state.context_revision += 1
                    self._refresh_session_tools(state)
                state.request_controllers.pop(commit_id, None)
                discard_closed = state.closed and not state.request_controllers
            if refresh_backend:
                await self._refresh_backend_contract()
            if discard_closed:
                self._discard_closed_session(state)

    async def close_session(self, session_id: str, reason: str) -> None:
        """Detach the frontend and fence every request that has not reached backend effects."""
        state = self._session(session_id)
        del reason
        async with state.lock:
            state.closed = True
            state.activity = replace(state.activity, connected=False)
            for request in self._requests(state):
                if not request.state.active or request.effects_started:
                    continue
                controller = state.request_controllers.get(request.local_request_id)
                disposition = self._ledger.cancel_before_effects(state.session_id, request.local_request_id)
                if disposition is VolatileCancelDisposition.CANCELLED:
                    state.context_revision += 1
                if controller is not None:
                    controller.cancel_requested.set()
            self._refresh_session_tools(state)
            discard_closed = not state.request_controllers
        self._state_store.discard_ephemeral_session(session_id)
        if discard_closed:
            self._discard_closed_session(state)

    def _discard_closed_session(self, state: _SessionState) -> None:
        """Release volatile memory only after every request coroutine has fenced itself."""
        if self._sessions.get(state.session_id) is not state or not state.closed:
            return
        self._ledger.discard_session(state.session_id, force=True)
        self._sessions.pop(state.session_id, None)

    def _control_failure(
        self,
        state: _SessionState,
        call_id: str,
        operation_id: str,
        code: str,
        *,
        local_request_id: str | None = None,
    ) -> InteractionUpdate:
        """Reject a local control without creating or mutating a Work record."""
        failure_copy = self._model_contracts.failure_copy(code)
        correlation = self._backend_correlation(
            state,
            call_id=call_id,
            operation_id=operation_id,
            error_code=code,
            **({"local_request_id": local_request_id} if local_request_id is not None else {}),
        )
        return InteractionUpdate(
            kind=ResponseOnlyUpdateKind.BACKEND_TURN,
            phase=ResponseOnlyRequestState.FAILED,
            title=failure_copy.title,
            text=failure_copy.display,
            correlation=correlation,
            tool_output={"status": ResponseOnlyRequestState.FAILED, "error": code},
            frontend_response=FrontendResponse(
                purpose=FrontendResponsePurpose.FAILURE_DELIVERY,
                payload_text=failure_copy.speech,
                local_request_id=local_request_id,
            ),
        )

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
        request = self._request(state, commit_id)
        if request is None:
            try:
                request = self._ledger.admit(
                    state.session_id,
                    commit_id,
                    summary=request_summary or "Request details unavailable",
                )
                state.context_revision += 1
                state.admissions.setdefault(
                    commit_id,
                    _LocalRequestAdmission(request=request, projection_revision=state.context_revision),
                )
                self._prune_session_records(state)
            except (VolatileCapacityError, VolatileTransitionError):
                return self._control_failure(state, call_id, commit_id, "session_busy")
        if request.state.active:
            request = self._ledger.fail(
                state.session_id,
                commit_id,
                failure_code=code,
                outcome_unknown=uncertain and request.effects_started,
            )
            state.context_revision += 1
        terminal_phase = _request_state(request)
        correlation = self._backend_correlation(
            state,
            call_id=call_id,
            commit_id=commit_id,
            error_code=code,
        )
        changed_tools = self._refresh_session_tools(state)
        changed_target = self._target_snapshot(state) if changed_tools is not None else None
        failure_copy = self._model_contracts.failure_copy(code)
        return InteractionUpdate(
            kind=ResponseOnlyUpdateKind.BACKEND_TURN,
            phase=terminal_phase,
            title=failure_copy.title,
            text=failure_copy.display,
            correlation=correlation,
            request_summary=request.summary,
            tool_output={"status": terminal_phase, "error": code} if tool_output else None,
            frontend_response=FrontendResponse(
                purpose=FrontendResponsePurpose.FAILURE_DELIVERY,
                payload_text=failure_copy.speech,
                local_request_id=commit_id,
            ),
            frontend_tools=changed_tools,
            backend_target=changed_target,
        )

    def _prune_session_records(self, state: _SessionState) -> None:
        retained = {request.local_request_id for request in self._requests(state)}
        for mapping in (state.admissions, state.speech_claims, state.request_controllers):
            for local_request_id in tuple(mapping):
                if local_request_id not in retained:
                    mapping.pop(local_request_id, None)

    def _backend_correlation(self, state: _SessionState, **values: str) -> dict[str, str]:
        """Keep request and target identities visible without copying user content."""
        correlation = dict(values)
        commit_id = correlation.get("commit_id")
        if commit_id is not None:
            correlation["local_request_id"] = commit_id
            request = self._request(state, commit_id)
            if request is not None:
                correlation["work_id"] = request.local_work_id
                correlation["attempt_id"] = request.attempt_id
                if request.result is not None:
                    correlation.setdefault("result_id", request.result.result_id)
                    correlation.setdefault("delivery_id", request.result.delivery_id)
            correlation["identity_authority"] = "voiceclaw_volatile"
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
