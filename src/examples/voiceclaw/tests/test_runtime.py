# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

import asyncio
import json
from pathlib import Path

import pytest

from voiceclaw.adapters.state.sqlite import SqliteStateStore
from voiceclaw.application.runtime import RealtimeInteractionManager
from voiceclaw.domain import (
    RUNTIME_PROJECTION_SCHEMA,
    ResponseOnlyRequestState,
    ResponseOnlyResultEventKind,
    ResponseOnlySpeechSource,
    ResponseOnlyUpdateKind,
)
from voiceclaw.domain.models import (
    BackendCapabilities,
    BackendOperation,
    CapabilitySource,
    Durability,
    EventDelivery,
    FrontendActivity,
    InputActivity,
)
from voiceclaw.interaction_profiles import INTERACTION_PROFILE_SCHEMA, MAX_DELEGATED_GOAL_BYTES
from voiceclaw.model_contracts import load_model_contract_catalog
from voiceclaw.ports.runtime import (
    FrontendConversationDeliveryState,
    FrontendConversationTurn,
    FrontendPlaybackReceipt,
    FrontendPlaybackReceiptState,
    FrontendResponsePurpose,
    InteractionUpdate,
    TurnDirective,
    TurnDirectiveKind,
)
from voiceclaw.ports.turns import (
    MAX_COMMITTED_TURN_GOAL_BYTES,
    CommittedTurnBackend,
    CommittedTurnCompleted,
    CommittedTurnDisplayDelta,
    CommittedTurnError,
    CommittedTurnResult,
)

_MODEL_CONTRACTS = Path(__file__).parents[1] / "src" / "voiceclaw" / "resources" / "model_contracts.v1.yaml"


def test_interaction_update_preserves_existing_positional_constructor() -> None:
    correlation = {"local_request_id": "request-1"}
    tool_output = {"status": "locally_queued"}
    update = InteractionUpdate(
        ResponseOnlyUpdateKind.BACKEND_TURN,
        ResponseOnlyRequestState.LOCALLY_QUEUED,
        "Request queued locally",
        "VoiceClaw queued the request.",
        correlation,
        tool_output,
        None,
        (),
    )

    assert update.correlation == correlation
    assert update.tool_output == tool_output
    assert update.frontend_tools == ()
    assert update.request_summary is None


class _Turns:
    async def inspect(self):
        return CommittedTurnBackend(
            label="Configured agent",
            target_ref="server-selected agent",
            mode="response_only",
            capabilities=BackendCapabilities(
                backend_kind="response_only",
                target_label="Configured agent",
                revision="test-runtime-v1",
                operations=frozenset({BackendOperation.SUBMIT}),
                durability=Durability.NONE,
                event_delivery=EventDelivery.RESPONSE_ONLY,
                max_parallel_work=1,
            ),
            capability_source=CapabilitySource.OPERATOR_CONFIGURED,
            capability_source_id="test_committed_turn",
        )

    async def commit_turn(self, request):
        return CommittedTurnResult(
            backend_session_id="backend-session-1",
            turn_id="turn-1",
            response_id="response-1",
            display_text=f"done: {request.text}",
            speak_text=f"Done: {request.text}.",
        )

    async def stream_turn(self, request):
        result = await self.commit_turn(request)
        for sequence, start in enumerate(range(0, len(result.display_text), 7)):
            yield CommittedTurnDisplayDelta(
                backend_session_id=result.backend_session_id,
                turn_id=result.turn_id,
                response_id=result.response_id,
                sequence=sequence,
                delta=result.display_text[start : start + 7],
            )
        yield CommittedTurnCompleted(result=result)


class _TestRoutingPolicy:
    def decide(self, text: str) -> TurnDirective:
        if text == "Hello":
            return TurnDirective.direct(reason_code="test_direct")
        return TurnDirective.tool("work.delegate", reason_code="test_delegate")


async def _collect(iterator):
    return [update async for update in iterator]


def _request_phases(updates):
    return [update.phase for update in updates if update.kind is ResponseOnlyUpdateKind.BACKEND_TURN]


def _display_updates(updates):
    return [update for update in updates if update.kind is ResponseOnlyUpdateKind.RESULT_DISPLAY]


def _display_text(updates):
    return "".join(
        update.text for update in _display_updates(updates) if update.phase is ResponseOnlyResultEventKind.DISPLAY_DELTA
    )


def _completed_display(updates):
    completed = [
        update for update in _display_updates(updates) if update.phase is ResponseOnlyResultEventKind.COMPLETED
    ]
    assert len(completed) == 1
    return completed[0]


def _delegation(goal: str) -> dict[str, str]:
    """Build the required model-authored delegation payload for runtime tests."""
    return {"goal": goal}


def test_frontend_conversation_and_playback_values_validate_delivery_evidence() -> None:
    with pytest.raises(ValueError, match="user conversation turns must be committed"):
        FrontendConversationTurn(
            turn_id="item-user",
            role="user",
            text="Hello",
            delivery_state=FrontendConversationDeliveryState.DELIVERED,
        )
    with pytest.raises(ValueError, match="assistant conversation turns cannot be committed"):
        FrontendConversationTurn(
            turn_id="item-assistant",
            role="assistant",
            text="Hello",
            delivery_state=FrontendConversationDeliveryState.COMMITTED,
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        FrontendPlaybackReceipt(
            turn_id="item-assistant",
            state=FrontendPlaybackReceiptState.INTERRUPTED,
            presentation_id="presentation-assistant",
            heard_through_ms=501,
            audio_end_ms=500,
        )


def test_runtime_routes_only_to_an_advertised_operation() -> None:
    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            attached = RealtimeInteractionManager(
                backend_profile="attached",
                state_store=store,
                committed_turns=_Turns(),
                turn_routing_policy=_TestRoutingPolicy(),
            )
            unavailable = RealtimeInteractionManager(
                backend_profile="unavailable",
                state_store=store,
                committed_turns=None,
                turn_routing_policy=_TestRoutingPolicy(),
            )
            await attached.open_session("session-attached", "conversation-attached")
            await unavailable.open_session("session-unavailable", "conversation-unavailable")

            task = attached.route_finalized_turn("session-attached", "Write a binary search tree.")
            greeting = attached.route_finalized_turn("session-attached", "Hello")
            missing = unavailable.route_finalized_turn("session-unavailable", "Write a binary search tree.")

            assert task.kind is TurnDirectiveKind.TOOL
            assert task.logical_tool == "work.delegate"
            assert greeting.kind is TurnDirectiveKind.DIRECT
            assert missing.kind is TurnDirectiveKind.REJECT
            assert missing.reason_code == "operation_unavailable"

    asyncio.run(exercise())


def test_runtime_persists_mapping_and_labels_ephemeral_turn_non_durable() -> None:
    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_Turns(),
            )
            snapshot = await runtime.open_session("session-1", "conversation-1")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-1",
                    commit_id="commit-1",
                    call_id="call-1",
                    tool_name="work.delegate",
                    arguments=_delegation("build it"),
                    finalized_user_text="build it",
                )
            )

            assert snapshot.recoverable_inflight is False
            assert [tool.name for tool in snapshot.frontend_tools] == ["work.delegate"]
            assert snapshot.capability_source is CapabilitySource.OPERATOR_CONFIGURED
            assert snapshot.capability_source_id == "test_committed_turn"
            assert snapshot.capability_revision == "test-runtime-v1"
            assert snapshot.capability_hash is not None
            assert snapshot.model_contract_profile == "default"
            assert snapshot.model_contract_hash is not None
            assert snapshot.interaction_profile_schema == INTERACTION_PROFILE_SCHEMA
            assert snapshot.interaction_profile_name == "stateless"
            assert snapshot.interaction_profile_hash is not None
            assert store.get_session("session-1") is not None
            assert _request_phases(updates) == [
                ResponseOnlyRequestState.LOCALLY_QUEUED,
                ResponseOnlyRequestState.DISPATCHING,
                ResponseOnlyRequestState.WAITING_FOR_RESPONSE,
                ResponseOnlyRequestState.SUCCEEDED,
            ]
            display_updates = _display_updates(updates)
            assert display_updates
            assert [update.phase for update in display_updates] == [
                ResponseOnlyResultEventKind.DISPLAY_DELTA,
                ResponseOnlyResultEventKind.DISPLAY_DELTA,
                ResponseOnlyResultEventKind.COMPLETED,
            ]
            assert _display_text(updates) == "done: build it"
            assert _completed_display(updates).text == "done: build it"
            assert updates[0].tool_output == {
                "status": "locally_queued",
                "request_state": "locally_queued",
                "local_request_id": "commit-1",
                "identity_authority": "voiceclaw_local",
                "evidence": "voiceclaw_local",
                "backend_acceptance": "unknown",
                "durability": "none",
            }
            assert updates[0].frontend_response is not None
            assert updates[0].frontend_response.purpose is FrontendResponsePurpose.DELEGATION_ACK
            assert updates[0].frontend_response.local_request_id == "commit-1"
            assert updates[0].frontend_response.payload_text == "build it"
            assert updates[0].request_summary == "build it"
            assert "request_summary" not in updates[0].correlation
            assert sum(update.tool_output is not None for update in updates) == 1
            assert updates[1].frontend_response is None
            assert updates[-1].frontend_response is not None
            assert updates[-1].frontend_response.purpose is FrontendResponsePurpose.RESULT_DELIVERY
            assert updates[-1].frontend_response.local_request_id == "commit-1"
            assert updates[-1].frontend_response.payload_text == "Done: build it."
            assert updates[-1].correlation["backend_name"] == "Configured agent"
            assert updates[-1].correlation["backend_mode"] == "response_only"
            assert updates[-1].correlation["target_ref"] == "server-selected agent"
            assert updates[-1].correlation["backend_session_id"] == "backend-session-1"
            assert updates[-1].correlation["turn_id"] == "turn-1"
            assert updates[-1].request_summary is None
            assert "request_summary" not in updates[-1].correlation
            assert updates[-1].correlation["speech_source"] == ResponseOnlySpeechSource.BACKEND_AUTHORED.value
            projection_text = runtime.projection("session-1")
            projection = json.loads(projection_text)
            assert projection["schema"] == RUNTIME_PROJECTION_SCHEMA
            assert projection["durability"] == "none"
            assert projection["gateway_reachable"] is True
            assert projection["connection"] == {
                "attachment_mode": "ephemeral_per_request",
                "attachment_state": "not_persistent",
                "frontend_state": "connected",
                "gateway_state": "reachable",
                "recoverable_inflight": False,
                "reconnect_mode": "fresh_frontend_session",
                "resume_supported": False,
            }
            assert projection["contract"] == {
                "backend_capabilities": ["work.submit"],
                "capability_hash": snapshot.capability_hash,
                "capability_revision": "test-runtime-v1",
                "capability_source": "operator_configured",
                "capability_source_id": "test_committed_turn",
                "durability": "none",
                "event_delivery": "response_only",
                "frontend_tools": ["work.delegate"],
                "interaction_profile_hash": snapshot.interaction_profile_hash,
                "interaction_profile_name": "stateless",
                "interaction_profile_schema": INTERACTION_PROFILE_SCHEMA,
                "max_parallel_requests": 1,
                "model_contract_hash": snapshot.model_contract_hash,
                "model_contract_profile": "default",
                "model_contract_schema": "voiceclaw.model-contracts.v1",
                "operations": ["work.submit"],
                "session_scope": "none",
                "sessionful": False,
                "supports_parallel_work": False,
                "work_cardinality": "many",
            }
            assert projection["execution"] == {
                "any_request_active": False,
                "durable_work_id_issued": False,
                "identity_authority": "voiceclaw_local",
                "latest_failure_code": None,
                "latest_terminal_outcome": "succeeded",
                "local_request_id": "commit-1",
                "operation": "work.delegate",
                "request_active": False,
                "request_state": "succeeded",
                "request_summary": "build it",
            }
            assert projection["local_requests"] == [
                {
                    "durability": "none",
                    "durable_work_id_issued": False,
                    "failure_code": None,
                    "identity_authority": "voiceclaw_local",
                    "local_request_id": "commit-1",
                    "operation": "work.delegate",
                    "request_active": False,
                    "request_state": "succeeded",
                    "request_summary": "build it",
                    "result": {
                        "display_available": True,
                        "speech_source": ResponseOnlySpeechSource.BACKEND_AUTHORED.value,
                        "state": "available",
                    },
                    "terminal_outcome": "succeeded",
                }
            ]
            assert projection["active_work"] == []
            assert projection["active_local_requests"] == []
            assert projection["pending_interactions"] == []
            assert projection["ready_results"] == [
                {
                    "delivery_state": "ready",
                    "display_available": True,
                    "identity_authority": "voiceclaw_local",
                    "local_request_id": "commit-1",
                    "speech_source": ResponseOnlySpeechSource.BACKEND_AUTHORED.value,
                    "state": "available",
                }
            ]
            assert projection["recent_conversation"] == []
            assert projection["heard_through_presentation_id"] is None
            assert projection["heard_through_turn_id"] is None
            assert projection["latest_result"] == {
                "display_available": True,
                "display_body_in_context": False,
                "provider_identifiers_in_context": False,
                "speech_source": ResponseOnlySpeechSource.BACKEND_AUTHORED.value,
                "state": "available",
            }
            assert projection["delivery"]["heard_state"] == "unknown"
            assert projection["delivery"]["authoritative_client_receipt"] is False
            assert "done: build it" not in projection_text
            assert "Done: build it." not in projection_text
            assert "payload_text" not in projection_text
            assert "work_id" not in projection
            assert "backend_session_id" not in projection_text
            assert "turn-1" not in projection_text
            assert "response_id" not in projection_text
            assert "call_id" not in projection_text
            assert "target_ref" not in projection_text

    asyncio.run(exercise())


def test_runtime_uses_the_bounded_delegated_goal_as_acknowledgement_payload() -> None:
    async def exercise() -> None:
        source_turn = "Please use the same language and include the complexity discussion."
        delegated_goal = "Create a concise Python binary search tree example and explain its time and space complexity."
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_Turns(),
            )
            await runtime.open_session("session-routing", "conversation-routing")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-routing",
                    commit_id="commit-routing",
                    call_id="call-routing",
                    tool_name="work.delegate",
                    arguments=_delegation(delegated_goal),
                    finalized_user_text=source_turn,
                )
            )

            acknowledgement = updates[0].frontend_response
            assert acknowledgement is not None
            assert acknowledgement.purpose is FrontendResponsePurpose.DELEGATION_ACK
            assert acknowledgement.payload_text == delegated_goal
            assert updates[0].request_summary == delegated_goal
            assert "request_summary" not in updates[0].correlation
            assert _display_text(updates) == f"done: {delegated_goal}"
            assert source_turn not in _display_text(updates)
            assert updates[-1].text == ""

    asyncio.run(exercise())


def test_runtime_sends_the_validated_model_authored_goal_not_the_source_turn_to_backend() -> None:
    class _RecordingTurns(_Turns):
        def __init__(self) -> None:
            self.committed_text: str | None = None

        async def commit_turn(self, request):
            self.committed_text = request.text
            return await super().commit_turn(request)

    async def exercise() -> None:
        turns = _RecordingTurns()
        source_turn = "Use those exact formatting requirements for this one."
        delegated_goal = '  Preserve "quotes", newlines\n,and Unicode ☃ exactly.  '
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=turns,
            )
            await runtime.open_session("session-exact", "conversation-exact")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-exact",
                    commit_id="commit-exact",
                    call_id="call-exact",
                    tool_name="work.delegate",
                    arguments=_delegation(delegated_goal),
                    finalized_user_text=source_turn,
                )
            )

            assert turns.committed_text == delegated_goal.strip()
            assert turns.committed_text != source_turn
            assert updates[0].request_summary == 'Preserve "quotes", newlines ,and Unicode ☃ exactly.'
            assert "request_summary" not in updates[0].correlation

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"goal": ""},
        {"goal": "   \n\t"},
        {"goal": 42},
        {"goal": "Valid standalone goal", "unexpected": "value"},
    ],
    ids=["missing", "empty", "whitespace", "not-a-string", "unknown-property"],
)
def test_runtime_rejects_invalid_model_authored_delegation_goals(arguments: dict[str, object]) -> None:
    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_Turns(),
            )
            await runtime.open_session("session-invalid-goal", "conversation-invalid-goal")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-invalid-goal",
                    commit_id="commit-invalid-goal",
                    call_id="call-invalid-goal",
                    tool_name="work.delegate",
                    arguments=arguments,
                    finalized_user_text="Use the earlier requirements.",
                )
            )

            assert _request_phases(updates) == [ResponseOnlyRequestState.FAILED]
            assert updates[0].tool_output == {
                "status": ResponseOnlyRequestState.FAILED,
                "error": "invalid_tool_arguments",
            }
            projection = json.loads(runtime.projection("session-invalid-goal"))
            assert projection["execution"]["latest_failure_code"] == "invalid_tool_arguments"
            assert projection["execution"]["request_summary"] == "Request details unavailable"

    asyncio.run(exercise())


def test_runtime_rejects_a_delegation_goal_over_the_utf8_byte_limit() -> None:
    async def exercise() -> None:
        oversized_goal = "☃" * ((MAX_DELEGATED_GOAL_BYTES // len("☃".encode())) + 1)
        assert len(oversized_goal) < MAX_DELEGATED_GOAL_BYTES
        assert len(oversized_goal.encode()) > MAX_DELEGATED_GOAL_BYTES
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_Turns(),
            )
            await runtime.open_session("session-oversized-goal", "conversation-oversized-goal")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-oversized-goal",
                    commit_id="commit-oversized-goal",
                    call_id="call-oversized-goal",
                    tool_name="work.delegate",
                    arguments=_delegation(oversized_goal),
                    finalized_user_text="Use all of the earlier requirements.",
                )
            )

            assert _request_phases(updates) == [ResponseOnlyRequestState.FAILED]
            assert updates[0].tool_output == {
                "status": ResponseOnlyRequestState.FAILED,
                "error": "invalid_tool_arguments",
            }

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "source_turn",
    [
        "valid\x00invalid",
        "\U0001f600" * (MAX_COMMITTED_TURN_GOAL_BYTES // 4 + 1),
    ],
    ids=["nul", "over-utf8-byte-limit"],
)
def test_runtime_rejects_invalid_finalized_source_turns_before_backend_dispatch(source_turn: str) -> None:
    class _RecordingTurns(_Turns):
        def __init__(self) -> None:
            self.calls = 0

        async def commit_turn(self, request):
            self.calls += 1
            return await super().commit_turn(request)

    async def exercise() -> None:
        turns = _RecordingTurns()
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=turns,
            )
            await runtime.open_session("session-invalid-source", "conversation-invalid-source")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-invalid-source",
                    commit_id="commit-invalid-source",
                    call_id="call-invalid-source",
                    tool_name="work.delegate",
                    arguments=_delegation("Inspect the configured workspace."),
                    finalized_user_text=source_turn,
                )
            )

            assert _request_phases(updates) == [ResponseOnlyRequestState.FAILED]
            assert updates[0].tool_output == {
                "status": ResponseOnlyRequestState.FAILED,
                "error": "invalid_tool_arguments",
            }
            assert turns.calls == 0

    asyncio.run(exercise())


def test_runtime_accepts_a_finalized_source_turn_at_the_utf8_byte_limit() -> None:
    async def exercise() -> None:
        source_turn = "\U0001f600" * (MAX_COMMITTED_TURN_GOAL_BYTES // 4)
        assert len(source_turn.encode("utf-8")) == MAX_COMMITTED_TURN_GOAL_BYTES
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_Turns(),
            )
            await runtime.open_session("session-source-boundary", "conversation-source-boundary")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-source-boundary",
                    commit_id="commit-source-boundary",
                    call_id="call-source-boundary",
                    tool_name="work.delegate",
                    arguments=_delegation("Inspect the configured workspace."),
                    finalized_user_text=source_turn,
                )
            )

            assert _request_phases(updates)[-1] is ResponseOnlyRequestState.SUCCEEDED

    asyncio.run(exercise())


def test_runtime_emits_one_bounded_request_summary_separately_from_correlation() -> None:
    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_Turns(),
                request_summary_character_limit=240,
            )
            await runtime.open_session("session-summary", "conversation-summary")
            request_text = "Explain the tradeoffs " + "carefully " * 40
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-summary",
                    commit_id="commit-summary",
                    call_id="call-summary",
                    tool_name="work.delegate",
                    arguments=_delegation(request_text),
                    finalized_user_text=request_text,
                )
            )

            assert all("request_summary" not in update.correlation for update in updates)
            projection = json.loads(runtime.projection("session-summary"))
            summary = projection["execution"]["request_summary"]
            assert updates[0].request_summary == summary
            assert all(update.request_summary is None for update in updates[1:])
            assert len(summary) <= 240
            assert summary.startswith("Explain the tradeoffs carefully")
            assert not summary.endswith(" ")
            assert projection["local_requests"][0]["request_summary"] == summary

    asyncio.run(exercise())


def test_runtime_tracks_activity_and_discards_ephemeral_session_on_close() -> None:
    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=None,
            )
            await runtime.open_session("session-1", "conversation-1")
            await runtime.update_activity(
                "session-1",
                FrontendActivity(connected=True, input=InputActivity.LISTENING),
            )
            projection = json.loads(runtime.projection("session-1"))
            assert projection["activity"]["input"] == "listening"
            assert projection["connection"]["frontend_state"] == "connected"

            await runtime.close_session("session-1", "client_disconnected")
            assert store.get_session("session-1") is None
            try:
                runtime.projection("session-1")
            except LookupError:
                pass
            else:  # pragma: no cover - failure branch
                raise AssertionError("closed sessions must be evicted from process memory")

    asyncio.run(exercise())


def test_runtime_projects_public_conversation_and_terminal_playback_receipts() -> None:
    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=None,
            )
            await runtime.open_session("session-conversation", "conversation-public")
            runtime.record_conversation_turn(
                "session-conversation",
                FrontendConversationTurn(
                    turn_id="item-user",
                    role="user",
                    text="What did you find?",
                    delivery_state=FrontendConversationDeliveryState.COMMITTED,
                ),
            )
            runtime.record_conversation_turn(
                "session-conversation",
                FrontendConversationTurn(
                    turn_id="item-assistant",
                    role="assistant",
                    text="I found the requested file.",
                    delivery_state=FrontendConversationDeliveryState.DELIVERED,
                    presentation_id="presentation-assistant",
                ),
            )

            before_receipt = json.loads(runtime.projection("session-conversation"))
            assert before_receipt["recent_conversation"] == [
                {
                    "audio_end_ms": None,
                    "delivery_state": "committed",
                    "heard_through_ms": None,
                    "local_request_id": None,
                    "presentation_id": None,
                    "role": "user",
                    "text": "What did you find?",
                    "text_truncated": False,
                    "turn_id": "item-user",
                },
                {
                    "audio_end_ms": None,
                    "delivery_state": "delivered",
                    "heard_through_ms": None,
                    "local_request_id": None,
                    "presentation_id": "presentation-assistant",
                    "role": "assistant",
                    "text": "I found the requested file.",
                    "text_truncated": False,
                    "turn_id": "item-assistant",
                },
            ]
            assert before_receipt["heard_through_presentation_id"] is None
            assert before_receipt["heard_through_turn_id"] is None
            assert before_receipt["delivery"]["authoritative_client_receipt"] is False

            receipt = FrontendPlaybackReceipt(
                turn_id="item-assistant",
                state=FrontendPlaybackReceiptState.HEARD,
                presentation_id="presentation-assistant",
                heard_through_ms=840,
                audio_end_ms=840,
            )
            runtime.record_playback_receipt("session-conversation", receipt)
            runtime.record_playback_receipt("session-conversation", receipt)

            after_receipt = json.loads(runtime.projection("session-conversation"))
            assistant_turn = after_receipt["recent_conversation"][-1]
            assert assistant_turn["delivery_state"] == "heard"
            assert assistant_turn["heard_through_ms"] == 840
            assert assistant_turn["audio_end_ms"] == 840
            assert after_receipt["heard_through_presentation_id"] == "presentation-assistant"
            assert after_receipt["heard_through_turn_id"] == "item-assistant"
            assert after_receipt["delivery"] == {
                "audio_end_ms": 840,
                "authoritative_client_receipt": True,
                "heard_state": "heard",
                "heard_through_ms": 840,
                "local_request_id": None,
                "playback_state": "heard",
                "presentation_id": "presentation-assistant",
                "speech_handoff_state": "client_reported",
                "speech_policy": "approved_payload_only",
                "turn_id": "item-assistant",
            }

            with pytest.raises(ValueError, match="different terminal playback receipt"):
                runtime.record_playback_receipt(
                    "session-conversation",
                    FrontendPlaybackReceipt(
                        turn_id="item-assistant",
                        state=FrontendPlaybackReceiptState.INTERRUPTED,
                        presentation_id="presentation-assistant",
                        heard_through_ms=400,
                        audio_end_ms=840,
                    ),
                )
            with pytest.raises(ValueError, match="assistant conversation turn"):
                runtime.record_playback_receipt(
                    "session-conversation",
                    FrontendPlaybackReceipt(
                        turn_id="item-user",
                        state=FrontendPlaybackReceiptState.HEARD,
                        presentation_id="presentation-user-invalid",
                    ),
                )
            runtime.record_playback_receipt(
                "session-conversation",
                FrontendPlaybackReceipt(
                    turn_id="item-without-transcript",
                    state=FrontendPlaybackReceiptState.FAILED,
                    presentation_id="presentation-failed",
                    heard_through_ms=0,
                    audio_end_ms=100,
                ),
            )
            failed_receipt = json.loads(runtime.projection("session-conversation"))
            assert failed_receipt["recent_conversation"] == after_receipt["recent_conversation"]
            assert failed_receipt["heard_through_turn_id"] == "item-assistant"
            assert failed_receipt["delivery"]["playback_state"] == "failed"
            assert failed_receipt["delivery"]["authoritative_client_receipt"] is False

    asyncio.run(exercise())


def test_runtime_bounds_recent_conversation_without_rewriting_backend_state() -> None:
    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=None,
            )
            await runtime.open_session("session-bounded", "conversation-bounded")
            for index in range(10):
                runtime.record_conversation_turn(
                    "session-bounded",
                    FrontendConversationTurn(
                        turn_id=f"item-{index}",
                        role="user",
                        text=f"{index}:" + ("x" * 600),
                        delivery_state=FrontendConversationDeliveryState.COMMITTED,
                    ),
                )

            projection = json.loads(runtime.projection("session-bounded"))
            assert [turn["turn_id"] for turn in projection["recent_conversation"]] == [
                f"item-{index}" for index in range(2, 10)
            ]
            assert all(len(turn["text"]) == 512 for turn in projection["recent_conversation"])
            assert all(turn["text_truncated"] is True for turn in projection["recent_conversation"])
            assert projection["active_work"] == []
            assert projection["active_local_requests"] == []
            assert projection["pending_interactions"] == []
            assert projection["ready_results"] == []

    asyncio.run(exercise())


def test_runtime_accepts_receipt_without_transcript_and_correlates_ready_result() -> None:
    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_Turns(),
            )
            await runtime.open_session("session-receipt-only", "conversation-receipt-only")
            await _collect(
                runtime.execute_tool(
                    session_id="session-receipt-only",
                    commit_id="commit-receipt-only",
                    call_id="call-receipt-only",
                    tool_name="work.delegate",
                    arguments=_delegation("find the file"),
                    finalized_user_text="find the file",
                )
            )

            runtime.record_playback_receipt(
                "session-receipt-only",
                FrontendPlaybackReceipt(
                    turn_id="item-without-transcript",
                    state=FrontendPlaybackReceiptState.INTERRUPTED,
                    presentation_id="presentation-without-transcript",
                    local_request_id="commit-receipt-only",
                    heard_through_ms=250,
                    audio_end_ms=900,
                ),
            )

            projection = json.loads(runtime.projection("session-receipt-only"))
            assert projection["recent_conversation"] == []
            assert projection["heard_through_presentation_id"] == "presentation-without-transcript"
            assert projection["heard_through_turn_id"] == "item-without-transcript"
            assert projection["ready_results"] == [
                {
                    "delivery_state": "interrupted",
                    "display_available": True,
                    "identity_authority": "voiceclaw_local",
                    "local_request_id": "commit-receipt-only",
                    "speech_source": "backend_authored",
                    "state": "available",
                }
            ]
            assert "done: find the file" not in runtime.projection("session-receipt-only")
            assert "backend-session-1" not in runtime.projection("session-receipt-only")

    asyncio.run(exercise())


def test_runtime_projection_exposes_only_truthful_active_request_state() -> None:
    class _BlockingTurns(_Turns):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def commit_turn(self, request):
            self.started.set()
            await self.release.wait()
            return await super().commit_turn(request)

    async def exercise() -> None:
        turns = _BlockingTurns()
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=turns,
            )
            await runtime.open_session("session-active", "conversation-active")
            updates = runtime.execute_tool(
                session_id="session-active",
                commit_id="commit-active",
                call_id="call-active",
                tool_name="work.delegate",
                arguments=_delegation("do the work"),
                finalized_user_text="do the work",
            )

            queued = await anext(updates)
            assert queued.phase == "locally_queued"
            assert queued.tool_output == {
                "status": "locally_queued",
                "request_state": "locally_queued",
                "local_request_id": "commit-active",
                "identity_authority": "voiceclaw_local",
                "evidence": "voiceclaw_local",
                "backend_acceptance": "unknown",
                "durability": "none",
            }
            assert queued.frontend_response is not None
            assert queued.frontend_response.purpose is FrontendResponsePurpose.DELEGATION_ACK
            assert queued.frontend_response.local_request_id == "commit-active"
            assert queued.frontend_response.payload_text == "do the work"
            dispatching = await anext(updates)
            assert dispatching.phase == "dispatching"
            assert dispatching.tool_output is None
            assert dispatching.frontend_response is None
            waiting = await anext(updates)
            assert waiting.phase == "waiting_for_response"
            assert waiting.tool_output is None
            assert waiting.frontend_response is None
            remainder = asyncio.create_task(_collect(updates))
            await turns.started.wait()

            projection_text = runtime.projection("session-active")
            projection = json.loads(projection_text)
            assert projection["execution"]["request_state"] == "waiting_for_response"
            assert projection["execution"]["request_active"] is True
            assert projection["execution"]["any_request_active"] is True
            assert projection["execution"]["durable_work_id_issued"] is False
            assert projection["execution"]["local_request_id"] == "commit-active"
            assert projection["execution"]["identity_authority"] == "voiceclaw_local"
            assert projection["execution"]["request_summary"] == "do the work"
            assert projection["local_requests"][0]["local_request_id"] == "commit-active"
            assert projection["local_requests"][0]["request_state"] == "waiting_for_response"
            assert projection["local_requests"][0]["request_summary"] == "do the work"
            assert projection["active_work"] == []
            assert projection["pending_interactions"] == []
            assert projection["ready_results"] == []
            assert projection["active_local_requests"] == projection["local_requests"]
            assert projection["latest_result"]["state"] == "none"
            assert "do the work" in projection_text
            assert "commit-active" in projection_text
            assert "call-active" not in projection_text

            turns.release.set()
            remainder_updates = await remainder
            assert _display_text(remainder_updates) == "done: do the work"
            completed = remainder_updates[-1]
            assert completed.phase == "succeeded"
            assert completed.tool_output is None
            assert completed.frontend_response is not None
            assert completed.frontend_response.purpose is FrontendResponsePurpose.RESULT_DELIVERY
            assert completed.frontend_response.local_request_id == "commit-active"
            assert completed.frontend_response.payload_text == "Done: do the work."

    asyncio.run(exercise())


def test_runtime_execution_activity_is_consistent_when_a_busy_attempt_fails() -> None:
    class _BlockingTurns(_Turns):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def commit_turn(self, request):
            self.started.set()
            await self.release.wait()
            return await super().commit_turn(request)

    async def exercise() -> None:
        turns = _BlockingTurns()
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=turns,
            )
            await runtime.open_session("session-busy", "conversation-busy")
            first = runtime.execute_tool(
                session_id="session-busy",
                commit_id="commit-active",
                call_id="call-active",
                tool_name="work.delegate",
                arguments=_delegation("first request"),
                finalized_user_text="first request",
            )
            assert (await anext(first)).phase == "locally_queued"
            assert (await anext(first)).phase == "dispatching"
            assert (await anext(first)).phase == "waiting_for_response"
            remainder = asyncio.create_task(_collect(first))
            await turns.started.wait()

            second = await _collect(
                runtime.execute_tool(
                    session_id="session-busy",
                    commit_id="commit-busy",
                    call_id="call-busy",
                    tool_name="work.delegate",
                    arguments=_delegation("second request"),
                    finalized_user_text="second request",
                )
            )
            assert _request_phases(second) == [ResponseOnlyRequestState.FAILED]
            projection = json.loads(runtime.projection("session-busy"))
            assert projection["execution"]["local_request_id"] == "commit-busy"
            assert projection["execution"]["request_state"] == "failed"
            assert projection["execution"]["request_active"] is False
            assert projection["execution"]["any_request_active"] is True

            turns.release.set()
            first_remainder = await remainder
            assert _display_text(first_remainder) == "done: first request"
            assert first_remainder[-1].phase == "succeeded"

    asyncio.run(exercise())


def test_runtime_serializes_backend_admission_before_reporting_dispatch() -> None:
    class _SerialTurns(_Turns):
        def __init__(self) -> None:
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()
            self.active = 0
            self.max_active = 0

        async def commit_turn(self, request):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                if request.commit_id == "commit-first":
                    self.first_started.set()
                    await self.release_first.wait()
                return await super().commit_turn(request)
            finally:
                self.active -= 1

    async def exercise() -> None:
        turns = _SerialTurns()
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=turns,
            )
            await runtime.open_session("session-first", "conversation-first")
            await runtime.open_session("session-second", "conversation-second")
            first = runtime.execute_tool(
                session_id="session-first",
                commit_id="commit-first",
                call_id="call-first",
                tool_name="work.delegate",
                arguments=_delegation("first request"),
                finalized_user_text="first request",
            )
            second = runtime.execute_tool(
                session_id="session-second",
                commit_id="commit-second",
                call_id="call-second",
                tool_name="work.delegate",
                arguments=_delegation("second request"),
                finalized_user_text="second request",
            )

            first_receipt = await anext(first)
            second_receipt = await anext(second)
            assert first_receipt.phase == second_receipt.phase == "locally_queued"
            assert first_receipt.frontend_response is not None
            assert second_receipt.frontend_response is not None
            assert first_receipt.frontend_response.local_request_id == "commit-first"
            assert second_receipt.frontend_response.local_request_id == "commit-second"

            assert (await anext(first)).phase == "dispatching"
            assert (await anext(first)).phase == "waiting_for_response"
            first_remainder = asyncio.create_task(_collect(first))
            await turns.first_started.wait()

            second_dispatch = asyncio.create_task(anext(second))
            await asyncio.sleep(0)
            assert second_dispatch.done() is False
            second_projection = json.loads(runtime.projection("session-second"))
            assert second_projection["execution"]["request_state"] == "locally_queued"
            assert second_projection["execution"]["request_active"] is True
            assert second_projection["execution"]["any_request_active"] is True

            turns.release_first.set()
            first_updates = await asyncio.wait_for(first_remainder, timeout=1)
            first_result = first_updates[-1]
            assert _display_text(first_updates) == "done: first request"
            assert first_result.phase == "succeeded"
            assert (await asyncio.wait_for(second_dispatch, timeout=1)).phase == "dispatching"
            assert (await anext(second)).phase == "waiting_for_response"
            second_updates = await _collect(second)
            second_result = second_updates[-1]
            assert _display_text(second_updates) == "done: second request"
            assert second_result.phase == "succeeded"
            assert turns.max_active == 1

    asyncio.run(exercise())


def test_runtime_preserves_completed_result_identity_when_a_new_request_starts() -> None:
    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_Turns(),
            )
            await runtime.open_session("session-shared", "conversation-shared")
            first_updates = await _collect(
                runtime.execute_tool(
                    session_id="session-shared",
                    commit_id="commit-first",
                    call_id="call-first",
                    tool_name="work.delegate",
                    arguments=_delegation("first result"),
                    finalized_user_text="first result",
                )
            )
            first_delivery = first_updates[-1].frontend_response
            assert first_delivery is not None
            assert first_delivery.local_request_id == "commit-first"

            second = runtime.execute_tool(
                session_id="session-shared",
                commit_id="commit-second",
                call_id="call-second",
                tool_name="work.delegate",
                arguments=_delegation("second result"),
                finalized_user_text="second result",
            )
            second_receipt = await anext(second)
            assert second_receipt.frontend_response is not None
            assert second_receipt.frontend_response.local_request_id == "commit-second"

            projection = json.loads(runtime.projection("session-shared"))
            assert projection["execution"]["local_request_id"] == "commit-second"
            requests = {request["local_request_id"]: request for request in projection["local_requests"]}
            assert requests["commit-first"]["request_state"] == "succeeded"
            assert requests["commit-first"]["result"]["state"] == "available"
            assert requests["commit-second"]["request_state"] == "locally_queued"
            assert requests["commit-second"]["result"]["state"] == "none"
            assert first_delivery.payload_text == "Done: first result."
            assert "commit-first" not in first_delivery.payload_text
            await second.aclose()

    asyncio.run(exercise())


def test_cancelled_ephemeral_turn_releases_process_admission_lane() -> None:
    class _CancellableTurns(_Turns):
        def __init__(self) -> None:
            self.calls = 0
            self.started = asyncio.Event()

        async def commit_turn(self, request):
            self.calls += 1
            if self.calls == 1:
                self.started.set()
                await asyncio.Future()
            return await super().commit_turn(request)

    async def exercise() -> None:
        turns = _CancellableTurns()
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=turns,
            )
            await runtime.open_session("session-abandoned", "conversation-abandoned")
            await runtime.open_session("session-next", "conversation-next")

            abandoned = runtime.execute_tool(
                session_id="session-abandoned",
                commit_id="commit-abandoned",
                call_id="call-abandoned",
                tool_name="work.delegate",
                arguments=_delegation("a request whose client disconnects"),
                finalized_user_text="a request whose client disconnects",
            )
            assert (await anext(abandoned)).phase == "locally_queued"
            assert (await anext(abandoned)).phase == "dispatching"
            assert (await anext(abandoned)).phase == "waiting_for_response"
            terminal = asyncio.create_task(anext(abandoned))
            await turns.started.wait()
            terminal.cancel()
            try:
                await terminal
            except asyncio.CancelledError:
                pass
            else:  # pragma: no cover - failure branch
                raise AssertionError("cancelling the abandoned turn must propagate")

            abandoned_projection = json.loads(runtime.projection("session-abandoned"))
            assert abandoned_projection["execution"]["request_active"] is False
            assert abandoned_projection["execution"]["any_request_active"] is False
            assert abandoned_projection["execution"]["request_state"] == "failed"
            assert abandoned_projection["execution"]["latest_failure_code"] == "runtime_interrupted"
            assert abandoned_projection["local_requests"][0]["request_state"] == "failed"

            following = runtime.execute_tool(
                session_id="session-next",
                commit_id="commit-next",
                call_id="call-next",
                tool_name="work.delegate",
                arguments=_delegation("the next admitted request"),
                finalized_user_text="the next admitted request",
            )
            updates = await asyncio.wait_for(_collect(following), timeout=1)
            assert _request_phases(updates) == [
                "locally_queued",
                "dispatching",
                "waiting_for_response",
                "succeeded",
            ]
            assert _display_text(updates) == "done: the next admitted request"

    asyncio.run(exercise())


def test_runtime_keeps_realtime_available_when_backend_inspection_fails() -> None:
    class _UnavailableTurns(_Turns):
        async def inspect(self):
            raise OSError("backend offline")

    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_UnavailableTurns(),
            )
            snapshot = await runtime.open_session("session-offline", "conversation-offline")

            assert snapshot.gateway_reachable is False
            assert snapshot.backend_label == "Backend unavailable"
            assert store.get_session("session-offline") is not None
            projection = json.loads(runtime.projection("session-offline"))
            assert projection["gateway_reachable"] is False
            assert projection["connection"]["gateway_state"] == "unavailable"
            assert projection["connection"]["attachment_state"] == "unavailable"
            assert projection["contract"]["operations"] == []
            assert projection["contract"]["frontend_tools"] == []

    asyncio.run(exercise())


def test_runtime_keeps_markdown_code_display_only_without_inferred_speech() -> None:
    class _LongResultTurns(_Turns):
        async def commit_turn(self, request):
            del request
            return CommittedTurnResult(
                backend_session_id="backend-session-long",
                turn_id="turn-long",
                response_id="response-long",
                display_text="```python\n" + ("print('display only')\n" * 40) + "```",
            )

    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_LongResultTurns(),
            )
            await runtime.open_session("session-long", "conversation-long")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-long",
                    commit_id="commit-long",
                    call_id="call-long",
                    tool_name="work.delegate",
                    arguments=_delegation("write code"),
                    finalized_user_text="write code",
                )
            )

            terminal = updates[-1]
            assert "print('display only')" in _display_text(updates)
            assert terminal.text == ""
            assert terminal.frontend_response is None
            assert terminal.correlation["speech_source"] == ResponseOnlySpeechSource.NONE.value
            projection = json.loads(runtime.projection("session-long"))
            assert projection["latest_result"]["speech_source"] == ResponseOnlySpeechSource.NONE.value

    asyncio.run(exercise())


def test_runtime_does_not_interpret_unclosed_markdown_as_speech() -> None:
    class _UnclosedCodeTurns(_Turns):
        async def commit_turn(self, request):
            del request
            return CommittedTurnResult(
                backend_session_id="backend-session-unclosed",
                turn_id="turn-unclosed",
                response_id="response-unclosed",
                display_text="Implementation:\n```python\nprint('never speak this')",
            )

    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_UnclosedCodeTurns(),
            )
            await runtime.open_session("session-unclosed", "conversation-unclosed")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-unclosed",
                    commit_id="commit-unclosed",
                    call_id="call-unclosed",
                    tool_name="work.delegate",
                    arguments=_delegation("write code"),
                    finalized_user_text="write code",
                )
            )

            terminal = updates[-1]
            assert _display_text(updates) == "Implementation:\n```python\nprint('never speak this')"
            assert terminal.text == ""
            assert terminal.frontend_response is None

    asyncio.run(exercise())


def test_runtime_keeps_markdown_lists_display_only_without_summarizing_them() -> None:
    class _RichResultTurns(_Turns):
        async def commit_turn(self, request):
            del request
            return CommittedTurnResult(
                backend_session_id="backend-session-rich",
                turn_id="turn-rich",
                response_id="response-rich",
                display_text="There are 2 entries.\n\n- AGENTS.md\n- USER.md",
            )

    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_RichResultTurns(),
            )
            await runtime.open_session("session-rich", "conversation-rich")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-rich",
                    commit_id="commit-rich",
                    call_id="call-rich",
                    tool_name="work.delegate",
                    arguments=_delegation("inspect the workspace"),
                    finalized_user_text="inspect the workspace",
                )
            )

            terminal = updates[-1]
            assert _display_text(updates).endswith("- AGENTS.md\n- USER.md")
            assert terminal.text == ""
            assert terminal.frontend_response is None

    asyncio.run(exercise())


def test_runtime_never_derives_speech_from_large_display_documents() -> None:
    class _BoundedResultTurns(_Turns):
        def __init__(self, display_text: str) -> None:
            self.display_text = display_text

        async def commit_turn(self, request):
            del request
            return CommittedTurnResult(
                backend_session_id="backend-session-bounded",
                turn_id="turn-bounded",
                response_id="response-bounded",
                display_text=self.display_text,
            )

    async def render(display_text: str, suffix: str) -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_BoundedResultTurns(display_text),
            )
            await runtime.open_session(f"session-bounded-{suffix}", f"conversation-bounded-{suffix}")
            updates = await _collect(
                runtime.execute_tool(
                    session_id=f"session-bounded-{suffix}",
                    commit_id=f"commit-bounded-{suffix}",
                    call_id=f"call-bounded-{suffix}",
                    tool_name="work.delegate",
                    arguments=_delegation("render the requested document"),
                    finalized_user_text="render the requested document",
                )
            )
            assert _display_text(updates) == display_text
            assert updates[-1].text == ""
            assert updates[-1].frontend_response is None

    async def exercise() -> None:
        await render(
            "First approved sentence. Second approved sentence. Third display-only sentence.",
            "sentences",
        )
        await render(" ".join("word" for _ in range(50)), "bounded")
        await render(
            " ".join(f"fact-{index:04d}." for index in range(600)),
            "basis-overflow",
        )

    asyncio.run(exercise())


def test_runtime_does_not_filter_or_summarize_markdown_tables() -> None:
    class _TableResultTurns(_Turns):
        async def commit_turn(self, request):
            del request
            return CommittedTurnResult(
                backend_session_id="backend-session-table",
                turn_id="turn-table",
                response_id="response-table",
                display_text=(
                    "Search complexity depends on balance. Would you like a walkthrough?\n"
                    "The comparison is complete, and I can also explain more.\n\n"
                    "The American I interviewed reported O(log n) performance.\n\n"
                    "| **Case** | **Behavior** |\n"
                    "| --- | --- |\n"
                    "| Source | https://example.com/details |\n"
                    "| File | /tmp/tree.py |\n"
                    "| Relative file | src/tree.py |\n"
                    "| Docs | [guide](docs/guide.md) |\n"
                    "| Code | `print(tree)` |\n"
                    "| Average case | O(log n) when balanced |\n"
                    "| Worst case | O(n) when degenerate |\n"
                    "| Supported operations | O(log n) for search/insert/delete |\n"
                    "| Mitigation | Use a self-balancing tree |\n"
                    "| Extra | This fourth safe row stays display-only |\n\n"
                    "The implementation is in src/tree.py.\n"
                    "Use `print(tree)` to inspect it.\n"
                    "Documentation: www.example.com/tree.\n"
                    "I can also provide another example.\n"
                    "Do you want implementation details?"
                ),
            )

    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_TableResultTurns(),
            )
            await runtime.open_session("session-table", "conversation-table")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-table",
                    commit_id="commit-table",
                    call_id="call-table",
                    tool_name="work.delegate",
                    arguments=_delegation("compare tree search complexity"),
                    finalized_user_text="compare tree search complexity",
                )
            )

            terminal = updates[-1]
            assert terminal.frontend_response is None
            streamed_display = _display_text(updates)
            assert "https://example.com/details" in streamed_display
            assert "/tmp/tree.py" in streamed_display
            assert "Would you like a walkthrough?" in streamed_display
            assert "Do you want implementation details?" in streamed_display
            projection = json.loads(runtime.projection("session-table"))
            assert projection["latest_result"]["speech_source"] == ResponseOnlySpeechSource.NONE.value

    asyncio.run(exercise())


def test_runtime_rejects_a_whitespace_only_committed_result() -> None:
    class _EmptyResultTurns(_Turns):
        async def commit_turn(self, request):
            del request
            return CommittedTurnResult(
                backend_session_id="backend-session-empty",
                turn_id="turn-empty",
                response_id="response-empty",
                display_text=" \n\t ",
            )

    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_EmptyResultTurns(),
            )
            await runtime.open_session("session-empty", "conversation-empty")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-empty",
                    commit_id="commit-empty",
                    call_id="call-empty",
                    tool_name="work.delegate",
                    arguments=_delegation("complete the operation"),
                    finalized_user_text="complete the operation",
                )
            )

            assert _request_phases(updates) == [
                ResponseOnlyRequestState.LOCALLY_QUEUED,
                ResponseOnlyRequestState.DISPATCHING,
                ResponseOnlyRequestState.WAITING_FOR_RESPONSE,
                ResponseOnlyRequestState.FAILED,
            ]
            assert [update.phase for update in _display_updates(updates)] == [
                ResponseOnlyResultEventKind.DISPLAY_DELTA,
            ]
            assert _display_text(updates) == " \n\t "
            terminal = updates[-1]
            assert terminal.text == "I couldn't use the response from the configured target."
            assert terminal.correlation["error_code"] == "agent_protocol_error"
            assert terminal.tool_output is None
            assert terminal.frontend_response is not None
            assert terminal.frontend_response.purpose is FrontendResponsePurpose.FAILURE_DELIVERY
            assert terminal.frontend_response.payload_text == "I couldn't use the response from the configured target."
            projection = json.loads(runtime.projection("session-empty"))
            assert projection["execution"]["latest_terminal_outcome"] == "failed"
            assert projection["execution"]["latest_failure_code"] == "agent_protocol_error"
            assert projection["latest_result"]["state"] == "none"
            assert projection["latest_result"]["display_available"] is False

    asyncio.run(exercise())


def test_runtime_rejects_out_of_order_and_mismatched_display_streams() -> None:
    class _OutOfOrderTurns(_Turns):
        async def stream_turn(self, request):
            result = await self.commit_turn(request)
            yield CommittedTurnDisplayDelta(
                backend_session_id=result.backend_session_id,
                turn_id=result.turn_id,
                response_id=result.response_id,
                sequence=1,
                delta=result.display_text,
            )
            yield CommittedTurnCompleted(result=result)

    class _MismatchedTurns(_Turns):
        async def stream_turn(self, request):
            result = await self.commit_turn(request)
            yield CommittedTurnDisplayDelta(
                backend_session_id=result.backend_session_id,
                turn_id=result.turn_id,
                response_id=result.response_id,
                sequence=0,
                delta="provisional",
            )
            yield CommittedTurnCompleted(result=result)

    async def assert_protocol_failure(turns, suffix: str, provisional_display: str) -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=turns,
            )
            await runtime.open_session(f"session-{suffix}", f"conversation-{suffix}")
            updates = await _collect(
                runtime.execute_tool(
                    session_id=f"session-{suffix}",
                    commit_id=f"commit-{suffix}",
                    call_id=f"call-{suffix}",
                    tool_name="work.delegate",
                    arguments=_delegation("run the request"),
                    finalized_user_text="run the request",
                )
            )

            assert _request_phases(updates) == [
                ResponseOnlyRequestState.LOCALLY_QUEUED,
                ResponseOnlyRequestState.DISPATCHING,
                ResponseOnlyRequestState.WAITING_FOR_RESPONSE,
                ResponseOnlyRequestState.FAILED,
            ]
            assert _display_text(updates) == provisional_display
            assert not any(
                update.phase is ResponseOnlyResultEventKind.COMPLETED for update in _display_updates(updates)
            )
            assert updates[-1].correlation["error_code"] == "agent_protocol_error"

    async def exercise() -> None:
        await assert_protocol_failure(_OutOfOrderTurns(), "out-of-order", "")
        await assert_protocol_failure(_MismatchedTurns(), "mismatched", "provisional")

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("agent_failed", "The configured target couldn't complete that request."),
        ("agent_protocol_error", "I couldn't use the response from the configured target."),
        ("protocol_error", "I couldn't use the response from the configured target."),
        (
            "response_too_large",
            "The configured target returned more detail than this session can present.",
        ),
    ],
)
def test_backend_failure_copy_distinguishes_execution_protocol_and_capacity(code: str, message: str) -> None:
    copy = load_model_contract_catalog().failure_copy(code)
    assert (copy.display, copy.speech) == (message, message)


def test_runtime_leaves_markdown_code_and_bullets_entirely_in_display() -> None:
    class _ComplexityResultTurns(_Turns):
        async def commit_turn(self, request):
            del request
            return CommittedTurnResult(
                backend_session_id="backend-session-complexity",
                turn_id="turn-complexity",
                response_id="response-complexity",
                display_text=(
                    "```python\nprint('display only')\n```\n\n"
                    "**Search complexity:**\n\n"
                    "- **Average case: O(log n)** when the tree is reasonably balanced.\n"
                    "- **Worst case: O(n)** when the tree degenerates into a linked list.\n"
                    "- Use a self-balancing tree to preserve logarithmic search.\n"
                    "- A fourth detail remains available only in the display."
                ),
            )

    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_ComplexityResultTurns(),
            )
            await runtime.open_session("session-complexity", "conversation-complexity")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-complexity",
                    commit_id="commit-complexity",
                    call_id="call-complexity",
                    tool_name="work.delegate",
                    arguments=_delegation("explain search complexity"),
                    finalized_user_text="explain search complexity",
                )
            )

            terminal = updates[-1]
            assert terminal.frontend_response is None
            streamed_display = _display_text(updates)
            assert "print('display only')" in streamed_display
            assert "A fourth detail remains available only in the display." in streamed_display

    asyncio.run(exercise())


def test_runtime_does_not_extract_speech_from_markdown_sections() -> None:
    class _ComparisonTurns(_Turns):
        async def commit_turn(self, request):
            del request
            return CommittedTurnResult(
                backend_session_id="backend-session-comparison",
                turn_id="turn-comparison",
                response_id="response-comparison",
                display_text=(
                    "# Search comparison\n\n"
                    "## Breadth-First Search (BFS)\n\n"
                    "How it works: It explores nodes level by level.\n\n"
                    "**Practical use case:** **Web crawlers** – Web crawlers discover nearby pages first.\n\n"
                    "## Depth-First Search (DFS)\n\n"
                    "How it works: It explores one branch before backtracking.\n\n"
                    "**Practical use case:** **Maze solving** – Maze solving explores one corridor before backtracking."
                ),
            )

    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_ComparisonTurns(),
            )
            await runtime.open_session("session-comparison", "conversation-comparison")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-comparison",
                    commit_id="commit-comparison",
                    call_id="call-comparison",
                    tool_name="work.delegate",
                    arguments=_delegation(
                        "Compare breadth-first search and depth-first search and give one practical use case for each."
                    ),
                    finalized_user_text=(
                        "Compare breadth-first search and depth-first search and give one practical use case for each."
                    ),
                )
            )

            terminal = updates[-1]
            assert terminal.frontend_response is None
            streamed_display = _display_text(updates)
            assert "Breadth-First Search (BFS)" in streamed_display
            assert "Web crawlers" in streamed_display
            assert "Depth-First Search (DFS)" in streamed_display
            assert "Maze solving" in streamed_display

    asyncio.run(exercise())


def test_runtime_does_not_extract_speech_from_markdown_tables_and_sections() -> None:
    class _ComparisonTableTurns(_Turns):
        async def commit_turn(self, request):
            del request
            return CommittedTurnResult(
                backend_session_id="backend-session-comparison-table",
                turn_id="turn-comparison-table",
                response_id="response-comparison-table",
                display_text=(
                    "Breadth-First Search (BFS) vs. Depth-First Search (DFS)\n\n"
                    "| Aspect | Breadth-First Search (BFS) | Depth-First Search (DFS) |\n"
                    "|---|---|---|\n"
                    "| **Traversal order** | Level by level | One branch before backtracking |\n"
                    "| **Data structure** | Queue (FIFO) | Stack (LIFO) or recursion |\n\n"
                    "### Practical Use Cases\n\n"
                    "#### BFS: **Web crawlers / shortest path finding**\n"
                    "Additional display detail for BFS.\n\n"
                    "#### DFS: **Maze solving / dependency resolution**\n"
                    "Additional display detail for DFS."
                ),
            )

    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_ComparisonTableTurns(),
            )
            await runtime.open_session("session-comparison-table", "conversation-comparison-table")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-comparison-table",
                    commit_id="commit-comparison-table",
                    call_id="call-comparison-table",
                    tool_name="work.delegate",
                    arguments=_delegation(
                        "Compare breadth-first search and depth-first search and give one practical use case for each."
                    ),
                    finalized_user_text=(
                        "Compare breadth-first search and depth-first search and give one practical use case for each."
                    ),
                )
            )

            terminal = updates[-1]
            assert terminal.frontend_response is None
            streamed_display = _display_text(updates)
            assert "Traversal order" in streamed_display
            assert "Queue (FIFO)" in streamed_display
            assert "Web crawlers / shortest path finding" in streamed_display
            assert "Maze solving / dependency resolution" in streamed_display

    asyncio.run(exercise())


def test_runtime_keeps_a_heading_and_list_display_only_without_speech() -> None:
    class _HeadingResultTurns(_Turns):
        async def commit_turn(self, request):
            del request
            return CommittedTurnResult(
                backend_session_id="backend-session-heading",
                turn_id="turn-heading",
                response_id="response-heading",
                display_text="Workspace entries:\n\n- AGENTS.md\n- USER.md",
            )

    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_HeadingResultTurns(),
            )
            await runtime.open_session("session-heading", "conversation-heading")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-heading",
                    commit_id="commit-heading",
                    call_id="call-heading",
                    tool_name="work.delegate",
                    arguments=_delegation("inspect the workspace"),
                    finalized_user_text="inspect the workspace",
                )
            )

            frontend_response = updates[-1].frontend_response
            assert frontend_response is None
            projection = json.loads(runtime.projection("session-heading"))
            assert projection["latest_result"]["speech_source"] == ResponseOnlySpeechSource.NONE.value

    asyncio.run(exercise())


def test_runtime_preserves_explicit_backend_speech_provenance() -> None:
    class _StructuredResultTurns(_Turns):
        async def commit_turn(self, request):
            del request
            return CommittedTurnResult(
                backend_session_id="backend-session-structured",
                turn_id="turn-structured",
                response_id="response-structured",
                display_text="A long display result that stays out of model context.",
                speak_text="The structured result is ready.",
            )

    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_StructuredResultTurns(),
            )
            await runtime.open_session("session-structured", "conversation-structured")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-structured",
                    commit_id="commit-structured",
                    call_id="call-structured",
                    tool_name="work.delegate",
                    arguments=_delegation("do structured work"),
                    finalized_user_text="do structured work",
                )
            )

            frontend_response = updates[-1].frontend_response
            assert frontend_response is not None
            assert frontend_response.purpose is FrontendResponsePurpose.RESULT_DELIVERY
            assert frontend_response.local_request_id == "commit-structured"
            assert frontend_response.payload_text == "The structured result is ready."
            serialized_projection = runtime.projection("session-structured")
            projection = json.loads(serialized_projection)
            assert projection["latest_result"]["speech_source"] == ResponseOnlySpeechSource.BACKEND_AUTHORED.value
            assert projection["latest_result"]["display_available"] is True
            assert projection["latest_result"]["display_body_in_context"] is False
            assert projection["latest_result"]["provider_identifiers_in_context"] is False
            assert "A long display result that stays out of model context." not in serialized_projection
            assert "The structured result is ready." not in serialized_projection

    asyncio.run(exercise())


def test_runtime_routes_backend_authored_url_speech_without_content_filtering() -> None:
    class _UnsafeSpeechTurns(_Turns):
        async def commit_turn(self, request):
            del request
            return CommittedTurnResult(
                backend_session_id="backend-session-unsafe-speech",
                turn_id="turn-unsafe-speech",
                response_id="response-unsafe-speech",
                display_text="The operation completed successfully.",
                speak_text="Open https://example.invalid/result to see it.",
            )

    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_UnsafeSpeechTurns(),
            )
            await runtime.open_session("session-unsafe-speech", "conversation-unsafe-speech")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-unsafe-speech",
                    commit_id="commit-unsafe-speech",
                    call_id="call-unsafe-speech",
                    tool_name="work.delegate",
                    arguments=_delegation("complete the operation"),
                    finalized_user_text="complete the operation",
                )
            )

            response = updates[-1].frontend_response
            assert response is not None
            assert response.purpose is FrontendResponsePurpose.RESULT_DELIVERY
            assert response.payload_text == "Open https://example.invalid/result to see it."
            projection = json.loads(runtime.projection("session-unsafe-speech"))
            assert projection["latest_result"]["speech_source"] == ResponseOnlySpeechSource.BACKEND_AUTHORED.value

    asyncio.run(exercise())


def test_runtime_routes_backend_authored_question_speech_without_content_filtering() -> None:
    class _FollowUpSpeechTurns(_Turns):
        async def commit_turn(self, request):
            del request
            return CommittedTurnResult(
                backend_session_id="backend-session-follow-up-speech",
                turn_id="turn-follow-up-speech",
                response_id="response-follow-up-speech",
                display_text="The operation completed successfully. Would you like more detail?",
                speak_text="The operation completed successfully. Would you like more detail?",
            )

    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_FollowUpSpeechTurns(),
            )
            await runtime.open_session("session-follow-up-speech", "conversation-follow-up-speech")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-follow-up-speech",
                    commit_id="commit-follow-up-speech",
                    call_id="call-follow-up-speech",
                    tool_name="work.delegate",
                    arguments=_delegation("complete the operation"),
                    finalized_user_text="complete the operation",
                )
            )

            response = updates[-1].frontend_response
            assert response is not None
            assert response.purpose is FrontendResponsePurpose.RESULT_DELIVERY
            assert response.payload_text == "The operation completed successfully. Would you like more detail?"
            projection = json.loads(runtime.projection("session-follow-up-speech"))
            assert projection["latest_result"]["speech_source"] == ResponseOnlySpeechSource.BACKEND_AUTHORED.value

    asyncio.run(exercise())


def test_runtime_preserves_safe_adapter_failure_code() -> None:
    class _FailedTurns(_Turns):
        async def commit_turn(self, request):
            del request
            raise CommittedTurnError("turn_timeout")

    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_FailedTurns(),
            )
            await runtime.open_session("session-failed", "conversation-failed")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-failed",
                    commit_id="commit-failed",
                    call_id="call-failed",
                    tool_name="work.delegate",
                    arguments=_delegation("do slow work"),
                    finalized_user_text="do slow work",
                )
            )

            assert _request_phases(updates) == [
                ResponseOnlyRequestState.LOCALLY_QUEUED,
                ResponseOnlyRequestState.DISPATCHING,
                ResponseOnlyRequestState.WAITING_FOR_RESPONSE,
                ResponseOnlyRequestState.FAILED,
            ]
            assert updates[0].tool_output == {
                "status": "locally_queued",
                "request_state": "locally_queued",
                "local_request_id": "commit-failed",
                "identity_authority": "voiceclaw_local",
                "evidence": "voiceclaw_local",
                "backend_acceptance": "unknown",
                "durability": "none",
            }
            assert updates[-1].tool_output is None
            assert updates[-1].frontend_response is not None
            assert updates[-1].frontend_response.purpose is FrontendResponsePurpose.FAILURE_DELIVERY
            assert updates[-1].frontend_response.local_request_id == "commit-failed"
            assert updates[-1].frontend_response.payload_text == "I couldn't finish that request in time."
            projection = json.loads(runtime.projection("session-failed"))
            assert projection["execution"]["latest_terminal_outcome"] == "failed"
            assert projection["execution"]["latest_failure_code"] == "turn_timeout"
            assert projection["latest_result"]["state"] == "none"
            assert "timed out" not in runtime.projection("session-failed")

    asyncio.run(exercise())


def test_runtime_uses_selected_failure_display_and_speech_copy(tmp_path: Path) -> None:
    class _FailedTurns(_Turns):
        async def commit_turn(self, request):
            del request
            raise CommittedTurnError("turn_timeout")

    override = tmp_path / "model-contracts.yaml"
    override.write_text(
        _MODEL_CONTRACTS.read_text(encoding="utf-8")
        .replace("title: Request failed", "title: Operation unavailable", 1)
        .replace(
            "display: I couldn't finish that request in time.\n"
            "            speech: I couldn't finish that request in time.",
            "display: The configured operation exceeded its time budget.\n"
            "            speech: I couldn't finish that before the time limit.",
            1,
        ),
        encoding="utf-8",
    )
    contracts = load_model_contract_catalog(override)

    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_FailedTurns(),
                model_contracts=contracts,
            )
            await runtime.open_session("session-custom-failure", "conversation-custom-failure")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-custom-failure",
                    commit_id="commit-custom-failure",
                    call_id="call-custom-failure",
                    tool_name="work.delegate",
                    arguments=_delegation("do slow work"),
                    finalized_user_text="do slow work",
                )
            )

        terminal = updates[-1]
        assert terminal.correlation["error_code"] == "turn_timeout"
        assert terminal.title == "Operation unavailable"
        assert terminal.text == "The configured operation exceeded its time budget."
        assert terminal.frontend_response is not None
        assert terminal.frontend_response.payload_text == "I couldn't finish that before the time limit."

    asyncio.run(exercise())
