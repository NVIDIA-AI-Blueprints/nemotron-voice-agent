# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from voiceclaw.adapters.state.sqlite import SqliteStateStore
from voiceclaw.application.context import FrontendContextProjector, FrontendContextSnapshot, LocalRequestContext
from voiceclaw.application.routing import build_turn_routing_policy
from voiceclaw.application.runtime import RealtimeInteractionManager
from voiceclaw.backends import BackendComposition
from voiceclaw.domain import (
    RUNTIME_PROJECTION_SCHEMA,
    ResponseOnlyRequestState,
    ResponseOnlyResultEventKind,
    ResponseOnlySpeechSource,
    ResponseOnlyTargetAvailability,
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
from voiceclaw.interaction_profiles import (
    INTERACTION_PROFILE_SCHEMA,
    MAX_DELEGATED_GOAL_BYTES,
    load_interaction_profile_catalog,
)
from voiceclaw.model_contracts import load_model_contract_catalog
from voiceclaw.ports.runtime import (
    FrontendContextPurpose,
    FrontendContextRequest,
    FrontendConversationDeliveryState,
    FrontendConversationTurn,
    FrontendPlaybackReceipt,
    FrontendPlaybackReceiptState,
    FrontendResponsePurpose,
    FrontendSpeechDeliveryOutcome,
    FrontendSpeechDeliveryOutcomeState,
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


def _contracts_without_optional_templates(tmp_path: Path, *names: str):
    raw = yaml.safe_load(_MODEL_CONTRACTS.read_text(encoding="utf-8"))
    templates = raw["profiles"]["default"]["frontend"]["instruction_templates"]
    for name in names:
        templates.pop(name)
    override = tmp_path / "legacy-model-contracts.v1.yaml"
    override.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return load_model_contract_catalog(override)


@pytest.mark.parametrize(
    ("omitted", "expected_initial", "expected_queued"),
    [
        (
            frozenset({"work.status"}),
            ("work.delegate",),
            ("work.cancel", "work.delegate"),
        ),
        (
            frozenset({"work.cancel"}),
            ("work.delegate", "work.status"),
            ("work.delegate", "work.status"),
        ),
        (
            frozenset({"work.status", "work.cancel"}),
            ("work.delegate",),
            ("work.delegate",),
        ),
    ],
)
def test_runtime_projects_only_local_controls_present_in_interaction_profile(
    omitted: frozenset[str],
    expected_initial: tuple[str, ...],
    expected_queued: tuple[str, ...],
) -> None:
    async def exercise() -> None:
        base_profile = load_interaction_profile_catalog().resolve("stateless")
        profile = replace(
            base_profile,
            tools={name: tool for name, tool in base_profile.tools.items() if name not in omitted},
        )
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_ControlCapableTurns(),
                interaction_profile=profile,
            )
            snapshot = await runtime.open_session("session-profile-controls", "conversation-profile-controls")
            assert tuple(tool.name for tool in snapshot.frontend_tools) == expected_initial

            updates = runtime.execute_tool(
                session_id="session-profile-controls",
                commit_id="commit-profile-controls",
                call_id="call-profile-controls",
                tool_name="work.delegate",
                arguments={"goal": "inspect the repository"},
                finalized_user_text="inspect the repository",
            )
            queued = await anext(updates)
            assert queued.phase is ResponseOnlyRequestState.LOCALLY_QUEUED
            assert tuple(json.loads(runtime.projection("session-profile-controls"))["frontend_tools"]) == (
                expected_queued
            )
            assert (await _collect(updates))[-1].phase is ResponseOnlyRequestState.SUCCEEDED

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("missing_templates", "expected_initial", "expected_queued"),
    [
        (
            ("status_delivery",),
            ("work.delegate",),
            ("work.cancel", "work.delegate"),
        ),
        (
            ("control_delivery",),
            ("work.delegate", "work.status"),
            ("work.delegate", "work.status"),
        ),
        (
            ("status_delivery", "control_delivery"),
            ("work.delegate",),
            ("work.delegate",),
        ),
    ],
)
def test_runtime_hides_local_controls_without_optional_model_contract_templates(
    tmp_path: Path,
    missing_templates: tuple[str, ...],
    expected_initial: tuple[str, ...],
    expected_queued: tuple[str, ...],
) -> None:
    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_ControlCapableTurns(),
                model_contracts=_contracts_without_optional_templates(tmp_path, *missing_templates),
            )
            snapshot = await runtime.open_session("session-contract-controls", "conversation-contract-controls")
            assert tuple(tool.name for tool in snapshot.frontend_tools) == expected_initial

            updates = runtime.execute_tool(
                session_id="session-contract-controls",
                commit_id="commit-contract-controls",
                call_id="call-contract-controls",
                tool_name="work.delegate",
                arguments={"goal": "inspect the repository"},
                finalized_user_text="inspect the repository",
            )
            queued = await anext(updates)
            assert queued.phase is ResponseOnlyRequestState.LOCALLY_QUEUED
            assert tuple(json.loads(runtime.projection("session-contract-controls"))["frontend_tools"]) == (
                expected_queued
            )
            assert (await _collect(updates))[-1].phase is ResponseOnlyRequestState.SUCCEEDED

    asyncio.run(exercise())


def test_frontend_context_budget_keeps_newest_local_request() -> None:
    older = LocalRequestContext(
        summary="older request",
        operation="work.delegate",
        state=ResponseOnlyRequestState.WAITING_FOR_RESPONSE,
    )
    newest = LocalRequestContext(
        summary="newest request",
        operation="work.delegate",
        state=ResponseOnlyRequestState.WAITING_FOR_RESPONSE,
    )
    single = FrontendContextSnapshot(
        purpose=FrontendContextPurpose.CONVERSATION,
        projection_revision=2,
        active_local_requests=(newest,),
    )
    single_projection = FrontendContextProjector(maximum_characters=4096).project(single)
    bounded = FrontendContextProjector(maximum_characters=len(single_projection)).project(
        FrontendContextSnapshot(
            purpose=FrontendContextPurpose.CONVERSATION,
            projection_revision=2,
            active_local_requests=(older, newest),
        )
    )

    assert json.loads(bounded)["active_local_requests"] == [json.loads(single_projection)["active_local_requests"][0]]


def test_frontend_context_projects_volatile_correlations_additively() -> None:
    request = LocalRequestContext(
        summary="compare trees",
        operation="work.delegate",
        state=ResponseOnlyRequestState.SUCCEEDED,
        local_work_id="work-local-1",
        attempt_id="attempt-1",
        result_id="result-1",
        delivery_id="delivery-1",
    )

    projected = json.loads(
        FrontendContextProjector(maximum_characters=4096).project(
            FrontendContextSnapshot(
                purpose=FrontendContextPurpose.RESULT_DELIVERY,
                projection_revision=3,
                target_request=request,
            )
        )
    )

    assert projected["schema"] == "voiceclaw.frontend-context.v2"
    target = projected["target_request"]
    assert target["local_work_id"] == "work-local-1"
    assert target["attempt_id"] == "attempt-1"
    assert target["state"] == "succeeded"
    assert target["result"]["result_id"] == "result-1"
    assert target["result"]["delivery_id"] == "delivery-1"
    assert target["result"]["state"] == "none"
    assert target["result"]["delivery_state"] == "not_requested"


def test_frontend_context_preserves_legacy_shape_when_correlations_are_absent() -> None:
    request = LocalRequestContext(
        summary="compare trees",
        operation="work.delegate",
        state=ResponseOnlyRequestState.LOCALLY_QUEUED,
    )

    projected = json.loads(
        FrontendContextProjector(maximum_characters=4096).project(
            FrontendContextSnapshot(
                purpose=FrontendContextPurpose.DELEGATION_ACK,
                projection_revision=1,
                target_request=request,
            )
        )
    )

    assert projected["target_request"] == {
        "backend_acceptance": "unknown",
        "durability": "none",
        "operation": "work.delegate",
        "request_summary": "compare trees",
        "state": "locally_queued",
    }


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
        if not request.dispatch_marker.mark():
            raise CommittedTurnError("invocation_cancelled")
        return CommittedTurnResult(
            backend_session_id="backend-session-1",
            turn_id="turn-1",
            response_id="response-1",
            display_text=f"done: {request.text}",
            speak_text=f"Done: {request.text}.",
        )

    async def stream_turn(self, request):
        if not request.dispatch_marker.mark():
            raise CommittedTurnError("invocation_cancelled")
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


class _ControlCapableTurns(_Turns):
    async def inspect(self):
        backend = await super().inspect()
        return replace(
            backend,
            capabilities=replace(
                backend.capabilities,
                operations=frozenset(
                    {
                        BackendOperation.SUBMIT,
                        BackendOperation.STATUS,
                        BackendOperation.CANCEL,
                    }
                ),
            ),
        )


class _StaticResultTurns(_Turns):
    def __init__(self, display_text: str, speak_text: str | None = None) -> None:
        self.display_text = display_text
        self.speak_text = speak_text

    async def commit_turn(self, request):
        del request
        return CommittedTurnResult(
            backend_session_id="backend-session-static",
            turn_id="turn-static",
            response_id="response-static",
            display_text=self.display_text,
            speak_text=self.speak_text,
        )


class _OneShotTurns(_Turns):
    def __init__(self) -> None:
        self.consumed = False
        self.invocations = 0

    async def inspect(self):
        backend = await super().inspect()
        if not self.consumed:
            return backend
        return CommittedTurnBackend(
            label=backend.label,
            target_ref=backend.target_ref,
            mode=backend.mode,
            capabilities=BackendCapabilities(
                backend_kind=backend.capabilities.backend_kind,
                target_label=backend.capabilities.target_label,
                revision=backend.capabilities.revision,
                operations=frozenset(),
                durability=backend.capabilities.durability,
                event_delivery=backend.capabilities.event_delivery,
                max_parallel_work=backend.capabilities.max_parallel_work,
            ),
            capability_source=CapabilitySource.OPERATOR_CONFIGURED,
            capability_source_id="test_committed_turn",
            target_availability=ResponseOnlyTargetAvailability.CONSUMED,
        )

    async def commit_turn(self, request):
        self.invocations += 1
        self.consumed = True
        return await super().commit_turn(request)


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


def test_frontend_context_freezes_acknowledgement_and_refreshes_result_state() -> None:
    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_Turns(),
            )
            await runtime.open_session("session-context", "conversation-context")
            updates = runtime.execute_tool(
                session_id="session-context",
                commit_id="commit-context",
                call_id="call-context",
                tool_name="work.delegate",
                arguments=_delegation("compare balanced search trees"),
                finalized_user_text="Please compare balanced search trees.",
            )

            queued = await anext(updates)
            assert queued.phase is ResponseOnlyRequestState.LOCALLY_QUEUED
            acknowledgement_request = FrontendContextRequest(
                FrontendContextPurpose.DELEGATION_ACK,
                "commit-context",
            )
            frozen_before = runtime.frontend_context(
                "session-context",
                acknowledgement_request,
                maximum_characters=4096,
            )
            remaining_updates = await _collect(updates)
            result_response = next(
                update.frontend_response
                for update in remaining_updates
                if update.frontend_response is not None
                and update.frontend_response.purpose is FrontendResponsePurpose.RESULT_DELIVERY
            )
            assert result_response.presentation_id is not None
            frozen_after = runtime.frontend_context(
                "session-context",
                acknowledgement_request,
                maximum_characters=4096,
            )
            assert frozen_after == frozen_before
            frozen = json.loads(frozen_before)
            assert frozen["purpose"] == "delegation_ack"
            target_request = frozen["target_request"]
            assert target_request.pop("attempt_id").startswith("attempt_vc_")
            assert target_request.pop("local_work_id").startswith("work_vc_")
            assert target_request == {
                "backend_acceptance": "unknown",
                "durability": "none",
                "operation": "work.delegate",
                "request_summary": "compare balanced search trees",
                "state": "locally_queued",
            }
            assert "local_request_id" not in frozen_before
            assert "voice_activity" not in frozen

            result = json.loads(
                runtime.frontend_context(
                    "session-context",
                    FrontendContextRequest(FrontendContextPurpose.RESULT_DELIVERY, "commit-context"),
                    maximum_characters=4096,
                )
            )
            assert result["purpose"] == "result_delivery"
            assert result["target_request"]["state"] == "succeeded"
            assert result["target_request"]["terminal_outcome"] == "succeeded"
            result_evidence = result["target_request"]["result"]
            assert result_evidence.pop("delivery_id").startswith("delivery_vc_")
            assert result_evidence.pop("result_id").startswith("result_vc_")
            assert result_evidence == {
                "delivery_state": "ready",
                "display_available": True,
                "speech_source": "backend_authored",
                "state": "available",
            }
            assert result["voice_activity"] == {
                "connected": True,
                "input": "idle",
                "model": "idle",
                "output": "idle",
            }

            conversation_before = json.loads(
                runtime.frontend_context(
                    "session-context",
                    FrontendContextRequest(FrontendContextPurpose.CONVERSATION),
                    maximum_characters=4096,
                )
            )
            assert len(conversation_before["ready_results"]) == 1
            runtime.record_conversation_turn(
                "session-context",
                FrontendConversationTurn(
                    turn_id="assistant-acknowledgement",
                    role="assistant",
                    text="I am checking that comparison.",
                    delivery_state=FrontendConversationDeliveryState.DELIVERED,
                    presentation_id="presentation-acknowledgement",
                    local_request_id="commit-context",
                ),
            )
            conversation_after_acknowledgement = json.loads(
                runtime.frontend_context(
                    "session-context",
                    FrontendContextRequest(FrontendContextPurpose.CONVERSATION),
                    maximum_characters=4096,
                )
            )
            assert len(conversation_after_acknowledgement["ready_results"]) == 1
            runtime.record_conversation_turn(
                "session-context",
                FrontendConversationTurn(
                    turn_id="assistant-result",
                    role="assistant",
                    text="The comparison is ready.",
                    delivery_state=FrontendConversationDeliveryState.DELIVERED,
                    presentation_id=result_response.presentation_id,
                    local_request_id="commit-context",
                ),
            )
            conversation_after = json.loads(
                runtime.frontend_context(
                    "session-context",
                    FrontendContextRequest(FrontendContextPurpose.CONVERSATION),
                    maximum_characters=4096,
                )
            )
            assert conversation_after["ready_results"] == []
            assert conversation_after["recent_conversation"][-1] == {
                "delivery_state": "delivered",
                "role": "assistant",
                "text": "The comparison is ready.",
            }
            result_after_generation = json.loads(
                runtime.frontend_context(
                    "session-context",
                    FrontendContextRequest(FrontendContextPurpose.RESULT_DELIVERY, "commit-context"),
                    maximum_characters=4096,
                )
            )
            assert result_after_generation["target_request"]["result"]["delivery_state"] == (
                "awaiting_playback_receipt"
            )

    asyncio.run(exercise())


def test_local_status_reads_active_request_without_calling_backend_again() -> None:
    class _BlockingTurns(_Turns):
        def __init__(self) -> None:
            self.calls = 0
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def commit_turn(self, request):
            self.calls += 1
            self.started.set()
            await self.release.wait()
            return await super().commit_turn(request)

    async def exercise() -> None:
        turns = _BlockingTurns()
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(backend_profile="default", state_store=store, committed_turns=turns)
            await runtime.open_session("session-status", "conversation-status")
            delegated = runtime.execute_tool(
                session_id="session-status",
                commit_id="commit-status-target",
                call_id="call-status-target",
                tool_name="work.delegate",
                arguments=_delegation("inspect the repository"),
                finalized_user_text="inspect the repository",
            )
            queued = await anext(delegated)
            work_id = str(queued.tool_output["work_id"])
            assert (await anext(delegated)).phase is ResponseOnlyRequestState.DISPATCHING
            assert (await anext(delegated)).phase is ResponseOnlyRequestState.WAITING_FOR_RESPONSE
            delegated_remainder = asyncio.create_task(_collect(delegated))
            await turns.started.wait()

            status = await _collect(
                runtime.execute_tool(
                    session_id="session-status",
                    commit_id="commit-status-read",
                    call_id="call-status-read",
                    tool_name="work.status",
                    arguments={"work_id": work_id},
                    finalized_user_text="What is the status?",
                )
            )
            assert len(status) == 1
            assert status[0].phase is ResponseOnlyRequestState.WAITING_FOR_RESPONSE
            assert status[0].tool_output == {
                "status": "waiting_for_response",
                "request_state": "waiting_for_response",
                "local_request_id": "commit-status-target",
                "work_id": work_id,
                "identity_authority": "voiceclaw_volatile",
                "terminal_outcome": "none",
                "result_state": "none",
            }
            assert status[0].frontend_response is not None
            assert status[0].frontend_response.purpose is FrontendResponsePurpose.STATUS_DELIVERY
            assert turns.calls == 1

            turns.release.set()
            assert (await delegated_remainder)[-1].phase is ResponseOnlyRequestState.SUCCEEDED

    asyncio.run(exercise())


def test_local_cancel_prevents_a_queued_request_from_reaching_backend() -> None:
    class _SerialTurns(_Turns):
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()

        async def commit_turn(self, request):
            self.calls.append(request.commit_id)
            if request.commit_id == "commit-running":
                self.first_started.set()
                await self.release_first.wait()
            return await super().commit_turn(request)

    async def exercise() -> None:
        turns = _SerialTurns()
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(backend_profile="default", state_store=store, committed_turns=turns)
            await runtime.open_session("session-running", "conversation-running")
            await runtime.open_session("session-queued", "conversation-queued")
            running = runtime.execute_tool(
                session_id="session-running",
                commit_id="commit-running",
                call_id="call-running",
                tool_name="work.delegate",
                arguments=_delegation("hold the execution lane"),
                finalized_user_text="hold the execution lane",
            )
            for _ in range(3):
                await anext(running)
            running_remainder = asyncio.create_task(_collect(running))
            await turns.first_started.wait()

            queued = runtime.execute_tool(
                session_id="session-queued",
                commit_id="commit-queued",
                call_id="call-queued",
                tool_name="work.delegate",
                arguments=_delegation("never dispatch this request"),
                finalized_user_text="never dispatch this request",
            )
            queued_receipt = await anext(queued)
            work_id = str(queued_receipt.tool_output["work_id"])
            queued_remainder = asyncio.create_task(_collect(queued))

            cancelled = await _collect(
                runtime.execute_tool(
                    session_id="session-queued",
                    commit_id="commit-cancel-operation",
                    call_id="call-cancel-operation",
                    tool_name="work.cancel",
                    arguments={"work_id": work_id},
                    finalized_user_text="Cancel that request.",
                )
            )
            assert len(cancelled) == 1
            assert cancelled[0].phase is ResponseOnlyRequestState.CANCELLED
            assert cancelled[0].frontend_response is not None
            assert cancelled[0].frontend_response.purpose is FrontendResponsePurpose.CONTROL_DELIVERY
            cancelled_target = cancelled[0].backend_target
            assert cancelled_target is not None
            assert tuple(tool.name for tool in cancelled_target.frontend_tools) == (
                "work.delegate",
                "work.status",
            )
            assert cancelled_target.local_control_tools == ("work.status",)
            assert await asyncio.wait_for(queued_remainder, timeout=1) == []
            assert turns.calls == ["commit-running"]

            turns.release_first.set()
            await running_remainder

    asyncio.run(exercise())


def test_local_cancel_after_dispatch_still_fences_the_unentered_backend_port() -> None:
    class _ObservedTurns(_Turns):
        def __init__(self) -> None:
            self.invocations = 0

        async def stream_turn(self, request):
            if not request.dispatch_marker.mark():
                raise CommittedTurnError("invocation_cancelled")
            self.invocations += 1
            async for event in super().stream_turn(request):
                yield event

    async def exercise() -> None:
        turns = _ObservedTurns()
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=turns,
            )
            await runtime.open_session("session-dispatch-cancel", "conversation-dispatch-cancel")
            delegated = runtime.execute_tool(
                session_id="session-dispatch-cancel",
                commit_id="commit-dispatch-cancel",
                call_id="call-dispatch-cancel",
                tool_name="work.delegate",
                arguments=_delegation("cancel after local dispatch"),
                finalized_user_text="cancel after local dispatch",
            )
            queued = await anext(delegated)
            assert (await anext(delegated)).phase is ResponseOnlyRequestState.DISPATCHING
            projection = json.loads(runtime.projection("session-dispatch-cancel"))
            assert projection["execution"]["request_state"] == "dispatching"
            assert "work.cancel" in projection["frontend_tools"]

            cancelled = await _collect(
                runtime.execute_tool(
                    session_id="session-dispatch-cancel",
                    commit_id="commit-dispatch-cancel-operation",
                    call_id="call-dispatch-cancel-operation",
                    tool_name="work.cancel",
                    arguments={"work_id": queued.tool_output["work_id"]},
                    finalized_user_text="Cancel it.",
                )
            )
            assert cancelled[0].phase is ResponseOnlyRequestState.CANCELLED
            assert await _collect(delegated) == []
            assert turns.invocations == 0
            projection = json.loads(runtime.projection("session-dispatch-cancel"))
            assert projection["execution"]["request_state"] == "cancelled"
            assert projection["execution"]["latest_terminal_outcome"] == "cancelled"

    asyncio.run(exercise())


def test_session_close_after_dispatch_cancels_before_backend_port_entry() -> None:
    class _ObservedTurns(_Turns):
        def __init__(self) -> None:
            self.invocations = 0

        async def stream_turn(self, request):
            self.invocations += 1
            async for event in super().stream_turn(request):
                yield event

    async def exercise() -> None:
        turns = _ObservedTurns()
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=turns,
            )
            await runtime.open_session("session-dispatch-close", "conversation-dispatch-close")
            delegated = runtime.execute_tool(
                session_id="session-dispatch-close",
                commit_id="commit-dispatch-close",
                call_id="call-dispatch-close",
                tool_name="work.delegate",
                arguments=_delegation("close after local dispatch"),
                finalized_user_text="close after local dispatch",
            )
            assert (await anext(delegated)).phase is ResponseOnlyRequestState.LOCALLY_QUEUED
            assert (await anext(delegated)).phase is ResponseOnlyRequestState.DISPATCHING

            await runtime.close_session("session-dispatch-close", "client_disconnected")
            projection = json.loads(runtime.projection("session-dispatch-close"))
            assert projection["execution"]["request_state"] == "cancelled"
            assert projection["execution"]["latest_terminal_outcome"] == "cancelled"
            assert projection["execution"]["latest_failure_code"] is None
            assert turns.invocations == 0

            await delegated.aclose()
            with pytest.raises(LookupError):
                runtime.projection("session-dispatch-close")

    asyncio.run(exercise())


def test_cancellation_during_the_port_entry_handshake_remains_pre_effect() -> None:
    class _ObservedTurns(_Turns):
        def __init__(self) -> None:
            self.invocations = 0
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def stream_turn(self, request):
            self.entered.set()
            await self.release.wait()
            if not request.dispatch_marker.mark():
                raise CommittedTurnError("invocation_cancelled")
            self.invocations += 1
            async for event in super().stream_turn(request):
                yield event

    async def exercise() -> None:
        turns = _ObservedTurns()
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=turns,
            )
            await runtime.open_session("session-entry-race", "conversation-entry-race")
            delegated = runtime.execute_tool(
                session_id="session-entry-race",
                commit_id="commit-entry-race",
                call_id="call-entry-race",
                tool_name="work.delegate",
                arguments=_delegation("cancel during port entry"),
                finalized_user_text="cancel during port entry",
            )
            assert (await anext(delegated)).phase is ResponseOnlyRequestState.LOCALLY_QUEUED
            assert (await anext(delegated)).phase is ResponseOnlyRequestState.DISPATCHING

            waiting = asyncio.create_task(anext(delegated))
            await turns.entered.wait()
            assert waiting.done() is False
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting

            projection = json.loads(runtime.projection("session-entry-race"))
            assert projection["execution"]["request_state"] == "cancelled"
            assert projection["execution"]["latest_terminal_outcome"] == "cancelled"
            assert projection["execution"]["latest_failure_code"] is None
            assert turns.invocations == 0

    asyncio.run(exercise())


def test_runtime_records_zero_audio_result_as_skipped_without_a_playback_receipt() -> None:
    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_Turns(),
            )
            await runtime.open_session("session-skipped", "conversation-skipped")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-skipped",
                    commit_id="commit-skipped",
                    call_id="call-skipped",
                    tool_name="work.delegate",
                    arguments=_delegation("inspect the result"),
                    finalized_user_text="inspect the result",
                )
            )
            result_response = next(
                update.frontend_response
                for update in updates
                if update.frontend_response is not None
                and update.frontend_response.purpose is FrontendResponsePurpose.RESULT_DELIVERY
            )
            assert result_response.presentation_id is not None
            with pytest.raises(ValueError, match="different result presentation"):
                runtime.record_speech_delivery_outcome(
                    "session-skipped",
                    FrontendSpeechDeliveryOutcome(
                        local_request_id="commit-skipped",
                        presentation_id="presentation-acknowledgement",
                        state=FrontendSpeechDeliveryOutcomeState.SKIPPED,
                        reason_code="frontend_emitted_no_audio",
                    ),
                )
            runtime.record_speech_delivery_outcome(
                "session-skipped",
                FrontendSpeechDeliveryOutcome(
                    local_request_id="commit-skipped",
                    presentation_id=result_response.presentation_id,
                    state=FrontendSpeechDeliveryOutcomeState.SKIPPED,
                    reason_code="frontend_emitted_no_audio",
                ),
            )

            context = json.loads(
                runtime.frontend_context(
                    "session-skipped",
                    FrontendContextRequest(FrontendContextPurpose.RESULT_DELIVERY, "commit-skipped"),
                    maximum_characters=4096,
                )
            )
            assert context["target_request"]["result"]["delivery_state"] == "skipped"
            assert context["ready_results"] == []
            projection = json.loads(runtime.projection("session-skipped"))
            assert projection["ready_results"][0]["delivery_state"] == "skipped"
            assert projection["delivery"]["authoritative_client_receipt"] is False

    asyncio.run(exercise())


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


def test_agent_routing_rejects_when_delegation_is_unavailable() -> None:
    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="unavailable",
                state_store=store,
                committed_turns=None,
                turn_routing_policy=build_turn_routing_policy("agent"),
            )
            await runtime.open_session("session-agent", "conversation-agent")
            directive = runtime.route_finalized_turn("session-agent", "Hello")
            assert directive.kind is TurnDirectiveKind.REJECT
            assert directive.reason_code == "operation_unavailable"

    asyncio.run(exercise())


def test_runtime_withdraws_delegation_after_a_one_shot_target_is_consumed() -> None:
    async def exercise() -> None:
        turns = _OneShotTurns()
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=turns,
            )
            await runtime.open_session("session-one-shot", "conversation-one-shot")
            first = await _collect(
                runtime.execute_tool(
                    session_id="session-one-shot",
                    commit_id="commit-one",
                    call_id="call-one",
                    tool_name="work.delegate",
                    arguments=_delegation("first request"),
                    finalized_user_text="first request",
                )
            )
            assert first[-1].phase is ResponseOnlyRequestState.SUCCEEDED
            assert first[-1].frontend_tools is not None
            assert tuple(tool.name for tool in first[-1].frontend_tools) == ("work.status",)
            assert first[-1].backend_target is not None
            assert first[-1].backend_target.target_binding_verified is True
            assert first[-1].backend_target.target_availability is ResponseOnlyTargetAvailability.CONSUMED
            assert first[-1].backend_target.capabilities == ()
            assert "capabilities" not in first[-1].correlation
            assert "frontend_tools" not in first[-1].correlation

            projection = json.loads(runtime.projection("session-one-shot"))
            assert projection["connection"]["target_state"] == ResponseOnlyTargetAvailability.CONSUMED
            assert projection["contract"]["operations"] == []
            assert projection["contract"]["frontend_tools"] == ["work.status"]

            second = await _collect(
                runtime.execute_tool(
                    session_id="session-one-shot",
                    commit_id="commit-two",
                    call_id="call-two",
                    tool_name="work.delegate",
                    arguments=_delegation("second request"),
                    finalized_user_text="second request",
                )
            )
            assert len(second) == 1
            assert second[0].phase is ResponseOnlyRequestState.FAILED
            assert second[0].tool_output == {"status": "failed", "error": "capability_unsupported"}
            assert second[0].frontend_response is not None
            assert second[0].frontend_response.purpose is FrontendResponsePurpose.FAILURE_DELIVERY
            assert turns.invocations == 1

    asyncio.run(exercise())


def test_runtime_atomically_marks_the_target_unavailable_when_refresh_fails() -> None:
    class _RefreshFailureTurns(_Turns):
        def __init__(self) -> None:
            self.inspections = 0

        async def inspect(self):
            self.inspections += 1
            if self.inspections > 1:
                raise OSError("private backend failure")
            return await super().inspect()

    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_RefreshFailureTurns(),
            )
            await runtime.open_session("session-refresh", "conversation-refresh")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-refresh",
                    commit_id="commit-refresh",
                    call_id="call-refresh",
                    tool_name="work.delegate",
                    arguments=_delegation("run one request"),
                    finalized_user_text="run one request",
                )
            )

            target = updates[-1].backend_target
            assert target is not None
            assert target.target_binding_verified is False
            assert target.backend_label == "Backend unavailable"
            assert tuple(tool.name for tool in target.frontend_tools) == ("work.status",)
            assert target.local_control_tools == ("work.status",)
            projection = json.loads(runtime.projection("session-refresh"))
            assert projection["connection"]["target_binding_state"] == "unavailable"
            assert projection["contract"]["operations"] == []

    asyncio.run(exercise())


def test_runtime_persists_mapping_and_labels_ephemeral_turn_non_durable() -> None:
    class _InspectableTurns(_Turns):
        def __init__(self) -> None:
            self.availability = ResponseOnlyTargetAvailability.AVAILABLE

        async def inspect(self):
            return replace(
                await super().inspect(),
                target_availability=self.availability,
            )

        async def stream_turn(self, request):
            if not request.dispatch_marker.mark():
                raise CommittedTurnError("invocation_cancelled")
            self.availability = ResponseOnlyTargetAvailability.IN_FLIGHT
            try:
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
            finally:
                self.availability = ResponseOnlyTargetAvailability.AVAILABLE

    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_InspectableTurns(),
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
            queued_output = dict(updates[0].tool_output or {})
            work_id = queued_output.pop("work_id")
            assert work_id.startswith("work_vc_")
            assert queued_output == {
                "status": "locally_queued",
                "request_state": "locally_queued",
                "local_request_id": "commit-1",
                "identity_authority": "voiceclaw_volatile",
                "evidence": "voiceclaw_volatile",
                "backend_acceptance": "unknown",
                "durability": "none",
            }
            assert updates[0].frontend_response is not None
            assert updates[0].frontend_response.purpose is FrontendResponsePurpose.DELEGATION_ACK
            assert updates[0].frontend_response.local_request_id == "commit-1"
            assert updates[0].frontend_response.payload_text is None
            assert updates[0].request_summary == "build it"
            assert "request_summary" not in updates[0].correlation
            queued_target = updates[0].backend_target
            assert queued_target is not None
            assert tuple(tool.name for tool in queued_target.frontend_tools) == (
                "work.cancel",
                "work.delegate",
                "work.status",
            )
            assert queued_target.local_control_tools == ("work.cancel", "work.status")
            waiting = next(
                update
                for update in updates
                if update.kind is ResponseOnlyUpdateKind.BACKEND_TURN
                and update.phase is ResponseOnlyRequestState.WAITING_FOR_RESPONSE
            )
            waiting_target = waiting.backend_target
            assert waiting_target is not None
            assert waiting_target.target_availability is ResponseOnlyTargetAvailability.IN_FLIGHT
            assert tuple(tool.name for tool in waiting_target.frontend_tools) == (
                "work.delegate",
                "work.status",
            )
            assert waiting_target.local_control_tools == ("work.status",)
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
            assert projection["target_binding_verified"] is True
            assert projection["connection"] == {
                "attachment_mode": "terminal_invocation",
                "attachment_state": "not_persistent",
                "backend_context_continuity": "unqualified",
                "frontend_state": "connected",
                "target_binding_state": "verified",
                "target_state": "available",
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
                "frontend_tools": ["work.delegate", "work.status"],
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
                "attempt_id": projection["execution"]["attempt_id"],
                "any_request_active": False,
                "durable_work_id_issued": False,
                "identity_authority": "voiceclaw_volatile",
                "latest_failure_code": None,
                "latest_terminal_outcome": "succeeded",
                "local_request_id": "commit-1",
                "local_work_id": work_id,
                "operation": "work.delegate",
                "request_active": False,
                "request_state": "succeeded",
                "request_summary": "build it",
            }
            assert len(projection["local_requests"]) == 1
            local_request = projection["local_requests"][0]
            assert local_request["local_request_id"] == "commit-1"
            assert local_request["local_work_id"] == work_id
            assert local_request["attempt_id"].startswith("attempt_vc_")
            assert local_request["identity_authority"] == "voiceclaw_volatile"
            assert local_request["request_state"] == "succeeded"
            assert local_request["terminal_outcome"] == "succeeded"
            assert local_request["request_summary"] == "build it"
            assert local_request["result"]["state"] == "available"
            assert local_request["result"]["result_id"].startswith("result_vc_")
            assert local_request["result"]["delivery_id"].startswith("delivery_vc_")
            assert projection["active_work"] == []
            assert projection["active_local_requests"] == []
            assert projection["pending_interactions"] == []
            assert len(projection["ready_results"]) == 1
            ready_result = projection["ready_results"][0]
            assert ready_result["local_request_id"] == "commit-1"
            assert ready_result["local_work_id"] == work_id
            assert ready_result["identity_authority"] == "voiceclaw_volatile"
            assert ready_result["delivery_state"] == "ready"
            assert ready_result["state"] == "available"
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


def test_runtime_keeps_the_delegated_goal_out_of_the_acknowledgement_payload() -> None:
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
            assert acknowledgement.payload_text is None
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


@pytest.mark.parametrize("mode", ["model", "agent"])
def test_configured_agent_mode_preserves_the_finalized_request_instead_of_a_model_rewrite(mode: str) -> None:
    class _RecordingTurns(_Turns):
        committed_text = None

        async def commit_turn(self, request):
            self.committed_text = request.text
            return await super().commit_turn(request)

    async def exercise() -> None:
        source_turn = '  Now convert that same word to hexadecimal. Preserve "quotes"\nand Unicode ☃.  '
        proposed_goal = "Convert the word 'that' to hexadecimal."
        turns = _RecordingTurns()
        with SqliteStateStore(":memory:") as store:
            runtime = BackendComposition(turns, "configured").create_runtime(
                backend_profile="default", state_store=store, turn_routing_mode=mode
            )
            await runtime.open_session("session-original", "conversation-original")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-original",
                    commit_id="commit-original",
                    call_id="call-original",
                    tool_name="work.delegate",
                    arguments=_delegation(proposed_goal),
                    finalized_user_text=source_turn,
                )
            )
            assert turns.committed_text == (source_turn.strip() if mode == "agent" else proposed_goal)
            assert updates[-1].phase is ResponseOnlyRequestState.SUCCEEDED

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
            queued_output = dict(updates[0].tool_output or {})
            assert queued_output == {
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
            queued_output = dict(updates[0].tool_output or {})
            assert queued_output == {
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
@pytest.mark.parametrize("delegate_original_text", [False, True])
def test_runtime_rejects_invalid_finalized_source_turns_before_backend_dispatch(
    source_turn: str, delegate_original_text: bool
) -> None:
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
                delegate_original_text=delegate_original_text,
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


def test_runtime_rejects_concurrent_duplicate_live_session_ids() -> None:
    class _ConcurrentInspectTurns(_Turns):
        def __init__(self) -> None:
            self.inspections = 0
            self.both_started = asyncio.Event()
            self.release = asyncio.Event()

        async def inspect(self):
            self.inspections += 1
            if self.inspections == 2:
                self.both_started.set()
            await self.release.wait()
            return await super().inspect()

    async def exercise() -> None:
        turns = _ConcurrentInspectTurns()
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=turns,
            )
            first = asyncio.create_task(runtime.open_session("session-duplicate", "conversation-first"))
            second = asyncio.create_task(runtime.open_session("session-duplicate", "conversation-second"))
            await asyncio.wait_for(turns.both_started.wait(), timeout=1)
            turns.release.set()
            outcomes = await asyncio.gather(first, second, return_exceptions=True)

            assert sum(isinstance(outcome, ValueError) for outcome in outcomes) == 1
            error = next(outcome for outcome in outcomes if isinstance(outcome, ValueError))
            assert str(error) == "realtime session_id is already open"
            assert runtime.projection("session-duplicate")

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
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-receipt-only",
                    commit_id="commit-receipt-only",
                    call_id="call-receipt-only",
                    tool_name="work.delegate",
                    arguments=_delegation("find the file"),
                    finalized_user_text="find the file",
                )
            )
            result_response = next(
                update.frontend_response
                for update in updates
                if update.frontend_response is not None
                and update.frontend_response.purpose is FrontendResponsePurpose.RESULT_DELIVERY
            )
            assert result_response.presentation_id is not None

            runtime.record_playback_receipt(
                "session-receipt-only",
                FrontendPlaybackReceipt(
                    turn_id="item-without-transcript",
                    state=FrontendPlaybackReceiptState.INTERRUPTED,
                    presentation_id=result_response.presentation_id,
                    local_request_id="commit-receipt-only",
                    heard_through_ms=250,
                    audio_end_ms=900,
                ),
            )

            projection = json.loads(runtime.projection("session-receipt-only"))
            assert projection["recent_conversation"] == []
            assert projection["heard_through_presentation_id"] == result_response.presentation_id
            assert projection["heard_through_turn_id"] == "item-without-transcript"
            assert len(projection["ready_results"]) == 1
            ready_result = projection["ready_results"][0]
            assert ready_result["delivery_state"] == "interrupted"
            assert ready_result["display_available"] is True
            assert ready_result["identity_authority"] == "voiceclaw_volatile"
            assert ready_result["local_request_id"] == "commit-receipt-only"
            assert ready_result["local_work_id"].startswith("work_vc_")
            assert ready_result["result_id"].startswith("result_vc_")
            assert ready_result["delivery_id"].startswith("delivery_vc_")
            assert ready_result["speech_source"] == "backend_authored"
            assert ready_result["state"] == "available"
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
            queued_output = dict(queued.tool_output or {})
            work_id = queued_output.pop("work_id")
            assert work_id.startswith("work_vc_")
            assert queued_output == {
                "status": "locally_queued",
                "request_state": "locally_queued",
                "local_request_id": "commit-active",
                "identity_authority": "voiceclaw_volatile",
                "evidence": "voiceclaw_volatile",
                "backend_acceptance": "unknown",
                "durability": "none",
            }
            assert queued.frontend_response is not None
            assert queued.frontend_response.purpose is FrontendResponsePurpose.DELEGATION_ACK
            assert queued.frontend_response.local_request_id == "commit-active"
            assert queued.frontend_response.payload_text is None
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
            assert projection["execution"]["identity_authority"] == "voiceclaw_volatile"
            assert projection["execution"]["local_work_id"] == work_id
            assert projection["execution"]["attempt_id"].startswith("attempt_vc_")
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


def test_runtime_queues_a_second_request_without_overwriting_the_active_one() -> None:
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

            second = runtime.execute_tool(
                session_id="session-busy",
                commit_id="commit-busy",
                call_id="call-busy",
                tool_name="work.delegate",
                arguments=_delegation("second request"),
                finalized_user_text="second request",
            )
            second_queued = await anext(second)
            assert second_queued.phase is ResponseOnlyRequestState.LOCALLY_QUEUED
            second_remainder = asyncio.create_task(_collect(second))
            projection = json.loads(runtime.projection("session-busy"))
            assert projection["execution"]["local_request_id"] == "commit-busy"
            assert projection["execution"]["request_state"] == "locally_queued"
            assert projection["execution"]["request_active"] is True
            assert projection["execution"]["any_request_active"] is True
            requests = {request["local_request_id"]: request for request in projection["local_requests"]}
            assert requests["commit-active"]["request_state"] == "waiting_for_response"
            assert requests["commit-busy"]["request_state"] == "locally_queued"

            turns.release.set()
            first_remainder = await remainder
            assert _display_text(first_remainder) == "done: first request"
            assert first_remainder[-1].phase == "succeeded"
            second_updates = await asyncio.wait_for(second_remainder, timeout=1)
            assert _request_phases(second_updates) == [
                ResponseOnlyRequestState.DISPATCHING,
                ResponseOnlyRequestState.WAITING_FOR_RESPONSE,
                ResponseOnlyRequestState.SUCCEEDED,
            ]

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
            assert abandoned_projection["execution"]["request_state"] == "outcome_unknown"
            assert abandoned_projection["execution"]["latest_failure_code"] == "runtime_interrupted"
            assert abandoned_projection["execution"]["latest_terminal_outcome"] == "outcome_unknown"
            assert abandoned_projection["local_requests"][0]["request_state"] == "outcome_unknown"

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


@pytest.mark.parametrize("updates_before_close", [1, 2, 3])
def test_abandoned_preinvoke_request_has_a_known_local_outcome(updates_before_close: int) -> None:
    class _ObservedTurns(_Turns):
        def __init__(self) -> None:
            self.invocations = 0

        async def stream_turn(self, request):
            self.invocations += 1
            async for event in super().stream_turn(request):
                yield event

    async def exercise() -> None:
        turns = _ObservedTurns()
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=turns,
            )
            await runtime.open_session("session-local-cancel", "conversation-local-cancel")
            updates = runtime.execute_tool(
                session_id="session-local-cancel",
                commit_id="commit-local-cancel",
                call_id="call-local-cancel",
                tool_name="work.delegate",
                arguments=_delegation("a request cancelled before invocation"),
                finalized_user_text="a request cancelled before invocation",
            )
            observed = [await anext(updates) for _ in range(updates_before_close)]
            assert [update.phase for update in observed] == [
                "locally_queued",
                "dispatching",
                "waiting_for_response",
            ][:updates_before_close]
            await updates.aclose()

            projection = json.loads(runtime.projection("session-local-cancel"))
            if updates_before_close <= 2:
                assert projection["execution"]["request_state"] == "cancelled"
                assert projection["execution"]["latest_terminal_outcome"] == "cancelled"
                assert projection["execution"]["latest_failure_code"] is None
                assert turns.invocations == 0
            else:
                assert projection["execution"]["request_state"] == "outcome_unknown"
                assert projection["execution"]["latest_terminal_outcome"] == "outcome_unknown"
                assert projection["execution"]["latest_failure_code"] == "runtime_interrupted"
                assert turns.invocations == 1

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

            assert snapshot.target_binding_verified is False
            assert snapshot.backend_label == "Backend unavailable"
            assert store.get_session("session-offline") is not None
            projection = json.loads(runtime.projection("session-offline"))
            assert projection["target_binding_verified"] is False
            assert projection["connection"]["target_binding_state"] == "unavailable"
            assert projection["connection"]["attachment_state"] == "unavailable"
            assert projection["contract"]["operations"] == []
            assert projection["contract"]["frontend_tools"] == []

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("display_text", "key_fragments"),
    [
        pytest.param(
            "```python\n" + ("print('display only')\n" * 40) + "```",
            ("print('display only')",),
            id="fenced-code",
        ),
        pytest.param(
            "Implementation:\n```python\nprint('never speak this')",
            ("Implementation:", "print('never speak this')"),
            id="unclosed-code",
        ),
        pytest.param(
            "There are 2 entries.\n\n- AGENTS.md\n- USER.md",
            ("- AGENTS.md", "- USER.md"),
            id="markdown-list",
        ),
        pytest.param(
            "First approved sentence. Second approved sentence. Third display-only sentence.",
            ("First approved sentence.", "Third display-only sentence."),
            id="multi-sentence-document",
        ),
        pytest.param(
            " ".join("word" for _ in range(50)),
            ("word word word",),
            id="bounded-document",
        ),
        pytest.param(
            " ".join(f"fact-{index:04d}." for index in range(600)),
            ("fact-0000.", "fact-0599."),
            id="large-document",
        ),
        pytest.param(
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
            "Do you want implementation details?",
            (
                "https://example.com/details",
                "/tmp/tree.py",
                "Would you like a walkthrough?",
                "Do you want implementation details?",
            ),
            id="markdown-table-links-and-paths",
        ),
        pytest.param(
            "```python\nprint('display only')\n```\n\n"
            "**Search complexity:**\n\n"
            "- **Average case: O(log n)** when the tree is reasonably balanced.\n"
            "- **Worst case: O(n)** when the tree degenerates into a linked list.\n"
            "- Use a self-balancing tree to preserve logarithmic search.\n"
            "- A fourth detail remains available only in the display.",
            ("print('display only')", "A fourth detail remains available only in the display."),
            id="code-and-bullets",
        ),
        pytest.param(
            "# Search comparison\n\n"
            "## Breadth-First Search (BFS)\n\n"
            "How it works: It explores nodes level by level.\n\n"
            "**Practical use case:** **Web crawlers** – Web crawlers discover nearby pages first.\n\n"
            "## Depth-First Search (DFS)\n\n"
            "How it works: It explores one branch before backtracking.\n\n"
            "**Practical use case:** **Maze solving** – Maze solving explores one corridor before backtracking.",
            ("Breadth-First Search (BFS)", "Web crawlers", "Depth-First Search (DFS)", "Maze solving"),
            id="markdown-sections",
        ),
        pytest.param(
            "Breadth-First Search (BFS) vs. Depth-First Search (DFS)\n\n"
            "| Aspect | Breadth-First Search (BFS) | Depth-First Search (DFS) |\n"
            "|---|---|---|\n"
            "| **Traversal order** | Level by level | One branch before backtracking |\n"
            "| **Data structure** | Queue (FIFO) | Stack (LIFO) or recursion |\n\n"
            "### Practical Use Cases\n\n"
            "#### BFS: **Web crawlers / shortest path finding**\n"
            "Additional display detail for BFS.\n\n"
            "#### DFS: **Maze solving / dependency resolution**\n"
            "Additional display detail for DFS.",
            (
                "Traversal order",
                "Queue (FIFO)",
                "Web crawlers / shortest path finding",
                "Maze solving / dependency resolution",
            ),
            id="markdown-table-and-sections",
        ),
        pytest.param(
            "Workspace entries:\n\n- AGENTS.md\n- USER.md",
            ("Workspace entries:", "- AGENTS.md", "- USER.md"),
            id="heading-and-list",
        ),
    ],
)
def test_runtime_keeps_display_only_results_out_of_speech(
    display_text: str,
    key_fragments: tuple[str, ...],
) -> None:
    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_StaticResultTurns(display_text),
            )
            await runtime.open_session("session-display-only", "conversation-display-only")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-display-only",
                    commit_id="commit-display-only",
                    call_id="call-display-only",
                    tool_name="work.delegate",
                    arguments=_delegation("render the requested document"),
                    finalized_user_text="render the requested document",
                )
            )

            streamed_display = _display_text(updates)
            terminal = updates[-1]
            assert streamed_display == display_text
            assert all(fragment in streamed_display for fragment in key_fragments)
            assert terminal.text == ""
            assert terminal.frontend_response is None
            assert terminal.correlation["speech_source"] == ResponseOnlySpeechSource.NONE.value
            projection = json.loads(runtime.projection("session-display-only"))
            assert projection["latest_result"]["speech_source"] == ResponseOnlySpeechSource.NONE.value
            assert projection["latest_result"]["display_available"] is True
            assert projection["latest_result"]["display_body_in_context"] is False
            assert projection["latest_result"]["provider_identifiers_in_context"] is False

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
                ResponseOnlyRequestState.OUTCOME_UNKNOWN,
            ]
            assert [update.phase for update in _display_updates(updates)] == [
                ResponseOnlyResultEventKind.DISPLAY_DELTA,
            ]
            assert _display_text(updates) == " \n\t "
            terminal = updates[-1]
            expected = load_model_contract_catalog().failure_copy("invocation_result_unusable")
            assert terminal.text == expected.display
            assert terminal.correlation["error_code"] == "invocation_result_unusable"
            assert terminal.tool_output is None
            assert terminal.frontend_response is not None
            assert terminal.frontend_response.purpose is FrontendResponsePurpose.FAILURE_DELIVERY
            assert terminal.frontend_response.payload_text == expected.speech
            projection = json.loads(runtime.projection("session-empty"))
            assert projection["execution"]["latest_terminal_outcome"] == "outcome_unknown"
            assert projection["execution"]["latest_failure_code"] == "invocation_result_unusable"
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
                ResponseOnlyRequestState.OUTCOME_UNKNOWN,
            ]
            assert _display_text(updates) == provisional_display
            assert not any(
                update.phase is ResponseOnlyResultEventKind.COMPLETED for update in _display_updates(updates)
            )
            assert updates[-1].correlation["error_code"] == "invocation_result_unusable"

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


@pytest.mark.parametrize(
    ("display_text", "speak_text"),
    [
        pytest.param(
            "A long display result that stays out of model context.",
            "The structured result is ready.",
            id="structured-summary",
        ),
        pytest.param(
            "The operation completed successfully.",
            "Open https://example.invalid/result to see it.",
            id="url",
        ),
        pytest.param(
            "The operation completed successfully. Would you like more detail?",
            "The operation completed successfully. Would you like more detail?",
            id="question",
        ),
    ],
)
def test_runtime_preserves_backend_authored_speech(
    display_text: str,
    speak_text: str,
) -> None:
    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_StaticResultTurns(display_text, speak_text),
            )
            await runtime.open_session("session-explicit-speech", "conversation-explicit-speech")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-explicit-speech",
                    commit_id="commit-explicit-speech",
                    call_id="call-explicit-speech",
                    tool_name="work.delegate",
                    arguments=_delegation("complete the operation"),
                    finalized_user_text="complete the operation",
                )
            )

            terminal = updates[-1]
            response = terminal.frontend_response
            assert _display_text(updates) == display_text
            assert terminal.correlation["speech_source"] == ResponseOnlySpeechSource.BACKEND_AUTHORED.value
            assert response is not None
            assert response.purpose is FrontendResponsePurpose.RESULT_DELIVERY
            assert response.local_request_id == "commit-explicit-speech"
            assert response.payload_text == speak_text
            serialized_projection = runtime.projection("session-explicit-speech")
            projection = json.loads(serialized_projection)
            assert projection["latest_result"]["speech_source"] == ResponseOnlySpeechSource.BACKEND_AUTHORED.value
            assert projection["latest_result"]["display_available"] is True
            assert projection["latest_result"]["display_body_in_context"] is False
            assert projection["latest_result"]["provider_identifiers_in_context"] is False
            assert display_text not in serialized_projection
            assert speak_text not in serialized_projection
            assert "backend-session-static" not in serialized_projection
            assert "turn-static" not in serialized_projection
            assert "response-static" not in serialized_projection

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
            queued_output = dict(updates[0].tool_output or {})
            assert queued_output.pop("work_id").startswith("work_vc_")
            assert queued_output == {
                "status": "locally_queued",
                "request_state": "locally_queued",
                "local_request_id": "commit-failed",
                "identity_authority": "voiceclaw_volatile",
                "evidence": "voiceclaw_volatile",
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


def test_failure_before_adapter_dispatch_has_no_waiting_or_uncertain_state() -> None:
    class _PreDispatchFailureTurns(_Turns):
        async def stream_turn(self, request):
            del request
            raise CommittedTurnError("invocation_outcome_unknown")
            yield  # pragma: no cover

    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_PreDispatchFailureTurns(),
            )
            await runtime.open_session("session-pre-dispatch-failure", "conversation-pre-dispatch-failure")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-pre-dispatch-failure",
                    commit_id="commit-pre-dispatch-failure",
                    call_id="call-pre-dispatch-failure",
                    tool_name="work.delegate",
                    arguments=_delegation("fail before backend dispatch"),
                    finalized_user_text="fail before backend dispatch",
                )
            )

            assert _request_phases(updates) == [
                ResponseOnlyRequestState.LOCALLY_QUEUED,
                ResponseOnlyRequestState.DISPATCHING,
                ResponseOnlyRequestState.FAILED,
            ]
            projection = json.loads(runtime.projection("session-pre-dispatch-failure"))
            assert projection["execution"]["request_state"] == "failed"
            assert projection["execution"]["latest_terminal_outcome"] == "failed"

    asyncio.run(exercise())


def test_runtime_distinguishes_an_uncertain_invocation_from_a_confirmed_failure() -> None:
    class _UncertainTurns(_Turns):
        async def commit_turn(self, request):
            del request
            raise CommittedTurnError("invocation_outcome_unknown")

    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_UncertainTurns(),
            )
            await runtime.open_session("session-uncertain", "conversation-uncertain")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-uncertain",
                    commit_id="commit-uncertain",
                    call_id="call-uncertain",
                    tool_name="work.delegate",
                    arguments=_delegation("perform one side effect"),
                    finalized_user_text="perform one side effect",
                )
            )

            assert _request_phases(updates)[-1] is ResponseOnlyRequestState.OUTCOME_UNKNOWN
            terminal = updates[-1]
            assert terminal.correlation["error_code"] == "invocation_outcome_unknown"
            assert terminal.frontend_response is not None
            assert terminal.frontend_response.payload_text == "I couldn't confirm whether that request completed."
            projection = json.loads(runtime.projection("session-uncertain"))
            assert projection["execution"]["latest_terminal_outcome"] == "outcome_unknown"
            assert projection["execution"]["latest_failure_code"] == "invocation_outcome_unknown"

    asyncio.run(exercise())


def test_runtime_reports_retained_target_context_as_an_uncertain_outcome() -> None:
    class _ReusedTargetTurns(_Turns):
        async def commit_turn(self, request):
            del request
            raise CommittedTurnError("target_context_reused")

    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_ReusedTargetTurns(),
            )
            await runtime.open_session("session-reused", "conversation-reused")
            updates = await _collect(
                runtime.execute_tool(
                    session_id="session-reused",
                    commit_id="commit-reused",
                    call_id="call-reused",
                    tool_name="work.delegate",
                    arguments=_delegation("perform isolated work"),
                    finalized_user_text="perform isolated work",
                )
            )

            terminal = updates[-1]
            assert terminal.phase is ResponseOnlyRequestState.OUTCOME_UNKNOWN
            assert terminal.correlation["error_code"] == "target_context_reused"
            assert terminal.frontend_response is not None
            assert "couldn't verify the agent's history" in terminal.frontend_response.payload_text

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


def test_cancelling_a_request_waiting_for_the_backend_lane_does_not_leak_the_lane() -> None:
    class _LaneTurns(_Turns):
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()

        async def commit_turn(self, request):
            self.calls.append(request.commit_id)
            if request.commit_id == "commit-lane-first":
                self.first_started.set()
                await self.release_first.wait()
            return await super().commit_turn(request)

    async def exercise() -> None:
        turns = _LaneTurns()
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=turns,
            )
            for suffix in ("first", "waiting", "following"):
                await runtime.open_session(f"session-lane-{suffix}", f"conversation-lane-{suffix}")

            first = runtime.execute_tool(
                session_id="session-lane-first",
                commit_id="commit-lane-first",
                call_id="call-lane-first",
                tool_name="work.delegate",
                arguments=_delegation("hold the backend lane"),
                finalized_user_text="hold the backend lane",
            )
            for _ in range(3):
                await anext(first)
            first_remainder = asyncio.create_task(_collect(first))
            await turns.first_started.wait()

            waiting = runtime.execute_tool(
                session_id="session-lane-waiting",
                commit_id="commit-lane-waiting",
                call_id="call-lane-waiting",
                tool_name="work.delegate",
                arguments=_delegation("wait for the backend lane"),
                finalized_user_text="wait for the backend lane",
            )
            assert (await anext(waiting)).phase is ResponseOnlyRequestState.LOCALLY_QUEUED
            waiting_remainder = asyncio.create_task(_collect(waiting))
            await asyncio.sleep(0)
            assert waiting_remainder.done() is False
            waiting_remainder.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting_remainder
            waiting_projection = json.loads(runtime.projection("session-lane-waiting"))
            assert waiting_projection["execution"]["request_state"] == "cancelled"
            assert turns.calls == ["commit-lane-first"]

            turns.release_first.set()
            await first_remainder
            following = await asyncio.wait_for(
                _collect(
                    runtime.execute_tool(
                        session_id="session-lane-following",
                        commit_id="commit-lane-following",
                        call_id="call-lane-following",
                        tool_name="work.delegate",
                        arguments=_delegation("use the released backend lane"),
                        finalized_user_text="use the released backend lane",
                    )
                ),
                timeout=1,
            )
            assert following[-1].phase is ResponseOnlyRequestState.SUCCEEDED
            assert turns.calls == ["commit-lane-first", "commit-lane-following"]

    asyncio.run(exercise())


def test_closing_a_session_fences_its_queued_request_before_dispatch() -> None:
    class _LaneTurns(_Turns):
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()

        async def commit_turn(self, request):
            self.calls.append(request.commit_id)
            if request.commit_id == "commit-close-running":
                self.first_started.set()
                await self.release_first.wait()
            return await super().commit_turn(request)

    async def exercise() -> None:
        turns = _LaneTurns()
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=turns,
            )
            await runtime.open_session("session-close-running", "conversation-close-running")
            await runtime.open_session("session-close-queued", "conversation-close-queued")
            running = runtime.execute_tool(
                session_id="session-close-running",
                commit_id="commit-close-running",
                call_id="call-close-running",
                tool_name="work.delegate",
                arguments=_delegation("hold the backend lane"),
                finalized_user_text="hold the backend lane",
            )
            for _ in range(3):
                await anext(running)
            running_remainder = asyncio.create_task(_collect(running))
            await turns.first_started.wait()

            queued = runtime.execute_tool(
                session_id="session-close-queued",
                commit_id="commit-close-queued",
                call_id="call-close-queued",
                tool_name="work.delegate",
                arguments=_delegation("do not dispatch after disconnect"),
                finalized_user_text="do not dispatch after disconnect",
            )
            assert (await anext(queued)).phase is ResponseOnlyRequestState.LOCALLY_QUEUED
            queued_remainder = asyncio.create_task(_collect(queued))
            await asyncio.sleep(0)
            await runtime.close_session("session-close-queued", "client_disconnected")
            assert await asyncio.wait_for(queued_remainder, timeout=1) == []
            with pytest.raises(LookupError):
                runtime.projection("session-close-queued")
            assert turns.calls == ["commit-close-running"]

            turns.release_first.set()
            await running_remainder

    asyncio.run(exercise())


def test_invalid_local_control_does_not_create_a_work_record() -> None:
    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_Turns(),
            )
            await runtime.open_session("session-control-error", "conversation-control-error")
            delegated = await _collect(
                runtime.execute_tool(
                    session_id="session-control-error",
                    commit_id="commit-control-target",
                    call_id="call-control-target",
                    tool_name="work.delegate",
                    arguments=_delegation("produce one result"),
                    finalized_user_text="produce one result",
                )
            )
            work_id = str(delegated[0].tool_output["work_id"])
            before = json.loads(runtime.projection("session-control-error"))["local_requests"]

            missing = await _collect(
                runtime.execute_tool(
                    session_id="session-control-error",
                    commit_id="commit-missing-status",
                    call_id="call-missing-status",
                    tool_name="work.status",
                    arguments={"work_id": f"{work_id}-missing"},
                    finalized_user_text="Check the missing request.",
                )
            )
            assert len(missing) == 1
            assert missing[0].correlation["operation_id"] == "commit-missing-status"
            assert missing[0].correlation["error_code"] == "local_request_not_found"
            after = json.loads(runtime.projection("session-control-error"))["local_requests"]
            assert after == before
            assert all(request["local_request_id"] != "commit-missing-status" for request in after)

    asyncio.run(exercise())


def test_completed_result_is_committed_before_its_completed_display_update() -> None:
    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_Turns(),
            )
            await runtime.open_session("session-result-fence", "conversation-result-fence")
            updates = runtime.execute_tool(
                session_id="session-result-fence",
                commit_id="commit-result-fence",
                call_id="call-result-fence",
                tool_name="work.delegate",
                arguments=_delegation("produce a fenced result"),
                finalized_user_text="produce a fenced result",
            )
            completed = None
            while completed is None:
                update = await anext(updates)
                if (
                    update.kind is ResponseOnlyUpdateKind.RESULT_DISPLAY
                    and update.phase is ResponseOnlyResultEventKind.COMPLETED
                ):
                    completed = update
            await updates.aclose()

            projection = json.loads(runtime.projection("session-result-fence"))
            request = projection["local_requests"][0]
            assert request["request_state"] == "succeeded"
            assert request["result"]["result_id"] == completed.correlation["result_id"]
            assert request["result"]["delivery_id"] == completed.correlation["delivery_id"]
            assert projection["execution"]["latest_terminal_outcome"] == "succeeded"

            status = await _collect(
                runtime.execute_tool(
                    session_id="session-result-fence",
                    commit_id="commit-result-recap",
                    call_id="call-result-recap",
                    tool_name="work.status",
                    arguments={"work_id": request["local_work_id"]},
                    finalized_user_text="Tell me the result.",
                )
            )
            assert status[0].frontend_response is not None
            assert status[0].frontend_response.purpose is FrontendResponsePurpose.RESULT_DELIVERY
            assert status[0].frontend_response.local_request_id == "commit-result-fence"

    asyncio.run(exercise())


def test_status_does_not_claim_result_speech_twice() -> None:
    async def exercise() -> None:
        with SqliteStateStore(":memory:") as store:
            runtime = RealtimeInteractionManager(
                backend_profile="default",
                state_store=store,
                committed_turns=_Turns(),
            )
            await runtime.open_session("session-result-claim", "conversation-result-claim")
            delegated = await _collect(
                runtime.execute_tool(
                    session_id="session-result-claim",
                    commit_id="commit-result-claim",
                    call_id="call-result-claim",
                    tool_name="work.delegate",
                    arguments=_delegation("produce one spoken result"),
                    finalized_user_text="produce one spoken result",
                )
            )
            work_id = str(delegated[0].tool_output["work_id"])
            automatic = delegated[-1].frontend_response
            assert automatic is not None
            assert automatic.purpose is FrontendResponsePurpose.RESULT_DELIVERY

            status = await _collect(
                runtime.execute_tool(
                    session_id="session-result-claim",
                    commit_id="commit-result-status",
                    call_id="call-result-status",
                    tool_name="work.status",
                    arguments={"work_id": work_id},
                    finalized_user_text="What is the status?",
                )
            )
            assert status[0].frontend_response is not None
            assert status[0].frontend_response.purpose is FrontendResponsePurpose.STATUS_DELIVERY
            assert status[0].frontend_response.payload_text is None

    asyncio.run(exercise())
