# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

import pytest

from voiceclaw.domain.capabilities import (
    CapabilityToolRegistry,
    PendingToolQuery,
    ToolProjectionState,
    ToolWork,
)
from voiceclaw.domain.models import (
    AgentQueryKind,
    BackendCapabilities,
    BackendOperation,
    CapabilityEvidence,
    CapabilitySource,
    Durability,
    EventDelivery,
    WorkState,
)
from voiceclaw.interaction_profiles import MAX_DELEGATED_GOAL_CHARACTERS, load_interaction_profile_catalog

_INTERACTION_PROFILES = load_interaction_profile_catalog()


def _registry(profile_name: str) -> CapabilityToolRegistry:
    return CapabilityToolRegistry(profile=_INTERACTION_PROFILES.resolve(profile_name))


def _state(
    *works: ToolWork,
    queries: tuple[PendingToolQuery, ...] = (),
    **overrides: object,
) -> ToolProjectionState:
    return ToolProjectionState(works=works, pending_queries=queries, **overrides)


def test_capability_evidence_digest_is_deterministic_for_the_typed_claim() -> None:
    first = BackendCapabilities(
        backend_kind="stateful_agent",
        target_label="Configured agent",
        revision="cap-v2",
        operations=frozenset({BackendOperation.SUBMIT, BackendOperation.CANCEL}),
        durability=Durability.BACKEND,
        event_delivery=EventDelivery.ORDERED_REPLAY,
        sessionful=True,
    )
    reordered = BackendCapabilities(
        backend_kind="stateful_agent",
        target_label="Configured agent",
        revision="cap-v2",
        operations=frozenset({BackendOperation.CANCEL, BackendOperation.SUBMIT}),
        durability=Durability.BACKEND,
        event_delivery=EventDelivery.ORDERED_REPLAY,
        sessionful=True,
    )

    evidence = CapabilityEvidence.capture(
        first,
        source=CapabilitySource.BACKEND_NEGOTIATED,
        source_id="test.adapter",
    )
    repeated = CapabilityEvidence.capture(
        reordered,
        source=CapabilitySource.BACKEND_NEGOTIATED,
        source_id="test.adapter",
    )

    assert evidence == repeated
    assert evidence.revision == "cap-v2"
    assert evidence.source_id == "test.adapter"
    assert evidence.digest.startswith("sha256:")
    assert len(evidence.digest) == 71


def test_capability_evidence_digest_changes_when_authorization_changes() -> None:
    submit_only = BackendCapabilities(
        backend_kind="stateless_model",
        target_label="Frontier model",
        revision="cap-v1",
        operations=frozenset({BackendOperation.SUBMIT}),
    )
    submit_and_cancel = BackendCapabilities(
        backend_kind="stateless_model",
        target_label="Frontier model",
        revision="cap-v1",
        operations=frozenset({BackendOperation.SUBMIT, BackendOperation.CANCEL}),
    )

    first = CapabilityEvidence.capture(
        submit_only,
        source=CapabilitySource.OPERATOR_CONFIGURED,
        source_id="test.adapter",
    )
    second = CapabilityEvidence.capture(
        submit_and_cancel,
        source=CapabilitySource.OPERATOR_CONFIGURED,
        source_id="test.adapter",
    )

    assert first.digest != second.digest


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"durability": Durability.BACKEND}, "durable backends must be sessionful"),
        (
            {"event_delivery": EventDelivery.ORDERED_REPLAY},
            "ordered replay requires a durable sessionful backend",
        ),
        (
            {
                "operations": frozenset({BackendOperation.REPLAY}),
                "durability": Durability.BACKEND,
                "sessionful": True,
            },
            "work.replay requires ordered replay delivery",
        ),
        (
            {
                "operations": frozenset({BackendOperation.STEER}),
            },
            "stateful operations require a sessionful backend",
        ),
    ],
)
def test_capability_claim_rejects_cross_field_contradictions(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        BackendCapabilities(
            backend_kind="test",
            target_label="test target",
            **overrides,
        )


def test_response_only_stateless_profile_needs_no_persisted_state() -> None:
    capabilities = BackendCapabilities(
        backend_kind="response_only",
        target_label="configured target",
        revision="compat-v1",
        operations=frozenset({BackendOperation.SUBMIT, BackendOperation.CANCEL}),
        durability=Durability.NONE,
        event_delivery=EventDelivery.RESPONSE_ONLY,
    )
    registry = _registry("stateless")

    tools = registry.project(capabilities)

    assert [tool.name for tool in tools] == ["work.delegate"]
    assert tools[0].input_schema["required"] == ["goal"]
    assert tools[0].input_schema["properties"]["goal"]["maxLength"] == MAX_DELEGATED_GOAL_CHARACTERS
    assert registry.operation_for_tool("work.delegate", capabilities) is BackendOperation.SUBMIT


def test_delegate_goal_is_core_required_bounded_and_normalized() -> None:
    registry = _registry("stateless")

    with pytest.raises(ValueError, match="missing arguments.*goal"):
        registry.validate_arguments("work.delegate", {})
    with pytest.raises(ValueError, match="goal is too short"):
        registry.validate_arguments("work.delegate", {"goal": "   "})
    with pytest.raises(ValueError, match="goal is too short"):
        registry.validate_arguments("work.delegate", {"goal": "valid\x00invalid"})
    with pytest.raises(ValueError, match="goal is too long"):
        registry.validate_arguments("work.delegate", {"goal": "a" * (MAX_DELEGATED_GOAL_CHARACTERS + 1)})

    assert dict(registry.validate_arguments("work.delegate", {"goal": "  compare both trees  "})) == {
        "goal": "compare both trees"
    }


def test_stateful_projection_fails_closed_without_authoritative_state() -> None:
    capabilities = BackendCapabilities(
        backend_kind="stateful",
        target_label="stateful target",
        operations=frozenset({BackendOperation.SUBMIT}),
        durability=Durability.SESSION,
        sessionful=True,
    )

    with pytest.raises(ValueError, match="authoritative state snapshot"):
        _registry("single_stateful").project(capabilities)


def test_conductor_projects_only_four_normalized_tools_when_each_is_applicable() -> None:
    capabilities = BackendCapabilities(
        backend_kind="conductor",
        target_label="conductor",
        operations=frozenset(BackendOperation),
        durability=Durability.BACKEND,
        event_delivery=EventDelivery.ORDERED_REPLAY,
        sessionful=True,
    )
    state = _state(
        ToolWork("work-1", WorkState.RUNNING),
        queries=(
            PendingToolQuery(
                query_id="query-1",
                work_id="work-1",
                kind=AgentQueryKind.INFORMATION,
                blocking=False,
            ),
        ),
    )

    tools = _registry("conductor").project(capabilities, state=state)

    assert [tool.name for tool in tools] == [
        "work.answer_agent",
        "work.cancel",
        "work.delegate",
        "work.status",
    ]
    assert tools[0].input_schema["properties"]["query_id"]["enum"] == ["query-1"]
    assert tools[1].input_schema["properties"]["work_id"]["enum"] == ["work-1"]
    assert "steer_work" not in {tool.name for tool in tools}


@pytest.mark.parametrize(
    ("state", "tool_name", "projected", "expected"),
    [
        (ToolProjectionState(), "work.delegate", ["work.delegate"], BackendOperation.SUBMIT),
        (
            _state(ToolWork("work-1", WorkState.RUNNING)),
            "work.delegate",
            ["work.delegate"],
            BackendOperation.STEER,
        ),
        (
            _state(
                ToolWork("work-1", WorkState.WAITING_INPUT),
                queries=(
                    PendingToolQuery(
                        query_id="query-1",
                        work_id="work-1",
                        kind=AgentQueryKind.INFORMATION,
                        blocking=True,
                    ),
                ),
            ),
            "work.answer_agent",
            ["work.answer_agent"],
            BackendOperation.ANSWER_QUERY,
        ),
        (
            _state(
                ToolWork("work-1", WorkState.WAITING_PERMISSION),
                queries=(
                    PendingToolQuery(
                        query_id="query-1",
                        work_id="work-1",
                        kind=AgentQueryKind.PERMISSION,
                        blocking=True,
                    ),
                ),
            ),
            "work.answer_agent",
            ["work.answer_agent"],
            BackendOperation.RESPOND_PERMISSION,
        ),
    ],
)
def test_single_stateful_tools_resolve_from_work_and_query_state(
    state: ToolProjectionState,
    tool_name: str,
    projected: list[str],
    expected: BackendOperation,
) -> None:
    capabilities = BackendCapabilities(
        backend_kind="stateful",
        target_label="stateful target",
        operations=frozenset(
            {
                BackendOperation.SUBMIT,
                BackendOperation.STEER,
                BackendOperation.ANSWER_QUERY,
                BackendOperation.RESPOND_PERMISSION,
            }
        ),
        durability=Durability.SESSION,
        sessionful=True,
    )
    registry = _registry("single_stateful")

    assert [tool.name for tool in registry.project(capabilities, state=state)] == projected
    assert registry.operation_for_tool(tool_name, capabilities, state=state) is expected


def test_single_stateful_nonblocking_query_does_not_hijack_steering() -> None:
    capabilities = BackendCapabilities(
        backend_kind="stateful",
        target_label="stateful target",
        operations=frozenset({BackendOperation.STEER, BackendOperation.ANSWER_QUERY}),
        durability=Durability.SESSION,
        sessionful=True,
    )
    state = _state(
        ToolWork("work-1", WorkState.RUNNING),
        queries=(
            PendingToolQuery(
                query_id="query-1",
                work_id="work-1",
                kind=AgentQueryKind.INFORMATION,
                blocking=False,
            ),
        ),
    )

    registry = _registry("single_stateful")

    assert [tool.name for tool in registry.project(capabilities, state=state)] == [
        "work.answer_agent",
        "work.delegate",
    ]
    assert registry.operation_for_tool("work.delegate", capabilities, state=state) is BackendOperation.STEER
    assert (
        registry.operation_for_tool(
            "work.answer_agent",
            capabilities,
            state=state,
            arguments={"query_id": "query-1"},
        )
        is BackendOperation.ANSWER_QUERY
    )


def test_claimed_blocking_query_withholds_duplicate_response() -> None:
    capabilities = BackendCapabilities(
        backend_kind="stateful",
        target_label="stateful target",
        operations=frozenset({BackendOperation.ANSWER_QUERY}),
        durability=Durability.SESSION,
        sessionful=True,
    )
    state = _state(
        ToolWork("work-1", WorkState.WAITING_INPUT),
        queries=(
            PendingToolQuery(
                query_id="query-1",
                work_id="work-1",
                kind=AgentQueryKind.INFORMATION,
                blocking=True,
                claimed=True,
            ),
        ),
    )

    assert _registry("single_stateful").project(capabilities, state=state) == ()


def test_cancelling_work_consumes_capacity_but_cannot_be_steered_or_cancelled_again() -> None:
    capabilities = BackendCapabilities(
        backend_kind="stateful",
        target_label="stateful target",
        operations=frozenset(
            {BackendOperation.SUBMIT, BackendOperation.STEER, BackendOperation.CANCEL, BackendOperation.STATUS}
        ),
        durability=Durability.SESSION,
        sessionful=True,
    )

    tools = _registry("single_stateful").project(
        capabilities,
        state=_state(ToolWork("work-1", WorkState.CANCELLING)),
    )

    assert [tool.name for tool in tools] == ["work.status"]


def test_stateless_cancel_is_restricted_to_cancellable_backend_work() -> None:
    capabilities = BackendCapabilities(
        backend_kind="stateless",
        target_label="configured target",
        operations=frozenset({BackendOperation.SUBMIT, BackendOperation.CANCEL}),
        supports_parallel_work=True,
        max_parallel_work=2,
    )
    state = _state(ToolWork("work-1", WorkState.RUNNING))

    tools = _registry("stateless").project(capabilities, state=state)

    assert [tool.name for tool in tools] == ["work.cancel", "work.delegate"]
    assert tools[0].input_schema["properties"]["work_id"]["enum"] == ["work-1"]


def test_specialized_routes_are_derived_from_backend_owned_work_targets() -> None:
    capabilities = BackendCapabilities(
        backend_kind="specialized",
        target_label="specialized targets",
        operations=frozenset(BackendOperation),
        durability=Durability.BACKEND,
        event_delivery=EventDelivery.ORDERED_REPLAY,
        agent_targets=("finance", "procurement"),
        sessionful=True,
        supports_parallel_work=True,
        max_parallel_work=2,
    )
    state = _state(
        ToolWork("work-finance", WorkState.RUNNING, agent_target="finance"),
        queries=(
            PendingToolQuery(
                query_id="query-finance",
                work_id="work-finance",
                kind=AgentQueryKind.INFORMATION,
                blocking=False,
            ),
        ),
    )
    registry = _registry("specialized")

    tools = {tool.name: tool for tool in registry.project(capabilities, state=state)}

    assert list(tools) == ["work.answer_agent", "work.cancel", "work.delegate", "work.status"]
    assert tools["work.delegate"].input_schema["properties"]["agent"]["enum"] == [
        "finance",
        "procurement",
    ]
    assert tools["work.cancel"].input_schema["properties"]["agent"]["enum"] == ["finance"]
    assert (
        registry.operation_for_tool(
            "work.delegate",
            capabilities,
            state=state,
            arguments={"agent": "finance", "goal": "Continue the current finance analysis."},
        )
        is BackendOperation.STEER
    )
    assert (
        registry.operation_for_tool(
            "work.delegate",
            capabilities,
            state=state,
            arguments={"agent": "procurement", "goal": "Analyze the procurement request."},
        )
        is BackendOperation.SUBMIT
    )


def test_specialized_work_without_backend_target_evidence_fails_closed() -> None:
    capabilities = BackendCapabilities(
        backend_kind="specialized",
        target_label="specialized targets",
        operations=frozenset({BackendOperation.SUBMIT}),
        durability=Durability.BACKEND,
        agent_targets=("finance",),
        sessionful=True,
    )

    with pytest.raises(ValueError, match="backend-authored agent_target"):
        _registry("specialized").project(
            capabilities,
            state=_state(ToolWork("work-1", WorkState.RUNNING)),
        )


def test_conductor_delegate_is_hidden_when_submit_and_steer_are_ambiguous() -> None:
    capabilities = BackendCapabilities(
        backend_kind="conductor",
        target_label="conductor",
        operations=frozenset({BackendOperation.SUBMIT, BackendOperation.STEER}),
        durability=Durability.BACKEND,
        sessionful=True,
        supports_parallel_work=True,
        max_parallel_work=2,
    )
    state = _state(ToolWork("work-1", WorkState.RUNNING))

    registry = _registry("conductor")

    assert registry.project(capabilities, state=state) == ()
    with pytest.raises(ValueError, match="server-side route context"):
        registry.operation_for_tool("work.delegate", capabilities, state=state)


def test_status_keeps_terminal_work_visible_but_cancel_does_not() -> None:
    capabilities = BackendCapabilities(
        backend_kind="conductor",
        target_label="conductor",
        operations=frozenset({BackendOperation.CANCEL, BackendOperation.STATUS}),
        durability=Durability.BACKEND,
        sessionful=True,
    )
    state = _state(ToolWork("work-1", WorkState.SUCCEEDED))

    tools = _registry("conductor").project(capabilities, state=state)

    assert [tool.name for tool in tools] == ["work.status"]
    assert tools[0].input_schema["properties"]["work_id"]["enum"] == ["work-1"]


def test_local_recovery_evidence_only_reduces_mutating_tool_availability() -> None:
    capabilities = BackendCapabilities(
        backend_kind="conductor",
        target_label="conductor",
        operations=frozenset({BackendOperation.SUBMIT, BackendOperation.STATUS}),
        durability=Durability.BACKEND,
        sessionful=True,
    )
    state = ToolProjectionState(
        anonymous_capacity_reservations=1,
        command_recovery_pending=True,
    )

    assert [tool.name for tool in _registry("conductor").project(capabilities, state=state)] == ["work.status"]


def test_tool_projection_state_rejects_queries_for_missing_or_terminal_work() -> None:
    query = PendingToolQuery(
        query_id="query-1",
        work_id="work-1",
        kind=AgentQueryKind.INFORMATION,
        blocking=True,
    )

    with pytest.raises(ValueError, match="non-terminal Work"):
        ToolProjectionState(pending_queries=(query,))
    with pytest.raises(ValueError, match="non-terminal Work"):
        _state(ToolWork("work-1", WorkState.SUCCEEDED), queries=(query,))


def test_shape_only_validation_does_not_authorize_a_tool() -> None:
    capabilities = BackendCapabilities(
        backend_kind="single",
        target_label="single target",
        operations=frozenset({BackendOperation.CANCEL}),
        durability=Durability.SESSION,
        sessionful=True,
    )
    registry = _registry("single_stateful")

    assert dict(registry.validate_arguments("work.cancel", {}, capabilities=capabilities)) == {}
    with pytest.raises(KeyError):
        registry.validate_arguments(
            "work.cancel",
            {},
            capabilities=capabilities,
            state=ToolProjectionState(),
        )
