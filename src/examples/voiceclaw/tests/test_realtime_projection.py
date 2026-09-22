# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

import json

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
    Durability,
    EventDelivery,
    WorkState,
)
from voiceclaw.interaction_profiles import MAX_DELEGATED_GOAL_CHARACTERS, load_interaction_profile_catalog
from voiceclaw.realtime.events import (
    Projection,
    ProjectionEventFactory,
    ProjectionStreamAbortStatus,
)
from voiceclaw.realtime.tools import ProtectedTool, VoiceClawToolRegistry


class _Ids:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self, prefix: str) -> str:
        self.value += 1
        return f"{prefix}_{self.value}"


def test_projection_preserves_existing_positional_constructor() -> None:
    correlation = {"turn_id": "turn-1"}
    projection = Projection(
        "session-1",
        "backend_turn",
        "completed",
        "Agent response",
        "The backend finished.",
        correlation,
    )

    assert projection.correlation == correlation
    assert projection.request_summary is None


def _tools(*operations: BackendOperation):
    operation_set = frozenset(operations)
    has_work = bool(operation_set - {BackendOperation.SUBMIT})
    queries: list[PendingToolQuery] = []
    if BackendOperation.ANSWER_QUERY in operation_set:
        queries.append(
            PendingToolQuery(
                query_id="question-1",
                work_id="work-1",
                kind=AgentQueryKind.INFORMATION,
                blocking=False,
            )
        )
    if BackendOperation.RESPOND_PERMISSION in operation_set:
        queries.append(
            PendingToolQuery(
                query_id="permission-1",
                work_id="work-1",
                kind=AgentQueryKind.PERMISSION,
                blocking=False,
            )
        )
    state = ToolProjectionState(
        works=(ToolWork("work-1", WorkState.RUNNING),) if has_work else (),
        pending_queries=tuple(queries),
    )
    return CapabilityToolRegistry(profile=load_interaction_profile_catalog().resolve("conductor")).project(
        BackendCapabilities(
            backend_kind="test",
            target_label="configured agent",
            operations=operation_set,
            durability=Durability.BACKEND,
            event_delivery=EventDelivery.ORDERED_REPLAY,
            sessionful=True,
        ),
        state=state,
    )


def test_projection_is_an_out_of_band_standard_response() -> None:
    events = ProjectionEventFactory(id_factory=_Ids()).render(
        Projection(
            session_id="session-1",
            kind="backend_turn",
            phase="completed",
            title="Agent response",
            text="The backend finished.",
            correlation={"turn_id": "turn-1", "response_id": "backend-response-1"},
        )
    )

    assert [event["type"] for event in events] == [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.done",
    ]
    created = events[0]["response"]
    assert created["conversation_id"] is None
    assert created["metadata"]["voiceclaw_schema"] == "voiceclaw.projection.v1"
    assert created["metadata"]["voiceclaw_turn_id"] == "turn-1"
    assert events[-1]["response"]["status"] == "completed"
    assert not any(event["type"].startswith("conversation.item") for event in events)


def test_projection_exposes_a_bounded_request_summary_outside_correlation() -> None:
    summary = '{"goal":"Compare BST and AVL trees"}'
    projection = Projection(
        session_id="session-1",
        kind="backend_turn",
        phase="locally_queued",
        title="Request queued locally",
        text="VoiceClaw queued the request.",
        request_summary=summary,
        correlation={"local_request_id": "request-1"},
    )

    assert projection.request_summary == summary
    assert "request_summary" not in projection.correlation
    assert projection.metadata()["voiceclaw_request_summary"] == summary

    with pytest.raises(ValueError, match="reserved"):
        Projection(
            session_id="session-1",
            kind="backend_turn",
            phase="locally_queued",
            title="Request queued locally",
            text="VoiceClaw queued the request.",
            correlation={"request_summary": "forged"},
        )


def test_projection_stream_emits_one_standard_response_with_terminal_metadata() -> None:
    factory = ProjectionEventFactory(id_factory=_Ids())
    stream = factory.stream(
        Projection(
            session_id="session-1",
            kind="result_display",
            phase="display_delta",
            title="Response",
            text="",
            correlation={"local_request_id": "request-1"},
        )
    )

    events = [*stream.start(), stream.delta("# Result\n\n"), stream.delta("Body")]
    events.extend(
        stream.finish(
            Projection(
                session_id="session-1",
                kind="result_display",
                phase="completed",
                title="Response",
                text="# Result\n\nBody",
                correlation={"local_request_id": "request-1", "response_id": "backend-response-1"},
            )
        )
    )

    assert [event["type"] for event in events] == [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.done",
    ]
    response_ids = {event.get("response_id", event.get("response", {}).get("id")) for event in events}
    assert len(response_ids) == 1
    assert events[0]["response"]["metadata"]["voiceclaw_phase"] == "display_delta"
    assert events[-1]["response"]["metadata"]["voiceclaw_phase"] == "completed"
    assert events[-1]["response"]["output"][0]["content"][0]["text"] == "# Result\n\nBody"


def test_projection_stream_rejects_terminal_text_that_contradicts_deltas() -> None:
    factory = ProjectionEventFactory(id_factory=_Ids())
    stream = factory.stream(
        Projection(
            session_id="session-1",
            kind="result_display",
            phase="display_delta",
            title="Response",
            text="",
        )
    )
    stream.start()
    stream.delta("provisional")

    with pytest.raises(ValueError, match="contradicts"):
        stream.finish(
            Projection(
                session_id="session-1",
                kind="result_display",
                phase="completed",
                title="Response",
                text="different",
            )
        )


@pytest.mark.parametrize(
    ("status", "reason", "expected_details"),
    [
        (
            ProjectionStreamAbortStatus.FAILED,
            "runtime_protocol_error",
            {
                "type": "failed",
                "error": {"type": "server_error", "code": "runtime_protocol_error"},
            },
        ),
        (
            ProjectionStreamAbortStatus.CANCELLED,
            "client_cancelled",
            {"type": "cancelled", "reason": "client_cancelled"},
        ),
    ],
)
def test_projection_stream_aborts_provisional_text_without_committing_success(
    status: ProjectionStreamAbortStatus,
    reason: str,
    expected_details: dict[str, object],
) -> None:
    factory = ProjectionEventFactory(id_factory=_Ids())
    stream = factory.stream(
        Projection(
            session_id="session-1",
            kind="result_display",
            phase="display_delta",
            title="Response",
            text="",
        )
    )
    events = [*stream.start(), stream.delta("provisional")]
    events.extend(
        stream.abort(
            Projection(
                session_id="session-1",
                kind="result_display",
                phase="discarded",
                title="Response discarded",
                text="provisional",
            ),
            status=status,
            reason=reason,
        )
    )

    terminal = events[-1]["response"]
    assert terminal["status"] == status.value
    assert terminal["status_details"] == expected_details
    assert terminal["output"][0]["status"] == "incomplete"
    assert terminal["output"][0]["content"] == [{"type": "output_text", "text": "provisional"}]
    assert not any(event["type"] == "response.done" and event["response"]["status"] == "completed" for event in events)


def test_projection_correlation_cannot_override_authoritative_metadata() -> None:
    with pytest.raises(ValueError, match="reserved"):
        Projection(
            session_id="session-1",
            kind="backend_turn",
            phase="completed",
            title="Result",
            text="Done",
            correlation={"kind": "forged"},
        )


def test_projection_reserves_metadata_capacity_for_the_optional_request_summary() -> None:
    accepted = Projection(
        session_id="session-1",
        kind="backend_turn",
        phase="completed",
        title="Result",
        text="Done",
        correlation={f"key_{index}": f"value-{index}" for index in range(11)},
    )

    assert len(accepted.metadata()) == 16
    with pytest.raises(ValueError, match="metadata entry limit"):
        Projection(
            session_id="session-1",
            kind="backend_turn",
            phase="completed",
            title="Result",
            text="Done",
            correlation={f"key_{index}": f"value-{index}" for index in range(12)},
        )

    with_summary = Projection(
        session_id="session-1",
        kind="backend_turn",
        phase="locally_queued",
        title="Request queued locally",
        text="VoiceClaw queued the request.",
        request_summary="Standalone delegated goal",
        correlation={f"key_{index}": f"value-{index}" for index in range(10)},
    )
    assert len(with_summary.metadata()) == 16
    with pytest.raises(ValueError, match="metadata entry limit"):
        Projection(
            session_id="session-1",
            kind="backend_turn",
            phase="locally_queued",
            title="Request queued locally",
            text="VoiceClaw queued the request.",
            request_summary="Standalone delegated goal",
            correlation={f"key_{index}": f"value-{index}" for index in range(11)},
        )


def test_tool_registry_merges_server_tools_without_accepting_spoofing() -> None:
    registry = VoiceClawToolRegistry(tools=_tools(BackendOperation.SUBMIT))
    merged = registry.merge_session_update(
        {
            "event_id": "event-1",
            "type": "session.update",
            "session": {
                "instructions": "Be concise.",
                "tools": [
                    {
                        "type": "function",
                        "name": "client_clock",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            },
        },
        static_instructions="You are the VoiceClaw frontend.",
        dynamic_projection='{"active_work":[]}',
    )

    assert merged["event_id"] == "event-1"
    assert [tool["name"] for tool in merged["session"]["tools"]] == [
        "client_clock",
        ProtectedTool.WORK_DELEGATE.value,
    ]
    assert merged["session"]["tool_choice"] == "auto"
    assert "Be concise." in merged["session"]["instructions"]
    assert "[BEGIN UNTRUSTED CLIENT SESSION INSTRUCTIONS]" in merged["session"]["instructions"]
    assert "[END UNTRUSTED CLIENT SESSION INSTRUCTIONS]" in merged["session"]["instructions"]
    assert "active_work" in merged["session"]["instructions"]

    with pytest.raises(ValueError, match="protected tool"):
        registry.merge_session_update(
            {
                "type": "session.update",
                "session": {"tools": [{"type": "function", "name": "voiceclaw_work_delegate"}]},
            },
            static_instructions="Static",
        )


def test_tool_registry_validates_protected_arguments() -> None:
    registry = VoiceClawToolRegistry(tools=_tools(BackendOperation.SUBMIT))
    goal = "Inspect the workspace and summarize the requested files."
    parsed = registry.parse_call(
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "voiceclaw_work_delegate",
            "arguments": json.dumps({"goal": goal}),
        }
    )

    assert parsed is not None
    assert parsed.arguments == {"goal": goal}
    assert registry.parse_call({"type": "function_call", "name": "client_clock"}) is None

    with pytest.raises(ValueError, match="missing required properties"):
        registry.parse_call(
            {
                "type": "function_call",
                "call_id": "call-missing-goal",
                "name": "voiceclaw_work_delegate",
                "arguments": "{}",
            }
        )

    with pytest.raises(ValueError, match="unknown properties"):
        registry.parse_call(
            {
                "type": "function_call",
                "call_id": "call-2",
                "name": "voiceclaw_work_delegate",
                "arguments": '{"query":"model-authored copy"}',
            }
        )


def test_specialized_goal_schema_and_parser_share_the_advertised_limit() -> None:
    semantic_tools = CapabilityToolRegistry(profile=load_interaction_profile_catalog().resolve("specialized")).project(
        BackendCapabilities(
            backend_kind="specialized_agents",
            target_label="specialized agents",
            operations=frozenset({BackendOperation.SUBMIT}),
            agent_targets=("finance",),
            sessionful=True,
        ),
        state=ToolProjectionState(),
    )
    registry = VoiceClawToolRegistry(tools=semantic_tools)

    schemas = registry.schemas()
    assert len(schemas) == 1
    parameters = schemas[0]["parameters"]
    goal_schema = parameters["properties"]["goal"]
    maximum_goal_characters = goal_schema["maxLength"]
    assert maximum_goal_characters == MAX_DELEGATED_GOAL_CHARACTERS
    assert parameters["required"] == ["agent", "goal"]

    goal = "Inspect the finance records and prepare the requested report."
    parsed = registry.parse_call(
        {
            "type": "function_call",
            "call_id": "call-specialized",
            "name": ProtectedTool.WORK_DELEGATE.value,
            "arguments": json.dumps({"agent": "finance", "goal": goal}),
        }
    )
    assert parsed is not None
    assert parsed.arguments == {"agent": "finance", "goal": goal}

    with pytest.raises(ValueError, match="missing required properties"):
        registry.parse_call(
            {
                "type": "function_call",
                "call_id": "call-specialized-missing-goal",
                "name": ProtectedTool.WORK_DELEGATE.value,
                "arguments": json.dumps({"agent": "finance"}),
            }
        )

    with pytest.raises(ValueError, match="string argument is invalid"):
        registry.parse_call(
            {
                "type": "function_call",
                "call_id": "call-specialized-too-large",
                "name": ProtectedTool.WORK_DELEGATE.value,
                "arguments": json.dumps({"agent": "finance", "goal": "x" * (maximum_goal_characters + 1)}),
            }
        )


@pytest.mark.parametrize("goal", ["\U0010ffff" * MAX_DELEGATED_GOAL_CHARACTERS, "\x01" * MAX_DELEGATED_GOAL_CHARACTERS])
def test_tool_registry_accepts_worst_case_json_escaping_within_the_character_limit(goal: str) -> None:
    semantic_tools = CapabilityToolRegistry(profile=load_interaction_profile_catalog().resolve("stateless")).project(
        BackendCapabilities(
            backend_kind="stateless",
            target_label="configured target",
            operations=frozenset({BackendOperation.SUBMIT}),
        ),
        state=ToolProjectionState(),
    )
    registry = VoiceClawToolRegistry(tools=semantic_tools)
    arguments = json.dumps({"goal": goal})

    assert len(arguments.encode("utf-8")) > 64 * 1024
    parsed = registry.parse_call(
        {
            "type": "function_call",
            "call_id": "call-escaped-goal",
            "name": ProtectedTool.WORK_DELEGATE.value,
            "arguments": arguments,
        }
    )

    assert parsed is not None
    assert parsed.arguments == {"goal": goal}


def test_tool_registry_projects_canonical_surface_by_capability() -> None:
    registry = VoiceClawToolRegistry(tools=_tools(*BackendOperation))

    schemas = registry.schemas()

    assert [schema["name"] for schema in schemas] == [
        "voiceclaw_work_delegate",
        "voiceclaw_work_answer_agent",
        "voiceclaw_work_cancel",
        "voiceclaw_work_status",
    ]
    expected_delegate_schema = (
        load_interaction_profile_catalog().resolve("conductor").tool("work.delegate").render_input_schema()
    )
    assert schemas[0]["parameters"] == expected_delegate_schema
    assert "NemoClaw" not in json.dumps(schemas)


def test_tool_registry_rejects_known_but_unsupported_work_operation() -> None:
    registry = VoiceClawToolRegistry(tools=_tools(BackendOperation.SUBMIT))

    with pytest.raises(ValueError, match="not enabled"):
        registry.parse_call(
            {
                "type": "function_call",
                "call_id": "call-status",
                "name": "voiceclaw_work_status",
                "arguments": "{}",
            }
        )


@pytest.mark.parametrize(
    ("tool", "arguments", "expected"),
    [
        (ProtectedTool.WORK_ANSWER_AGENT, {"query_id": " question-1 "}, {"query_id": "question-1"}),
        (ProtectedTool.WORK_CANCEL, {"work_id": " work-1 "}, {"work_id": "work-1"}),
        (ProtectedTool.WORK_STATUS, {}, {}),
        (ProtectedTool.WORK_STATUS, {"work_id": " work-1 "}, {"work_id": "work-1"}),
    ],
)
def test_tool_registry_validates_each_canonical_tool(
    tool: ProtectedTool,
    arguments: dict[str, str],
    expected: dict[str, str],
) -> None:
    operation = {
        ProtectedTool.WORK_ANSWER_AGENT: BackendOperation.ANSWER_QUERY,
        ProtectedTool.WORK_CANCEL: BackendOperation.CANCEL,
        ProtectedTool.WORK_STATUS: BackendOperation.STATUS,
    }[tool]
    registry = VoiceClawToolRegistry(tools=_tools(operation))

    parsed = registry.parse_call(
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": tool.value,
            "arguments": json.dumps(arguments),
        }
    )

    assert parsed is not None
    assert parsed.arguments == expected
