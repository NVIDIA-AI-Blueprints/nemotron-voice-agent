# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

import json

from voiceclaw.application.context import (
    ActiveWorkContext,
    ContextSnapshot,
    ConversationTurn,
    FrontendInstructionBuilder,
    PendingInteractionContext,
    PromptContextBuilder,
    ReadyResultContext,
)
from voiceclaw.domain.models import WorkState
from voiceclaw.ports.runtime import FrontendConversationDeliveryState


def test_prompt_order_keeps_dynamic_projection_bounded_and_user_turn_last() -> None:
    builder = PromptContextBuilder(dynamic_character_budget=600)
    snapshot = ContextSnapshot(
        session_id="session-a",
        heard_through_presentation_id="presentation-3",
        heard_through_turn_id="turn-3",
        active_work=(ActiveWorkContext(work_id="work-a", state=WorkState.RUNNING, summary="Searching records"),),
        pending_interactions=(
            PendingInteractionContext(
                interaction_id="question-a",
                work_id="work-a",
                kind="question",
                prompt="Which date should I use?",
            ),
        ),
        ready_results=(ReadyResultContext(result_id="result-old", work_id="work-old", summary="Older work completed"),),
        recent_conversation=tuple(
            ConversationTurn(turn_id=f"turn-{index}", role="user", text="x" * 180) for index in range(5)
        ),
    )

    context = builder.build(
        static_instructions="Static tool and safety contract",
        profile_instructions="User prefers concise speech",
        snapshot=snapshot,
        finalized_user_turn="What happened with the search?",
    )

    assert [message.role for message in context.messages] == ["system", "system", "system", "user"]
    assert context.messages[-1].content == "What happened with the search?"
    assert context.dynamic_characters <= 600
    dynamic = json.loads(context.messages[-2].content)
    assert dynamic["active_work"][0]["work_id"] == "work-a"
    assert dynamic["pending_interactions"][0]["interaction_id"] == "question-a"
    assert dynamic["heard_through_presentation_id"] == "presentation-3"
    assert dynamic["heard_through_turn_id"] == "turn-3"
    assert all(
        turn["delivery_state"] == FrontendConversationDeliveryState.COMMITTED.value
        for turn in dynamic["recent_conversation"]
    )
    assert len(dynamic["recent_conversation"]) < 5


def test_context_snapshot_is_captured_per_build() -> None:
    builder = PromptContextBuilder(dynamic_character_budget=512)
    first = builder.build(
        static_instructions="Static",
        profile_instructions=None,
        snapshot=ContextSnapshot(session_id="session-a"),
        finalized_user_turn="First turn",
    )
    second = builder.build(
        static_instructions="Static",
        profile_instructions=None,
        snapshot=ContextSnapshot(
            session_id="session-a",
            ready_results=(ReadyResultContext(result_id="result-a", work_id="work-a", summary="Ready"),),
        ),
        finalized_user_turn="Second turn",
    )

    assert "result-a" not in first.messages[-2].content
    assert "result-a" in second.messages[-2].content
    assert first.messages[-1].content == "First turn"
    assert second.messages[-1].content == "Second turn"


def test_prompt_context_projects_conversation_delivery_identity_and_boundary() -> None:
    builder = PromptContextBuilder(dynamic_character_budget=1024)
    context = builder.build(
        static_instructions="Static",
        profile_instructions=None,
        snapshot=ContextSnapshot(
            session_id="session-delivery",
            heard_through_presentation_id="presentation-a",
            heard_through_turn_id="turn-a",
            recent_conversation=(
                ConversationTurn(
                    turn_id="turn-a",
                    role="assistant",
                    text="The result was partially played.",
                    delivery_state=FrontendConversationDeliveryState.INTERRUPTED,
                    presentation_id="presentation-a",
                    local_request_id="request-a",
                    heard_through_ms=420,
                    audio_end_ms=900,
                ),
            ),
        ),
        finalized_user_turn="Continue.",
    )

    dynamic = json.loads(context.messages[-2].content)
    assert dynamic["heard_through_presentation_id"] == "presentation-a"
    assert dynamic["heard_through_turn_id"] == "turn-a"
    assert dynamic["recent_conversation"] == [
        {
            "audio_end_ms": 900,
            "delivery_state": "interrupted",
            "heard_through_ms": 420,
            "local_request_id": "request-a",
            "presentation_id": "presentation-a",
            "role": "assistant",
            "text": "The result was partially played.",
            "turn_id": "turn-a",
        }
    ]


def test_realtime_instruction_builder_combines_projection_with_server_response_context() -> None:
    builder = FrontendInstructionBuilder(maximum_characters=2048)

    interactive = builder.build(
        static_instructions="Stable policy",
        client_session_instructions="Prefer short answers",
        client_response_instructions="Use a calm tone",
        dynamic_projection='{"active_work":[]}',
    )
    delivery = builder.build(
        static_instructions="Stable policy",
        client_session_instructions="Prefer short answers",
        dynamic_projection='{"active_work":[]}',
        response_context={"response_purpose": "delegation_ack"},
    )

    assert "Current VoiceClaw projection" in interactive
    assert "voiceclaw_response_context" not in interactive
    assert "voiceclaw_response_context" in delivery
    assert "active_work" in delivery
    assert "UNTRUSTED CLIENT SESSION" in delivery
