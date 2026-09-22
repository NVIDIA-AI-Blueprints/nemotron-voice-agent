# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Bounded response-boundary context projection for frontend models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from voiceclaw.domain.models import WorkState
from voiceclaw.model_contracts import ModelContractCatalog, load_model_contract_catalog
from voiceclaw.ports.runtime import FrontendConversationDeliveryState


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """Delivered conversational text eligible for bounded model context."""

    turn_id: str
    role: str
    text: str
    delivery_state: FrontendConversationDeliveryState = FrontendConversationDeliveryState.COMMITTED
    presentation_id: str | None = None
    local_request_id: str | None = None
    heard_through_ms: int | None = None
    audio_end_ms: int | None = None


@dataclass(frozen=True, slots=True)
class ActiveWorkContext:
    """Compact current Work projection."""

    work_id: str
    state: WorkState
    summary: str


@dataclass(frozen=True, slots=True)
class PendingInteractionContext:
    """Blocking backend question or permission awaiting user input."""

    interaction_id: str
    work_id: str
    kind: str
    prompt: str


@dataclass(frozen=True, slots=True)
class ReadyResultContext:
    """Compact result available for presentation or follow-up."""

    result_id: str
    work_id: str
    summary: str


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    """Immutable local projection captured at a response boundary."""

    session_id: str
    heard_through_presentation_id: str | None = None
    heard_through_turn_id: str | None = None
    active_work: tuple[ActiveWorkContext, ...] = ()
    pending_interactions: tuple[PendingInteractionContext, ...] = ()
    ready_results: tuple[ReadyResultContext, ...] = ()
    recent_conversation: tuple[ConversationTurn, ...] = ()


@dataclass(frozen=True, slots=True)
class PromptMessage:
    """Provider-neutral ordered prompt message."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class PromptContext:
    """Static, projected, and current-turn messages in required order."""

    messages: tuple[PromptMessage, ...]
    dynamic_characters: int


class PromptContextBuilder:
    """Build a bounded dynamic snapshot without mutating an active response."""

    def __init__(self, *, dynamic_character_budget: int) -> None:
        """Configure the maximum serialized dynamic projection size."""
        if dynamic_character_budget < 256:
            raise ValueError("dynamic_character_budget must be at least 256")
        self._budget = dynamic_character_budget

    def build(
        self,
        *,
        static_instructions: str,
        profile_instructions: str | None,
        snapshot: ContextSnapshot,
        finalized_user_turn: str,
    ) -> PromptContext:
        """Capture context once and keep the finalized user turn last."""
        projection: dict[str, Any] = {
            "schema": "voiceclaw.context.v1",
            "session_id": snapshot.session_id,
            "heard_through_presentation_id": snapshot.heard_through_presentation_id,
            "heard_through_turn_id": snapshot.heard_through_turn_id,
            "active_work": [],
            "pending_interactions": [],
            "ready_results": [],
            "recent_conversation": [],
        }
        if len(self._serialize(projection)) > self._budget:
            raise ValueError("dynamic_character_budget cannot hold the base context envelope")

        for work in snapshot.active_work:
            self._try_append(
                projection,
                "active_work",
                {"work_id": work.work_id, "state": work.state.value, "summary": work.summary},
            )
        for pending in snapshot.pending_interactions:
            self._try_append(
                projection,
                "pending_interactions",
                {
                    "interaction_id": pending.interaction_id,
                    "work_id": pending.work_id,
                    "kind": pending.kind,
                    "prompt": pending.prompt,
                },
            )
        for result in snapshot.ready_results:
            self._try_append(
                projection,
                "ready_results",
                {"result_id": result.result_id, "work_id": result.work_id, "summary": result.summary},
            )

        accepted_turns: list[dict[str, Any]] = []
        for turn in reversed(snapshot.recent_conversation):
            candidate: dict[str, Any] = {
                "turn_id": turn.turn_id,
                "role": turn.role,
                "text": turn.text,
                "delivery_state": turn.delivery_state.value,
            }
            if turn.presentation_id is not None:
                candidate["presentation_id"] = turn.presentation_id
            if turn.local_request_id is not None:
                candidate["local_request_id"] = turn.local_request_id
            if turn.heard_through_ms is not None:
                candidate["heard_through_ms"] = turn.heard_through_ms
            if turn.audio_end_ms is not None:
                candidate["audio_end_ms"] = turn.audio_end_ms
            projection["recent_conversation"] = [candidate, *accepted_turns]
            if len(self._serialize(projection)) <= self._budget:
                accepted_turns.insert(0, candidate)
            else:
                projection["recent_conversation"] = accepted_turns

        dynamic = self._serialize(projection)
        messages = [PromptMessage(role="system", content=static_instructions)]
        if profile_instructions:
            messages.append(PromptMessage(role="system", content=profile_instructions))
        messages.extend(
            (
                PromptMessage(role="system", content=dynamic),
                PromptMessage(role="user", content=finalized_user_turn),
            )
        )
        return PromptContext(messages=tuple(messages), dynamic_characters=len(dynamic))

    def _try_append(self, projection: dict[str, Any], key: str, item: dict[str, Any]) -> bool:
        values: list[dict[str, Any]] = projection[key]
        values.append(item)
        if len(self._serialize(projection)) <= self._budget:
            return True
        values.pop()
        return False

    @staticmethod
    def _serialize(projection: dict[str, Any]) -> str:
        return json.dumps(projection, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class FrontendInstructionBuilder:
    """Assemble one authoritative Realtime response instruction boundary.

    Conversation items remain owned by the public Realtime protocol.  This
    builder keeps server policy, explicitly untrusted client preferences, and
    a read-only runtime projection, and an optional least-privilege server
    response context in one application-owned instruction path.
    """

    def __init__(
        self,
        *,
        maximum_characters: int,
        contracts: ModelContractCatalog | None = None,
    ) -> None:
        """Set the hard bound applied after all instruction sections are assembled."""
        if maximum_characters < 256:
            raise ValueError("maximum_characters must be at least 256")
        self._maximum_characters = maximum_characters
        self._contracts = contracts or load_model_contract_catalog()

    def build(
        self,
        *,
        static_instructions: str,
        client_session_instructions: str = "",
        client_response_instructions: str = "",
        dynamic_projection: str = "",
        response_context: Mapping[str, str] | None = None,
    ) -> str:
        """Return one bounded instruction string for the actual response boundary."""
        values = (
            static_instructions,
            client_session_instructions,
            client_response_instructions,
            dynamic_projection,
        )
        if not all(isinstance(value, str) and "\x00" not in value for value in values):
            raise ValueError("frontend instruction sections must be text without NUL bytes")
        if not static_instructions.strip():
            raise ValueError("static frontend instructions must not be empty")
        sections = [self._contracts.render_instruction("server_policy", content=static_instructions.strip())]
        if client_session_instructions.strip():
            sections.append(
                self._contracts.render_instruction("untrusted_session", content=client_session_instructions.strip())
            )
        if client_response_instructions.strip():
            sections.append(
                self._contracts.render_instruction("untrusted_response", content=client_response_instructions.strip())
            )
        if dynamic_projection.strip():
            sections.append(
                self._contracts.render_instruction("dynamic_projection", projection=dynamic_projection.strip())
            )
        if response_context is not None:
            context = dict(response_context)
            if not all(
                isinstance(key, str) and key and "\x00" not in key and isinstance(value, str) and "\x00" not in value
                for key, value in context.items()
            ):
                raise ValueError("frontend response context must contain bounded text facts")
            serialized = json.dumps(context, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            sections.append(self._contracts.render_instruction("response_context", context_json=serialized))
        instructions = "\n\n".join(sections)
        if len(instructions) > self._maximum_characters:
            raise ValueError("frontend instructions exceed the configured bound")
        return instructions
