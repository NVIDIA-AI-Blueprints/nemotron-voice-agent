# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delegation contract constants for Responses requests and events, and a resolved model endpoint."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Final, Literal, Protocol, TypedDict

TERMINALS: Final = frozenset({"response.completed", "response.failed", "response.incomplete"})
# The request options besides the model, the instructions, the tools, the tool choice and the output limit.
OPTIONS = ("parallel_tool_calls", "reasoning", "text", "service_tier")
# Metadata keys: the delegation a request starts or continues, the mark of a typed run, valued "true", and a
# conversation's opt-in to keep work after its call, valued "true" (a Live session's store: true).
DELEGATION_ID = "live_delegation_id"
CALLER_EXCERPT = "live_caller_excerpt"
CALLER_EXCERPT_CHARS = 300
CALLER_CONTEXT = "_nva_caller_context"
TYPED_RUN = "live_typed_run"
STORE = "live_store"
PROGRESS_PHASE = "commentary"
# The API a model endpoint speaks when its services.yaml entry names none: chat completions.
CHAT_COMPLETIONS: Final = "chat-completions"
RetainedAppCall = Literal["publish", "pending", "withhold"]
AppResultState = float | Literal["closed", "cancelled"]
AppResultWindow = Callable[[str, str, AppResultState], None]
APP_RESULT_WINDOW = "_nva_app_result_window"
WORK_PROVENANCE_KEY = "_nva_work"


@dataclass(frozen=True)
class AppCallDelivery:
    """Local receipt and result-window lookup for one conversation."""

    queued: Callable[[str], None]
    window: Callable[[str], AppResultState | None]


class CallerContext(TypedDict):
    """Caller text and playback evidence carried between the frontend and Core."""

    caller_turns: list[tuple[int, str]]
    playback: list[dict]


class SettleCallerTurn(Protocol):
    """Outcome of the frontend's temporary caller-input dispatch hold."""

    def __call__(self, *, routing_failed: bool = False) -> None:
        """Release dispatch after routing, or mark the caller input unresolved on failure."""


def is_progress(item: dict) -> bool:
    """Return whether a message item is progress (phase ``commentary``) rather than a final answer."""
    return item.get("phase") == PROGRESS_PHASE


def strip_provenance(event: Mapping[str, object]) -> dict:
    """Copy a public event without its host-only work provenance."""
    return {key: value for key, value in event.items() if key != WORK_PROVENANCE_KEY}


@dataclass(frozen=True)
class ModelEndpoint:
    """A model endpoint from a ``services.yaml`` entry and the session body; ``extra_params`` reach the client as is."""

    model: str
    base_url: str
    api_key: str = field(default="not-needed", repr=False)
    extra_params: dict = field(default_factory=dict)
    # The API its server speaks: the entry's ``api``, a provider name of the delegation core's ``BACKENDS``.
    api: str = CHAT_COMPLETIONS
