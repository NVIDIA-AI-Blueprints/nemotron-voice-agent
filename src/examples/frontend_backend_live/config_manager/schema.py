# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for the example. Pure data: no provider is imported here."""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_CONTINUERS = ("uh-huh", "uh huh", "mm-hmm", "mm hmm", "mhm", "hmm", "mm", "mmm", "ah", "oh", "i see", "got it")


@dataclass(frozen=True)
class ModelEndpoint:
    """One chat-style model endpoint, resolved from a ``services.yaml`` entry and the session body.

    ``extra_params`` are passed to the client verbatim (``temperature``, ``max_tokens``, ``extra_body``,
    ``stream_options`` ...), so settings written there by ``settings.yaml`` take effect unchanged.
    """

    model: str
    base_url: str
    api_key: str = "not-needed"
    extra_params: dict = field(default_factory=dict)
    # Frontend only: json_schema -> json_object -> none; a rejecting 400 steps down once.
    response_format: str = "json_schema"


@dataclass(frozen=True)
class RoleConfig:
    """Which implementation serves a role and where its model and static prompt come from.

    ``slot`` is the ``examples_registry.yaml`` services slot that supplies the model; ``prompt`` is the
    ``prompts.yaml`` entry used as the role's instructions when a session supplies none.
    """

    provider: str
    slot: str
    prompt: str = ""


@dataclass(frozen=True)
class ReliabilityConfig:
    """Timeouts, retries and recovery. Retries never wrap an action that could repeat."""

    provider_retries: int = 2
    talker_attempt_timeout_seconds: float = 10.0
    api_timeout_seconds: float = 45.0
    tool_timeout_seconds: float = 60.0
    max_tool_rounds: int = 8
    max_pending_delegations: int = 16
    # Recover from a failed talker/commentary/thinker call instead of surfacing an error.
    recover_failures: bool = True


@dataclass(frozen=True)
class GuardConfig:
    """Deterministic floors beneath the prompts."""

    repeat_guard: bool = True
    # Drop acknowledgment-only turns before the talker ("okay", "mm-hmm"); shadow mode only traces them.
    turn_router: bool = False
    turn_router_shadow: bool = False
    continuers: tuple[str, ...] = DEFAULT_CONTINUERS


@dataclass(frozen=True)
class LiveConfig:
    """Everything ``config.yaml`` controls."""

    frontend: RoleConfig
    backend: RoleConfig
    prompt_version: str = "v1"
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    guards: GuardConfig = field(default_factory=GuardConfig)
    # Backend (thinker) token budget for the delegate's resent history, per session.
    backend_context_tokens: int = 100_000
    # Conversation turns kept in the talker's context.
    chat_history_recent_turns: int = 20
