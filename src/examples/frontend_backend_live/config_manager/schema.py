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
class DelegationConfig:
    """How one call's several delegations are handled."""

    # A delegation that arrives while another is still waiting joins it, so one backend run answers both.
    merge_pending: bool = True
    # A request that arrives while the running delegation has not called a tool yet is folded into it: the run
    # restarts once with both requests, so no model judgement is needed to notice a correction.
    fold_before_acting: bool = True
    # A short "any update?" turn is answered from the work in progress instead of starting a delegation.
    status_from_ledger: bool = True
    # Speak a short status line when work has run this long without the assistant saying anything.
    progress_speech: bool = True
    progress_after_seconds: float = 8.0
    progress_interval_seconds: float = 12.0
    progress_max_per_delegation: int = 1


@dataclass(frozen=True)
class LiveConfig:
    """Everything ``config.yaml`` controls."""

    frontend: RoleConfig
    backend: RoleConfig
    prompt_version: str = "v1"
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    guards: GuardConfig = field(default_factory=GuardConfig)
    delegation: DelegationConfig = field(default_factory=DelegationConfig)
    # Backend (thinker) token budget for the delegate's resent history, per session.
    backend_context_tokens: int = 100_000
    # Conversation turns kept in the talker's context.
    chat_history_recent_turns: int = 20
