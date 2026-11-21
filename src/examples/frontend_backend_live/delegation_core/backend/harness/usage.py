# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reported native counters, without treating an absent observation as zero."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from .protocol import Agent, PromptEnded, SessionBinding

COUNTERS = ("inputTokens", "outputTokens", "totalTokens", "thoughtTokens", "cachedReadTokens", "cachedWriteTokens")


def snapshot(usage: Mapping | None) -> dict[str, int]:
    """Copy only valid, explicitly reported counters from one observation."""
    return {
        key: value
        for key in COUNTERS
        if isinstance(value := (usage or {}).get(key), int) and not isinstance(value, bool) and value >= 0
    }


def round_counters(
    current: Mapping | None,
    before: Mapping | None,
    invocation: Mapping | None = None,
    *,
    agent: Agent | None = None,
    epoch: str | None = None,
    before_epoch: str | None = None,
) -> dict[str, int]:
    """Return exact invocation counters, or comparable cumulative counter differences.

    A missing prior field or a counter reset leaves that field unknown. An explicit empty invocation observation
    overrides cumulative history: the current invocation reported no counters.
    """
    if invocation is not None:
        return snapshot(invocation)
    scope = agent.capabilities.counter_scope if agent is not None else "unscoped"
    if scope == "invocation":
        return snapshot(current)
    if scope != "cumulative_epoch" or epoch is None or epoch != before_epoch:
        return {}
    old = snapshot(before)
    return {key: value - old[key] for key, value in snapshot(current).items() if key in old and value >= old[key]}


def observation(agent: Agent, current: Mapping | None, invocation: Mapping | None) -> dict:
    """Keep raw observations and explicit counter scope separate from attributable invocation totals."""
    return {
        "raw_reported_counters": snapshot(current) if current is not None else None,
        "usage_scope": "invocation" if invocation is not None else agent.capabilities.counter_scope,
        "usage_observation": "absent" if invocation is None and current is None else "reported",
    }


class InvocationUsage:
    """One authoritative native invocation observation, updated under the same identity at its terminal."""

    def __init__(self, binding: SessionBinding, prompt: int, invocation_id: str):
        """Correlate every replacement to one actual native session/prompt operation."""
        self.binding, self.prompt, self.id = binding, prompt, invocation_id
        self.version = 0

    def record(self, agent: Agent, *, terminal: PromptEnded | None = None, dispatched: bool = False) -> None:
        """Record a replacement observation; callers must not sum versions or supporting worker summaries."""
        self.version += 1
        try:
            self._record(agent, terminal=terminal, dispatched=dispatched)
        except Exception:
            # Measurement must not convert a submitted native request into an apparent non-dispatch.
            with suppress(Exception):
                logger.warning("harness.invocation_usage_observer_failed invocation_id={}", self.id)

    def _record(self, agent: Agent, *, terminal: PromptEnded | None, dispatched: bool) -> None:
        current = terminal.usage if terminal is not None else None
        invocation = terminal.invocation_usage if terminal is not None else None
        counts = round_counters(current, None, invocation, agent=agent)
        completion = agent.completion(terminal) if terminal is not None else None
        logger.debug(
            "harness.invocation.usage invocation_id={} observation_version={} replaces_version={} "
            "authority=native_invocation connection={} session={} prompt={} role={} dispatch={} "
            "terminal={} model_invoked={} usage={} usage_missing={} observation={}",
            self.id,
            self.version,
            self.version - 1 or None,
            self.binding.connection_id,
            self.binding.session_id,
            self.prompt,
            self.binding.role,
            "dispatched" if dispatched else terminal.dispatch if terminal is not None else "scheduled",
            completion.outcome if completion is not None else "pending",
            completion.model_invoked if completion is not None else None,
            responses_usage(counts),
            not bool(counts),
            observation(agent, current, invocation),
        )


def responses_usage(counts: Mapping) -> dict | None:
    """Map known counters to standard Responses fields; cache writes remain private reported counters."""
    result = {
        target: counts[source]
        for source, target in (
            ("inputTokens", "input_tokens"),
            ("outputTokens", "output_tokens"),
            ("totalTokens", "total_tokens"),
        )
        if source in counts
    }
    if "cachedReadTokens" in counts:
        result["input_tokens_details"] = {"cached_tokens": counts["cachedReadTokens"]}
    if "thoughtTokens" in counts:
        result["output_tokens_details"] = {"reasoning_tokens": counts["thoughtTokens"]}
    return result or None
