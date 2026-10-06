# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent port the engine talks to.

The engine never sees the text prototype directly. It asks an :class:`AgentPort`
to respond to user text or to resume with tool outputs, and gets back an
:class:`AgentReply` carrying spoken text *or* tool calls, never both, mirroring
the text prototype's ``AgentTurn``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class OutgoingCall:
    """One tool call to surface as a Realtime ``function_call`` item."""

    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class RoleUsage:
    """One LLM role's calls, tokens and summed call latency in one agent step."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0

    def as_record(self) -> dict[str, int | float]:
        """The event-log form."""
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": round(self.latency_ms, 1),
        }


@dataclass(frozen=True, slots=True)
class ReplyUsage:
    """Token usage for ``response.done.usage``, plus the per-role split for the event log."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    frontend: RoleUsage = field(default_factory=RoleUsage)
    backend: RoleUsage = field(default_factory=RoleUsage)


@dataclass(frozen=True, slots=True)
class AgentReply:
    """Spoken text xor tool calls.

    ``staged`` is true when the turn finished while staging was on (a barge-in
    review is open): the result is held by the agent, not committed, until
    :meth:`AgentPort.end_staging` commits or drops it.
    """

    text: str | None = None
    calls: tuple[OutgoingCall, ...] = ()
    usage: ReplyUsage = field(default_factory=ReplyUsage)
    staged: bool = False

    def __post_init__(self) -> None:
        """Enforce the text/calls exclusivity."""
        if (self.text is None) == (not self.calls):
            raise ValueError("AgentReply carries exactly one of text or calls")


@dataclass(frozen=True, slots=True)
class InFlight:
    """The running turn's delegation, recorded once the frontend has delegated."""

    query: str
    filler_text: str


#: Probe verdicts: keep the running request, or carry out the probe's decision instead.
VERDICT_CONTINUE = "continue"
VERDICT_NEW = "new"


@dataclass(frozen=True, slots=True)
class Probe:
    """The frontend's decision for speech during a running turn, not yet carried out.

    ``step`` is agent-specific and opaque to the engine; only :meth:`AgentPort.proceed`
    reads it.
    """

    verdict: str
    reason: str
    running_query: str
    probe_query: str
    step: Any
    latency_ms: float = 0.0
    frontend: RoleUsage = field(default_factory=RoleUsage)
    #: The ``task`` the model sent (``""`` when missing or invalid), before the same-query guard.
    model_task: str = ""


@runtime_checkable
class AgentPort(Protocol):
    """One session's conversational agent."""

    @property
    def backend_only(self) -> bool:
        """Whether the agent runs without a frontend (no filler source)."""

    def configure(self, *, tools: Sequence[Mapping[str, Any]], instructions: str) -> None:
        """(Re)build for the session's tools and instructions; conversation state is kept."""

    async def respond(self, text: str) -> AgentReply:
        """Answer one user turn. Cancellation must leave the conversation state untouched."""

    async def resume(self, outputs: Mapping[str, str]) -> AgentReply:
        """Continue a turn with the client's function-call outputs, keyed by ``call_id``."""

    def repair_last_answer(self, full_text: str, replacement: str) -> None:
        """Rewrite every stored copy of the last spoken answer (raises ``HistoryRepairError``)."""

    def seed_assistant(self, text: str) -> None:
        """Record assistant speech the agent did not produce (greetings)."""

    # -- barge-in frontend verdict (barge_in.while_thinking: frontend_verdict) ------

    @property
    def in_flight(self) -> InFlight | None:
        """The running turn's delegation, or ``None`` before the frontend delegated (or when idle)."""

    async def probe(self, text: str, *, new_words: str, filler_spoken: bool) -> Probe:
        """Frontend-only decision for speech during a running turn. Must not touch conversation state."""

    async def proceed(self, probe: Probe) -> AgentReply:
        """Carry out a probe's decision on the state it was made on; commits like :meth:`respond`."""

    def begin_staging(self) -> None:
        """Hold a text result of the call in flight (``AgentReply.staged``) instead of committing it."""

    def end_staging(self, *, commit: bool) -> None:
        """Stop staging; a held result is committed (``commit``) or dropped as if the turn was cancelled."""
