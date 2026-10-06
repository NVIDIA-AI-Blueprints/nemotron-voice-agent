# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Session-local planner/executor backend for the generic domain."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger
from openai import APIConnectionError, APIError, APITimeoutError, InternalServerError, RateLimitError

from examples.frontend_backend_agent.generic import plan_checks
from examples.frontend_backend_agent.generic.argument_screen import ArgumentScreen
from examples.frontend_backend_agent.generic.client_tools import (
    ClientToolRoundExecutor,
    ClientToolSpec,
    build_client_tool_specs,
    client_call_fingerprint,
    format_client_result,
)
from examples.frontend_backend_agent.generic.consent import ConsentBook
from examples.frontend_backend_agent.generic.dispatcher import (
    PlanValidationError,
    combine_accumulated_results,
    dispatch_plan,
)
from examples.frontend_backend_agent.generic.planner import (
    GenericPlanner,
    GenericPlannerSessionUpdate,
    PlanTruncatedError,
)
from examples.frontend_backend_agent.generic.result_formatters import nothing_further, planner_failure, timeout_failure
from examples.frontend_backend_agent.generic.state import GenericThinkerSessionState
from examples.frontend_backend_agent.src.delegation import DelegationRun, current_run_id
from examples.frontend_backend_agent.src.flags import flag
from examples.frontend_backend_agent.src.history import DelegationLedger
from examples.frontend_backend_agent.src.normalization import tool_kind
from examples.frontend_backend_agent.src.protocol import ThinkerLifecycleEvent
from examples.frontend_backend_agent.src.stage_metrics import STREAM_ACTIVITY
from examples.frontend_backend_agent.src.tools import ToolSpec, session_server_tools
from utils import parse_env_float, parse_env_int

if TYPE_CHECKING:
    from examples.frontend_backend_agent.src.stage_metrics import StageMetricsCoordinator

_PLANNER_MAX_ATTEMPTS = parse_env_int("GENERIC_PLANNER_MAX_ATTEMPTS", 2, min_value=1)
# How many distinct lookups one session keeps for later turns. Keyed per call,
# so a caller with several records keeps each of them.
_SESSION_TOOL_MEMORY_LIMIT = 12
_PLANNER_RETRY_BACKOFF_SECONDS = parse_env_float("GENERIC_PLANNER_RETRY_BACKOFF_SECONDS", 0.2, min_value=0.0)
#: A plan cut off at the token limit is retried once like a transient failure.
_RETRIABLE_PLANNER_EXCEPTIONS = (
    TimeoutError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    PlanTruncatedError,
)
#: Shared inference endpoints report transient saturation as a streamed
#: ``APIError`` with no HTTP status, so the class alone cannot separate it from
#: a malformed request. Match the wording instead of retrying every APIError.
_TRANSIENT_PLANNER_ERROR_TEXT = (
    "overloaded",
    "temporarily unavailable",
    "service unavailable",
    "capacity",
    "try again",
    "too many requests",
)


class PlannerCeilingError(TimeoutError):
    """A live planning stream ran to its per-attempt ceiling; asking again would start over."""


def _is_retriable_planner_error(exc: BaseException) -> bool:
    """Return whether one planner failure is worth another attempt."""
    if isinstance(exc, PlannerCeilingError):
        return False
    if isinstance(exc, _RETRIABLE_PLANNER_EXCEPTIONS):
        return True
    if isinstance(exc, APIError):
        message = str(exc).lower()
        return any(fragment in message for fragment in _TRANSIENT_PLANNER_ERROR_TEXT)
    return False


_DEFAULT_MAX_PLANNING_ROUNDS = 8
#: A clarification question kept for a later progress check. It is code-written
#: (schema labels or a fixed sentence), so carrying it forward never makes
#: planner text speakable.
_PENDING_QUESTION_REASONS = frozenset({"params_missing", "params_invalid"})
#: A pending question is offered again at most this many times, then dropped.
_PENDING_QUESTION_MAX_REDELIVERIES = 2
_PENDING_QUESTION_TTL_SECONDS = 60.0
#: How long a superseded run's answer waits for newer calls to settle.
_LATE_ANSWER_MAX_WAIT_SECONDS = 20.0
#: Superseded runs allowed to keep planning; an older one beyond this stops.
_DETACHED_RUN_CAP = parse_env_int("FRONTEND_BACKEND_DETACHED_RUN_CAP", 1, min_value=0)
#: Thinker text handed to the Talker as a draft answer or a refusal reason is bounded.
_MAX_THINKER_TEXT_CHARS = 600


@dataclass(frozen=True, slots=True)
class GenericBackendSessionUpdate:
    """Prepared planner and client-tool state for one session transaction."""

    planner: GenericPlannerSessionUpdate
    client_tools: dict[str, ClientToolSpec]
    enabled_tools: tuple[str, ...]
    argument_screen: ArgumentScreen | None = None


@dataclass(frozen=True, slots=True)
class GenericBackendSessionSnapshot:
    """Rollback snapshot for the dynamic Generic backend policy."""

    planner: GenericPlannerSessionUpdate
    client_tools: dict[str, ClientToolSpec]
    enabled_tools: tuple[str, ...]
    argument_screen: ArgumentScreen | None = None


class GenericThinkerBackend:
    """Run one bounded, replaceable backend task per voice session."""

    # Generic formatters already produce grounded, TTS-safe speech; avoid a second
    # Talker pass over Pipecat's asynchronous started/final result envelope.
    tool_result_mode_default = "direct"
    talker_result_tools = ("get_weather",)
    # A later call_backend can take over the run still working (frontend verdict).
    supports_task_continuation = True

    def __init__(
        self,
        *,
        planner: GenericPlanner,
        enabled_tools: tuple[str, ...],
        tools: Mapping[str, ToolSpec],
        client_tools: Mapping[str, ClientToolSpec] | None = None,
        client_tool_executor: ClientToolRoundExecutor | None = None,
        client_tool_timeout_seconds: float = 25.0,
        overall_timeout_seconds: float = 40.0,
        planner_timeout_seconds: float = 6.0,
        planner_first_chunk_timeout_seconds: float | None = None,
        planner_stall_timeout_seconds: float | None = None,
        max_planning_rounds: int = _DEFAULT_MAX_PLANNING_ROUNDS,
        state: GenericThinkerSessionState | None = None,
        on_tool_started: Callable[[str], Awaitable[None]] | None = None,
        stage_metrics: StageMetricsCoordinator | None = None,
        conversation_ledger: DelegationLedger | None = None,
    ) -> None:
        """Create a backend with bounded planner and end-to-end deadlines."""
        self._planner = planner
        self._tools = dict(tools)
        self._client_tools = dict(client_tools or {})
        self._ledger = conversation_ledger
        # Client tools carry no mutation metadata. A call is recorded at the
        # executor, the real side-effect boundary where it is published and its
        # output awaited: as a read only when its name and description both say
        # it reads (``tool_kind``), otherwise as a potential write.
        self._client_tool_executor = (
            _recording_client_executor(client_tool_executor, conversation_ledger, self.client_tool_kind)
            if client_tool_executor is not None and conversation_ledger is not None
            else client_tool_executor
        )
        self._client_tool_timeout_seconds = max(1.0, client_tool_timeout_seconds)
        self._server_enabled_tools = tuple(name for name in enabled_tools if name in self._tools)
        client_enabled = tuple(name for name in enabled_tools if name not in self._tools)
        self._enabled_tools = (*session_server_tools(self._server_enabled_tools, client_enabled), *client_enabled)
        self._overall_timeout_seconds = max(1.0, overall_timeout_seconds)
        self._planner_timeout_seconds = min(max(1.0, planner_timeout_seconds), self._overall_timeout_seconds)
        # With both liveness limits set, a planning attempt ends only when its
        # stream has not started or has stopped producing chunks; the planner
        # timeout is then the ceiling for one attempt. A reasoning model that is
        # still thinking at the old fixed limit used to be cancelled and asked
        # the same question again from scratch.
        self._planner_liveness = (
            (max(0.5, planner_first_chunk_timeout_seconds), max(0.5, planner_stall_timeout_seconds))
            if planner_first_chunk_timeout_seconds is not None and planner_stall_timeout_seconds is not None
            else None
        )
        self._max_planning_rounds = max(1, max_planning_rounds)
        self._on_tool_started = on_tool_started
        self._stage_metrics = stage_metrics
        # Session-scoped so a client call that keeps failing stays suppressed across
        # turns; the frontend re-delegates per turn, and per-call state would reset.
        self._seen_client_calls: set[str] = set()
        # Client calls a run of this session is executing right now. A newer
        # run that plans the same call waits for that result instead of having
        # it suppressed as a failed repeat or sending it twice.
        self._running_client_calls: dict[str, asyncio.Future[dict[str, Any]]] = {}
        # What this session has already established. A delegation starts with an
        # empty result list, so without this the planner re-plans its opening
        # lookup on every user turn -- and once one of those rounds is lost to
        # the caller speaking over it, the repeat is suppressed as a duplicate
        # and the conversation can never reach a second step.
        self._session_tool_memory: list[dict[str, Any]] = []
        # Superseded plans still running. Held only so the event loop keeps a
        # strong reference to them until they finish.
        self._detached: set[asyncio.Task[dict[str, Any]]] = set()
        self._active_run: DelegationRun | None = None
        # Late answers of superseded runs (frontend verdict ``new``).
        self._late_answers_enabled = flag("FRONTEND_BACKEND_LATE_ANSWERS")
        self._pending_question_enabled = flag("FRONTEND_BACKEND_PENDING_QUESTION")
        self._pending_question: _PendingQuestion | None = None
        self._superseded_runs: set[str] = set()
        self._owner_gone: set[str] = set()
        self._late_answers_waiting: set[str] = set()
        self._late_delivered: set[str] = set()
        self._last_late_question_tool: str | None = None
        self._direct_write_enabled = flag("FRONTEND_BACKEND_DIRECT_WRITE")
        self._done_guard_enabled = flag("FRONTEND_BACKEND_DONE_GUARD")
        self._plan_checks_enabled = flag("FRONTEND_BACKEND_PLAN_CHECKS")
        # Approval needs the caller's turns, so the gate runs only with the history's transcript.
        transcript = conversation_ledger.transcript if conversation_ledger is not None else None
        self._consent_gate_enabled = flag("FRONTEND_BACKEND_CONSENT_GATE") and transcript is not None
        self._write_consent = _write_consent_mode()
        self._consent = ConsentBook()
        if self._consent_gate_enabled and transcript is not None:
            transcript.add_user_turn_listener(self._observe_consent_turn)
        # Superseded runs, oldest first, and what each is doing now (planning,
        # or a tool round with or without a pending client write).
        self._detached_runs: dict[str, asyncio.Task[dict[str, Any]]] = {}
        self._run_phase: dict[str, str] = {}
        #: Runs that must end once their pending tool round has returned.
        self._stop_after_tools: set[str] = set()
        #: Withdrawn runs whose dispatched write is still reported.
        self._withdrawn_writes: set[str] = set()
        #: Incremented by every committed session update; voids pending approvals.
        self._session_generation = 0
        self._argument_screen = _argument_screen_for(
            [
                {"type": "function", "name": spec.name, "description": spec.description, "parameters": spec.parameters}
                for spec in self._client_tools.values()
            ]
        )
        self.state = state or GenericThinkerSessionState()

    @property
    def session_instruction_context(self):
        """Return the Thinker context that owns session.instructions verbatim."""
        return self._planner.session_instruction_context

    def render_session_instructions(self, instructions: str) -> list[dict[str, Any]]:
        """Render the exact client-owned Thinker instruction message."""
        return self._planner.render_session_instructions(instructions)

    def prepare_session_update(
        self,
        *,
        instructions: str,
        client_tools: Sequence[Mapping[str, Any]],
    ) -> GenericBackendSessionUpdate:
        """Validate a live prompt/tool update without changing active state."""
        if self.state.active_task is not None and not self.state.active_task.done():
            raise RuntimeError("Generic planner policy cannot change during an active backend call")
        client_specs = build_client_tool_specs(client_tools)
        planner_update = self._planner.prepare_session_update(
            instructions=instructions,
            client_tools=client_tools,
        )
        return GenericBackendSessionUpdate(
            planner=planner_update,
            client_tools=client_specs,
            enabled_tools=(*session_server_tools(self._server_enabled_tools, client_specs), *client_specs),
            argument_screen=_argument_screen_for(client_tools),
        )

    def snapshot_session_update(self) -> GenericBackendSessionSnapshot:
        """Capture dynamic planner and client-tool state for rollback."""
        return GenericBackendSessionSnapshot(
            planner=self._planner.snapshot_session_update(),
            client_tools=dict(self._client_tools),
            enabled_tools=self._enabled_tools,
            argument_screen=self._argument_screen,
        )

    def commit_session_update(self, prepared: GenericBackendSessionUpdate) -> None:
        """Install one prepared dynamic planner and client-tool policy."""
        if not isinstance(prepared, GenericBackendSessionUpdate):
            raise TypeError("Generic backend session update has an invalid receipt")
        if self.state.active_task is not None and not self.state.active_task.done():
            raise RuntimeError("Generic planner policy cannot change during an active backend call")
        self._planner.commit_session_update(prepared.planner)
        self._session_generation += 1
        self.clear_pending_question("session_update")
        self._client_tools = dict(prepared.client_tools)
        self._enabled_tools = prepared.enabled_tools
        if prepared.argument_screen is not None:
            self._argument_screen = prepared.argument_screen
            _log_argument_screen(self._argument_screen)

    def restore_session_update(self, snapshot: GenericBackendSessionSnapshot) -> None:
        """Restore dynamic planner and client-tool state after a failed commit."""
        if not isinstance(snapshot, GenericBackendSessionSnapshot):
            raise TypeError("Generic backend rollback has an invalid snapshot")
        self._planner.restore_session_update(snapshot.planner)
        self._client_tools = dict(snapshot.client_tools)
        self._enabled_tools = snapshot.enabled_tools
        if snapshot.argument_screen is not None:
            self._argument_screen = snapshot.argument_screen

    def client_tool_kind(self, name: str) -> str:
        """Return ``read`` or ``write`` for one of this session's client tools; unknown tools are writes."""
        spec = self._client_tools.get(name)
        return tool_kind(spec.name, spec.description) if spec is not None else "write"

    @property
    def conversation_ledger(self) -> DelegationLedger | None:
        """Return the session's delegation history, when backend history is enabled."""
        return self._ledger

    def running_query(self) -> str | None:
        """Return the query of the run still working, if any."""
        run = self._active_run
        return run.query if run is not None and run.running else None

    async def call(
        self,
        query: str,
        slots: dict[str, Any] | None = None,
        *,
        on_started: Callable[[ThinkerLifecycleEvent], Awaitable[None]] | None = None,
        continue_active: bool = False,
    ) -> dict[str, Any]:
        """Cancel superseded work and suppress stale results.

        With ``continue_active`` and a run still working, this call takes over
        that run instead of starting another: the previous caller ends at once
        as superseded and this caller receives the run's single result.
        """
        del slots
        clean_query = query.strip()
        if not clean_query:
            return planner_failure()
        call_id = uuid.uuid4().hex[:12]
        self._observe_consent_turn()
        run = self._active_run
        if continue_active and run is not None and run.running:
            self.state.active_call_id = call_id
            future = run.adopt(on_started)
            continued = ThinkerLifecycleEvent(marker="ThinkerContinued", call_id=run.run_id, query=clean_query)
            self.state.add_event(continued)
            if on_started:
                await on_started(continued)
        else:
            previous = self.state.active_task
            if previous is not None and not previous.done():
                superseded_id = self._active_run.run_id if self._active_run is not None else None
                if superseded_id is not None:
                    self._superseded_runs.add(superseded_id)
                # Let the superseded plan finish instead of discarding it. A caller
                # who adds a detail or says "okay" mid-thought used to destroy the
                # round already running, and the next turn began again at step
                # one -- so a plan that needs a lookup before it can act never got
                # to act. Its speech is still dropped (a newer turn owns the
                # microphone); only its work and what it learned survive.
                self._detached.add(previous)
                previous.add_done_callback(self._detached.discard)
                # A consent question is never carried across a changed request.
                self._consent.expire_for_new_request(self._latest_user_text())
                if superseded_id is not None:
                    self._detached_runs[superseded_id] = previous
                    previous.add_done_callback(lambda _task, run_id=superseded_id: self._forget_run(run_id))
                    self._enforce_detached_cap()
            self.state.active_call_id = call_id
            if self._ledger is not None:
                self._ledger.open(call_id, clean_query)
            started = ThinkerLifecycleEvent(marker="ThinkerStarted", call_id=call_id, query=clean_query)
            self.state.add_event(started)
            if on_started:
                await on_started(started)
            approved = self._take_approved_write(clean_query)
            run = DelegationRun(
                call_id,
                clean_query,
                lambda progress: self._run_call(call_id, clean_query, on_progress=progress, approved_call=approved),
                on_started,
            )
            future = run.attach()
            self._active_run = run
            self.state.active_task = run.task
        try:
            payload = self._with_write_outcome(run.run_id, await run.wait(future))
            if self.state.active_call_id != call_id:
                late = await self._late_answer(run.run_id, payload)
                if late is None:
                    self._retain_superseded_question(run.run_id, payload, run.query)
                    raise asyncio.CancelledError
                self._retain_question(late, run.query)
                return late
            if payload.get("reason") == "no_action_needed" and self._late_answers_waiting:
                # A superseded run finished with a question or a client result
                # while this turn needed nothing; let that answer be the one
                # spoken instead of "nothing further" followed by it.
                return {**payload, "response_text": "", "speakable": False}
            self._retain_question(payload, run.query)
            return payload
        except asyncio.CancelledError:
            self.state.add_event(
                ThinkerLifecycleEvent(marker="ThinkerAborted", call_id=call_id, query=clean_query, reason="cancelled")
            )
            raise
        finally:
            if run.owns(future) and self.state.active_task is run.task:
                self.state.active_task = None
                self.state.active_call_id = None
                self._active_run = None

    def cancel_active(self, reason: str = "new_user_query") -> bool:
        """Cancel and immediately invalidate the active task generation."""
        # A withdrawal also withdraws what superseded runs would still say,
        # even when nothing is running any more.
        self._owner_gone.update(self._superseded_runs)
        self.clear_pending_question("cancelled")
        self._consent.expire_all("withdrawn")
        task = self.state.active_task
        if task is None or task.done():
            return False
        call_id = self.state.active_call_id
        logger.info(f"Generic Thinker call {call_id or '(unknown)'} cancelled: {reason}")
        self.state.active_call_id = None
        phase = self._run_phase.get(call_id or "")
        if call_id is not None and phase == "tools_write":
            # A dispatched write cannot be undone by stopping: let it finish and report it.
            self._withdrawn_writes.add(call_id)
            self._owner_gone.add(call_id)
            self._stop_after_tools.add(call_id)
            logger.bind(event="withdrawal", run_id=call_id, outcome="write_pending").info(
                "Withdrawal leaves a dispatched write to finish and be reported"
            )
            return True
        if call_id is not None and phase == "tools":
            # A pending read finishes silently into session memory; planning stops after it.
            self._owner_gone.add(call_id)
            self._stop_after_tools.add(call_id)
            logger.bind(event="withdrawal", run_id=call_id, outcome="read_pending").info(
                "Withdrawal stops the run after its read returns"
            )
            return True
        task.cancel()
        return True

    def take_pending_question(self) -> dict[str, Any] | None:
        """Return the clarification question still owed to the caller, or None.

        Called when the caller only acknowledges or asks for progress. The
        question was asked by a run whose answer the caller did not hear in
        full: it was superseded, or the caller spoke over it. Offering it again
        replaces another filler and another identical lookup. It is offered at
        most twice, and never while a backend request is running.
        """
        pending = self._pending_question
        if pending is None or self.running_query() is not None:
            return None
        if asyncio.get_running_loop().time() - pending.created_at > _PENDING_QUESTION_TTL_SECONDS:
            self.clear_pending_question("expired")
            return None
        if pending.redelivered >= _PENDING_QUESTION_MAX_REDELIVERIES:
            self.clear_pending_question("redelivery_limit")
            return None
        pending.redelivered += 1
        logger.bind(event="pending_question", tool=pending.tool, redelivered=pending.redelivered).info(
            f"Pending clarification question offered again: tool={pending.tool} count={pending.redelivered}"
        )
        return dict(pending.payload)

    def clear_pending_question(self, reason: str) -> None:
        """Forget the pending clarification question, if any."""
        if self._pending_question is None:
            return
        logger.bind(event="pending_question", tool=self._pending_question.tool, cleared=reason).info(
            f"Pending clarification question cleared: {reason}"
        )
        self._pending_question = None

    def repeats_pending_request(self, query: str) -> bool:
        """Return whether ``query`` only repeats the request whose clarification question is still pending."""
        pending = self._pending_question
        return pending is not None and bool(pending.query) and _comparable(pending.query) == _comparable(query)

    def has_pending_confirmation(self) -> bool:
        """Return whether a consent question still awaits the caller's answer."""
        if self._ledger is None:
            return False
        confirmation = self._ledger.latest_confirmation()
        return confirmation is not None and confirmation.outcome == "pending"

    def _retain_question(self, payload: Mapping[str, Any], query: str = "") -> None:
        """Keep a code-written clarification question for a later progress check."""
        if not self._pending_question_enabled or payload.get("type") != "response_hint":
            return
        if payload.get("reason") not in _PENDING_QUESTION_REASONS or payload.get("speakable") is False:
            return
        tool = str(payload.get("context") or "")
        text = str(payload.get("response_text") or "").strip()
        if not tool or tool == "call_backend" or not text:
            return
        current = self._pending_question
        if current is not None and current.tool == tool and current.payload.get("response_text") == text:
            # The same question asked again keeps its count, so it cannot loop.
            return
        self._pending_question = _PendingQuestion(
            payload=dict(payload), tool=tool, created_at=asyncio.get_running_loop().time(), query=query
        )

    def _retain_superseded_question(self, run_id: str, payload: Mapping[str, Any], query: str = "") -> None:
        """Keep a superseded run's question unless newer work covered its tool or was withdrawn."""
        if run_id in self._owner_gone:
            return
        tool = _late_answer_tool(payload)
        if tool is not None and self._ledger is not None:
            blocker = self._late_answer_blocker(run_id, tool, payload)
            if blocker in {"withdrawn", "later_write", "same_tool_later"}:
                return
        self._retain_question(payload, query)

    def is_late_answer(self, run_id: str | None) -> bool:
        """Return whether ``run_id`` was returned to its caller as a late answer."""
        return run_id in self._late_delivered

    async def _late_answer(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Return a superseded run's answer when it is still worth speaking, else None.

        The run's own ``call_backend`` call still owns the answer; it is spoken
        only when nothing newer has covered the same ground. Newer calls settle
        first. Only a question for the caller or a successful client result
        qualifies, and never when a newer run produced speech, touched the same
        tool or wrote anything.
        """
        self._superseded_runs.discard(run_id)
        if run_id in self._withdrawn_writes:
            return self._withdrawn_write_report(run_id, payload)
        if not self._late_answers_enabled or self._ledger is None or run_id in self._owner_gone:
            return None
        tool = _late_answer_tool(payload)
        if tool is None:
            return None
        self._late_answers_waiting.add(run_id)
        try:
            if not await self._await_no_running_call():
                return None
        finally:
            self._late_answers_waiting.discard(run_id)
        verdict = self._late_answer_blocker(run_id, tool, payload)
        logger.bind(event="late_answer", run_id=run_id, tool=tool, delivered=verdict is None, reason=verdict).info(
            f"Superseded run answer {'delivered' if verdict is None else 'dropped: ' + verdict}"
        )
        if verdict is not None:
            return None
        if payload.get("type") == "response_hint":
            self._last_late_question_tool = tool
        self._late_delivered.add(run_id)
        return payload

    def _withdrawn_write_report(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Report what a withdrawn run's dispatched write did; the caller must hear it."""
        self._withdrawn_writes.discard(run_id)
        writes = self._ledger.entry_writes(run_id) if self._ledger is not None else []
        delivered = bool(writes)
        logger.bind(event="late_answer", run_id=run_id, delivered=delivered, reason="withdrawn_write").info(
            f"Withdrawn run's write {'reported' if delivered else 'not reported: no write recorded'}"
        )
        if not delivered:
            return None
        self._late_delivered.add(run_id)
        return payload

    async def _await_no_running_call(self) -> bool:
        """Wait, bounded, until no newer backend call is still running."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _LATE_ANSWER_MAX_WAIT_SECONDS
        while True:
            task = self.state.active_task
            if task is None or task.done():
                return True
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            await asyncio.wait({task}, timeout=remaining)

    def _late_answer_blocker(self, run_id: str, tool: str, payload: Mapping[str, Any]) -> str | None:
        """Return why a superseded run's answer must not be spoken, or None when it may be."""
        assert self._ledger is not None
        if run_id in self._owner_gone:
            return "withdrawn"
        if payload.get("type") == "response_hint" and self._last_late_question_tool == tool:
            return "repeated_question"
        for later in self._ledger.runs_after(run_id):
            calls = self._ledger.entry_calls(later)
            if any(call.kind == "write" for call in calls):
                return "later_write"
            if any(call.tool == tool for call in calls):
                return "same_tool_later"
            outcome = self._ledger.outcome(later)
            if outcome is None:
                return "later_run_open"
            result, reason, context = outcome
            if context == tool:
                return "same_tool_later"
            if result not in {"cancelled"} and reason != "no_action_needed":
                return "later_speech"
        return None

    def cancel_pending_work(self) -> bool:
        """Generic tools have no draft state outside the active call."""
        return False

    def cancel_pending_booking(self) -> bool:
        """Retain compatibility with older shared-handler test doubles."""
        return self.cancel_pending_work()

    async def _plan_with_retry(
        self,
        call_id: str,
        query: str,
        *,
        planning_round: int,
        prior_tool_results: list[dict[str, Any]],
        validation_error: str | None = None,
    ) -> dict[str, Any]:
        """Retry one transient planner failure inside the existing overall deadline.

        ``validation_error`` tells a re-plan why the previous plan of this round
        was rejected. It carries the dispatcher's message only, never argument values.
        """
        _prior_tools = ",".join(str(entry.get("tool")) for entry in prior_tool_results)
        logger.debug(
            f"plan-in call={call_id[:8]} round={planning_round} "
            f"prior={len(prior_tool_results)} tools={_prior_tools} query={query[:160]!r}"
        )
        for attempt in range(1, _PLANNER_MAX_ATTEMPTS + 1):
            state: dict[str, Any] = {
                "active_call_id": call_id,
                "planner_attempt": attempt,
                "planning_round": planning_round,
                "max_planning_rounds": self._max_planning_rounds,
                "prior_tool_results": prior_tool_results,
            }
            if validation_error is not None:
                state["validation_error"] = validation_error
            try:
                plan = await self._await_plan(
                    self._planner.plan(query=query, state=state, **self._history_argument(call_id))
                )
            except asyncio.CancelledError:
                self._log_round(call_id, planning_round, attempt, "cancelled")
                raise
            except Exception as exc:
                self._log_round(call_id, planning_round, attempt, _round_failure_outcome(exc))
                if attempt >= _PLANNER_MAX_ATTEMPTS or not _is_retriable_planner_error(exc):
                    raise
                backoff = _PLANNER_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    f"Generic Thinker planner transient failure: attempt={attempt}/{_PLANNER_MAX_ATTEMPTS} "
                    f"error={type(exc).__name__}: {exc}; retrying in {backoff:.1f}s"
                )
                await asyncio.sleep(backoff)
                continue
            self._log_round(call_id, planning_round, attempt, "ok")
            return plan
        raise AssertionError("planner retry loop exited unexpectedly")

    def _log_round(self, call_id: str, planning_round: int, attempt: int, outcome: str) -> None:
        """Log how one Thinker planning attempt ended, with its prompt size when known."""
        usage = getattr(self._planner, "last_usage", None)
        prompt_tokens = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
        logger.bind(
            event="thinker_round",
            call_id=call_id,
            planning_round=planning_round,
            attempt=attempt,
            outcome=outcome,
            prompt_tokens=prompt_tokens,
        ).info(f"Thinker round {planning_round} attempt {attempt}: {outcome}")

    async def _await_plan(self, plan: Awaitable[dict[str, Any]]) -> dict[str, Any]:
        """Await one planning attempt under the fixed limit or the stream-liveness limits."""
        if self._planner_liveness is None:
            return await asyncio.wait_for(plan, timeout=self._planner_timeout_seconds)
        first_chunk_seconds, stall_seconds = self._planner_liveness
        loop = asyncio.get_running_loop()
        last_activity: list[float] = []

        def touch() -> None:
            last_activity[:] = [loop.time()]

        token = STREAM_ACTIVITY.set(touch)
        try:
            # The task copies the current context, so the planner's stream sees ``touch``.
            task = asyncio.ensure_future(plan)
        finally:
            STREAM_ACTIVITY.reset(token)
        started = loop.time()
        ceiling = started + self._planner_timeout_seconds
        try:
            while True:
                deadline = min(
                    ceiling, last_activity[0] + stall_seconds if last_activity else started + first_chunk_seconds
                )
                remaining = deadline - loop.time()
                if remaining <= 0:
                    elapsed = loop.time() - started
                    if deadline == ceiling:
                        raise PlannerCeilingError(f"Thinker stream reached its ceiling after {elapsed:.1f}s")
                    reason = "stalled" if last_activity else "did not start"
                    raise TimeoutError(f"Thinker stream {reason} after {elapsed:.1f}s")
                done, _pending = await asyncio.wait({task}, timeout=remaining)
                if done:
                    return task.result()
        finally:
            if not task.done():
                task.cancel()
                # Collect the abandoned attempt's outcome without swallowing a
                # cancellation aimed at this caller.
                await asyncio.gather(task, return_exceptions=True)

    def _settle(self, accumulated_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Close one delegation without inventing a failure that did not happen.

        A turn can legitimately need no new tool: the caller said "okay", or the
        session already holds what was asked for. That used to reach an empty
        combine and surface as "I couldn't complete that request reliably" --
        a plain untruth, and the most common thing the caller heard.
        """
        if accumulated_results:
            return combine_accumulated_results(accumulated_results)
        if self._session_tool_memory:
            return combine_accumulated_results(list(self._session_tool_memory))
        return nothing_further()

    def _prior_results_for_round(self, accumulated_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Supplement this delegation's own results with what the session already knows.

        Anything this call has fetched wins outright: the carried copy of the
        same tool is older by definition and must not appear twice.
        """
        fetched = {_memory_key(entry) for entry in accumulated_results}
        carried = [entry for entry in self._session_tool_memory if _memory_key(entry) not in fetched]
        return [*carried, *accumulated_results]

    def _clear_answered_question(self, result: Mapping[str, Any]) -> None:
        """A successful call of the tool a pending question was about answers it."""
        pending = self._pending_question
        if (
            pending is not None
            and result.get("type") == "tool_result"
            and result.get("status") == "success"
            and result.get("tool") == pending.tool
        ):
            self.clear_pending_question("answered")

    def _remember_session_result(self, payload: Mapping[str, Any]) -> None:
        """Keep one successful lookup per distinct call for later turns.

        A later call with the same arguments replaces the earlier entry, so a
        record re-read after it changed supersedes the stale copy rather than
        joining it.
        """
        if payload.get("type") != "tool_result" or str(payload.get("status") or "") != "success":
            return
        tool = str(payload.get("tool") or "")
        if tool not in self._client_tools:
            return
        if self.client_tool_kind(tool) == "write":
            # A successful change makes every earlier read possibly stale.
            self._session_tool_memory = [
                entry for entry in self._session_tool_memory if self.client_tool_kind(str(entry.get("tool"))) == "write"
            ]
        key = _memory_key(payload)
        self._session_tool_memory = [entry for entry in self._session_tool_memory if _memory_key(entry) != key]
        self._session_tool_memory.append(dict(payload))
        del self._session_tool_memory[:-_SESSION_TOOL_MEMORY_LIMIT]

    async def _run_call(
        self,
        call_id: str,
        query: str,
        *,
        on_progress: Callable[[ThinkerLifecycleEvent], Awaitable[None]] | None = None,
        approved_call: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        accumulated_results: list[dict[str, Any]] = []
        seen_client_calls = self._seen_client_calls
        checks = _CallChecks()
        try:
            async with asyncio.timeout(self._overall_timeout_seconds):
                for planning_round in range(1, self._max_planning_rounds + 1):
                    if call_id in self._stop_after_tools:
                        if call_id in self._withdrawn_writes:
                            break
                        raise asyncio.CancelledError
                    plan, round_payload, result_count_before_dispatch = await self._plan_and_dispatch(
                        call_id,
                        query,
                        planning_round,
                        accumulated_results,
                        seen_client_calls,
                        preset_plan=approved_call if planning_round == 1 else None,
                        checks=checks,
                    )
                    if round_payload is None:
                        break
                    if len(accumulated_results) == result_count_before_dispatch:
                        accumulated_results.append(round_payload)
                    for result in accumulated_results[result_count_before_dispatch:]:
                        self._argument_screen.add_result_hints(
                            result, is_read=self.client_tool_kind(str(result.get("tool") or "")) == "read"
                        )
                    # One entry per call: a multi-call round's combined payload
                    # would otherwise never be remembered.
                    for result in accumulated_results[result_count_before_dispatch:]:
                        self._remember_session_result(result)
                        self._clear_answered_question(result)
                    self._record_reads(call_id, accumulated_results[result_count_before_dispatch:])
                    self._record_confirmation(call_id, round_payload)
                    if self._record_local_answer(call_id, round_payload):
                        # A question for the caller: nothing more can be planned until they answer.
                        break
                    requests_follow_up = _requests_follow_up(plan, self._client_tools)
                    if requests_follow_up and planning_round < self._max_planning_rounds:
                        progress = ThinkerLifecycleEvent(
                            marker="IntermediateResponse",
                            call_id=call_id,
                            query=query,
                            payload=combine_accumulated_results(accumulated_results),
                        )
                        self.state.add_event(progress)
                        if on_progress is not None:
                            await on_progress(progress)
                    if not requests_follow_up:
                        break
                    if planning_round == self._max_planning_rounds:
                        logger.warning(
                            "Generic Thinker reached the configured planning-round limit: "
                            f"rounds={self._max_planning_rounds}"
                        )
                payload = self._with_thinker_text(self._settle(accumulated_results), checks)
        except asyncio.CancelledError:
            self._end_run_state(call_id)
            if self._ledger is not None:
                self._ledger.cancelled(call_id)
            raise
        except TimeoutError:
            logger.warning("Generic Thinker exhausted its bounded planner/overall deadline")
            payload = combine_accumulated_results(accumulated_results) if accumulated_results else timeout_failure()
        except PlanValidationError as exc:
            # Every other branch here logs. This one did not, so a rejected plan
            # reached the speaker as "I couldn't complete that request reliably"
            # with nothing on the server saying which rule rejected it.
            logger.warning(f"Generic Thinker plan rejected: {exc}")
            payload = self._settle(accumulated_results)
        except Exception as exc:  # noqa: BLE001 - planner boundary fails closed
            logger.warning(f"Generic Thinker planning failed: {type(exc).__name__}: {exc}")
            payload = combine_accumulated_results(accumulated_results) if accumulated_results else planner_failure()
        self.state.add_event(
            ThinkerLifecycleEvent(marker="IntermediateResponse", call_id=call_id, query=query, payload=payload)
        )
        self.state.add_event(
            ThinkerLifecycleEvent(marker="ThinkerCompleted", call_id=call_id, query=query, payload=payload)
        )
        self._end_run_state(call_id)
        if self._ledger is not None:
            self._ledger.close(call_id, payload)
        return payload

    def _end_run_state(self, call_id: str) -> None:
        self._run_phase.pop(call_id, None)
        self._stop_after_tools.discard(call_id)

    def _with_thinker_text(self, payload: dict[str, Any], checks: _CallChecks) -> dict[str, Any]:
        """Attach the Thinker's final answer as a draft, or its refusal reason, to the closing payload."""
        if checks.draft_answer and payload.get("type") == "tool_result":
            return {**payload, "draft_answer": checks.draft_answer}
        if checks.refusal_reason:
            return {**payload, "refusal_reason": checks.refusal_reason}
        return payload

    async def _plan_and_dispatch(
        self,
        call_id: str,
        query: str,
        planning_round: int,
        accumulated_results: list[dict[str, Any]],
        seen_client_calls: set[str],
        *,
        preset_plan: dict[str, Any] | None = None,
        checks: _CallChecks | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, int]:
        """Plan and execute one round; a rejected plan is re-planned once with the reason.

        Returns the plan, the round's payload (None for a completion plan) and
        how many results existed before dispatch. A plan is rejected before any
        side effect, so the re-plan starts from the same state. ``preset_plan``
        is an approved call issued without planning; if it no longer validates,
        the round is planned normally.

        A refusal, a handoff or a refusal-labelled answer may be sent back once
        per backend call with a code-written reason (:mod:`.plan_checks`); that
        uses the round's single re-plan, and the second plan is accepted.
        """
        checks = checks if checks is not None else _CallChecks()
        validation_error: str | None = None
        rechecking: tuple[tuple[str, ...], dict[str, Any]] | None = None
        while True:
            prior = self._prior_results_for_round(accumulated_results)
            self._run_phase[call_id] = "planning"
            if preset_plan is not None:
                plan, preset_plan = preset_plan, None
                preset = True
            else:
                preset = False
                plan = await self._plan_with_retry(
                    call_id,
                    query,
                    planning_round=planning_round,
                    prior_tool_results=prior,
                    validation_error=validation_error,
                )
            _plan_text = json.dumps(plan)[:240]
            logger.debug(f"plan-out call={call_id[:8]} round={planning_round} plan={_plan_text}")
            if rechecking is not None:
                plan = self._after_check(rechecking, plan, query, prior, accumulated_results, checks)
                rechecking = None
            elif not preset and validation_error is None:
                check = self._plan_check(plan, query, prior, accumulated_results, checks)
                if check is not None:
                    kinds, reason = check
                    checks.used = True
                    rechecking = (kinds, plan)
                    validation_error = reason
                    self._log_round(call_id, planning_round, 0, "plan_rechecked")
                    continue
            if _is_completion_plan(plan):
                self._take_completion_text(plan, prior, checks)
                return plan, None, len(accumulated_results)
            plan = self._gate_writes(plan)
            result_count_before_dispatch = len(accumulated_results)
            self._run_phase[call_id] = "tools_write" if self._client_writes(plan) else "tools"
            try:
                round_payload = await dispatch_plan(
                    plan,
                    self._tools,
                    self._enabled_tools,
                    source_query=query,
                    on_tool_started=self._on_tool_started,
                    stage_metrics=self._stage_metrics,
                    backend_call_id=call_id,
                    accumulated_results=accumulated_results,
                    tool_ordinal_offset=len(accumulated_results),
                    client_tools=self._client_tools,
                    client_tool_executor=self._client_tool_executor,
                    client_tool_timeout_seconds=self._client_tool_timeout_seconds,
                    seen_client_calls=seen_client_calls,
                    argument_screen=self._argument_screen,
                    running_client_calls=self._running_client_calls,
                )
            except PlanValidationError as exc:
                if preset:
                    logger.warning(f"Approved call no longer validates; planning instead: {exc}")
                    continue
                self._log_round(call_id, planning_round, 0, "plan_rejected")
                if validation_error is not None:
                    raise
                validation_error = str(exc)[:200]
                continue
            finally:
                self._run_phase[call_id] = "planning"
            if self._client_writes(plan):
                self._consent.write_dispatched()
            respell = self._argument_screen.respell_question(round_payload, query)
            if respell is not None:
                logger.bind(event="respell_question", tool=respell.get("context")).info(
                    f"Not-found identifier answered with a re-spell question: tool={respell.get('context')}"
                )
                round_payload = respell
            return plan, round_payload, result_count_before_dispatch

    # -- one-time plan checks (refusals, handoffs, refusal-labelled answers) --

    def _client_tool_infos(self) -> list[plan_checks.ToolInfo]:
        return [
            plan_checks.ToolInfo(spec.name, spec.description, self.client_tool_kind(spec.name))
            for spec in self._client_tools.values()
            if spec.name in self._enabled_tools
        ]

    def _is_handoff_call(self, name: str) -> bool:
        spec = self._client_tools.get(name)
        return spec is not None and plan_checks.is_handoff_tool(
            plan_checks.ToolInfo(spec.name, spec.description, self.client_tool_kind(spec.name))
        )

    def _client_writes(self, plan: Mapping[str, Any]) -> list[tuple[str, Any]]:
        """Return the plan's client write calls; a handoff is not one (the handoff check governs it)."""
        return [
            (name, params)
            for name, params in plan_checks.planned_calls(plan)
            if name in self._client_tools and self.client_tool_kind(name) == "write" and not self._is_handoff_call(name)
        ]

    def _session_instructions(self) -> str:
        context = getattr(self._planner, "session_instruction_context", None)
        messages = context.get_messages() if context is not None else []
        return str(messages[0].get("content") or "") if messages and isinstance(messages[0], Mapping) else ""

    def _latest_user_text(self) -> str:
        transcript = self._ledger.transcript if self._ledger is not None else None
        return transcript.latest_user_text() if transcript is not None else ""

    def _plan_check(
        self,
        plan: dict[str, Any],
        query: str,
        prior: list[dict[str, Any]],
        accumulated_results: list[dict[str, Any]],
        checks: _CallChecks,
    ) -> tuple[tuple[str, ...], str] | None:
        """Return the check kinds and code-written reason to re-plan with once, or None to accept the plan."""
        if not self._plan_checks_enabled or not self._client_tools:
            return None
        try:
            quote = plan_checks.policy_quote(plan)
        except plan_checks.MalformedBasisError as exc:
            return ("basis",), f"Rejected basis: {exc}."
        refusal = plan_checks.is_refusal(plan)
        handoffs = [name for name, _params in plan_checks.planned_calls(plan) if self._is_handoff_call(name)]
        if not refusal and not handoffs:
            return None
        if checks.used:
            self._log_check("handoff_check" if handoffs else "refusal_check", "bypassed:check_used")
            return None
        quoted = quote is not None and plan_checks.quote_in_instructions(quote, self._session_instructions())
        tools = self._client_tool_infos()
        if refusal:
            kinds: list[str] = []
            reasons: list[str] = []
            if prior:
                if " ".join(str(plan.get("response_text") or "").split()) == plan_checks.PLACEHOLDER_REFUSAL_TEXT:
                    reasons.append(plan_checks.PLACEHOLDER_REASON)
                kinds.append("answer")
                reasons.append(plan_checks.ANSWER_CHECK_REASON)
            bypass = self._refusal_bypass(query, quoted, accumulated_results)
            candidates: list[str] = []
            if bypass is None:
                writes = [tool for tool in tools if not plan_checks.is_handoff_tool(tool)]
                candidates = plan_checks.candidate_tools(query, writes, writes_only=True)
                bypass = None if candidates else "no_candidate"
            if bypass is None:
                kinds.append("refusal")
                reasons.append(plan_checks.refusal_check_reason(candidates))
            else:
                self._log_check("refusal_check", f"bypassed:{bypass}")
            return (tuple(kinds), " ".join(reasons)) if kinds else None
        bypass = self._handoff_bypass(query, quoted)
        candidates = []
        if bypass is None:
            change = plan_checks.is_change_request(query)
            pool = [tool for tool in tools if not plan_checks.is_handoff_tool(tool) and (tool.kind == "read" or change)]
            candidates = plan_checks.candidate_tools(query, pool, writes_only=False)
            bypass = None if candidates else "no_candidate"
        if bypass is not None:
            self._log_check("handoff_check", f"bypassed:{bypass}")
            return None
        return ("handoff",), plan_checks.handoff_check_reason(candidates)

    def _refusal_bypass(self, query: str, quoted: bool, accumulated_results: list[dict[str, Any]]) -> str | None:
        """Return why a refusal is accepted without the refusal check, or None to check it."""
        if not self._consent_gate_enabled:
            # A correction away from refusing must never reach an unconsented write.
            return "no_consent_gate"
        if not plan_checks.is_change_request(query):
            return "not_a_change"
        if quoted:
            return "policy_quote"
        for result in accumulated_results:
            tool = str(result.get("tool") or "")
            is_write = tool in self._client_tools and self.client_tool_kind(tool) == "write"
            if is_write and result.get("status") == "error":
                return "tool_error"
        return None

    def _handoff_bypass(self, query: str, quoted: bool) -> str | None:
        """Return why a handoff goes ahead without the handoff check, or None to check it."""
        if not self._consent_gate_enabled:
            return "no_consent_gate"
        if plan_checks.asks_for_person(self._latest_user_text()) or plan_checks.asks_for_person(query):
            return "person_request"
        if quoted:
            return "policy_quote"
        return None

    def _after_check(
        self,
        rechecking: tuple[tuple[str, ...], dict[str, Any]],
        plan: dict[str, Any],
        query: str,
        prior: list[dict[str, Any]],
        accumulated_results: list[dict[str, Any]],
        checks: _CallChecks,
    ) -> dict[str, Any]:
        """Log how a re-checked plan came back, and return the plan to carry out."""
        kinds, original = rechecking
        writes = self._client_writes(plan)
        if "handoff" in kinds:
            kept = any(self._is_handoff_call(name) for name, _params in plan_checks.planned_calls(plan))
            self._log_check("handoff_check", "rechecked:kept" if kept else "rechecked:changed")
            return plan
        if "basis" in kinds:
            return plan
        if plan_checks.is_refusal(plan):
            outcome, refusal_outcome = "refused", "kept"
            text = " ".join(str(plan.get("response_text") or "").split())
            if prior and text and text != plan_checks.PLACEHOLDER_REFUSAL_TEXT:
                checks.refusal_reason = text[:_MAX_THINKER_TEXT_CHARS]
        elif plan_checks.completion_text(plan):
            outcome, refusal_outcome = "answered", "changed_to_answer"
        elif writes and not self._consent_gate_enabled:
            # Without the consent gate a write from a re-plan is never dispatched.
            outcome, refusal_outcome = "write_blocked", "kept"
            plan = original
        elif plan_checks.planned_calls(plan):
            outcome, refusal_outcome = "call", "changed_to_call"
        else:
            outcome, refusal_outcome = "question", "changed_to_question"
        if "answer" in kinds:
            wrote = any(
                self.client_tool_kind(str(result.get("tool") or "")) == "write"
                for result in accumulated_results
                if str(result.get("tool") or "") in self._client_tools
            )
            logger.bind(
                event="answer_check",
                outcome=outcome,
                change_request=plan_checks.is_change_request(query),
                wrote=wrote,
            ).info(f"Answer check: {outcome}")
        if "refusal" in kinds:
            self._log_check("refusal_check", f"rechecked:{refusal_outcome}")
        return plan

    def _take_completion_text(self, plan: Mapping[str, Any], prior: list[dict[str, Any]], checks: _CallChecks) -> None:
        """Keep a final answer's text as the Talker's draft, only when it can rest on tool results."""
        text = plan_checks.completion_text(plan)
        if not text or not self._plan_checks_enabled:
            return
        if not prior:
            logger.bind(event="answer_check", outcome="draft_ignored").info("Final-answer text without tool results")
            return
        checks.draft_answer = text[:_MAX_THINKER_TEXT_CHARS]

    @staticmethod
    def _log_check(event: str, outcome: str) -> None:
        logger.bind(event=event, outcome=outcome).info(f"{event}: {outcome}")

    # -- consent gate --

    def _gate_writes(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Return the plan to dispatch: unchanged when every client write is approved, else a consent question.

        Each write must match its own granted approval (same tool, identical
        canonical arguments). If any lacks one, none of the plan's writes is
        sent and the caller is asked about the first unapproved write.
        """
        writes = self._client_writes(plan)
        if not self._consent_gate_enabled or not writes:
            return plan
        approvals = []
        for name, params in writes:
            arguments = dict(params) if isinstance(params, Mapping) else {}
            try:
                fingerprint = client_call_fingerprint(name, arguments)
            except (TypeError, ValueError):
                return plan  # the dispatcher rejects the arguments before any side effect
            approval = self._consent.take(fingerprint, self._session_generation)
            if approval is None and self._write_consent == "policy" and self._policy_waives_consent(plan):
                logger.bind(event="consent_gate", tool=name, outcome="allowed", reason="policy_quote").info(
                    f"Consent gate allowed a write on a policy quote: tool={name}"
                )
                continue
            if approval is None:
                logger.bind(
                    event="consent_gate", tool=name, outcome="asked", arguments_sha=fingerprint_digest(fingerprint)
                ).info(f"Consent gate turned an unapproved write into a consent question: tool={name}")
                return {
                    "tool": "response_hint",
                    "reason": "confirmation_needed",
                    "action": "req_confirmation",
                    "context": name,
                    "params": arguments,
                }
            approvals.append((name, fingerprint, approval))
        for name, fingerprint, approval in approvals:
            self._consent.consume(approval)
            logger.bind(
                event="consent_gate", tool=name, outcome="allowed", arguments_sha=fingerprint_digest(fingerprint)
            ).info(f"Consent gate allowed an approved write: tool={name}")
        return plan

    def _policy_waives_consent(self, plan: Mapping[str, Any]) -> bool:
        try:
            quote = plan_checks.policy_quote(plan)
        except plan_checks.MalformedBasisError:
            return False
        return quote is not None and plan_checks.quote_in_instructions(quote, self._session_instructions())

    def _observe_consent_turn(self, turn: int | None = None, utterance: str | None = None) -> None:
        """Grant or expire write approvals as of the caller's latest turn."""
        transcript = self._ledger.transcript if self._ledger is not None else None
        if not self._consent_gate_enabled or transcript is None:
            return
        reply = transcript.reply_before_latest_user()
        self._consent.observe_turn(
            turn=transcript.user_turns if turn is None else turn,
            utterance=transcript.latest_user_text() if utterance is None else utterance,
            reply_completed=None if reply is None else reply[1] == "completed",
            generation=self._session_generation,
        )

    # -- superseded runs --

    def _forget_run(self, run_id: str) -> None:
        self._detached_runs.pop(run_id, None)
        self._run_phase.pop(run_id, None)
        self._stop_after_tools.discard(run_id)

    def _enforce_detached_cap(self) -> None:
        """Keep at most the configured number of superseded runs planning; stop the oldest beyond it.

        A run with a client write pending is never stopped and does not
        count. A run in a read round stops once the read returns, so its
        result still reaches session memory.
        """
        counted: list[tuple[str, asyncio.Task[dict[str, Any]], str]] = []
        for run_id, task in self._detached_runs.items():
            if task.done() or run_id in self._stop_after_tools:
                continue
            phase = self._run_phase.get(run_id, "planning")
            if phase == "tools_write":
                logger.bind(event="detached_cap_exempt_write", run_id=run_id).info(
                    "Superseded run with a pending write is exempt from the cap"
                )
                continue
            counted.append((run_id, task, phase))
        for run_id, task, phase in counted[: max(0, len(counted) - _DETACHED_RUN_CAP)]:
            self._owner_gone.add(run_id)
            if phase == "tools":
                self._stop_after_tools.add(run_id)
            else:
                task.cancel()
            logger.bind(event="detached_cap", run_id=run_id, phase=phase).info(
                f"Superseded run stopped by the detached-run cap: phase={phase}"
            )

    def _history_argument(self, call_id: str) -> dict[str, Any]:
        """Return the planner's ``history`` keyword only when there is history to send."""
        if self._ledger is None:
            return {}
        history = self._ledger.render(call_id)
        return {} if history is None else {"history": history}

    def _record_confirmation(self, call_id: str, payload: Mapping[str, Any]) -> None:
        """Record a consent question with the exact call and text it asked about."""
        if self._ledger is None or payload.get("reason") != "confirmation_needed":
            return
        params = payload.get("params_resolved")
        record = self._ledger.confirmation(
            call_id,
            str(payload.get("context") or ""),
            params if isinstance(params, Mapping) else {},
            str(payload.get("response_text") or ""),
            summarized=bool(payload.get("summarized", True)),
        )
        if record is not None:
            record.generation = self._session_generation
        transcript = self._ledger.transcript
        tool = str(payload.get("context") or "")
        if self._consent_gate_enabled and transcript is not None and tool in self._client_tools:
            try:
                fingerprint = client_call_fingerprint(tool, params if isinstance(params, Mapping) else {})
            except (TypeError, ValueError):
                return
            self._consent.record(
                tool=tool,
                fingerprint=fingerprint,
                run_id=call_id,
                generation=self._session_generation,
                turn=transcript.user_turns,
            )

    def _with_write_outcome(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Tell the Talker exactly which writes of this run succeeded, or that nothing changed.

        The Talker may state completion of a listed tool only. A question or
        failure with no successful write carries a fixed notice instead, and
        code-authored question text that claims completion is replaced.
        """
        if not self._done_guard_enabled or self._ledger is None or not self._client_tools:
            return payload
        written = [
            call.tool
            for call in self._ledger.entry_writes(run_id)
            if call.state == "confirmed" and call.status == "success"
        ]
        guarded = dict(payload)
        if written:
            guarded["completed_actions"] = list(dict.fromkeys(written))
            return guarded
        guarded["no_change_notice"] = NO_CHANGE_NOTICE
        if guarded.get("type") == "response_hint":
            text = str(guarded.get("response_text") or "")
            replaced = COMPLETION_CLAIM_RE.sub("", text)
            if replaced != text:
                logger.bind(event="done_guard", run_id=run_id).warning("Removed a completion claim with no write")
                guarded["response_text"] = " ".join(replaced.split()) or "I'm still working on that."
        return guarded

    def _take_approved_write(self, query: str) -> dict[str, Any] | None:
        """Return the confirmed call to issue without a new plan, or None to plan as usual.

        The newest consent question is approved only when every condition
        holds: it stated every value in full; the message that carried it
        completed and the caller can have heard all of it (a truncation cuts
        the journal text); the very next user turn is a bare "yes"; and no
        other turn, write or session update came in between. An approval is
        used once; any later turn retires it.
        """
        if self._ledger is None or not self._direct_write_enabled:
            return None
        confirmation = self._ledger.latest_confirmation()
        if confirmation is None or confirmation.outcome != "pending":
            return None
        blocker = self._direct_write_blocker(confirmation)
        confirmation.outcome = "expired" if blocker is not None else "consumed"
        logger.bind(
            event="direct_write",
            tool=confirmation.tool,
            approved=blocker is None,
            reason=blocker or "approved",
            query_chars=len(query),
        ).info(f"Confirmed call {'issued' if blocker is None else 'not issued: ' + blocker}")
        if blocker is not None:
            return None
        return {"tool": confirmation.tool, "params": dict(confirmation.params)}

    def _direct_write_blocker(self, confirmation: Any) -> str | None:
        """Return why a bare "yes" cannot approve ``confirmation``, or None when it does."""
        assert self._ledger is not None
        if confirmation.summarized:
            return "summarized"
        if confirmation.tool not in self._client_tools or self.client_tool_kind(confirmation.tool) != "write":
            return "not_a_client_write"
        if confirmation.generation != self._session_generation:
            return "session_updated"
        transcript = self._ledger.transcript
        reply = transcript.reply_before_latest_user() if transcript is not None else None
        if reply is None:
            return "not_observable"
        text, status, user_turns = reply
        if status != "completed":
            return "response_not_completed"
        if user_turns != 1:
            return "other_turn"
        if _comparable(confirmation.text) not in _comparable(text):
            return "not_heard_in_full"
        if _comparable(transcript.latest_user_text()) not in BARE_AFFIRMATIVES:
            return "not_a_bare_yes"
        for later in self._ledger.runs_after(confirmation.run_id):
            if any(call.kind == "write" for call in self._ledger.entry_calls(later)):
                return "other_write"
        return None

    def _record_local_answer(self, call_id: str, payload: Mapping[str, Any]) -> bool:
        """Ledger a read that was answered locally (unfinished identifier); return whether one was."""
        local = payload.get("answered_locally")
        if not isinstance(local, Mapping):
            return False
        if self._ledger is not None:
            self._ledger.read(call_id, local.get("tool"), local.get("arguments"), "error", state="answered_locally")
        return True

    def _record_reads(self, call_id: str, payloads: list[dict[str, Any]]) -> None:
        """Record server-tool results; client calls are already recorded as writes at the executor."""
        if self._ledger is None:
            return
        for payload in payloads:
            data = payload.get("data")
            if payload.get("type") != "tool_result" or not isinstance(data, dict) or data.get("owner") == "client":
                continue
            self._ledger.read(call_id, payload.get("tool"), data.get("arguments"), payload.get("status"))


def _argument_screen_for(client_tools: Sequence[Mapping[str, Any]]) -> ArgumentScreen:
    """Build one session's identifier screen; read per session so a switch change applies to the next."""
    return ArgumentScreen.for_tools(
        client_tools,
        enabled=flag("FRONTEND_BACKEND_NORMALIZATION"),
        phone_hints=flag("FRONTEND_BACKEND_PHONE_FORMAT"),
    )


def _log_argument_screen(screen: ArgumentScreen) -> None:
    logger.bind(
        event="schema_rules",
        enabled=screen.enabled,
        session_tools_sha256=screen.tools_sha256,
        snapshot=screen.rules.snapshot(),
    ).info(f"Session tool rules: tools_sha256={screen.tools_sha256[:12]}")


NO_CHANGE_NOTICE = (
    "No change has been made. Do not say that anything was done, booked, cancelled, updated or is being processed."
)
#: Whole completion phrases; amounts and identifiers around them are never touched.
COMPLETION_CLAIM_RE = re.compile(
    r"(?i)[^.!?]*\b(?:has been (?:updated|cancelled|canceled|booked|changed|processed)|is booked|"
    r"is being processed|you're all set|you are all set|it's done|it is done)\b[^.!?]*[.!?]?"
)

#: Whole utterances that approve a confirmed call on their own.
BARE_AFFIRMATIVES = frozenset(
    {"yes", "yes please", "yeah", "go ahead", "do it", "that's right", "thats right", "correct", "please do", "sure"}
)
_COMPARABLE_DROP_RE = re.compile(r"[^\w\s']")


def _comparable(text: object) -> str:
    """Case-fold, drop punctuation and collapse spaces, for comparing spoken text."""
    return " ".join(_COMPARABLE_DROP_RE.sub(" ", str(text or "").casefold().replace("’", "'")).split())


def _late_answer_tool(payload: Mapping[str, Any]) -> str | None:
    """Return the tool a deliverable late answer is about, or None when it is not deliverable.

    Deliverable: a question for the caller (missing or invalid details, a
    consent question) or a successful client-tool result.
    """
    if payload.get("type") == "response_hint":
        if payload.get("reason") in {"params_missing", "params_invalid", "confirmation_needed"}:
            return str(payload.get("context") or "") or None
        return None
    data = payload.get("data")
    if (
        payload.get("type") == "tool_result"
        and payload.get("status") == "success"
        and isinstance(data, Mapping)
        and data.get("owner") == "client"
    ):
        return str(payload.get("tool") or "") or None
    return None


def _task_cancellation_requested() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def _requests_follow_up(
    plan: Mapping[str, Any],
    client_tools: Mapping[str, ClientToolSpec] | None = None,
) -> bool:
    """Continue explicit dependent work and finalize implicit client-tool results."""
    requested = plan.get("continue_after_results")
    if requested is not None:
        return requested is True
    client_names = frozenset(client_tools or ())
    if not client_names:
        return False
    raw_calls = plan.get("tool_calls")
    if raw_calls is None and plan.get("tool"):
        raw_calls = [plan]
    if not isinstance(raw_calls, list):
        return False
    return any(isinstance(call, Mapping) and call.get("tool") in client_names for call in raw_calls)


def _is_completion_plan(plan: Mapping[str, Any]) -> bool:
    """Treat explicit completion or a plan with no executable call as complete."""
    return plan.get("complete") is True or ("tool" not in plan and not plan.get("tool_calls"))


@dataclass(slots=True)
class _CallChecks:
    """One backend call's plan-check state: at most one check, and what the Thinker wrote."""

    used: bool = False
    draft_answer: str = ""
    refusal_reason: str = ""


def _write_consent_mode() -> str:
    """Return ``always`` (every client write needs an approval) or ``policy`` (a verified quote may waive it)."""
    raw = os.getenv("FRONTEND_BACKEND_WRITE_CONSENT", "always").strip().lower()
    return raw if raw in {"always", "policy"} else "always"


def fingerprint_digest(fingerprint: str) -> str:
    """Return a short digest of a call fingerprint for logs; never the argument values."""
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:12]


@dataclass
class _PendingQuestion:
    """One clarification question the caller has not answered yet."""

    payload: dict[str, Any]
    tool: str
    created_at: float
    redelivered: int = 0
    #: The delegated request that asked the question.
    query: str = ""


def _memory_key(entry: Mapping[str, Any]) -> str:
    """Identify one remembered call by its tool and exact arguments."""
    data = entry.get("data")
    arguments = data.get("arguments") if isinstance(data, Mapping) else None
    if not isinstance(arguments, Mapping):
        arguments = {}
    try:
        return client_call_fingerprint(str(entry.get("tool") or ""), arguments)
    except (TypeError, ValueError):
        return f"{entry.get('tool')}\0<unencodable>"


def _round_failure_outcome(exc: BaseException) -> str:
    """Name how one planning attempt failed, for the per-round log."""
    if isinstance(exc, PlanTruncatedError):
        return "truncated"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ValueError):
        return "parse_error"
    return "error"


def _recording_client_executor(
    executor: ClientToolRoundExecutor,
    ledger: DelegationLedger,
    kind_of: Callable[[str], str] = lambda _name: "write",
) -> ClientToolRoundExecutor:
    """Record each client call where its round is published and awaited.

    Writes are recorded as started before publication, so a write whose output
    never arrives still appears; reads are recorded once their status is known.
    """

    async def execute(
        calls: tuple[tuple[str, dict[str, Any]], ...],
        timeout_secs: float,
    ) -> list[str | dict[str, Any]]:
        run_id = current_run_id()
        kinds = [kind_of(name) for name, _arguments in calls]
        records = [
            ledger.write_started(run_id, name, arguments) if kind == "write" else None
            for kind, (name, arguments) in zip(kinds, calls, strict=True)
        ]
        try:
            outputs = await executor(calls, timeout_secs)
        except BaseException:
            for kind, record, (name, arguments) in zip(kinds, records, calls, strict=True):
                if kind == "write":
                    ledger.write_unconfirmed(record)
                else:
                    ledger.read(run_id, name, arguments, "unavailable")
            raise
        for kind, record, (name, arguments), output in zip(kinds, records, calls, outputs, strict=False):
            timed_out = _client_tool_timed_out(output)
            if kind != "write":
                status = "unavailable" if timed_out else format_client_result(name, arguments, output).get("status")
                ledger.read(run_id, name, arguments, status)
            elif timed_out:
                # The client may still run a call it received after the deadline.
                ledger.write_unconfirmed(record)
            else:
                ledger.write_confirmed(record, format_client_result(name, arguments, output).get("status"))
        return outputs

    return execute


def _client_tool_timed_out(output: object) -> bool:
    error = output.get("error") if isinstance(output, dict) else None
    return isinstance(error, dict) and error.get("code") == "client_tool_timeout"
