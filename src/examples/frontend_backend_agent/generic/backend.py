# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Session-local planner/executor backend for the generic domain."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger
from openai import APIConnectionError, APIError, APITimeoutError, InternalServerError, RateLimitError

from examples.frontend_backend_agent.generic.argument_screen import ArgumentScreen
from examples.frontend_backend_agent.generic.client_tools import (
    ClientToolRoundExecutor,
    ClientToolSpec,
    build_client_tool_specs,
    format_client_result,
)
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
from examples.frontend_backend_agent.src.tools import ToolSpec, session_server_tools
from utils import parse_env_float, parse_env_int

if TYPE_CHECKING:
    from examples.frontend_backend_agent.src.stage_metrics import StageMetricsCoordinator

_PLANNER_MAX_ATTEMPTS = parse_env_int("GENERIC_PLANNER_MAX_ATTEMPTS", 2, min_value=1)
# How many distinct lookups one session keeps for later turns.
_SESSION_TOOL_MEMORY_LIMIT = 6
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


def _is_retriable_planner_error(exc: BaseException) -> bool:
    """Return whether one planner failure is worth another attempt."""
    if isinstance(exc, _RETRIABLE_PLANNER_EXCEPTIONS):
        return True
    if isinstance(exc, APIError):
        message = str(exc).lower()
        return any(fragment in message for fragment in _TRANSIENT_PLANNER_ERROR_TEXT)
    return False


_DEFAULT_MAX_PLANNING_ROUNDS = 8
#: How long a superseded run's answer waits for newer calls to settle.
_LATE_ANSWER_MAX_WAIT_SECONDS = 20.0


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
        self._max_planning_rounds = max(1, max_planning_rounds)
        self._on_tool_started = on_tool_started
        self._stage_metrics = stage_metrics
        # Session-scoped so a client call that keeps failing stays suppressed across
        # turns; the frontend re-delegates per turn, and per-call state would reset.
        self._seen_client_calls: set[str] = set()
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
        self._superseded_runs: set[str] = set()
        self._owner_gone: set[str] = set()
        self._late_answers_waiting: set[str] = set()
        self._late_delivered: set[str] = set()
        self._last_late_question_tool: str | None = None
        self._direct_write_enabled = flag("FRONTEND_BACKEND_DIRECT_WRITE")
        self._done_guard_enabled = flag("FRONTEND_BACKEND_DONE_GUARD")
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
                if self._active_run is not None:
                    self._superseded_runs.add(self._active_run.run_id)
                # Let the superseded plan finish instead of discarding it. A caller
                # who adds a detail or says "okay" mid-thought used to destroy the
                # round already running, and the next turn began again at step
                # one -- so a plan that needs a lookup before it can act never got
                # to act. Its speech is still dropped (a newer turn owns the
                # microphone); only its work and what it learned survive.
                self._detached.add(previous)
                previous.add_done_callback(self._detached.discard)
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
                    raise asyncio.CancelledError
                return late
            if payload.get("reason") == "no_action_needed" and self._late_answers_waiting:
                # A superseded run finished with a question or a client result
                # while this turn needed nothing; let that answer be the one
                # spoken instead of "nothing further" followed by it.
                return {**payload, "response_text": "", "speakable": False}
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
        task = self.state.active_task
        if task is None or task.done():
            return False
        logger.info(f"Generic Thinker call {self.state.active_call_id or '(unknown)'} cancelled: {reason}")
        self.state.active_call_id = None
        task.cancel()
        return True

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
                plan = await asyncio.wait_for(
                    self._planner.plan(query=query, state=state, **self._history_argument(call_id)),
                    timeout=self._planner_timeout_seconds,
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
        fetched = {entry.get("tool") for entry in accumulated_results}
        carried = [entry for entry in self._session_tool_memory if entry.get("tool") not in fetched]
        return [*carried, *accumulated_results]

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
        self._session_tool_memory = [entry for entry in self._session_tool_memory if entry.get("tool") != tool]
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
        try:
            async with asyncio.timeout(self._overall_timeout_seconds):
                for planning_round in range(1, self._max_planning_rounds + 1):
                    plan, round_payload, result_count_before_dispatch = await self._plan_and_dispatch(
                        call_id,
                        query,
                        planning_round,
                        accumulated_results,
                        seen_client_calls,
                        preset_plan=approved_call if planning_round == 1 else None,
                    )
                    if round_payload is None:
                        break
                    if len(accumulated_results) == result_count_before_dispatch:
                        accumulated_results.append(round_payload)
                    for result in accumulated_results[result_count_before_dispatch:]:
                        self._argument_screen.add_result_hints(
                            result, is_read=self.client_tool_kind(str(result.get("tool") or "")) == "read"
                        )
                    self._remember_session_result(round_payload)
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
                payload = self._settle(accumulated_results)
        except asyncio.CancelledError:
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
        if self._ledger is not None:
            self._ledger.close(call_id, payload)
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
    ) -> tuple[dict[str, Any], dict[str, Any] | None, int]:
        """Plan and execute one round; a rejected plan is re-planned once with the reason.

        Returns the plan, the round's payload (None for a completion plan) and
        how many results existed before dispatch. A plan is rejected before any
        side effect, so the re-plan starts from the same state. ``preset_plan``
        is an approved call issued without planning; if it no longer validates,
        the round is planned normally.
        """
        validation_error: str | None = None
        while True:
            if preset_plan is not None:
                plan, preset_plan = preset_plan, None
                preset = True
            else:
                preset = False
                plan = await self._plan_with_retry(
                    call_id,
                    query,
                    planning_round=planning_round,
                    prior_tool_results=self._prior_results_for_round(accumulated_results),
                    validation_error=validation_error,
                )
            _plan_text = json.dumps(plan)[:240]
            logger.debug(f"plan-out call={call_id[:8]} round={planning_round} plan={_plan_text}")
            if _is_completion_plan(plan):
                return plan, None, len(accumulated_results)
            result_count_before_dispatch = len(accumulated_results)
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
            return plan, round_payload, result_count_before_dispatch

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
