# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Session-local planner/executor backend for the generic domain."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger
from openai import APIConnectionError, APIError, APITimeoutError, InternalServerError, RateLimitError

from examples.frontend_backend_agent.generic.client_tools import (
    ClientToolRoundExecutor,
    ClientToolSpec,
    build_client_tool_specs,
)
from examples.frontend_backend_agent.generic.dispatcher import (
    PlanValidationError,
    combine_accumulated_results,
    dispatch_plan,
)
from examples.frontend_backend_agent.generic.planner import GenericPlanner, GenericPlannerSessionUpdate
from examples.frontend_backend_agent.generic.result_formatters import planner_failure, timeout_failure
from examples.frontend_backend_agent.generic.state import GenericThinkerSessionState
from examples.frontend_backend_agent.src.protocol import ThinkerLifecycleEvent
from examples.frontend_backend_agent.src.tools import ToolSpec
from utils import parse_env_float, parse_env_int

if TYPE_CHECKING:
    from examples.frontend_backend_agent.src.stage_metrics import StageMetricsCoordinator

_PLANNER_MAX_ATTEMPTS = parse_env_int("GENERIC_PLANNER_MAX_ATTEMPTS", 2, min_value=1)
_PLANNER_RETRY_BACKOFF_SECONDS = parse_env_float("GENERIC_PLANNER_RETRY_BACKOFF_SECONDS", 0.2, min_value=0.0)
_RETRIABLE_PLANNER_EXCEPTIONS = (TimeoutError, APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
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


_MAX_PLANNING_ROUNDS = 3


@dataclass(frozen=True, slots=True)
class GenericBackendSessionUpdate:
    """Prepared planner and client-tool state for one session transaction."""

    planner: GenericPlannerSessionUpdate
    client_tools: dict[str, ClientToolSpec]
    enabled_tools: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GenericBackendSessionSnapshot:
    """Rollback snapshot for the dynamic Generic backend policy."""

    planner: GenericPlannerSessionUpdate
    client_tools: dict[str, ClientToolSpec]
    enabled_tools: tuple[str, ...]


class GenericThinkerBackend:
    """Run one bounded, replaceable backend task per voice session."""

    # Generic formatters already produce grounded, TTS-safe speech; avoid a second
    # Talker pass over Pipecat's asynchronous started/final result envelope.
    tool_result_mode_default = "direct"
    talker_result_tools = ("get_weather",)

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
        state: GenericThinkerSessionState | None = None,
        on_tool_started: Callable[[str], Awaitable[None]] | None = None,
        stage_metrics: StageMetricsCoordinator | None = None,
    ) -> None:
        """Create a backend with bounded planner and end-to-end deadlines."""
        self._planner = planner
        self._tools = dict(tools)
        self._client_tools = dict(client_tools or {})
        self._client_tool_executor = client_tool_executor
        self._client_tool_timeout_seconds = max(1.0, client_tool_timeout_seconds)
        self._server_enabled_tools = tuple(name for name in enabled_tools if name in self._tools)
        self._enabled_tools = enabled_tools
        self._overall_timeout_seconds = max(1.0, overall_timeout_seconds)
        self._planner_timeout_seconds = min(max(1.0, planner_timeout_seconds), self._overall_timeout_seconds)
        self._on_tool_started = on_tool_started
        self._stage_metrics = stage_metrics
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
            enabled_tools=(*self._server_enabled_tools, *client_specs),
        )

    def snapshot_session_update(self) -> GenericBackendSessionSnapshot:
        """Capture dynamic planner and client-tool state for rollback."""
        return GenericBackendSessionSnapshot(
            planner=self._planner.snapshot_session_update(),
            client_tools=dict(self._client_tools),
            enabled_tools=self._enabled_tools,
        )

    def commit_session_update(self, prepared: GenericBackendSessionUpdate) -> None:
        """Install one prepared dynamic planner and client-tool policy."""
        if not isinstance(prepared, GenericBackendSessionUpdate):
            raise TypeError("Generic backend session update has an invalid receipt")
        if self.state.active_task is not None and not self.state.active_task.done():
            raise RuntimeError("Generic planner policy cannot change during an active backend call")
        self._planner.commit_session_update(prepared.planner)
        self._client_tools = dict(prepared.client_tools)
        self._enabled_tools = prepared.enabled_tools

    def restore_session_update(self, snapshot: GenericBackendSessionSnapshot) -> None:
        """Restore dynamic planner and client-tool state after a failed commit."""
        if not isinstance(snapshot, GenericBackendSessionSnapshot):
            raise TypeError("Generic backend rollback has an invalid snapshot")
        self._planner.restore_session_update(snapshot.planner)
        self._client_tools = dict(snapshot.client_tools)
        self._enabled_tools = snapshot.enabled_tools

    async def call(
        self,
        query: str,
        slots: dict[str, Any] | None = None,
        *,
        on_started: Callable[[ThinkerLifecycleEvent], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Cancel superseded work and suppress stale results."""
        del slots
        clean_query = query.strip()
        if not clean_query:
            return planner_failure()
        previous = self.state.active_task
        if previous is not None and not previous.done():
            self.cancel_active("superseded")
            try:
                await previous
            except asyncio.CancelledError:
                if _task_cancellation_requested():
                    raise
            except Exception:  # noqa: BLE001 - the superseded result is intentionally discarded
                pass
        call_id = uuid.uuid4().hex[:12]
        self.state.active_call_id = call_id
        started = ThinkerLifecycleEvent(marker="ThinkerStarted", call_id=call_id, query=clean_query)
        self.state.add_event(started)
        if on_started:
            await on_started(started)
        task = asyncio.create_task(self._run_call(call_id, clean_query, on_progress=on_started))
        self.state.active_task = task
        try:
            payload = await task
            if self.state.active_call_id != call_id:
                raise asyncio.CancelledError
            return payload
        except asyncio.CancelledError:
            self.state.add_event(
                ThinkerLifecycleEvent(marker="ThinkerAborted", call_id=call_id, query=clean_query, reason="cancelled")
            )
            raise
        finally:
            if self.state.active_task is task:
                self.state.active_task = None
                self.state.active_call_id = None

    def cancel_active(self, reason: str = "new_user_query") -> bool:
        """Cancel and immediately invalidate the active task generation."""
        task = self.state.active_task
        if task is None or task.done():
            return False
        logger.info(f"Generic Thinker call {self.state.active_call_id or '(unknown)'} cancelled: {reason}")
        self.state.active_call_id = None
        task.cancel()
        return True

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
    ) -> dict[str, Any]:
        """Retry one transient planner failure inside the existing overall deadline."""
        for attempt in range(1, _PLANNER_MAX_ATTEMPTS + 1):
            try:
                return await asyncio.wait_for(
                    self._planner.plan(
                        query=query,
                        state={
                            "active_call_id": call_id,
                            "planner_attempt": attempt,
                            "planning_round": planning_round,
                            "max_planning_rounds": _MAX_PLANNING_ROUNDS,
                            "prior_tool_results": prior_tool_results,
                        },
                    ),
                    timeout=self._planner_timeout_seconds,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if attempt >= _PLANNER_MAX_ATTEMPTS or not _is_retriable_planner_error(exc):
                    raise
                backoff = _PLANNER_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    f"Generic Thinker planner transient failure: attempt={attempt}/{_PLANNER_MAX_ATTEMPTS} "
                    f"error={type(exc).__name__}: {exc}; retrying in {backoff:.1f}s"
                )
                await asyncio.sleep(backoff)
        raise AssertionError("planner retry loop exited unexpectedly")

    async def _run_call(
        self,
        call_id: str,
        query: str,
        *,
        on_progress: Callable[[ThinkerLifecycleEvent], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        accumulated_results: list[dict[str, Any]] = []
        seen_client_calls: set[str] = set()
        try:
            async with asyncio.timeout(self._overall_timeout_seconds):
                for planning_round in range(1, _MAX_PLANNING_ROUNDS + 1):
                    plan = await self._plan_with_retry(
                        call_id,
                        query,
                        planning_round=planning_round,
                        prior_tool_results=accumulated_results,
                    )
                    if _is_completion_plan(plan):
                        break
                    result_count_before_dispatch = len(accumulated_results)
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
                    )
                    if len(accumulated_results) == result_count_before_dispatch:
                        accumulated_results.append(round_payload)
                    if _requests_follow_up(plan) and planning_round < _MAX_PLANNING_ROUNDS:
                        progress = ThinkerLifecycleEvent(
                            marker="IntermediateResponse",
                            call_id=call_id,
                            query=query,
                            payload=combine_accumulated_results(accumulated_results),
                        )
                        self.state.add_event(progress)
                        if on_progress is not None:
                            await on_progress(progress)
                    if not _requests_follow_up(plan):
                        break
                payload = combine_accumulated_results(accumulated_results)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            logger.warning("Generic Thinker exhausted its bounded planner/overall deadline")
            payload = combine_accumulated_results(accumulated_results) if accumulated_results else timeout_failure()
        except PlanValidationError:
            payload = combine_accumulated_results(accumulated_results) if accumulated_results else planner_failure()
        except Exception as exc:  # noqa: BLE001 - planner boundary fails closed
            logger.warning(f"Generic Thinker planning failed: {type(exc).__name__}: {exc}")
            payload = combine_accumulated_results(accumulated_results) if accumulated_results else planner_failure()
        self.state.add_event(
            ThinkerLifecycleEvent(marker="IntermediateResponse", call_id=call_id, query=query, payload=payload)
        )
        self.state.add_event(
            ThinkerLifecycleEvent(marker="ThinkerCompleted", call_id=call_id, query=query, payload=payload)
        )
        return payload


def _task_cancellation_requested() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def _requests_follow_up(plan: Mapping[str, Any]) -> bool:
    """Honor only the Thinker's explicit, bounded request for another planning round."""
    return plan.get("continue_after_results") is True


def _is_completion_plan(plan: Mapping[str, Any]) -> bool:
    """Treat explicit completion or a plan with no executable call as complete."""
    return plan.get("complete") is True or ("tool" not in plan and not plan.get("tool_calls"))
