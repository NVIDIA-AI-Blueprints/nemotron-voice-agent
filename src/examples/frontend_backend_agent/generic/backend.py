# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Session-local planner/executor backend for the generic domain."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from loguru import logger
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

from examples.frontend_backend_agent.generic.dispatcher import (
    PlanValidationError,
    combine_accumulated_results,
    dispatch_plan,
)
from examples.frontend_backend_agent.generic.planner import GenericPlanner
from examples.frontend_backend_agent.generic.result_formatters import planner_failure, timeout_failure
from examples.frontend_backend_agent.generic.state import GenericThinkerSessionState
from examples.frontend_backend_agent.src.protocol import ThinkerLifecycleEvent
from examples.frontend_backend_agent.src.tools import ToolSpec

if TYPE_CHECKING:
    from examples.frontend_backend_agent.src.stage_metrics import StageMetricsCoordinator

_PLANNER_MAX_ATTEMPTS = 2
_PLANNER_RETRY_BACKOFF_SECONDS = 0.2
_RETRIABLE_PLANNER_EXCEPTIONS = (TimeoutError, APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
_MAX_PLANNING_ROUNDS = 3


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
        overall_timeout_seconds: float = 40.0,
        planner_timeout_seconds: float = 6.0,
        state: GenericThinkerSessionState | None = None,
        on_tool_started: Callable[[str], Awaitable[None]] | None = None,
        stage_metrics: StageMetricsCoordinator | None = None,
    ) -> None:
        """Create a backend with bounded planner and end-to-end deadlines."""
        self._planner = planner
        self._tools = dict(tools)
        self._enabled_tools = enabled_tools
        self._overall_timeout_seconds = max(1.0, overall_timeout_seconds)
        self._planner_timeout_seconds = min(max(1.0, planner_timeout_seconds), self._overall_timeout_seconds)
        self._on_tool_started = on_tool_started
        self._stage_metrics = stage_metrics
        self.state = state or GenericThinkerSessionState()

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
            except _RETRIABLE_PLANNER_EXCEPTIONS as exc:
                if attempt >= _PLANNER_MAX_ATTEMPTS:
                    raise
                logger.warning(
                    f"Generic Thinker planner transient failure: attempt={attempt}/{_PLANNER_MAX_ATTEMPTS} "
                    f"error={type(exc).__name__}; retrying once"
                )
                await asyncio.sleep(_PLANNER_RETRY_BACKOFF_SECONDS)
        raise AssertionError("planner retry loop exited unexpectedly")

    async def _run_call(
        self,
        call_id: str,
        query: str,
        *,
        on_progress: Callable[[ThinkerLifecycleEvent], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        accumulated_results: list[dict[str, Any]] = []
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
            logger.warning(f"Generic Thinker planning failed: {type(exc).__name__}")
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
