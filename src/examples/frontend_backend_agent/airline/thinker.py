# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Thinker implementation for the independent Frontend/Backend Agent example."""

from __future__ import annotations

import asyncio
import random
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from examples.frontend_backend_agent.airline.backend import BookingBackend
from examples.frontend_backend_agent.airline.booking_tool import BookingTool
from examples.frontend_backend_agent.airline.flight_search import flight_search
from examples.frontend_backend_agent.airline.plan_parsing import (
    combine_parallel_payloads,
    plan_tool_calls,
    string_list,
)
from examples.frontend_backend_agent.airline.pnr_status import pnr_status
from examples.frontend_backend_agent.airline.state import ThinkerSessionState
from examples.frontend_backend_agent.src.planner import ThinkerPlanner
from examples.frontend_backend_agent.src.protocol import ThinkerLifecycleEvent, response_hint

# Thinker tools that only read session state (or are pure). These are safe to
# run concurrently with each other and with the serialized mutating-tool chain.
# Everything else (flight_search, booking) writes shared session state and must
# be sequenced — see _dispatch_parallel_tool_calls.
_READ_ONLY_TOOLS = frozenset({"pnr_status", "response_hint"})


class ThinkerBackend:
    """Pluggable backend boundary behind the frontend LLM's ``call_backend`` tool.

    The frontend LLM provides a detailed natural-language query with conversation context.
    The Thinker planner LLM owns intent detection and parameter extraction from
    that query plus session state. This class validates the plan and executes
    the internal Python tools/backends.
    """

    def __init__(
        self,
        *,
        state: ThinkerSessionState | None = None,
        backend: BookingBackend | None = None,
        planner: ThinkerPlanner | None = None,
        tool_delay_seconds: float = 0.0,
        tool_delay_min_seconds: float | None = None,
        overall_timeout_seconds: float = 30.0,
        planner_timeout_seconds: float = 30.0,
    ) -> None:
        """Create a Thinker backend for one voice session."""
        if planner is None:
            raise ValueError("ThinkerBackend requires a ThinkerPlanner")
        if backend is None:
            raise ValueError("ThinkerBackend requires a BookingBackend")
        self.state = state or ThinkerSessionState()
        self._backend = backend
        self._planner = planner
        self._tool_delay_seconds = tool_delay_seconds
        self._tool_delay_min_seconds = tool_delay_min_seconds
        self._overall_timeout_seconds = max(1.0, overall_timeout_seconds)
        self._planner_timeout_seconds = min(max(1.0, planner_timeout_seconds), self._overall_timeout_seconds)

    async def call(
        self,
        query: str,
        slots: dict[str, Any] | None = None,
        *,
        on_started: Callable[[ThinkerLifecycleEvent], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Run one Thinker invocation and return a speakable protocol payload."""
        clean_query = query.strip()
        clean_slots = dict(slots or {})
        call_id = uuid.uuid4().hex[:12]
        previous_task = self.state.active_task
        if previous_task and not previous_task.done():
            self.cancel_active("new_thinker_call")
            try:
                await previous_task
            except asyncio.CancelledError:
                if _task_cancellation_requested():
                    raise
        self.state.active_call_id = call_id
        started_event = ThinkerLifecycleEvent(marker="ThinkerStarted", call_id=call_id, query=clean_query)
        self.state.add_event(started_event)
        if on_started:
            await on_started(started_event)
        task = asyncio.create_task(self._run_call(call_id, clean_query, clean_slots))
        self.state.active_task = task
        try:
            payload = await task
            if self.state.active_call_id != call_id:
                raise asyncio.CancelledError
        except asyncio.CancelledError:
            self.state.add_event(
                ThinkerLifecycleEvent(
                    marker="ThinkerAborted",
                    call_id=call_id,
                    query=clean_query,
                    reason="new_user_query",
                )
            )
            raise
        finally:
            if self.state.active_task is task:
                self.state.active_task = None
                self.state.active_call_id = None
        return payload

    def cancel_active(self, reason: str = "new_user_query") -> bool:
        """Abort the active Thinker task if one is running."""
        task = self.state.active_task
        if task is None or task.done():
            return False
        logger.info(f"Thinker call {self.state.active_call_id or '(unknown)'} aborted: {reason}")
        self.state.active_call_id = None
        task.cancel()
        return True

    def cancel_pending_booking(self) -> bool:
        """Clear pending search/booking context if one exists."""
        has_pending_work = (
            bool(self.state.search_context)
            or bool(self.state.search_results)
            or self.state.booking_draft is not None
            or self.state.waiting_for_preferences
            or self.state.waiting_for_confirmation
        )
        if has_pending_work:
            self.state.reset_search_and_booking()
        return has_pending_work

    def cancel_pending_work(self) -> bool:
        """Implement the domain-neutral cancellation contract."""
        return self.cancel_pending_booking()

    async def _run_call(self, call_id: str, query: str, slots: dict[str, Any]) -> dict[str, Any]:
        """Dispatch the query to internal Thinker tools."""
        try:
            async with asyncio.timeout(self._overall_timeout_seconds):
                delay_seconds = self._next_tool_delay_seconds()
                if delay_seconds > 0:
                    await asyncio.sleep(delay_seconds)
                payload = await self._dispatch(query, slots)
        except TimeoutError:
            payload = _timeout_failure()
        self.state.add_event(
            ThinkerLifecycleEvent(marker="IntermediateResponse", call_id=call_id, query=query, payload=payload)
        )
        self.state.add_event(
            ThinkerLifecycleEvent(marker="ThinkerCompleted", call_id=call_id, query=query, payload=payload)
        )
        return payload

    def _next_tool_delay_seconds(self) -> float:
        """Return the synthetic tool delay for this call."""
        max_delay = max(0.0, self._tool_delay_seconds)
        if max_delay <= 0:
            return 0.0
        if self._tool_delay_min_seconds is None:
            return max_delay
        min_delay = max(0.0, min(self._tool_delay_min_seconds, max_delay))
        return random.uniform(min_delay, max_delay)

    async def _dispatch(self, query: str, slots: dict[str, Any]) -> dict[str, Any]:
        try:
            plan = await asyncio.wait_for(
                self._planner.plan(query=query, slots=slots, state=self._planner_state()),
                timeout=self._planner_timeout_seconds,
            )
        except TimeoutError:
            return _timeout_failure()
        except Exception as exc:
            logger.warning(f"Thinker planner failed: {exc}")
            return response_hint(
                reason="planner_error",
                action="retry",
                response_text="I could not plan that request. Could you say it again?",
                context="general",
                error=str(exc),
            )

        tool_calls = plan_tool_calls(plan)
        if len(tool_calls) > 1:
            return await self._dispatch_parallel_tool_calls(slots, plan, tool_calls)
        if not tool_calls:
            return response_hint(
                reason="unsupported_request",
                action="answer_directly",
                response_text="I can help search flights, book a selected flight, or check a PNR status.",
                context="general",
            )
        return await self._dispatch_tool_call_safely(slots, tool_calls[0])

    async def _dispatch_parallel_tool_calls(
        self,
        slots: dict[str, Any],
        plan: dict[str, Any],
        tool_calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # State-mutating tools (flight_search, booking) write shared session
        # state (search_results / search_context / booking_draft). Running them
        # in a bare asyncio.gather() would interleave those writes
        # (last-writer-wins), so we serialize the mutating tools in planner order
        # while still running read-only tools concurrently with that chain. The
        # result list stays in planner order so combine_parallel_payloads is
        # unaffected.
        payloads: list[dict[str, Any] | None] = [None] * len(tool_calls)

        async def run_one(index: int, tool_call: dict[str, Any]) -> None:
            payloads[index] = await self._dispatch_tool_call_safely(slots, tool_call)

        async def run_mutating_chain(items: list[tuple[int, dict[str, Any]]]) -> None:
            for index, tool_call in items:
                await run_one(index, tool_call)

        mutating: list[tuple[int, dict[str, Any]]] = []
        coros: list[Awaitable[None]] = []
        for index, tool_call in enumerate(tool_calls):
            tool_name = str(tool_call.get("tool", "") or "").strip()
            if tool_name in _READ_ONLY_TOOLS:
                coros.append(run_one(index, tool_call))
            else:
                mutating.append((index, tool_call))
        if mutating:
            coros.append(run_mutating_chain(mutating))

        results = await asyncio.gather(*coros, return_exceptions=True)
        for result in results:
            if isinstance(result, asyncio.CancelledError) or _task_cancellation_requested():
                raise asyncio.CancelledError
            if isinstance(result, BaseException):
                logger.warning(f"parallel Thinker tool chain failed unexpectedly: {result}")
        return combine_parallel_payloads(plan, [payload for payload in payloads if payload is not None])

    async def _dispatch_tool_call_safely(
        self,
        slots: dict[str, Any],
        tool_call: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            return await self._dispatch_tool_call(slots, tool_call)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return _tool_exception_hint(tool_call, exc)

    async def _dispatch_tool_call(self, slots: dict[str, Any], tool_call: dict[str, Any]) -> dict[str, Any]:
        tool_name = str(tool_call.get("tool", "") or "").strip()
        planned_params = tool_call.get("params") if isinstance(tool_call.get("params"), dict) else {}
        planned_slots = {**slots, **planned_params}
        if tool_name == "booking":
            return await BookingTool(state=self.state, backend=self._backend).continue_booking(planned_slots)
        if tool_name == "pnr_status":
            return await pnr_status(backend=self._backend, slots=planned_slots)
        if tool_name == "flight_search":
            return await flight_search(state=self.state, backend=self._backend, slots=planned_slots)
        if tool_name == "response_hint":
            return self._planner_response_hint(tool_call)
        return response_hint(
            reason="unsupported_request",
            action="answer_directly",
            response_text="I can help search flights, book a selected flight, or check a PNR status.",
            context="general",
        )

    def _planner_state(self) -> dict[str, Any]:
        draft = self.state.booking_draft
        return {
            "search_context": self.state.search_context,
            "search_results": self.state.search_results,
            "booking_draft": {
                "flight": draft.flight,
                "seat_pref": draft.seat_pref,
                "meal_pref": draft.meal_pref,
                "passenger_name": draft.passenger_name,
            }
            if draft
            else None,
            "waiting_for_preferences": self.state.waiting_for_preferences,
            "waiting_for_confirmation": self.state.waiting_for_confirmation,
        }

    def _planner_response_hint(self, plan: dict[str, Any]) -> dict[str, Any]:
        return response_hint(
            reason=str(plan.get("reason") or "params_missing"),
            action=str(plan.get("action") or "req_params"),
            response_text=str(plan.get("response_text") or "Could you share the missing details?"),
            context=str(plan.get("context") or "general"),
            params_needed=string_list(plan.get("params_needed")),
            params_resolved=plan.get("params_resolved") if isinstance(plan.get("params_resolved"), dict) else None,
            error=str(plan.get("error")) if plan.get("error") is not None else None,
        )


def _timeout_failure() -> dict[str, Any]:
    """Return a bounded-deadline fallback without exposing internal timing."""
    return response_hint(
        reason="timeout",
        action="retry",
        response_text="That check took too long, so I stopped it. Would you like me to try again?",
        context="general",
    )


def _tool_exception_hint(tool_call: dict[str, Any], exc: Exception) -> dict[str, Any]:
    """Return a speakable fallback for unexpected tool failures."""
    tool_name = str(tool_call.get("tool") or "tool").strip() or "tool"
    logger.warning(f"Thinker tool {tool_name} failed: {exc}")
    return response_hint(
        reason="tool_error",
        action="retry",
        error=str(exc),
        response_text="I could not complete that request right now. Please try again.",
        context=tool_name,
    )


def _task_cancellation_requested() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0
