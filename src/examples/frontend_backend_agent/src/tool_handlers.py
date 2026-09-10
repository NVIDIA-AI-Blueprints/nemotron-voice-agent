# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Pipecat function handlers for the frontend/backend-agent tools."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.services.llm_service import FunctionCallResultProperties

from examples.frontend_backend_agent.src.protocol import ThinkerLifecycleEvent, is_speakable_payload, response_hint
from examples.frontend_backend_agent.src.runtime_context import runtime_today

_ISO_DATE_PATTERN = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")
_NAMED_DATE_PATTERN = re.compile(
    r"\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*|\s+)(\d{4})\b",
    re.IGNORECASE,
)
_MAX_PLANNER_ERROR_ATTEMPTS = 2

if TYPE_CHECKING:
    from pipecat.services.llm_service import FunctionCallParams


class ThinkerBackend(Protocol):
    """Minimal runtime interface required by the frontend tool handlers."""

    async def call(
        self,
        query: str,
        slots: dict[str, Any] | None = None,
        *,
        on_started: Callable[[ThinkerLifecycleEvent], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Run one Thinker invocation."""

    def cancel_active(self, reason: str = "new_user_query") -> bool:
        """Cancel any active Thinker invocation."""

    def cancel_pending_booking(self) -> bool:
        """Cancel pending domain work that has no active task."""


def build_handlers(thinker: ThinkerBackend, *, filler_threshold_seconds: float = 0.8) -> dict[str, Callable]:
    """Return tool handlers bound to one session-local backend agent."""
    consecutive_planner_errors = 0

    async def handle_call_backend(params: FunctionCallParams) -> None:
        nonlocal consecutive_planner_errors
        arguments = _normalize_arguments(params.arguments or {})
        query = str(arguments.get("query", "") or "").strip()
        if not query:
            consecutive_planner_errors = 0
            await params.result_callback(
                {
                    "type": "response_hint",
                    "reason": "params_missing",
                    "action": "req_params",
                    "params_needed": ["query"],
                    "response_text": "What would you like me to check?",
                    "context": "call_backend",
                }
            )
            return
        past_date = _past_date_in_query(query)
        if past_date is not None:
            consecutive_planner_errors = 0
            payload = response_hint(
                reason="past_date",
                action="request_future_date",
                response_text=(
                    f"{past_date.strftime('%B')} {past_date.day}, {past_date.year} has already passed. "
                    "Please provide a future travel date."
                ),
                context="flight_search",
            )
            await _emit_terminal_payload(params, payload)
            return
        try:
            filler_text = str(arguments.get("filler_text", "") or "").strip()
            slots = {key: value for key, value in arguments.items() if key not in {"query", "intent", "filler_text"}}
            filler_task: asyncio.Task | None = None
            filler_started = False

            async def emit_filler_after_threshold() -> None:
                try:
                    await asyncio.sleep(filler_threshold_seconds)
                    await _emit_talker_response(params.llm, filler_text)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(f"Failed to emit Talker filler: {exc}")

            async def schedule_thinker_started_filler(event: ThinkerLifecycleEvent) -> None:
                nonlocal filler_started, filler_task
                if event.marker != "ThinkerStarted" or not filler_text:
                    return
                if filler_started or (filler_task is not None and not filler_task.done()):
                    return
                filler_started = True
                if filler_threshold_seconds <= 0:
                    await _emit_talker_response(params.llm, filler_text)
                    return
                filler_task = asyncio.create_task(emit_filler_after_threshold())

            try:
                payload = await thinker.call(query, slots=slots, on_started=schedule_thinker_started_filler)
            finally:
                await _cancel_pending_filler(filler_task)
        except asyncio.CancelledError:
            consecutive_planner_errors = 0
            logger.info("call_backend result suppressed after Thinker abort")
            await params.result_callback(
                {
                    "type": "response_hint",
                    "reason": "aborted",
                    "action": "internal_abort",
                    "response_text": "",
                    "context": "call_backend",
                    "speakable": False,
                },
                properties=FunctionCallResultProperties(run_llm=False),
            )
            return
        except Exception as exc:
            consecutive_planner_errors = 0
            logger.exception(f"call_backend failed before producing a result: {exc}")
            await params.result_callback(
                {
                    "type": "response_hint",
                    "reason": "tool_error",
                    "action": "retry",
                    "error": str(exc),
                    "response_text": "I could not complete that request right now. Please try again.",
                    "context": "call_backend",
                }
            )
            return
        if payload.get("reason") == "planner_error":
            consecutive_planner_errors += 1
            if consecutive_planner_errors >= _MAX_PLANNER_ERROR_ATTEMPTS:
                logger.warning(f"Planner failed {_MAX_PLANNER_ERROR_ATTEMPTS} consecutive times; ending retries")
                terminal_payload = dict(payload)
                terminal_payload.update(
                    {
                        "reason": "planner_error_exhausted",
                        "action": "answer_directly",
                        "response_text": (
                            "I could not process that request after a few attempts. Please try again later."
                        ),
                    }
                )
                await _emit_terminal_payload(params, terminal_payload)
                consecutive_planner_errors = 0
                return
        else:
            consecutive_planner_errors = 0
        if _direct_tool_response_enabled() and is_speakable_payload(payload):
            await _emit_talker_response(params.llm, str(payload.get("response_text") or ""))
            await params.result_callback(payload, properties=FunctionCallResultProperties(run_llm=False))
            return
        await params.result_callback(payload)

    async def handle_cancel_backend(params: FunctionCallParams) -> None:
        nonlocal consecutive_planner_errors
        consecutive_planner_errors = 0
        cancelled = thinker.cancel_active("user_cancelled")
        cleared_pending_booking = thinker.cancel_pending_booking()
        did_cancel = cancelled or cleared_pending_booking
        payload = {
            "type": "response_hint",
            "reason": "cancelled" if did_cancel else "nothing_to_cancel",
            "action": "cancelled" if did_cancel else "nothing_to_cancel",
            "response_text": "Okay, I stopped that." if did_cancel else "There is nothing pending right now.",
            "context": "cancel_backend",
        }
        if _direct_tool_response_enabled():
            await _emit_talker_response(params.llm, str(payload["response_text"]))
            await params.result_callback(payload, properties=FunctionCallResultProperties(run_llm=False))
            return
        await params.result_callback(payload)

    return {"call_backend": handle_call_backend, "cancel_backend": handle_cancel_backend}


async def _emit_talker_response(llm, text: str) -> None:
    """Emit Talker-authored filler through the normal LLM text/TTS path."""
    if _task_cancellation_requested():
        return
    started = False
    try:
        started = True
        await llm.push_frame(LLMFullResponseStartFrame())
        await llm.push_frame(LLMTextFrame(text=text))
    finally:
        if started:
            await llm.push_frame(LLMFullResponseEndFrame())


async def _emit_terminal_payload(params: FunctionCallParams, payload: dict[str, Any]) -> None:
    """Speak a validated terminal payload without asking the Talker to reinterpret it."""
    await _emit_talker_response(params.llm, str(payload.get("response_text") or ""))
    await params.result_callback(payload, properties=FunctionCallResultProperties(run_llm=False))


async def _cancel_pending_filler(task: asyncio.Task | None) -> None:
    """Cancel a delayed filler if the Thinker returned before it fired."""
    if task is None or task.done():
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


def _normalize_arguments(arguments: dict) -> dict:
    """Recover from LLMs that wrap the tool payload under ``original_args``."""
    original_args = arguments.get("original_args")
    if isinstance(original_args, str) and "query" not in arguments:
        try:
            decoded = json.loads(original_args)
        except json.JSONDecodeError:
            return arguments
        if isinstance(decoded, dict):
            return decoded
    return arguments


def _past_date_in_query(query: str, *, today: date | None = None) -> date | None:
    """Return a past ISO travel date when the query contains no future date.

    The Talker contract supplies known travel dates as ISO values. If a correction
    contains both an old and a new date, the future date wins and the Thinker still
    receives the request.
    """
    dates: list[date] = []
    for match in _ISO_DATE_PATTERN.finditer(query):
        try:
            dates.append(date.fromisoformat(match.group(1)))
        except ValueError:
            continue
    for match in _NAMED_DATE_PATTERN.finditer(query):
        candidate = " ".join(match.groups())
        for date_format in ("%B %d %Y", "%b %d %Y"):
            try:
                dates.append(datetime.strptime(candidate, date_format).date())
                break
            except ValueError:
                continue
    if not dates:
        return None
    today = today or runtime_today()
    if any(value >= today for value in dates):
        return None
    return max(dates)


def _direct_tool_response_enabled() -> bool:
    return os.getenv("FRONTEND_BACKEND_DIRECT_TOOL_RESPONSE", "").strip().lower() in {"1", "true", "yes", "on"}


def _task_cancellation_requested() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0
