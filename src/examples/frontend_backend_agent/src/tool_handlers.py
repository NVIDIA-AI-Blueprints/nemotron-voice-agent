# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Pipecat function handlers for the frontend/backend-agent tools."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.services.llm_service import FunctionCallResultProperties

from examples.frontend_backend_agent.src.protocol import ThinkerLifecycleEvent, is_speakable_payload

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

    async def handle_call_backend(params: FunctionCallParams) -> None:
        arguments = _normalize_arguments(params.arguments or {})
        query = str(arguments.get("query", "") or "").strip()
        if not query:
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
        if _direct_tool_response_enabled() and is_speakable_payload(payload):
            await _emit_talker_response(params.llm, str(payload.get("response_text") or ""))
            await params.result_callback(payload, properties=FunctionCallResultProperties(run_llm=False))
            return
        await params.result_callback(payload)

    async def handle_cancel_backend(params: FunctionCallParams) -> None:
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


def _direct_tool_response_enabled() -> bool:
    return os.getenv("FRONTEND_BACKEND_DIRECT_TOOL_RESPONSE", "").strip().lower() in {"1", "true", "yes", "on"}


def _task_cancellation_requested() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0
