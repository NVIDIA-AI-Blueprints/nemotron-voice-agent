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
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.services.llm_service import FunctionCallResultProperties

from examples.frontend_backend_agent.src.protocol import ThinkerLifecycleEvent, is_speakable_payload

if TYPE_CHECKING:
    from pipecat.services.llm_service import FunctionCallParams

    from examples.frontend_backend_agent.src.domain import FillerPolicy
    from examples.frontend_backend_agent.src.stage_metrics import StageMetricsCoordinator


_FILLER_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_FILLER_TOKEN_RE = re.compile(r"[a-z0-9]+")
_FILLER_INTERNAL_RE = re.compile(
    r"\b(?:backend|function|hidden|llm|model|prompt|reasoning|system|tool)\b|"
    r"</?(?:think|tool_call|function|parameter)[^>]*>|```|https?://|www\.",
    re.IGNORECASE,
)
_FILLER_RESULT_CLAIM_RE = re.compile(
    r"\b(?:completed|done|finished|found|got|result|succeeded|successful|shows?|the answer is|turns out)\b",
    re.IGNORECASE,
)
_FILLER_PROGRESS_WORDS = frozenset(
    {
        "a",
        "about",
        "and",
        "for",
        "i",
        "it",
        "latest",
        "let",
        "look",
        "me",
        "please",
        "that",
        "the",
        "those",
        "to",
        "up",
        "verify",
        "will",
        "your",
    }
)


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

    def cancel_pending_work(self) -> bool:
        """Cancel pending domain state that has no active task."""


def build_handlers(
    thinker: ThinkerBackend,
    *,
    filler_threshold_seconds: float = 0.8,
    filler_policy: FillerPolicy = "planner_authored",
    filler_selector: Callable[[str], str] | None = None,
    interrupted_speech_consumer: Callable[[], bool] | None = None,
    max_query_chars: int = 4000,
    stage_metrics: StageMetricsCoordinator | None = None,
) -> dict[str, Callable]:
    """Return tool handlers bound to one session-local backend agent."""
    tool_result_mode_default = getattr(thinker, "tool_result_mode_default", "talker")

    async def handle_call_backend(params: FunctionCallParams) -> None:
        arguments = _normalize_arguments(params.arguments or {})
        query = str(arguments.get("query", "") or "").strip()
        if not query or len(query) > max_query_chars:
            await params.result_callback(
                {
                    "type": "response_hint",
                    "reason": "params_missing" if not query else "params_invalid",
                    "action": "req_params",
                    "params_needed": ["query"],
                    "response_text": "What would you like me to check?",
                    "context": "call_backend",
                }
            )
            return
        try:
            if filler_policy == "talker_authored":
                filler_text = _validated_talker_filler(query, arguments.get("filler_text"))
            elif filler_policy == "planner_authored":
                filler_text = " ".join(str(arguments.get("filler_text") or "").split()).strip()
            elif filler_policy == "code_authored":
                filler_text = filler_selector(query) if filler_selector is not None else ""
            else:
                raise ValueError(f"Unknown filler policy: {filler_policy}")
            filler_mode = _talker_filler_mode()
            if filler_policy == "talker_authored":
                logger.bind(
                    event="talker_filler_candidate",
                    mode=filler_mode,
                    accepted=bool(filler_text),
                    word_count=len(_FILLER_WORD_RE.findall(filler_text)),
                ).info("Processed Talker-authored filler candidate")
            if filler_mode != "emit":
                filler_text = ""
            slots = {key: value for key, value in arguments.items() if key not in {"query", "intent", "filler_text"}}
            filler_task: asyncio.Task | None = None
            filler_started = False

            async def emit_filler_after_threshold() -> None:
                try:
                    await asyncio.sleep(filler_threshold_seconds)
                    await _emit_talker_response(params.llm, filler_text, append_to_context=False)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(f"Failed to emit Talker filler: {exc}")

            async def schedule_thinker_started_filler(event: ThinkerLifecycleEvent) -> None:
                nonlocal filler_started, filler_task
                if stage_metrics is not None:
                    await stage_metrics.bind_backend_call(params.tool_call_id, event.call_id)
                if event.marker != "ThinkerStarted" or not filler_text:
                    return
                if filler_started or (filler_task is not None and not filler_task.done()):
                    return
                filler_started = True
                if filler_threshold_seconds <= 0:
                    await _emit_talker_response(params.llm, filler_text, append_to_context=False)
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
            if stage_metrics is not None:
                await stage_metrics.cleanup_tool_call(params.tool_call_id)
            return
        except Exception as exc:
            logger.exception(f"call_backend failed before producing a result: {exc}")
            payload = {
                "type": "response_hint",
                "reason": "tool_error",
                "action": "retry",
                "response_text": "I could not complete that request right now. Please try again.",
                "context": "call_backend",
            }
        await _deliver_tool_payload(
            params,
            payload,
            default_mode=tool_result_mode_default,
            stage_metrics=stage_metrics,
        )

    async def handle_cancel_backend(params: FunctionCallParams) -> None:
        cancelled = thinker.cancel_active("user_cancelled")
        cancel_pending = getattr(thinker, "cancel_pending_work", None)
        if not callable(cancel_pending):
            # Compatibility for third-party/older airline backends while they
            # migrate to the domain-neutral protocol.
            cancel_pending = getattr(thinker, "cancel_pending_booking", None)
        cleared_pending_work = bool(cancel_pending()) if callable(cancel_pending) else False
        interrupted_speech = bool(interrupted_speech_consumer()) if interrupted_speech_consumer else False
        did_cancel = cancelled or cleared_pending_work or interrupted_speech
        if cancelled or cleared_pending_work:
            reason = "cancelled"
        elif interrupted_speech:
            reason = "interrupted_speech"
        else:
            reason = "nothing_to_cancel"
        payload = {
            "type": "response_hint",
            "reason": reason,
            "action": "cancelled" if did_cancel else "nothing_to_cancel",
            "response_text": "Okay, I stopped that." if did_cancel else "There is nothing pending right now.",
            "context": "cancel_backend",
        }
        if _tool_result_mode(tool_result_mode_default) == "direct":
            await _emit_talker_response(params.llm, str(payload["response_text"]), append_to_context=False)
            await params.result_callback(payload, properties=FunctionCallResultProperties(run_llm=False))
            if stage_metrics is not None:
                await stage_metrics.cleanup_tool_call(params.tool_call_id)
            return
        await params.result_callback(payload)

    return {"call_backend": handle_call_backend, "cancel_backend": handle_cancel_backend}


async def _emit_talker_response(llm, text: str, *, append_to_context: bool = True) -> None:
    """Emit Talker-authored filler through the normal LLM text/TTS path."""
    if _task_cancellation_requested():
        return
    started = False
    try:
        started = True
        await llm.push_frame(LLMFullResponseStartFrame())
        text_frame = LLMTextFrame(text=text)
        text_frame.append_to_context = append_to_context
        await llm.push_frame(text_frame)
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


def _remember_backend_response(llm, text: str, payload: dict[str, Any]) -> None:
    remember_result = getattr(llm, "remember_backend_result", None)
    if callable(remember_result):
        remember_result(payload)
        return
    remember = getattr(llm, "remember_backend_response", None)
    if callable(remember):
        remember(text)


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


async def _deliver_tool_payload(
    params: FunctionCallParams,
    payload: dict[str, Any],
    *,
    default_mode: object = "talker",
    stage_metrics: StageMetricsCoordinator | None = None,
) -> None:
    """Deliver one grounded payload through the configured final-response path."""
    if not is_speakable_payload(payload):
        await params.result_callback(payload, properties=FunctionCallResultProperties(run_llm=False))
        if stage_metrics is not None:
            await stage_metrics.cleanup_tool_call(params.tool_call_id)
        return
    response_text = str(payload.get("response_text") or "")
    _remember_backend_response(params.llm, response_text, payload)
    if _should_deliver_directly(payload, default_mode=default_mode):
        await _emit_talker_response(params.llm, response_text, append_to_context=False)
        await params.result_callback(payload, properties=FunctionCallResultProperties(run_llm=False))
        if stage_metrics is not None:
            await stage_metrics.cleanup_tool_call(params.tool_call_id)
        return
    await params.result_callback(
        _talker_result_projection(payload),
        properties=FunctionCallResultProperties(run_llm=True),
    )


def _tool_result_mode(default_mode: object = "talker") -> str:
    raw = os.getenv("FRONTEND_BACKEND_TOOL_RESULT_MODE", "").strip().lower()
    if raw in {"direct", "hybrid", "talker"}:
        return raw
    if _direct_tool_response_enabled():
        return "direct"
    normalized_default = str(default_mode or "").strip().lower()
    return normalized_default if normalized_default in {"direct", "hybrid", "talker"} else "talker"


def _should_deliver_directly(payload: dict[str, Any], *, default_mode: object = "talker") -> bool:
    mode = _tool_result_mode(default_mode)
    if mode == "direct":
        return True
    if mode == "hybrid":
        return _payload_outcome(payload) == "success"
    return False


def _payload_outcome(payload: dict[str, Any]) -> str:
    if payload.get("type") == "tool_result":
        status = str(payload.get("status") or "error")
        return "success" if status == "success" else "partial" if status == "partial" else "failure"
    reason = str(payload.get("reason") or "")
    if reason in {"params_missing", "params_invalid"}:
        return "needs_input"
    if reason in {"aborted", "cancelled"}:
        return "cancelled"
    return "failure"


def _talker_result_projection(payload: dict[str, Any]) -> dict[str, Any]:
    """Expose only the trusted spoken contract, never raw provider data."""
    allowed = {
        "type",
        "tool",
        "status",
        "response_text",
        "reason",
        "action",
        "context",
        "params_needed",
        "params_resolved",
    }
    return {key: value for key, value in payload.items() if key in allowed}


def _talker_filler_mode() -> str:
    raw = os.getenv("FRONTEND_BACKEND_TALKER_FILLER_MODE", "emit").strip().lower()
    return raw if raw in {"off", "observe", "emit"} else "emit"


def _validated_talker_filler(query: str, raw_filler: object) -> str:
    """Accept a short grounded progress phrase or suppress it without replacement."""
    original = str(raw_filler or "")
    filler = " ".join(original.split()).strip()
    if not filler or len(filler) > 96 or "\n" in original:
        return ""
    words = _FILLER_WORD_RE.findall(filler)
    if not 3 <= len(words) <= 12:
        return ""
    if "?" in filler or len(re.findall(r"[.!]", filler)) > 1:
        return ""
    if any(character.isdigit() for character in filler):
        return ""
    if _FILLER_INTERNAL_RE.search(filler) or _FILLER_RESULT_CLAIM_RE.search(filler):
        return ""
    filler_tokens = set(_FILLER_TOKEN_RE.findall(filler.casefold())) - _FILLER_PROGRESS_WORDS
    query_tokens = set(_FILLER_TOKEN_RE.findall(query.casefold()))
    if not filler_tokens or filler_tokens.isdisjoint(query_tokens):
        return ""
    return filler


def _task_cancellation_requested() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0
