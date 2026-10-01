# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

from examples.frontend_backend_agent.src.frontend_verdict import decide
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
#: Bound on the client-tool facts handed to the Talker to compose from.
#: Large enough for an ordinary record, small enough that an unexpectedly
#: large one cannot crowd out the prompt.
_MAX_CLIENT_FACT_CHARS = 2_000

if TYPE_CHECKING:
    from pipecat.services.llm_service import FunctionCallParams

    from examples.frontend_backend_agent.src.domain import FillerPolicy
    from examples.frontend_backend_agent.src.history import ConversationTranscript
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
    allow_talker_frames: bool = True,
    realtime_filler_emitter: Callable[[str], Awaitable[bool]] | None = None,
    frontend_verdict: bool = False,
    transcript: ConversationTranscript | None = None,
) -> dict[str, Callable]:
    """Return tool handlers bound to one session-local backend agent.

    Realtime sessions suppress unowned Talker frames. When provided, the
    deferred filler emitter claims a separate pipeline-created response only
    after the delegated function-call response has closed.

    With ``frontend_verdict``, a ``call_backend`` made while earlier work still
    runs can take over that run instead of restarting it (see
    ``frontend_verdict.decide``). ``transcript`` supplies the user's latest words.
    """
    consecutive_planner_errors = 0
    continuation_supported = frontend_verdict and bool(getattr(thinker, "supports_task_continuation", False))
    tool_result_mode_default = getattr(thinker, "tool_result_mode_default", "talker")
    talker_result_tools = frozenset(getattr(thinker, "talker_result_tools", ()))

    async def handle_call_backend(params: FunctionCallParams) -> None:
        nonlocal consecutive_planner_errors
        arguments = _normalize_arguments(params.arguments or {})
        query = str(arguments.get("query", "") or "").strip()
        if not query or len(query) > max_query_chars:
            consecutive_planner_errors = 0
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
            await _emit_terminal_payload(
                params,
                payload,
                allow_talker_frames=allow_talker_frames,
            )
            return
        continue_active = continuation_supported and _continues_running_request(thinker, arguments, query, transcript)
        ledger = getattr(thinker, "conversation_ledger", None)
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
            if filler_mode != "emit" or (not allow_talker_frames and realtime_filler_emitter is None):
                filler_text = ""
            slots = {
                key: value for key, value in arguments.items() if key not in {"query", "intent", "filler_text", "task"}
            }
            filler_task: asyncio.Task | None = None
            filler_started = False
            filler_emitted = False
            run_id: str | None = None

            async def emit_filler_once() -> None:
                nonlocal filler_emitted
                if filler_emitted or not filler_text:
                    return
                filler_emitted = True
                if allow_talker_frames:
                    await _emit_talker_response(params.llm, filler_text, append_to_context=False)
                elif realtime_filler_emitter is not None:
                    await realtime_filler_emitter(filler_text)

            async def emit_filler_after_threshold() -> None:
                try:
                    await asyncio.sleep(filler_threshold_seconds)
                    await emit_filler_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(f"Failed to emit Talker filler: {exc}")

            async def schedule_thinker_started_filler(event: ThinkerLifecycleEvent) -> None:
                nonlocal filler_started, filler_task, run_id
                if event.marker in {"ThinkerStarted", "ThinkerContinued"}:
                    run_id = event.call_id
                if stage_metrics is not None:
                    await stage_metrics.bind_backend_call(params.tool_call_id, event.call_id)
                if event.marker == "ThinkerContinued":
                    # The run already spoke its own progress phrase if it needed one.
                    # Speak this call's phrase only when the Talker chose to answer a
                    # progress check, and do it now rather than after the threshold.
                    if filler_text and not filler_started:
                        filler_started = True
                        filler_task = asyncio.create_task(emit_filler_once())
                    return
                if event.marker == "IntermediateResponse" and filler_text and not filler_emitted:
                    await _cancel_pending_filler(filler_task)
                    filler_task = None
                    await emit_filler_once()
                    return

                if event.marker != "ThinkerStarted" or not filler_text:
                    return
                if filler_started or (filler_task is not None and not filler_task.done()):
                    return
                filler_started = True
                if filler_threshold_seconds <= 0:
                    await emit_filler_once()
                    return
                filler_task = asyncio.create_task(emit_filler_after_threshold())

            try:
                if continue_active:
                    payload = await thinker.call(
                        query, slots=slots, on_started=schedule_thinker_started_filler, continue_active=True
                    )
                else:
                    payload = await thinker.call(query, slots=slots, on_started=schedule_thinker_started_filler)
            finally:
                if filler_emitted and filler_task is not None and not filler_task.done():
                    with suppress(asyncio.CancelledError):
                        await filler_task
                else:
                    await _cancel_pending_filler(filler_task)
        except asyncio.CancelledError:
            if _task_cancellation_requested():
                logger.info("call_backend cancelled by Pipecat; allowing it to settle the function call")
                raise
            consecutive_planner_errors = 0
            logger.info("call_backend result suppressed after Thinker abort")
            if ledger is not None:
                ledger.record_delivery(run_id, "not_delivered")
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
            consecutive_planner_errors = 0
            logger.exception(f"call_backend failed before producing a result: {exc}")
            payload = {
                "type": "response_hint",
                "reason": "tool_error",
                "action": "retry",
                "error": str(exc),
                "response_text": "I could not complete that request right now. Please try again.",
                "context": "call_backend",
            }
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
                await _emit_terminal_payload(
                    params,
                    terminal_payload,
                    allow_talker_frames=allow_talker_frames,
                )
                if ledger is not None:
                    spoken = allow_talker_frames
                    ledger.record_delivery(
                        run_id,
                        "spoken_direct" if spoken else "via_talker",
                        str(terminal_payload.get("response_text") or "") if spoken else "",
                    )
                if stage_metrics is not None:
                    await stage_metrics.cleanup_tool_call(params.tool_call_id)
                consecutive_planner_errors = 0
                return
        else:
            consecutive_planner_errors = 0
        delivery = await _deliver_tool_payload(
            params,
            payload,
            default_mode=tool_result_mode_default,
            talker_result_tools=talker_result_tools,
            stage_metrics=stage_metrics,
            allow_talker_frames=allow_talker_frames,
        )
        if ledger is not None:
            ledger.record_delivery(
                run_id,
                delivery,
                str(payload.get("response_text") or "") if delivery == "spoken_direct" else "",
            )

    async def handle_cancel_backend(params: FunctionCallParams) -> None:
        nonlocal consecutive_planner_errors
        consecutive_planner_errors = 0
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
        if allow_talker_frames and _tool_result_mode(tool_result_mode_default) == "direct":
            await _emit_talker_response(params.llm, str(payload["response_text"]), append_to_context=False)
            await params.result_callback(payload, properties=FunctionCallResultProperties(run_llm=False))
            if stage_metrics is not None:
                await stage_metrics.cleanup_tool_call(params.tool_call_id)
            return
        await params.result_callback(payload)

    return {"call_backend": handle_call_backend, "cancel_backend": handle_cancel_backend}


def _continues_running_request(
    thinker: object,
    arguments: dict[str, Any],
    query: str,
    transcript: ConversationTranscript | None,
) -> bool:
    """Apply the frontend verdict to a call_backend made while earlier work may still run."""
    running_query = getattr(thinker, "running_query", None)
    running = running_query() if callable(running_query) else None
    raw_task = arguments.get("task")
    if running is None:
        if raw_task is not None:
            logger.bind(event="frontend_verdict", decision="new", reason="no_running_task", model_task=raw_task).info(
                "Frontend verdict found no running backend request"
            )
        return False
    utterance = transcript.latest_user_text() if transcript is not None else ""
    verdict = decide(raw_task, query, running, utterance)
    logger.bind(
        event="frontend_verdict",
        decision=verdict.decision,
        reason=verdict.reason,
        model_task=verdict.model_task,
        query_chars=len(query),
        running_query_chars=len(running),
        utterance_words=len(utterance.split()),
    ).info(f"Frontend verdict: {verdict.decision} ({verdict.reason})")
    return verdict.decision == "continue"


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


async def _emit_terminal_payload(
    params: FunctionCallParams,
    payload: dict[str, Any],
    *,
    allow_talker_frames: bool = True,
) -> None:
    """Deliver a validated terminal payload through the protocol-safe path."""
    if not allow_talker_frames:
        await params.result_callback(
            _talker_result_projection(payload),
            properties=FunctionCallResultProperties(run_llm=True),
        )
        return
    await _emit_talker_response(params.llm, str(payload.get("response_text") or ""))
    await params.result_callback(payload, properties=FunctionCallResultProperties(run_llm=False))


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


async def _deliver_tool_payload(
    params: FunctionCallParams,
    payload: dict[str, Any],
    *,
    default_mode: object = "talker",
    stage_metrics: StageMetricsCoordinator | None = None,
    talker_result_tools: frozenset[str] = frozenset(),
    allow_talker_frames: bool = True,
) -> str:
    """Deliver one grounded payload through the configured final-response path.

    Returns how it was delivered: ``not_speakable``, ``spoken_direct``, or ``via_talker``.
    """
    if not is_speakable_payload(payload):
        await params.result_callback(payload, properties=FunctionCallResultProperties(run_llm=False))
        if stage_metrics is not None:
            await stage_metrics.cleanup_tool_call(params.tool_call_id)
        return "not_speakable"
    response_text = str(payload.get("response_text") or "")
    _remember_backend_response(params.llm, response_text, payload)
    if not allow_talker_frames:
        await params.result_callback(
            _talker_result_projection(payload),
            properties=FunctionCallResultProperties(run_llm=True),
        )
        return "via_talker"
    if _should_deliver_directly(payload, default_mode=default_mode, talker_result_tools=talker_result_tools):
        await _emit_talker_response(params.llm, response_text, append_to_context=False)
        await params.result_callback(payload, properties=FunctionCallResultProperties(run_llm=False))
        if stage_metrics is not None:
            await stage_metrics.cleanup_tool_call(params.tool_call_id)
        return "spoken_direct"
    await params.result_callback(
        _talker_result_projection(payload),
        properties=FunctionCallResultProperties(run_llm=True),
    )
    return "via_talker"


def _tool_result_mode(default_mode: object = "talker") -> str:
    raw = os.getenv("FRONTEND_BACKEND_TOOL_RESULT_MODE", "").strip().lower()
    if raw in {"direct", "hybrid", "talker"}:
        return raw
    if _direct_tool_response_enabled():
        return "direct"
    normalized_default = str(default_mode or "").strip().lower()
    return normalized_default if normalized_default in {"direct", "hybrid", "talker"} else "talker"


def _payload_is_client_owned(payload: dict[str, Any]) -> bool:
    data = payload.get("data")
    return isinstance(data, dict) and data.get("owner") == "client"


def _should_deliver_directly(
    payload: dict[str, Any],
    *,
    default_mode: object = "talker",
    talker_result_tools: frozenset[str] = frozenset(),
) -> bool:
    # A client-owned result is whatever the caller's own tool returned, in
    # whatever shape it chose. Direct delivery speaks response_text as-is,
    # which is safe only for results this repository composes itself. Route
    # these through the Talker in every mode so a sentence is written for
    # them, rather than a serialized record being read out.
    if _payload_is_client_owned(payload):
        return False
    mode = _tool_result_mode(default_mode)
    if mode == "direct":
        return True
    if mode == "hybrid":
        if talker_result_tools:
            dynamic_success = _payload_outcome(payload) == "success" and payload.get("tool") in talker_result_tools
            return not dynamic_success
        return _payload_outcome(payload) == "success"
    return False


def _payload_outcome(payload: dict[str, Any]) -> str:
    if payload.get("type") == "tool_result":
        status = str(payload.get("status") or "error")
        return "success" if status == "success" else "partial" if status == "partial" else "failure"
    reason = str(payload.get("reason") or "")
    if reason in {"params_missing", "params_invalid", "confirmation_needed"}:
        return "needs_input"
    if reason in {"aborted", "cancelled"}:
        return "cancelled"
    return "failure"


def _talker_result_projection(payload: dict[str, Any]) -> dict[str, Any]:
    """Expose the trusted spoken contract plus a bounded weather fact projection."""
    allowed = {
        "type",
        "tool",
        "status",
        "response_text",
        "reason",
        "action",
        "context",
        "params_needed",
        # The action awaiting consent. The caller's policy requires its details
        # be stated back, and the Talker is the component trusted to phrase a
        # client-owned payload.
        "params_resolved",
    }
    projected = {key: value for key, value in payload.items() if key in allowed}
    data = payload.get("data")
    result = data.get("result") if isinstance(data, dict) else None
    if payload.get("tool") == "get_weather" and isinstance(result, dict):
        weather_keys = {
            "city",
            "temperature",
            "temperature_unit",
            "condition",
            "feels_like",
            "humidity_percent",
            "wind_kph",
        }
        projected["data"] = {key: result[key] for key in weather_keys if key in result}
    elif _payload_is_client_owned(payload) and result is not None:
        # A client tool's schema is declared per session, so there is no key
        # whitelist to apply as there is for weather. Give the Talker a
        # bounded rendering to compose a sentence from: it needs the facts,
        # and the bound keeps an unexpectedly large record from filling the
        # prompt. The Talker writes the sentence; this is never spoken as-is.
        projected["data"] = _bounded_client_facts(result)
    return projected


def _bounded_client_facts(result: Any) -> Any:
    """Return a size-bounded rendering of one client tool's result."""
    rendered = json.dumps(result, ensure_ascii=False, allow_nan=False, default=str, sort_keys=True)
    if len(rendered) <= _MAX_CLIENT_FACT_CHARS:
        return result
    return {"truncated": True, "preview": rendered[:_MAX_CLIENT_FACT_CHARS]}


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
