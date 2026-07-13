# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""User-facing Speaker Omni agent."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from loguru import logger
from pipecat.frames.frames import ErrorFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext

from examples.omni_assistant.nvidia_omni_multimodal_service import (
    NvidiaOmniMultimodalService,
    NvidiaOmniSettings,
    NvidiaOmniTurnResult,
)
from utils import parse_env_float, parse_env_int

_TURN_ACTIONS = frozenset({"respond", "think", "analyze_attachment", "capture_highres", "clarify"})
_MEDIA_ACTIONS = frozenset({"none", "new", "rerun"})
_INPUT_SOURCES = frozenset({"none", "live_webcam", "uploaded_attachment"})
_MEDIA_FIELD_PREFIXES = ("- selected_input_source:", "- media_analysis_action:", "- media_analysis_prompt:")

_BRIDGE_FILLERS = (
    "Hmm, let me look back over our conversation for a second.",
    "Let me think this through more carefully.",
    "One moment — let me reconsider that.",
)
_REPEAT_MIN_WORDS = 4
_REPEAT_HISTORY = 4

_AFFIRMATION_TOKENS = frozenset(
    {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "please", "go", "do", "fine", "alright", "right"}
)
_CAPTURE_ESCALATION_COOLDOWN = 3
_ACTION_CORRECTION_MAX_TOKENS = 2048
_ACTION_FALLBACK_RESPONSE = "Let me think that through carefully."


def _lean_contract(full_instruction: str) -> str:
    """Derive the lean contract by dropping the media-routing field lines."""
    lines = [line for line in full_instruction.splitlines() if not line.strip().startswith(_MEDIA_FIELD_PREFIXES)]
    return "\n".join(lines)


def _is_affirmation(transcript: str) -> bool:
    """Whether a user turn is a short agreement (a follow-up, not a "stuck" signal)."""
    words = _normalize_text(transcript).split()
    if not words or len(words) > 6:
        return False
    return words[0] in _AFFIRMATION_TOKENS or _AFFIRMATION_TOKENS.issuperset(set(words))


class _RepeatGuard:
    """Detects verbatim response repeats and bridges them with a rotating filler.

    The model can restate the same line even when it believes the turn went fine,
    so repetition is detected deterministically here to force a Thinker escalation.
    """

    def __init__(self) -> None:
        self._recent: deque[str] = deque(maxlen=_REPEAT_HISTORY)
        self._filler_index = 0
        self.suppressing = False
        self.detected = False
        self.filler = ""
        self.emitted = False

    def bridge_filler(self, text: str) -> str | None:
        """Return a bridging filler when ``text`` is the turn's first chunk and repeats a recent reply."""
        if self.emitted or self.suppressing or not self._is_repeat(text):
            return None
        self.suppressing = True
        self.detected = True
        self.emitted = True
        self.filler = _BRIDGE_FILLERS[self._filler_index % len(_BRIDGE_FILLERS)]
        self._filler_index += 1
        return self.filler

    def _is_repeat(self, text: str) -> bool:
        normalized = _normalize_text(text)
        if len(normalized.split()) < _REPEAT_MIN_WORDS:
            return False
        return normalized in self._recent

    def note_reply(self, response: str, *, track: bool) -> None:
        """Remember the model's own reply so the next turn can detect a repeat."""
        normalized = _normalize_text(response)
        if normalized and track:
            self._recent.append(normalized)

    def reset(self) -> None:
        self.suppressing = False
        self.detected = False
        self.filler = ""
        self.emitted = False


class SubagentsSpeakerOmniService(NvidiaOmniMultimodalService):
    """Speaker Omni wrapper that turns each strict-JSON turn into one owned action.

    Parses the per-turn action envelope, gates malformed output out of TTS, runs one
    bounded self-correction, and dispatches media analysis / high-res capture / Thinker
    escalation to the transport agent via the provided handler callbacks.
    """

    def __init__(
        self,
        *,
        audio_response_instruction: str,
        media_analysis_prompt_handler: Callable[[str, str, str, str], Awaitable[None]] | None = None,
        uploaded_attachment_available: Callable[[], bool] | None = None,
        attachment_pending: Callable[[], bool] | None = None,
        thinking_handler: Callable[[str, str, str], Awaitable[None]] | None = None,
        highres_capture_handler: Callable[[str], Awaitable[None]] | None = None,
        visual_status_provider: Callable[[], str] | None = None,
        **kwargs,
    ) -> None:
        """Configure the wrapper with the per-turn JSON contract from ``prompts.yaml``."""
        super().__init__(**kwargs)
        self._media_analysis_prompt_handler = media_analysis_prompt_handler
        self._uploaded_attachment_available = uploaded_attachment_available
        self._attachment_pending = attachment_pending
        self._thinking_handler = thinking_handler
        self._highres_capture_handler = highres_capture_handler
        self._visual_status_provider = visual_status_provider
        self._repeat = _RepeatGuard()
        self._capture_cooldown = 0
        self._audio_response_instruction_content = audio_response_instruction.strip()
        if not self._audio_response_instruction_content:
            raise ValueError("SpeakerAgent audio_response_instruction must be provided from prompts.yaml")

    def _audio_response_instruction(self) -> str:
        contract = (
            self._audio_response_instruction_content
            if self._routing_enabled()
            else _lean_contract(self._audio_response_instruction_content)
        )
        reminder = self._current_visual_reminder()
        if not reminder:
            return contract
        return f"{reminder}\n\n{contract}"

    def _current_visual_reminder(self) -> str:
        """Per-turn pointer to where the current visual sources live.

        Instead of re-injecting the live webcam view and uploaded-file details each turn
        (which over-weighted the camera), this just reminds the Speaker that both sources
        sit on the pinned Subagents board and should be read there, kept separate.
        """
        if self._visual_status_provider is None and self._attachment_pending is None:
            return ""
        return (
            "Reminder: your current visual sources are on the pinned Subagents board — the live webcam "
            "(your eyes) under the webcam entry, and any uploaded file under the media analyzer entry. "
            "Read them there for this turn and keep the two sources separate."
        )

    def _routing_enabled(self) -> bool:
        """Whether this turn offers the media-routing fields (only while an upload is pending).

        Once an upload is analyzed it is past context, so the lean contract keeps the model
        from re-routing a live-visual turn to a stale file and reaches ``response`` sooner.
        """
        check = self._attachment_pending or self._uploaded_attachment_available
        if check is None:
            return False
        try:
            return bool(check())
        except Exception:
            return True

    def _parse_turn_result(self, raw_content: str, *, parse_json: bool) -> NvidiaOmniTurnResult:
        result = super()._parse_turn_result(raw_content, parse_json=parse_json)
        response = _clean_spoken_response_artifacts(result.response)
        payload, recovery = _normalize_action_envelope(
            result.payload,
            transcript=result.transcript,
            response=response,
        )
        selected_input_source = _normalize_selected_input_source(payload.get("selected_input_source"))
        media_action = _normalize_media_analysis_action(payload.get("media_analysis_action"))
        if self._is_missing_uploaded_attachment_route(selected_input_source, media_action):
            payload["turn_action"] = "clarify"
            payload["selected_input_source"] = "none"
            payload["media_analysis_action"] = "none"
            payload["media_analysis_prompt"] = ""
            response = _missing_uploaded_attachment_response(result.transcript)
            payload["response"] = response
            return NvidiaOmniTurnResult(
                transcript=result.transcript,
                response=response,
                raw_content=result.raw_content,
                payload=payload,
            )
        if recovery:
            payload["_action_recovery"] = recovery
        if payload.get("_action_fallback"):
            response = ""
        payload["response"] = response
        if response == result.response and payload == result.payload:
            return result
        return NvidiaOmniTurnResult(
            transcript=result.transcript,
            response=response,
            raw_content=result.raw_content,
            payload=payload,
        )

    async def _emit_assistant_text(self, text: str, *, response_started: bool) -> bool:
        cleaned = _clean_spoken_response_artifacts(text)
        if not cleaned:
            return response_started
        filler = self._repeat.bridge_filler(cleaned)
        if filler is not None:
            logger.info(f"Speaker Omni suppressed a verbatim repeat; bridging with filler={filler!r}")
            return await super()._emit_assistant_text(filler, response_started=response_started)
        if self._repeat.suppressing:
            return response_started
        self._repeat.emitted = True
        return await super()._emit_assistant_text(cleaned, response_started=response_started)

    def _structured_response_control_fields(self) -> tuple[str, ...]:
        return ("turn_action",)

    def _should_emit_streamed_structured_response(self, field_values: Mapping[str, str]) -> bool:
        return _normalize_turn_action(field_values.get("turn_action")) in _TURN_ACTIONS

    def _is_missing_uploaded_attachment_route(self, selected_input_source: str, media_action: str) -> bool:
        if selected_input_source == "live_webcam":
            return False
        if selected_input_source != "uploaded_attachment" and media_action not in {"new", "rerun"}:
            return False
        if self._uploaded_attachment_available is None:
            return False
        return not self._uploaded_attachment_available()

    async def _on_turn_result(self, result: NvidiaOmniTurnResult) -> NvidiaOmniTurnResult | None:
        """Correct one unsafe envelope, then handle it or fall back to Thinker."""
        if not result.payload.get("_action_fallback"):
            await self._handle_turn_result(result)
            return None
        corrected = await self._attempt_action_correction(result)
        if corrected is not None:
            await self._handle_turn_result(corrected, track_response=True)
            return corrected
        logger.warning(
            f"Speaker Omni action correction failed; falling back to Thinker: "
            f"reason={result.payload.get('_action_recovery', 'invalid envelope')!r}"
        )
        fallback_payload = dict(result.payload)
        fallback_payload["_action_fallback"] = False
        fallback_payload["response"] = _ACTION_FALLBACK_RESPONSE
        fallback = NvidiaOmniTurnResult(
            transcript=result.transcript,
            response=_ACTION_FALLBACK_RESPONSE,
            raw_content=result.raw_content,
            payload=fallback_payload,
        )
        await self._handle_turn_result(fallback, track_response=False)
        return fallback

    async def _attempt_action_correction(self, result: NvidiaOmniTurnResult) -> NvidiaOmniTurnResult | None:
        """Run exactly one Speaker regeneration for a structurally unsafe envelope."""
        reason = str(result.payload.get("_action_recovery", "invalid or contradictory action envelope"))
        instruction = _action_correction_instruction(result, reason=reason)
        try:
            raw_correction = await self.run_inference(
                self._context,
                max_tokens=_ACTION_CORRECTION_MAX_TOKENS,
                system_instruction=instruction,
            )
        except Exception as exc:
            logger.warning(f"Speaker Omni action correction request failed: {exc}")
            return None
        if not raw_correction:
            return None
        corrected = self._parse_turn_result(raw_correction, parse_json=True)
        if corrected.payload.get("_action_fallback") or corrected.payload.get("_action_recovery"):
            logger.warning("Speaker Omni rejected structurally invalid action correction")
            return None
        if _normalize_turn_action(corrected.payload.get("turn_action")) == "think":
            logger.warning("Speaker Omni rejected a think action-correction; deferring to the Thinker fallback")
            return None
        logger.info(f"Speaker Omni accepted one action-envelope correction: action={corrected.payload['turn_action']}")
        return corrected

    async def _handle_turn_result(self, result: NvidiaOmniTurnResult, *, track_response: bool = True) -> None:
        """Record and dispatch one structurally normalized turn result."""
        transcript = result.transcript.strip()
        response = _clean_spoken_response_artifacts(result.response)
        user_text = transcript or response or result.raw_content.strip()
        if not user_text:
            return

        self._repeat.note_reply(response, track=track_response)

        turn_action = _normalize_turn_action(result.payload.get("turn_action"))
        selected_input_source = _normalize_selected_input_source(result.payload.get("selected_input_source"))
        media_prompt = str(result.payload.get("media_analysis_prompt", "")).strip()
        media_action = _normalize_media_analysis_action(result.payload.get("media_analysis_action"))
        capture_requested = turn_action == "capture_highres"
        highres_query = str(result.payload.get("highres_query", "")).strip()
        if capture_requested:
            self._capture_cooldown = _CAPTURE_ESCALATION_COOLDOWN
        should_analyze_media = (
            turn_action == "analyze_attachment"
            and selected_input_source == "uploaded_attachment"
            and (bool(media_prompt) or media_action in {"new", "rerun"})
        )
        if selected_input_source != "uploaded_attachment" and (media_prompt or media_action in {"new", "rerun"}):
            logger.info(
                f"Speaker Omni ignored media trigger for source={selected_input_source!r}, "
                f"transcript_chars={len(transcript)}"
            )

        media_dispatched = False
        if should_analyze_media and self._media_analysis_prompt_handler:
            media_prompt = media_prompt or transcript or response
            media_action = "new" if media_action == "none" else media_action
            try:
                logger.info(
                    f"Speaker Omni queued media analysis: action={media_action}, transcript_chars={len(transcript)}"
                )
                await self._media_analysis_prompt_handler(user_text, media_prompt, media_action, selected_input_source)
                media_dispatched = True
            except Exception as exc:
                logger.warning(f"Speaker Omni media-analysis prompt handler failed: {exc}")
        if should_analyze_media and not media_dispatched:
            await self.push_error_frame(
                ErrorFrame(error="Could not start media analysis. Please try again.", fatal=False)
            )

        capture_dispatched = False
        if capture_requested and self._highres_capture_handler:
            query = highres_query or transcript or response
            try:
                logger.info(f"Speaker Omni requested a high-res webcam capture: query_chars={len(query)}")
                await self._highres_capture_handler(query)
                capture_dispatched = True
            except Exception as exc:
                logger.warning(f"Speaker Omni high-res capture handler failed: {exc}")
        if capture_requested and not capture_dispatched:
            await self.push_error_frame(
                ErrorFrame(error="Could not start the high-resolution capture. Please try again.", fatal=False)
            )

        await self._maybe_escalate_thinking(
            transcript=transcript,
            repeated=self._repeat.detected,
            payload=result.payload,
            media_pending=media_dispatched or capture_dispatched,
        )
        self._repeat.reset()
        if self._capture_cooldown > 0:
            self._capture_cooldown -= 1

    async def _maybe_escalate_thinking(
        self, *, transcript: str, repeated: bool, payload: Mapping[str, Any], media_pending: bool
    ) -> None:
        """Escalate to the reasoning-ON Thinker on a ``think`` action or a repetition backstop.

        Never escalates alongside subagent work, nor on a live-visual follow-up (a bare
        affirmation or the post-capture cooldown), where the vision-less Thinker dead-ends.
        """
        if media_pending or not (self._thinking_handler and transcript):
            return
        needs_thinking = _normalize_turn_action(payload.get("turn_action")) == "think"
        if not (needs_thinking or repeated):
            return
        if repeated and not needs_thinking and (self._capture_cooldown > 0 or _is_affirmation(transcript)):
            logger.info(
                "Speaker Omni skipped repetition escalation for a visual/affirmation follow-up: "
                f"transcript_chars={len(transcript)}"
            )
            return
        reason = "repetition" if repeated else ""
        effort = "high" if repeated else "medium"
        try:
            logger.info(f"Speaker Omni escalating to Thinker: reason={reason or 'needs_thinking'}, effort={effort}")
            await self._thinking_handler(transcript, effort, reason)
        except Exception as exc:
            logger.warning(f"Speaker Omni thinking handler failed: {exc}")
            await self.push_error_frame(
                ErrorFrame(error="Could not start deliberate thinking. Please try again.", fatal=False)
            )


def _clean_spoken_response_artifacts(text: str) -> str:
    """Remove worker-only prompt fragments if the model leaks them into speech."""
    cleaned = text.strip()
    cleaned = cleaned.replace("Answer only with the final user-facing result.", "")
    cleaned = cleaned.replace("Answer only with the final user-facing result", "")
    return re.sub(r"\s+", " ", cleaned).strip()


def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for repeat comparison."""
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").lower()).split())


def _one_of(value: Any, allowed: frozenset[str], default: str) -> str:
    """Return the normalized value when it is one of ``allowed``, else ``default``."""
    candidate = str(value or default).strip().lower()
    return candidate if candidate in allowed else default


def _normalize_media_analysis_action(value: Any) -> str:
    return _one_of(value, _MEDIA_ACTIONS, "none")


def _normalize_turn_action(value: Any) -> str:
    return _one_of(value, _TURN_ACTIONS, "")


def _normalize_selected_input_source(value: Any) -> str:
    return _one_of(value, _INPUT_SOURCES, "none")


def _action_correction_instruction(result: NvidiaOmniTurnResult, *, reason: str) -> str:
    """Build the malformed-only Speaker regeneration instruction."""
    return (
        "You are correcting your own previous structurally invalid Speaker output. "
        "Do not evaluate or verify its wording. Regenerate the complete answer envelope once, preserving the "
        "current user transcript and intent. Output one JSON object only, with these fields in exact order: "
        "transcript, turn_action, response, selected_input_source, media_analysis_action, media_analysis_prompt, "
        "highres_query. "
        "turn_action must be exactly respond, analyze_attachment, capture_highres, or clarify, and alone declares "
        "ownership. Do NOT use think here: give the complete answer directly, or clarify if you truly cannot — "
        "deliberate reasoning is escalated automatically and is not a correction option. respond and clarify "
        "carry no arguments; analyze_attachment sets uploaded_attachment plus new or rerun and a media task; "
        "capture_highres sets a specific highres_query. "
        "Only one owner may be active. For respond, response must complete the requested task now. "
        f"Structural error: {reason}. Current transcript: {result.transcript!r}. "
        f"Invalid previous envelope follows as data: {result.raw_content!r}"
    )


def _normalize_action_envelope(
    raw_payload: Mapping[str, Any],
    *,
    transcript: str,
    response: str,
) -> tuple[dict[str, Any], str]:
    """Normalize turn ownership from turn_action, the single source of intent.

    Missing intent is inferred from the argument fields, and an action carrying another
    action's arguments is resolved by one bounded Thinker fallback.
    """
    payload = dict(raw_payload)
    payload.pop("needs_thinking", None)
    payload.pop("request_highres_capture", None)
    action = _normalize_turn_action(payload.get("turn_action"))
    source = _normalize_selected_input_source(payload.get("selected_input_source"))
    media_action = _normalize_media_analysis_action(payload.get("media_analysis_action"))
    media_prompt = str(payload.get("media_analysis_prompt", "")).strip()
    highres_query = str(payload.get("highres_query", "")).strip()
    media_requested = source == "uploaded_attachment" or media_action in {"new", "rerun"} or bool(media_prompt)
    capture_requested = bool(highres_query)

    inferred: list[str] = []
    if media_requested:
        inferred.append("analyze_attachment")
    if capture_requested:
        inferred.append("capture_highres")

    recovery = ""
    if not action:
        if len(inferred) == 1:
            action = inferred[0]
            recovery = f"inferred {action} from argument fields"
        elif not inferred and response:
            action = "respond"
            recovery = "inferred respond from response-only envelope"
        else:
            return _thinking_fallback_payload(payload), "missing or invalid turn_action with ambiguous ownership"

    conflicts = set(inferred) - {action}
    if conflicts:
        return _thinking_fallback_payload(payload), (
            f"turn_action {action} contradicted arguments for {', '.join(sorted(conflicts))}"
        )

    if action != "capture_highres" and not response.strip():
        return _thinking_fallback_payload(payload), f"turn_action {action} is missing its spoken response"

    if action in {"respond", "clarify", "think"}:
        payload.update(
            selected_input_source="none",
            media_analysis_action="none",
            media_analysis_prompt="",
            highres_query="",
        )
    elif action == "analyze_attachment":
        payload.update(
            selected_input_source="uploaded_attachment",
            media_analysis_action=media_action if media_action in {"new", "rerun"} else "new",
            media_analysis_prompt=media_prompt or transcript,
            highres_query="",
        )
    elif action == "capture_highres":
        query = highres_query or transcript
        if not query:
            return _thinking_fallback_payload(payload), "high-resolution capture is missing a query"
        payload.update(
            selected_input_source="none",
            media_analysis_action="none",
            media_analysis_prompt="",
            highres_query=query,
        )

    payload["turn_action"] = action
    return payload, recovery


def _thinking_fallback_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized.update(
        turn_action="think",
        highres_query="",
        selected_input_source="none",
        media_analysis_action="none",
        media_analysis_prompt="",
        _action_fallback=True,
    )
    return normalized


def _missing_uploaded_attachment_response(transcript: str) -> str:
    if "video" in transcript.lower():
        return "Please upload or attach the video first, then I can take a look."
    return "Please upload or attach the media first, then I can take a look."


class SpeakerOmniAgent(PipelineWorker):
    """Main conversational agent backed by the upstream-style Omni service.

    A bus-bridged ``PipelineWorker`` that receives user frames teed from the transport
    worker and is the only worker that emits spoken responses.
    """

    AGENT_NAME = "speaker_omni"

    def __init__(
        self,
        name: str | None = None,
        *,
        context: LLMContext,
        api_key: str,
        base_url: str,
        model_id: str,
        audio_response_instruction: str,
        extra_params: dict[str, Any] | None = None,
        media_analysis_prompt_handler: Callable[[str, str, str, str], Awaitable[None]] | None = None,
        uploaded_attachment_available: Callable[[], bool] | None = None,
        attachment_pending: Callable[[], bool] | None = None,
        thinking_handler: Callable[[str, str, str], Awaitable[None]] | None = None,
        highres_capture_handler: Callable[[str], Awaitable[None]] | None = None,
        visual_status_provider: Callable[[], str] | None = None,
    ) -> None:
        """Initialize the bridged Speaker Omni agent.

        ``enable_rtvi`` is False so only the transport worker emits the user
        transcript (the speaker must not convert it a second time).
        """
        omni = SubagentsSpeakerOmniService(
            api_key=api_key,
            base_url=base_url,
            context=context,
            extra=dict(extra_params or {}),
            settings=NvidiaOmniSettings(
                model=model_id,
                max_tokens=parse_env_int("OMNI_MAX_TOKENS", 8192, min_value=64),
                temperature=parse_env_float("OMNI_TEMPERATURE", 0.7, min_value=0.0),
                top_p=parse_env_float("OMNI_TOP_P", 0.95, min_value=0.0),
                response_format={"type": "json_object"},
                emit_transcriptions=True,
                min_user_audio_secs=parse_env_float("OMNI_MIN_USER_AUDIO_SECS", 0.3, min_value=0.0),
            ),
            media_analysis_prompt_handler=media_analysis_prompt_handler,
            uploaded_attachment_available=uploaded_attachment_available,
            attachment_pending=attachment_pending,
            thinking_handler=thinking_handler,
            highres_capture_handler=highres_capture_handler,
            visual_status_provider=visual_status_provider,
            audio_response_instruction=audio_response_instruction,
        )
        super().__init__(
            Pipeline([omni]),
            name=name or self.AGENT_NAME,
            active=True,
            bridged=(),
            enable_rtvi=False,
        )
