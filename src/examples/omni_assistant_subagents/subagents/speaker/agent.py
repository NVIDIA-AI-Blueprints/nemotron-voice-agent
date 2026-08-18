# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""User-facing Speaker Omni agent."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any

from loguru import logger
from openai.types.chat import ChatCompletionChunk
from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.frames.frames import ErrorFrame, LLMServiceMetadataFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import LLMService

from examples.omni_assistant.nvidia_omni_multimodal_service import (
    NvidiaOmniLLMService,
    NvidiaOmniSettings,
    text_message_part,
)
from examples.omni_assistant_subagents.subagents.speaker.action_envelope import (
    ACTION_FALLBACK_RESPONSE,
    TURN_ACTIONS,
    SpeakerTurnResult,
    action_correction_instruction,
    clean_spoken_response_artifacts,
    lean_contract,
    missing_uploaded_attachment_response,
    normalize_action_envelope,
    normalize_media_analysis_action,
    normalize_selected_input_source,
    normalize_turn_action,
)
from examples.omni_assistant_subagents.subagents.speaker.json_stream import JsonStringFieldStreamer
from examples.omni_assistant_subagents.subagents.speaker.repeat_guard import RepeatGuard, is_affirmation
from examples.shared.json_parsing import extract_json_object
from utils import parse_env_float, parse_env_int

_CAPTURE_ESCALATION_COOLDOWN = 3
_ACTION_CORRECTION_MAX_TOKENS = 2048


class SubagentsSpeakerOmniService(NvidiaOmniLLMService):
    """Speaker Omni wrapper that turns each strict-JSON turn into one owned action.

    The Speaker uses no tools, so it can afford the forced ``json_object`` response
    format that a single turn's action envelope needs. The envelope parser is
    layered on top of the inherited completion stream, which already has reasoning
    removed, so this class only deals with JSON.

    It gates malformed output out of TTS, runs one bounded self-correction, and
    dispatches media analysis / high-res capture / Thinker escalation to the
    transport agent via the provided handler callbacks.
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
        self._repeat = RepeatGuard()
        self._capture_cooldown = 0
        self._audio_response_instruction_content = audio_response_instruction.strip()
        if not self._audio_response_instruction_content:
            raise ValueError("SpeakerAgent audio_response_instruction must be provided from prompts.yaml")

    def _audio_response_instruction(self) -> str:
        contract = (
            self._audio_response_instruction_content
            if self._routing_enabled()
            else lean_contract(self._audio_response_instruction_content)
        )
        reminder = self._current_visual_reminder()
        if not reminder:
            return contract
        return f"{reminder}\n\n{contract}"

    def _current_visual_reminder(self) -> str:
        """Per-turn statement of the live view, plus where the visual sources live.

        The pinned board carries the same live view, but it sits above the whole
        conversation, and a reply that already claimed to see something outweighs it —
        the model then keeps repeating that claim with the camera off. Stating the live
        view next to the user's turn, the most salient position, is what keeps each
        reply grounded in the present scene. The pointer keeps the camera and an
        uploaded file from being read as one source.
        """
        live_view = self._live_view()
        if not live_view and self._attachment_pending is None:
            return ""
        pointer = (
            "Reminder: your current visual sources are on the pinned Subagents board — the live webcam "
            "(your eyes) under the webcam entry, and any uploaded file under the media analyzer entry. "
            "Read them there for this turn and keep the two sources separate."
        )
        if not live_view:
            return pointer
        return f"Live view right now: {live_view}.\n\n{pointer}"

    def _live_view(self) -> str:
        """What the live camera shows right now, or "" when no visual source is wired."""
        if self._visual_status_provider is None:
            return ""
        try:
            return self._visual_status_provider().strip()
        except Exception as exc:
            logger.debug(f"Speaker Omni live view unavailable: {exc}")
            return ""

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

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Carry the action-envelope contract on every request, with audio or without.

        An audio turn already appends the contract beside its audio. A turn driven
        by context alone — the opening introduction, or the out-of-band action
        correction — has no audio parts and would otherwise be asked for an
        envelope it was never given the shape of, so it gets the contract here.
        """
        params = super().build_chat_completion_params(params_from_context)
        if self._active_turn_parts:
            return params
        messages = list(params.get("messages") or [])
        messages.append({"role": "user", "content": [text_message_part(self._audio_response_instruction())]})
        params["messages"] = messages
        return params

    async def get_chat_completions(self, context: LLMContext) -> AsyncIterator[ChatCompletionChunk]:
        """Layer the action-envelope parser over the reasoning-filtered stream.

        Args:
            context: The LLM context for the completion request.

        Returns:
            An async iterator whose visible content is only the envelope's
            spoken ``response`` field.
        """
        stream = await super().get_chat_completions(context)
        return self._stream_action_envelope(stream)

    async def _stream_action_envelope(
        self, stream: AsyncIterator[ChatCompletionChunk]
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Speak the ``response`` field as it streams, then own the parsed turn.

        The envelope declares ``turn_action`` before ``response``, so ownership is
        known by the time there is anything to say. If it is not, nothing is
        streamed and the fully parsed envelope decides what the Speaker says.
        """
        transcript_field = JsonStringFieldStreamer("transcript")
        action_field = JsonStringFieldStreamer("turn_action")
        response_field = JsonStringFieldStreamer("response")
        transcript_text = ""
        action_text = ""
        raw_content = ""
        spoken_text = ""
        response_buffer = ""
        stream_released = False
        transcript_emitted = False
        spoke = False
        gated = False
        self._repeat.reset()

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            content = delta.content if delta is not None else None
            if not content:
                yield chunk
                continue

            raw_content += content
            if not transcript_field.done:
                transcript_text += transcript_field.feed(content)
                if transcript_field.done and transcript_text.strip() and self.current_turn_has_user_audio():
                    transcript_emitted = True
                    await self._emit_user_transcript(transcript_text.strip())
            if not action_field.done:
                action_text += action_field.feed(content)

            response_delta = response_field.feed(content)
            if response_delta and not gated and not spoke:
                gated = normalize_turn_action(action_text) not in TURN_ACTIONS
                if gated:
                    logger.info("Speaker Omni withheld streamed text: turn ownership was not declared first")
            if gated:
                spoken = ""
            elif stream_released:
                spoken = response_delta
            else:
                response_buffer += response_delta
                spoken = ""
                if response_buffer and response_field.done:
                    filler = self._repeat.bridge_filler(response_buffer)
                    if filler is not None:
                        logger.info(
                            f"Speaker Omni suppressed a streamed verbatim repeat; bridging with filler={filler!r}"
                        )
                    spoken = filler or response_buffer
                    stream_released = True
                elif response_buffer and not self._repeat.could_be_repeat_prefix(response_buffer):
                    spoken = response_buffer
                    stream_released = True
            spoke = spoke or bool(spoken)
            spoken_text += spoken
            delta.content = spoken or None
            yield chunk

        logger.debug(f"Speaker Omni envelope: {raw_content}")
        result = self._parse_turn_result(raw_content)
        if spoken_text:
            self._repeat.note_spoken(spoken_text)
        final = await self._resolve_turn(result)
        logger.info(
            "Speaker Omni turn: "
            f"live_view={(self._live_view() or '<none wired>')!r}, "
            f"heard={(final.transcript or transcript_text.strip())!r}, "
            f"action={normalize_turn_action(final.payload.get('turn_action'))}, "
            f"streamed={spoken_text.strip()!r}, "
            f"resolved={final.response.strip()!r}"
        )
        if final.transcript and not transcript_emitted:
            await self._emit_user_transcript(final.transcript)
        if not spoke or final is not result:
            await self._speak_response(final.response)

    async def _speak_response(self, text: str) -> None:
        """Clean a whole resolved response and bridge verbatim repeats before TTS.

        Only complete responses come through here. Streamed deltas reach TTS
        untouched, because the whitespace that separates two words is carried by
        the delta that starts the next one, so per-delta trimming would glue the
        spoken text together. It also takes a whole response to recognize a
        leaked prompt fragment or a verbatim repeat at all.
        """
        cleaned = clean_spoken_response_artifacts(text)
        if not cleaned:
            return
        filler = self._repeat.bridge_filler(cleaned)
        if filler is not None:
            logger.info(f"Speaker Omni suppressed a verbatim repeat; bridging with filler={filler!r}")
            await self._push_llm_text(filler)
            return
        if self._repeat.suppressing:
            return
        self._repeat.emitted = True
        await self._push_llm_text(cleaned)

    def service_metadata_frame(self) -> LLMServiceMetadataFrame:
        """Announce a plain service: no aggregator pair owns this conversation.

        Omni declares itself a realtime service so that a paired aggregator
        writes the spoken turn once the model reports it. Here the Speaker writes
        its own history, and the transport worker's assistant aggregator, which
        sees every frame bridged out of this worker, has no user half to pair
        with, so the mode would only fail on arrival.
        """
        return LLMService.service_metadata_frame(self)

    async def _emit_user_transcript(self, transcript: str) -> None:
        """Write the spoken turn into the Speaker's own context, then report it.

        The Speaker runs as a worker pipeline holding this service alone, so
        there is no user aggregator to write the conversation history from the
        reported frame, and the Speaker keeps that write itself.
        """
        if self._context is not None:
            self._context.add_message({"role": "user", "content": transcript})
        await super()._emit_user_transcript(transcript)

    def _parse_turn_result(self, raw_content: str) -> SpeakerTurnResult:
        raw_payload = extract_json_object(raw_content)
        if not raw_payload:
            logger.warning(f"Speaker Omni response did not parse as JSON: {raw_content[:500]!r}")
        transcript = str(raw_payload.get("transcript", "")).strip()
        if transcript and not self.current_turn_has_user_audio():
            # Nothing was spoken this turn, so a reported transcript is the model
            # echoing context back at us. Keeping it would enter the conversation
            # as something the user said and steer every later turn.
            logger.info(f"Speaker Omni dropped a transcript claimed without user audio: chars={len(transcript)}")
            transcript = ""
        response = clean_spoken_response_artifacts(str(raw_payload.get("response", "")))
        payload, recovery = normalize_action_envelope(
            raw_payload,
            transcript=transcript,
            response=response,
        )
        selected_input_source = normalize_selected_input_source(payload.get("selected_input_source"))
        media_action = normalize_media_analysis_action(payload.get("media_analysis_action"))
        if self._is_missing_uploaded_attachment_route(selected_input_source, media_action):
            payload["turn_action"] = "clarify"
            payload["selected_input_source"] = "none"
            payload["media_analysis_action"] = "none"
            payload["media_analysis_prompt"] = ""
            response = missing_uploaded_attachment_response(transcript)
            payload["response"] = response
            return SpeakerTurnResult(
                transcript=transcript,
                response=response,
                raw_content=raw_content,
                payload=payload,
            )
        if recovery:
            payload["_action_recovery"] = recovery
        if payload.get("_action_fallback"):
            response = ""
        payload["response"] = response
        return SpeakerTurnResult(
            transcript=transcript,
            response=response,
            raw_content=raw_content,
            payload=payload,
        )

    def _is_missing_uploaded_attachment_route(self, selected_input_source: str, media_action: str) -> bool:
        if selected_input_source == "live_webcam":
            return False
        if selected_input_source != "uploaded_attachment" and media_action not in {"new", "rerun"}:
            return False
        if self._uploaded_attachment_available is None:
            return False
        return not self._uploaded_attachment_available()

    async def _resolve_turn(self, result: SpeakerTurnResult) -> SpeakerTurnResult:
        """Correct one unsafe envelope, then handle it or fall back to Thinker.

        Returns the envelope the Speaker actually owns, which is ``result``
        itself whenever the model's first attempt was already usable.
        """
        if not result.payload.get("_action_fallback"):
            await self._handle_turn_result(result)
            return result
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
        fallback_payload["response"] = ACTION_FALLBACK_RESPONSE
        fallback = SpeakerTurnResult(
            transcript=result.transcript,
            response=ACTION_FALLBACK_RESPONSE,
            raw_content=result.raw_content,
            payload=fallback_payload,
        )
        await self._handle_turn_result(fallback, track_response=False)
        return fallback

    async def _attempt_action_correction(self, result: SpeakerTurnResult) -> SpeakerTurnResult | None:
        """Run exactly one Speaker regeneration for a structurally unsafe envelope."""
        reason = str(result.payload.get("_action_recovery", "invalid or contradictory action envelope"))
        instruction = action_correction_instruction(result, reason=reason)
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
        corrected = self._parse_turn_result(raw_correction)
        if corrected.payload.get("_action_fallback") or corrected.payload.get("_action_recovery"):
            logger.warning("Speaker Omni rejected structurally invalid action correction")
            return None
        if normalize_turn_action(corrected.payload.get("turn_action")) == "think":
            logger.warning("Speaker Omni rejected a think action-correction; deferring to the Thinker fallback")
            return None
        logger.info(f"Speaker Omni accepted one action-envelope correction: action={corrected.payload['turn_action']}")
        return corrected

    async def _handle_turn_result(self, result: SpeakerTurnResult, *, track_response: bool = True) -> None:
        """Record and dispatch one structurally normalized turn result."""
        transcript = result.transcript.strip()
        response = clean_spoken_response_artifacts(result.response)
        user_text = transcript or response or result.raw_content.strip()
        if not user_text:
            return

        self._repeat.note_reply(response, track=track_response)

        turn_action = normalize_turn_action(result.payload.get("turn_action"))
        selected_input_source = normalize_selected_input_source(result.payload.get("selected_input_source"))
        media_prompt = str(result.payload.get("media_analysis_prompt", "")).strip()
        media_action = normalize_media_analysis_action(result.payload.get("media_analysis_action"))
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
        needs_thinking = normalize_turn_action(payload.get("turn_action")) == "think"
        if not (needs_thinking or repeated):
            return
        if repeated and not needs_thinking and (self._capture_cooldown > 0 or is_affirmation(transcript)):
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
            # The Speaker registers no tools, so it can use the forced JSON
            # response format its action envelope needs. It parses and emits the
            # user transcript itself, so the service's own tag-based transcript
            # extraction stays off.
            extra={"response_format": {"type": "json_object"}, **dict(extra_params or {})},
            settings=NvidiaOmniSettings(
                model=model_id,
                max_tokens=parse_env_int("OMNI_MAX_TOKENS", 8192, min_value=64),
                temperature=parse_env_float("OMNI_TEMPERATURE", 0.7, min_value=0.0),
                top_p=parse_env_float("OMNI_TOP_P", 0.95, min_value=0.0),
                emit_transcriptions=False,
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
