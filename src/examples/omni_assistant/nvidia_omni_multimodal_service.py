# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Nemotron Omni multimodal-input, text-output LLM service.

Designed to be upstream-compatible with Pipecat's :class:`LLMService`
contract:

* accepts text, audio, image, and video inputs as OpenAI-compatible
  multimodal content parts;
* emits standard Pipecat frames only, with optional ``TranscriptionFrame``
  output when structured audio parsing is enabled via
  ``Settings.emit_transcriptions``;
* records processing, TTFB, and token-usage metrics through the standard
  :class:`LLMService` helpers;
* exposes the ``_on_turn_result``,
  ``_structured_response_control_fields``, and
  ``_should_emit_streamed_structured_response`` extension hooks so
  application-layer policy (attachment routing, webcam summaries, visual
  barge-in orchestration, etc.) lives outside this provider class.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import time
import wave
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger
from openai import AsyncOpenAI
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InputImageRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMTextFrame,
    StartFrame,
    TranscriptionFrame,
    UserImageRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import NOT_GIVEN, LLMSettings, _NotGiven
from pipecat.utils.time import time_now_iso8601

InputModality = Literal["text", "audio", "image", "video"]
OutputModality = Literal["text"]
OpenAIContentPart = dict[str, Any]
OpenAIMessage = dict[str, Any]

DEFAULT_AUDIO_RESPONSE_INSTRUCTION = (
    "Listen to the user's speech in the attached audio and answer them helpfully in plain "
    "speech-ready text. Do not use markdown, bullets, asterisks, or visual formatting symbols."
)

JSON_AUDIO_RESPONSE_INSTRUCTION = (
    "Listen to the user's speech in the attached audio. First, transcribe what they said exactly. "
    "Then answer them helpfully in plain speech-ready text. Return strict JSON with exactly "
    "these fields: "
    '{"transcript": "...", "response": "..."}'
)

DEFAULT_MULTIMODAL_RESPONSE_INSTRUCTION = (
    "Answer the user's latest request using the provided context and any attached media. "
    "Return plain text suitable for speech. Do not use markdown, bullets, asterisks, or visual "
    "formatting symbols."
)

DEFAULT_INPUT_MODALITIES: tuple[InputModality, ...] = ("text", "audio", "image", "video")
SUPPORTED_INPUT_MODALITIES: set[str] = set(DEFAULT_INPUT_MODALITIES)


@dataclass
class NvidiaOmniSettings(LLMSettings):
    """Runtime settings for ``NvidiaOmniMultimodalService``.

    ``output_modality`` is intentionally text-only for cascaded pipelines: the
    service replaces ASR+LLM, while a normal downstream TTS service speaks the
    emitted ``LLMTextFrame`` content.
    """

    stream: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    response_format: dict[str, Any] | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    input_modalities: tuple[InputModality, ...] | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    output_modality: OutputModality | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    emit_transcriptions: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    audio_response_instruction: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    multimodal_response_instruction: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    min_user_audio_secs: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    pre_speech_buffer_secs: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    image_mime_type: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


@dataclass(frozen=True)
class NvidiaOmniTurnResult:
    """Parsed Omni response."""

    transcript: str = ""
    response: str = ""
    raw_content: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NvidiaOmniInferenceResult:
    """Result from an out-of-pipeline Omni inference."""

    text: str = ""
    reasoning: str = ""
    finish_reason: str = ""


class NvidiaOmniMultimodalService(LLMService):
    """Nemotron Omni multimodal-input, text-output LLM service.

    Typical cascaded pipeline:

    ``transport.input() -> VAD/UserTurnProcessor -> NvidiaOmniMultimodalService
    -> TTS -> transport.output() -> LLMAssistantAggregator``

    By default, the assistant response is plain model text.  If callers opt into
    ``emit_transcriptions=True``, the service can also parse and emit a
    ``TranscriptionFrame`` from the same Omni response.
    """

    Settings = NvidiaOmniSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str | None = None,
        context: LLMContext | None = None,
        settings: Settings | None = None,
        request_timeout_secs: float = 120.0,
        extra: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Initialize the multimodal Omni service.

        Args:
            api_key: NVIDIA API key. For local deployments, an empty key is accepted.
            base_url: OpenAI-compatible endpoint base URL.
            model: Deprecated direct model override. Prefer ``settings.model``.
            context: Shared LLM context. If omitted, the first ``LLMContextFrame`` supplies it.
            settings: Runtime-updatable service settings.
            request_timeout_secs: HTTP client timeout.
            extra: Extra request fields merged into every chat completion call.
            **kwargs: Additional ``LLMService`` arguments.
        """
        default_settings = self.Settings(
            model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            system_instruction=None,
            temperature=0.6,
            max_tokens=65536,
            top_p=0.95,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
            stream=True,
            response_format=None,
            input_modalities=DEFAULT_INPUT_MODALITIES,
            output_modality="text",
            emit_transcriptions=False,
            audio_response_instruction=None,
            multimodal_response_instruction=DEFAULT_MULTIMODAL_RESPONSE_INSTRUCTION,
            min_user_audio_secs=0.3,
            pre_speech_buffer_secs=0.2,
            image_mime_type="image/jpeg",
        )
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model
        if settings is not None:
            default_settings.apply_update(settings)
        self._validate_settings(default_settings)

        super().__init__(settings=default_settings, **kwargs)
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",
            timeout=request_timeout_secs,
        )
        self._base_url = base_url
        self._context = context
        self._extra = dict(extra or {})

        self._audio_buffer: list[bytes] = []
        self._pre_speech_buffer: list[bytes] = []
        self._pending_content_parts: list[OpenAIContentPart] = []
        self._sample_rate = 16000
        self._channels = 1
        self._user_speaking = False
        self._bot_responding = False
        self._pending_request: asyncio.Task[None] | None = None
        self._pending_request_is_audio = False
        self._last_user_eou_at: float | None = None

    def can_generate_metrics(self) -> bool:
        """Return whether Pipecat metrics can be emitted."""
        return True

    async def start(self, frame: StartFrame) -> None:
        """Start the service and reset per-session buffers."""
        await super().start(frame)
        self._audio_buffer = []
        self._pre_speech_buffer = []
        self._pending_content_parts = []
        self._user_speaking = False
        self._bot_responding = False
        self._pending_request_is_audio = False
        self._last_user_eou_at = None

    async def stop(self, frame) -> None:
        """Stop the service and cancel an in-flight Omni request."""
        await self._cancel_pending_request()
        await super().stop(frame)

    async def cancel(self, frame) -> None:
        """Cancel the service and any in-flight Omni request."""
        await self._cancel_pending_request()
        await super().cancel(frame)

    async def run_inference(
        self,
        context: LLMContext,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
    ) -> str | None:
        """Run a text-only, out-of-pipeline inference."""
        messages = self._messages_from_context(context)
        if system_instruction:
            messages.insert(0, {"role": "system", "content": system_instruction})
        request_kwargs = self._build_request_kwargs(messages=messages, has_audio=False)
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        request_kwargs["stream"] = False
        completion = await self._client.chat.completions.create(**request_kwargs)
        message = completion.choices[0].message if completion.choices else None
        return _extract_text_content(getattr(message, "content", ""))

    async def run_multimodal_inference(
        self,
        context: LLMContext,
        *,
        max_tokens: int | None = None,
        reasoning_budget: int | None = None,
        temperature: float | None = None,
        stream: bool = False,
        on_text_delta: Callable[[str], Awaitable[None]] | None = None,
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> NvidiaOmniInferenceResult:
        """Run an out-of-pipeline multimodal inference with optional streaming callbacks.

        This is useful for worker-style agents that need Omni multimodal input
        support without inserting the service into a Pipecat pipeline.
        """
        request_kwargs = self._build_request_kwargs(messages=self._messages_from_context(context), has_audio=False)
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if reasoning_budget is not None:
            extra_body = dict(request_kwargs.get("extra_body") or {})
            extra_body["reasoning_budget"] = reasoning_budget
            request_kwargs["extra_body"] = extra_body
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if not stream:
            request_kwargs["stream"] = False
            completion = await self._client.chat.completions.create(**request_kwargs)
            choice = completion.choices[0] if completion.choices else None
            message = choice.message if choice else None
            return NvidiaOmniInferenceResult(
                text=_extract_text_content(getattr(message, "content", "")).strip(),
                finish_reason=str(getattr(choice, "finish_reason", "") or ""),
            )

        request_kwargs["stream"] = True
        text = ""
        reasoning = ""
        finish_reason = ""
        response_stream = await self._client.chat.completions.create(**request_kwargs)
        async for chunk in response_stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if choice.finish_reason:
                finish_reason = str(choice.finish_reason)
            reasoning_delta = _extract_delta_reasoning_content(delta)
            text_delta = _extract_text_content(getattr(delta, "content", ""))
            if reasoning_delta:
                reasoning += reasoning_delta
                if on_reasoning_delta is not None:
                    await on_reasoning_delta(reasoning_delta)
            if text_delta:
                text += text_delta
                if on_text_delta is not None:
                    await on_text_delta(text_delta)
        return NvidiaOmniInferenceResult(
            text=text.strip(),
            reasoning=reasoning.strip(),
            finish_reason=finish_reason,
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process pipeline frames for multimodal Omni turns."""
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            await self.stop_all_metrics()
            await self._cancel_pending_request()
            self._bot_responding = False
            if not self._user_speaking:
                self._audio_buffer = []
                self._pre_speech_buffer = []
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_responding = True
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_responding = False
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMContextFrame):
            self._context = frame.context
            await self._maybe_run_text_or_multimodal_turn(frame.context)
            return

        if isinstance(frame, LLMRunFrame):
            await self._maybe_run_text_or_multimodal_turn(self._context)
            return

        if isinstance(frame, InputImageRawFrame):
            await self._handle_image_frame(frame)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, InputAudioRawFrame):
            self._handle_audio_frame(frame)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)):
            await self._handle_user_started(frame)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, UserStoppedSpeakingFrame):
            await self._handle_user_stopped(frame)
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    async def _handle_user_started(self, frame: Frame) -> None:
        if self._user_speaking:
            return
        if self._bot_responding:
            logger.info("NVIDIA Omni: barge-in detected, interrupting bot output")
            await self._cancel_pending_request()
            self._bot_responding = False
            await self.broadcast_interruption()
        self._user_speaking = True
        self._audio_buffer = list(self._pre_speech_buffer)
        self._pre_speech_buffer = []

    async def _handle_user_stopped(self, frame: Frame) -> None:
        if not self._user_speaking:
            return
        self._user_speaking = False
        self._last_user_eou_at = time.time()
        await self._maybe_run_audio_turn()

    def _handle_audio_frame(self, frame: InputAudioRawFrame) -> None:
        self._sample_rate = frame.sample_rate
        self._channels = frame.num_channels
        if not self._modality_enabled("audio"):
            return
        if self._user_speaking:
            self._audio_buffer.append(frame.audio)
        else:
            self._append_pre_speech_audio(frame)

    async def _handle_image_frame(self, frame: InputImageRawFrame) -> None:
        if not self._modality_enabled("image"):
            return
        if isinstance(frame, UserImageRawFrame) and frame.append_to_context is False:
            return
        try:
            self._pending_content_parts.append(
                input_image_frame_to_message_part(frame, mime_type=str(self._settings.image_mime_type))
            )
            if isinstance(frame, UserImageRawFrame) and frame.text:
                self._pending_content_parts.append(text_message_part(frame.text))
        except Exception as exc:
            logger.warning(f"NVIDIA Omni: could not encode input image frame: {exc}")

    async def _maybe_run_audio_turn(self) -> None:
        if self._bot_responding:
            logger.debug("NVIDIA Omni: ignoring audio turn while bot is responding")
            return

        audio_chunks = list(self._audio_buffer)
        content_parts = list(self._pending_content_parts)
        self._audio_buffer = []
        self._pre_speech_buffer = []
        self._pending_content_parts = []
        if not audio_chunks:
            logger.debug("NVIDIA Omni: no buffered audio for audio turn, skipping")
            return

        audio_payload = b"".join(audio_chunks)
        min_bytes = int(self._sample_rate * self._channels * 2 * float(self._settings.min_user_audio_secs))
        if len(audio_payload) < min_bytes:
            logger.debug(
                f"NVIDIA Omni: dropping {len(audio_payload)} bytes "
                f"(< {float(self._settings.min_user_audio_secs) * 1000:.0f} ms)"
            )
            return

        if self._pending_request is not None and not self._pending_request.done():
            logger.debug("NVIDIA Omni: previous turn still in flight, cancelling it to run the newer turn")
            await self.stop_all_metrics()
            await self._cancel_pending_request()

        eou_at = self._last_user_eou_at
        self._last_user_eou_at = None

        async def _run() -> None:
            await self._run_omni_turn(
                audio_payload=audio_payload,
                content_parts=content_parts,
                context=self._context,
                metrics_start_time=eou_at,
            )

        self._pending_request = self.create_task(_run(), name="nvidia-omni-audio-turn")
        self._pending_request_is_audio = True

    async def _maybe_run_text_or_multimodal_turn(self, context: LLMContext | None) -> None:
        if self._bot_responding:
            logger.debug("NVIDIA Omni: ignoring text/multimodal trigger while bot is responding")
            return
        if not _context_has_pending_user_message(context) and not self._pending_content_parts:
            logger.debug("NVIDIA Omni: no pending user text or media, skipping text/multimodal turn")
            return

        if self._pending_request is not None and not self._pending_request.done():
            if self._pending_request_is_audio:
                logger.debug("NVIDIA Omni: audio turn in flight, ignoring redundant context/run trigger")
                return
            logger.debug("NVIDIA Omni: previous turn still in flight, cancelling it to run the newer turn")
            await self.stop_all_metrics()
            await self._cancel_pending_request()

        content_parts = list(self._pending_content_parts)
        self._pending_content_parts = []

        async def _run() -> None:
            await self._run_omni_turn(
                audio_payload=b"",
                content_parts=content_parts,
                context=context,
                metrics_start_time=None,
            )

        self._pending_request = self.create_task(_run(), name="nvidia-omni-text-turn")
        self._pending_request_is_audio = False

    async def _run_omni_turn(
        self,
        *,
        audio_payload: bytes,
        content_parts: Sequence[OpenAIContentPart],
        context: LLMContext | None,
        metrics_start_time: float | None,
    ) -> None:
        has_audio = bool(audio_payload)
        messages = self._messages_from_context(context)
        current_parts = list(content_parts)
        if has_audio:
            current_parts.append(audio_message_part(audio_payload, self._sample_rate, self._channels))
            current_parts.append(text_message_part(self._audio_response_instruction()))
        elif current_parts:
            current_parts.append(text_message_part(str(self._settings.multimodal_response_instruction)))

        if current_parts:
            messages.append({"role": "user", "content": current_parts})

        request_kwargs = self._build_request_kwargs(messages=messages, has_audio=has_audio)
        logger.info(
            "NVIDIA Omni request: "
            f"base_url={self._base_url}, model={self._settings.model}, "
            f"mode={'audio' if has_audio else 'text'}, "
            f"content_parts={len(current_parts)}, context_messages={len(messages) - (1 if current_parts else 0)}"
        )

        if bool(self._settings.stream):
            await self._run_streaming_omni_turn(request_kwargs, has_audio, metrics_start_time)
        else:
            await self._run_non_streaming_omni_turn(request_kwargs, has_audio, metrics_start_time)

    def _build_request_kwargs(self, *, messages: list[OpenAIMessage], has_audio: bool) -> dict[str, Any]:
        request_kwargs: dict[str, Any] = {
            "model": self._settings.model,
            "messages": messages,
            "max_tokens": self._settings.max_tokens,
            "temperature": self._settings.temperature,
        }
        if self._settings.top_p is not None:
            request_kwargs["top_p"] = self._settings.top_p
        if self._expects_structured_audio_response(has_audio) and self._settings.response_format:
            request_kwargs["response_format"] = self._settings.response_format
        request_kwargs.update(self._extra)
        return request_kwargs

    def _audio_response_instruction(self) -> str:
        if self._settings.audio_response_instruction:
            return str(self._settings.audio_response_instruction)
        if self._settings.emit_transcriptions:
            return JSON_AUDIO_RESPONSE_INSTRUCTION
        return DEFAULT_AUDIO_RESPONSE_INSTRUCTION

    def _expects_structured_audio_response(self, has_audio: bool) -> bool:
        return has_audio and bool(self._settings.emit_transcriptions)

    async def _run_non_streaming_omni_turn(
        self,
        request_kwargs: dict[str, Any],
        has_audio: bool,
        metrics_start_time: float | None,
    ) -> None:
        await self.start_processing_metrics(start_time=metrics_start_time)
        await self.start_ttfb_metrics(start_time=metrics_start_time)
        try:
            completion = await self._client.chat.completions.create(**request_kwargs)
        except asyncio.CancelledError:
            await self.stop_processing_metrics()
            raise
        except Exception as exc:
            await self.stop_processing_metrics()
            logger.exception(f"NVIDIA Omni request failed: {exc}")
            await self.push_error_frame(ErrorFrame(error=f"NVIDIA Omni request failed: {exc}", fatal=False))
            return

        message = completion.choices[0].message if completion.choices else None
        raw_content = _extract_text_content(getattr(message, "content", ""))
        await self._emit_llm_usage_metrics(getattr(completion, "usage", None))
        structured_audio_response = self._expects_structured_audio_response(has_audio)
        result = self._parse_turn_result(raw_content, parse_json=structured_audio_response)
        result = await self._on_turn_result(result) or result
        if structured_audio_response and result.transcript:
            await self._emit_user_transcript(result.transcript)

        response = result.response
        if not response:
            response = "Sorry, I could not generate a response."
        await self.push_frame(LLMFullResponseStartFrame())
        await self.stop_ttfb_metrics()
        await self.push_frame(LLMTextFrame(text=response))
        await self.stop_processing_metrics()
        await self.push_frame(LLMFullResponseEndFrame())

    async def _run_streaming_omni_turn(
        self,
        request_kwargs: dict[str, Any],
        has_audio: bool,
        metrics_start_time: float | None,
    ) -> None:
        request_kwargs = dict(request_kwargs)
        request_kwargs["stream"] = True
        request_kwargs.setdefault("stream_options", {"include_usage": True})
        structured_audio_response = self._expects_structured_audio_response(has_audio)
        transcript_streamer = _JsonStringFieldStreamer("transcript") if structured_audio_response else None
        response_streamer = _JsonStringFieldStreamer("response") if structured_audio_response else None
        control_streamers = (
            {
                field_name: _JsonStringFieldStreamer(field_name)
                for field_name in self._structured_response_control_fields()
            }
            if structured_audio_response
            else {}
        )
        control_field_values: dict[str, str] = {}
        suppress_structured_response_stream = False
        transcript_emitted = False
        streamed_transcript = ""
        raw_content = ""
        response_started = False
        sentence_buffer = ""
        ttfb_stopped = False

        await self.start_processing_metrics(start_time=metrics_start_time)
        await self.start_ttfb_metrics(start_time=metrics_start_time)
        try:
            stream = await self._client.chat.completions.create(**request_kwargs)
            async for chunk in stream:
                await self._emit_llm_usage_metrics(getattr(chunk, "usage", None))
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = _extract_text_content(getattr(delta, "content", ""))
                if not content:
                    continue
                raw_content += content

                if not ttfb_stopped:
                    await self.stop_ttfb_metrics()
                    ttfb_stopped = True

                if transcript_streamer and not transcript_emitted:
                    streamed_transcript += transcript_streamer.feed(content)
                    if transcript_streamer.done:
                        transcript = streamed_transcript.strip()
                        if transcript:
                            await self._emit_user_transcript(transcript)
                            transcript_emitted = True

                for field_name, streamer in control_streamers.items():
                    if streamer.done:
                        continue
                    control_field_values[field_name] = control_field_values.get(field_name, "") + streamer.feed(content)

                response_chunk = response_streamer.feed(content) if response_streamer else content
                if not response_chunk:
                    continue
                if structured_audio_response:
                    if not suppress_structured_response_stream and not self._should_emit_streamed_structured_response(
                        control_field_values
                    ):
                        suppress_structured_response_stream = True
                    if suppress_structured_response_stream:
                        continue
                if not response_started:
                    response_started = True
                    await self.push_frame(LLMFullResponseStartFrame())
                    if not ttfb_stopped:
                        await self.stop_ttfb_metrics()
                        ttfb_stopped = True
                sentence_buffer += response_chunk
                sentences, sentence_buffer = _pop_complete_sentences(sentence_buffer)
                for sentence in sentences:
                    response_started = await self._emit_assistant_text(
                        sentence,
                        response_started=response_started,
                    )
        except asyncio.CancelledError:
            await self.stop_processing_metrics()
            raise
        except Exception as exc:
            await self.stop_processing_metrics()
            logger.exception(f"NVIDIA Omni streaming request failed: {exc}")
            await self.push_error_frame(ErrorFrame(error=f"NVIDIA Omni streaming request failed: {exc}", fatal=False))
            return

        trailing_text = sentence_buffer.strip()
        if structured_audio_response:
            result = self._parse_turn_result(raw_content, parse_json=True)
            corrected_result = await self._on_turn_result(result)
            if corrected_result is not None:
                result = corrected_result
                if response_started and result.response:
                    response_started = await self._emit_assistant_text(
                        result.response,
                        response_started=response_started,
                    )
            if result.transcript and not transcript_emitted:
                await self._emit_user_transcript(result.transcript)
            if trailing_text:
                response_started = await self._emit_assistant_text(
                    trailing_text,
                    response_started=response_started,
                )
            elif not response_started and result.response:
                response_started = await self._emit_assistant_text(
                    result.response,
                    response_started=response_started,
                )
        elif trailing_text:
            response_started = await self._emit_assistant_text(
                trailing_text,
                response_started=response_started,
            )
        elif not response_started and raw_content.strip():
            response_started = await self._emit_assistant_text(
                raw_content.strip(),
                response_started=response_started,
            )

        await self.stop_processing_metrics()
        if response_started:
            await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.stop_ttfb_metrics()

    async def _emit_assistant_text(self, text: str, *, response_started: bool) -> bool:
        cleaned = text.strip()
        if not cleaned:
            return response_started
        if not response_started:
            response_started = True
            await self.push_frame(LLMFullResponseStartFrame())
            await self.stop_ttfb_metrics()
        await self.push_frame(LLMTextFrame(text=f"{cleaned} "))
        return response_started

    def _structured_response_control_fields(self) -> tuple[str, ...]:
        """Return structured JSON string fields needed before streaming response text."""
        return ()

    def _should_emit_streamed_structured_response(self, field_values: Mapping[str, str]) -> bool:
        """Allow subclasses to suppress streamed response text until full JSON is parsed."""
        return True

    async def _emit_llm_usage_metrics(self, usage: Any) -> None:
        tokens = _llm_token_usage_from_openai_usage(usage)
        if tokens is not None:
            await self.start_llm_usage_metrics(tokens)

    async def _emit_user_transcript(self, transcript: str) -> None:
        if self._context is not None:
            self._context.add_message({"role": "user", "content": transcript})
        await self.push_frame(
            TranscriptionFrame(
                text=transcript,
                user_id="user",
                timestamp=time_now_iso8601(),
                result=transcript,
            ),
            FrameDirection.UPSTREAM,
        )

    def _parse_turn_result(self, raw_content: str, *, parse_json: bool) -> NvidiaOmniTurnResult:
        if not parse_json:
            return NvidiaOmniTurnResult(response=raw_content.strip(), raw_content=raw_content)
        payload = _extract_json_payload(raw_content)
        transcript = str(payload.get("transcript", "")).strip()
        response = str(payload.get("response", "")).strip()
        if not payload:
            logger.warning(
                "NVIDIA Omni: audio response did not parse as JSON; "
                f"user transcript will be missing. Raw response: {raw_content[:500]!r}"
            )
        return NvidiaOmniTurnResult(
            transcript=transcript,
            response=response,
            raw_content=raw_content,
            payload=payload,
        )

    async def _on_turn_result(self, result: NvidiaOmniTurnResult) -> NvidiaOmniTurnResult | None:
        """Inspect a parsed response and optionally replace it before final emission."""
        return None

    def _messages_from_context(self, context: LLMContext | None) -> list[OpenAIMessage]:
        if context is None:
            return []
        messages: list[OpenAIMessage] = []
        for message in context.get_messages():
            converted = self._normalize_context_message(message)
            if converted is not None:
                messages.append(converted)
        return messages

    def _normalize_context_message(self, message: Any) -> OpenAIMessage | None:
        if not isinstance(message, Mapping):
            return None
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            return None
        content = message.get("content")
        if isinstance(content, str):
            return {"role": role, "content": content}
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
            parts = [part for part in content if self._content_part_allowed(part)]
            if parts:
                return {"role": role, "content": parts}
        return None

    def _content_part_allowed(self, part: Any) -> bool:
        if not isinstance(part, Mapping):
            return False
        modality = _content_part_modality(part)
        return modality is not None and self._modality_enabled(modality)

    def _append_pre_speech_audio(self, frame: InputAudioRawFrame) -> None:
        self._pre_speech_buffer.append(frame.audio)
        bytes_per_second = max(frame.sample_rate * frame.num_channels * 2, 1)
        max_bytes = int(bytes_per_second * float(self._settings.pre_speech_buffer_secs))
        total = sum(len(chunk) for chunk in self._pre_speech_buffer)
        while self._pre_speech_buffer and total > max_bytes:
            total -= len(self._pre_speech_buffer.pop(0))

    def _modality_enabled(self, modality: InputModality) -> bool:
        return modality in set(self._settings.input_modalities)

    async def _cancel_pending_request(self) -> None:
        if self._pending_request and not self._pending_request.done():
            self._pending_request.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pending_request
        self._pending_request = None
        self._pending_request_is_audio = False

    @staticmethod
    def _validate_settings(settings: Settings) -> None:
        modalities = tuple(settings.input_modalities)
        unknown = sorted(set(modalities) - SUPPORTED_INPUT_MODALITIES)
        if unknown:
            raise ValueError(f"Unsupported NvidiaOmni input modalities: {unknown}")
        if settings.output_modality != "text":
            raise ValueError("NvidiaOmniMultimodalService currently supports output_modality='text' only")


# Short alias used by pipelines that prefer ``NvidiaOmniService``.
NvidiaOmniService = NvidiaOmniMultimodalService


def text_message_part(text: str) -> OpenAIContentPart:
    """Create an OpenAI-compatible text content part."""
    return {"type": "text", "text": text}


def audio_message_part(audio: bytes, sample_rate: int, channels: int) -> OpenAIContentPart:
    """Create an OpenAI-compatible audio content part from int16 PCM audio."""
    return {"type": "audio_url", "audio_url": {"url": audio_to_data_url(audio, sample_rate, channels)}}


def image_message_part(data: bytes, mime_type: str = "image/jpeg") -> OpenAIContentPart:
    """Create an OpenAI-compatible image content part."""
    return {"type": "image_url", "image_url": {"url": data_to_data_url(data, mime_type)}}


def video_message_part(data: bytes, mime_type: str = "video/mp4") -> OpenAIContentPart:
    """Create an OpenAI-compatible video content part."""
    return {"type": "video_url", "video_url": {"url": data_to_data_url(data, mime_type)}}


def media_message_part(data: bytes, *, modality: InputModality, mime_type: str) -> OpenAIContentPart:
    """Create a multimodal content part for text/audio/image/video media."""
    if modality == "text":
        return text_message_part(data.decode("utf-8"))
    if modality == "audio":
        return {"type": "audio_url", "audio_url": {"url": data_to_data_url(data, mime_type)}}
    if modality == "image":
        return image_message_part(data, mime_type)
    if modality == "video":
        return video_message_part(data, mime_type)
    raise ValueError(f"Unsupported modality: {modality}")


def input_image_frame_to_message_part(
    frame: InputImageRawFrame,
    *,
    mime_type: str = "image/jpeg",
) -> OpenAIContentPart:
    """Encode an ``InputImageRawFrame`` as an image content part."""
    return image_message_part(encode_image_frame(frame, mime_type=mime_type), mime_type=mime_type)


def data_to_data_url(data: bytes, mime_type: str) -> str:
    """Encode bytes as a data URL."""
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def audio_to_data_url(audio: bytes, sample_rate: int, channels: int) -> str:
    """Encode little-endian int16 PCM bytes as a WAV data URL."""
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio)
        return data_to_data_url(buffer.getvalue(), "audio/wav")


def encode_image_frame(frame: InputImageRawFrame, *, mime_type: str = "image/jpeg") -> bytes:
    """Encode a raw Pipecat image frame to JPEG/PNG bytes for multimodal APIs."""
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required to encode InputImageRawFrame media parts") from exc

    image_format = (frame.format or "RGB").upper()
    image = Image.frombytes(image_format, frame.size, frame.image)
    with io.BytesIO() as buffer:
        if mime_type == "image/png":
            image.save(buffer, format="PNG")
        else:
            if image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
            image.save(buffer, format="JPEG")
        return buffer.getvalue()


def _content_part_modality(part: Mapping[str, Any]) -> InputModality | None:
    part_type = str(part.get("type") or "")
    if part_type == "text":
        return "text"
    if part_type == "audio_url":
        return "audio"
    if part_type == "image_url":
        return "image"
    if part_type == "video_url":
        return "video"
    return None


def _extract_text_content(raw: Any) -> str:
    """Flatten OpenAI content payload variants into plain text."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, Mapping):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(p for p in parts if p)
    return str(raw)


def _extract_delta_reasoning_content(delta: Any) -> str:
    """Return provider-specific reasoning content from a streamed delta."""
    for attr in ("reasoning", "reasoning_content"):
        value = getattr(delta, attr, None)
        if isinstance(value, str) and value:
            return value

    model_extra = getattr(delta, "model_extra", None)
    if isinstance(model_extra, dict):
        for key in ("reasoning", "reasoning_content"):
            value = model_extra.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _extract_json_payload(text: str) -> dict[str, Any]:
    """Parse raw or fenced JSON from model output."""
    if not text:
        return {}
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            parsed = json.loads(candidate[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def _llm_token_usage_from_openai_usage(usage: Any) -> LLMTokenUsage | None:
    """Convert OpenAI-compatible usage objects into Pipecat token metrics."""
    if usage is None:
        return None

    prompt_tokens = int(_usage_value(usage, "prompt_tokens") or 0)
    completion_tokens = int(_usage_value(usage, "completion_tokens") or 0)
    total_tokens = int(_usage_value(usage, "total_tokens") or (prompt_tokens + completion_tokens))
    if prompt_tokens == 0 and completion_tokens == 0 and total_tokens == 0:
        return None

    prompt_details = _usage_value(usage, "prompt_tokens_details")
    completion_details = _usage_value(usage, "completion_tokens_details")
    return LLMTokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cache_read_input_tokens=_usage_value(prompt_details, "cached_tokens"),
        reasoning_tokens=_usage_value(completion_details, "reasoning_tokens"),
    )


def _usage_value(source: Any, key: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _context_has_pending_user_message(context: LLMContext | None) -> bool:
    if context is None:
        return False
    for message in reversed(list(context.get_messages())):
        if not isinstance(message, Mapping):
            continue
        role = message.get("role")
        if role == "assistant":
            return False
        if role == "user" and message.get("content"):
            return True
    return False


def _pop_complete_sentences(buffer: str) -> tuple[list[str], str]:
    """Return complete sentence chunks and the remaining incomplete suffix."""
    sentences: list[str] = []
    start = 0
    for idx, ch in enumerate(buffer):
        if ch not in ".!?":
            continue
        next_idx = idx + 1
        if next_idx < len(buffer) and buffer[next_idx] not in " \n\r\t\"'”’)]}":
            continue
        sentence = buffer[start:next_idx].strip()
        if sentence:
            sentences.append(sentence)
        start = next_idx
    return sentences, buffer[start:]


class _JsonStringFieldStreamer:
    """Incrementally extract a string field from streamed JSON text."""

    def __init__(self, field_name: str) -> None:
        self._needle = f'"{field_name}"'
        self._state = "search"
        self._buffer = ""
        self._escaped = False
        self._unicode_remaining = 0
        self._unicode_buffer = ""
        self.done = False

    def feed(self, text: str) -> str:
        if self.done or not text:
            return ""
        if self._state != "in_string":
            self._buffer += text
            emitted = self._advance_to_string()
            if self._state != "in_string":
                return ""
            text = emitted
        return self._consume_string_chars(text)

    def _advance_to_string(self) -> str:
        while True:
            if self._state == "search":
                idx = self._buffer.find(self._needle)
                if idx < 0:
                    self._buffer = self._buffer[-len(self._needle) :]
                    return ""
                self._buffer = self._buffer[idx + len(self._needle) :]
                self._state = "colon"
            if self._state == "colon":
                stripped = self._buffer.lstrip()
                if not stripped:
                    self._buffer = ""
                    return ""
                if stripped[0] != ":":
                    self._state = "search"
                    self._buffer = stripped
                    continue
                self._buffer = stripped[1:]
                self._state = "quote"
            if self._state == "quote":
                stripped = self._buffer.lstrip()
                if not stripped:
                    self._buffer = ""
                    return ""
                if stripped[0] != '"':
                    self._state = "search"
                    self._buffer = stripped
                    continue
                self._state = "in_string"
                emitted = stripped[1:]
                self._buffer = ""
                return emitted

    def _consume_string_chars(self, text: str) -> str:
        out: list[str] = []
        for ch in text:
            if self._unicode_remaining:
                self._unicode_buffer += ch
                self._unicode_remaining -= 1
                if self._unicode_remaining == 0:
                    try:
                        out.append(chr(int(self._unicode_buffer, 16)))
                    except ValueError:
                        out.append(f"\\u{self._unicode_buffer}")
                    self._unicode_buffer = ""
                continue
            if self._escaped:
                self._escaped = False
                if ch == "u":
                    self._unicode_remaining = 4
                    self._unicode_buffer = ""
                else:
                    out.append(
                        {
                            '"': '"',
                            "\\": "\\",
                            "/": "/",
                            "b": "\b",
                            "f": "\f",
                            "n": "\n",
                            "r": "\r",
                            "t": "\t",
                        }.get(ch, ch)
                    )
                continue
            if ch == "\\":
                self._escaped = True
                continue
            if ch == '"':
                self.done = True
                break
            out.append(ch)
        return "".join(out)
