# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""NVIDIA Nemotron Omni service implementation.

This module provides a service for interacting with NVIDIA's Nemotron Omni
models, which accept speech as well as text input and reply with text. It
extends ``NvidiaLLMService``, so reasoning frames, NIM's incremental token
accounting, and every OpenAI-compatible behaviour come from there unchanged.

``Settings.input_modalities`` selects what starts a pipeline turn. ``"text"``
expects an upstream STT service to produce the user turn, while ``"audio"`` lets
Omni buffer user speech on VAD boundaries and perform ASR and generation in a
single request. Both turn kinds run through the inherited completion path, so
tool calling, streaming, metrics, and context aggregation are unchanged, and a
tool result is answered whichever modality asked for the call.

Media travels in Pipecat's universal LLM context, so ``create_audio_message()``,
``add_audio_frames_message()``, and the image equivalents work here as they do
for any other service: ``NvidiaOmniLLMAdapter`` converts each content part to
the shape NIM names it by. Video, which the universal context has no part for,
and one-shot inference with streaming or reasoning callbacks are available
through ``run_multimodal_inference()`` and the module-level part helpers.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import time
import wave
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast, get_args

import httpx
from loguru import logger
from openai import AsyncOpenAI, DefaultAsyncHttpxClient, NotGiven
from openai.types.chat import ChatCompletionMessageParam
from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter, OpenAILLMInvocationParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMServiceMetadataFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextMessage
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.nvidia.llm import NvidiaLLMService, NvidiaLLMSettings
from pipecat.services.settings import NOT_GIVEN, _NotGiven, assert_given
from pipecat.utils.time import time_now_iso8601

InputModality = Literal["text", "audio"]
MediaModality = Literal["text", "audio", "image", "video"]
OpenAIContentPart = dict[str, Any]

DEFAULT_AUDIO_RESPONSE_INSTRUCTION = "Listen to the user's speech in the attached audio and answer them."

TRANSCRIPT_AUDIO_RESPONSE_INSTRUCTION = (
    "Listen to the user's speech in the attached audio. First write exactly what the user "
    "said inside <transcript>...</transcript>, then write your spoken reply inside "
    "<response>...</response>."
)

SUPPORTED_INPUT_MODALITIES: frozenset[str] = frozenset(get_args(InputModality))
DEFAULT_INPUT_MODALITIES: tuple[InputModality, ...] = ("text", "audio")

_TRANSCRIPT_OPEN = "<transcript>"
_TRANSCRIPT_CLOSE = "</transcript>"
_RESPONSE_OPEN = "<response>"
_RESPONSE_CLOSE = "</response>"


@dataclass
class NvidiaOmniSettings(NvidiaLLMSettings):
    """Settings for NvidiaOmniLLMService.

    Extends ``NvidiaLLMSettings``, so every standard OpenAI-compatible field
    (sampling, penalties, seed, tools, ``extra``) is inherited unchanged.

    Parameters:
        input_modalities: What starts a pipeline turn. ``"text"`` runs turns from
            upstream STT output; ``"audio"`` buffers user speech and lets Omni do
            ASR itself. This does not limit what a turn may carry: media placed
            in the context, an image message for instance, is sent with whichever
            turn kind picks the context up.
        emit_transcriptions: Whether audio turns should also produce a
            ``TranscriptionFrame`` for the user's speech. Enabling it opts into a
            response contract: the model is asked for
            ``<transcript>``/``<response>`` sections, which the service strips
            before any text reaches TTS. Buffered audio is sent as a transient
            message rather than stored, so this frame is what gives the user side
            of an audio conversation its history: enable it for conversations
            that need one across turns, and for tool calling on audio turns,
            whose follow-up completion would otherwise hold no record of the
            request the user spoke.
        audio_response_instruction: Overrides the instruction appended to audio
            turns.
        min_user_audio_secs: Shortest buffered utterance that starts a turn.
        pre_speech_buffer_secs: Amount of pre-speech audio retained so the start
            of an utterance is not clipped.
    """

    input_modalities: tuple[InputModality, ...] | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    emit_transcriptions: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    audio_response_instruction: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    min_user_audio_secs: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    pre_speech_buffer_secs: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


@dataclass(frozen=True)
class NvidiaOmniInferenceResult:
    """Result of an out-of-pipeline Omni inference."""

    text: str = ""
    reasoning: str = ""
    finish_reason: str = ""


class NvidiaOmniLLMAdapter(OpenAILLMAdapter):
    """Names the universal context's media parts the way NIM's Omni endpoint does.

    Pipecat's universal context carries audio as an ``input_audio`` part holding
    base64 WAV data, which is what ``LLMContext.create_audio_message()`` and
    ``add_audio_frames_message()`` produce. Omni reads a data URL under
    ``audio_url`` instead, so the two are reconciled here, at the provider
    boundary, leaving every caller free to build a context the standard way.
    Image parts already agree, and text is untouched.
    """

    def _from_universal_context_messages(
        self,
        messages: list[LLMContextMessage],
        *,
        convert_developer_to_user: bool,
    ) -> list[ChatCompletionMessageParam]:
        converted = super()._from_universal_context_messages(
            messages, convert_developer_to_user=convert_developer_to_user
        )
        return [_to_omni_message(message) for message in converted]


class NvidiaOmniLLMService(NvidiaLLMService):
    """Nemotron Omni LLM service accepting text and speech pipeline input.

    Cascaded pipeline with an upstream STT service::

        transport.input() -> stt -> user_aggregator -> NvidiaOmniLLMService
        -> tts -> transport.output() -> assistant_aggregator

    Or with Omni handling ASR itself, replacing the STT stage::

        transport.input() -> user_aggregator -> NvidiaOmniLLMService
        -> tts -> transport.output() -> assistant_aggregator

    Every turn is executed by the inherited completion path, so tool calling,
    function-call re-prompting, token-level streaming, and metrics match a
    standard OpenAI-compatible service regardless of input modality. Reasoning
    handling, both the ``reasoning_content`` delta field and a leading
    ``<think>`` block, is inherited from ``NvidiaLLMService``; this service adds
    audio turns on top of it.

    A transcribed spoken turn is reported as a ``TranscriptionFrame`` and written
    to the context by the user aggregator, as an STT service's transcript is, so
    the aggregator remains the only writer of the conversation history.
    """

    Settings = NvidiaOmniSettings
    _settings: Settings
    adapter_class = NvidiaOmniLLMAdapter

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
        """Initialize the Omni service.

        Args:
            api_key: NVIDIA API key. An empty key is accepted for local deployments.
            base_url: OpenAI-compatible endpoint base URL.
            model: Deprecated direct model override.

                .. deprecated::
                    Use ``settings=NvidiaOmniLLMService.Settings(model=...)`` instead.

            context: Shared LLM context. If omitted, the first ``LLMContextFrame``
                supplies it.
            settings: Runtime-updatable service settings.
            request_timeout_secs: HTTP client timeout.
            extra: Extra request fields merged into every chat completion call.
            **kwargs: Additional arguments passed to ``NvidiaLLMService``.
        """
        default_settings = self.Settings(
            model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            system_instruction=None,
            temperature=0.6,
            top_p=0.95,
            extra=dict(extra or {}),
            input_modalities=DEFAULT_INPUT_MODALITIES,
            emit_transcriptions=False,
            audio_response_instruction=None,
            min_user_audio_secs=0.3,
            pre_speech_buffer_secs=0.2,
        )
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model
        if settings is not None:
            default_settings.apply_update(settings)
        self._validate_settings(default_settings)

        self._request_timeout_secs = request_timeout_secs
        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)

        self._context = context

        self._audio_buffer: list[bytes] = []
        self._pre_speech_buffer: list[bytes] = []
        self._sample_rate = 16000
        self._channels = 1
        self._user_speaking = False
        self._bot_responding = False
        self._pending_request: asyncio.Task[None] | None = None
        self._pending_request_is_audio = False
        self._last_user_eou_at: float | None = None

        self._active_turn_parts: list[OpenAIContentPart] | None = None
        self._transcript_extractor: _TranscriptResponseExtractor | None = None
        self._transcript_emitted = False
        self._answered_transcript = ""

    def create_client(
        self,
        api_key=None,
        base_url=None,
        organization=None,
        project=None,
        default_headers=None,
        **kwargs,
    ):
        """Create the AsyncOpenAI client with connection pooling and the Omni timeout.

        Args:
            api_key: NVIDIA API key. Empty is accepted for local deployments.
            base_url: OpenAI-compatible endpoint base URL.
            organization: OpenAI organization ID.
            project: OpenAI project ID.
            default_headers: Additional HTTP headers.
            **kwargs: Service arguments forwarded by the base class. Unused
                here, as in the base implementation, and kept for signature
                parity.

        Returns:
            Configured AsyncOpenAI client instance.
        """
        return AsyncOpenAI(
            api_key=api_key or "not-needed",
            base_url=base_url,
            timeout=self._request_timeout_secs,
            organization=organization,
            project=project,
            http_client=DefaultAsyncHttpxClient(
                limits=httpx.Limits(max_keepalive_connections=100, max_connections=1000, keepalive_expiry=None)
            ),
            default_headers=default_headers,
        )

    def service_metadata_frame(self) -> LLMServiceMetadataFrame:
        """Announce an audio pipeline as a realtime service, a cascade as a plain one.

        With audio input Omni transcribes the user itself, so the user's turn
        only exists once the model reports it — after the boundary at which a
        cascade aggregator writes the user message. Realtime mode moves that
        write to the start of the assistant's response, which is late enough for
        the transcript to have landed, and keeps the aggregator the single owner
        of the conversation history.

        Turns stay pipeline-driven, so no turn strategies are recommended and no
        warning about absent turn frames is raised: Omni has no server-side turn
        of its own to align with, and the pipeline's VAD and turn analyzer decide
        every boundary.

        Returns:
            The metadata frame broadcast at pipeline start.
        """
        if not self._modality_enabled("audio"):
            return super().service_metadata_frame()
        return LLMServiceMetadataFrame(service_name=self.name, is_realtime_service=True)

    async def start(self, frame: StartFrame) -> None:
        """Start the service and reset per-session audio state."""
        await super().start(frame)
        self._reset_audio_state()

    async def stop(self, frame: EndFrame) -> None:
        """Stop the service and cancel an in-flight turn."""
        await self._cancel_pending_request()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame) -> None:
        """Cancel the service and any in-flight turn."""
        await self._cancel_pending_request()
        await super().cancel(frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process pipeline frames, driving text and audio Omni turns.

        This calls ``LLMService.process_frame()`` rather than the
        ``BaseOpenAILLMService`` implementation because Omni owns its turn
        timing: an audio turn starts on a VAD boundary, not on every
        ``LLMContextFrame``. The grandparent still applies settings updates, tool
        registration, and function-call cancellation.

        ``LLMContextFrame`` and ``LLMRunFrame`` are consumed here; every other
        frame is forwarded. A user aggregator turns ``LLMRunFrame`` into a
        context frame, so it only reaches this service when the pipeline drives
        it without one, as a worker-style pipeline does.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await LLMService.process_frame(self, frame, direction)

        if isinstance(frame, InterruptionFrame):
            await self.stop_all_metrics()
            await self._cancel_pending_request()
            self._bot_responding = False
            if not self._user_speaking:
                self._audio_buffer = []
                self._pre_speech_buffer = []
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_responding = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_responding = False
        elif isinstance(frame, LLMContextFrame):
            self._context = frame.context
            await self._maybe_run_text_turn(frame.context)
            return
        elif isinstance(frame, LLMRunFrame):
            await self._maybe_run_text_turn(self._context, force=True)
            return
        elif isinstance(frame, InputAudioRawFrame):
            self._handle_audio_frame(frame)
        elif isinstance(frame, (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)):
            await self._handle_user_started()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._handle_user_stopped()

        await self.push_frame(frame, direction)

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
        """Run a one-shot, out-of-pipeline inference over any Omni modality.

        Extends ``run_inference()`` with what a worker-style agent needs from a
        one-shot call and the standard contract does not offer: the reasoning the
        model produced, deltas as they arrive, and a per-call reasoning budget.
        Request parameters are built with ``build_chat_completion_params()``, so
        sampling settings and media conversion match pipeline turns.

        Args:
            context: Context whose messages carry the media content parts.
            max_tokens: Overrides the configured token limit.
            reasoning_budget: Optional NVIDIA ``reasoning_budget`` extra body field.
            temperature: Overrides the configured temperature.
            stream: Whether to stream the completion and invoke the delta callbacks.
            on_text_delta: Called with each visible text delta while streaming.
            on_reasoning_delta: Called with each reasoning delta while streaming.

        Returns:
            The generated text, reasoning, and finish reason.
        """
        request_kwargs = self._out_of_band_request_kwargs(context)
        if max_tokens is not None:
            # One limit, one field: an endpoint given both is free to honour
            # either, so the one this call asks for replaces both.
            request_kwargs.pop("max_completion_tokens", None)
            request_kwargs["max_tokens"] = max_tokens
        if reasoning_budget is not None:
            extra_body = dict(request_kwargs.get("extra_body") or {})
            extra_body["reasoning_budget"] = reasoning_budget
            request_kwargs["extra_body"] = extra_body
        if temperature is not None:
            request_kwargs["temperature"] = temperature

        if not stream:
            request_kwargs["stream"] = False
            request_kwargs.pop("stream_options", None)
            completion = await self._client.chat.completions.create(**request_kwargs)
            choice = completion.choices[0] if completion.choices else None
            message = choice.message if choice else None
            return NvidiaOmniInferenceResult(
                text=_extract_text_content(getattr(message, "content", "")).strip(),
                reasoning=_extract_reasoning_content(message).strip(),
                finish_reason=str(getattr(choice, "finish_reason", "") or ""),
            )

        request_kwargs["stream"] = True
        request_kwargs.setdefault("stream_options", {"include_usage": True})
        text = ""
        reasoning = ""
        finish_reason = ""
        response_stream = await self._client.chat.completions.create(**request_kwargs)
        async for chunk in response_stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = str(choice.finish_reason)
            reasoning_delta = _extract_reasoning_content(choice.delta)
            text_delta = _extract_text_content(getattr(choice.delta, "content", ""))
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

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build chat completion params, appending the active audio turn.

        Delegates to the base implementation so inherited settings, tools, and
        tool choice are forwarded unchanged. While an audio turn is active, its
        content parts are appended as a transient trailing user message instead
        of mutating the shared context.

        A field nobody configured is dropped rather than sent as ``NOT_GIVEN``,
        which the client omits from the request anyway. That keeps one token
        limit on the wire: the inherited one-shot path overrides
        ``max_completion_tokens`` whenever that key is merely present, and would
        otherwise send it beside the ``max_tokens`` from settings.

        Args:
            params_from_context: Parameters derived from the LLM context.

        Returns:
            Dictionary of parameters for the chat completion request.
        """
        params = {
            name: value
            for name, value in super().build_chat_completion_params(params_from_context).items()
            if not isinstance(value, (NotGiven, _NotGiven))
        }
        if self._active_turn_parts:
            messages = list(params.get("messages") or [])
            messages.append(
                {"role": "user", "content": [_to_omni_content_part(part) for part in self._active_turn_parts]}
            )
            params["messages"] = messages
        return params

    async def _push_llm_text(self, text: str) -> None:
        """Split the transcript section out of visible content before TTS.

        The base loop routes every content delta through here. When transcripts
        are enabled, the ``<transcript>`` section becomes a
        ``TranscriptionFrame`` and only the ``<response>`` text travels onward,
        so neither TTS nor the assistant's context sees the tags.

        Args:
            text: Visible content from the model, reasoning already removed.
        """
        extractor = self._transcript_extractor
        if extractor is None:
            await super()._push_llm_text(text)
            return
        response_text = extractor.feed(text)
        await self._maybe_emit_transcript(extractor)
        if response_text:
            await super()._push_llm_text(response_text)

    async def _finalize_reasoning_state(self, *, flush_buffered_text: bool) -> None:
        """Close reasoning state, then flush what the transcript split still holds.

        Args:
            flush_buffered_text: Whether buffered text may still be forwarded.
                ``False`` when the stream ended early through interruption or
                cancellation.
        """
        await super()._finalize_reasoning_state(flush_buffered_text=flush_buffered_text)
        extractor = self._transcript_extractor
        if extractor is None:
            return
        # The split is over, so clear it before pushing: the remainder is plain
        # response text and must not be fed through the extractor again.
        self._transcript_extractor = None
        pending = extractor.finalize()
        await self._maybe_emit_transcript(extractor)
        if pending and flush_buffered_text:
            await self._push_llm_text(pending)

    async def _handle_user_started(self) -> None:
        """Open a fresh audio buffer for the utterance the user just started.

        Interrupting the bot is the turn controller's job, not this service's, so
        this only drops the turn Omni is generating for the previous utterance.
        """
        if self._user_speaking or not self._modality_enabled("audio"):
            return
        if self._bot_responding:
            logger.debug(f"{self}: barge-in detected, dropping the turn being generated")
            await self._cancel_pending_request()
            self._bot_responding = False
        self._user_speaking = True
        self._audio_buffer = list(self._pre_speech_buffer)
        self._pre_speech_buffer = []

    async def _handle_user_stopped(self) -> None:
        """Close the utterance at the end-of-turn boundary and answer it."""
        if not self._user_speaking:
            return
        self._user_speaking = False
        self._last_user_eou_at = time.time()
        await self._maybe_run_audio_turn()

    def _handle_audio_frame(self, frame: InputAudioRawFrame) -> None:
        """Buffer input audio, tracking the format the transport delivers."""
        self._sample_rate = frame.sample_rate
        self._channels = frame.num_channels
        if not self._modality_enabled("audio"):
            return
        if self._user_speaking:
            self._audio_buffer.append(frame.audio)
        else:
            self._append_pre_speech_audio(frame)

    def _append_pre_speech_audio(self, frame: InputAudioRawFrame) -> None:
        """Keep a rolling window of pre-speech audio so utterances start intact."""
        self._pre_speech_buffer.append(frame.audio)
        bytes_per_second = max(frame.sample_rate * frame.num_channels * 2, 1)
        max_bytes = int(bytes_per_second * float(self._settings.pre_speech_buffer_secs))
        total = sum(len(chunk) for chunk in self._pre_speech_buffer)
        while self._pre_speech_buffer and total > max_bytes:
            total -= len(self._pre_speech_buffer.pop(0))

    async def _maybe_run_audio_turn(self) -> None:
        """Answer the buffered utterance, letting Omni transcribe it itself."""
        if self._bot_responding:
            logger.debug(f"{self}: ignoring audio turn while bot is responding")
            return

        audio_payload = b"".join(self._audio_buffer)
        self._audio_buffer = []
        self._pre_speech_buffer = []
        if not audio_payload:
            return

        bytes_per_second = max(self._sample_rate * self._channels * 2, 1)
        min_secs = float(self._settings.min_user_audio_secs)
        if len(audio_payload) < int(bytes_per_second * min_secs):
            logger.debug(f"{self}: dropping utterance shorter than {min_secs * 1000:.0f} ms")
            return

        if self._pending_request is not None and not self._pending_request.done():
            logger.debug(f"{self}: preempting the in-flight turn with a newer one")
            await self.stop_all_metrics()
            await self._cancel_pending_request()

        eou_at = self._last_user_eou_at
        self._last_user_eou_at = None
        context = self._context
        if context is None:
            # A pipeline that never sends a context frame still has a user
            # waiting for an answer, so the utterance is answered on its own
            # rather than dropped, with no conversation history behind it.
            logger.warning(f"{self}: answering an utterance that arrived before any context")
            context = LLMContext([])
        instruction = self._audio_response_instruction()
        turn_parts = [
            audio_message_part(audio_payload, self._sample_rate, self._channels),
            text_message_part(instruction),
        ]
        expect_transcript = bool(self._settings.emit_transcriptions)
        # The audio and its instruction never enter the context, so this is the
        # only place they can be seen: the inherited context log won't show them.
        logger.debug(
            f"{self}: audio turn of {len(audio_payload) / bytes_per_second:.2f}s with a "
            f"{len(instruction)}-character instruction, transcript={expect_transcript}"
        )

        async def run() -> None:
            await self._run_turn(
                context,
                turn_parts=turn_parts,
                expect_transcript=expect_transcript,
                metrics_start_time=eou_at,
            )

        self._pending_request = self.create_task(run(), name="nvidia-omni-audio-turn")
        self._pending_request_is_audio = True

    async def _maybe_run_text_turn(self, context: LLMContext | None, *, force: bool = False) -> None:
        """Answer a context frame, unless the audio path owns the same turn.

        A tool result is answered whatever the input modality is: the completion
        that requested the call can only be finished by another completion, and
        an audio-only pipeline would otherwise leave the function's result
        unspoken.

        Args:
            context: The context to complete, or ``None`` when none arrived yet.
            force: Whether the frame asks for a completion outright, as
                ``LLMRunFrame`` does, rather than carrying a new user turn.
        """
        if context is None:
            logger.warning(f"{self}: asked for a completion before any context arrived, ignoring")
            return

        trigger = "user" if force else _completion_trigger(context)
        if trigger != "tool":
            if not self._modality_enabled("text"):
                return
            if self._modality_enabled("audio") and self._echoes_an_audio_turn(trigger, context, force=force):
                return

        pending = self._pending_request
        if pending is not None and not pending.done():
            if trigger == "tool":
                # The in-flight turn is the one that requested the tool call, so
                # let it finish instead of preempting its own follow-up.
                with contextlib.suppress(asyncio.CancelledError):
                    await pending
            elif self._pending_request_is_audio:
                logger.debug(f"{self}: audio turn in flight, ignoring the context echo")
                return
            else:
                logger.debug(f"{self}: preempting the in-flight turn with a newer one")
                await self.stop_all_metrics()
                await self._cancel_pending_request()

        async def run() -> None:
            await self._run_turn(context, turn_parts=self._unwritten_spoken_turn(trigger, context))

        self._pending_request = self.create_task(run(), name="nvidia-omni-text-turn")
        self._pending_request_is_audio = False

    def _echoes_an_audio_turn(
        self,
        trigger: Literal["user", "tool"] | None,
        context: LLMContext | None,
        *,
        force: bool,
    ) -> bool:
        """Whether a context frame repeats a user turn the audio path already owns.

        A pipeline that also accepts audio sees the same user turn twice: once as
        buffered speech, and again as the context frame the aggregator pushes
        once a transcript exists for it. Only asked when audio is enabled, so a
        text-only pipeline completes every context frame it is given, like any
        other OpenAI-compatible service.
        """
        if trigger is None:
            # Omni starts audio turns on speech boundaries rather than on context
            # frames, so a frame carrying no unanswered user turn is an echo.
            return True
        if force:
            return False
        if self._answered_transcript and _latest_user_text(context) == self._answered_transcript:
            logger.debug(f"{self}: the audio turn already answered this transcript, ignoring the echo")
            return True
        return False

    def _unwritten_spoken_turn(
        self,
        trigger: Literal["user", "tool"] | None,
        context: LLMContext | None,
    ) -> list[OpenAIContentPart] | None:
        """The spoken turn a tool follow-up still needs, as transient content parts.

        A spoken turn enters the context through the user aggregator, which
        writes it once the assistant response starts. The follow-up completion a
        tool result asks for belongs to that same response, so it can run before
        that write lands, and replaying the context alone would drop the request
        the user spoke. ``None`` when the context already carries the turn, or
        when nothing was transcribed for it.
        """
        transcript = self._answered_transcript
        if trigger != "tool" or not transcript or _latest_user_text(context) == transcript:
            return None
        return [text_message_part(transcript)]

    async def _run_turn(
        self,
        context: LLMContext,
        *,
        turn_parts: Sequence[OpenAIContentPart] | None = None,
        expect_transcript: bool = False,
        metrics_start_time: float | None = None,
    ) -> None:
        """Run one turn through the base OpenAI completion path.

        Mirrors the frame and metrics contract of the base ``LLMContextFrame``
        handler while letting Omni decide when a turn starts and inject audio
        content parts into it. Reasoning state is reset by the inherited
        ``_process_context()``.
        """
        self._active_turn_parts = list(turn_parts) if turn_parts else None
        self._transcript_extractor = _TranscriptResponseExtractor() if expect_transcript else None
        self._transcript_emitted = False

        await self.push_frame(LLMFullResponseStartFrame())
        await self.start_processing_metrics(start_time=metrics_start_time)
        try:
            await self._process_context(context)
        except httpx.TimeoutException as exc:
            await self._call_event_handler("on_completion_timeout")
            await self.push_error(error_msg="LLM completion timeout", exception=exc)
        except Exception as exc:
            await self.push_error(error_msg=f"Error during completion: {exc}", exception=exc)
        finally:
            self._active_turn_parts = None
            self._transcript_extractor = None
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())

    async def _maybe_emit_transcript(self, extractor: _TranscriptResponseExtractor) -> None:
        """Report the user's speech once the transcript section is complete."""
        if self._transcript_emitted:
            return
        if extractor.transcript_done and extractor.transcript:
            self._transcript_emitted = True
            await self._emit_user_transcript(extractor.transcript)

    async def _emit_user_transcript(self, transcript: str) -> None:
        """Report the user's spoken turn upstream, for the aggregator and the UI.

        The transcript is not written to the context here. The user aggregator
        upstream writes what this frame carries, exactly as it does for an STT
        service, and writing it here as well would leave the same spoken turn in
        the conversation twice.
        """
        self._answered_transcript = transcript
        await self.push_frame(
            TranscriptionFrame(
                text=transcript,
                user_id="user",
                timestamp=time_now_iso8601(),
                result=transcript,
            ),
            FrameDirection.UPSTREAM,
        )

    async def run_inference(
        self,
        context: LLMContext,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
    ) -> str | None:
        """Run a one-shot completion without attaching the active audio turn.

        Args:
            context: The LLM context containing conversation history.
            max_tokens: Optional override for the generated token limit.
            system_instruction: Optional system instruction for this inference.

        Returns:
            The model's response text, or ``None`` when nothing was generated.
        """
        with self._without_active_turn():
            return await super().run_inference(context, max_tokens, system_instruction)

    def current_turn_has_user_audio(self) -> bool:
        """Whether the request being generated carries the user's speech.

        True on an audio turn, where the model hears the user; false on a text
        turn, which sends the context alone.
        """
        return any(part.get("type") == "input_audio" for part in self._active_turn_parts or ())

    @contextlib.contextmanager
    def _without_active_turn(self):
        """Hide the in-flight audio turn from out-of-band requests."""
        active = self._active_turn_parts
        self._active_turn_parts = None
        try:
            yield
        finally:
            self._active_turn_parts = active

    def _out_of_band_request_kwargs(self, context: LLMContext) -> dict[str, Any]:
        """Build request kwargs for a one-shot call outside the pipeline turn."""
        invocation_params = self.get_llm_adapter().get_llm_invocation_params(
            context,
            system_instruction=assert_given(self._settings.system_instruction),
            convert_developer_to_user=not self.supports_developer_role,
        )
        with self._without_active_turn():
            return self.build_chat_completion_params(invocation_params)

    def _audio_response_instruction(self) -> str:
        """The instruction that accompanies buffered speech in an audio turn."""
        if self._settings.audio_response_instruction:
            return str(self._settings.audio_response_instruction)
        if self._settings.emit_transcriptions:
            return TRANSCRIPT_AUDIO_RESPONSE_INSTRUCTION
        return DEFAULT_AUDIO_RESPONSE_INSTRUCTION

    def _modality_enabled(self, modality: InputModality) -> bool:
        """Whether this pipeline input kind is configured."""
        return modality in tuple(self._settings.input_modalities)

    def _reset_audio_state(self) -> None:
        """Drop buffered speech and turn state at session start."""
        self._audio_buffer = []
        self._pre_speech_buffer = []
        self._user_speaking = False
        self._bot_responding = False
        self._pending_request_is_audio = False
        self._last_user_eou_at = None

    async def _cancel_pending_request(self) -> None:
        """Cancel the turn being generated, if any, and wait for it to unwind."""
        if self._pending_request and not self._pending_request.done():
            self._pending_request.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pending_request
        self._pending_request = None
        self._pending_request_is_audio = False

    @staticmethod
    def _validate_settings(settings: Settings) -> None:
        """Reject input kinds that cannot start a pipeline turn."""
        unknown = sorted(set(settings.input_modalities) - SUPPORTED_INPUT_MODALITIES)
        if unknown:
            raise ValueError(
                f"Unsupported pipeline input modalities: {unknown}. "
                f"Supported: {sorted(SUPPORTED_INPUT_MODALITIES)}. "
                "Other media travels in the context instead of starting a turn."
            )
        if not settings.input_modalities:
            raise ValueError("At least one pipeline input modality is required")


def text_message_part(text: str) -> OpenAIContentPart:
    """Create an OpenAI-compatible text content part."""
    return {"type": "text", "text": text}


def audio_message_part(audio: bytes, sample_rate: int, channels: int) -> OpenAIContentPart:
    """Create the universal audio content part from int16 PCM audio.

    Produces the same shape as ``LLMContext.create_audio_message()``;
    ``NvidiaOmniLLMAdapter`` renames it for the endpoint.
    """
    return {
        "type": "input_audio",
        "input_audio": {"data": audio_to_wav_base64(audio, sample_rate, channels), "format": "wav"},
    }


def image_message_part(data: bytes, mime_type: str = "image/jpeg") -> OpenAIContentPart:
    """Create an OpenAI-compatible image content part."""
    return {"type": "image_url", "image_url": {"url": data_to_data_url(data, mime_type)}}


def video_message_part(data: bytes, mime_type: str = "video/mp4") -> OpenAIContentPart:
    """Create an OpenAI-compatible video content part."""
    return {"type": "video_url", "video_url": {"url": data_to_data_url(data, mime_type)}}


def media_message_part(data: bytes, *, modality: MediaModality, mime_type: str) -> OpenAIContentPart:
    """Create a content part for text, audio, image, or video media."""
    if modality == "text":
        return text_message_part(data.decode("utf-8"))
    if modality == "audio":
        encoded = base64.b64encode(data).decode("ascii")
        return {"type": "input_audio", "input_audio": {"data": encoded, "format": _audio_format(mime_type)}}
    if modality == "image":
        return image_message_part(data, mime_type)
    if modality == "video":
        return video_message_part(data, mime_type)
    raise ValueError(f"Unsupported modality: {modality}")


def data_to_data_url(data: bytes, mime_type: str) -> str:
    """Encode bytes as a data URL."""
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def audio_to_wav_base64(audio: bytes, sample_rate: int, channels: int) -> str:
    """Encode little-endian int16 PCM bytes as base64 WAV."""
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio)
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def _audio_format(mime_type: str) -> str:
    """The audio format name a universal ``input_audio`` part carries."""
    return mime_type.split("/")[-1] or "wav"


def _to_omni_content_part(part: OpenAIContentPart) -> OpenAIContentPart:
    """Rewrite one content part in the shape Omni's endpoint reads.

    Only audio is renamed: Omni takes a data URL under ``audio_url`` where the
    universal context carries base64 data under ``input_audio``.
    """
    if part.get("type") != "input_audio":
        return part
    payload = part.get("input_audio") or {}
    audio_format = str(payload.get("format") or "wav")
    encoded = str(payload.get("data") or "")
    return {"type": "audio_url", "audio_url": {"url": f"data:audio/{audio_format};base64,{encoded}"}}


def _to_omni_message(message: ChatCompletionMessageParam) -> ChatCompletionMessageParam:
    """Rewrite a message's media parts, leaving plain text messages untouched.

    Returns a new message: the ones the base adapter hands back are the objects
    the context stores, and the provider's shape must not leak back into it.
    """
    content = message.get("content")
    if not isinstance(content, list):
        return message
    parts = [_to_omni_content_part(part) if isinstance(part, Mapping) else part for part in content]
    return cast(ChatCompletionMessageParam, {**message, "content": parts})


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


def _extract_reasoning_content(payload: Any) -> str:
    """Return provider-specific reasoning content from a delta or message."""
    for attr in ("reasoning", "reasoning_content"):
        value = getattr(payload, attr, None)
        if isinstance(value, str) and value:
            return value

    model_extra = getattr(payload, "model_extra", None)
    if isinstance(model_extra, dict):
        for key in ("reasoning", "reasoning_content"):
            value = model_extra.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _completion_trigger(context: LLMContext | None) -> Literal["user", "tool"] | None:
    """Why the context needs a completion, or ``None`` when it already has one.

    ``"tool"`` marks the follow-up completion a function-call result asks for,
    which must never preempt the turn that requested it.
    """
    if context is None:
        return None
    for message in reversed(list(context.get_messages())):
        if not isinstance(message, Mapping):
            continue
        role = message.get("role")
        if role == "assistant":
            return None
        if role == "tool":
            return "tool"
        if role == "user" and message.get("content"):
            return "user"
    return None


def _latest_user_text(context: LLMContext | None) -> str:
    """Return the newest user message rendered as plain text."""
    if context is None:
        return ""
    for message in reversed(list(context.get_messages())):
        if isinstance(message, Mapping) and message.get("role") == "user":
            return _extract_text_content(message.get("content")).strip()
    return ""


def _tag_match(text: str, tag: str) -> Literal["match", "partial", "no"]:
    """Return ``match``, ``partial``, or ``no`` for a tag at the start of ``text``."""
    if text.startswith(tag):
        return "match"
    if len(text) < len(tag) and tag.startswith(text):
        return "partial"
    return "no"


class _TranscriptResponseExtractor:
    """Incrementally splits ``<transcript>`` and ``<response>`` sections.

    ``feed()`` returns the response text to stream onward; the transcript is
    accumulated and exposed through ``transcript`` once ``transcript_done`` is
    set. Output that does not open with a ``<transcript>`` tag is treated
    entirely as response text, so an unexpected plain reply still reaches TTS.
    """

    def __init__(self) -> None:
        self._state = "detecting"
        self._buffer = ""
        self._transcript = ""
        self.transcript = ""
        self.transcript_done = False

    def feed(self, text: str) -> str:
        """Consume a content delta and return response text ready to stream."""
        if self._state == "done":
            return ""
        self._buffer += text
        out: list[str] = []
        while self._buffer:
            if self._state == "detecting":
                stripped = self._buffer.lstrip()
                if not stripped:
                    return "".join(out)
                match = _tag_match(stripped, _TRANSCRIPT_OPEN)
                if match == "partial":
                    return "".join(out)
                if match == "match":
                    self._buffer = stripped[len(_TRANSCRIPT_OPEN) :]
                    self._state = "transcript"
                else:
                    self._buffer = stripped
                    self._state = "passthrough"
                continue

            if self._state == "transcript":
                idx = self._buffer.find(_TRANSCRIPT_CLOSE)
                if idx == -1:
                    safe = len(self._buffer) - (len(_TRANSCRIPT_CLOSE) - 1)
                    if safe > 0:
                        self._transcript += self._buffer[:safe]
                        self._buffer = self._buffer[safe:]
                    return "".join(out)
                self._transcript += self._buffer[:idx]
                self.transcript = self._transcript.strip()
                self.transcript_done = True
                self._buffer = self._buffer[idx + len(_TRANSCRIPT_CLOSE) :]
                self._state = "between"
                continue

            if self._state == "between":
                stripped = self._buffer.lstrip()
                if not stripped:
                    self._buffer = ""
                    return "".join(out)
                match = _tag_match(stripped, _RESPONSE_OPEN)
                if match == "partial":
                    self._buffer = stripped
                    return "".join(out)
                if match == "match":
                    self._buffer = stripped[len(_RESPONSE_OPEN) :]
                    self._state = "response"
                else:
                    self._buffer = stripped
                    self._state = "passthrough"
                continue

            if self._state == "passthrough":
                out.append(self._buffer)
                self._buffer = ""
                return "".join(out)

            idx = self._buffer.find(_RESPONSE_CLOSE)
            if idx != -1:
                out.append(self._buffer[:idx])
                self._buffer = ""
                self._state = "done"
                return "".join(out)
            safe = len(self._buffer) - (len(_RESPONSE_CLOSE) - 1)
            if safe > 0:
                out.append(self._buffer[:safe])
                self._buffer = self._buffer[safe:]
            return "".join(out)

        return "".join(out)

    def finalize(self) -> str:
        """Flush buffered response text once the stream has ended."""
        remainder = self._buffer
        self._buffer = ""
        if self._state == "transcript":
            if self._transcript and not self.transcript_done:
                self.transcript = self._transcript.strip()
                self.transcript_done = bool(self.transcript)
            return ""
        if self._state == "done":
            return ""
        return remainder
