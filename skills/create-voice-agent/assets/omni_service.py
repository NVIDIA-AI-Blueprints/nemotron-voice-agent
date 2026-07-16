# ruff: noqa: D101,D102,D103,D107

"""Reference omni_service.py — COPY to project root. Nemotron Omni multimodal LLM for Pipecat."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import re
import wave
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from openai import AsyncOpenAI
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
    LLMTextFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import NOT_GIVEN, LLMSettings, _NotGiven

VOICE_JSON_INSTRUCTION = (
    "Respond with strict JSON only: "
    '{"transcript":"<what the user said>","response":"<spoken reply>"}. '
    "Keep response brief, conversational, plain text suitable for TTS."
)

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a helpful voice assistant. Respond in plain spoken language only. "
    "Keep answers brief and direct. Do not use markdown, lists, code blocks, "
    "or emojis. Your replies will be read aloud."
)


def _is_unusable_spoken(text: str) -> bool:
    """Drop short or numeric garbage that should not be sent to TTS."""
    spoken = text.strip()
    if len(spoken) < 8:
        return True
    if spoken.isdigit():
        return True
    alnum = re.sub(r"\s+", "", spoken)
    return bool(alnum.isdigit())


@dataclass
class NvidiaOmniMultimodalSettings(LLMSettings):
    """Runtime settings for Nemotron Omni audio-in LLM."""

    input_modalities: list[str] | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    emit_transcriptions: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    min_user_audio_secs: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    pre_speech_buffer_secs: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    stream: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    response_format: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class NvidiaOmniMultimodalService(LLMService):
    """Audio-in Nemotron Omni service (replaces separate STT + text LLM)."""

    Settings = NvidiaOmniMultimodalSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        settings: Settings | None = None,
        extra_body: dict[str, Any] | None = None,
        **kwargs,
    ):
        default_settings = self.Settings(
            model=model,
            system_instruction=DEFAULT_SYSTEM_INSTRUCTION,
            max_tokens=8192,
            temperature=0.6,
            top_p=0.95,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
            input_modalities=["text", "audio"],
            emit_transcriptions=True,
            min_user_audio_secs=0.3,
            pre_speech_buffer_secs=0.2,
            stream=True,
            response_format="json_object",
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(settings=default_settings, **kwargs)
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=120.0,
        )
        self._extra_body = extra_body or {}
        self._audio_buffer: list[bytes] = []
        self._pre_speech_buffer: list[bytes] = []
        self._sample_rate = 16000
        self._channels = 1
        self._context: LLMContext | None = None
        self._pending_request: asyncio.Task | None = None
        self._user_speaking = False
        self._bot_responding = False
        self._request_lock = asyncio.Lock()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._sample_rate = frame.audio_in_sample_rate or self._sample_rate
        elif isinstance(frame, InputAudioRawFrame):
            if self._user_speaking:
                self._audio_buffer.append(frame.audio)
            else:
                self._pre_speech_buffer.append(frame.audio)
                self._trim_pre_speech_buffer()
        elif isinstance(frame, (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)):
            self._user_speaking = True
            if self._pre_speech_buffer:
                self._audio_buffer = list(self._pre_speech_buffer) + self._audio_buffer
                self._pre_speech_buffer.clear()
            if self._bot_responding:
                await self._cancel_request()
                await self.push_frame(InterruptionFrame())
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            await self._maybe_run_audio_turn()
        elif isinstance(frame, LLMContextFrame):
            self._context = frame.context
            await self._run_text_turn()
            return
        elif isinstance(frame, LLMRunFrame):
            await self._run_text_turn()
            return
        elif isinstance(frame, InterruptionFrame):
            await self._cancel_request()
            if not self._user_speaking:
                self._audio_buffer.clear()
                self._pre_speech_buffer.clear()
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_responding = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_responding = False
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._cancel_request()

        await self.push_frame(frame, direction)

    def _trim_pre_speech_buffer(self):
        pre_secs = float(self._settings.pre_speech_buffer_secs or 0.2)
        if pre_secs <= 0 or not self._pre_speech_buffer:
            return
        max_bytes = int(pre_secs * self._sample_rate * self._channels * 2)
        joined = b"".join(self._pre_speech_buffer)
        if len(joined) > max_bytes:
            joined = joined[-max_bytes:]
            self._pre_speech_buffer = [joined]

    async def _cancel_request(self):
        if self._pending_request and not self._pending_request.done():
            self._pending_request.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pending_request
        self._pending_request = None

    def _pcm_duration_secs(self, chunks: list[bytes]) -> float:
        if not chunks:
            return 0.0
        nbytes = sum(len(c) for c in chunks)
        return nbytes / (self._sample_rate * self._channels * 2)

    def _pcm_to_wav_b64(self, chunks: list[bytes]) -> str:
        pcm = b"".join(chunks)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self._channels)
            wf.setsampwidth(2)
            wf.setframerate(self._sample_rate)
            wf.writeframes(pcm)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _context_messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        system = self._settings.system_instruction
        if isinstance(system, str) and system:
            messages.append({"role": "system", "content": system})

        if self._context is not None:
            for msg in self._context.get_messages():
                role = msg.get("role")
                content = msg.get("content")
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content})
        return messages

    def _build_messages(self, *, audio_b64: str | None, text_prompt: str) -> list[dict[str, Any]]:
        messages = self._context_messages()

        if audio_b64:
            content: list[dict[str, Any]] = [
                {"type": "text", "text": text_prompt},
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_b64, "format": "wav"},
                },
            ]
            messages.append({"role": "user", "content": content})

        return messages

    async def _maybe_run_audio_turn(self):
        if self._bot_responding or self._pending_request:
            return
        chunks = self._audio_buffer
        if not chunks:
            return
        if self._pcm_duration_secs(chunks) < float(self._settings.min_user_audio_secs or 0.3):
            self._audio_buffer.clear()
            return

        audio_b64 = self._pcm_to_wav_b64(chunks)
        self._audio_buffer.clear()
        instruction = (
            VOICE_JSON_INSTRUCTION
            if self._settings.emit_transcriptions
            else ("Listen to the audio and respond to the user.")
        )
        await self._start_request(
            messages=self._build_messages(audio_b64=audio_b64, text_prompt=instruction),
            emit_transcript=bool(self._settings.emit_transcriptions),
        )

    async def _run_text_turn(self):
        if self._pending_request:
            return
        messages = self._context_messages()
        if not any(m.get("role") == "user" for m in messages):
            return
        await self._start_request(messages=messages, emit_transcript=False)

    async def _start_request(
        self,
        *,
        messages: list[dict[str, Any]],
        emit_transcript: bool,
    ):
        async with self._request_lock:
            if self._pending_request and not self._pending_request.done():
                return
            self._pending_request = asyncio.create_task(
                self._stream_completion(messages=messages, emit_transcript=emit_transcript)
            )
            try:
                await self._pending_request
            finally:
                self._pending_request = None

    async def _stream_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        emit_transcript: bool,
    ):
        await self.push_frame(LLMFullResponseStartFrame())
        response_text = ""
        try:
            kwargs: dict[str, Any] = {
                "model": self._settings.model,
                "messages": messages,
                "stream": bool(self._settings.stream),
                "temperature": self._settings.temperature,
                "top_p": self._settings.top_p,
                "max_tokens": self._settings.max_tokens,
            }
            if self._extra_body:
                kwargs["extra_body"] = self._extra_body
            if emit_transcript and self._settings.response_format == "json_object":
                kwargs["response_format"] = {"type": "json_object"}

            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    response_text += delta

            transcript, spoken = self._parse_response(response_text, emit_transcript)
            logger.info(
                "Omni response mode={} spoken={!r} raw_len={}",
                "audio" if emit_transcript else "text",
                spoken,
                len(response_text),
            )
            if emit_transcript and transcript:
                await self.push_frame(TranscriptionFrame(text=transcript, user_id="user"))
            if spoken:
                await self.push_frame(LLMTextFrame(text=spoken))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Omni completion failed: {}", exc)
        finally:
            await self.push_frame(LLMFullResponseEndFrame())

    def _parse_response(self, text: str, emit_transcript: bool) -> tuple[str, str]:
        if not text:
            return "", ""
        if not emit_transcript:
            spoken = text.strip()
            if _is_unusable_spoken(spoken):
                logger.warning("Omni text response too short or numeric, dropping: {!r}", spoken)
                return "", ""
            return "", spoken

        try:
            payload = json.loads(text)
            transcript = str(payload.get("transcript", "")).strip()
            response = str(payload.get("response", "")).strip()
            if response and not _is_unusable_spoken(response):
                return transcript, response
            if response:
                logger.warning("Omni JSON response unusable for TTS, dropping: {!r}", response[:120])
        except json.JSONDecodeError:
            pass

        spoken = text.strip()
        if _is_unusable_spoken(spoken):
            logger.warning("Omni audio response unusable, dropping: {!r}", spoken[:120])
            return "", ""
        return "", spoken


NvidiaOmniService = NvidiaOmniMultimodalService
