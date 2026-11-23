# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cascade's speech services, and the TTS modes that keep each line's text behind its audio."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    DataFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    TextType,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.nvidia.stt import NvidiaSTTService, NvidiaSTTSettings
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings
from pipecat.utils.text.transforms import expand_currency

from nemotron_voice_agent.core.catalog import NVCF_GRPC_SERVER
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.media.silent_lines import (
    STITCHED_FAILURE_KEY,
    SentenceSynthesisFailed,
    SilentSentences,
)
from nemotron_voice_agent.pipecat.runtime.utils import is_nvcf, load_ipa_dictionary, normalize_lang_code, nvidia_api_key
from nemotron_voice_agent.pipecat.shared.nemotron_speech_text_filter import nemotron_speech_markdown_text_transform


def speech_server(entry: Mapping[str, Any]) -> str:
    """Return the gRPC address of a speech service's catalog entry: its ``server``, else NVIDIA's cloud."""
    return str(entry.get("server") or NVCF_GRPC_SERVER)


def build_stt(body: dict, default_asr: dict) -> NvidiaSTTService:
    """Build the streaming speech-to-text service from the catalog entry and body overrides."""
    server = body.get("asr_server", "") or speech_server(default_asr)
    kwargs: dict = {"api_key": nvidia_api_key(), "server": server, "use_ssl": is_nvcf(server)}
    function_id = body.get("asr_function_id", "") or default_asr.get("function_id", "")
    model = body.get("asr_model", "") or default_asr.get("model", "")
    language = body.get("asr_language_code", "") or default_asr.get("language_code", "")
    punctuation = default_asr.get("automatic_punctuation", True)
    punctuation = str(body.get("asr_automatic_punctuation", punctuation)).lower() != "false"
    if function_id or model:
        kwargs["model_function_map"] = {"function_id": function_id, "model_name": model or "custom-asr"}
    kwargs["settings"] = NvidiaSTTSettings(automatic_punctuation=punctuation)
    if language:
        kwargs["settings"].language = language
    logger.info(f"ASR: server={server}, function_id={function_id or '(default)'}, language={language or '(default)'}")
    return NvidiaSTTService(**kwargs, stop_history=320)


@dataclass
class StitchedSynthesisResult(DataFrame):
    """The terminal state of a stitched stream, ordered with its audio."""

    context_id: str
    complete: bool


class PlaybackTTSService(NvidiaTTSService):
    """Keep synthesis evidence in audio order for the frontend transcript."""

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Forward errors and mark the affected synthesis before Pipecat queues its text."""
        per_sentence = self.settings.synthesis_mode == PerSentence.synthesis_mode
        async for frame in super().run_tts(text, context_id):
            if isinstance(frame, ErrorFrame):
                yield (
                    SentenceSynthesisFailed(context_id) if per_sentence else StitchedSynthesisResult(context_id, False)
                )
            yield frame

    def _responding_context(self) -> str | None:
        stream = self._stream_state
        if stream is not None and stream.response_task is asyncio.current_task():
            return stream.context_id
        return None

    async def push_error_frame(self, error: ErrorFrame, force_treat_as_permanent: bool = False) -> None:
        """Mark an error from the active stitched response task, without changing error reporting."""
        if context_id := self._responding_context():
            marker = StitchedSynthesisResult(context_id, False)
            if self.audio_context_available(context_id):
                await self.append_to_audio_context(context_id, marker)
            else:
                await self.push_frame(marker)
        await super().push_error_frame(error, force_treat_as_permanent=force_treat_as_permanent)

    async def remove_audio_context(self, context_id: str | None) -> None:
        """Mark the stitched response task's normal drain before it closes its queue."""
        if context_id is not None and context_id == self._responding_context():
            await self.append_to_audio_context(context_id, StitchedSynthesisResult(context_id, True))
        await super().remove_audio_context(context_id)


def build_tts(body: dict, default_tts: dict, *, expand_currency_amounts: bool = False) -> NvidiaTTSService:
    """Build the text-to-speech from the catalog entry and body overrides, stripping Markdown from what it speaks."""
    server = body.get("tts_server", "") or speech_server(default_tts)
    voice = body.get("tts_voice_id", "") or default_tts.get("voice_id", "")
    mode = body.get("tts_synthesis_mode", "") or default_tts.get("synthesis_mode", "")
    raw_function_id = body.get("tts_function_id")
    function_id = str(raw_function_id) if raw_function_id is not None else default_tts.get("function_id", "")
    model = body.get("tts_model", "") or default_tts.get("model", "")
    zero_shot = body.get("tts_zero_shot_audio_prompt_file", "") or default_tts.get("zero_shot_audio_prompt_file", "")
    language = body.get("tts_language_code", "") or default_tts.get("language_code", "")
    settings_kwargs: dict = {"voice": voice}
    if mode:
        settings_kwargs["synthesis_mode"] = mode
    if language:
        settings_kwargs["language"] = normalize_lang_code(language)
    kwargs: dict = {
        "api_key": nvidia_api_key(),
        "server": server,
        "settings": NvidiaTTSSettings(**settings_kwargs),
        "use_ssl": is_nvcf(server),
        "text_transforms": [("*", nemotron_speech_markdown_text_transform)],
        "custom_dictionary": load_ipa_dictionary(),
    }
    if function_id or model:
        kwargs["model_function_map"] = {"function_id": function_id, "model_name": model}
    if zero_shot:
        kwargs["zero_shot_audio_prompt_file"] = zero_shot
    logger.info(
        f"TTS: server={server}, voice={voice}, model={model or '(default)'}, synthesis_mode={mode or '(default)'}, "
        f"language={language or '(default)'}"
    )
    tts = PlaybackTTSService(**kwargs)
    if expand_currency_amounts:

        async def english_currency(text: str, text_type: str | TextType) -> str:
            if str(tts.settings.language).split("-", 1)[0].lower() == "en":
                return await expand_currency(text, text_type)
            return text

        tts.add_text_transformer(english_currency)
    return tts


class SpeechMode(Protocol):
    """How the text-to-speech takes a line, and what keeps the line's text behind its audio."""

    synthesis_mode: ClassVar[str]

    def text_hold(self) -> list[FrameProcessor]:
        """Return new processors for right after the text-to-speech, which keep its text behind its audio or mark it."""


class PerSentence:
    """One synthesis request per sentence: Pipecat yields a sentence's audio, then its text, so nothing is held."""

    synthesis_mode: ClassVar[str] = "per_sentence"

    def text_hold(self) -> list[FrameProcessor]:
        """Return a processor that marks failed sentences and sentences with no audio."""
        return [SilentSentences()]


@dataclass(frozen=True)
class Stitched:
    """One stitched stream per line: smoother across sentences, with each sentence's text queued before its audio."""

    synthesis_mode: ClassVar[str] = "stitched"
    require_completion: bool = False

    def text_hold(self) -> list[FrameProcessor]:
        """Return a :class:`LineTextHold`, so a line's text follows all of its audio."""
        return [LineTextHold(require_completion=self.require_completion)]


SPEECH_MODES: dict[str, type[SpeechMode]] = {"per_sentence": PerSentence, "stitched": Stitched}


def speech_mode(tts: Any) -> SpeechMode:
    """Return the TTS mode a text-to-speech service runs in, by its synthesis mode setting."""
    mode = named_mode(getattr(getattr(tts, "settings", None), "synthesis_mode", None))
    if isinstance(mode, Stitched) and isinstance(tts, PlaybackTTSService):
        return Stitched(require_completion=True)
    return mode


def named_mode(synthesis_mode: Any) -> SpeechMode:
    """Return the TTS mode a synthesis mode names; :class:`PerSentence`, the NVIDIA service's default, for any other."""
    return SPEECH_MODES.get(str(synthesis_mode), PerSentence)()


class LineTextHold(FrameProcessor):
    """Release stitched text behind the audio, using completion when the provider reports it."""

    def __init__(self, *, require_completion: bool = False, **kwargs):
        """Start with nothing held."""
        super().__init__(**kwargs)
        self._require_completion = require_completion
        self._held: dict[str | None, list[TTSTextFrame]] = {}
        self._failed: set[str] = set()

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        """Hold each stream's text while its audio is still unconfirmed."""
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM:
            if isinstance(frame, TTSTextFrame):
                if frame.context_id not in self._failed:
                    self._held.setdefault(frame.context_id, []).append(frame)
                return
            if isinstance(frame, StitchedSynthesisResult):
                held = self._held.pop(frame.context_id, ())
                if not frame.complete:
                    self._failed.add(frame.context_id)
                elif frame.context_id not in self._failed:
                    for text in held:
                        await self.push_frame(text, direction)
                return
            if not self._require_completion and isinstance(frame, (TTSStoppedFrame, LLMFullResponseEndFrame, EndFrame)):
                for held in self._held.values():
                    for text in held:
                        await self.push_frame(text, direction)
                self._held.clear()
            if isinstance(frame, LLMFullResponseEndFrame) and self._failed:
                frame.metadata[STITCHED_FAILURE_KEY] = True
            if isinstance(frame, (InterruptionFrame, CancelFrame, LLMFullResponseEndFrame, EndFrame)):
                self._held.clear()
                self._failed.clear()
        await self.push_frame(frame, direction)
