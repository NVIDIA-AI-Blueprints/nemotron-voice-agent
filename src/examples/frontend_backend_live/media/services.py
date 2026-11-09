# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cascade's speech services, and the TTS modes that keep each line's text behind its audio."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Protocol

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.nvidia.stt import NvidiaSTTService, NvidiaSTTSettings
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings

from examples.frontend_backend_live.media.silent_lines import SilentSentences
from examples.shared.nemotron_speech_text_filter import nemotron_speech_markdown_text_transform
from utils import NVCF_GRPC_SERVER, is_nvcf, load_ipa_dictionary, normalize_lang_code, nvidia_api_key


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


def build_tts(body: dict, default_tts: dict) -> NvidiaTTSService:
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
    return NvidiaTTSService(**kwargs)


class SpeechMode(Protocol):
    """How the text-to-speech takes a line, and what keeps the line's text behind its audio."""

    synthesis_mode: ClassVar[str]
    reports_sentences: ClassVar[bool]

    def text_hold(self) -> list[FrameProcessor]:
        """Return new processors for right after the text-to-speech, which keep its text behind its audio or mark it."""


class PerSentence:
    """One synthesis request per sentence: Pipecat yields a sentence's audio, then its text, so nothing is held."""

    synthesis_mode: ClassVar[str] = "per_sentence"
    reports_sentences: ClassVar[bool] = True

    def text_hold(self) -> list[FrameProcessor]:
        """Return a :class:`SilentSentences`, which marks the text of each sentence that got no audio."""
        return [SilentSentences()]


class Stitched:
    """One stitched stream per line: smoother across sentences, with each sentence's text queued before its audio."""

    synthesis_mode: ClassVar[str] = "stitched"
    reports_sentences: ClassVar[bool] = False

    def text_hold(self) -> list[FrameProcessor]:
        """Return a :class:`LineTextHold`, so a line's text follows all of its audio."""
        return [LineTextHold()]


SPEECH_MODES: dict[str, type[SpeechMode]] = {"per_sentence": PerSentence, "stitched": Stitched}


def speech_mode(tts: Any) -> SpeechMode:
    """Return the TTS mode a text-to-speech service runs in, by its synthesis mode setting."""
    return named_mode(getattr(getattr(tts, "settings", None), "synthesis_mode", None))


def named_mode(synthesis_mode: Any) -> SpeechMode:
    """Return the TTS mode a synthesis mode names; :class:`PerSentence`, the NVIDIA service's default, for any other."""
    return SPEECH_MODES.get(str(synthesis_mode), PerSentence)()


class LineTextHold(FrameProcessor):
    """Holds each line's text until the line's audio has all been produced, then lets it follow that audio."""

    def __init__(self, **kwargs):
        """Start with nothing held."""
        super().__init__(**kwargs)
        self._held: list[TTSTextFrame] = []

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        """Hold a line's text frames; release them just before the line ends; drop them on a cut."""
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM:
            if isinstance(frame, TTSTextFrame):
                self._held.append(frame)
                return
            if isinstance(frame, (InterruptionFrame, CancelFrame)):
                self._held.clear()
            elif isinstance(frame, (TTSStoppedFrame, LLMFullResponseEndFrame, EndFrame)):
                # A context Pipecat reopens for late audio has no TTSStoppedFrame: the response's end releases it.
                held, self._held = self._held, []
                for text in held:
                    await self.push_frame(text, direction)
        await self.push_frame(frame, direction)
