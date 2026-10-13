# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cascade's speech services: streaming speech-to-text in, text-to-speech out."""

from __future__ import annotations

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    SystemFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.nvidia.stt import NvidiaSTTSettings
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings

from examples.shared.nemotron_speech_text_filter import NemotronSpeechTextFilter
from examples.shared.nvidia_force_eou_stt import NvidiaForceEouSTTService
from utils import is_nvcf, load_ipa_dictionary, nvidia_api_key


def build_stt(body: dict, default_asr: dict) -> NvidiaForceEouSTTService:
    """Build the streaming speech-to-text service from the catalog entry and body overrides."""
    server = body.get("asr_server", "") or default_asr.get("server", "grpc.nvcf.nvidia.com:443")
    kwargs: dict = {"api_key": nvidia_api_key(), "server": server, "use_ssl": is_nvcf(server)}
    function_id = body.get("asr_function_id", "") or default_asr.get("function_id", "")
    model = body.get("asr_model", "") or default_asr.get("model", "")
    language = body.get("asr_language_code", "") or default_asr.get("language_code", "")
    punctuation = str(body.get("asr_automatic_punctuation", "true")).lower() != "false"
    if function_id or model:
        kwargs["model_function_map"] = {"function_id": function_id, "model_name": model or "custom-asr"}
    kwargs["settings"] = NvidiaSTTSettings(automatic_punctuation=punctuation)
    if language:
        kwargs["settings"].language = language
    logger.info(f"ASR: server={server}, function_id={function_id or '(default)'}, language={language or '(default)'}")
    return NvidiaForceEouSTTService(**kwargs, stop_history=400)


def build_tts(body: dict, default_tts: dict) -> NvidiaTTSService:
    """Build the text-to-speech service from the catalog entry and body overrides."""
    server = body.get("tts_server", "") or default_tts.get("server", "grpc.nvcf.nvidia.com:443")
    voice = body.get("tts_voice_id", "") or default_tts.get("voice_id", "")
    mode = body.get("tts_synthesis_mode", "") or default_tts.get("synthesis_mode", "")
    raw_function_id = body.get("tts_function_id")
    function_id = str(raw_function_id) if raw_function_id is not None else default_tts.get("function_id", "")
    model = body.get("tts_model", "") or default_tts.get("model", "")
    zero_shot = body.get("tts_zero_shot_audio_prompt_file", "") or default_tts.get("zero_shot_audio_prompt_file", "")
    settings_kwargs: dict = {"voice": voice}
    if mode:
        settings_kwargs["synthesis_mode"] = mode
    kwargs: dict = {
        "api_key": nvidia_api_key(),
        "server": server,
        "settings": NvidiaTTSSettings(**settings_kwargs),
        "use_ssl": is_nvcf(server),
        "text_filters": [NemotronSpeechTextFilter()],
        "custom_dictionary": load_ipa_dictionary(),
    }
    if function_id or model:
        kwargs["model_function_map"] = {"function_id": function_id, "model_name": model}
    if zero_shot:
        kwargs["zero_shot_audio_prompt_file"] = zero_shot
    logger.info(
        f"TTS: server={server}, voice={voice}, model={model or '(default)'}, synthesis_mode={mode or '(default)'}"
    )
    return NvidiaTTSService(**kwargs)


class SpeechPreroll(FrameProcessor):
    """Hold the start of each reply until ``preroll_ms`` of audio is ready, then play it on.

    Synthesis runs several times faster than playback but delivers audio in uneven chunks; without a head start,
    playback runs dry between chunks and the caller hears a stutter. Every downstream frame of the reply is held in
    order (so word timings stay with their audio) until enough audio is buffered or the reply ends. An interruption
    drops what is held. System frames pass at once.
    """

    def __init__(self, preroll_ms: int, **kwargs):
        """Hold ``preroll_ms`` of audio at the start of each reply (0 passes everything through)."""
        super().__init__(**kwargs)
        self.preroll_ms = preroll_ms
        self._held: list = []
        self._held_ms = 0.0
        self._holding = False

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        """Hold a reply's first frames, release them once enough audio is ready, and pass everything else."""
        await super().process_frame(frame, direction)
        if direction is not FrameDirection.DOWNSTREAM or not self.preroll_ms:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, (InterruptionFrame, CancelFrame)):
            self._drop()
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, TTSStartedFrame) and not self._holding:
            self._holding = True
        if not self._holding:
            await self.push_frame(frame, direction)
            return
        self._held.append(frame)
        if isinstance(frame, OutputAudioRawFrame):
            self._held_ms += len(frame.audio) / 2 / frame.num_channels / frame.sample_rate * 1000
        if self._held_ms >= self.preroll_ms or isinstance(frame, (TTSStoppedFrame, EndFrame)):
            await self._release()

    async def _release(self) -> None:
        held, self._held, self._held_ms, self._holding = self._held, [], 0.0, False
        for frame in held:
            await self.push_frame(frame, FrameDirection.DOWNSTREAM)

    def _drop(self) -> None:
        self._held, self._held_ms, self._holding = [], 0.0, False
