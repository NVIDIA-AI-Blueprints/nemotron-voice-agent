# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cascade's speech services: streaming speech-to-text in, text-to-speech out."""

from __future__ import annotations

from loguru import logger
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
