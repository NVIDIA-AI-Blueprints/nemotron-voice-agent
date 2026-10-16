# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared Pipecat speech services this adapter uses, built from FBV's resolved endpoints.

Ported from FDH ``adapters/pipecat/services.py`` (``a0ef50c``). The ASR and TTS endpoints come
from the profile through FBV's loader (``core/config/catalog.py``), the same values v1 used
(plan §3.2). Nothing here implements speech: Silero is Pipecat's, the STT service is the
repository's shared ``NvidiaForceEouSTTService`` and the TTS service is Pipecat's
``NvidiaTTSService`` with the shared Nemotron text filter.
"""

from __future__ import annotations

from typing import Any

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.processors.frame_processor import FrameProcessor

from examples.frontend_backend_verdict.adapters.pipecat.settings import PipecatSettings
from examples.frontend_backend_verdict.core.config.catalog import SpeechEndpoint
from examples.frontend_backend_verdict.core.config.voice import VoiceConfig


def _model_function_map(endpoint: SpeechEndpoint, default_model: str) -> dict[str, str] | None:
    if endpoint.function_id or endpoint.model:
        return {"function_id": endpoint.function_id, "model_name": endpoint.model or default_model}
    return None


def build_vad(settings: PipecatSettings) -> VADAnalyzer:
    """Silero with Pipecat's ``VADParams`` defaults, overridden only by the settings file (B1)."""
    return SileroVADAnalyzer(params=VADParams(**settings.vad.overrides()))


def build_stt(config: VoiceConfig, settings: PipecatSettings) -> FrameProcessor:
    """The shared ``NvidiaForceEouSTTService`` on the profile's ASR endpoint (B2)."""
    from pipecat.services.nvidia.stt import NvidiaSTTSettings  # noqa: PLC0415

    from examples.shared.nvidia_force_eou_stt import NvidiaForceEouSTTService  # noqa: PLC0415

    endpoint = config.asr.endpoint
    kwargs: dict[str, Any] = {
        "api_key": endpoint.api_key or None,
        "server": endpoint.server,
        "use_ssl": endpoint.use_ssl,
        "stop_history": 400,
    }
    if settings.turn.stt_ttfs_p99 is not None:
        kwargs["ttfs_p99_latency"] = settings.turn.stt_ttfs_p99
    function_map = _model_function_map(endpoint, "custom-asr")
    if function_map:
        kwargs["model_function_map"] = function_map
    if endpoint.language_code:
        kwargs["settings"] = NvidiaSTTSettings(language=endpoint.language_code)
    return NvidiaForceEouSTTService(**kwargs)


def build_tts(config: VoiceConfig, settings: PipecatSettings) -> FrameProcessor:
    """Pipecat's ``NvidiaTTSService`` on the profile's TTS endpoint, one request per sentence (B3)."""
    from pipecat.services.nvidia.tts import (  # noqa: PLC0415
        NvidiaTTSService,
        NvidiaTTSSettings,
        NvidiaTTSSynthesisMode,
    )

    from examples.shared.nemotron_speech_text_filter import NemotronSpeechTextFilter  # noqa: PLC0415

    endpoint = config.tts.endpoint
    tts_settings: dict[str, Any] = {"synthesis_mode": NvidiaTTSSynthesisMode(settings.tts.synthesis_mode)}
    if endpoint.voice_id:
        tts_settings["voice"] = endpoint.voice_id
    if endpoint.language_code:
        tts_settings["language"] = endpoint.language_code
    kwargs: dict[str, Any] = {
        "api_key": endpoint.api_key or None,
        "server": endpoint.server,
        "use_ssl": endpoint.use_ssl,
        "sample_rate": config.tts.sample_rate,
        "settings": NvidiaTTSSettings(**tts_settings),
    }
    if config.tts.normalize_text:
        kwargs["text_filters"] = [NemotronSpeechTextFilter()]
    function_map = _model_function_map(endpoint, "")
    if function_map:
        kwargs["model_function_map"] = function_map
    return NvidiaTTSService(**kwargs)
