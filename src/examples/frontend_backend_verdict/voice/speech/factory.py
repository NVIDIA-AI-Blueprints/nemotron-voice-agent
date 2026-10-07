# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the shared speech services from configuration.

Riva and Silero are imported lazily, so ``--stub-speech`` runs (tau3 gates,
smoke tests) need neither a speech endpoint nor the ONNX model.
"""

from __future__ import annotations

from examples.frontend_backend_verdict.voice.config import VoiceConfig
from examples.frontend_backend_verdict.voice.speech.ports import (
    IdentityNormalizer,
    SpeechServices,
    TextNormalizer,
    VadFactory,
)
from examples.frontend_backend_verdict.voice.speech.stubs import StubRecognizer, ToneSynthesizer
from examples.frontend_backend_verdict.voice.speech.vad_energy import EnergyVad


def _vad_factory(kind: str) -> VadFactory:
    if kind == "energy":
        return EnergyVad
    from examples.frontend_backend_verdict.voice.speech.vad_silero import SileroVad

    return SileroVad


def _normalizer(enabled: bool) -> TextNormalizer:
    if not enabled:
        return IdentityNormalizer()
    from examples.frontend_backend_verdict.voice.speech.text_normalizer import NemotronTextNormalizer

    return NemotronTextNormalizer()


def build_speech_services(config: VoiceConfig, *, stub: bool = False) -> SpeechServices:
    """Riva ASR/TTS and the configured VAD, or deterministic stand-ins when ``stub``."""
    if stub:
        return SpeechServices(
            recognizer=StubRecognizer(),
            synthesizer=ToneSynthesizer(sample_rate=config.tts.sample_rate),
            vad_factory=EnergyVad,
            normalizer=IdentityNormalizer(),
        )
    from examples.frontend_backend_verdict.voice.speech.asr_riva import RivaStreamingRecognizer
    from examples.frontend_backend_verdict.voice.speech.tts_riva import RivaSynthesizer

    sessions = config.server.max_sessions
    return SpeechServices(
        recognizer=RivaStreamingRecognizer(
            config.asr.endpoint,
            max_streams=config.asr.max_concurrent_streams or sessions,
            interim_results=config.asr.interim_results,
        ),
        synthesizer=RivaSynthesizer(
            config.tts.endpoint,
            sample_rate=config.tts.sample_rate,
            max_streams=config.tts.max_concurrent_streams or 2 * sessions,
        ),
        vad_factory=_vad_factory(config.turn_detection.vad),
        normalizer=_normalizer(config.tts.normalize_text),
    )
