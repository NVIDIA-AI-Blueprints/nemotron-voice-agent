# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speech services of the Pipecat adapter, from the same resolved catalog endpoints as the τ³ adapter.

ASR is the repository's ``NvidiaForceEouSTTService`` (a VAD stop forces the end of the
utterance, as in every repository pipeline); TTS is Pipecat's ``NvidiaTTSService`` with the
shared Nemotron speech text filter, in ``per_sentence`` mode. NVCF endpoints get SSL, the
API key and the function id; local ``nemo-speech`` endpoints get none of them.

The stubs at the bottom replace both services and the VAD for the no-GPU tests: an energy
VAD (the τ² replay tools send tones, which Silero does not treat as speech), a stub ASR
that emits one finalized transcript per turn, and a tone TTS built on Pipecat's own
``TTSService`` so the real serialization queue, stop frames and end markers are exercised.
"""

from __future__ import annotations

import math
import struct
from collections.abc import AsyncGenerator
from typing import Any

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings, NvidiaTTSSynthesisMode
from pipecat.services.tts_service import TTSService
from pipecat.utils.time import time_now_iso8601

from examples.frontend_delegation_hermes.adapters.pipecat.settings import PipecatHeadSettings
from examples.frontend_delegation_hermes.core.catalog import SpeechEndpoint
from examples.frontend_delegation_hermes.core.voice_config import VoiceConfig


def _model_function_map(endpoint: SpeechEndpoint, default_model: str) -> dict[str, str] | None:
    if endpoint.function_id or endpoint.model:
        return {"function_id": endpoint.function_id, "model_name": endpoint.model or default_model}
    return None


def build_vad(settings: PipecatHeadSettings) -> VADAnalyzer:
    """Silero with Pipecat's ``VADParams`` defaults, overridden only by the settings file."""
    return SileroVADAnalyzer(params=VADParams(**settings.vad.overrides()))


def build_stt(config: VoiceConfig) -> FrameProcessor:
    """``NvidiaForceEouSTTService`` on the catalog's ASR endpoint."""
    from pipecat.services.nvidia.stt import NvidiaSTTSettings  # noqa: PLC0415

    from examples.shared.nvidia_force_eou_stt import NvidiaForceEouSTTService  # noqa: PLC0415

    endpoint = config.asr.endpoint
    kwargs: dict[str, Any] = {
        "api_key": endpoint.api_key or None,
        "server": endpoint.server,
        "use_ssl": endpoint.use_ssl,
        "stop_history": 400,
    }
    function_map = _model_function_map(endpoint, "custom-asr")
    if function_map:
        kwargs["model_function_map"] = function_map
    if endpoint.language_code:
        kwargs["settings"] = NvidiaSTTSettings(language=endpoint.language_code)
    return NvidiaForceEouSTTService(**kwargs)


def build_tts(config: VoiceConfig, settings: PipecatHeadSettings) -> FrameProcessor:
    """``NvidiaTTSService`` on the catalog's TTS endpoint (``per_sentence``, Nemotron text filter)."""
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
        "text_filters": [NemotronSpeechTextFilter()],
    }
    function_map = _model_function_map(endpoint, "")
    if function_map:
        kwargs["model_function_map"] = function_map
    return NvidiaTTSService(**kwargs)


# -- no-GPU stand-ins (tests and the stub smoke) ---------------------------------------------


class EnergyVADAnalyzer(VADAnalyzer):
    """Speech confidence from frame energy (a tone is speech, silence is not)."""

    def __init__(self, *, sample_rate: int | None = None, params: VADParams | None = None) -> None:
        """Pipecat's VAD state machine over an energy measure."""
        super().__init__(sample_rate=sample_rate, params=params or VADParams())

    def num_frames_required(self) -> int:
        """20 ms frames."""
        return int(self.sample_rate * 0.02) if self.sample_rate else 320

    def voice_confidence(self, buffer: bytes) -> float:
        """1.0 above a fixed RMS, else 0.0."""
        count = len(buffer) // 2
        if not count:
            return 0.0
        samples = struct.unpack(f"<{count}h", buffer[: count * 2])
        rms = math.sqrt(sum(sample * sample for sample in samples) / count)
        return 1.0 if rms > 500 else 0.0


class StubSTT(FrameProcessor):
    """Emits one finalized ``TranscriptionFrame("utterance N")`` at each VAD stop."""

    def __init__(self, **kwargs: Any) -> None:
        """Count utterances from one."""
        super().__init__(**kwargs)
        self.count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Forward everything; add a transcript after a VAD stop."""
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, VADUserStoppedSpeakingFrame) and direction == FrameDirection.DOWNSTREAM:
            self.count += 1
            await self.push_frame(
                TranscriptionFrame(
                    text=f"utterance {self.count}", user_id="", timestamp=time_now_iso8601(), finalized=True
                )
            )


class ToneTTS(TTSService):
    """A 440 Hz tone, ``ms_per_char`` long per character, through Pipecat's own TTS machinery."""

    def __init__(self, *, sample_rate: int = 16000, ms_per_char: float = 5.0, **kwargs: Any) -> None:
        """Pipecat pushes the stop frame per sentence, like ``NvidiaTTSService``."""
        super().__init__(sample_rate=sample_rate, push_stop_frames=True, **kwargs)
        self._ms_per_char = ms_per_char

    def can_generate_metrics(self) -> bool:
        """No metrics."""
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """One tone per sentence, in 20 ms chunks."""
        rate = self.sample_rate
        total = int(rate * self._ms_per_char * max(1, len(text)) / 1000)
        chunk = int(rate * 0.02)
        for start in range(0, total, chunk):
            count = min(chunk, total - start)
            pcm = struct.pack(
                f"<{count}h",
                *(int(8000 * math.sin(2 * math.pi * 440 * (start + i) / rate)) for i in range(count)),
            )
            yield TTSAudioRawFrame(audio=pcm, sample_rate=rate, num_channels=1, context_id=context_id)
