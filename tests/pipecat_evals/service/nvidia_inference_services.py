# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""CI-only Pipecat services backed by NVIDIA's hosted Inference API."""

from __future__ import annotations

import io
import os
import wave
from collections.abc import AsyncGenerator

import httpx
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.nvidia.llm import NvidiaLLMService
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

from examples.omni_assistant.nvidia_omni_multimodal_service import NvidiaOmniLLMService

MAGPIE_TTS_ENDPOINT = "https://inference-api.nvidia.com/v1/audio/nvidia/magpie-tts-multilingual-357m/synthesize"


def _api_key() -> str:
    key = os.getenv("NVIDIA_INFERENCE_API_KEY")
    if not key:
        raise RuntimeError("NVIDIA_INFERENCE_API_KEY is required for CI Inference API services")
    return key


class InferenceNvidiaLLMService(NvidiaLLMService):
    """Use the CI Inference API credential instead of the NVCF credential."""

    def __init__(self, *args, api_key: str | None = None, **kwargs) -> None:
        """Ignore the example-provided credential and use the CI-only key."""
        super().__init__(*args, api_key=_api_key(), **kwargs)


class InferenceNvidiaOmniLLMService(NvidiaOmniLLMService):
    """Use the CI Inference API credential for Omni requests."""

    def __init__(self, *args, api_key: str | None = None, **kwargs) -> None:
        """Ignore the example-provided credential and use the CI-only key."""
        super().__init__(*args, api_key=_api_key(), **kwargs)


class InferenceMagpieTTSService(TTSService):
    """Translate Pipecat TTS calls to NVIDIA's hosted multipart endpoint."""

    def can_generate_metrics(self) -> bool:
        """Indicate that this service reports TTFB and usage metrics."""
        return True

    def __init__(self, *args, settings=None, text_filters=None, text_transforms=None, **kwargs) -> None:
        """Accept the NVIDIA gRPC constructor shape used by the examples."""
        voice = getattr(settings, "voice", None) or "Magpie-Multilingual.EN-US.Aria"
        language = getattr(settings, "language", None) or "en-US"
        super().__init__(
            sample_rate=22050,
            push_start_frame=True,
            push_stop_frames=True,
            settings=TTSSettings(
                model="nvidia/nvidia/magpie-tts-multilingual-357m",
                voice=str(voice),
                language=str(language),
            ),
            text_filters=text_filters,
            text_transforms=text_transforms,
            stop_frame_timeout_s=float(kwargs.get("stop_frame_timeout_s", 3.0)),
        )

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Synthesize one text segment and emit its decoded PCM audio."""
        try:
            await self.start_tts_usage_metrics(text)
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(
                    MAGPIE_TTS_ENDPOINT,
                    headers={"Authorization": f"Bearer {_api_key()}"},
                    files={
                        "text": (None, text),
                        "language": (None, str(self._settings.language or "en-US")),
                        "voice": (None, str(self._settings.voice)),
                        "encoding": (None, "LINEAR_PCM"),
                    },
                )
            response.raise_for_status()
            with wave.open(io.BytesIO(response.content), "rb") as wav:
                if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
                    raise ValueError("Inference API TTS did not return 16-bit mono WAV audio")
                sample_rate = wav.getframerate()
                audio = wav.readframes(wav.getnframes())
            await self.stop_ttfb_metrics()
            yield TTSAudioRawFrame(audio=audio, sample_rate=sample_rate, num_channels=1, context_id=context_id)
        except Exception as exc:
            await self.stop_ttfb_metrics()
            yield ErrorFrame(error=f"Inference API TTS failed: {exc}")
