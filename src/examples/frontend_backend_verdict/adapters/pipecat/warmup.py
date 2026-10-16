# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speech warm-up at start-up (S12): one short ASR request and one short TTS request.

v1 warmed its own Riva clients the same way (a silent utterance through streaming ASR, and
"Hello." through TTS), so the first real turn does not pay the channel setup. Here the probes
go directly through ``riva.client`` on a worker thread; the Pipecat services open their own
channels per session. A failure is reported to clients as ``server_busy``, as in v1.
"""

from __future__ import annotations

import asyncio

from examples.frontend_backend_verdict.core.config.catalog import SpeechEndpoint
from examples.frontend_backend_verdict.core.config.voice import VoiceConfig
from examples.frontend_backend_verdict.core.errors import SpeechServiceError

WARMUP_TIMEOUT_S = 30.0
_SILENCE_16K = b"\x00\x00" * 8000  # 0.5 s of 16 kHz PCM16


def _auth(endpoint: SpeechEndpoint):  # noqa: ANN202 - riva.client.Auth
    import riva.client  # noqa: PLC0415

    return riva.client.Auth(None, endpoint.use_ssl, endpoint.server, endpoint.metadata())


def _warm_asr(endpoint: SpeechEndpoint) -> None:
    import riva.client  # noqa: PLC0415

    service = riva.client.ASRService(_auth(endpoint))
    config = riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            language_code=endpoint.language_code or "en-US",
            model=endpoint.model or "",
            sample_rate_hertz=16000,
            audio_channel_count=1,
            max_alternatives=1,
        ),
        interim_results=False,
    )
    for _ in service.streaming_response_generator(audio_chunks=iter([_SILENCE_16K]), streaming_config=config):
        pass


def _warm_tts(endpoint: SpeechEndpoint, sample_rate: int) -> None:
    import riva.client  # noqa: PLC0415

    service = riva.client.SpeechSynthesisService(_auth(endpoint))
    response = service.synthesize(
        "Hello.",
        voice_name=endpoint.voice_id or None,
        language_code=endpoint.language_code or "en-US",
        sample_rate_hz=sample_rate,
    )
    if not response.audio:
        raise SpeechServiceError("TTS returned no audio")


async def warm_up(config: VoiceConfig) -> None:
    """Probe ASR and TTS once; raises :class:`SpeechServiceError` naming the endpoint that failed."""
    asr, tts = config.asr.endpoint, config.tts.endpoint
    try:
        await asyncio.wait_for(asyncio.to_thread(_warm_asr, asr), timeout=WARMUP_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 - reported as server_busy
        raise SpeechServiceError(f"ASR warm-up failed against {asr.describe()}: {exc}") from exc
    try:
        await asyncio.wait_for(asyncio.to_thread(_warm_tts, tts, config.tts.sample_rate), timeout=WARMUP_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 - reported as server_busy
        raise SpeechServiceError(f"TTS warm-up failed against {tts.describe()}: {exc}") from exc
