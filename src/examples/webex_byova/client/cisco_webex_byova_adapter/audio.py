# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Audio conversion helpers for the Cisco Webex BYOVA adapter."""

from __future__ import annotations

import audioop

from cisco_webex_byova_adapter.generated import voicevirtualagent_pb2

# Nemotron accepts caller input at 16 kHz. Output frames declare their own rate.
TARGET_SAMPLE_RATE = 16000

# Webex BYoVA Prompt.audio_content is consumed as 8 kHz mono **G.711 μ-law**
# (the PSTN telephony standard, also what the Cisco-published reference
# connectors output). Sending 16 kHz PCM or even 8 kHz LINEAR16 plays back as
# noise / chipmunked audio on the call.
BYOVA_OUT_SAMPLE_RATE = 8000


def normalize_caller_audio(
    audio: bytes,
    encoding: int,
    sample_rate_hz: int,
    resample_state: object = None,
) -> tuple[bytes, object]:
    """Convert supported BYoVA caller audio into 16 kHz mono signed 16-bit PCM.

    Webex Universal Harness omits the ``encoding`` field on caller audio
    (sends ``UNSPECIFIED_FORMAT = 0``) and the implicit default for PSTN
    telephony is G.711 μ-law at 8 kHz. We treat ``UNSPECIFIED`` as μ-law for
    that reason.

    ``resample_state`` is the opaque state returned by ``audioop.ratecv``.
    Persist it across calls within a single caller session so the resampling
    is continuous (otherwise every Webex audio chunk — ~20 ms — would have a
    small discontinuity at its boundary, which can be enough to break STT
    transcription even when VAD still picks up energy).
    """
    if encoding == voicevirtualagent_pb2.VoiceInput.UNSPECIFIED_FORMAT:
        encoding = voicevirtualagent_pb2.VoiceInput.MULAW_FORMAT
    if not sample_rate_hz:
        sample_rate_hz = 8000

    pcm = audio
    if encoding == voicevirtualagent_pb2.VoiceInput.MULAW_FORMAT:
        pcm = audioop.ulaw2lin(audio, 2)
    elif encoding == voicevirtualagent_pb2.VoiceInput.ALAW_FORMAT:
        pcm = audioop.alaw2lin(audio, 2)
    elif encoding != voicevirtualagent_pb2.VoiceInput.LINEAR16_FORMAT:
        raise ValueError(f"Unsupported caller audio encoding: {encoding}")

    if sample_rate_hz != TARGET_SAMPLE_RATE:
        pcm, resample_state = audioop.ratecv(pcm, 2, 1, sample_rate_hz, TARGET_SAMPLE_RATE, resample_state)

    return pcm, resample_state


def to_byova_audio(
    audio_pcm16: bytes,
    sample_rate_hz: int = TARGET_SAMPLE_RATE,
    num_channels: int = 1,
) -> bytes:
    """Convert Nemotron PCM into the G.711 μ-law payload expected by Webex.

    Input is mono LINEAR16 PCM at its declared rate. Output is 8 kHz mono
    μ-law for ``Prompt.audio_content``.
    """
    if not audio_pcm16:
        return audio_pcm16
    if not isinstance(sample_rate_hz, int) or sample_rate_hz <= 0:
        raise ValueError("PCM16 sample rate must be a positive integer")
    if num_channels != 1:
        raise ValueError("only mono PCM16 output is supported")
    if len(audio_pcm16) % 2:
        raise ValueError("PCM16 audio must contain complete samples")
    pcm_8k = audio_pcm16
    if sample_rate_hz != BYOVA_OUT_SAMPLE_RATE:
        pcm_8k, _ = audioop.ratecv(audio_pcm16, 2, 1, sample_rate_hz, BYOVA_OUT_SAMPLE_RATE, None)
    return audioop.lin2ulaw(pcm_8k, 2)
