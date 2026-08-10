# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Realtime audio helpers: base64 PCM + Pipecat stream resampling.

Uses Pipecat's ``create_stream_resampler()`` (SOXR stream). v1 is linear PCM
only (``audio/pcm`` / ``pcm16``).
"""

from __future__ import annotations

import base64
from typing import Any

from pipecat.audio.utils import create_stream_resampler

PIPELINE_PCM_RATE = 16000
DEFAULT_CLIENT_PCM_RATE = 24000
SUPPORTED_PCM_RATES = frozenset({8000, 16000, 24000, 48000})
SUPPORTED_PCM_FORMAT_TYPES = frozenset({"audio/pcm", "pcm16"})
# Manual (push-to-talk) buffer: 60s of 16-bit mono at pipeline rate.
MAX_PENDING_INPUT_BYTES = PIPELINE_PCM_RATE * 2 * 60


def decode_base64_audio(audio_b64: str) -> bytes:
    """Decode ``input_audio_buffer.append`` audio."""
    if not isinstance(audio_b64, str) or not audio_b64:
        raise ValueError("audio must be a non-empty base64 string")
    try:
        return base64.b64decode(audio_b64, validate=False)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"invalid base64 audio: {exc}") from exc


def encode_base64_audio(pcm: bytes) -> str:
    """Encode PCM for ``response.output_audio.delta``."""
    return base64.b64encode(pcm).decode("ascii")


def require_supported_pcm_rate(rate: Any, *, param: str = "rate") -> int:
    """Return ``rate`` if supported; raise ``ValueError`` otherwise."""
    if isinstance(rate, bool) or not isinstance(rate, int):
        raise ValueError(f"{param} must be an integer PCM sample rate")
    if rate not in SUPPORTED_PCM_RATES:
        supported = ", ".join(str(r) for r in sorted(SUPPORTED_PCM_RATES))
        raise ValueError(f"unsupported PCM rate {rate}; supported: {supported}")
    return rate


def normalize_pcm_rate(rate: Any, *, default: int = DEFAULT_CLIENT_PCM_RATE) -> int:
    """Return a supported rate for internal resampling (already-validated sessions)."""
    if isinstance(rate, bool) or not isinstance(rate, int):
        return default
    if rate in SUPPORTED_PCM_RATES:
        return rate
    return default


def _validate_format_object(fmt: Any, *, param: str) -> None:
    if isinstance(fmt, str):
        if fmt not in SUPPORTED_PCM_FORMAT_TYPES:
            raise ValueError(
                f"{param} must be audio/pcm (pcm16); got {fmt!r}. G.711 and other codecs are not supported in v1"
            )
        return
    if not isinstance(fmt, dict):
        raise ValueError(f"{param} must be a format object or pcm16 string")
    typ = fmt.get("type")
    if not isinstance(typ, str) or typ not in SUPPORTED_PCM_FORMAT_TYPES:
        raise ValueError(
            f"{param}.type must be audio/pcm (pcm16); got {typ!r}. G.711 and other codecs are not supported in v1"
        )
    if "rate" in fmt and fmt.get("rate") is not None:
        require_supported_pcm_rate(fmt.get("rate"), param=f"{param}.rate")


def validate_session_audio_config(session_patch: dict[str, Any]) -> None:
    """Reject unsupported audio formats/rates in a session.update patch.

    Call before acknowledging ``session.updated``. Omitted fields are fine
    (defaults apply); explicit unsupported values fail closed.
    """
    if not isinstance(session_patch, dict):
        return

    for key in ("input_audio_format", "output_audio_format"):
        if key in session_patch and session_patch.get(key) is not None:
            _validate_format_object(session_patch.get(key), param=f"session.{key}")

    audio = session_patch.get("audio")
    if not isinstance(audio, dict):
        return
    for side in ("input", "output"):
        block = audio.get(side)
        if not isinstance(block, dict) or "format" not in block:
            continue
        if block.get("format") is None:
            continue
        _validate_format_object(block.get("format"), param=f"session.audio.{side}.format")


def _format_rate(fmt: Any) -> int | None:
    if isinstance(fmt, dict) and "rate" in fmt and fmt.get("rate") is not None:
        return require_supported_pcm_rate(fmt.get("rate"), param="format.rate")
    return None


def extract_client_pcm_rate(session_view: dict[str, Any] | None) -> int:
    """Client input PCM rate from the session view."""
    if not isinstance(session_view, dict):
        return DEFAULT_CLIENT_PCM_RATE
    audio = session_view.get("audio")
    if not isinstance(audio, dict):
        return DEFAULT_CLIENT_PCM_RATE
    inp = audio.get("input")
    if not isinstance(inp, dict):
        return DEFAULT_CLIENT_PCM_RATE
    try:
        rate = _format_rate(inp.get("format"))
    except ValueError:
        return DEFAULT_CLIENT_PCM_RATE
    return rate if rate is not None else DEFAULT_CLIENT_PCM_RATE


def extract_client_output_pcm_rate(session_view: dict[str, Any] | None) -> int:
    """Client output PCM rate from the session view (falls back to input)."""
    if not isinstance(session_view, dict):
        return DEFAULT_CLIENT_PCM_RATE
    audio = session_view.get("audio")
    if not isinstance(audio, dict):
        return DEFAULT_CLIENT_PCM_RATE
    out = audio.get("output")
    if not isinstance(out, dict):
        return DEFAULT_CLIENT_PCM_RATE
    try:
        rate = _format_rate(out.get("format"))
    except ValueError:
        return extract_client_pcm_rate(session_view)
    if rate is not None:
        return rate
    return extract_client_pcm_rate(session_view)


def extract_client_input_format_type(session_view: dict[str, Any] | None) -> str:
    """Client input format type (default ``audio/pcm``)."""
    if not isinstance(session_view, dict):
        return "audio/pcm"
    audio = session_view.get("audio")
    if not isinstance(audio, dict):
        return "audio/pcm"
    inp = audio.get("input")
    if not isinstance(inp, dict):
        return "audio/pcm"
    fmt = inp.get("format")
    if isinstance(fmt, str) and fmt in SUPPORTED_PCM_FORMAT_TYPES:
        return "audio/pcm"
    if isinstance(fmt, dict) and isinstance(fmt.get("type"), str) and fmt["type"]:
        if fmt["type"] in SUPPORTED_PCM_FORMAT_TYPES:
            return "audio/pcm"
        return fmt["type"]
    return "audio/pcm"


class AudioResampler:
    """Directional wrapper around Pipecat ``create_stream_resampler()``.

    Separate stream instances for uplink (client to pipeline) and downlink
    (pipeline to client) so SOXR stream state does not cross directions.
    """

    def __init__(self) -> None:
        """Create uplink and downlink Pipecat stream resamplers."""
        self._uplink = create_stream_resampler()
        self._downlink = create_stream_resampler()

    def reset(self) -> None:
        """Recreate stream resamplers (required when client PCM rates change).

        Pipecat ``SOXRStreamAudioResampler`` cannot switch ``in_rate``/``out_rate``
        after the first chunk; mid-session ``session.update`` rate changes must
        replace the streams or subsequent audio is dropped.
        """
        self._uplink = create_stream_resampler()
        self._downlink = create_stream_resampler()

    async def to_pipeline(
        self,
        pcm: bytes,
        client_rate: int,
        *,
        pipeline_rate: int = PIPELINE_PCM_RATE,
    ) -> bytes:
        """Resample client PCM → pipeline rate (default 16 kHz)."""
        if not pcm:
            return b""
        rate = normalize_pcm_rate(client_rate)
        target = normalize_pcm_rate(pipeline_rate, default=PIPELINE_PCM_RATE)
        if rate == target:
            return pcm
        return await self._uplink.resample(pcm, rate, target)

    async def from_pipeline(
        self,
        pcm: bytes,
        client_rate: int,
        *,
        pipeline_rate: int = PIPELINE_PCM_RATE,
    ) -> bytes:
        """Resample pipeline PCM → client rate."""
        if not pcm:
            return b""
        rate = normalize_pcm_rate(client_rate)
        source = normalize_pcm_rate(pipeline_rate, default=PIPELINE_PCM_RATE)
        if rate == source:
            return pcm
        return await self._downlink.resample(pcm, source, rate)
