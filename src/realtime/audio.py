# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""OpenAI Realtime PCM and G.711 codec helpers with streaming resampling.

Wire audio is mono linear PCM16 or fixed-rate 8 kHz G.711. Codec conversion
always happens on the wire side of resampling, while the pipeline continues to
receive and produce signed 16-bit PCM.
"""

from __future__ import annotations

import base64
import math
from typing import Any

import numpy as np
import soxr
from pipecat.audio.utils import alaw_to_pcm, create_stream_resampler, pcm_to_alaw, pcm_to_ulaw, ulaw_to_pcm

PIPELINE_PCM_RATE = 16000
PIPELINE_OUTPUT_PCM_RATE = 22050
DEFAULT_CLIENT_PCM_RATE = 24000
G711_PCM_RATE = 8000
SUPPORTED_PCM_RATES = frozenset({8000, 16000, 24000, 48000})
SUPPORTED_PCM_FORMAT_TYPES = frozenset({"audio/pcm"})
SUPPORTED_G711_FORMAT_TYPES = frozenset({"audio/pcmu", "audio/pcma"})
SUPPORTED_AUDIO_FORMAT_TYPES = SUPPORTED_PCM_FORMAT_TYPES | SUPPORTED_G711_FORMAT_TYPES
MAX_PENDING_INPUT_SECONDS = 60
# Pipeline-side cap: 60s of 16-bit mono PCM at the pipeline rate.
MAX_PENDING_INPUT_BYTES = PIPELINE_PCM_RATE * 2 * MAX_PENDING_INPUT_SECONDS


class _FlushableSoxrStream:
    """One mono PCM16 SOXR stream with an explicit terminal flush."""

    def __init__(self) -> None:
        self._in_rate: int | None = None
        self._out_rate: int | None = None
        self._stream: soxr.ResampleStream | None = None

    def reset(self) -> None:
        """Discard stream history and any pending output samples."""
        self._in_rate = None
        self._out_rate = None
        self._stream = None

    def _require_stream(self, in_rate: int, out_rate: int) -> soxr.ResampleStream:
        if self._stream is None:
            self._in_rate = in_rate
            self._out_rate = out_rate
            self._stream = soxr.ResampleStream(
                in_rate=in_rate,
                out_rate=out_rate,
                num_channels=1,
                quality="VHQ",
                dtype="int16",
            )
        elif self._in_rate != in_rate or self._out_rate != out_rate:
            raise ValueError(
                "SOXR output resampler cannot change rates without a response boundary: "
                f"expected {self._in_rate}->{self._out_rate}, got {in_rate}->{out_rate}"
            )
        return self._stream

    async def resample(self, audio: bytes, in_rate: int, out_rate: int) -> bytes:
        """Resample one continuous PCM16 chunk."""
        if in_rate == out_rate:
            return audio
        stream = self._require_stream(in_rate, out_rate)
        source = np.frombuffer(audio, dtype=np.int16)
        return stream.resample_chunk(source).astype(np.int16).tobytes()

    async def flush(self, in_rate: int, out_rate: int) -> bytes:
        """Return delayed output samples, then make the stream reusable."""
        if in_rate == out_rate or self._stream is None:
            self.reset()
            return b""
        if self._in_rate != in_rate or self._out_rate != out_rate:
            raise ValueError(
                "SOXR output flush rates do not match the active stream: "
                f"expected {self._in_rate}->{self._out_rate}, got {in_rate}->{out_rate}"
            )
        try:
            pending = self._stream.resample_chunk(np.empty(0, dtype=np.int16), last=True)
            return pending.astype(np.int16).tobytes()
        finally:
            self.reset()


def require_supported_audio_format_type(format_type: Any, *, param: str = "format.type") -> str:
    """Return a canonical Realtime wire format type or fail closed."""
    if not isinstance(format_type, str) or format_type not in SUPPORTED_AUDIO_FORMAT_TYPES:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_FORMAT_TYPES))
        raise ValueError(f"{param} must be one of: {supported}")
    return format_type


def require_audio_format_rate(
    format_type: Any,
    rate: Any | None,
    *,
    param: str = "rate",
    default_pcm_rate: int = DEFAULT_CLIENT_PCM_RATE,
) -> int:
    """Resolve a PCM rate while enforcing the fixed 8 kHz G.711 clock."""
    canonical_type = require_supported_audio_format_type(format_type)
    if canonical_type in SUPPORTED_G711_FORMAT_TYPES:
        if rate is not None and (isinstance(rate, bool) or not isinstance(rate, int) or rate != G711_PCM_RATE):
            raise ValueError(f"{param} must be {G711_PCM_RATE} for {canonical_type}")
        return G711_PCM_RATE
    return require_supported_pcm_rate(default_pcm_rate if rate is None else rate, param=param)


def max_pending_input_bytes(
    format_type: Any,
    rate: Any | None = None,
    *,
    duration_seconds: int | float = MAX_PENDING_INPUT_SECONDS,
) -> int:
    """Return the encoded wire-byte cap for a duration of mono audio."""
    if (
        isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, int | float)
        or not math.isfinite(float(duration_seconds))
        or duration_seconds <= 0
    ):
        raise ValueError("duration_seconds must be a positive finite number")
    canonical_type = require_supported_audio_format_type(format_type)
    sample_rate = require_audio_format_rate(canonical_type, rate)
    bytes_per_sample = 2 if canonical_type == "audio/pcm" else 1
    return int(sample_rate * float(duration_seconds) * bytes_per_sample)


def max_base64_audio_chars(max_decoded_bytes: int) -> int:
    """Return the exact padded-base64 character bound for a byte limit."""
    if isinstance(max_decoded_bytes, bool) or not isinstance(max_decoded_bytes, int) or max_decoded_bytes < 0:
        raise ValueError("max_decoded_bytes must be a non-negative integer")
    return 4 * ((max_decoded_bytes + 2) // 3)


def decode_base64_audio(
    audio_b64: str,
    *,
    format_type: str = "audio/pcm",
    max_decoded_bytes: int | None = None,
) -> bytes:
    """Decode and validate one format-aware ``input_audio_buffer.append`` payload."""
    canonical_type = require_supported_audio_format_type(format_type)
    if not isinstance(audio_b64, str) or not audio_b64:
        raise ValueError("audio must be a non-empty base64 string")
    if max_decoded_bytes is not None:
        max_base64_chars = max_base64_audio_chars(max_decoded_bytes)
        if len(audio_b64) > max_base64_chars:
            raise ValueError(f"audio exceeds the {max_decoded_bytes}-byte decoded limit")
    try:
        audio = base64.b64decode(audio_b64, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"invalid base64 audio: {exc}") from exc
    if not audio:
        raise ValueError("audio must decode to at least one byte")
    if max_decoded_bytes is not None and len(audio) > max_decoded_bytes:
        raise ValueError(f"audio exceeds the {max_decoded_bytes}-byte decoded limit")
    if canonical_type == "audio/pcm" and len(audio) % 2:
        raise ValueError("PCM16 audio must contain an even number of bytes")
    return audio


def encode_base64_audio(audio: bytes, *, format_type: str = "audio/pcm") -> str:
    """Validate and encode wire audio for ``response.output_audio.delta``."""
    canonical_type = require_supported_audio_format_type(format_type)
    if not isinstance(audio, bytes | bytearray | memoryview):
        raise ValueError("audio must be bytes-like")
    wire_audio = bytes(audio)
    if canonical_type == "audio/pcm" and len(wire_audio) % 2:
        raise ValueError("PCM16 audio must contain an even number of bytes")
    return base64.b64encode(wire_audio).decode("ascii")


def require_supported_pcm_rate(rate: Any, *, param: str = "rate") -> int:
    """Return ``rate`` if supported; raise ``ValueError`` otherwise."""
    if isinstance(rate, bool) or not isinstance(rate, int):
        raise ValueError(f"{param} must be an integer PCM sample rate")
    if rate not in SUPPORTED_PCM_RATES:
        supported = ", ".join(str(r) for r in sorted(SUPPORTED_PCM_RATES))
        raise ValueError(f"unsupported PCM rate {rate}; supported: {supported}")
    return rate


def require_supported_pipeline_pcm_rate(rate: Any, *, param: str = "pipeline_rate") -> int:
    """Return a supported internal PCM rate, including Magpie's 22.05 kHz output."""
    if isinstance(rate, bool) or not isinstance(rate, int):
        raise ValueError(f"{param} must be an integer PCM sample rate")
    if rate == PIPELINE_OUTPUT_PCM_RATE:
        return PIPELINE_OUTPUT_PCM_RATE
    return require_supported_pcm_rate(rate, param=param)


def _format_type(fmt: Any) -> str | None:
    if fmt is None:
        return None
    if isinstance(fmt, str):
        return require_supported_audio_format_type(fmt)
    if isinstance(fmt, dict):
        return require_supported_audio_format_type(fmt.get("type"))
    raise ValueError("format must be an audio format object or type string")


def _format_rate(fmt: Any) -> int | None:
    format_type = _format_type(fmt)
    if format_type in SUPPORTED_G711_FORMAT_TYPES:
        if isinstance(fmt, dict) and "rate" in fmt:
            raise ValueError(f"{format_type} has a fixed {G711_PCM_RATE} Hz rate and does not accept format.rate")
        return G711_PCM_RATE
    if isinstance(fmt, dict) and "rate" in fmt and fmt.get("rate") is not None:
        return require_supported_pcm_rate(fmt.get("rate"), param="format.rate")
    return None


def extract_client_pcm_rate(session_view: dict[str, Any] | None) -> int:
    """Client input wire rate from the session view, including fixed-rate G.711."""
    if not isinstance(session_view, dict):
        return DEFAULT_CLIENT_PCM_RATE
    audio = session_view.get("audio")
    if not isinstance(audio, dict):
        return DEFAULT_CLIENT_PCM_RATE
    inp = audio.get("input")
    if not isinstance(inp, dict):
        return DEFAULT_CLIENT_PCM_RATE
    rate = _format_rate(inp.get("format"))
    return rate if rate is not None else DEFAULT_CLIENT_PCM_RATE


def extract_client_output_pcm_rate(session_view: dict[str, Any] | None) -> int:
    """Client output wire rate from the session view (falls back to input)."""
    if not isinstance(session_view, dict):
        return DEFAULT_CLIENT_PCM_RATE
    audio = session_view.get("audio")
    if not isinstance(audio, dict):
        return DEFAULT_CLIENT_PCM_RATE
    out = audio.get("output")
    if not isinstance(out, dict):
        return DEFAULT_CLIENT_PCM_RATE
    rate = _format_rate(out.get("format"))
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
    return _format_type(inp.get("format")) or "audio/pcm"


def extract_client_output_format_type(session_view: dict[str, Any] | None) -> str:
    """Client output format type (default ``audio/pcm``)."""
    if not isinstance(session_view, dict):
        return "audio/pcm"
    audio = session_view.get("audio")
    if not isinstance(audio, dict):
        return "audio/pcm"
    out = audio.get("output")
    if not isinstance(out, dict):
        return "audio/pcm"
    return _format_type(out.get("format")) or "audio/pcm"


class _PassthroughResampler:
    """Minimal identity resampler used around Pipecat's codec primitives."""

    async def resample(self, audio: bytes, in_rate: int, out_rate: int) -> bytes:
        if in_rate != out_rate:
            raise ValueError(f"identity resampler cannot convert {in_rate} Hz to {out_rate} Hz")
        return audio


_PASSTHROUGH_RESAMPLER = _PassthroughResampler()


def _require_pcm16_bytes(pcm: bytes, *, param: str = "pcm") -> None:
    if len(pcm) % 2:
        raise ValueError(f"{param} must contain complete signed 16-bit PCM samples")


async def _decode_g711(audio: bytes, format_type: str, out_rate: int, resampler: Any) -> bytes:
    if format_type == "audio/pcmu":
        return await ulaw_to_pcm(audio, G711_PCM_RATE, out_rate, resampler)
    if format_type == "audio/pcma":
        return await alaw_to_pcm(audio, G711_PCM_RATE, out_rate, resampler)
    raise ValueError(f"unsupported G.711 input format {format_type!r}")


async def _encode_g711(pcm: bytes, format_type: str, in_rate: int, resampler: Any) -> bytes:
    if format_type == "audio/pcmu":
        return await pcm_to_ulaw(pcm, in_rate, G711_PCM_RATE, resampler)
    if format_type == "audio/pcma":
        return await pcm_to_alaw(pcm, in_rate, G711_PCM_RATE, resampler)
    raise ValueError(f"unsupported G.711 output format {format_type!r}")


class AudioResampler:
    """Directional wire-codec converters with independent PCM stream state."""

    def __init__(self) -> None:
        """Create independent uplink and downlink stream resamplers."""
        self._uplink = create_stream_resampler()
        self._downlink = _FlushableSoxrStream()
        self._uplink_signature: tuple[str, int, int] | None = None
        self._downlink_signature: tuple[str, int, int] | None = None

    def reset(self) -> None:
        """Reset both directions before applying a wire-format transition."""
        self.reset_uplink()
        self.reset_downlink()

    def reset_uplink(self) -> None:
        """Discard input stream history before an input-format transition."""
        self._uplink = create_stream_resampler()
        self._uplink_signature = None

    def reset_downlink(self) -> None:
        """Discard output history after cancellation or before a format transition."""
        self._downlink.reset()
        self._downlink_signature = None

    def for_format_transition(
        self,
        *,
        reset_uplink: bool,
        reset_downlink: bool,
    ) -> AudioResampler:
        """Prepare independent direction state for an atomic format transition."""
        prepared = AudioResampler()
        if not reset_uplink:
            prepared._uplink = self._uplink
            prepared._uplink_signature = self._uplink_signature
        if not reset_downlink:
            prepared._downlink = self._downlink
            prepared._downlink_signature = self._downlink_signature
        return prepared

    @staticmethod
    def _require_stream_signature(
        active: tuple[str, int, int] | None,
        requested: tuple[str, int, int],
        *,
        direction: str,
    ) -> tuple[str, int, int]:
        if active is not None and active != requested:
            raise ValueError(
                f"Realtime {direction} audio format changed from {active} to {requested}; "
                f"reset_{direction}() is required before the transition"
            )
        return requested

    async def to_pipeline(
        self,
        audio: bytes,
        client_rate: int,
        *,
        pipeline_rate: int = PIPELINE_PCM_RATE,
        format_type: str = "audio/pcm",
    ) -> bytes:
        """Decode client wire audio and stream PCM16 at the pipeline rate."""
        canonical_type = require_supported_audio_format_type(format_type)
        rate = require_audio_format_rate(canonical_type, client_rate, param="client_rate")
        target = require_supported_pcm_rate(pipeline_rate, param="pipeline_rate")
        if not audio:
            return b""
        signature = (canonical_type, rate, target)
        self._uplink_signature = self._require_stream_signature(
            self._uplink_signature,
            signature,
            direction="uplink",
        )
        if canonical_type in SUPPORTED_G711_FORMAT_TYPES:
            return await _decode_g711(audio, canonical_type, target, self._uplink)
        _require_pcm16_bytes(audio, param="client PCM16 audio")
        if rate == target:
            return audio
        return await self._uplink.resample(audio, rate, target)

    @staticmethod
    async def complete_input_to_pipeline(
        audio: bytes,
        client_rate: int,
        *,
        pipeline_rate: int = PIPELINE_PCM_RATE,
        format_type: str = "audio/pcm",
    ) -> bytes:
        """Decode and one-shot resample a complete manually committed buffer."""
        canonical_type = require_supported_audio_format_type(format_type)
        rate = require_audio_format_rate(canonical_type, client_rate, param="client_rate")
        target = require_supported_pcm_rate(pipeline_rate, param="pipeline_rate")
        if not audio:
            return b""
        if canonical_type in SUPPORTED_G711_FORMAT_TYPES:
            pcm = await _decode_g711(audio, canonical_type, rate, _PASSTHROUGH_RESAMPLER)
        else:
            pcm = audio
        _require_pcm16_bytes(pcm, param="decoded input PCM16 audio")
        if rate == target:
            return pcm
        source = np.frombuffer(pcm, dtype=np.int16)
        return soxr.resample(source, rate, target, quality="VHQ").astype(np.int16).tobytes()

    async def from_pipeline(
        self,
        pcm: bytes,
        client_rate: int,
        *,
        pipeline_rate: int = PIPELINE_PCM_RATE,
        format_type: str = "audio/pcm",
    ) -> bytes:
        """Resample pipeline PCM16, then encode it in the client wire format."""
        canonical_type = require_supported_audio_format_type(format_type)
        rate = require_audio_format_rate(canonical_type, client_rate, param="client_rate")
        source = require_supported_pipeline_pcm_rate(pipeline_rate)
        _require_pcm16_bytes(pcm, param="pipeline PCM16 audio")
        if not pcm:
            return b""
        signature = (canonical_type, source, rate)
        self._downlink_signature = self._require_stream_signature(
            self._downlink_signature,
            signature,
            direction="downlink",
        )
        if canonical_type in SUPPORTED_G711_FORMAT_TYPES:
            return await _encode_g711(pcm, canonical_type, source, self._downlink)
        if rate == source:
            return pcm
        return await self._downlink.resample(pcm, source, rate)

    async def flush_from_pipeline(
        self,
        client_rate: int,
        *,
        pipeline_rate: int = PIPELINE_OUTPUT_PCM_RATE,
        format_type: str = "audio/pcm",
    ) -> bytes:
        """Flush delayed PCM and encode the tail at a response boundary."""
        canonical_type = require_supported_audio_format_type(format_type)
        rate = require_audio_format_rate(canonical_type, client_rate, param="client_rate")
        source = require_supported_pipeline_pcm_rate(pipeline_rate)
        signature = (canonical_type, source, rate)
        if self._downlink_signature is None:
            return b""
        self._require_stream_signature(self._downlink_signature, signature, direction="downlink")
        pcm = await self._downlink.flush(source, rate)
        if not pcm or canonical_type == "audio/pcm":
            return pcm
        return await _encode_g711(pcm, canonical_type, rate, _PASSTHROUGH_RESAMPLER)
