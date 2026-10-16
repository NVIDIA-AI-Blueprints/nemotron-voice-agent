# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated wire records for a sealed, finite audio session.

Offsets count mono PCM16 samples at 16 kHz. Neither arrival time nor upload
packet boundaries have presentation meaning.
"""

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from typing import Any

from .models import RATE, AudioSource, SessionResult

PROTOCOL = "nva.offline.v1"
HEADER = struct.Struct("!4sQQI")
CAPABILITIES = ("aligned_output", "explicit_eof", "offset_audio", "service_trace")
_PRIVATE_KEYS = frozenset({"api_key", "authorization", "password", "secret", "token", "credentials"})


class OfflineProtocolError(ValueError):
    """A client-safe protocol failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        """Store public error information; provider exceptions never enter this record."""
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class SessionLimits:
    """Bound memory, packets, and failure watchdogs independently of audio time."""

    max_input_samples: int = RATE * 3600
    max_output_samples: int = RATE * 7200
    max_packet_samples: int = 65536
    output_packet_samples: int = 1600
    max_control_bytes: int = 32768
    upload_timeout_seconds: float = 300
    execution_timeout_seconds: float = 1800
    output_timeout_seconds: float = 300

    def __post_init__(self):
        """Reject invalid bounds before accepting a WebSocket."""
        for name in (
            "max_input_samples",
            "max_output_samples",
            "max_packet_samples",
            "output_packet_samples",
            "max_control_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("upload_timeout_seconds", "execution_timeout_seconds", "output_timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def integer_field(message: dict[str, Any], name: str) -> int:
    """Read a nonnegative integer without accepting booleans or coercion."""
    value = message.get(name)
    if type(value) is not int or value < 0:
        raise OfflineProtocolError("invalid_control", f"{name} must be a nonnegative integer")
    return value


def parse_control(text: str, limits: SessionLimits) -> dict[str, Any]:
    """Read one bounded JSON object and reject nonfinite JSON numbers."""
    if len(text.encode("utf-8")) > limits.max_control_bytes:
        raise OfflineProtocolError("control_too_large", "Control message exceeds the configured limit")
    try:
        message = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (ValueError, RecursionError) as error:
        raise OfflineProtocolError("invalid_control", "Control message must be valid finite JSON") from error
    if not isinstance(message, dict):
        raise OfflineProtocolError("invalid_control", "Control message must be a JSON object")
    return message


def validate_start(message: dict[str, Any], limits: SessionLimits) -> dict[str, Any]:
    """Validate format and declared extent before any input or provider starts."""
    if message.get("type") != "offline.session.start" or message.get("protocol") != PROTOCOL:
        raise OfflineProtocolError("unsupported_protocol", "Expected an nva.offline.v1 session start")
    if (
        type(message.get("sample_rate")) is not int
        or message["sample_rate"] != RATE
        or type(message.get("channels")) is not int
        or message["channels"] != 1
        or message.get("encoding") != "pcm_s16le"
    ):
        raise OfflineProtocolError("unsupported_format", "Input must be 16 kHz mono PCM16 little endian")
    extent = integer_field(message, "input_samples")
    if extent > limits.max_input_samples:
        raise OfflineProtocolError("input_too_large", "Declared input exceeds the configured duration limit")
    if "config" in message and not isinstance(message["config"], dict):
        raise OfflineProtocolError("invalid_control", "config must be a JSON object")
    return message


def decode_packet(data: bytes, magic: bytes, max_samples: int) -> tuple[int, int, bytes]:
    """Validate an audio envelope without inferring time from packet arrival."""
    if len(data) < HEADER.size:
        raise OfflineProtocolError("invalid_audio", "Truncated audio packet header")
    actual_magic, sequence, offset, count = HEADER.unpack_from(data)
    if actual_magic != magic or count == 0 or len(data) != HEADER.size + count * 2:
        raise OfflineProtocolError("invalid_audio", "Audio packet magic or PCM sample count is invalid")
    if count > max_samples:
        raise OfflineProtocolError("packet_too_large", "Audio packet exceeds the configured sample limit")
    return sequence, offset, data[HEADER.size :]


def encode_packet(magic: bytes, sequence: int, offset: int, pcm: bytes) -> bytes:
    """Serialize an ordered input or committed output span."""
    if (
        magic not in (b"NVAI", b"NVAO")
        or type(sequence) is not int
        or sequence < 0
        or type(offset) is not int
        or offset < 0
        or not pcm
        or len(pcm) % 2
    ):
        raise ValueError("Invalid PCM packet")
    return HEADER.pack(magic, sequence, offset, len(pcm) // 2) + pcm


class InputUpload:
    """Own contiguous, bounded input until explicit extent/hash-verified EOF."""

    def __init__(self, declared_samples: int, limits: SessionLimits):
        """Create the upload independently of the conversation runtime."""
        self.declared_samples = declared_samples
        self.limits = limits
        self.packet_count = 0
        self._pcm = bytearray()
        self._digest = hashlib.sha256()
        self._sealed = False

    def append(self, packet: bytes) -> None:
        """Accept all source samples once, in exact sequence and offset order."""
        if self._sealed:
            raise OfflineProtocolError("input_sealed", "Input has already been sealed")
        sequence, offset, pcm = decode_packet(packet, b"NVAI", self.limits.max_packet_samples)
        if sequence != self.packet_count or offset != len(self._pcm) // 2:
            raise OfflineProtocolError("input_order", "Input packet sequence and offset must be contiguous")
        if len(self._pcm) + len(pcm) > self.declared_samples * 2:
            raise OfflineProtocolError("input_extent", "Input exceeds its declared sample count")
        self._pcm.extend(pcm)
        self._digest.update(pcm)
        self.packet_count += 1

    def seal(self, message: dict[str, Any]) -> AudioSource:
        """Produce an immutable source only after an explicit, consistent EOF."""
        if self._sealed:
            raise OfflineProtocolError("input_sealed", "Input has already been sealed")
        if message.get("type") != "offline.input.complete":
            raise OfflineProtocolError("unexpected_control", "Expected explicit offline.input.complete")
        if (
            integer_field(message, "input_samples") != self.declared_samples
            or len(self._pcm) != self.declared_samples * 2
            or integer_field(message, "input_packets") != self.packet_count
        ):
            raise OfflineProtocolError("input_extent", "Input completion extent or packet count is inconsistent")
        if message.get("sha256") != self._digest.hexdigest():
            raise OfflineProtocolError("input_digest", "Input SHA-256 does not match the received PCM")
        source = AudioSource(bytes(self._pcm))
        self._pcm.clear()
        self._sealed = True
        return source


def redact_private(value: Any) -> Any:
    """Exclude common authentication values from wire metadata and fingerprints."""
    if isinstance(value, dict):
        return {
            key: "[redacted]"
            if str(key).lower() in _PRIVATE_KEYS
            or str(key).lower().endswith(("_api_key", "_token", "_password", "_secret"))
            else redact_private(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [redact_private(item) for item in value]
    return value


def configuration_fingerprint(configuration: dict[str, Any]) -> str:
    """Fingerprint public configuration while treating secret rotation as equivalent."""
    encoded = json.dumps(redact_private(configuration), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_result(result: SessionResult, source: AudioSource, limits: SessionLimits) -> None:
    """Reject overlapping, retroactive, malformed, or unbounded committed output."""
    if not isinstance(result, SessionResult):
        raise ValueError("Runtime must return a SessionResult")
    if type(result.rendered_samples) is not int or not source.sample_count <= result.rendered_samples:
        raise ValueError("Output must include the source horizon")
    if result.rendered_samples > limits.max_output_samples:
        raise ValueError("Output exceeds the configured duration limit")
    cursor = 0
    for span in result.spans:
        if type(span.start_sample) is not int or span.start_sample < cursor or not span.pcm or len(span.pcm) % 2:
            raise ValueError("Committed output spans must be nonempty, chronological mono PCM16")
        if span.end_sample > result.rendered_samples:
            raise ValueError("Committed output exceeds the rendered horizon")
        cursor = span.end_sample
