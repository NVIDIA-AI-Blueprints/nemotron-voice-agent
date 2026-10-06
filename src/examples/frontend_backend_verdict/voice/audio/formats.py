# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Realtime GA audio formats and conversion to and from PCM16.

GA format objects only: ``{"type": "audio/pcm", "rate": 24000}``,
``{"type": "audio/pcmu"}`` and ``{"type": "audio/pcma"}``. Beta format strings
(``pcm16``, ``g711_ulaw``, ``g711_alaw``) are rejected by the session view, not
here (see the plan, section 7.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from examples.frontend_backend_verdict.voice.audio import g711

PCM = "audio/pcm"
PCMU = "audio/pcmu"
PCMA = "audio/pcma"

SUPPORTED_PCM_RATES = (8000, 16000, 24000, 48000)
G711_RATE = 8000


@dataclass(frozen=True, slots=True)
class AudioFormat:
    """One client-side audio format."""

    type: str
    rate: int

    @property
    def bytes_per_sample(self) -> int:
        """Encoded bytes per sample (2 for PCM16, 1 for G.711)."""
        return 2 if self.type == PCM else 1

    def samples_in(self, payload: bytes) -> int:
        """Number of samples in an encoded payload."""
        return len(payload) // self.bytes_per_sample

    def decode(self, payload: bytes) -> bytes:
        """Encoded client audio -> PCM16 at :attr:`rate`."""
        if self.type == PCMU:
            return g711.ulaw_decode(payload)
        if self.type == PCMA:
            return g711.alaw_decode(payload)
        return payload[: len(payload) - len(payload) % 2]

    def encode(self, pcm: bytes) -> bytes:
        """PCM16 at :attr:`rate` -> encoded client audio."""
        if self.type == PCMU:
            return g711.ulaw_encode(pcm)
        if self.type == PCMA:
            return g711.alaw_encode(pcm)
        return pcm

    def to_wire(self) -> dict[str, Any]:
        """GA format object for ``session.created`` / ``session.updated``."""
        if self.type == PCM:
            return {"type": PCM, "rate": self.rate}
        return {"type": self.type}


def parse_format(value: Any, *, param: str) -> AudioFormat:
    """Parse a GA format object, raising ``ValueError`` naming ``param``."""
    if not isinstance(value, dict):
        raise ValueError(f"{param} must be a GA format object such as {{'type': 'audio/pcm', 'rate': 24000}}")
    kind = value.get("type")
    if kind == PCM:
        rate = value.get("rate", 24000)
        if isinstance(rate, bool) or not isinstance(rate, int) or rate not in SUPPORTED_PCM_RATES:
            raise ValueError(f"{param}.rate must be one of {SUPPORTED_PCM_RATES}, got {rate!r}")
        return AudioFormat(PCM, rate)
    if kind in (PCMU, PCMA):
        rate = value.get("rate", G711_RATE)
        if rate != G711_RATE:
            raise ValueError(f"{param}.rate must be {G711_RATE} for {kind}, got {rate!r}")
        return AudioFormat(kind, G711_RATE)
    raise ValueError(f"{param}.type must be one of {PCM!r}, {PCMU!r}, {PCMA!r}; got {kind!r}")
