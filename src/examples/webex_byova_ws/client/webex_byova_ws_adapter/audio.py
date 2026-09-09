"""Raw telephony audio conversion and bounded output chunking."""

from __future__ import annotations

import audioop
import base64
import binascii
from dataclasses import dataclass, field

PIPELINE_RATE = 16_000
WEBEX_RATE = 8_000
MIN_CHUNK = 320
TARGET_CHUNK = 8_000
MAX_CHUNK = 32 * 1024


@dataclass(slots=True)
class CallerAudioConverter:
    """Statefully convert raw 8 kHz mu-law caller audio to PCM16 16 kHz."""

    _state: object | None = None

    def convert(self, encoded: str, encoding: str, sample_rate: int) -> bytes:
        """Decode Base64 and resample one contiguous caller chunk."""
        if encoding != "MULAW_FORMAT" or sample_rate != WEBEX_RATE:
            raise ValueError("only raw G.711 mu-law 8000 Hz mono audio is supported")
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("caller_audio_b64 must be a nonempty string")
        try:
            mulaw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("caller_audio_b64 is invalid Base64") from exc
        pcm = audioop.ulaw2lin(mulaw, 2)
        pcm, self._state = audioop.ratecv(pcm, 2, 1, WEBEX_RATE, PIPELINE_RATE, self._state)
        return pcm


@dataclass(slots=True)
class WebexAudioChunker:
    """Convert mono PCM16 at its declared rate to bounded raw mu-law chunks."""

    _resample_state: object | None = None
    _source_rate: int | None = None
    _buffer: bytearray = field(default_factory=bytearray)

    def feed(self, pcm: bytes, sample_rate: int, num_channels: int) -> list[bytes]:
        """Convert and return every currently complete output chunk."""
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("PCM16 sample rate must be a positive integer")
        if num_channels != 1:
            raise ValueError("only mono PCM16 output is supported")
        if len(pcm) % 2:
            raise ValueError("PCM16 audio must contain complete samples")
        if self._source_rate != sample_rate:
            self._resample_state = None
            self._source_rate = sample_rate
        downsampled, self._resample_state = audioop.ratecv(
            pcm,
            2,
            1,
            sample_rate,
            WEBEX_RATE,
            self._resample_state,
        )
        self._buffer.extend(audioop.lin2ulaw(downsampled, 2))
        chunks: list[bytes] = []
        while len(self._buffer) >= TARGET_CHUNK:
            chunks.append(bytes(self._buffer[:TARGET_CHUNK]))
            del self._buffer[:TARGET_CHUNK]
        return chunks

    def flush(self) -> list[bytes]:
        """Flush a short tail padded with mu-law silence."""
        if not self._buffer:
            return []
        self._buffer.extend(b"\xff" * (MIN_CHUNK - len(self._buffer)))
        chunk = bytes(self._buffer)
        self._buffer.clear()
        return [chunk]


def encode_audio(raw: bytes) -> str:
    """Encode raw mu-law bytes for the AsyncAPI prompt field."""
    return base64.b64encode(raw).decode("ascii")
