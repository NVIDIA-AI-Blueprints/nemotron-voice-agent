"""Telephony audio conversion and bounded output chunking."""

from __future__ import annotations

import audioop
import base64
import binascii
import io
import wave
from dataclasses import dataclass, field

PIPELINE_RATE = 16_000
WEBEX_RATE = 8_000
SAMPLE_WIDTH = 2
MIN_CHUNK = 640
MAX_CHUNK = 32 * 1024

# Cisco's documented media profile is raw mu-law with no container, which the
# connector renders once a dtmf_config accompanies INPUT_VOICE_DTMF. Without
# that config it discards raw audio, which is why the WAV form is kept
# selectable.
WAV_PCM16 = "wav-pcm16"
RAW_MULAW = "raw-mulaw"
PROMPT_AUDIO_SUFFIX = {WAV_PCM16: "wav", RAW_MULAW: "ulaw"}


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
    """Resample mono PCM16 to 8 kHz and release it as generated.

    Generated audio is forwarded on arrival rather than gathered into fixed
    chunks, so the only audio ever held back is a remainder too short for
    Cisco's minimum prompt size, which the reply's flush releases.
    """

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
        self._buffer.extend(downsampled)
        chunks: list[bytes] = []
        while len(self._buffer) >= MIN_CHUNK:
            size = min(len(self._buffer), MAX_CHUNK)
            chunks.append(bytes(self._buffer[:size]))
            del self._buffer[:size]
        return chunks

    def flush(self) -> list[bytes]:
        """Flush a short tail padded with silence."""
        if not self._buffer:
            return []
        self._buffer.extend(bytes(max(0, MIN_CHUNK - len(self._buffer))))
        chunk = bytes(self._buffer)
        self._buffer.clear()
        return [chunk]


def wav_container(pcm: bytes) -> bytes:
    """Wrap PCM16 8 kHz audio in a WAV container.

    Webex renders prompt audio only when it carries a WAV header, matching the
    canned prompts its own connector plays.
    """
    container = io.BytesIO()
    with wave.open(container, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(WEBEX_RATE)
        wav.writeframes(pcm)
    return container.getvalue()


def prompt_audio(pcm: bytes, profile: str) -> bytes:
    """Render PCM16 8 kHz audio into the wire form of the chosen profile."""
    if profile == RAW_MULAW:
        return audioop.lin2ulaw(pcm, SAMPLE_WIDTH)
    return wav_container(pcm)


def encode_audio(payload: bytes) -> str:
    """Encode rendered prompt audio for the AsyncAPI prompt field."""
    return base64.b64encode(payload).decode("ascii")
