"""Media conversion tests."""

from __future__ import annotations

import audioop
import base64
import io
import wave

import pytest

from webex_byova_ws_adapter.audio import (
    RAW_MULAW,
    WAV_PCM16,
    CallerAudioConverter,
    WebexAudioChunker,
    encode_audio,
    prompt_audio,
)


def test_caller_mulaw_is_statefully_resampled_to_pcm16() -> None:
    pcm8 = b"".join(int(index * 20 - 3000).to_bytes(2, "little", signed=True) for index in range(300))
    mulaw = audioop.lin2ulaw(pcm8, 2)
    converter = CallerAudioConverter()
    first = converter.convert(base64.b64encode(mulaw[:100]).decode(), "MULAW_FORMAT", 8000)
    second = converter.convert(base64.b64encode(mulaw[100:]).decode(), "MULAW_FORMAT", 8000)
    one_shot = CallerAudioConverter().convert(base64.b64encode(mulaw).decode(), "MULAW_FORMAT", 8000)
    assert first + second == one_shot
    assert len(one_shot) > len(pcm8)


def test_caller_rejects_nonqualified_media_and_invalid_base64() -> None:
    converter = CallerAudioConverter()
    with pytest.raises(ValueError, match="mu-law"):
        converter.convert("AAAA", "LINEAR16_FORMAT", 16000)
    with pytest.raises(ValueError, match="Base64"):
        converter.convert("***", "MULAW_FORMAT", 8000)


def test_output_chunks_obey_cisco_bounds_and_flush_tail() -> None:
    chunker = WebexAudioChunker()
    chunks = chunker.feed(b"\x00\x00" * 80_000, 16_000, 1)
    chunks.extend(chunker.flush())
    assert chunks
    assert all(640 <= len(chunk) <= 32 * 1024 for chunk in chunks)


def test_generated_audio_is_forwarded_without_fixed_size_buffering() -> None:
    """A frame long enough to send must not wait for more audio to accumulate."""
    chunker = WebexAudioChunker()
    assert chunker.feed(b"\x00\x00" * 500, 8_000, 1) == [b"\x00\x00" * 500]


def test_short_output_is_padded_to_minimum_raw_chunk() -> None:
    chunker = WebexAudioChunker()
    assert chunker.feed(b"\x00\x00" * 100, 16_000, 1) == []
    assert len(chunker.flush()[0]) == 640


def test_output_uses_declared_pipeline_sample_rate() -> None:
    one_second_pcm = b"\x00\x00" * 22_050
    chunks = WebexAudioChunker().feed(one_second_pcm, 22_050, 1)
    assert sum(map(len, chunks)) == 16_000


def test_prompt_audio_is_a_playable_pcm16_8khz_wav() -> None:
    """Webex renders prompt audio only when it arrives in a WAV container."""
    pcm = b"\x01\x02" * 4_000
    payload = prompt_audio(pcm, WAV_PCM16)
    assert base64.b64decode(encode_audio(payload)) == payload
    with wave.open(io.BytesIO(payload), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 8_000)
        assert wav.readframes(wav.getnframes()) == pcm


def test_documented_profile_renders_raw_mulaw_without_a_container() -> None:
    pcm = b"\x01\x02" * 4_000
    payload = prompt_audio(pcm, RAW_MULAW)
    assert not payload.startswith(b"RIFF")
    assert payload == audioop.lin2ulaw(pcm, 2)


def test_output_rejects_missing_rate_and_non_mono_audio() -> None:
    chunker = WebexAudioChunker()
    with pytest.raises(ValueError, match="sample rate"):
        chunker.feed(b"\x00\x00", 0, 1)
    with pytest.raises(ValueError, match="mono"):
        chunker.feed(b"\x00\x00", 16_000, 2)
