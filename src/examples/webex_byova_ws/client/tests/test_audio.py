"""Media conversion tests."""

from __future__ import annotations

import audioop
import base64

import pytest

from webex_byova_ws_adapter.audio import CallerAudioConverter, WebexAudioChunker


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
    chunks = chunker.feed(b"\x00\x00" * 80_000)
    chunks.extend(chunker.flush())
    assert chunks
    assert all(320 <= len(chunk) <= 32 * 1024 for chunk in chunks)
    assert all(not chunk.startswith(b"RIFF") for chunk in chunks)


def test_short_output_is_padded_to_minimum_raw_chunk() -> None:
    chunker = WebexAudioChunker()
    assert chunker.feed(b"\x00\x00" * 100) == []
    assert len(chunker.flush()[0]) == 320
