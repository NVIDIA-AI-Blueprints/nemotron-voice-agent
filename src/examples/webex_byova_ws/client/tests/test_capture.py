"""Diagnostic prompt audio capture tests."""

from __future__ import annotations

import json
from pathlib import Path

from webex_byova_ws_adapter.audio import RAW_MULAW, WAV_PCM16, prompt_audio
from webex_byova_ws_adapter.capture import PromptAudioRecorder


def test_capture_is_disabled_without_a_configured_directory() -> None:
    assert PromptAudioRecorder.for_conversation("", "conv-1", WAV_PCM16) is None


def test_each_chunk_is_stored_raw_as_sent_and_in_the_manifest(tmp_path: Path) -> None:
    recorder = PromptAudioRecorder.for_conversation(str(tmp_path), "conv-1", WAV_PCM16)
    assert recorder is not None
    for pcm in (b"\x01\x02" * 320, b"\x03\x04" * 320):
        recorder.record(pcm, prompt_audio(pcm, WAV_PCM16))

    directory = tmp_path / "conv-1"
    assert (directory / "chunk-0001.pcm").read_bytes() == b"\x01\x02" * 320
    assert (directory / "chunk-0002.wav").read_bytes().startswith(b"RIFF")
    entries = [json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines()]
    assert [entry["chunk"] for entry in entries] == [1, 2]
    assert entries[0]["pcm_bytes"] == 640
    assert entries[0]["payload_bytes"] == 640 + 44


def test_capture_names_files_and_manifest_by_profile(tmp_path: Path) -> None:
    recorder = PromptAudioRecorder.for_conversation(str(tmp_path), "conv-2", RAW_MULAW)
    assert recorder is not None
    pcm = b"\x01\x02" * 320
    recorder.record(pcm, prompt_audio(pcm, RAW_MULAW))

    directory = tmp_path / "conv-2"
    assert (directory / "chunk-0001.ulaw").read_bytes() == prompt_audio(pcm, RAW_MULAW)
    entry = json.loads((directory / "manifest.jsonl").read_text().splitlines()[0])
    assert entry["profile"] == RAW_MULAW
    assert entry["payload_bytes"] == 320
