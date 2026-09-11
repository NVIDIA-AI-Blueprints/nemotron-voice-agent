# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session audio recorder using Pipecat's AudioBufferProcessor.

Captures user (ASR input) and bot (TTS output) audio to separate WAV files.
Controlled via environment variables:
  - ENABLE_ASR_AUDIO_DUMP  (default: false)
  - ENABLE_TTS_AUDIO_DUMP  (default: false)
  - AUDIO_DUMP_PATH        (default: <project_root>/audio_dumps)
"""

import os
import uuid
import wave
from pathlib import Path

from loguru import logger
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor

from utils import PROJECT_ROOT


def _env_bool(key: str, default: str = "false") -> bool:
    return os.getenv(key, default).lower() == "true"


ENABLE_ASR_DUMP = _env_bool("ENABLE_ASR_AUDIO_DUMP")
ENABLE_TTS_DUMP = _env_bool("ENABLE_TTS_AUDIO_DUMP")
_raw_dump_path = Path(os.getenv("AUDIO_DUMP_PATH", "audio_dumps"))
AUDIO_DUMP_PATH = _raw_dump_path if _raw_dump_path.is_absolute() else PROJECT_ROOT / _raw_dump_path


def _write_wav(filepath: Path, audio: bytes, sample_rate: int) -> None:
    """Write raw PCM16 mono audio to a WAV file."""
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)
    logger.info(f"Audio saved: {filepath} ({len(audio)} bytes, {sample_rate}Hz)")


def _validate_dump_dir(dump_dir: Path) -> None:
    """Create the dump directory and verify write permissions."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    test_file = dump_dir / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
    except PermissionError:
        raise PermissionError(
            f"Cannot write to audio dump directory: {dump_dir}. Fix: sudo chown -R $(id -u):$(id -g) {dump_dir}"
        ) from None


def create_audio_recorder() -> AudioBufferProcessor | None:
    """Create an AudioBufferProcessor that saves per-turn audio clips to WAV files.

    Returns None if both ASR and TTS dumps are disabled.
    Each session gets a unique stream ID, and each turn gets an incrementing index:
      asr_{stream}_{turn}.wav, tts_{stream}_{turn}.wav
    Caller must await recorder.start_recording() on client connect.
    """
    if not ENABLE_ASR_DUMP and not ENABLE_TTS_DUMP:
        return None

    _validate_dump_dir(AUDIO_DUMP_PATH)
    stream_id = uuid.uuid4().hex[:8]
    turn_counter = {"asr": 0, "tts": 0}

    recorder = AudioBufferProcessor(num_channels=1, enable_turn_audio=True)

    @recorder.event_handler("on_user_turn_audio_data")
    async def on_user_turn(processor, audio: bytes, sample_rate: int, num_channels: int):
        if not ENABLE_ASR_DUMP or not audio:
            return
        idx = turn_counter["asr"]
        turn_counter["asr"] = idx + 1
        _write_wav(AUDIO_DUMP_PATH / f"asr_{stream_id}_{idx:03d}.wav", audio, sample_rate)

    @recorder.event_handler("on_bot_turn_audio_data")
    async def on_bot_turn(processor, audio: bytes, sample_rate: int, num_channels: int):
        if not ENABLE_TTS_DUMP or not audio:
            return
        idx = turn_counter["tts"]
        turn_counter["tts"] = idx + 1
        _write_wav(AUDIO_DUMP_PATH / f"tts_{stream_id}_{idx:03d}.wav", audio, sample_rate)

    logger.info(
        f"Audio recorder enabled (per-turn) — ASR={ENABLE_ASR_DUMP}, TTS={ENABLE_TTS_DUMP}, "
        f"path={AUDIO_DUMP_PATH}, stream={stream_id}"
    )
    return recorder
