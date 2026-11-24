# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session audio recorder using Pipecat's AudioBufferProcessor.

Records each session to one stereo WAV file, with the user (ASR input) on the left channel and the bot
(TTS output) on the right channel.
Controlled via environment variables:
  - ENABLE_AUDIO_DUMP  (default: false)
  - AUDIO_DUMP_PATH    (default: <project_root>/audio_dumps)
"""

import os
import uuid
import wave
from pathlib import Path

from loguru import logger
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor

from nemotron_voice_agent.core.paths import REPO_ROOT


def _env_bool(key: str, default: str = "false") -> bool:
    return os.getenv(key, default).lower() == "true"


ENABLE_AUDIO_DUMP = _env_bool("ENABLE_AUDIO_DUMP")
_raw_dump_path = Path(os.getenv("AUDIO_DUMP_PATH", "audio_dumps"))
AUDIO_DUMP_PATH = _raw_dump_path if _raw_dump_path.is_absolute() else REPO_ROOT / _raw_dump_path


def _write_wav(filepath: Path, audio: bytes, sample_rate: int, num_channels: int) -> None:
    """Write raw PCM16 audio to a WAV file."""
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(num_channels)
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
    """Create an AudioBufferProcessor that saves each session to `session_{stream}.wav`.

    Returns None if `ENABLE_AUDIO_DUMP` is off. The file is written when the session ends.
    Caller must await recorder.start_recording() on client connect.
    """
    if not ENABLE_AUDIO_DUMP:
        return None

    _validate_dump_dir(AUDIO_DUMP_PATH)
    stream_id = uuid.uuid4().hex[:8]

    recorder = AudioBufferProcessor(num_channels=2)

    @recorder.event_handler("on_audio_data")
    async def on_audio_data(processor, audio: bytes, sample_rate: int, num_channels: int):
        if audio:
            _write_wav(AUDIO_DUMP_PATH / f"session_{stream_id}.wav", audio, sample_rate, num_channels)

    logger.info(f"Audio recorder enabled (per-session) — path={AUDIO_DUMP_PATH}, stream={stream_id}")
    return recorder
