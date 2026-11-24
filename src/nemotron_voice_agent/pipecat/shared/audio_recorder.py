# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session audio dumps for debugging.

Each session writes three WAV files when it ends:
  - session_{stream}.wav: stereo playback from Pipecat's AudioBufferProcessor, user left and bot right
  - asr_{stream}_session.wav: the audio that ASR received, tapped before STT
  - tts_{stream}_session.wav: the audio that TTS produced, tapped before the output transport
The ASR and TTS files share one wall-clock timeline, with silence where a stream paused.
Controlled via environment variables:
  - ENABLE_AUDIO_DUMP  (default: false)
  - AUDIO_DUMP_PATH    (default: <project_root>/audio_dumps)
"""

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from pipecat.frames.frames import CancelFrame, EndFrame, Frame, InputAudioRawFrame, OutputAudioRawFrame
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from nemotron_voice_agent.core.paths import REPO_ROOT
from nemotron_voice_agent.core.session_audio import Recording, SessionAudio, write_wav


def _env_bool(key: str, default: str = "false") -> bool:
    return os.getenv(key, default).lower() == "true"


ENABLE_AUDIO_DUMP = _env_bool("ENABLE_AUDIO_DUMP")
_raw_dump_path = Path(os.getenv("AUDIO_DUMP_PATH", "audio_dumps"))
AUDIO_DUMP_PATH = _raw_dump_path if _raw_dump_path.is_absolute() else REPO_ROOT / _raw_dump_path


def _write_wav(filepath: Path, audio: bytes, sample_rate: int, num_channels: int) -> None:
    """Write raw PCM16 audio to a WAV file."""
    write_wav(filepath, Recording(audio, sample_rate, num_channels))
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


class SessionAudioDump(FrameProcessor):
    """Pass every frame through, and write the session's frames of one audio type to a WAV file when it ends."""

    def __init__(self, path: Path, frame_type: type[Frame], **kwargs):
        """Record ``frame_type`` frames to ``path`` once :meth:`start_recording` runs."""
        super().__init__(**kwargs)
        self._path = path
        self._frame_type = frame_type
        self._audio = SessionAudio()

    async def start_recording(self) -> None:
        """Start the session's timeline."""
        self._audio.start()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Record this stream's audio, and write the file at the end of the session."""
        await super().process_frame(frame, direction)
        if isinstance(frame, self._frame_type):
            self._audio.add(frame.audio, frame.sample_rate, frame.num_channels)
        elif isinstance(frame, (EndFrame, CancelFrame)) and (recording := self._audio.finish()):
            try:
                await asyncio.to_thread(_write_wav, self._path, *recording)
            except OSError as exc:
                logger.warning(f"Cannot save the session audio to {self._path}: {exc}")
        await self.push_frame(frame, direction)


@dataclass(frozen=True)
class AudioDump:
    """The session's dump processors, by where they go: after the input, before the output, and after the output."""

    inputs: list[FrameProcessor] = field(default_factory=list)
    outputs: list[FrameProcessor] = field(default_factory=list)
    after: list[FrameProcessor] = field(default_factory=list)

    async def start_recording(self) -> None:
        """Start every dump at once, so that the files share one timeline."""
        for processor in (*self.inputs, *self.outputs, *self.after):
            await processor.start_recording()


def create_audio_recorder() -> AudioDump:
    """Create the session's audio dumps, which are empty when `ENABLE_AUDIO_DUMP` is off.

    Place ``inputs`` right after the input transport, ``outputs`` right before the output transport, and ``after``
    after it. Call :meth:`AudioDump.start_recording` on client connect. The files are written when the session ends.
    """
    if not ENABLE_AUDIO_DUMP:
        return AudioDump()

    _validate_dump_dir(AUDIO_DUMP_PATH)
    stream_id = uuid.uuid4().hex[:8]

    recorder = AudioBufferProcessor(num_channels=2)

    @recorder.event_handler("on_audio_data")
    async def on_audio_data(processor, audio: bytes, sample_rate: int, num_channels: int):
        if audio:
            await asyncio.to_thread(
                _write_wav, AUDIO_DUMP_PATH / f"session_{stream_id}.wav", audio, sample_rate, num_channels
            )

    logger.info(f"Audio recorder enabled (per-session) — path={AUDIO_DUMP_PATH}, stream={stream_id}")
    return AudioDump(
        inputs=[SessionAudioDump(AUDIO_DUMP_PATH / f"asr_{stream_id}_session.wav", InputAudioRawFrame)],
        outputs=[SessionAudioDump(AUDIO_DUMP_PATH / f"tts_{stream_id}_session.wav", OutputAudioRawFrame)],
        after=[recorder],
    )
