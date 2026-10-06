# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Swappable audio I/O for the Realtime chat client.

``AudioIO`` is the whole contract: the input frames to stream (already in the
session's client format, 20 ms each, including silence), and sinks for received
audio. Adding an I/O means adding one class.

* :class:`MicSpeakerIO`: microphone and speaker through ``sounddevice``
  (optional extra ``prototypes-voice-mic``; needs the PortAudio system library).
* :class:`WavFileIO`: scripted turns from WAV files, separated by silence; the
  agent's audio is written to a WAV file.
* :class:`TextIO`: no audio at all (text in, text out).
"""

from __future__ import annotations

import asyncio
import time
import wave
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from examples.frontend_backend_verdict.voice.audio.formats import AudioFormat
from examples.frontend_backend_verdict.voice.audio.resample import StreamResampler

FRAME_MS = 20
PORTAUDIO_HINT = (
    "microphone I/O needs PortAudio and sounddevice: `sudo apt install libportaudio2` (Debian/Ubuntu) or "
    "`brew install portaudio` (macOS), then `uv sync --dev --extra prototypes-voice-mic`"
)


class AudioIO(Protocol):
    """Audio source and sink for one client session."""

    @property
    def uses_audio(self) -> bool:
        """Whether this I/O streams audio at all."""

    def frames(self) -> AsyncIterator[bytes]:
        """Input frames in the client format, paced in real time; ends when the input is exhausted."""

    def play(self, audio: bytes) -> None:
        """Queue received agent audio (client format)."""

    def stop_playback(self) -> None:
        """Drop queued agent audio (the server detected user speech)."""

    def response_done(self, *, had_calls: bool) -> None:
        """A response finished (scripted inputs use this to pace turns)."""

    async def close(self) -> None:
        """Release devices and flush outputs."""


def read_wav(path: str | Path, target: AudioFormat) -> bytes:
    """Read a 16-bit WAV (mono or stereo, any rate) and return it encoded in ``target``."""
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV files are supported")
        channels, rate = handle.getnchannels(), handle.getframerate()
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype("<i2")
    pcm = StreamResampler(rate, target.rate).process(samples.tobytes(), last=True)
    return target.encode(pcm)


def write_wav(path: str | Path, pcm: bytes, rate: int) -> None:
    """Write mono PCM16."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)


class _Pacer:
    """Real-time pacing for 20 ms frames without drift."""

    def __init__(self) -> None:
        self._start = time.monotonic()
        self._sent = 0

    async def tick(self) -> None:
        self._sent += 1
        delay = self._start + self._sent * FRAME_MS / 1000.0 - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)


class WavFileIO:
    """Scripted turns: each WAV, then silence until the agent has answered and its audio has 'played'."""

    def __init__(
        self, inputs: Sequence[str], fmt: AudioFormat, *, out_path: str = "", tail_silence_ms: int = 2000
    ) -> None:
        """Pre-encode every input turn in ``fmt``."""
        self._fmt = fmt
        self._turns = [read_wav(path, fmt) for path in inputs]
        self._out_path = out_path
        self._tail_ms = tail_silence_ms
        self._received = bytearray()
        self._answered = asyncio.Event()
        self._pending_audio_ms = 0.0
        self.uses_audio = True

    def _frame_bytes(self) -> int:
        return self._fmt.rate * FRAME_MS // 1000 * self._fmt.bytes_per_sample

    def _silence(self) -> bytes:
        return self._fmt.encode(b"\x00\x00" * (self._fmt.rate * FRAME_MS // 1000))

    async def frames(self) -> AsyncIterator[bytes]:
        """Each turn's audio, then continuous silence until the agent answered, then the next turn."""
        size = self._frame_bytes()
        pacer = _Pacer()
        for turn in self._turns:
            self._answered.clear()
            for offset in range(0, len(turn), size):
                chunk = turn[offset : offset + size]
                yield chunk + self._silence()[len(chunk) :]
                await pacer.tick()
            while not self._answered.is_set():
                yield self._silence()
                await pacer.tick()
            # Let the agent's audio "play" before the next turn, like a polite caller.
            for _ in range(int(self._pending_audio_ms // FRAME_MS)):
                yield self._silence()
                await pacer.tick()
            self._pending_audio_ms = 0.0
        for _ in range(self._tail_ms // FRAME_MS):
            yield self._silence()
            await pacer.tick()

    def play(self, audio: bytes) -> None:
        """Collect agent audio for the output WAV."""
        self._received.extend(self._fmt.decode(audio))
        self._pending_audio_ms += self._fmt.samples_in(audio) * 1000.0 / self._fmt.rate

    def stop_playback(self) -> None:
        """Nothing is playing in real time."""
        self._pending_audio_ms = 0.0

    def response_done(self, *, had_calls: bool) -> None:
        """A spoken (non-tool) response ends the agent's turn."""
        if not had_calls:
            self._answered.set()

    async def close(self) -> None:
        """Write the agent's audio."""
        if self._out_path:
            write_wav(self._out_path, bytes(self._received), self._fmt.rate)


class MicSpeakerIO:
    """Microphone in, speaker out (24 kHz PCM16) through ``sounddevice``."""

    def __init__(self, fmt: AudioFormat) -> None:
        """Open the devices lazily; a missing PortAudio raises a one-line hint."""
        try:
            import sounddevice
        except (ImportError, OSError) as exc:
            raise SystemExit(f"{PORTAUDIO_HINT} ({exc})") from None
        self._sd = sounddevice
        self._fmt = fmt
        self._frames: asyncio.Queue[bytes] = asyncio.Queue()
        self._playback = bytearray()
        self._input: Any = None
        self._output: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.uses_audio = True

    def _mic_callback(self, data: Any, frames: int, time_info: Any, status: Any) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._frames.put_nowait, self._fmt.encode(bytes(data)))

    def _speaker_callback(self, data: Any, frames: int, time_info: Any, status: Any) -> None:
        size = len(data)
        chunk = bytes(self._playback[:size])
        del self._playback[:size]
        data[:] = chunk + b"\x00" * (size - len(chunk))

    async def frames(self) -> AsyncIterator[bytes]:
        """Microphone frames until cancelled."""
        self._loop = asyncio.get_running_loop()
        blocksize = self._fmt.rate * FRAME_MS // 1000
        self._input = self._sd.RawInputStream(
            samplerate=self._fmt.rate, channels=1, dtype="int16", blocksize=blocksize, callback=self._mic_callback
        )
        self._output = self._sd.RawOutputStream(
            samplerate=self._fmt.rate, channels=1, dtype="int16", blocksize=blocksize, callback=self._speaker_callback
        )
        self._input.start()
        self._output.start()
        while True:
            yield await self._frames.get()

    def play(self, audio: bytes) -> None:
        """Queue PCM16 for the speaker."""
        self._playback.extend(self._fmt.decode(audio))

    def stop_playback(self) -> None:
        """Barge-in: drop queued speaker audio."""
        self._playback.clear()

    def response_done(self, *, had_calls: bool) -> None:
        """Nothing to pace."""

    async def close(self) -> None:
        """Stop the devices."""
        for stream in (self._input, self._output):
            if stream is not None:
                stream.stop()
                stream.close()


class TextIO:
    """No audio: typed input, printed output."""

    uses_audio = False

    async def frames(self) -> AsyncIterator[bytes]:
        """No frames."""
        return
        yield b""  # pragma: no cover - makes this an async generator

    def play(self, audio: bytes) -> None:
        """Ignore audio."""

    def stop_playback(self) -> None:
        """Nothing playing."""

    def response_done(self, *, had_calls: bool) -> None:
        """Nothing to pace."""

    async def close(self) -> None:
        """Nothing to release."""
