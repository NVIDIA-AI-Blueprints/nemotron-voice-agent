# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

import wave
from unittest.mock import patch

from nemotron_voice_agent.core import session_audio
from nemotron_voice_agent.core.session_audio import Recording, SessionAudio, write_wav

FRAME = b"\x01\x01" * 100


def clock(*times):
    return patch.object(session_audio.time, "monotonic", side_effect=list(times))


def test_pauses_longer_than_the_gap_become_silence_on_the_wall_clock():
    audio = SessionAudio()
    with clock(0.0, 0.01, 0.02, 1.02, 1.03):
        audio.start()
        audio.add(FRAME, 10000, 1)
        audio.add(FRAME, 10000, 1)
        audio.add(FRAME, 10000, 1)
        recording = audio.finish()
    assert recording == Recording(FRAME * 2 + bytes(19800) + FRAME, 10000, 1)


def test_audio_faster_than_real_time_keeps_the_next_frame_on_the_wall_clock():
    audio = SessionAudio()
    with clock(0.0, 0.01, 0.01, 0.01, 1.04, 1.05):
        audio.start()
        for _ in range(3):
            audio.add(FRAME, 10000, 1)
        audio.add(FRAME, 10000, 1)
        recording = audio.finish()
    assert recording.pcm == FRAME * 3 + bytes(20000) + FRAME


def test_frames_in_another_format_are_skipped_and_the_audio_is_returned_once():
    audio = SessionAudio()
    with clock(0.0, 0.01, 0.02):
        audio.start()
        audio.add(FRAME, 10000, 1)
        audio.add(FRAME, 24000, 1)
        assert audio.finish() == Recording(FRAME, 10000, 1)
    assert audio.finish() is None


def test_nothing_is_returned_before_start_or_without_audio():
    audio = SessionAudio()
    audio.add(FRAME, 10000, 1)
    assert audio.finish() is None
    audio.start()
    assert audio.finish() is None


def test_a_recording_is_written_as_a_wav_file(tmp_path):
    assert write_wav(tmp_path / "asr.wav", Recording(FRAME, 10000, 2)) == tmp_path / "asr.wav"
    with wave.open(str(tmp_path / "asr.wav"), "rb") as wf:
        assert (wf.getframerate(), wf.getnchannels(), wf.readframes(wf.getnframes())) == (10000, 2, FRAME)
