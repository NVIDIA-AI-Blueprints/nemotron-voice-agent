# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102

import asyncio
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from nemotron_voice_agent.pipecat.shared import audio_recorder


class AudioRecorderTests(unittest.IsolatedAsyncioTestCase):
    def test_recorder_is_off_when_the_flag_is_false(self) -> None:
        with patch.object(audio_recorder, "ENABLE_AUDIO_DUMP", False):
            self.assertIsNone(audio_recorder.create_audio_recorder())

    async def test_session_audio_is_saved_as_one_stereo_wav(self) -> None:
        with (
            TemporaryDirectory() as tmp,
            patch.object(audio_recorder, "ENABLE_AUDIO_DUMP", True),
            patch.object(audio_recorder, "AUDIO_DUMP_PATH", Path(tmp)),
            patch.object(audio_recorder.uuid, "uuid4", return_value=type("_UUID", (), {"hex": "12345678abcdef"})()),
        ):
            recorder = audio_recorder.create_audio_recorder()
            self.assertIsNotNone(recorder)
            self.assertEqual(recorder._num_channels, 2)

            await recorder._call_event_handler("on_audio_data", b"\x01\x00\x02\x00", 16000, 2)
            await asyncio.sleep(0)

            with wave.open(str(Path(tmp) / "session_12345678.wav"), "rb") as wf:
                self.assertEqual((wf.getnchannels(), wf.getframerate()), (2, 16000))
                self.assertEqual(wf.readframes(wf.getnframes()), b"\x01\x00\x02\x00")
