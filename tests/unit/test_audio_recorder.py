# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102

import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch

from pipecat.processors.audio.audio_buffer_processor import TurnAudioData

from examples.shared import audio_recorder


class AudioRecorderTests(unittest.IsolatedAsyncioTestCase):
    async def test_turn_audio_handlers_preserve_filenames_and_counters(self) -> None:
        writes: list[tuple[Path, bytes, int]] = []
        with (
            patch.object(audio_recorder, "ENABLE_ASR_DUMP", True),
            patch.object(audio_recorder, "ENABLE_TTS_DUMP", True),
            patch.object(audio_recorder, "AUDIO_DUMP_PATH", Path("/tmp/audio")),
            patch.object(audio_recorder, "_validate_dump_dir"),
            patch.object(audio_recorder.uuid, "uuid4", return_value=type("_UUID", (), {"hex": "12345678abcdef"})()),
            patch.object(audio_recorder, "_write_wav", side_effect=lambda *args: writes.append(args)),
        ):
            recorder = audio_recorder.create_audio_recorder()
            self.assertIsNotNone(recorder)
            user_turn = TurnAudioData(turn_number=1, audio=b"user", sample_rate=16000, num_channels=1)
            bot_turn = TurnAudioData(turn_number=1, audio=b"bot", sample_rate=24000, num_channels=1)

            await recorder._call_event_handler("on_user_turn_audio", user_turn)
            await recorder._call_event_handler("on_user_turn_audio", user_turn)
            await recorder._call_event_handler("on_bot_turn_audio", bot_turn)
            await asyncio.sleep(0)

        self.assertEqual(
            writes,
            [
                (Path("/tmp/audio/asr_12345678_000.wav"), b"user", 16000),
                (Path("/tmp/audio/asr_12345678_001.wav"), b"user", 16000),
                (Path("/tmp/audio/tts_12345678_000.wav"), b"bot", 24000),
            ],
        )
