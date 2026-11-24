# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102

import asyncio
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pipecat.frames.frames import CancelFrame, InputAudioRawFrame, OutputAudioRawFrame
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor

from nemotron_voice_agent.pipecat.shared import audio_recorder
from tests.unit.pipecat.frontend_helpers import Probe, run_frames


class AudioRecorderTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_dump_is_empty_when_the_flag_is_false(self) -> None:
        with patch.object(audio_recorder, "ENABLE_AUDIO_DUMP", False):
            dump = audio_recorder.create_audio_recorder()
        self.assertEqual((dump.inputs, dump.outputs, dump.after), ([], [], []))
        await dump.start_recording()

    async def test_the_dump_records_asr_input_tts_output_and_the_stereo_session(self) -> None:
        with (
            TemporaryDirectory() as tmp,
            patch.object(audio_recorder, "ENABLE_AUDIO_DUMP", True),
            patch.object(audio_recorder, "AUDIO_DUMP_PATH", Path(tmp)),
            patch.object(audio_recorder.uuid, "uuid4", return_value=type("_UUID", (), {"hex": "12345678abcdef"})()),
        ):
            dump = audio_recorder.create_audio_recorder()
            self.assertEqual(dump.inputs[0]._path, Path(tmp) / "asr_12345678_session.wav")
            self.assertEqual(dump.outputs[0]._path, Path(tmp) / "tts_12345678_session.wav")
            self.assertIsInstance(dump.after[0], AudioBufferProcessor)
            self.assertEqual(dump.after[0]._num_channels, 2)

            await dump.after[0]._call_event_handler("on_audio_data", b"\x01\x00\x02\x00", 16000, 2)
            await asyncio.sleep(0.1)

            with wave.open(str(Path(tmp) / "session_12345678.wav"), "rb") as wf:
                self.assertEqual((wf.getnchannels(), wf.getframerate()), (2, 16000))
                self.assertEqual(wf.readframes(wf.getnframes()), b"\x01\x00\x02\x00")

    async def test_a_session_dump_passes_every_frame_and_writes_only_its_audio_type(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "asr.wav"
            dump, probe = audio_recorder.SessionAudioDump(path, InputAudioRawFrame), Probe()
            heard = InputAudioRawFrame(audio=b"\x01\x01" * 160, sample_rate=16000, num_channels=1)
            spoken = OutputAudioRawFrame(audio=b"\x02\x02" * 160, sample_rate=24000, num_channels=1)
            await dump.start_recording()
            await run_frames([dump, probe], [heard, spoken, heard], gap=0.0)

            self.assertEqual(len(probe.of(InputAudioRawFrame)), 2)
            self.assertEqual(len(probe.of(OutputAudioRawFrame)), 1)
            with wave.open(str(path), "rb") as wf:
                self.assertEqual((wf.getnchannels(), wf.getframerate()), (1, 16000))
                audio = wf.readframes(wf.getnframes())
            self.assertEqual(audio.strip(b"\x00"), heard.audio * 2)

    async def test_a_failed_write_still_passes_the_end_of_the_session_on(self) -> None:
        dump, probe = audio_recorder.SessionAudioDump(Path("/nonexistent/asr.wav"), InputAudioRawFrame), Probe()
        heard = InputAudioRawFrame(audio=b"\x01\x01" * 160, sample_rate=16000, num_channels=1)
        await dump.start_recording()
        with patch.object(audio_recorder, "_write_wav", side_effect=OSError("read-only")) as write:
            await run_frames([dump, probe], [heard], gap=0.0)
        write.assert_called_once()
        self.assertEqual(len(probe.of(CancelFrame)), 1)
