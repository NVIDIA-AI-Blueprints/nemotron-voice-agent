# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""A primary WebSocket's serializer: caller audio in and reply audio out in each wire format, and every other frame."""

import asyncio
import unittest

import numpy as np
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame

from examples.frontend_backend_live.adapter.serializer import CODECS, LiveFrameSerializer
from gateways.live.protocol import AudioFormat
from gateways.live.session.audio_input import SAMPLE_BYTES
from tests.unit.frontend_helpers import FakeSocket, RecordingPort

AUDIO_IN_RATE = 16000
FORMATS = {
    "pcm 16 kHz": AudioFormat(type="audio/pcm", rate=16000),
    "pcm 24 kHz": AudioFormat(type="audio/pcm", rate=24000),
    "mu-law 8 kHz": AudioFormat(type="audio/pcmu"),
    "A-law 8 kHz": AudioFormat(type="audio/pcma"),
}


def serializer(audio_format: AudioFormat, socket=None) -> tuple[LiveFrameSerializer, FakeSocket, RecordingPort]:
    socket = socket or FakeSocket(audio_format)
    port = RecordingPort(socket=socket)
    return LiveFrameSerializer(socket, port, AUDIO_IN_RATE), socket, port


class SerializerTests(unittest.IsolatedAsyncioTestCase):
    async def heard(self, audio_format, chunks, pause_at=None):
        """Return the samples the frontend has heard after each of ``chunks`` 100 ms appends, pausing before one."""
        serialize, socket, _ = serializer(audio_format)
        samples = audio_format.rate // 10
        socket.audio["append"] = (bytes(range(256)) * samples)[: samples * SAMPLE_BYTES[audio_format.type]]
        heard, total = [], 0
        for count in range(chunks):
            if count == pause_at:
                await asyncio.sleep(0.25)  # longer than Pipecat's default for clearing a resampler
            frame = await serialize.deserialize("append")
            if frame is not None:
                self.assertIsInstance(frame, InputAudioRawFrame)
                self.assertEqual((frame.sample_rate, frame.num_channels), (AUDIO_IN_RATE, 1))
                total += len(frame.audio) // 2
            heard.append(total)
        return heard

    async def test_caller_audio_becomes_frames_at_the_frontends_rate(self):
        # The resampler holds a little audio back (none at 16 kHz), and a pause drops none of it.
        for name, audio_format in FORMATS.items():
            with self.subTest(name):
                steady, paused = await self.heard(audio_format, 30), await self.heard(audio_format, 30, pause_at=10)
                sent = [AUDIO_IN_RATE // 10 * count for count in range(1, 31)]
                held = [want - got for want, got in zip(sent, steady, strict=True)]
                self.assertEqual(paused, steady)
                self.assertTrue(all(0 <= samples <= AUDIO_IN_RATE * 120 // 1000 for samples in held), held)
                self.assertEqual(held[-1] == 0, audio_format.rate == AUDIO_IN_RATE)

    async def test_every_other_frame_goes_to_the_socket_as_no_frame(self):
        serialize, socket, port = serializer(FORMATS["pcm 24 kHz"])
        sent = ['{"type": "session.input_audio.mute"}', "not json", b"\x00\x01"]
        self.assertEqual([await serialize.deserialize(data) for data in sent], [None] * 3)
        self.assertEqual((socket.handed, port.log), (sent, []))

    async def test_it_never_raises(self):
        class FailingSocket(FakeSocket):
            async def handle(self, data):
                raise RuntimeError("the socket failed")

        broken, _, _ = serializer(FORMATS["pcm 24 kHz"], FailingSocket())
        self.assertIsNone(await broken.deserialize("anything"))
        serialize, socket, _ = serializer(FORMATS["pcm 24 kHz"])
        socket.audio["half a sample"] = b"\x01"
        self.assertIsNone(await serialize.deserialize("half a sample"))

    async def test_reply_audio_goes_out_in_the_wire_format(self):
        samples = np.arange(-32768, 32768, 64, dtype="<i2").tobytes()
        for name, audio_format in FORMATS.items():
            with self.subTest(name):
                serialize, _, port = serializer(audio_format)
                frame = OutputAudioRawFrame(audio=samples, sample_rate=audio_format.rate, num_channels=1)
                self.assertIsNone(await serialize.serialize(frame))
                [audio] = port.packets()
                expected = samples
                if audio_format.type != "audio/pcm":
                    encode, rate = CODECS[audio_format.type].encode, audio_format.rate
                    expected = await encode(samples, rate, rate, create_stream_resampler())
                self.assertEqual(audio, expected)
