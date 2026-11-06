# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""A primary WebSocket's serializer: caller audio in and reply audio out in each wire format, and every other frame."""

import asyncio
import unittest

import numpy as np
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
)

from examples.frontend_backend_live.adapter.serializer import CODECS, LiveFrameSerializer
from gateways.live.protocol import AudioFormat
from tests.unit.frontend_backend_live.adapter.adapter_helpers import FakeSocket, RecordingPort, tone

AUDIO_IN_RATE = 16000  # the rate the frontend hears the caller at
FORMATS = {  # each wire format the protocol allows, at each of its rates
    "pcm 16 kHz": AudioFormat(type="audio/pcm", rate=16000),
    "pcm 24 kHz": AudioFormat(type="audio/pcm", rate=24000),
    "mu-law 8 kHz": AudioFormat(type="audio/pcmu"),
    "A-law 8 kHz": AudioFormat(type="audio/pcma"),
}
SAMPLE_BYTES = {"audio/pcm": 2, "audio/pcmu": 1, "audio/pcma": 1}


def reference(sample: int, alaw: bool) -> int:
    """ITU-T G.711 of one 16-bit sample, as the CCITT reference (Sun g711.c) encodes it: A-law or mu-law."""
    if alaw:
        pcm, mask = (sample >> 3, 0xD5) if sample >= 0 else (-(sample >> 3) - 1, 0x55)
        segment = sum(pcm > end for end in (0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF))
        code = 0x7F if segment >= 8 else (segment << 4) | ((pcm >> (1 if segment < 2 else segment)) & 0xF)
        return code ^ mask
    pcm, mask = (-(sample >> 2), 0x7F) if sample < 0 else (sample >> 2, 0xFF)
    pcm = min(pcm, 8159) + 33
    segment = sum(pcm > end for end in (0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF))
    return (0x7F if segment >= 8 else (segment << 4) | ((pcm >> (segment + 1)) & 0xF)) ^ mask


def reference_decode(code: int, alaw: bool) -> int:
    """ITU-T G.711 of one code back to its 16-bit sample, as the CCITT reference decodes it."""
    if alaw:
        code ^= 0x55
        sample, exponent = (code & 15) << 4, (code & 112) >> 4
        sample += 8 if exponent == 0 else 264
        if exponent > 1:
            sample <<= exponent - 1
        return sample if code & 128 else -sample
    code = ~code & 255
    sample = (((code & 15) << 3) + 132) << ((code & 112) >> 4)
    return 132 - sample if code & 128 else sample - 132


def serializer(audio_format: AudioFormat) -> tuple[LiveFrameSerializer, FakeSocket, RecordingPort]:
    socket = FakeSocket(audio_format)
    port = RecordingPort(socket=socket)
    return LiveFrameSerializer(socket, port, AUDIO_IN_RATE), socket, port


class G711Tests(unittest.IsolatedAsyncioTestCase):
    """The G.711 step, at one rate: the conversion the reference defines, byte for byte."""

    async def test_every_16_bit_sample_encodes_as_the_reference_does(self):
        samples = np.arange(-32768, 32768, dtype="<i2")
        for kind, alaw in (("audio/pcmu", False), ("audio/pcma", True)):
            with self.subTest(kind=kind):
                encoded = await CODECS[kind].encode(samples.tobytes(), 8000, 8000, create_stream_resampler())
                expected = bytes(reference(int(sample), alaw) for sample in samples)
                self.assertEqual(encoded, expected)

    async def test_every_code_decodes_as_the_reference_does(self):
        for kind, alaw in (("audio/pcmu", False), ("audio/pcma", True)):
            with self.subTest(kind=kind):
                decoded = await CODECS[kind].decode(bytes(range(256)), 8000, 8000, create_stream_resampler())
                expected = np.array([reference_decode(code, alaw) for code in range(256)], dtype="<i2").tobytes()
                self.assertEqual(decoded, expected)


class CallerAudioTests(unittest.IsolatedAsyncioTestCase):
    async def heard(self, audio_format, chunks, pause_at=None):
        """The samples the frontend hears after each of ``chunks`` appends of 100 ms, with a pause before one."""
        serialize, socket, _ = serializer(audio_format)
        samples = audio_format.rate // 10
        socket.audio["append"] = (bytes(range(256)) * samples)[: samples * SAMPLE_BYTES[audio_format.type]]
        heard, total = [], 0
        for count in range(chunks):
            if count == pause_at:
                await asyncio.sleep(0.25)  # longer than Pipecat's default for clearing a resampler
            frame = await serialize.deserialize("append")
            if frame is not None:  # the resampler may still hold all of it
                self.assertIsInstance(frame, InputAudioRawFrame)
                self.assertEqual((frame.sample_rate, frame.num_channels), (AUDIO_IN_RATE, 1))
                total += len(frame.audio) // 2
            heard.append(total)
        return heard

    async def test_caller_audio_becomes_frames_at_the_frontends_rate_in_each_wire_format(self):
        # The stream resampler holds a little of the caller's audio back, in blocks (none at 16 kHz); a pause in the
        # caller's stream drops none of it, so the input clock never falls further behind.
        for name, audio_format in FORMATS.items():
            with self.subTest(name):
                steady, paused = await self.heard(audio_format, 30), await self.heard(audio_format, 30, pause_at=10)
                sent = [AUDIO_IN_RATE // 10 * count for count in range(1, 31)]
                held = [want - got for want, got in zip(sent, steady, strict=True)]
                self.assertEqual(paused, steady)
                self.assertTrue(all(0 <= samples <= AUDIO_IN_RATE * 120 // 1000 for samples in held), held)
                self.assertEqual(held[-1] == 0, audio_format.rate == AUDIO_IN_RATE)

    async def test_16_khz_pcm_passes_as_it_came(self):
        serialize, socket, _ = serializer(FORMATS["pcm 16 kHz"])
        socket.audio["append"] = tone(100, rate=16000)
        frame = await serialize.deserialize("append")
        self.assertEqual(frame.audio, tone(100, rate=16000))

    async def test_every_other_frame_goes_to_the_socket_and_becomes_no_frame(self):
        serialize, socket, port = serializer(FORMATS["pcm 24 kHz"])
        for data in ('{"type": "session.input_audio.mute"}', "not json", b"\x00\x01"):
            with self.subTest(data=data):
                self.assertIsNone(await serialize.deserialize(data))
        self.assertEqual(socket.handed, ['{"type": "session.input_audio.mute"}', "not json", b"\x00\x01"])
        self.assertEqual(port.log, [])

    async def test_it_never_raises(self):
        class FailingSocket(FakeSocket):
            async def handle(self, data):
                raise RuntimeError("the socket failed")

        failing = FailingSocket()
        broken = LiveFrameSerializer(failing, RecordingPort(socket=failing), AUDIO_IN_RATE)
        self.assertIsNone(await broken.deserialize("anything"))
        serialize, socket, _ = serializer(FORMATS["pcm 24 kHz"])
        socket.audio["half a sample"] = b"\x01"  # the gateway's socket refuses it; were it given, it is no frame
        self.assertIsNone(await serialize.deserialize("half a sample"))


class ReplyAudioTests(unittest.IsolatedAsyncioTestCase):
    async def test_reply_audio_goes_out_through_the_session_in_the_wire_format(self):
        samples = np.arange(-32768, 32768, 64, dtype="<i2").tobytes()
        for name, audio_format in FORMATS.items():
            with self.subTest(name):
                serialize, _, port = serializer(audio_format)
                frame = OutputAudioRawFrame(audio=samples, sample_rate=audio_format.rate, num_channels=1)
                self.assertIsNone(await serialize.serialize(frame))  # the transport itself sends nothing
                [audio] = port.packets()
                if audio_format.type == "audio/pcm":
                    self.assertEqual(audio, samples)  # PCM16 at the wire rate goes out as it came
                else:
                    alaw = audio_format.type == "audio/pcma"
                    pcm = np.frombuffer(samples, dtype="<i2")
                    self.assertEqual(audio, bytes(reference(int(sample), alaw) for sample in pcm))

    async def test_no_other_frame_sends_anything(self):
        serialize, _, port = serializer(FORMATS["mu-law 8 kHz"])
        message = OutputTransportMessageFrame(message={"type": "anything"})
        for frame in (InterruptionFrame(), EndFrame(), CancelFrame(), message):
            with self.subTest(frame=type(frame).__name__):
                self.assertIsNone(await serialize.serialize(frame))
        self.assertEqual(port.log, [])


if __name__ == "__main__":
    unittest.main()
