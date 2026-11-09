# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""A primary WebSocket on Starlette's own socket, through the live route and the cascade's Pipecat transport."""

import base64
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from fastapi.testclient import TestClient
from pipecat.frames.frames import InputAudioRawFrame
from starlette.websockets import WebSocketDisconnect

from examples.shared import audio_recorder
from tests.unit.cascade_helpers import (
    RESPONSES_SESSION,
    SILENCE_100MS,
    FakeSTT,
    FakeTTS,
    cascade_parts,
    chat_core,
    core_hook,
)
from tests.unit.frontend_helpers import ScriptedChat, decision
from tests.unit.live.test_live_gateway import SESSIONS, make_app


class RecordingSTT(FakeSTT):
    """The speech-to-text stand-in, which also keeps the caller audio frames it is handed."""

    def __init__(self):
        super().__init__()
        self.audio = []

    async def process_frame(self, frame, direction):
        if isinstance(frame, InputAudioRawFrame):
            self.audio.append(frame)
        await super().process_frame(frame, direction)


def read_until(ws, reader, kind, where=lambda event: True, timeout=10.0):
    """Return the events read off ``ws`` up to the first of ``kind`` that ``where`` accepts, within ``timeout``."""
    deadline, events = time.monotonic() + timeout, []
    while not events or events[-1]["type"] != kind or not where(events[-1]):
        events.append(reader.submit(ws.receive_json).result(max(deadline - time.monotonic(), 0)))
    return events


class PrimaryWebSocketTests(unittest.TestCase):
    @mock.patch.object(audio_recorder, "ENABLE_ASR_DUMP", False)
    @mock.patch.object(audio_recorder, "ENABLE_TTS_DUMP", False)
    def test_a_session_hears_replies_and_closes_on_starlettes_socket(self):
        chat, stt = ScriptedChat(decision("speak", "Hello there.")), RecordingSTT()

        async def factory(config, transport, ice_servers):
            return cascade_parts(config, transport, ice_servers, chat=chat, stt=stt, tts=FakeTTS())

        app, _, _ = make_app(factory=factory, core=core_hook(chat_core(chat)))
        with (
            ThreadPoolExecutor(max_workers=1) as reader,  # closes last: the socket's close ends a read still waiting
            TestClient(app) as client,
            client.websocket_connect(SESSIONS) as ws,
        ):
            ws.send_json({"type": "session.start", "event_id": "start_1", "session": RESPONSES_SESSION})
            self.assertEqual(read_until(ws, reader, "session.started")[-1]["client_event_id"], "start_1")
            for _ in range(10):
                ws.send_json({"type": "session.input_audio.append", "audio": SILENCE_100MS})
            deadline = time.monotonic() + 5
            while len(stt.audio) < 10 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual({(frame.sample_rate, frame.num_channels) for frame in stt.audio}, {(16000, 1)})
            ws.send_bytes(b"\x00\x01")
            self.assertIn("Binary frames are not supported", read_until(ws, reader, "error")[-1]["error"]["message"])
            client.portal.call(stt.say, "Hi.")
            reply = read_until(ws, reader, "session.output_transcript.delta", where=lambda e: "there." in e["delta"])
            heard = [e["delta"].strip() for e in reply if e["type"] == "session.input_transcript.delta"]
            audio = b"".join(base64.b64decode(e["delta"]) for e in reply if e["type"] == "session.output_audio.delta")
            self.assertEqual((heard, audio.replace(b"\x00", b"")), (["Hi."], b"\x10" * 9600))  # two words at 24 kHz
            ws.send_json({"type": "session.close", "event_id": "bye"})
            closed = read_until(ws, reader, "session.closed")[-1]
            self.assertEqual((closed["reason"], closed["client_event_id"]), ("close_requested", "bye"))
            with self.assertRaises(WebSocketDisconnect) as gone:
                reader.submit(ws.receive_json).result(5)
            self.assertEqual(gone.exception.code, 1000)


if __name__ == "__main__":
    unittest.main()
