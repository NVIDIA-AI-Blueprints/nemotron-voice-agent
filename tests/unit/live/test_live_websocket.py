# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""A primary WebSocket on Starlette's own socket, through the live route and the cascade adapter's Pipecat transport.

``TestClient`` serves the live routes in process: the socket the gateway and Pipecat's ``FastAPIWebsocketTransport``
read is Starlette's own, and the session's events go out on it. The session runs on the cascade adapter, with fake
speech and a scripted talker, and on the gateway's one core, as the whole-call tests build them
(``tests/unit/cascade_helpers.py``).
"""

import asyncio
import base64
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from fastapi.testclient import TestClient
from pipecat.frames.frames import InputAudioRawFrame
from starlette.websockets import WebSocketDisconnect

from examples.shared import audio_recorder
from tests.unit.cascade_helpers import FakeSTT, FakeTTS, _decision_chunks, cascade_parts, chat_core, core_hook
from tests.unit.cross_box.test_frontend_backend_live_engine import RESPONSES_SESSION, SILENCE_100MS
from tests.unit.frontend_helpers import ScriptedChat
from tests.unit.live.test_live_gateway import SESSIONS, make_app

APPEND = {"type": "session.input_audio.append", "audio": SILENCE_100MS}


class RecordingSTT(FakeSTT):
    """The speech-to-text stand-in, which also keeps the caller audio it is handed."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.audio: list[InputAudioRawFrame] = []
        self._arrived = asyncio.Event()

    async def process_frame(self, frame, direction):
        if isinstance(frame, InputAudioRawFrame):
            self.audio.append(frame)
            self._arrived.set()
        await super().process_frame(frame, direction)

    async def wait_for_audio(self, frames, timeout=5.0):
        async with asyncio.timeout(timeout):
            while len(self.audio) < frames:
                self._arrived.clear()
                await self._arrived.wait()


class Received:
    """The events the client of a primary WebSocket receives, read with ``receive_json`` as the test waits for them.

    Each read runs on ``reader``, a one-thread executor, and waits no longer than its caller allows.
    """

    def __init__(self, ws, reader):
        self.ws, self.reader, self.events = ws, reader, []

    def wait_for(self, kind, *, after=0, timeout=10.0, where=None):
        """The first event of ``kind`` from index ``after`` that ``where`` accepts, read off the socket until it comes.

        Each read waits only for the time left, so the wait ends by its deadline even when nothing comes.
        """
        deadline = time.monotonic() + timeout
        while True:
            for event in self.events[after:]:
                if event["type"] == kind and (where is None or where(event)):
                    return event
            after = len(self.events)
            try:
                self.events.append(self.read(max(deadline - time.monotonic(), 0)))
            except TimeoutError:
                raise AssertionError(f"no {kind} within {timeout} s; received {self.kinds()}") from None

    def read(self, timeout):
        """The next event; ``TimeoutError`` after ``timeout`` s, and ``WebSocketDisconnect`` on the server's close."""
        try:
            return self.reader.submit(self.ws.receive_json).result(timeout)
        except TimeoutError:
            raise TimeoutError(f"nothing on the socket within {timeout} s; received {self.kinds()}") from None

    def kinds(self):
        return [event["type"] for event in self.events if event["type"] != "session.output_audio.delta"]

    def audio(self, after=0):
        """The audio of each ``session.output_audio.delta`` from index ``after``."""
        deltas = self.events[after:]
        return [base64.b64decode(event["delta"]) for event in deltas if event["type"] == "session.output_audio.delta"]


class PrimaryWebSocketTests(unittest.TestCase):
    def setUp(self):
        # The shared audio recorder's switches at their defaults, whatever the environment sets: nothing records.
        self.enterContext(patch.object(audio_recorder, "ENABLE_ASR_DUMP", False))
        self.enterContext(patch.object(audio_recorder, "ENABLE_TTS_DUMP", False))

    def test_a_session_starts_hears_the_caller_replies_and_closes_on_starlettes_socket(self):
        chat = ScriptedChat(_decision_chunks('{"action":"speak","speech":"Hello there."}'))
        stt, tts = RecordingSTT(), FakeTTS()

        async def factory(config, transport, ice_servers):
            return cascade_parts(config, transport, ice_servers, chat=chat, stt=stt, tts=tts)

        app, _, _ = make_app(factory=factory, core=core_hook(chat_core(chat)))
        with (
            ThreadPoolExecutor(max_workers=1) as reader,  # closes last: the socket's close ends a read still waiting
            TestClient(app) as client,
            client.websocket_connect(SESSIONS) as ws,
        ):
            received = Received(ws, reader)
            ws.send_json({"type": "session.start", "event_id": "start_1", "session": RESPONSES_SESSION})
            started = received.wait_for("session.started")
            self.assertEqual(
                (started["client_event_id"], started["session"]["delegation"]["type"]), ("start_1", "responses")
            )

            # Each append reaches the speech-to-text: the serializer turns its 24 kHz wire audio into a 16 kHz frame.
            for _ in range(10):
                ws.send_json(APPEND)
            client.portal.call(stt.wait_for_audio, 10)
            self.assertEqual({(frame.sample_rate, frame.num_channels) for frame in stt.audio}, {(16000, 1)})

            # Pipecat's reader passes on a binary frame as Starlette gives it: an error event, and the session goes on.
            sent = len(received.events)
            ws.send_bytes(b"\x00\x01")
            refused = received.wait_for("error", after=sent)
            self.assertIn("Binary frames are not supported", refused["error"]["message"])

            # The caller's transcript, and the reply: its words, and its audio through Pipecat's output transport.
            said = len(received.events)
            client.portal.call(stt.say, "Hi.")
            received.wait_for("session.output_transcript.delta", after=said, where=lambda e: "there." in e["delta"])
            heard = received.wait_for("session.input_transcript.delta", after=said)
            self.assertEqual(heard["delta"].strip(), "Hi.")
            spoken = [
                e["delta"].strip() for e in received.events[said:] if e["type"] == "session.output_transcript.delta"
            ]
            self.assertEqual(spoken, ["Hello", "there."])
            reply = b"".join(audio for audio in received.audio(after=said) if any(audio))
            self.assertEqual(reply, b"\x10" * 9600)  # FakeTTS's two words, whole, at the wire's 24 kHz

            asked = len(received.events)
            ws.send_json({"type": "session.close", "event_id": "bye"})
            closed = received.wait_for("session.closed", after=asked)
            self.assertEqual((closed["reason"], closed["client_event_id"]), ("close_requested", "bye"))
            with self.assertRaises(WebSocketDisconnect) as gone:  # nothing follows session.closed but the close
                received.read(5.0)
            self.assertEqual(gone.exception.code, 1000)


if __name__ == "__main__":
    unittest.main()
