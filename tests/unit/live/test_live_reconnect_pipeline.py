# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""A retained result crosses fresh real Pipecat frontends without repeating backend execution."""

import asyncio
import unittest
from unittest.mock import patch

from examples.shared import audio_recorder
from gateways.live.gateway import LiveGateway, LiveSettings
from gateways.live.gateway.websocket import run_primary_websocket
from gateways.live.protocol import ProtocolError
from tests.unit.cascade_helpers import FakeSTT, FakeTTS, cascade_parts
from tests.unit.frontend_helpers import ScriptedChat, decision, wait_until
from tests.unit.live.test_live_reconnect import ANSWER, CONFIG, RestorableAdapter
from tests.unit.live_helpers import FakeWebSocket, Stream, response


class ReconnectPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_an_answer_finished_away_is_written_once_by_a_fresh_pipeline(self):
        core, frontends, voices = RestorableAdapter(), [], []

        async def factory(config, transport, ice_servers):
            tts = FakeTTS()
            chat = ScriptedChat(decision("speak", "Your reservation is confirmed."))
            parts = cascade_parts(config, transport, ice_servers, chat=chat, stt=FakeSTT(), tts=tts)
            frontends.append(parts.frontend)
            voices.append(tts)
            return parts

        gateway = LiveGateway(factory, LiveSettings(), core=core.core)
        sockets, runners = [], []

        async def connect(first):
            socket = FakeWebSocket()
            socket.send(first)
            sockets.append(socket)
            runners.append(asyncio.create_task(run_primary_websocket(gateway, socket)))
            return socket

        with (
            patch.object(audio_recorder, "ENABLE_ASR_DUMP", False),
            patch.object(audio_recorder, "ENABLE_TTS_DUMP", False),
        ):
            try:
                socket = await connect({"type": "session.start", "session": CONFIG, "reconnect": {"version": 1}})
                ready = await socket.wait_for("session.reconnect.ready")
                session = next(iter(gateway.sessions.values()))
                finish = asyncio.Event()
                core.streams.append(
                    Stream(response("resp1", end=None), until=finish, then=response("resp1", ANSWER)[1:])
                )
                await session.delegation.delegate([{"role": "user", "content": "Make my reservation"}])
                await socket.wait_for("response.event")
                old_view, old_port = session.adapter_view, session.delegation_port
                original = frontends[0].voice
                caller_turn = {"role": "user", "content": "Caller asked for a reservation."}
                original.context.set_messages([*original.context.get_messages(), caller_turn])
                original.decided[id(caller_turn)] = caller_turn
                socket.leave(1006)
                await asyncio.wait_for(runners[0], 5)
                self.assertTrue(frontends[0].voice._closed)
                self.assertFalse(old_view.is_active())
                self.assertEqual(session.attachment_state, "reconnecting")
                finish.set()
                await asyncio.gather(*session.delegation.readers)
                authority = ready["reconnect"]
                resumed = await connect({"type": "session.start", "resume": {**authority, "request_id": "rejoin"}})
                receipt = await resumed.wait_for("session.resumed")
                self.assertEqual(receipt["session"]["id"], session.id)
                self.assertNotEqual(authority["token"], receipt["reconnect"]["token"])
                await wait_until(
                    lambda: "confirmed." in " ".join(voices[-1].spoken), "the retained result's speech", timeout=5
                )
                self.assertEqual(len(frontends), 2)
                self.assertIsNot(frontends[0].voice, frontends[1].voice)
                self.assertEqual(voices[0].spoken, [])
                self.assertEqual(" ".join(voices[1].spoken), "Your reservation is confirmed.")
                self.assertFalse(session.frontend.voice.reader.open)
                self.assertIn(
                    caller_turn["content"],
                    " ".join(message["content"] for message in session.frontend.voice.context.get_messages()),
                )
                await old_view.emit("session.output_transcript.delta", delta="stale")
                old_view.request_close("remote_hangup")
                self.assertEqual(session.status, "active")
                self.assertFalse(any(event.get("delta") == "stale" for event in resumed.from_server))
                with self.assertRaises(ProtocolError):
                    old_port.listen(frontends[0].voice.reader)
                with self.assertRaises(ProtocolError):
                    await old_port.delegate([{"role": "user", "content": "Do it again."}])
                self.assertEqual(len(core.requests), 1)
            finally:
                await gateway.close_all()
                for socket in sockets:
                    socket.leave()
                await asyncio.gather(*runners, return_exceptions=True)
