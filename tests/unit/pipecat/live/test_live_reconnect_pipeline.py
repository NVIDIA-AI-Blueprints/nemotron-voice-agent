# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""A WebSocket resume restores the caller's media and delivers retained work through a fresh Pipecat pipeline."""

import asyncio
import base64
import unittest
from unittest.mock import patch

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.adapter.adapter import (
    LIVE_AUDIO_IN_RATE,
    CascadeAdapter,
)
from nemotron_voice_agent.pipecat.runtime.gateways.live.gateway import LiveGateway, LiveSettings
from nemotron_voice_agent.pipecat.runtime.gateways.live.gateway.websocket import run_primary_websocket
from nemotron_voice_agent.pipecat.runtime.gateways.live.protocol import ProtocolError
from nemotron_voice_agent.pipecat.shared import audio_recorder
from tests.unit.pipecat.cascade_helpers import FakeTTS, cascade_parts
from tests.unit.pipecat.frontend_helpers import ScriptedChat, decision, wait_until
from tests.unit.pipecat.live.test_live_reconnect import ANSWER, CONFIG, RestorableAdapter
from tests.unit.pipecat.live.test_live_websocket import RecordingSTT
from tests.unit.pipecat.live_helpers import FakeWebSocket, Stream, response


class ReconnectPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_an_answer_finished_away_is_written_once_by_a_fresh_pipeline(self):
        core, frontends, voices, listeners = RestorableAdapter(), [], [], []

        async def factory(config, transport, ice_servers):
            stt, tts = RecordingSTT(), FakeTTS()
            chat = ScriptedChat(decision("speak", "Your reservation is confirmed."), decision("wait", ""))
            parts = cascade_parts(config, transport, ice_servers, chat=chat, stt=stt, tts=tts)
            frontends.append(parts.frontend)
            voices.append(tts)
            listeners.append(stt)
            return parts

        gateway = LiveGateway(factory, LiveSettings(), core=core.core)
        sockets, runners = [], []

        async def connect(first):
            socket = FakeWebSocket()
            socket.send(first)
            sockets.append(socket)
            runners.append(asyncio.create_task(run_primary_websocket(gateway, socket)))
            return socket

        def send_audio(socket, duration_ms, *, sample=0):
            pcm = bytes([sample, 0]) * (LIVE_AUDIO_IN_RATE * duration_ms // 1000)
            socket.send({"type": "session.input_audio.append", "audio": base64.b64encode(pcm).decode()})

        restoring, restore = asyncio.Event(), asyncio.Event()
        prepare_resume = CascadeAdapter.prepare_resume

        async def hold_resume(frontend):
            restoring.set()
            await restore.wait()
            await prepare_resume(frontend)

        with (
            patch.object(audio_recorder, "ENABLE_ASR_DUMP", False),
            patch.object(audio_recorder, "ENABLE_TTS_DUMP", False),
            patch.object(CascadeAdapter, "prepare_resume", hold_resume),
        ):
            try:
                config = {**CONFIG, "audio": {"format": {"type": "audio/pcm", "rate": LIVE_AUDIO_IN_RATE}}}
                socket = await connect({"type": "session.start", "session": config, "reconnect": {"version": 1}})
                ready = await socket.wait_for("session.reconnect.ready")
                session = next(iter(gateway.sessions.values()))
                send_audio(socket, 300)
                await wait_until(lambda: bool(listeners[0].audio), "caller audio before the disconnect")
                self.assertEqual(session.input_ms, 300)
                socket.send({"type": "session.input_audio.mute"})
                await socket.wait_for("session.input_audio.muted")
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
                await asyncio.wait_for(restoring.wait(), 5)
                send_audio(resumed, 400)
                resumed.send({"type": "session.update", "event_id": "while-restoring", "session": {}})
                refused = await resumed.wait_for(
                    "error", where=lambda event: event.get("client_event_id") == "while-restoring"
                )
                self.assertEqual(refused["error"]["code"], "session_reconnecting")
                self.assertNotIn("session.resumed", resumed.kinds())
                restore.set()
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

                send_audio(resumed, 100)
                await wait_until(lambda: session.input_ms >= 400, "the resumed input clock while muted")
                self.assertEqual(session.input_ms, 400)
                resumed.send({"type": "session.input_audio.unmute"})
                await resumed.wait_for("session.input_audio.unmuted")
                send_audio(resumed, 100, sample=1)
                await wait_until(lambda: any(any(frame.audio) for frame in listeners[1].audio), "audio after unmute")
                self.assertEqual(sum(len(frame.audio) for frame in listeners[1].audio), LIVE_AUDIO_IN_RATE // 5)
                await listeners[1].say("Thank you.")
                transcript = await resumed.wait_for("session.input_transcript.delta")
                self.assertEqual(transcript["end_ms"], 500)
                self.assertGreaterEqual(transcript["start_ms"], 300)
            finally:
                restore.set()
                await gateway.close_all()
                for socket in sockets:
                    socket.leave()
                await asyncio.gather(*runners, return_exceptions=True)
