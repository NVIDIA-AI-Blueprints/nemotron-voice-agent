# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Whole calls through the Live gateway, the cascade adapter and the delegation core, on scripted models."""

import asyncio
import json
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import AudioStreamTrack
from fastapi import FastAPI
from loguru import logger
from pipecat.frames.frames import LLMTextFrame, MetricsFrame, VADUserStoppedSpeakingFrame
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.observers.base_observer import BaseObserver
from pipecat.processors.frame_processor import FrameProcessor

from examples.frontend_backend_live.adapter.adapter import CascadeAdapter
from examples.frontend_backend_live.delegation_core.im.input import compose
from examples.frontend_backend_live.heard_log import Heard
from examples.frontend_backend_live.prompts import load_prompts
from examples.shared import audio_recorder
from gateways.live.gateway import LiveGateway, LiveSettings
from gateways.live.gateway.websocket import run_primary_websocket
from gateways.live.mount import mount_live
from gateways.live.server_delegation.passthrough import COMPOSED_SOURCE
from tests.unit.cascade_helpers import (
    ENDPOINT,
    LIVE_CONFIG,
    MENU_TOOL,
    RESPONSES_SESSION,
    SILENCE_100MS,
    FakeSTT,
    FakeTTS,
    cascade_parts,
    chat_core,
    core_hook,
)
from tests.unit.core_helpers import chunk
from tests.unit.frontend_helpers import ScriptedChat, ToneTTS, decision, tool_call
from tests.unit.live_helpers import FakeWebSocket

PROMPTS, LINES = load_prompts().cascade, load_prompts().lines
CLIENT_SESSION = {"model": "live-1", "instructions": "You are a kiosk voice assistant."}
# A Live client with its own agent model: its instructions for the backend, one call per round.
OWN_MODEL = {"model": "client/agent-model", "instructions": "Be brief.", "tools": [MENU_TOOL], "tool_choice": "auto"}
OWN_MODEL_SESSION = {
    "model": "live-1",
    "instructions": "You are a kiosk voice assistant. Delegate every request.",
    "delegation": {
        "type": "responses",
        "responses": {**OWN_MODEL, "parallel_tool_calls": False, "reasoning": {"effort": "low"}},
    },
}


def menu_call():
    return tool_call("get_menu", '{"category":"tea"}')


class EngineCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.object(audio_recorder, "ENABLE_ASR_DUMP", False))
        self.enterContext(patch.object(audio_recorder, "ENABLE_TTS_DUMP", False))

    async def serve(self, session, script, *, live_config=LIVE_CONFIG, models=(), tts=FakeTTS, early=(), **delegation):
        """Serve ``session`` over a WebSocket, sending ``early`` before it starts; both models run on ``script``."""
        self.chat, self.stt, self.tts = ScriptedChat(*script), FakeSTT(), tts()
        core = chat_core(self.chat, models=models, **delegation)
        self.addAsyncCleanup(core.aclose)
        deadline = core.settings.delegation.client_tool_timeout_seconds
        wired = live_config.model_copy(update={"client_tool_timeout_seconds": deadline})

        async def factory(config, transport, connection):
            parts = cascade_parts(
                config, transport, connection, chat=self.chat, stt=self.stt, tts=self.tts, live_config=wired
            )
            self.engine = parts.frontend
            return parts

        self.hook = core_hook(core)
        self.gateway = LiveGateway(factory, LiveSettings(max_session_seconds=60), core=self.hook)
        self.ws = FakeWebSocket()
        self.server = asyncio.create_task(run_primary_websocket(self.gateway, self.ws))
        self.addAsyncCleanup(self.stop)
        for message in ({"type": "session.start", "event_id": "start_1", "session": session}, *early):
            self.ws.send(message)
        await self.ws.wait_for("session.started")

    @property
    def live_session(self):
        return next(iter(self.gateway.sessions.values()))

    def said(self):
        return [e["delta"].strip() for e in self.ws.from_server if e["type"] == "session.output_transcript.delta"]

    async def heard(self, text):
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: text in e["delta"])

    async def backend(self, kind):
        return await self.ws.wait_for("response.event", where=lambda e: e["event"]["type"] == kind)

    def calls(self):
        events = [e["event"] for e in self.ws.from_server if e["type"] == "response.event"]
        done = [e["item"] for e in events if e["type"] == "response.output_item.done"]
        return [item for item in done if item["type"] == "function_call"]

    async def until(self, predicate, what, timeout=4.0):
        try:
            async with asyncio.timeout(timeout):
                while not predicate():
                    await asyncio.sleep(0.01)
        except TimeoutError:
            raise AssertionError(f"timed out waiting for {what}") from None

    async def stop(self):
        self.ws.send({"type": "session.close"})
        try:
            async with asyncio.timeout(10):
                await self.server
        except (TimeoutError, asyncio.CancelledError):
            self.server.cancel()


class DelegationTests(EngineCase):
    async def test_the_clients_tool_runs_and_its_answer_is_spoken(self):
        answer = "We have Earl Grey and Matcha."
        script = [decision("delegate", "Let me check."), menu_call(), [chunk(answer, finish="stop")]]
        await self.serve(RESPONSES_SESSION, [*script, decision("speak", answer)])
        await self.stt.say("What teas do you have?")
        created = (await self.ws.wait_for("session.delegation.created"))["delegation"]
        self.assertEqual(created["target"], "responses")
        self.assertTrue(created["response_id"].startswith("resp_"))
        completed = await self.backend("response.completed")
        self.assertEqual(completed["event"]["response"]["output"], [])
        [call] = self.calls()
        self.assertEqual((call["name"], json.loads(call["arguments"])), ("get_menu", {"category": "tea"}))
        # The lead-in is said once the delegation is accepted; a result that arrived first would speak for it.
        await self.heard("check.")
        output = {"type": "function_call_output", "call_id": call["call_id"], "output": '{"teas": ["Earl Grey"]}'}
        self.ws.send({"type": "response.item.create", "item": output})
        self.ws.send({"type": "response.create", "event_id": "go"})
        await self.heard("Matcha")
        self.assertEqual(" ".join(self.said()), "Let me check. We have Earl Grey and Matcha.")
        tool_message = next(m for m in self.chat.requests[2]["messages"] if m["role"] == "tool")
        self.assertEqual(json.loads(tool_message["content"]), {"teas": ["Earl Grey"]})
        self.assertEqual(self.chat.requests[2]["tools"][0]["function"]["name"], "get_menu")

    async def test_a_repeated_result_is_refused_with_the_cores_reason(self):
        await self.serve(RESPONSES_SESSION, [decision("delegate"), menu_call()])
        await self.stt.say("Look it up.")
        await self.backend("response.completed")
        [call] = self.calls()
        result = {"type": "function_call_output", "call_id": call["call_id"], "output": "first"}
        self.ws.send({"type": "response.item.create", "item": result})
        self.ws.send({"type": "response.item.create", "item": {**result, "output": "second"}})
        self.ws.send({"type": "response.create", "event_id": "both"})
        refused = await self.ws.wait_for("error", where=lambda e: e["error"].get("client_event_id") == "both")
        reason = f"Duplicate result for call {call['call_id']}"
        self.assertEqual((refused["error"]["code"], refused["error"]["message"]), ("invalid_value", reason))

    async def test_the_backend_is_busy_while_the_app_runs_its_tool(self):
        delegation = LIVE_CONFIG.delegation.model_copy(update={"app_wait_seconds": 0.1})
        live_config = LIVE_CONFIG.model_copy(update={"delegation": delegation})
        script = [decision("delegate", "Let me check."), menu_call()]
        await self.serve(RESPONSES_SESSION, script, live_config=live_config, client_tool_timeout_seconds=2.0)
        await self.stt.say("What teas do you have?")
        await self.backend("response.completed")
        self.assertTrue((await self.live_session.history())["backend"]["busy"])
        await self.heard(LINES.app_wait_line.split()[-1])
        self.assertTrue((await self.live_session.history())["backend"]["busy"])
        await self.heard(LINES.unclear.split()[0])
        self.assertFalse((await self.live_session.history())["backend"]["busy"])

    async def test_a_typed_question_runs_its_model_and_the_writer_sees_it(self):
        # A client's session names its own agent model, which the catalog lists: it starts with no error,
        # and its typed run goes to that model.
        answer = "Your order is on its way."
        script = [[chunk(answer, finish="stop")], decision("speak", answer)]
        await self.serve(OWN_MODEL_SESSION, script, models=("client/agent-model",))
        await asyncio.gather(*self.live_session.delegation.checks)
        typed = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Where is my order?"}]}
        self.ws.send({"type": "response.item.create", "event_id": "q1", "item": typed})
        self.ws.send({"type": "response.create", "event_id": "go"})
        await self.heard("way.")
        self.assertNotIn("error", self.ws.kinds())
        self.assertEqual(self.chat.requests[0]["model"], "client/agent-model")
        question = PROMPTS.context_label + "Where is my order?"
        wording = self.chat.requests[1]["messages"][-1]
        self.assertEqual(wording["role"], "user")
        self.assertTrue(wording["content"].startswith(f"{question}\n"), wording["content"])
        context = [(m["role"], m["content"], m["kind"]) for m in self.engine.voice.conversation()]
        self.assertIn(("user", question, "typed"), context)
        self.assertNotIn("caller", [kind for _, _, kind in context])
        self.assertEqual([entry.kind for entry in self.engine.voice.heard_log.entries], [Heard.AGENT])

    async def test_client_mode_announces_and_speaks_the_commentary(self):
        await self.serve(CLIENT_SESSION, [decision("delegate", "One moment."), decision("speak", "It is sunny today.")])
        await self.stt.say("What is the weather?")
        created = (await self.ws.wait_for("session.delegation.created"))["delegation"]
        self.assertEqual(created["target"], "client")
        self.assertNotIn("response_id", created)
        # The client answers once the lead-in has played; commentary that came first would drop it.
        await self.heard("moment.")
        commentary = {"event_id": "c1", "delegation_id": created["id"], "content": "Weather: sunny, 72 degrees."}
        self.ws.send({"type": "session.commentary.append", **commentary})
        await self.heard("sunny")
        update = self.chat.requests[1]["messages"][-1]["content"]
        self.assertIn("Weather: sunny, 72 degrees.", update)
        self.assertIn('(the caller already heard: "One moment."; do not say that again)', update)
        schema = self.chat.requests[1]["response_format"]["json_schema"]["schema"]
        self.assertEqual(schema["properties"]["action"]["enum"], ["speak"])
        self.assertEqual(len(self.chat.requests), 2)
        self.assertNotIn("response.event", self.ws.kinds())

    async def test_history_shows_what_the_talker_and_the_backend_hold(self):
        backend_answers = asyncio.Event()
        answer = [chunk("We have Earl Grey.", finish="stop")]
        script = [decision("delegate", "Let me check."), backend_answers, answer]
        await self.serve(RESPONSES_SESSION, [*script, decision("speak", "We have Earl Grey.")])
        self.ws.send({"type": "session.thinking.append", "delegation_id": None, "content": "The caller is a regular."})
        await self.stt.say("What teas do you have?")
        await self.heard("check.")
        backend_answers.set()
        await self.heard("Grey")

        history = await self.live_session.history()
        self.assertEqual(history, json.loads(json.dumps(history)))
        self.assertEqual(history["frontend"]["model"], ENDPOINT.model)
        frontend = [m["content"] for m in history["frontend"]["messages"]]
        self.assertTrue(any("What teas do you have?" in text for text in frontend))
        self.assertTrue(any("The caller is a regular." in text for text in frontend))
        kinds = {m["content"]: m["kind"] for m in history["frontend"]["messages"]}
        self.assertEqual(kinds[PROMPTS.context_label + "The caller is a regular."], "app_context")
        self.assertEqual(kinds[PROMPTS.result_label + "We have Earl Grey."], "result")
        backend = history["backend"]
        request = backend["items"][0]["content"][0]["text"]
        self.assertTrue(request.startswith(PROMPTS.transcript.header), request)
        self.assertIn("The caller is a regular.", request)
        self.assertIn("What teas do you have?", request)
        self.assertIn("We have Earl Grey.", json.dumps(backend["items"][1:]))
        # The lead-in was said after that input was built, so the next delegation carries it.
        lead_in = {"kind": Heard.AGENT, "text": f"{PROMPTS.transcript.agent}: Let me check."}
        self.assertEqual(backend["pending"], [lead_in])
        im = self.hook.client.service.sessions[self.live_session.delegation.conversation]
        composed = compose("Look things up.", *im.instruction_notes())
        session_part = {"source": "delegation.responses.instructions", "text": "Look things up."}
        self.assertEqual(backend["prompt"], [session_part, {"source": COMPOSED_SOURCE, "text": composed}])


class SessionTests(EngineCase):
    async def test_a_start_runs_each_loop_once_then_early_commands(self):
        # Pipecat's transport reads the socket from the pipeline's start, before the session is active.
        await self.serve(RESPONSES_SESSION, [], early=[{"type": "session.input_audio.mute", "event_id": "early"}])
        muted = await self.ws.wait_for("session.input_audio.muted")
        self.assertEqual(muted["client_event_id"], "early")
        self.assertEqual(self.ws.kinds()[:2], ["session.started", "session.input_audio.muted"])
        self.assertNotIn("error", self.ws.kinds())

        def running(name):
            return [task for task in asyncio.all_tasks() if task.get_name() == name and not task.done()]

        self.assertEqual((len(running("delegation-progress")), len(running("im-placer"))), (1, 1))
        await self.live_session.delegation.open(self.live_session.delegation.view)
        self.assertEqual(len(running("im-placer")), 1)

    async def test_an_instruction_append_is_acted_on_and_acknowledged(self):
        await self.serve(RESPONSES_SESSION, [decision("speak", "Understood.")])
        append = {"event_id": "i", "delegation_id": None, "content": "Greet the caller."}
        self.ws.send({"type": "session.instructions.append", **append})
        await self.heard("Understood")
        self.assertIn("Greet the caller.", self.chat.requests[0]["messages"][0]["content"])
        for _ in range(6):  # the acknowledgement waits for the input clock, which caller audio moves
            self.ws.send({"type": "session.input_audio.append", "audio": SILENCE_100MS})
        ack = await self.ws.wait_for("session.instructions.appended", timeout=3)
        self.assertEqual(ack["client_event_id"], "i")

    async def test_a_direct_reply_is_spoken_whole_and_logs_its_latency(self):
        logged = []
        sink = logger.add(lambda message: logged.append(message.record["message"]), level="INFO")
        self.addCleanup(logger.remove, sink)
        # At 16 kHz the output sends whole 100 ms packets, so a line's first word passes it ahead of its audio.
        developer = {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Kiosk 7."}]}
        session = {**RESPONSES_SESSION, "input": [developer], "audio": {"format": {"type": "audio/pcm", "rate": 16000}}}
        await self.serve(session, [decision("speak", "Hello there.")])
        last, passed = self.engine.frontend.processors[-1], []

        class PassedMetrics(BaseObserver):
            async def on_push_frame(self, data):
                if isinstance(data.frame, MetricsFrame) and data.source is last:
                    passed.extend(data.frame.data)

        self.engine.task.add_observer(PassedMetrics())
        await self.stt.push_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
        await self.stt.say("Hi.")
        await self.heard("there.")
        self.assertEqual(self.said(), ["Hello", "there."])
        self.assertNotIn("session.delegation.created", self.ws.kinds())
        self.assertEqual(len(self.chat.requests), 1)
        system = self.chat.requests[0]["messages"][0]["content"]
        self.assertIn("You are a kiosk voice assistant. Delegate every request.", system)
        self.assertIn("Kiosk 7.", system)
        latency = [line for line in logged if line.startswith("User-to-bot latency: ")]
        breakdown = [line for line in logged if line.startswith("Latency breakdown: ")]
        self.assertEqual((len(latency), len(breakdown)), (1, 1))
        self.assertIn("endpointing wait", breakdown[0])
        self.assertTrue(breakdown[0].endswith("TOTAL"))
        talker = self.engine.voice.talker.name
        self.assertTrue([data for data in passed if isinstance(data, TTFBMetricsData) and data.processor == talker])

    async def test_three_lines_in_a_row_with_no_audio_end_the_call(self):
        # The line, its retry and the next line all get no audio: two notices, then the third writes the service off.
        script = [decision("speak", "Hello there."), decision("speak", "Are you there?")]
        dead = partial(ToneTTS, silent={"Hello there.", "Are you there?"}, stop_frame_timeout_s=0.2)
        await self.serve(RESPONSES_SESSION, script, tts=dead)

        def notices():
            return [(e["code"], e["message"]) for e in self.ws.from_server if e["type"] == "info"]

        await self.stt.say("Hi.")
        await self.until(lambda: len(notices()) >= 2, "the line and the line said once more, each with no audio", 5)
        await self.stt.say("Hello?")
        closed = await self.ws.wait_for("session.closed", timeout=15)
        self.assertEqual((closed["reason"], closed.get("client_event_id")), ("connection_lost", None))
        spoken = [sentence for _, sentence in self.tts.spoken]
        self.assertEqual(spoken, ["Hello there.", "Hello there.", "Are you there?"])
        self.assertEqual([code for code, _ in notices()], ["pipeline_error"] * 2)
        self.assertTrue(all(message.endswith("completed with no audio") for _, message in notices()))
        errors = [(e["error"]["code"], e["error"]["message"]) for e in self.ws.from_server if e["type"] == "error"]
        self.assertEqual(errors, [("pipeline_error", "3 consecutive TTS contexts completed with no audio")])
        self.assertEqual(self.ws.kinds()[-1], "session.closed")

    async def test_a_second_close_stops_only_a_pipeline_started_since(self):
        calls = []

        async def stop_pipeline():
            calls.append("pipeline stopped")

        async def close_voice():
            calls.append("voice closed")

        voice = SimpleNamespace(close=close_voice)
        adapter = SimpleNamespace(_closed=False, runner_task=None, stop_pipeline=stop_pipeline, voice=voice)
        # The session closes its frontend adapter, then its delegation side, and one adapter is both.
        await CascadeAdapter.close(adapter)
        await CascadeAdapter.close(adapter)
        self.assertEqual(calls, ["pipeline stopped", "voice closed"])
        # A session closed while starting closes again after the start: a new pipeline stops, the voice closes once.
        adapter.runner_task = SimpleNamespace(done=lambda: False)
        await CascadeAdapter.close(adapter)
        self.assertEqual(calls, ["pipeline stopped", "voice closed", "pipeline stopped"])
        adapter.runner_task = SimpleNamespace(done=lambda: True)
        await CascadeAdapter.close(adapter)
        self.assertEqual(calls, ["pipeline stopped", "voice closed", "pipeline stopped"])


class WrittenOffTTS(FakeTTS):
    """A voice that Pipecat's ``TTSService`` writes off, as after three lines with no audio, at its first line."""

    async def process_frame(self, frame, direction):
        if isinstance(frame, LLMTextFrame):
            await FrameProcessor.process_frame(self, frame, direction)
            if self.is_usable:
                message = "3 consecutive TTS contexts completed with no audio"
                await self.push_error(message, force_treat_as_permanent=True)
            return
        await super().process_frame(frame, direction)


class WebRTCTests(EngineCase):
    async def call(self, script, *, tts=FakeTTS, core_up=None, label="oai-events", hear=False):
        """Start a session over a real peer on loopback, as a WebRTC client does, and return its data channel."""
        chat, self.stt = ScriptedChat(*script), FakeSTT()

        async def factory(config, transport, ice_servers):
            return cascade_parts(config, transport, ice_servers, chat=chat, stt=self.stt, tts=tts())

        core = chat_core(chat)
        self.addAsyncCleanup(core.aclose)
        core_client = core_hook(core)

        async def hook():
            if core_up is not None:
                await core_up.wait()
            return await core_client()

        app = FastAPI()
        self.gateway = mount_live(app, factory=factory, core=hook)
        self.addAsyncCleanup(self.gateway.close_all)
        client = RTCPeerConnection()
        self.addAsyncCleanup(client.close)
        client.addTrack(AudioStreamTrack())
        channel = client.createDataChannel(label)
        self.seen, self.audio = [], []
        channel.on("message", lambda raw: self.seen.append(json.loads(raw)))
        channel.on("close", lambda: self.seen.append({"type": "<channel closed>"}))
        if hear:
            client.on("track", lambda track: asyncio.create_task(self.hear(track)))
        await client.setLocalDescription(await client.createOffer())
        offer = {"type": "webrtc", "sdp": client.localDescription.sdp}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as http:
            response = await http.post("/v1/live/sessions", json={"session": RESPONSES_SESSION, "transport": offer})
        self.assertEqual(response.status_code, 201, response.text)
        await client.setRemoteDescription(RTCSessionDescription(sdp=response.json()["transport"]["sdp"], type="answer"))
        return channel

    async def hear(self, track):
        try:
            while True:
                frame = await track.recv()
                self.audio.append(int(max(abs(frame.to_ndarray().astype("int32").ravel()), default=0)))
        except Exception:
            return

    def kinds(self):
        return [event["type"] for event in self.seen]

    async def received(self, kind, count=1):
        await self.until(lambda: self.kinds().count(kind) >= count, kind, 20)

    async def test_the_reply_is_heard_and_session_closed_ends_the_channel(self):
        script = [decision("speak", "Hello from the kiosk.")]
        channel = await self.call(script, label="", hear=True)  # some clients use an empty label
        await self.received("session.started")
        await self.stt.say("Hello.")
        await self.received("session.input_transcript.delta")
        await self.received("session.output_transcript.delta", count=4)
        words = [event["delta"].strip() for event in self.seen if event["type"] == "session.output_transcript.delta"]
        self.assertEqual(" ".join(words[:4]), "Hello from the kiosk.")
        await self.until(lambda: max(self.audio, default=0) >= 100, "reply audio on the media track", 10)
        channel.send(json.dumps({"type": "session.close", "event_id": "bye"}))
        await self.received("<channel closed>")
        ends = ("session.closed", "signalling", "<channel closed>")
        self.assertEqual([kind for kind in self.kinds() if kind in ends], ["session.closed", "<channel closed>"])
        await self.until(lambda: not self.gateway.sessions, "the session to leave the gateway", 5)
        self.assertEqual(self.gateway.sessions, {})

    async def test_a_written_off_voice_sends_its_error_then_session_closed(self):
        # The worker ends the pipeline before the session's close; the peer stays open for the error and session.closed.
        await self.call([decision("speak", "Hello there.")], tts=WrittenOffTTS)
        await self.received("session.started")
        await self.stt.say("Hi.")
        await self.received("<channel closed>")
        self.assertEqual(self.kinds()[-2:], ["session.closed", "<channel closed>"])
        self.assertEqual(self.seen[-2]["reason"], "connection_lost")
        errors = [event["error"]["message"] for event in self.seen if event["type"] == "error"]
        self.assertEqual(errors, ["3 consecutive TTS contexts completed with no audio"])
        self.assertNotIn("signalling", self.kinds())

    async def test_a_command_sent_before_the_start_runs_once_it_is_active(self):
        # The core comes up after the command is sent, so the session and Pipecat's transport start after it.
        core_up = asyncio.Event()
        channel = await self.call([], core_up=core_up)
        await self.until(lambda: channel.readyState == "open", "the data channel to open", 20)
        channel.send(json.dumps({"type": "session.input_audio.mute", "event_id": "early"}))
        core_up.set()
        await self.received("session.input_audio.muted")
        self.assertEqual(self.kinds()[:2], ["session.started", "session.input_audio.muted"])
        self.assertEqual(self.seen[1]["client_event_id"], "early")
        self.assertNotIn("error", self.kinds())
