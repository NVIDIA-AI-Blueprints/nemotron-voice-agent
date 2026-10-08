# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

import asyncio
import base64
import json
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.turns.user_start import ExternalUserTurnStartStrategy
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from examples.frontend_backend_live.cascade.engine import CascadeLiveEngine
from examples.frontend_backend_live.config_manager.schema import (
    LiveConfig,
    ModelEndpoint,
    ReliabilityConfig,
    RoleConfig,
)
from examples.frontend_backend_live.delegation.tasks import backend_task_for
from examples.frontend_backend_live.engine import factory as live_engine
from live.gateway import LiveGateway, LiveSettings
from live.protocol import SessionConfig
from tests.unit.live_helpers import FakeSTT, FakeTTS, FakeWebSocket
from tests.unit.test_frontend_backend_live import FakeChat, _decision_chunks
from tests.unit.test_frontend_backend_live_translation import chunk, tool_fragment

ENDPOINT = ModelEndpoint(model="org/org/model", base_url="http://localhost:1/v1")
LIVE_CONFIG = LiveConfig(
    frontend=RoleConfig("chat-completions", "llm"),
    backend=RoleConfig("chat-completions", "thinker-llm"),
    reliability=ReliabilityConfig(provider_retries=0, tool_timeout_seconds=10.0, api_timeout_seconds=10.0),
)
STATIC_BACKEND = "Static backend prompt from config."
MENU_TOOL = {
    "type": "function",
    "name": "get_menu",
    "description": "Look up the menu.",
    "parameters": {"type": "object", "properties": {"category": {"type": "string"}}},
}
RESPONSES_SESSION = {
    "model": "live-1",
    "instructions": "You are a kiosk voice assistant. Delegate every request.",
    "delegation": {
        "type": "responses",
        "responses": {"model": "org/org/model", "instructions": "Look things up.", "tools": [MENU_TOOL]},
    },
}


def tool_call_round(call_id="call_1", arguments='{"category":"tea"}'):
    return [
        chunk(tool_calls=[tool_fragment(0, call_id, "get_menu", arguments[:8])]),
        chunk(tool_calls=[tool_fragment(0, arguments=arguments[8:])], finish="tool_calls"),
    ]


def result_item(call_id, output="{}"):
    return {
        "type": "response.item.create",
        "item": {"type": "function_call_output", "call_id": call_id, "output": output},
    }


def spoken_words(ws):
    return " ".join(e["delta"].strip() for e in ws.from_server if e["type"] == "session.output_transcript.delta")


SILENCE_100MS = base64.b64encode(bytes(4800)).decode()


class EngineHarness(unittest.IsolatedAsyncioTestCase):
    def feed_silence(self, tenths):
        """Send caller audio so the input media clock advances (append acknowledgments wait for it)."""
        for _ in range(tenths):
            self.ws.send({"type": "session.input_audio.append", "audio": SILENCE_100MS})

    async def serve(self, session, script):
        """Run the real gateway with a real engine, scripted models and fake speech services."""
        self.chat = FakeChat(*script)
        self.stt, self.tts = FakeSTT(), FakeTTS()
        self.engines: list[CascadeLiveEngine] = []

        async def factory(config, transport, connection):
            engine = CascadeLiveEngine(
                config,
                transport,
                connection,
                live_config=LIVE_CONFIG,
                frontend_endpoint=ENDPOINT,
                backend_endpoint=ENDPOINT,
                instructions=config.instructions or "You are a kiosk assistant.",
                backend_instructions=STATIC_BACKEND,
                stt=self.stt,
                tts=self.tts,
                user_params=LLMUserAggregatorParams(
                    user_turn_strategies=UserTurnStrategies(
                        start=[ExternalUserTurnStartStrategy()], stop=[ExternalUserTurnStopStrategy()]
                    )
                ),
                frontend_client=self.chat,
                backend_client=self.chat,
            )
            self.engines.append(engine)
            return engine

        self.gateway = LiveGateway(factory, LiveSettings(max_session_seconds=60))
        self.ws = FakeWebSocket()
        self.server = asyncio.create_task(self.gateway.run_primary_websocket(self.ws))
        self.addAsyncCleanup(self.stop)
        self.ws.send({"type": "session.start", "event_id": "start_1", "session": session})
        return await self.ws.wait_for("session.started")

    async def stop(self):
        self.ws.send({"type": "session.close"})
        try:
            async with asyncio.timeout(10):
                await self.server
        except (TimeoutError, asyncio.CancelledError):
            self.server.cancel()


class ClientToolsEndToEndTests(EngineHarness):
    """The live-session loop with the client executing the backend's tools."""

    async def test_a_delegated_request_runs_the_clients_tool_and_speaks_the_answer(self):
        started = await self.serve(
            RESPONSES_SESSION,
            [
                _decision_chunks('{"action":"delegate","speech":"Let me check."}'),
                tool_call_round(),
                [chunk("We have Earl Grey and Matcha.", finish="stop")],
                _decision_chunks('{"action":"speak","speech":"We have Earl Grey and Matcha."}'),
            ],
        )
        self.assertEqual(started["client_event_id"], "start_1")
        await self.stt.say("What teas do you have?")

        transcript = await self.ws.wait_for("session.input_transcript.delta")
        self.assertEqual(transcript["delta"].strip(), "What teas do you have?")
        created = await self.ws.wait_for("session.delegation.created")
        self.assertEqual(created["delegation"]["target"], "responses")
        self.assertTrue(created["delegation"]["response_id"].startswith("resp_"))

        # The client sees the function call as it completes, then runs it and continues.
        call = await self.ws.wait_for(
            "response.event",
            where=lambda e: (
                e["event"]["type"] == "response.output_item.done" and e["event"]["item"]["type"] == "function_call"
            ),
        )
        item = call["event"]["item"]
        self.assertEqual((item["name"], json.loads(item["arguments"])), ("get_menu", {"category": "tea"}))
        completed = await self.ws.wait_for("response.event", where=lambda e: e["event"]["type"] == "response.completed")
        self.assertEqual(completed["event"]["response"]["output"], [])  # lifecycle snapshots are redacted
        self.ws.send(
            {
                "type": "response.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": item["call_id"],
                    "output": '{"teas": ["Earl Grey"]}',
                },
            }
        )
        self.ws.send({"type": "response.create", "event_id": "go"})

        await self.ws.wait_for(
            "session.output_transcript.delta", where=lambda e: e["delta"].strip().startswith("Matcha")
        )
        spoken = " ".join(
            e["delta"].strip() for e in self.ws.from_server if e["type"] == "session.output_transcript.delta"
        )
        self.assertEqual(spoken, "Let me check. We have Earl Grey and Matcha.")
        # the second backend round received the client's result
        tool_message = next(m for m in self.chat.requests[2]["messages"] if m["role"] == "tool")
        self.assertEqual(json.loads(tool_message["content"]), {"teas": ["Earl Grey"]})
        self.assertEqual(self.chat.requests[2]["tools"][0]["function"]["name"], "get_menu")
        # a steady stream of audio events, with real reply audio among silence
        audio = [e for e in self.ws.from_server if e["type"] == "session.output_audio.delta"]
        self.assertGreater(len(audio), 3)
        self.assertNotIn("event_id", audio[0])

    async def test_the_backend_runs_on_its_static_prompt_when_the_session_supplies_none(self):
        session = {**RESPONSES_SESSION, "delegation": {"type": "responses", "responses": {"model": "org/org/model"}}}
        task = backend_task_for(SessionConfig.model_validate(session), "fallback", STATIC_BACKEND)
        self.assertEqual(task.instructions, STATIC_BACKEND)
        explicit = backend_task_for(SessionConfig.model_validate(RESPONSES_SESSION), "fallback", STATIC_BACKEND)
        self.assertEqual(explicit.instructions, "Look things up.")
        client = {"model": "live-1"}
        self.assertIsNone(backend_task_for(SessionConfig.model_validate(client), "fallback", STATIC_BACKEND))

    async def test_a_truncated_backend_round_is_retried_and_its_failure_is_not_sent_to_the_client(self):
        await self.serve(
            RESPONSES_SESSION,
            [
                _decision_chunks('{"action":"delegate","speech":""}'),
                [chunk("Partial thought", finish="length")],  # hits the token limit
                [chunk("Your flight is on time.", finish="stop")],  # the retried round
                _decision_chunks('{"action":"speak","speech":"Your flight is on time."}'),
            ],
        )
        await self.stt.say("Is my flight on time?")
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "time." in e["delta"])
        events = [e["event"]["type"] for e in self.ws.from_server if e["type"] == "response.event"]
        self.assertNotIn("response.incomplete", events)
        self.assertEqual(events.count("response.completed"), 1)
        self.assertEqual(len(self.chat.requests), 4)

    async def test_response_create_before_every_result_is_returned_is_refused(self):
        await self.serve(
            RESPONSES_SESSION,
            [
                _decision_chunks('{"action":"delegate","speech":""}'),
                [
                    chunk(
                        tool_calls=[tool_fragment(0, "a", "get_menu", "{}"), tool_fragment(1, "b", "get_menu", "{}")]
                    ),
                    chunk(finish="tool_calls"),
                ],
            ],
        )
        await self.stt.say("Two lookups please.")
        await self.ws.wait_for("response.event", where=lambda e: e["event"]["type"] == "response.completed")
        self.ws.send(
            {"type": "response.item.create", "item": {"type": "function_call_output", "call_id": "a", "output": "{}"}}
        )
        self.ws.send({"type": "response.create", "event_id": "early"})
        error = await self.ws.wait_for("error")
        self.assertEqual((error["error"]["code"], error["error"]["client_event_id"]), ("missing_tool_results", "early"))

    async def test_unknown_and_conflicting_results_are_errors(self):
        await self.serve(
            RESPONSES_SESSION,
            [_decision_chunks('{"action":"delegate","speech":""}'), tool_call_round("known")],
        )
        await self.stt.say("Look it up.")
        await self.ws.wait_for("response.event", where=lambda e: e["event"]["type"] == "response.completed")
        self.ws.send(
            {
                "type": "response.item.create",
                "item": {"type": "function_call_output", "call_id": "nope", "output": "{}"},
            }
        )
        self.assertEqual((await self.ws.wait_for("error"))["error"]["param"], "item.call_id")
        known = {"type": "function_call_output", "call_id": "known", "output": "first"}
        self.ws.send({"type": "response.item.create", "item": known})
        self.ws.send({"type": "response.item.create", "item": {**known, "output": "second"}})
        errors = [e for e in self.ws.from_server if e["type"] == "error"]
        for _ in range(100):
            errors = [e for e in self.ws.from_server if e["type"] == "error"]
            if len(errors) == 2:
                break
            await asyncio.sleep(0.02)
        self.assertEqual(errors[1]["error"]["message"], "Conflicting duplicate tool result")

    async def test_a_direct_reply_is_spoken_without_a_delegation(self):
        await self.serve(RESPONSES_SESSION, [_decision_chunks('{"action":"speak","speech":"Hello there."}')])
        await self.stt.say("Hi.")
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "there." in e["delta"])
        self.assertNotIn("session.delegation.created", self.ws.kinds())
        self.assertEqual(len(self.chat.requests), 1)

    async def test_the_clients_instructions_and_history_reach_the_talker(self):
        session = {
            **RESPONSES_SESSION,
            "input": [
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Kiosk 7."}]}
            ],
        }
        await self.serve(session, [_decision_chunks('{"action":"speak","speech":"Hi."}')])
        await self.stt.say("Hello.")
        await self.ws.wait_for("session.output_transcript.delta")
        system = self.chat.requests[0]["messages"][0]["content"]
        self.assertIn("You are a kiosk voice assistant. Delegate every request.", system)
        self.assertIn("Kiosk 7.", system)

    async def test_mute_stops_input_audio_without_ending_the_session(self):
        await self.serve(RESPONSES_SESSION, [])
        self.ws.send({"type": "session.input_audio.mute", "event_id": "m"})
        await self.ws.wait_for("session.input_audio.muted")
        self.assertTrue(self.engines[0].gate.muted)
        self.ws.send({"type": "session.input_audio.unmute", "event_id": "u"})
        await self.ws.wait_for("session.input_audio.unmuted")
        self.assertFalse(self.engines[0].gate.muted)

    async def test_session_update_changes_the_backend_for_later_rounds(self):
        await self.serve(RESPONSES_SESSION, [])
        patch = {"delegation": {"responses": {"instructions": "New backend rules.", "max_output_tokens": 77}}}
        self.ws.send({"type": "session.update", "event_id": "u", "session": patch})
        await self.ws.wait_for("session.updated")
        task = self.engines[0].cascade.worker.task
        self.assertEqual((task.instructions, task.max_output_tokens), ("New backend rules.", 77))
        self.ws.send(
            {"type": "session.update", "session": {"delegation": {"responses": {"model": "unqualified-name"}}}}
        )
        error = await self.ws.wait_for("error")
        self.assertEqual(error["error"]["param"], "session.delegation.responses.model")

    async def test_instruction_append_interrupts_and_acts_on_the_new_direction(self):
        await self.serve(RESPONSES_SESSION, [_decision_chunks('{"action":"speak","speech":"Understood."}')])
        self.ws.send(
            {
                "type": "session.instructions.append",
                "event_id": "i",
                "delegation_id": None,
                "content": "Greet the caller.",
            }
        )
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "Understood" in e["delta"])
        self.assertIn("Greet the caller.", self.chat.requests[0]["messages"][0]["content"])
        self.feed_silence(6)
        ack = await self.ws.wait_for("session.instructions.appended", timeout=3)
        self.assertEqual(ack["client_event_id"], "i")

    async def test_thinking_append_is_silent_context_for_later_turns(self):
        await self.serve(RESPONSES_SESSION, [_decision_chunks('{"action":"speak","speech":"Noted."}')])
        self.ws.send({"type": "session.thinking.append", "delegation_id": None, "content": "The caller is a regular."})
        await self.stt.say("Hi.")
        await self.ws.wait_for("session.output_transcript.delta")
        history = [m["content"] for m in self.chat.requests[0]["messages"][1:]]
        self.assertTrue(any("The caller is a regular." in item for item in history))
        self.assertNotIn(
            "session.output_transcript.delta",
            self.ws.kinds()[: self.ws.kinds().index("session.input_transcript.delta")],
        )


class ClientModeEndToEndTests(EngineHarness):
    """The client owns delegated work: the server announces it and phrases what the client reports."""

    CLIENT_SESSION = {"model": "live-1", "instructions": "You are a kiosk voice assistant."}

    async def test_a_delegation_is_announced_and_the_clients_commentary_is_spoken(self):
        await self.serve(
            self.CLIENT_SESSION,
            [
                _decision_chunks('{"action":"delegate","speech":"One moment."}'),
                _decision_chunks('{"action":"speak","speech":"It is sunny today."}'),
            ],
        )
        await self.stt.say("What is the weather?")
        created = await self.ws.wait_for("session.delegation.created")
        self.assertEqual(created["delegation"]["target"], "client")
        delegation_id = created["delegation"]["id"]
        self.ws.send(
            {
                "type": "session.commentary.append",
                "event_id": "c1",
                "delegation_id": delegation_id,
                "content": "Weather: sunny, 72 degrees.",
            }
        )
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "sunny" in e["delta"])
        update = self.chat.requests[1]["messages"][-1]["content"]
        self.assertIn("Weather: sunny, 72 degrees.", update)
        self.assertIn('you already told the caller: "One moment."', update)
        self.assertEqual(
            self.chat.requests[1]["response_format"]["json_schema"]["schema"]["properties"]["action"]["enum"],
            ["speak", "listen"],
        )

    async def test_client_commentary_is_spoken_even_when_the_writer_would_stay_silent(self):
        await self.serve(
            self.CLIENT_SESSION,
            [
                _decision_chunks('{"action":"delegate","speech":""}'),
                _decision_chunks('{"action":"listen","speech":""}'),
            ],
        )
        await self.stt.say("Check something.")
        created = await self.ws.wait_for("session.delegation.created")
        self.ws.send(
            {
                "type": "session.commentary.append",
                "delegation_id": created["delegation"]["id"],
                "content": "Background fact.",
            }
        )
        self.feed_silence(6)
        await self.ws.wait_for("session.commentary.appended", timeout=3)
        await self.ws.wait_for("session.output_transcript.delta", timeout=3)
        self.assertEqual(" ".join(self.tts.spoken), "Background fact.")

    async def test_response_commands_are_refused_in_client_mode(self):
        await self.serve(self.CLIENT_SESSION, [])
        self.ws.send({"type": "response.create"})
        error = await self.ws.wait_for("error")
        self.assertIn("requires Responses delegation", error["error"]["message"])


class WebRTCEndToEndTests(unittest.IsolatedAsyncioTestCase):
    """The real engine over a real WebRTC session: Pipecat's transport on the live connection."""

    async def test_reply_audio_reaches_the_media_track_and_the_final_event_precedes_channel_close(self):
        import httpx
        from aiortc import RTCPeerConnection, RTCSessionDescription
        from aiortc.mediastreams import AudioStreamTrack
        from fastapi import FastAPI

        from live.mount import mount_live

        chat = FakeChat(_decision_chunks('{"action":"speak","speech":"Hello from the kiosk."}'))
        stt, tts = FakeSTT(), FakeTTS()

        async def factory(config, transport, connection):
            return CascadeLiveEngine(
                config,
                transport,
                connection,
                live_config=LIVE_CONFIG,
                frontend_endpoint=ENDPOINT,
                backend_endpoint=ENDPOINT,
                instructions="You are a kiosk assistant.",
                backend_instructions=STATIC_BACKEND,
                stt=stt,
                tts=tts,
                user_params=LLMUserAggregatorParams(
                    user_turn_strategies=UserTurnStrategies(
                        start=[ExternalUserTurnStartStrategy()], stop=[ExternalUserTurnStopStrategy()]
                    )
                ),
                frontend_client=chat,
                backend_client=chat,
            )

        app = FastAPI()
        gateway = mount_live(app, factory)
        client = RTCPeerConnection()
        self.addAsyncCleanup(client.close)
        client.addTrack(AudioStreamTrack())
        channel = client.createDataChannel("")  # some clients use an empty label
        events: asyncio.Queue = asyncio.Queue()
        order: list[str] = []
        closed = asyncio.Event()
        audio_frames: list[int] = []

        @channel.on("message")
        def on_message(raw):
            event = json.loads(raw)
            events.put_nowait(event)
            if event["type"] == "session.closed":
                order.append("final_event")

        @channel.on("close")
        def on_close():
            order.append("channel_closed")
            closed.set()

        @client.on("track")
        def on_track(track):
            async def read():
                try:
                    while True:
                        frame = await track.recv()
                        audio_frames.append(int(max(abs(frame.to_ndarray().astype("int32").ravel()), default=0)))
                except Exception:
                    return

            asyncio.create_task(read())

        async def next_event(kind, timeout=20):
            async with asyncio.timeout(timeout):
                while True:
                    event = await events.get()
                    if event["type"] == kind:
                        return event

        await client.setLocalDescription(await client.createOffer())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            response = await http.post(
                "/v1/live/sessions",
                json={
                    "session": RESPONSES_SESSION,
                    "transport": {"type": "webrtc", "sdp": client.localDescription.sdp},
                },
            )
        self.assertEqual(response.status_code, 201, response.text)
        await client.setRemoteDescription(RTCSessionDescription(sdp=response.json()["transport"]["sdp"], type="answer"))
        await next_event("session.started")
        await stt.say("Hello.")
        await next_event("session.input_transcript.delta")
        words = []
        async with asyncio.timeout(20):
            while len(words) < 4:
                words.append((await next_event("session.output_transcript.delta"))["delta"].strip())
        self.assertEqual(" ".join(words), "Hello from the kiosk.")
        # reply audio (a tone) arrived on the media track, not just silence
        async with asyncio.timeout(10):
            while max(audio_frames, default=0) < 100:
                await asyncio.sleep(0.05)
        channel.send(json.dumps({"type": "session.close", "event_id": "bye"}))
        await next_event("session.closed")
        async with asyncio.timeout(20):
            await closed.wait()
        self.assertEqual(order, ["final_event", "channel_closed"])
        for _ in range(100):
            if not gateway.sessions:
                break
            await asyncio.sleep(0.05)
        self.assertEqual(gateway.sessions, {})


class LifecycleTests(EngineHarness):
    async def test_closing_the_session_ends_with_a_final_event_and_stops_everything(self):
        await self.serve(RESPONSES_SESSION, [])
        self.ws.send({"type": "session.close", "event_id": "bye"})
        closed = await self.ws.wait_for("session.closed")
        self.assertEqual((closed["reason"], closed["client_event_id"]), ("close_requested", "bye"))
        async with asyncio.timeout(10):
            await self.server
        self.assertEqual(self.gateway.sessions, {})
        self.assertIsNone(self.engines[0].runner_task and (None if self.engines[0].runner_task.done() else 1))

    async def test_a_session_that_cannot_start_reports_an_error_and_closes(self):
        self.stt, self.tts = FakeSTT(), FakeTTS()

        async def factory(config, transport, connection):
            raise RuntimeError("no models")

        gateway = LiveGateway(factory, LiveSettings())
        ws = FakeWebSocket()
        task = asyncio.create_task(gateway.run_primary_websocket(ws))
        ws.send({"type": "session.start", "event_id": "s1", "session": RESPONSES_SESSION})
        async with asyncio.timeout(10):
            await task
        error = ws.from_server[0]["error"]
        self.assertEqual((error["code"], error["client_event_id"]), ("server_error", "s1"))
        self.assertTrue(ws.closed)
        self.assertEqual(gateway.sessions, {})


def patched_services(entries, config):
    """Stand in for the registry: ``entries`` by slot name, ``config`` as the example's ``config.yaml``."""
    lookup = lambda slot, _: entries.get(slot, {})  # noqa: E731
    return (
        patch.object(live_engine.examples_registry, "find"),
        patch.object(live_engine, "set_service_context"),
        patch.object(live_engine, "load_live_config", return_value=config),
        patch.object(live_engine, "load_service_entry", side_effect=lookup),
        patch("examples.frontend_backend_live.models.endpoints.load_service_entry", side_effect=lookup),
        patch.object(live_engine, "build_stt"),
        patch.object(live_engine, "build_tts"),
    )


class ContextMeterTests(unittest.TestCase):
    def test_the_cascade_reports_how_much_conversation_its_talker_holds(self):
        from types import SimpleNamespace

        from tests.unit.test_frontend_backend_live import ListConversation

        engine = SimpleNamespace(cascade=None)
        self.assertIsNone(CascadeLiveEngine.context_tokens(engine))
        engine.cascade = SimpleNamespace(conversation=ListConversation([{"role": "user", "content": "hi"}]))
        small = CascadeLiveEngine.context_tokens(engine)
        engine.cascade = SimpleNamespace(conversation=ListConversation([{"role": "user", "content": "hi " * 500}]))
        self.assertGreater(CascadeLiveEngine.context_tokens(engine), small + 300)


class FactoryTests(unittest.IsolatedAsyncioTestCase):
    async def create(self, entries, config):
        with ExitStack() as stack:
            for patcher in patched_services(entries, config):
                stack.enter_context(patcher)
            return await live_engine.create_engine(SessionConfig.model_validate({"model": "live-1"}), "websocket", None)

    async def test_each_role_resolves_its_model_from_the_slot_config_names(self):
        entries = {
            "front": {"model_id": "org/front-model", "base_url": "http://front/v1"},
            "back": {"model_id": "org/back-model", "base_url": "http://back/v1"},
        }
        config = LiveConfig(
            frontend=RoleConfig("chat-completions", "front", "talker"),
            backend=RoleConfig("chat-completions", "back", "thinker"),
        )
        engine = await self.create(entries, config)
        self.assertEqual(engine.frontend_endpoint.model, "org/front-model")
        self.assertEqual(engine.backend_endpoint.base_url, "http://back/v1")
        self.assertIn("Remy", engine.instructions)
        self.assertIn("backend for an assistant", engine.backend_instructions)

    async def test_the_backend_inherits_what_its_entry_leaves_unset_from_the_frontend(self):
        entries = {"front": {"model_id": "org/front-model", "base_url": "http://front/v1"}, "back": {}}
        config = LiveConfig(
            frontend=RoleConfig("chat-completions", "front", "talker"),
            backend=RoleConfig("chat-completions", "back", "thinker"),
        )
        engine = await self.create(entries, config)
        self.assertEqual(
            (engine.backend_endpoint.model, engine.backend_endpoint.base_url), ("org/front-model", "http://front/v1")
        )

    async def test_a_realtime_frontend_that_resolved_to_a_chat_endpoint_is_refused_with_the_reason(self):
        # What happens when a custom entry is not offered by the active recipe: the slot falls back to its default.
        entries = {
            "front": {"model_id": "nvidia/default-chat", "base_url": "https://integrate.api.nvidia.com/v1"},
            "back": {"model_id": "org/back-model", "base_url": "http://back/v1"},
        }
        config = LiveConfig(
            frontend=RoleConfig("realtime", "front", "talker"),
            backend=RoleConfig("chat-completions", "back", "thinker"),
        )
        with self.assertRaisesRegex(
            ValueError, "slot 'front' resolved to 'https://integrate.api.nvidia.com/v1'.*SERVICE_RECIPE"
        ):
            await self.create(entries, config)

    def test_describe_warns_when_the_realtime_frontend_is_not_on_a_websocket_url(self):
        entries = {"front": {"model_id": "m", "base_url": "https://chat.example.com/v1"}, "back": {}}
        config = LiveConfig(
            frontend=RoleConfig("realtime", "front", "talker"),
            backend=RoleConfig("chat-completions", "back", "thinker"),
        )
        with ExitStack() as stack:
            for patcher in patched_services(entries, config):
                stack.enter_context(patcher)
            warnings = live_engine.describe()["warnings"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("ws:// or wss://", warnings[0])

    async def test_the_realtime_provider_builds_the_realtime_engine(self):
        from examples.frontend_backend_live.realtime.engine import RealtimeLiveEngine

        entries = {
            "front": {"model_id": "remote-realtime", "base_url": "wss://remote.example/v1/realtime"},
            "back": {"model_id": "org/back-model", "base_url": "http://back/v1"},
        }
        config = LiveConfig(
            frontend=RoleConfig("realtime", "front", "talker"),
            backend=RoleConfig("chat-completions", "back", "thinker"),
        )
        engine = await self.create(entries, config)
        self.assertIsInstance(engine, RealtimeLiveEngine)
        self.assertEqual(engine.frontend_endpoint.model, "remote-realtime")
        self.assertEqual(engine.backend_endpoint.model, "org/back-model")
        self.assertIn("Remy", engine.instructions)


if __name__ == "__main__":
    unittest.main()
