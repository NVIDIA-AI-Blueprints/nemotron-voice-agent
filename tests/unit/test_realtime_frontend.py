# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D107

"""A remote realtime model as the frontend: the live protocol on one side, the realtime protocol on the other."""

import asyncio
import base64
import json
import unittest

from examples.frontend_backend_live.config_manager.schema import (
    LiveConfig,
    ModelEndpoint,
    ReliabilityConfig,
    RoleConfig,
)
from examples.frontend_backend_live.realtime.engine import RealtimeLiveEngine, realtime_url
from live.gateway import LiveGateway, LiveSettings
from tests.unit.live_helpers import FakeWebSocket
from tests.unit.realtime_fake import FakeRealtimeServer
from tests.unit.test_frontend_backend_live import FakeChat
from tests.unit.test_frontend_backend_live_engine import MENU_TOOL, STATIC_BACKEND, result_item, tool_call_round
from tests.unit.test_frontend_backend_live_translation import chunk

LIVE_CONFIG = LiveConfig(
    frontend=RoleConfig("realtime", "llm"),
    backend=RoleConfig("chat-completions", "thinker-llm"),
    reliability=ReliabilityConfig(provider_retries=0, tool_timeout_seconds=10.0, api_timeout_seconds=10.0),
)
INSTRUCTIONS = "You are a kiosk voice assistant."
RESPONSES_SESSION = {
    "model": "live-1",
    "instructions": INSTRUCTIONS,
    "input": [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Earlier question."}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Earlier answer."}]},
    ],
    "delegation": {
        "type": "responses",
        "responses": {"model": "org/org/model", "instructions": "Look things up.", "tools": [MENU_TOOL]},
    },
}
CLIENT_SESSION = {"model": "live-1", "instructions": INSTRUCTIONS}


class RealtimeHarness(unittest.IsolatedAsyncioTestCase):
    async def serve(self, session, fake, backend_script=()):
        self.fake, self.chat = fake, FakeChat(*backend_script)
        self.engines: list[RealtimeLiveEngine] = []

        async def factory(config, transport, connection):
            engine = RealtimeLiveEngine(
                config,
                transport,
                connection,
                live_config=LIVE_CONFIG,
                frontend_endpoint=ModelEndpoint(
                    model="remote-realtime", base_url=fake.url, api_key="remote-key", extra_params={}
                ),
                backend_endpoint=ModelEndpoint(model="org/org/model", base_url="http://localhost:1/v1"),
                instructions=config.instructions or INSTRUCTIONS,
                backend_instructions=STATIC_BACKEND,
                backend_client=self.chat,
            )
            self.engines.append(engine)
            return engine

        self.gateway = LiveGateway(factory, LiveSettings(max_session_seconds=60))
        self.ws = FakeWebSocket()
        self.server = asyncio.create_task(self.gateway.run_primary_websocket(self.ws))
        self.addAsyncCleanup(self.stop)
        self.ws.send({"type": "session.start", "event_id": "start_1", "session": session})

    async def stop(self):
        self.ws.send({"type": "session.close"})
        try:
            async with asyncio.timeout(10):
                await self.server
        except (TimeoutError, asyncio.CancelledError, RuntimeError):
            self.server.cancel()

    def spoken(self):
        return " ".join(
            e["delta"].strip() for e in self.ws.from_server if e["type"] == "session.output_transcript.delta"
        )


class SessionTests(RealtimeHarness):
    async def test_the_remote_session_is_configured_from_the_live_session(self):
        async with FakeRealtimeServer() as fake:
            await self.serve(RESPONSES_SESSION, fake)
            started = await self.ws.wait_for("session.started")
            self.assertEqual(started["client_event_id"], "start_1")
            update = await fake.wait_for("session.update")
            session = update["session"]
            self.assertEqual(session["model"], "remote-realtime")
            self.assertTrue(session["instructions"].startswith(INSTRUCTIONS))
            self.assertIn("delegate", session["instructions"])
            self.assertEqual([t["name"] for t in session["tools"]], ["delegate"])
            self.assertEqual(session["audio"]["output"]["voice"], "marin")
            self.assertEqual(session["audio"]["input"]["format"], {"type": "audio/pcm", "rate": 24000})
            self.assertEqual(fake.headers["Authorization"], "Bearer remote-key")
            self.assertIn("model=remote-realtime", fake.path)

    async def test_prior_messages_seed_the_remote_conversation_after_it_is_configured(self):
        async with FakeRealtimeServer() as fake:
            await self.serve(RESPONSES_SESSION, fake)
            await self.ws.wait_for("session.started")
            await fake.wait_for("conversation.item.create", where=lambda e: e["item"]["role"] == "assistant")
            kinds = fake.kinds()
            self.assertLess(kinds.index("session.update"), kinds.index("conversation.item.create"))
            self.assertEqual([i["role"] for i in fake.items()], ["user", "assistant"])
            self.assertEqual(fake.items()[1]["content"][0], {"type": "output_text", "text": "Earlier answer."})

    async def test_a_server_that_rejects_the_session_fails_the_start(self):
        async with FakeRealtimeServer(session_error={"code": "invalid_model", "message": "No such model."}) as fake:
            await self.serve(CLIENT_SESSION, fake)
            error = await self.ws.wait_for("error")
            self.assertEqual(error["error"]["code"], "pipeline_error")
            self.assertNotIn("session.started", self.ws.kinds())
            with self.assertRaisesRegex(RuntimeError, "No such model"):
                await asyncio.wait_for(self.server, 10)

    async def test_losing_the_remote_connection_closes_the_session(self):
        async with FakeRealtimeServer() as fake:
            await self.serve(CLIENT_SESSION, fake)
            await self.ws.wait_for("session.started")
            await fake.connected()
            await fake.ws.close()
            closed = await self.ws.wait_for("session.closed")
            self.assertEqual(closed["reason"], "connection_lost")

    def test_the_model_query_parameter_is_added_once(self):
        self.assertEqual(realtime_url("wss://h/v1/realtime", "m"), "wss://h/v1/realtime?model=m")
        self.assertEqual(realtime_url("wss://h/v1/realtime?model=x", "m"), "wss://h/v1/realtime?model=x")


class SpeechTests(RealtimeHarness):
    async def test_the_callers_words_and_the_models_speech_become_live_events(self):
        async with FakeRealtimeServer() as fake:
            await self.serve(CLIENT_SESSION, fake)
            await self.ws.wait_for("session.started")
            await fake.connected()
            await fake.hear("Hello there.")
            heard = await self.ws.wait_for("session.input_transcript.delta")
            self.assertEqual(heard["delta"].strip(), "Hello there.")
            await fake.speak("Welcome to the cafe today.")
            await self.ws.wait_for("session.output_audio.delta")
            await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "today." in e["delta"])
            # Sub-word transcript pieces are re-assembled into whole words.
            self.assertEqual(self.spoken(), "Welcome to the cafe today.")

    async def test_speech_the_server_sends_after_the_caller_barges_in_is_dropped(self):
        async with FakeRealtimeServer() as fake:
            await self.serve(CLIENT_SESSION, fake)
            await self.ws.wait_for("session.started")
            await fake.connected()
            response_id = await fake.speak("We have several teas on the menu", finish=False)
            await self.ws.wait_for("session.output_transcript.delta")
            await fake.send({"type": "input_audio_buffer.speech_started"})
            await asyncio.sleep(0.2)
            seen = len(self.ws.from_server)
            await fake.speak("and some pastries too.", response_id=response_id)
            await asyncio.sleep(0.3)
            late = self.ws.from_server[seen:]
            # The wire carries silence continuously; only speech (transcript, non-zero audio) must stop.
            self.assertEqual([e for e in late if e["type"] == "session.output_transcript.delta"], [])
            tones = [e for e in late if e["type"] == "session.output_audio.delta" and any(base64.b64decode(e["delta"]))]
            self.assertEqual(tones, [])


class DelegationTests(RealtimeHarness):
    async def test_a_delegate_call_runs_the_backend_and_the_result_is_given_to_the_model(self):
        async with FakeRealtimeServer() as fake:
            await self.serve(
                RESPONSES_SESSION,
                fake,
                [tool_call_round(), [chunk("We have Earl Grey and Matcha.", finish="stop")]],
            )
            await self.ws.wait_for("session.started")
            await fake.connected()
            await fake.hear("What teas do you have?")
            await fake.call("delegate", {"request": "List the teas."}, lead_in="Let me check.")

            # The call is answered at once, so the model keeps talking, then asked to continue.
            output = await fake.wait_for(
                "conversation.item.create", where=lambda e: e["item"]["type"] == "function_call_output"
            )
            self.assertEqual(output["item"]["call_id"], "call_1")
            self.assertEqual(json.loads(output["item"]["output"])["status"], "working")
            await fake.wait_for("response.create")

            created = await self.ws.wait_for("session.delegation.created")
            self.assertEqual(created["delegation"]["target"], "responses")
            call = await self.ws.wait_for(
                "response.event",
                where=lambda e: (
                    e["event"]["type"] == "response.output_item.done" and e["event"]["item"]["type"] == "function_call"
                ),
            )
            item = call["event"]["item"]
            self.assertEqual(item["name"], "get_menu")
            self.ws.send(result_item(item["call_id"], '{"teas": ["Earl Grey"]}'))
            self.ws.send({"type": "response.create"})

            # The verified result reaches the model as a system message, with a request to speak it.
            result = await fake.wait_for(
                "conversation.item.create",
                where=lambda e: e["item"].get("role") == "system" and "Verified backend result" in str(e["item"]),
            )
            self.assertIn("Earl Grey and Matcha", result["item"]["content"][0]["text"])
            responses = [e for e in fake.events if e["type"] == "response.create"]
            self.assertTrue(any("result" in e.get("response", {}).get("instructions", "") for e in responses))

    async def test_the_delegate_sees_the_spoken_conversation_not_only_the_request(self):
        async with FakeRealtimeServer() as fake:
            await self.serve(RESPONSES_SESSION, fake, [[chunk("Done.", finish="stop")]])
            await self.ws.wait_for("session.started")
            await fake.connected()
            await fake.hear("I'd like a large Earl Grey.")
            await fake.speak("Sure thing.")
            await fake.call("delegate", {"request": "Add a large Earl Grey."})
            await fake.wait_for("conversation.item.create", where=lambda e: e["item"]["type"] == "function_call_output")
            async with asyncio.timeout(10):
                while not self.chat.requests:
                    await asyncio.sleep(0.02)
            sent = json.dumps(self.chat.requests[0]["messages"])
            self.assertIn("I'd like a large Earl Grey.", sent)
            self.assertIn("Sure thing.", sent)
            self.assertIn("Current request: Add a large Earl Grey.", sent)

    async def test_in_client_mode_the_delegation_is_announced_and_the_clients_commentary_is_voiced(self):
        async with FakeRealtimeServer() as fake:
            await self.serve(CLIENT_SESSION, fake)
            await self.ws.wait_for("session.started")
            await fake.connected()
            await fake.call("delegate", {"request": "Check the order."})
            created = await self.ws.wait_for("session.delegation.created")
            self.assertEqual(created["delegation"]["target"], "client")
            await fake.wait_for("conversation.item.create", where=lambda e: e["item"]["type"] == "function_call_output")
            await fake.wait_for("response.create")
            before = len(fake.events)
            self.ws.send(
                {
                    "type": "session.commentary.append",
                    "delegation_id": created["delegation"]["id"],
                    "content": "The order total is $12.",
                }
            )
            note = await fake.wait_for(
                "conversation.item.create", after=before, where=lambda e: e["item"].get("role") == "system"
            )
            self.assertIn("The order total is $12.", note["item"]["content"][0]["text"])
            await fake.wait_for("response.create", after=before)

    async def test_thinking_is_added_without_asking_for_speech(self):
        async with FakeRealtimeServer() as fake:
            await self.serve(CLIENT_SESSION, fake)
            await self.ws.wait_for("session.started")
            await fake.connected()
            before = len(fake.events)
            self.ws.send(
                {"type": "session.thinking.append", "delegation_id": None, "content": "The caller is a regular."}
            )
            note = await fake.wait_for("conversation.item.create", after=before)
            self.assertIn("The caller is a regular.", note["item"]["content"][0]["text"])
            await asyncio.sleep(0.2)
            self.assertNotIn("response.create", [e["type"] for e in fake.events[before:]])

    async def test_an_instruction_interrupts_speech_and_asks_for_a_new_reply(self):
        async with FakeRealtimeServer() as fake:
            await self.serve(CLIENT_SESSION, fake)
            await self.ws.wait_for("session.started")
            await fake.connected()
            await fake.speak("We have several teas on the menu", finish=False)
            await self.ws.wait_for("session.output_transcript.delta")
            before = len(fake.events)
            self.ws.send({"type": "session.instructions.append", "delegation_id": None, "content": "Speak formally."})
            await fake.wait_for("response.cancel", after=before)
            note = await fake.wait_for("conversation.item.create", after=before)
            self.assertIn("Speak formally.", note["item"]["content"][0]["text"])
            await fake.wait_for("response.create", after=before)

    async def test_two_results_that_arrive_together_are_both_spoken(self):
        # The server refuses a response request while one is active, so the second must wait for the first.
        async with FakeRealtimeServer(strict=True, reply="Noted.") as fake:
            await self.serve(RESPONSES_SESSION, fake)
            await self.ws.wait_for("session.started")
            await fake.connected()
            response_id = await fake.speak("One moment", finish=False)  # a reply is in progress when both arrive
            await asyncio.sleep(0.2)
            task = type("Task", (), {"id": "item_1"})()
            engine = self.engines[0]
            both = asyncio.gather(
                engine.on_result("First result.", task, None, newer_requests=[]),
                engine.on_result("Second result.", task, None, newer_requests=[]),
            )
            await asyncio.sleep(0.2)
            await fake.finish("One moment", response_id)  # both are released at the same moment
            await both
            async with asyncio.timeout(10):
                while sum("Noted." in e.get("delta", "") for e in self.ws.from_server) < 2:
                    await asyncio.sleep(0.05)
            self.assertEqual(fake.refused, 0)

    async def test_a_request_the_server_refused_as_busy_is_repeated_once_it_is_idle(self):
        # The server can be busy with a reply of its own that the client has not heard about yet.
        async with FakeRealtimeServer(strict=True, reply="Noted.") as fake:
            await self.serve(RESPONSES_SESSION, fake)
            await self.ws.wait_for("session.started")
            await fake.connected()
            fake.active = True  # a response the client does not know about
            task = type("Task", (), {"id": "item_1"})()
            await self.engines[0].on_result("The result.", task, None, newer_requests=[])
            await fake.wait_for("response.create")
            await asyncio.sleep(0.3)
            fake.active = False
            await fake.send({"type": "response.created", "response": {"id": "resp_x"}})
            await fake.send({"type": "response.done", "response": {"id": "resp_x", "status": "completed"}})
            async with asyncio.timeout(10):
                while not any("Noted." in e.get("delta", "") for e in self.ws.from_server):
                    await asyncio.sleep(0.05)
            self.assertEqual(fake.refused, 1)

    async def test_a_result_is_spoken_knowing_the_newer_requests_in_progress(self):
        async with FakeRealtimeServer() as fake:
            await self.serve(RESPONSES_SESSION, fake)
            await self.ws.wait_for("session.started")
            await fake.connected()
            task = type("T", (), {"id": "item_1"})()
            await self.engines[0].on_result("Old answer.", task, None, newer_requests=["make it Friday"])
            request = await fake.wait_for("response.create")
            self.assertIn("make it Friday", request["response"]["instructions"])
            self.assertIn("updating", request["response"]["instructions"])


if __name__ == "__main__":
    unittest.main()
