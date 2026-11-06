# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

import asyncio
import base64
import json
import unittest
from functools import partial
from unittest.mock import patch

import httpx
from aiortc import RTCPeerConnection
from aiortc.mediastreams import AudioStreamTrack
from fastapi import FastAPI
from loguru import logger
from pipecat.frames.frames import InputAudioRawFrame, LLMTextFrame, MetricsFrame, VADUserStoppedSpeakingFrame
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.observers.base_observer import BaseObserver
from pipecat.processors.frame_processor import FrameProcessor

from examples.frontend_backend_live.adapter.adapter import LIVE_AUDIO_IN_RATE, CascadeAdapter
from examples.frontend_backend_live.adapter.connection import LiveWebRTCConnection
from examples.frontend_backend_live.delegation_core.im.input import BackendTask, compose, settings_for
from examples.frontend_backend_live.heard_log import Heard
from examples.frontend_backend_live.prompts import load_prompts
from examples.shared import audio_recorder
from gateways.live.gateway import LiveGateway, LiveSettings, create_live_router
from gateways.live.gateway.websocket import run_primary_websocket
from gateways.live.ports import SessionParts
from gateways.live.protocol import SessionConfig
from gateways.live.server_delegation.passthrough import COMPOSED_SOURCE, request_settings
from tests.unit.cascade_helpers import (
    ENDPOINT,
    LIVE_CONFIG,
    MENU_TOOL,
    STATIC_BACKEND,
    FakeSTT,
    FakeTTS,
    _decision_chunks,
    cascade_parts,
    chat_core,
    core_hook,
    tool_call_round,
)
from tests.unit.core_helpers import CORE_BLOCK, FakeChat, chunk, tool_fragment
from tests.unit.frontend_helpers import ScriptedChat, ToneTTS, decision
from tests.unit.live_helpers import FakeWebSocket

# The example's prompt file: its prompt set, whose words the cascade writes in these whole-call tests, and its lines.
LOADED = load_prompts()
PROMPTS, LINES = LOADED.cascade, LOADED.lines

RESPONSES_SESSION = {
    "model": "live-1",
    "instructions": "You are a kiosk voice assistant. Delegate every request.",
    "delegation": {
        "type": "responses",
        "responses": {"model": "org/org/model", "instructions": "Look things up.", "tools": [MENU_TOOL]},
    },
}

# What the benchmark's Live client starts a session with: its own agent model, its policy as the backend's instructions,
# its function tools, one call per round and an effort.
BENCHMARK_SESSION = {
    "model": "live-1",
    "instructions": "You are a customer service agent. Delegate every request.",
    "delegation": {
        "type": "responses",
        "responses": {
            "model": "benchmark/agent-model",
            "instructions": "Policy.",
            "tools": [MENU_TOOL],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "reasoning": {"effort": "low"},
        },
    },
}

SILENCE_100MS = base64.b64encode(bytes(4800)).decode()


class EngineHarness(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # The shared audio recorder's switches at their defaults, whatever the environment sets: nothing records.
        self.enterContext(patch.object(audio_recorder, "ENABLE_ASR_DUMP", False))
        self.enterContext(patch.object(audio_recorder, "ENABLE_TTS_DUMP", False))

    def feed_silence(self, tenths):
        """Send caller audio so the input media clock advances (append acknowledgments wait for it)."""
        for _ in range(tenths):
            self.ws.send({"type": "session.input_audio.append", "audio": SILENCE_100MS})

    async def serve(
        self,
        session,
        script,
        *,
        live_config=LIVE_CONFIG,
        delegation=None,
        core=CORE_BLOCK,
        models=(),
        tts=FakeTTS,
        websocket=FakeWebSocket,
        early=(),
    ):
        """Run the real gateway with a real engine, scripted models and fake speech services.

        ``live_config`` is the cascade's block, ``core`` the core's settings, ``models`` the models its catalog lists
        beside the endpoint's, and ``delegation`` sets keys of its ``delegation`` block; ``tts`` is the speech output's
        class, and ``websocket`` the client socket's. ``early`` are client events sent right after ``session.start``,
        before ``session.started``.
        """
        self.chat = ScriptedChat(*script)
        self.stt, self.tts = FakeSTT(), tts()
        self.engines: list[CascadeAdapter] = []
        self.parts: list[SessionParts] = []

        async def factory(config, transport, connection):
            parts = cascade_parts(
                config, transport, connection, chat=self.chat, stt=self.stt, tts=self.tts, live_config=live_config
            )
            self.engines.append(parts.frontend)
            self.parts.append(parts)
            return parts

        # The gateway's one core, as the root builds it: the shipped core settings, its backend LLM on the same script.
        self.core = chat_core(self.chat, core=core, models=models, **(delegation or {}))
        self.addAsyncCleanup(self.core.aclose)
        hook = core_hook(self.core)
        self.core_client = hook.client
        self.gateway = LiveGateway(factory, LiveSettings(max_session_seconds=60), core=hook)
        self.ws = websocket()
        self.server = asyncio.create_task(run_primary_websocket(self.gateway, self.ws))
        self.addAsyncCleanup(self.stop)
        self.ws.send({"type": "session.start", "event_id": "start_1", "session": session})
        for message in early:
            self.ws.send(message)
        return await self.ws.wait_for("session.started")

    @property
    def live_session(self):
        """The gateway's session that this harness serves."""
        return next(iter(self.gateway.sessions.values()))

    async def history(self):
        """What ``GET /v1/live/sessions/{id}/history`` serves: the session's halves, as the route asks."""
        return await self.live_session.history()

    def relayed_call_ids(self):
        """The call ids of the function calls relayed to the client so far, in order: the core's own."""
        return [
            e["event"]["item"]["call_id"]
            for e in self.ws.from_server
            if e["type"] == "response.event"
            and e["event"]["type"] == "response.output_item.done"
            and e["event"]["item"]["type"] == "function_call"
        ]

    def im(self):
        """The core's IM session that serves this call's conversation."""
        return self.core_client.service.sessions[self.live_session.delegation.conversation]

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
        # The lead-in is said once the delegation is accepted; a result that arrived first would speak for it.
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "check." in e["delta"])
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
        def task_for(session):  # the server delegation's request, with the core's defaults for what it leaves out
            defaults = BackendTask("fallback", STATIC_BACKEND, [])
            return settings_for(request_settings(SessionConfig.model_validate(session).delegation.responses), defaults)

        session = {**RESPONSES_SESSION, "delegation": {"type": "responses", "responses": {"model": "org/org/model"}}}
        task = task_for(session)
        self.assertEqual(task.instructions, STATIC_BACKEND)
        explicit = task_for(RESPONSES_SESSION)
        self.assertEqual(explicit.instructions, "Look things up.")
        # A request that sets nothing (the UI bot's, or a session not in responses mode) gets the default model and
        # prompt, and no tools.
        for nothing in ({"model": "live-1"}, {"model": "live-1", "delegation": {"type": "client"}}):
            omitted = task_for(nothing)
            self.assertEqual((omitted.model, omitted.instructions, omitted.tools), ("fallback", STATIC_BACKEND, []))

    async def test_a_round_cut_by_the_output_limit_ends_incomplete_and_the_voice_says_its_own_line(self):
        # The endings fixture's cut_by_max_output_tokens: the cut round is not tried again and its text is
        # no item; the delegation's response ends incomplete with the official reason, and the voice says its own
        # apology (no tool ran), never the cut text.
        await self.serve(
            RESPONSES_SESSION,
            [_decision_chunks('{"action":"delegate","speech":""}'), [chunk("Partial thought", finish="length")]],
        )
        await self.stt.say("Is my flight on time?")
        ended = await self.ws.wait_for("response.event", where=lambda e: e["event"]["type"] == "response.incomplete")
        self.assertEqual(ended["event"]["response"]["incomplete_details"], {"reason": "max_output_tokens"})
        apology = LINES.apology
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "again?" in e["delta"])
        self.assertEqual(" ".join(self.tts.spoken), " ".join(apology.split()))  # no lead-in: the apology alone
        events = [e["event"]["type"] for e in self.ws.from_server if e["type"] == "response.event"]
        self.assertNotIn("response.completed", events)
        self.assertEqual(len(self.chat.requests), 2)  # the talker and the one round: no retry, no writer

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
        first, _ = self.relayed_call_ids()  # the core's own call ids
        self.ws.send(
            {"type": "response.item.create", "item": {"type": "function_call_output", "call_id": first, "output": "{}"}}
        )
        self.ws.send({"type": "response.create", "event_id": "early"})
        error = await self.ws.wait_for("error")
        # The official code: nothing can be continued while a function result is owed.
        self.assertEqual(
            (error["error"]["code"], error["error"]["client_event_id"]), ("function_call_outputs_required", "early")
        )

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
        # The gateway passes a repeat on and the core judges it: the request that carries both results
        # is refused as the client's mistake, with the core's reason and the response.create's event id.
        [known] = self.relayed_call_ids()
        result = {"type": "function_call_output", "call_id": known, "output": "first"}
        self.ws.send({"type": "response.item.create", "item": result})
        self.ws.send({"type": "response.item.create", "item": {**result, "output": "second"}})
        self.ws.send({"type": "response.create", "event_id": "both"})
        refused = await self.ws.wait_for("error", where=lambda e: e["error"].get("client_event_id") == "both")
        self.assertEqual(
            (refused["error"]["code"], refused["error"]["message"]),
            ("invalid_value", f"Duplicate result for call {known}"),
        )

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

    async def test_session_update_changes_the_backend_for_later_requests(self):
        await self.serve(RESPONSES_SESSION, [])
        patch = {"delegation": {"responses": {"instructions": "New backend rules.", "max_output_tokens": 77}}}
        self.ws.send({"type": "session.update", "event_id": "u", "session": patch})
        await self.ws.wait_for("session.updated")
        # The update applies to the requests sent after it, which read the stored settings.
        settings = request_settings(self.live_session.config.delegation.responses)
        self.assertEqual((settings["instructions"], settings["max_output_tokens"]), ("New backend rules.", 77))
        # A model the core does not serve: the core checks the settings after the answer (with its input_tokens), and
        # the error is nonfatal: the session goes on, and the setting is kept.
        update = {"delegation": {"responses": {"model": "org/unlisted-model"}}}
        self.ws.send({"type": "session.update", "event_id": "u2", "session": update})
        error = (await self.ws.wait_for("error"))["error"]
        self.assertEqual(
            (error["code"], error["param"], error["client_event_id"]),
            ("responses_validation_failed", "session.delegation.responses.model", "u2"),
        )
        self.assertEqual(self.live_session.status, "active")
        self.assertEqual(request_settings(self.live_session.config.delegation.responses)["model"], "org/unlisted-model")

    async def test_the_benchmark_clients_session_starts_with_no_error_and_runs_its_model(self):
        # The second real-core case of the background check: the benchmark client's session names its own agent model,
        # which the catalog lists. It starts with no error, and its typed run goes to that model.
        await self.serve(
            BENCHMARK_SESSION,
            [
                [chunk("Your order is on its way.", finish="stop")],  # the backend: the typed run's round
                _decision_chunks('{"action":"speak","speech":"Your order is on its way."}'),  # the writer
            ],
            models=("benchmark/agent-model",),
        )
        await asyncio.gather(*self.live_session.delegation.checks)  # the start's settings check
        typed = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Where is my order?"}]}
        self.ws.send({"type": "response.item.create", "event_id": "q1", "item": typed})
        self.ws.send({"type": "response.create", "event_id": "go"})
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "way." in e["delta"])
        self.assertNotIn("error", self.ws.kinds())
        self.assertEqual(self.chat.requests[0]["model"], "benchmark/agent-model")

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

    async def test_history_shows_what_the_talker_and_the_backend_hold(self):
        backend_answers = asyncio.Event()  # the backend answers once the lead-in has played
        await self.serve(
            RESPONSES_SESSION,
            [
                _decision_chunks('{"action":"delegate","speech":"Let me check."}'),
                backend_answers,
                [chunk("We have Earl Grey.", finish="stop")],
                _decision_chunks('{"action":"speak","speech":"We have Earl Grey."}'),
            ],
        )
        self.ws.send({"type": "session.thinking.append", "delegation_id": None, "content": "The caller is a regular."})
        await self.stt.say("What teas do you have?")
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "check." in e["delta"])
        backend_answers.set()
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "Grey" in e["delta"])

        history = await self.history()
        self.assertEqual(history, json.loads(json.dumps(history)))
        frontend = [m["content"] for m in history["frontend"]["messages"]]
        self.assertEqual(history["frontend"]["model"], ENDPOINT.model)
        self.assertTrue(any("What teas do you have?" in text for text in frontend))
        self.assertTrue(any("The caller is a regular." in text for text in frontend))
        # Each message carries its kind.
        kinds = {m["content"]: m["kind"] for m in history["frontend"]["messages"]}
        self.assertEqual(kinds[PROMPTS.context_label + "The caller is a regular."], "app_context")
        self.assertEqual(kinds[PROMPTS.result_label + "We have Earl Grey."], "result")
        backend = history["backend"]
        request = backend["items"][0]["content"][0]["text"]
        # The delegation's input is the labeled transcript since the last request, the context included.
        self.assertTrue(request.startswith(PROMPTS.transcript.header), request)
        self.assertIn("The caller is a regular.", request)
        self.assertIn("What teas do you have?", request)
        self.assertIn("We have Earl Grey.", json.dumps(backend["items"][1:]))
        # The transcript went with the delegation. The lead-in was said after its input was built, so the next
        # delegation carries it; the answer the caller heard in full is the backend's own.
        self.assertEqual(
            backend["pending"], [{"kind": Heard.AGENT, "text": f"{PROMPTS.transcript.agent}: Let me check."}]
        )
        # What the session set, then the newest request's instructions as the core composed them.
        composed = compose(RESPONSES_SESSION["delegation"]["responses"]["instructions"], *self.im().instruction_notes())
        self.assertEqual(
            backend["prompt"],
            [
                {"source": "delegation.responses.instructions", "text": "Look things up."},
                {"source": COMPOSED_SOURCE, "text": composed},
            ],
        )

    async def test_history_names_where_each_part_of_each_prompt_comes_from(self):
        await self.serve(RESPONSES_SESSION, [])
        self.ws.send({"type": "session.instructions.append", "delegation_id": None, "content": "Be brief."})
        # An instruction append is pinned in the talker's prompt, after the session's instructions.
        async with asyncio.timeout(3):
            while "session.instructions.append" not in [
                p["source"] for p in (await self.history())["frontend"]["prompt"]
            ]:
                await asyncio.sleep(0.01)
        history = await self.history()
        talker = [part["source"] for part in history["frontend"]["prompt"]]
        session_part = talker.index("session.instructions")
        self.assertEqual(history["frontend"]["prompt"][session_part]["text"], RESPONSES_SESSION["instructions"])
        self.assertLess(session_part, talker.index("session.instructions.append"))
        self.assertIn("agent_prompts.Cascade.routing", talker)
        thinker = history["backend"]["prompt"]
        self.assertEqual(thinker[0], {"source": "delegation.responses.instructions", "text": "Look things up."})

        self.ws.send(
            {
                "type": "session.update",
                "session": {"delegation": {"type": "responses", "responses": {"instructions": "Use the tools."}}},
            }
        )
        await self.ws.wait_for("session.updated")
        self.assertEqual((await self.history())["backend"]["prompt"][0]["text"], "Use the tools.")

    async def test_history_shows_the_backend_busy_while_a_delegation_waits_for_the_app(self):
        # The backend's busy flag comes from the adapter's work note.
        # At the core's client-tool deadline (0.2 s here, the core's client_tool_timeout_seconds) the core's work ends
        # and nothing runs there any more, but the delegation still waits for the app in the work note: when the
        # voice's app wait (0.4 s here) runs out, it says the app-wait line once and keeps the delegation watched, so a
        # late result is still said, and the flag stays on.
        delegation = LIVE_CONFIG.delegation.model_copy(update={"app_wait_seconds": 0.4})
        live_config = LIVE_CONFIG.model_copy(update={"delegation": delegation})
        await self.serve(
            RESPONSES_SESSION,
            [_decision_chunks('{"action":"delegate","speech":"Let me check."}'), tool_call_round()],
            live_config=live_config,
            delegation={"client_tool_timeout_seconds": 0.2},
        )
        await self.stt.say("What teas do you have?")
        await self.ws.wait_for("response.event", where=lambda e: e["event"]["type"] == "response.completed")
        self.assertTrue((await self.history())["backend"]["busy"])  # the delegation waits for the app
        im = self.im()
        async with asyncio.timeout(2):  # the core's deadline passes: its work ended, and nothing runs there any more
            while im.busy:
                await asyncio.sleep(0.01)
        self.assertTrue((await self.history())["backend"]["busy"])
        line = LINES.app_wait_line
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: line.split()[-1] in e["delta"])
        self.assertTrue((await self.history())["backend"]["busy"])  # still watched after the app-wait line


class TalkerMovesEndToEndTests(EngineHarness):
    """The talker's moves in a live session, through the gateway, the cascade adapter and the core."""

    def said(self) -> str:
        """Everything the caller heard so far, from the session's output transcript."""
        return " ".join(
            e["delta"].strip() for e in self.ws.from_server if e["type"] == "session.output_transcript.delta"
        )

    async def until(self, predicate, what: str, timeout: float = 4.0) -> None:
        try:
            async with asyncio.timeout(timeout):
                while not predicate():
                    await asyncio.sleep(0.01)
        except TimeoutError:
            raise AssertionError(f"timed out waiting for {what}") from None

    def decided(self, count: int):
        """Whether the talker's ``count``-th call has been decided."""
        return lambda: len(self.chat.requests) >= count and self.engines[0].voice.call is None

    async def test_a_wait_says_nothing_and_the_next_turn_is_decided_after_it(self):
        await self.serve(
            RESPONSES_SESSION,
            [decision("wait"), decision("speak", "Earl Grey it is.")],
        )
        await self.stt.say("Hmm, let me think.")
        await self.until(self.decided(1), "the wait")
        await self.stt.say("A pot of Earl Grey.")
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "is." in e["delta"])
        self.assertEqual(self.said(), "Earl Grey it is.")  # the wait said nothing
        self.assertNotIn("session.delegation.created", self.ws.kinds())
        runtime = self.chat.requests[1]["messages"][-1]["content"]
        self.assertIn("Hmm, let me think.", runtime)  # the turn it waited on is the conversation's
        self.assertIn(PROMPTS.newest_turn.format(turn="A pot of Earl Grey."), runtime)  # the newest turn alone

    async def test_a_barge_in_in_the_middle_of_the_decision_cuts_it_and_both_turns_are_decided_together(self):
        held = asyncio.Event()  # the first decision stops in the middle of its JSON
        await self.serve(
            RESPONSES_SESSION,
            [
                [chunk('{"action":"speak","spe'), held, chunk('ech":"We have Earl Grey."}'), chunk(finish="stop")],
                decision("speak", "Coffee it is."),
            ],
        )
        self.addCleanup(held.set)
        await self.stt.say("Tell me about your teas.")
        await self.until(lambda: self.chat.requests, "the first decision")
        await self.stt.propose("Actually, a coffee.")
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "is." in e["delta"])
        self.assertEqual(self.said(), "Coffee it is.")  # nothing of the cut decision was said
        turns = PROMPTS.newest_turn.format(turn="Tell me about your teas. Actually, a coffee.")
        self.assertIn(turns, self.chat.requests[1]["messages"][-1]["content"])
        said = [m["content"] for m in self.engines[0].frontend.context.get_messages() if m["role"] == "assistant"]
        self.assertEqual(said, ["Coffee it is."])

    async def test_two_delegations_each_answer_and_the_second_carries_what_was_heard_since_the_first(self):
        first, second = asyncio.Event(), asyncio.Event()  # each answer comes once its lead-in has played
        await self.serve(
            RESPONSES_SESSION,
            [
                decision("delegate", "Let me check."),
                first,
                [chunk("We have Earl Grey.", finish="stop")],
                decision("speak", "We have Earl Grey."),
                decision("delegate", "One moment."),
                second,
                [chunk("Your order is placed.", finish="stop")],
                decision("speak", "Your order is placed."),
            ],
        )
        await self.stt.say("What teas do you have?")
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "check." in e["delta"])
        first.set()
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "Grey." in e["delta"])
        await self.until(self.decided(3), "the first answer")
        await self.stt.say("Order one, please.")
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "moment." in e["delta"])
        second.set()
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "placed." in e["delta"])
        created = [e["delegation"]["id"] for e in self.ws.from_server if e["type"] == "session.delegation.created"]
        self.assertEqual(len(set(created)), 2)
        self.assertEqual(self.said(), "Let me check. We have Earl Grey. One moment. Your order is placed.")
        # The second delegation's input is what the caller heard since the first one's: its lead-in, said after the
        # first input was built, and the newer turn.
        second_input = self.chat.requests[4]["messages"]  # the backend's round for the second delegation
        text = json.dumps(second_input, ensure_ascii=False)
        self.assertIn(f"{PROMPTS.transcript.agent}: Let me check.", text)
        self.assertIn("Order one, please.", text)


class TypedQuestionTests(EngineHarness):
    """The application's typed question enters the talker's context, with the gateway's half."""

    async def test_the_writer_words_a_typed_runs_answer_knowing_the_question_which_is_never_a_caller_turn(self):
        await self.serve(
            RESPONSES_SESSION,
            [
                [chunk("There is parking behind the cafe.", finish="stop")],  # the backend: the typed run's round
                _decision_chunks('{"action":"speak","speech":"Yes, there is parking behind the cafe."}'),  # the writer
            ],
        )
        typed = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Is there parking?"}]}
        self.ws.send({"type": "response.item.create", "event_id": "q1", "item": typed})
        self.ws.send({"type": "response.create", "event_id": "go"})
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "cafe." in e["delta"])
        self.assertNotIn("error", self.ws.kinds())
        question = {"role": "user", "content": PROMPTS.context_label + "Is there parking?"}
        # The writer's request: the question, then the answer to word, in its one user message.
        wording = self.chat.requests[1]["messages"][-1]
        self.assertEqual(wording["role"], "user")
        self.assertTrue(wording["content"].startswith(f"{question['content']}\n"), wording["content"])
        voice = self.engines[0].voice
        context = [(m["role"], m["content"], m["kind"]) for m in voice.conversation()]
        self.assertIn((question["role"], question["content"], "typed"), context)
        self.assertNotIn("caller", [kind for _, _, kind in context])  # no caller turn
        self.assertEqual([entry.kind for entry in voice.heard_log.entries], [Heard.AGENT])


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
        self.assertNotIn("response_id", created["delegation"])
        delegation_id = created["delegation"]["id"]
        # The client answers once the lead-in has played; commentary that came first would drop it.
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "moment." in e["delta"])
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
        self.assertIn('(the caller already heard: "One moment."; do not say that again)', update)
        # The writer words the delegation's answer as one the caller waits for, so it may only speak.
        self.assertEqual(
            self.chat.requests[1]["response_format"]["json_schema"]["schema"]["properties"]["action"]["enum"],
            ["speak"],
        )
        self.assertEqual(len(self.chat.requests), 2)  # the talker and the writer: the server runs no backend
        self.assertNotIn("response.event", self.ws.kinds())

    async def test_client_commentary_is_spoken_even_when_the_writer_would_stay_silent(self):
        await self.serve(
            self.CLIENT_SESSION,
            [
                _decision_chunks('{"action":"delegate","speech":""}'),
                _decision_chunks('{"action":"wait","speech":""}'),
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
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "fact." in e["delta"], timeout=3)
        # The talker delegated with no lead-in, so the voice says nothing first; the writer's answer may not
        # stay silent, so the client's update is said as written.
        self.assertEqual(" ".join(self.tts.spoken), "Background fact.")

    async def test_empty_thinking_for_a_delegation_cancels_its_waiting_lead_in(self):
        # The Live API's recipe: a delay at the start, then empty thinking that names the announced delegation.
        session = {**self.CLIENT_SESSION, "delegation": {"type": "client", "acknowledgement_delay_ms": 300}}
        script = [
            _decision_chunks('{"action":"delegate","speech":"One moment."}'),
            _decision_chunks('{"action":"speak","speech":"It is sunny today."}'),
        ]
        await self.serve(session, script)
        await self.stt.say("What is the weather?")
        delegation_id = (await self.ws.wait_for("session.delegation.created"))["delegation"]["id"]
        quiet = {"type": "session.thinking.append", "event_id": "quiet", "delegation_id": delegation_id, "content": ""}
        self.ws.send(quiet)
        self.assertEqual((await self.ws.wait_for("session.thinking.appended", timeout=3))["client_event_id"], "quiet")
        self.assertEqual(self.engines[0].voice.acknowledgements.pending, {})
        await asyncio.sleep(0.5)  # past the delay
        self.ws.send(
            {"type": "session.commentary.append", "delegation_id": delegation_id, "content": "Weather: sunny, 72."}
        )
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "sunny" in e["delta"])
        self.assertEqual(" ".join(self.tts.spoken), "It is sunny today.")
        self.assertNotIn("One moment.", self.chat.requests[1]["messages"][-1]["content"])  # never told
        messages = (await self.history())["frontend"]["messages"]
        context = [message["content"] for message in messages if message["kind"] == "app_context"]
        self.assertEqual(context, [PROMPTS.context_label + "Weather: sunny, 72."])  # the commentary only

    async def test_history_has_no_backend_when_the_client_runs_it(self):
        await self.serve(self.CLIENT_SESSION, [])
        history = await self.history()
        self.assertIsNone(history["backend"])
        # The talker's instructions are a part of its prompt; its messages are the voice history only.
        instructions = {"source": "session.instructions", "text": self.CLIENT_SESSION["instructions"]}
        self.assertIn(instructions, history["frontend"]["prompt"])
        self.assertNotIn("system", [message["role"] for message in history["frontend"]["messages"]])
        # A client delegation never enters the work note, so the talker's prompt has no work line.
        self.assertNotIn("open delegations (runtime)", [part["source"] for part in history["frontend"]["prompt"]])
        # Every way to ask for client mode (no block, null, {} or {"type": "client"}) is client mode, whose delegation
        # part (live/client_delegation.py) has no backend half.
        for delegation in (None, {}, {"type": "client"}):
            with self.subTest(delegation=delegation):
                self.assertEqual(SessionConfig(**{**self.CLIENT_SESSION, "delegation": delegation}).mode, "client")

    async def test_response_commands_are_refused_in_client_mode(self):
        await self.serve(self.CLIENT_SESSION, [])
        self.ws.send({"type": "response.create"})
        error = await self.ws.wait_for("error")
        self.assertIn("requires Responses delegation", error["error"]["message"])

    async def test_a_webrtc_offer_that_asks_for_client_delegation_starts_a_client_mode_session(self):
        built = []

        async def factory(config, transport, ice_servers):
            built.append(config.mode)
            return SessionParts(frontend=_IdleFrontend(), connection=LiveWebRTCConnection.from_ice_servers(ice_servers))

        router, gateway = create_live_router(factory, LiveSettings())
        app = FastAPI()
        app.include_router(router)
        client = RTCPeerConnection()
        self.addAsyncCleanup(client.close)
        client.addTrack(AudioStreamTrack())
        client.createDataChannel("oai-events")
        await client.setLocalDescription(await client.createOffer())
        offer = {"type": "webrtc", "sdp": client.localDescription.sdp}
        session = {**self.CLIENT_SESSION, "delegation": {"type": "client"}}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as http:
            response = await http.post("/v1/live/sessions", json={"session": session, "transport": offer})
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(built, ["client"])
        live = next(iter(gateway.sessions.values()))
        self.assertEqual((live.delegation.target, live.delegation.conversation), ("client", None))
        await gateway.close_all()


class _IdleFrontend:
    """A frontend adapter that starts and closes and does nothing else, for a session that only needs to start."""

    async def start(self, session, delegation):
        self.delegation = delegation

    async def close(self):
        pass


class WebRTCEndToEndTests(unittest.IsolatedAsyncioTestCase):
    """The real engine over a real WebRTC session: Pipecat's transport on the live connection."""

    def setUp(self):
        # The shared audio recorder's switches at their defaults, whatever the environment sets: nothing records.
        self.enterContext(patch.object(audio_recorder, "ENABLE_ASR_DUMP", False))
        self.enterContext(patch.object(audio_recorder, "ENABLE_TTS_DUMP", False))

    async def test_reply_audio_reaches_the_media_track_and_the_final_event_precedes_channel_close(self):
        import httpx
        from aiortc import RTCPeerConnection, RTCSessionDescription
        from aiortc.mediastreams import AudioStreamTrack
        from fastapi import FastAPI

        from gateways.live.mount import mount_live

        chat = FakeChat(_decision_chunks('{"action":"speak","speech":"Hello from the kiosk."}'))
        stt, tts = FakeSTT(), FakeTTS()

        async def factory(config, transport, connection):
            return cascade_parts(config, transport, connection, chat=chat, stt=stt, tts=tts)

        core = chat_core(chat)
        self.addAsyncCleanup(core.aclose)
        app = FastAPI()
        gateway = mount_live(app, factory=factory, core=core_hook(core))
        self.addAsyncCleanup(gateway.close_all)  # as at shutdown: a failure mid-call leaves no session open
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
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as http:
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
        # Pipecat's transport stopped with the pipeline; the gateway closed the socket once the final event went out.
        self.assertEqual((self.ws.closed, self.ws.sent_at_close[-1]), (True, "session.closed"))

    async def test_a_client_that_leaves_ends_the_session_by_its_close_code(self):
        for code, reason in ((1000, "remote_hangup"), (1001, "remote_hangup"), (1006, "connection_lost")):
            with self.subTest(code=code):
                await self.serve(RESPONSES_SESSION, [])
                self.ws.leave(code)
                async with asyncio.timeout(10):
                    await self.server
                [closed] = [event for event in self.ws.from_server if event["type"] == "session.closed"]
                self.assertEqual((closed["reason"], self.engines[-1].runner_task.done()), (reason, True))

    async def test_a_command_sent_before_the_session_started_runs_once_it_is_active(self):
        # Pipecat's transport reads the socket from the pipeline's start, before the session is active.
        mute = {"type": "session.input_audio.mute", "event_id": "early"}
        await self.serve(RESPONSES_SESSION, [], early=[mute])
        muted = await self.ws.wait_for("session.input_audio.muted")
        self.assertEqual(muted["client_event_id"], "early")
        self.assertEqual(self.ws.kinds()[:2], ["session.started", "session.input_audio.muted"])
        self.assertNotIn("error", self.ws.kinds())

    async def test_a_session_that_cannot_start_reports_an_error_and_closes(self):
        self.stt, self.tts = FakeSTT(), FakeTTS()

        async def factory(config, transport, connection):
            raise RuntimeError("no models")

        gateway = LiveGateway(factory, LiveSettings())
        ws = FakeWebSocket()
        task = asyncio.create_task(run_primary_websocket(gateway, ws))
        ws.send({"type": "session.start", "event_id": "s1", "session": RESPONSES_SESSION})
        async with asyncio.timeout(10):
            await task
        error = ws.from_server[0]["error"]
        self.assertEqual((error["code"], error["client_event_id"]), ("server_error", "s1"))
        self.assertTrue(ws.closed)
        self.assertEqual(gateway.sessions, {})


class WrittenOffTTS(FakeTTS):
    """A voice service that produces no audio, written off as Pipecat's ``TTSService`` writes one off.

    After three contexts in a row with no audio, ``TTSService`` reports a permanent error
    (``push_error(..., force_treat_as_permanent=True)``) and is given no more work, so it reports nothing more; here
    the first line does it at once.
    """

    async def process_frame(self, frame, direction):
        if isinstance(frame, LLMTextFrame):
            await FrameProcessor.process_frame(self, frame, direction)
            if self.is_usable:
                message = "3 consecutive TTS contexts completed with no audio"
                await self.push_error(message, force_treat_as_permanent=True)
            return
        await super().process_frame(frame, direction)


class VoiceServiceFailureTests(EngineHarness):
    async def test_a_written_off_voice_ends_the_call_with_one_clear_error_and_an_official_reason(self):
        # No voice means no service. The worker ends the pipeline (ProcessorUnusablePolicy.END), and
        # the session closes with the official connection_lost, after one error that says why.
        await self.serve(
            RESPONSES_SESSION, [_decision_chunks('{"action":"speak","speech":"Hello there."}')], tts=WrittenOffTTS
        )
        await self.stt.say("Hi.")
        closed = await self.ws.wait_for("session.closed", timeout=15)
        self.assertEqual((closed["reason"], closed.get("client_event_id")), ("connection_lost", None))
        errors = [event["error"] for event in self.ws.from_server if event["type"] == "error"]
        self.assertEqual(
            [(error["code"], error["message"]) for error in errors],
            [("pipeline_error", "3 consecutive TTS contexts completed with no audio")],
        )
        self.assertEqual(self.ws.kinds()[-1], "session.closed")

    async def test_a_voice_that_gives_three_lines_in_a_row_no_audio_ends_the_call(self):
        # Pipecat's own text-to-speech, at the example's limit: the line, said once more, and the next line all end
        # with no audio. The first two are notices; the third writes the service off, and the session closes.
        script = [
            _decision_chunks('{"action":"speak","speech":"Hello there."}'),
            _decision_chunks('{"action":"speak","speech":"Are you there?"}'),
        ]
        dead = partial(ToneTTS, silent={"Hello there.", "Are you there?"}, stop_frame_timeout_s=0.2)
        await self.serve(RESPONSES_SESSION, script, tts=dead)

        def notices():
            return [(e["code"], e["message"]) for e in self.ws.from_server if e["type"] == "info"]

        await self.stt.say("Hi.")
        async with asyncio.timeout(5):  # the line and the line said once more, each with no audio
            while len(notices()) < 2:
                await asyncio.sleep(0.01)
        await self.stt.say("Hello?")
        closed = await self.ws.wait_for("session.closed", timeout=15)
        self.assertEqual((closed["reason"], closed.get("client_event_id")), ("connection_lost", None))
        spoken = [sentence for _, sentence in self.tts.spoken]
        self.assertEqual(spoken, ["Hello there.", "Hello there.", "Are you there?"])
        self.assertEqual([code for code, _ in notices()], ["pipeline_error"] * 2)
        self.assertTrue(all(message.endswith("completed with no audio") for _, message in notices()))
        errors = [event["error"] for event in self.ws.from_server if event["type"] == "error"]
        self.assertEqual(
            [(error["code"], error["message"]) for error in errors],
            [("pipeline_error", "3 consecutive TTS contexts completed with no audio")],
        )
        self.assertEqual(self.ws.kinds()[-1], "session.closed")


class StartOnceTests(EngineHarness):
    """Who starts what in a live session: each loop runs once.

    The delegation side's ``open`` opens the conversation, whose IM session starts its placer, and the adapter's
    ``start`` the silence timer. Building the voice starts neither.
    """

    @staticmethod
    def running(name):
        return [task for task in asyncio.all_tasks() if task.get_name() == name and not task.done()]

    async def test_the_backend_loop_starts_once(self):
        await self.serve(RESPONSES_SESSION, [])
        self.assertEqual(len(self.running("im-placer")), 1)
        await self.live_session.delegation.open(self.live_session.delegation.view)  # a second open starts nothing
        self.assertEqual(len(self.running("im-placer")), 1)

    async def test_the_silence_timer_starts_once(self):
        await self.serve(RESPONSES_SESSION, [])
        self.assertEqual(len(self.running("delegation-progress")), 1)


class CloseOwnerTests(EngineHarness):
    """Who closes what in a live session: the delegation side's own close stops its worker.

    The adapter's close stops the pipeline, the voice's tasks and the silence timer, and leaves the delegation side
    alone. A whole session's close cannot tell the two apart: it closes the side right after the
    adapter, and a second close of the side does nothing.
    """

    async def test_the_adapters_close_leaves_the_delegation_side_to_its_own_close(self):
        await self.serve(RESPONSES_SESSION, [])
        # Closed here while the session is active, the adapter's pipeline end would ask the session to close, and the
        # session's own close would then close the side too; that request is held, so the adapter's close runs alone.
        with patch.object(self.engines[0].session, "request_close") as held:
            await self.parts[0].frontend.close()
        delegation = self.live_session.delegation
        placer = self.im().placer  # the IM session's loop in the core
        self.assertFalse(placer.done(), "the adapter's close stopped the delegation side's work")
        await delegation.close()
        self.assertTrue(placer.done())
        # The held request ends the session, as in a call: with the adapter closed, nothing reads the client's socket.
        self.engines[0].session.request_close(*held.call_args.args)


class CallerAudioTests(EngineHarness):
    async def test_websocket_caller_audio_enters_the_pipeline_at_the_cascade_rate(self):
        # The cascade hears the caller at 16 kHz (LIVE_AUDIO_IN_RATE). Audio from a primary WebSocket comes in its
        # wire format, here the default 24 kHz PCM16, and the transport's serializer resamples it to the cascade's
        # rate before the pipeline sees it.
        await self.serve(RESPONSES_SESSION, [])
        gate, frames = self.engines[0].gate, []

        class CallerAudio(BaseObserver):
            async def on_push_frame(self, data):
                if isinstance(data.frame, InputAudioRawFrame) and data.source is gate:
                    frames.append(data.frame)

        self.engines[0].task.add_observer(CallerAudio())
        self.feed_silence(3)
        async with asyncio.timeout(5):
            while len(frames) < 3:
                await asyncio.sleep(0.01)
        self.assertEqual({(frame.sample_rate, frame.num_channels) for frame in frames}, {(LIVE_AUDIO_IN_RATE, 1)})
        self.assertEqual(LIVE_AUDIO_IN_RATE, 16000)


class ReplyAudioTests(EngineHarness):
    async def test_a_lines_first_word_reaches_the_transcript_when_the_output_resamples_its_audio(self):
        # At 16 kHz the WebSocket's output resamples the reply audio and sends it in whole 100 ms packets, so the first
        # word's text reaches what reads the call after the output before its audio, and waits there for that audio.
        session = {**RESPONSES_SESSION, "audio": {"format": {"type": "audio/pcm", "rate": 16000}}}
        await self.serve(session, [_decision_chunks('{"action":"speak","speech":"Hello there."}')])
        await self.stt.say("Hi.")
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "there." in e["delta"])
        spoken = [e["delta"].strip() for e in self.ws.from_server if e["type"] == "session.output_transcript.delta"]
        self.assertEqual(spoken, ["Hello", "there."])


class LatencyLogTests(EngineHarness):
    async def test_each_turn_logs_its_latency_and_breakdown_and_the_metrics_pass_the_whole_pipeline(self):
        logged = []
        sink = logger.add(lambda message: logged.append(message.record["message"]), level="INFO")
        self.addCleanup(logger.remove, sink)
        await self.serve(RESPONSES_SESSION, [_decision_chunks('{"action":"speak","speech":"Hello there."}')])
        engine, passed = self.engines[0], []
        last = engine.frontend.processors[-1]  # the assistant aggregator

        class PassedMetrics(BaseObserver):
            async def on_push_frame(self, data):
                if isinstance(data.frame, MetricsFrame) and data.source is last:
                    passed.extend(data.frame.data)

        engine.task.add_observer(PassedMetrics())
        await self.stt.push_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))  # the caller's speech ended
        await self.stt.say("Hi.")
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "there." in e["delta"])
        latency = [line for line in logged if line.startswith("User-to-bot latency: ")]
        breakdown = [line for line in logged if line.startswith("Latency breakdown: ")]
        self.assertEqual((len(latency), len(breakdown)), (1, 1))  # one turn, one line
        self.assertIn("endpointing wait", breakdown[0])  # from the caller's silence to the bot's first audio
        self.assertTrue(breakdown[0].endswith("TOTAL"))
        # The talker's time to first byte passed every processor of the session's pipeline, the output's included.
        talker = engine.voice.talker.name
        self.assertTrue([data for data in passed if isinstance(data, TTFBMetricsData) and data.processor == talker])
        spoken = [e["delta"].strip() for e in self.ws.from_server if e["type"] == "session.output_transcript.delta"]
        self.assertEqual(spoken, ["Hello", "there."])  # the call is as it was


class ContextMeterTests(unittest.TestCase):
    def test_the_cascade_reports_how_much_conversation_its_talker_holds(self):
        from types import SimpleNamespace

        from pipecat.processors.aggregators.llm_context import LLMContext

        from examples.frontend_backend_live.delegation_core.items import message_text
        from examples.frontend_backend_live.voice import Voice
        from tests.unit.frontend_helpers import Port

        engine = SimpleNamespace(voice=None)
        self.assertIsNone(CascadeAdapter.context_tokens(engine))
        engine.voice = Voice(
            config=LIVE_CONFIG,
            prompts=PROMPTS,
            lines=LINES,
            json_rules=PROMPTS.json_rules,
            empty_turn_note=PROMPTS.empty_turn_note,
            delegation=Port(),
            item_text=message_text,
        )
        engine.voice.context = LLMContext([{"role": "user", "content": "hi"}])
        small = CascadeAdapter.context_tokens(engine)
        engine.voice.context = LLMContext([{"role": "user", "content": "hi " * 500}])
        self.assertGreater(CascadeAdapter.context_tokens(engine), small + 300)


class AdapterCloseTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_later_close_stops_only_a_pipeline_that_started_after_the_first(self):
        from types import SimpleNamespace

        calls = []

        async def stop_pipeline():
            calls.append("pipeline stopped")

        async def close_voice():
            calls.append("voice closed")

        adapter = SimpleNamespace(
            _closed=False, runner_task=None, stop_pipeline=stop_pipeline, voice=SimpleNamespace(close=close_voice)
        )
        # The session closes its frontend adapter, then its delegation side, and one adapter is both.
        await CascadeAdapter.close(adapter)
        await CascadeAdapter.close(adapter)
        self.assertEqual(calls, ["pipeline stopped", "voice closed"])
        # A session closed while it was starting: its pipeline started after the close, and the start's early-close
        # path closes the parts again. That late pipeline is stopped; the voice is not closed twice.
        adapter.runner_task = SimpleNamespace(done=lambda: False)
        await CascadeAdapter.close(adapter)
        self.assertEqual(calls, ["pipeline stopped", "voice closed", "pipeline stopped"])
        adapter.runner_task = SimpleNamespace(done=lambda: True)
        await CascadeAdapter.close(adapter)
        self.assertEqual(calls, ["pipeline stopped", "voice closed", "pipeline stopped"])


if __name__ == "__main__":
    unittest.main()
